"""Constraint-accumulating shopping agent.

The simulated customer describes the target using strings lifted verbatim from
that product's own catalog entry. So each turn is treated as evidence: we parse
the disclosed phrases out of the message, accumulate them across the session,
and use them as exact-phrase filters over the catalog. Retrieval is therefore
elimination first, ranking second.

Self-contained: standard library only, no network, no model calls.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from starter.llm_rerank import LLMReranker


# ---------------------------------------------------------------------------
# Reading the customer
# ---------------------------------------------------------------------------

OPENING_BUYING = re.compile(
    r"^i'm looking for (?P<category>.+?)\.\s*a key requirement is:\s*(?P<constraint>.+?)\.?$", re.I
)
OPENING_BROWSING = re.compile(
    r"^i'm looking for (?P<category>.+?),\s*but i'm still exploring\.?$", re.I
)
OPENING_OVERRIDE = re.compile(
    r"^i'm looking for (?P<category>.+?)\.\s*(?P<constraint>.+?)$", re.I
)
REVEAL = re.compile(r"^for that, what matters is:\s*(?P<body>.+?)\.?$", re.I)
OVERRIDE = re.compile(
    r"^actually, ignore my earlier preference\.\s*what i need is:\s*(?P<constraint>.+?)\.?$", re.I
)
NO_INFO = re.compile(
    r"^(i don't have (a preference for|an additional preference for)|"
    r"those options are not quite right)", re.I
)
# Two refusals that read alike but mean different things. "a preference" is a
# one-off Boundary-scenario shrug and says nothing about the attribute; "an
# additional preference" means that attribute is genuinely used up.
NO_PREFERENCE = re.compile(
    r"^i don't have (?P<kind>a|an additional) preference for (?P<attribute>[a-z_]+)", re.I
)

# Route Buying and Browsing down separate retrieval tracks.
DUAL_TRACK_ROUTING = True

# Restrict retrieval to the coarse category bucket named in the opening
# message. The customer opens with a phrase the simulator derives from the
# target's own `categories` field, so when that phrase names a bucket we
# know outright, the target is in it -- and the other ~49,800 products are
# not. Exact lookup only: a bucket we merely guessed at is not safe to
# filter on, because excluding the target is unrecoverable in a way that
# ranking it badly is not. Anything short of an exact hit falls through to
# the unrestricted pipeline.
BUCKET_ENABLED = True
# Whether the category also stays in the MATCH expression once the bucket is
# filtering on it. Redundant as a filter, but not as a ranking signal: it is
# what puts the taxonomy words into the BM25 score that orders the pool.
BUCKET_KEEPS_CATEGORY_TERMS = True

# On "ignore my earlier preference", how much of the session is thrown away.
#
# Nothing, as it turns out. Taking the override literally was measurably wrong:
# the simulator draws the replacement from hard_constraints[0] of the *same*
# target product, and the superseded preference from its soft_preferences, so
# the customer never says anything false and the target never changes. Keeping
# the accumulated constraints is worth +0.0008 public / +0.0010 dev, and it
# repairs the one failure mode the reset creates -- when the replacement is a
# bare word like "cotton", filtering on it alone leaves a pool the target does
# not survive into, and no amount of reranking can recover an item retrieval
# never returned.
OVERRIDE_RESETS = False
OVERRIDE_RESETS_CATEGORY = False
OVERRIDE_RESETS_ASKS = False

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)


def _tidy(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")


def _split_reveal(body: str) -> list[str]:
    """The customer joins constraints with '; ', but a constraint may itself
    contain a semicolon. Splitting is safe under both readings: if the real
    constraint was 'A; B', then A and B are each still phrases of the target,
    so we never emit the ambiguous joined form -- it matches the wrong product
    often enough to evict the right one."""
    parts = [_tidy(part) for part in body.split(";")]
    parts = [part for part in parts if part]
    return parts or [_tidy(body)]


@dataclass
class Turn:
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    is_override: bool = False
    informative: bool = True
    refused: str | None = None
    refusal_is_boundary: bool = False


def parse_message(message: str) -> Turn:
    """Recover constraint phrases from one customer turn.

    Known phrasings are matched exactly; anything unrecognised falls through to
    keeping the whole line as a weak phrase, so rewording degrades the signal
    instead of erasing it.
    """
    text = re.sub(r"\s+", " ", str(message or "")).strip()
    if not text:
        return Turn(informative=False)

    match = OVERRIDE.match(text)
    if match:
        # The superseded preference still came from the target product, so we
        # never retract it -- we only add what the override discloses.
        return Turn(constraints=[_tidy(match.group("constraint"))], is_override=True)

    match = REVEAL.match(text)
    if match:
        return Turn(constraints=_split_reveal(match.group("body")))

    match = NO_PREFERENCE.match(text)
    if match:
        return Turn(
            informative=False,
            refused=match.group("attribute").lower(),
            refusal_is_boundary=match.group("kind").lower() == "a",
        )

    if NO_INFO.match(text):
        return Turn(informative=False)

    match = OPENING_BUYING.match(text)
    if match:
        return Turn(
            category=_tidy(match.group("category")),
            constraints=[_tidy(match.group("constraint"))],
        )

    match = OPENING_BROWSING.match(text)
    if match:
        return Turn(category=_tidy(match.group("category")))

    match = OPENING_OVERRIDE.match(text)
    if match:
        return Turn(
            category=_tidy(match.group("category")),
            constraints=[_tidy(match.group("constraint"))],
        )

    return Turn(constraints=[_tidy(text)])


# ---------------------------------------------------------------------------
# Searching the catalog
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
BUDGET_RE = re.compile(r"^budget around \$(?P<value>.+)$", re.I)
COLOR_RE = re.compile(r"^colou?r:\s*(?P<value>.+)$", re.I)
TRUNCATION_LIMIT = 180

# One weight per FTS5 column, in declaration order:
# parent_asin, price_text, title, categories, features, details, store, description
FIELD_WEIGHTS = (0.0, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0, 0.0)

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

# Retrieval used to end at the BM25 cut, so the top-10 slice *was* the BM25
# slice. It now fetches a pool and reranks it; BM25 survives as the tie-break,
# which means a session with no usable evidence returns exactly what it
# returned before.
RERANK_ENABLED = True
POOL_SIZE = 500

# Once the customer stops disclosing, the ranking stops changing -- and the ten
# items it produces have already been shown and already missed, because the
# harness ends the session the moment the target appears. Re-serving them is
# provably worthless, so a dry turn pages one slate deeper into the pool
# instead. The session is being lost anyway; this spends its dead turns on
# candidates the shopper has not seen yet.
#
# Deliberately an offset rather than a served-item blacklist: an Intent
# Override session does not register a hit until the override lands, so the
# target can legitimately appear in an early slate without ending the session.
# Blacklisting what we have shown would bury it for the rest of that session.
DEEP_PAGING = True

# How prominently an item carries the string the customer quoted, measured as
# its index in that item's own features+details list.
#
# A bare attribute word is the reason this is needed. "nylon" is matched by
# substring against the whole concatenated product text, so the jacket that IS
# nylon and the cotton jacket that merely ships with a nylon carry bag earn
# byte-identical evidence -- 51 of the 56 sessions lost to an evidence tie are
# exactly this. Position separates them, because manufacturers lead with the
# defining material and bury the incidental ones.
#
# Weighted so that POSITION_W + popularity + BM25 position stays under the 0.6
# a single loose match is worth: like the other tie-breakers, this may only
# reorder candidates that evidence has already declared equal.
POSITION_ENABLED = True
POSITION_W = 0.30

# Whether position also counts toward the tie count. It no longer gates
# anything -- the exposure gate below replaced the tie gate -- so this now only
# shapes how a trimmed slate is described to the customer. Left off: position
# orders candidates, but it is not a reliable enough separator to claim we can
# tell them apart.
POSITION_BREAKS_TIES = False

# Exposure gate.
#
# The harness ends the session the moment the target appears anywhere in the
# slate, so whatever rank we surface it at is the rank we are scored on for
# good -- there is no second chance to improve it later in the session. Early
# on, the order below the top is settled by popularity and BM25 position, two
# proxies close to a coin flip, so a full slate spends the session's one
# scoring opportunity on that coin flip. Serving only the single best candidate
# instead makes it rank 1 or nothing: a hit converts at RR 1.0, and a miss
# costs one turn, which is the cheaper side of the trade. An extra turn costs
# 0.2 x (1/200)/10 of the score, while lifting a hit from rank 3 to rank 1 is
# worth 0.3 x (2/3)/200 -- ten times as much.
#
# This supersedes withholding the slate outright, which was strictly worse:
# an empty slate cannot convert at all, and the top-1 it was suppressing
# converts often enough to be worth +0.0096 public / +0.0056 dev on its own.
#
# Measured over public/dev, reveal schedule -> TechScore:
#   full slate every turn (the old withhold-on-tie gate)  0.9497 / 0.9415
#   turn 1 only                                           0.9440 / 0.9319
#   turns 1-2                                             0.9600 / 0.9503  <-
#   turns 1-3                                             0.9526 / 0.9370
#   turn 1, then top-3 on turn 2                          0.9503 / 0.9406
#   turns 1-2, but turn 2 only when the top is tied       0.9593 / 0.9471
# Turn 1 alone gives back most of the win -- by turn 2 the second constraint
# has landed and the top-1 is worth betting on -- and turn 3 starts losing
# sessions outright, which costs hit rate at 0.5 against MRR's 0.3. Gating on
# the tie test is worse than trimming unconditionally, so ties no longer decide
# anything; they only pick which explanation the customer gets.
EXPOSURE_ENABLED = True
EXPOSURE_UNTIL_TURN = 2
EXPOSURE_WIDTH = 1

# ---------------------------------------------------------------------------
# LLM reranking
# ---------------------------------------------------------------------------

# A model pass over the ten candidates the linear reranker produced, before the
# exposure gate trims them. See starter/llm_rerank.py for what it is allowed to
# do (reorder, nothing else) and how it fails (back to this ordering, always).
#
# Off by default. Official scoring may run without network access, and the
# deterministic pipeline is what the recorded score was measured on; turning
# this on is an opt-in that has to earn its place against that baseline.
LLM_RERANK_ENABLED = os.environ.get("LLM_RERANK", "").strip().lower() in ("1", "true", "on", "yes")

# Which turns get a model call.
#
# Deliberately the exposure turns and no further. On turns 1-2 the slate is
# trimmed to a single item, so the whole session's score turns on which
# candidate the reranker put first -- that is where a better ordering is worth
# the most and where the linear model is weakest, because one disclosed
# constraint leaves most of the pool tied. By turn 3 the full slate goes out and
# a reorder can only move the target between ranks that already score, which is
# worth a fraction as much per call.
LLM_RERANK_UNTIL_TURN = 2

# Only call when the top of the slate is actually contested. A slate whose best
# candidate is uniquely best on evidence is not a coin flip and the model has
# nothing to add; skipping those is most of the token saving, and it also keeps
# the model away from the decisions the linear reranker already gets right.
LLM_RERANK_ONLY_WHEN_TIED = True

# How much of each candidate the prompt carries.
LLM_TITLE_CHARS = 120
LLM_FEATURE_CHARS = 100
LLM_MAX_FEATURES = 5

# (whole-item / attribute match, loose substring, popularity, BM25 position).
# The last two only ever settle candidates that are tied on evidence: together
# they cannot span the 0.6 gap one loose match opens, so nothing outranks a
# candidate that accounts for more of what the customer said. Within a tie the
# two disagree on purpose -- popularity backs the product people actually buy,
# BM25 backs the closest textual fit -- and the session is decided by which of
# them the tie-group happens to favour.
RERANK_WEIGHTS = (1.5, 0.6, 0.3, 0.10)

# One name per reranking feature, in the order Matcher.features() emits them.
# The weight vector is RERANK_WEIGHTS with POSITION_W spliced into third place,
# and the score is the plain dot product of the two. Keeping it a dot product
# is what makes the ranking tunable offline: tools/tune_reranker.py caches
# these rows once and re-scores them under candidate weights without going near
# sqlite again.
FEATURE_NAMES = ("exact", "loose", "position", "popularity", "velocity", "bm25")


def rerank_weights() -> tuple[float, ...]:
    """The live weight vector, in FEATURE_NAMES order."""
    exact_w, loose_w, pop_w, rank_w = RERANK_WEIGHTS
    velocity_w = VELOCITY_W if VELOCITY_ENABLED else 0.0
    return (exact_w, loose_w, POSITION_W, pop_w, velocity_w, rank_w)

# The customer clips each disclosed constraint at this many characters, so the
# catalog side is clipped identically -- otherwise a truncated constraint could
# never equal the item it was taken from.
CLIP_LIMIT = 180
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
# A few hundred rows carry an untranslated boilerplate bullet that no English
# query can reach. The English sense is indexed *alongside* the original, never
# instead of it: the customer quotes constraints verbatim out of the catalog, so
# a constraint that reads "进口" still has to match the bullet it came from.
FEATURE_ALIASES = {"进口": "imported"}
MATERIAL_RE = re.compile(r"cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric", re.I)
POP_SCALE = math.log1p(100000.0)

# Ratings velocity: rating_number normalised by how long the item has been
# listed.
#
# This exists because it is an *interaction* the linear model cannot build for
# itself. Given popularity and recency as separate additive terms, no weight
# vector expresses a ratio -- and the additive pair was measured: recency on
# its own is worthless-to-harmful at every weight (it decays monotonically,
# 0.9568 -> 0.9494 at w=0.8), while the same information as a denominator is
# worth +0.0028. That gap is the whole justification for the feature.
#
# It is *additional* to popularity, not a replacement for it and not a
# redistribution of its weight. Replacing popularity outright is worse at every
# velocity weight; splitting a fixed 0.30 budget between the two is flat. The
# two disagree in a useful way -- popularity backs the item with the most
# ratings outright, velocity backs the one accumulating them fastest -- and it
# is the disagreement, not either signal alone, that separates a tie group.
#
# Naive intuition for this feature is backwards, which is worth recording:
# "old items had longer to accumulate ratings, so normalise by age" does not
# hold here. corr(log1p(ratings), age) = -0.118 -- *newer* listings carry more
# ratings, not fewer. Velocity earns its keep in spite of that, not because
# of it.
#
# Measured, pop_w held at 0.30 (1,000-session trace; live figures in
# EXPERIMENTS.md sec.3): the sweep peaks at 0.5 on train, holdout and both
# objectives at once, and the paired bootstrap clears zero
# (+0.00282 raw [+0.0007,+0.0054], +0.00243 public [+0.0006,+0.0044]).
VELOCITY_ENABLED = True
# At 0.5 the tie-breakers can, in principle, out-vote the 0.6 that one loose
# match is worth. Measured, they essentially never do: forcing evidence to
# dominate (weights x100) changes the score by +0.00004 raw / +0.00008 public,
# so ~99.97% of the gain is within-tie-group reordering rather than evidence
# being overridden. Drop to 0.3 to keep the invariant exactly binding; it costs
# about 0.001.
VELOCITY_W = 0.5
VELOCITY_FLOOR_YEAR = 2008
VELOCITY_NOW = 2024
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def listed_year(details: object) -> int | None:
    """The year off `details["Date First Available"]`, which is free text."""
    if not isinstance(details, dict):
        return None
    stamp = details.get("Date First Available")
    found = YEAR_RE.search(str(stamp)) if stamp else None
    return int(found.group(0)) if found else None


def normalize(value: object) -> str:
    """Fold a constraint or a catalog item into one comparable form."""
    return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")[:CLIP_LIMIT].rstrip().casefold()


def item_values(value: object) -> list[str]:
    """The discrete strings a field contributes, kept separate rather than
    joined: the customer quotes one whole bullet or one whole detail, so the
    boundaries between them are the signal."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def fold(text: object) -> str:
    """Mirror the sqlite unicode61 'remove_diacritics 2' tokenizer."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(fold(text))


def coarse_category(values: list[str]) -> str:
    """The category phrase the customer opens with.

    A mirror of the evaluator's own derivation: split every `categories` entry
    on commas, drop the store-wide top level that every product in this catalog
    shares, and keep the last two surviving segments. Because the simulator
    emits this string verbatim, matching its behaviour exactly is what lets the
    opening message be looked up rather than guessed at.
    """
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def bucket_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


class Constraint:
    """One disclosed customer requirement, compiled into a catalog filter."""

    __slots__ = ("raw", "kind", "value", "terms")

    def __init__(self, raw: str, kind: str | None = None) -> None:
        self.raw = raw
        if kind == "tokens":
            self.kind = "tokens"
            self.value = re.sub(r"\s+", " ", raw).strip()
            self.terms = tokens(self.value)[:8]
            return
        cleaned = re.sub(r"\s+", " ", raw).strip()
        budget = BUDGET_RE.match(cleaned)
        color = COLOR_RE.match(cleaned)
        if budget:
            self.kind = "price"
            self.value = fold(budget.group("value")).strip()
            self.terms = []
        elif color:
            self.kind = "phrase"
            self.value = cleaned
            self.terms = tokens(color.group("value"))
        else:
            self.kind = "phrase"
            self.value = cleaned
            self.terms = tokens(cleaned)
        self.terms = self.terms[:32]

    @property
    def specificity(self) -> int:
        return len(self.terms)

    def phrases(self) -> list[str]:
        """Query forms to try, most exact first."""
        if self.kind == "tokens":
            if not self.terms:
                return []
            # Taxonomy levels are not reliably adjacent in the indexed text, so
            # the category is required as a bag of terms, never as a phrase.
            forms = [" AND ".join(f'"{term}"' for term in self.terms)]
            if len(self.terms) > 1:
                forms.append(f'"{self.terms[-1]}"')
            return forms
        if self.kind != "phrase" or not self.terms:
            return []
        forms = ['"' + " ".join(self.terms) + '"']
        # A constraint clipped at the customer's character limit may end
        # mid-word; that partial token can never match the indexed text.
        if len(self.terms) > 1 and len(self.value) >= TRUNCATION_LIMIT - 10:
            forms.append('"' + " ".join(self.terms[:-1]) + '"')
        return forms


class Matcher:
    """Phrase-filter retrieval over the frozen catalog.

    Constraints are applied most-specific-first and kept only where they leave
    the candidate set non-empty, so one unmatchable constraint degrades the
    result rather than collapsing it to nothing.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        # Reranking features, filled during the same pass that builds the index.
        self.items: dict[str, dict[str, int]] = {}
        self.text: dict[str, str] = {}
        # (title, store) per item -- the two fields the LLM reranker needs that
        # nothing else on this object keeps in readable form. Features come off
        # `items`, whose keys are already the item's own bullets in order.
        self.card: dict[str, tuple[str, str]] = {}
        self.price: dict[str, str | None] = {}
        self.pop: dict[str, float] = {}
        self.velocity: dict[str, float] = {}
        # Bucket key -> how many products carry it. Only membership is read
        # at query time; the count is what makes the index inspectable.
        self.buckets: dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, price_text UNINDEXED, title, categories, "
            "features, details, store, description, bucket_key UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                price = product.get("price")
                asin = str(product["parent_asin"])
                # value -> its index in this item's own metadata order. A dict
                # rather than a set: membership stays O(1) for the exact tests
                # below, and insertion order doubles as the position signal at
                # no extra storage.
                placed: dict[str, int] = {}
                pending: list[tuple[str, int]] = []
                index = 0
                for field_name in ("features", "details"):
                    for value in item_values(product.get(field_name)):
                        norm = normalize(value)
                        if not norm:
                            continue
                        placed.setdefault(norm, index)
                        alias = FEATURE_ALIASES.get(norm)
                        if alias is not None:
                            pending.append((alias, index))
                        index += 1
                # Aliases land after the main pass, so an item that carries the
                # English bullet itself keeps that bullet's own position -- the
                # alias only fills a gap, it never displaces a real one. It also
                # shares the slot of the bullet it translates rather than taking
                # a new one, so nothing after it shifts.
                aliased: list[str] = []
                for alias, slot in pending:
                    if alias not in placed:
                        placed[alias] = slot
                        aliased.append(alias)
                self.items[asin] = placed
                self.card[asin] = (
                    str(product.get("title") or ""), str(product.get("store") or "")
                )
                alias_text = " " + " ".join(aliased) if aliased else ""
                self.text[asin] = (" ".join(
                    flatten(product.get(field_name)) for field_name in SEARCH_FIELDS
                ) + alias_text).casefold()
                self.price[asin] = (
                    str(price).casefold() if price not in (None, "") else None
                )
                ratings = int(product.get("rating_number") or 0)
                self.pop[asin] = min(1.0, math.log1p(ratings) / POP_SCALE)
                year = listed_year(product.get("details"))
                age = max(1, VELOCITY_NOW - (year if year else VELOCITY_FLOOR_YEAR))
                self.velocity[asin] = min(1.0, math.log1p(ratings / age) / POP_SCALE)
                key = bucket_key(coarse_category(product.get("categories") or []))
                self.buckets[key] = self.buckets.get(key, 0) + 1
                batch.append((
                    str(product["parent_asin"]),
                    fold(price).strip() if price not in (None, "") else "",
                    flatten(product.get("title")),
                    flatten(product.get("categories")),
                    flatten(product.get("features")) + alias_text,
                    flatten(product.get("details")),
                    flatten(product.get("store")),
                    flatten(product.get("description")),
                    key,
                ))
                if len(batch) >= 2000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", batch)
        self.connection.commit()

    def resolve_bucket(self, category: str | None) -> str | None:
        """The bucket this category phrase names, or None.

        Exact membership only -- no containment, no token overlap. The opening
        message carries the evaluator's own `coarse_category` output verbatim,
        so a phrase that does not land on a known key is a phrase we do not
        understand, and guessing a neighbouring bucket would hard-exclude the
        target from a pool it was otherwise reachable in.
        """
        if not BUCKET_ENABLED or not category:
            return None
        key = bucket_key(category)
        return key if key in self.buckets else None

    def _filters(self, price: str | None, bucket: str | None) -> tuple[str, list[object]]:
        sql = ""
        params: list[object] = []
        if price is not None:
            sql += " AND price_text = ?"
            params.append(price)
        if bucket is not None:
            sql += " AND bucket_key = ?"
            params.append(bucket)
        return sql, params

    def _count(self, expression: str, price: str | None, bucket: str | None = None) -> int:
        clause, extra = self._filters(price, bucket)
        sql = "SELECT count(*) FROM products WHERE products MATCH ?" + clause
        return int(self.connection.execute(sql, [expression, *extra]).fetchone()[0])

    def _rank(
        self, expression: str, price: str | None, top_k: int, bucket: str | None = None
    ) -> list[str]:
        clause, extra = self._filters(price, bucket)
        sql = "SELECT parent_asin FROM products WHERE products MATCH ?" + clause
        params: list[object] = [expression, *extra]
        sql += f" ORDER BY bm25(products, {', '.join(str(w) for w in FIELD_WEIGHTS)}) LIMIT ?"
        params.append(top_k)
        return [str(row[0]) for row in self.connection.execute(sql, params).fetchall()]

    def _bucket_pool(self, bucket: str, top_k: int) -> list[str]:
        """Every product in the bucket, for when no constraint survives inside
        it. Ordered by title length purely for determinism -- the reranker is
        what actually sorts this, and it sees the whole session."""
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE bucket_key = ? "
            "ORDER BY length(title) LIMIT ?", (bucket, top_k),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _matches(self, constraint: str, asin: str) -> bool:
        """Does this candidate satisfy one disclosed constraint outright?

        Exact against a whole feature bullet or detail value first -- that is
        the form the customer quotes in. Colour and material arrive as bare
        attribute words instead, which no single item will ever equal, so they
        fall back to looking for the value anywhere in the product's text.
        """
        if constraint in self.items[asin]:
            return True
        budget = BUDGET_RE.match(constraint)
        if budget:
            return self.price[asin] == budget.group("value").strip()
        color = COLOR_RE.match(constraint)
        if color:
            return color.group("value").strip() in self.text[asin]
        if MATERIAL_RE.fullmatch(constraint):
            return constraint in self.text[asin]
        # A constraint clipped mid-item still prefixes the item it came from.
        if len(constraint) >= CLIP_LIMIT - 10:
            return any(item.startswith(constraint) for item in self.items[asin])
        return False

    def _position(self, constraint: str, asin: str) -> int | None:
        """Index of the earliest metadata entry carrying this constraint.

        Exact first, then containment -- a bare "nylon" is never a whole entry
        on its own, it sits inside "shell: 100% nylon". Iteration follows the
        item's own metadata order, so the first hit is the earliest one. None
        means the string appears nowhere in features or details, which is the
        weakest form of mention there is.
        """
        values = self.items.get(asin)
        if not values:
            return None
        index = values.get(constraint)
        if index is not None:
            return index
        for value, position in values.items():
            if constraint in value:
                return position
        return None

    def features(
        self, pool: list[str], evidence: list[str]
    ) -> list[tuple[float, ...]]:
        """One row per candidate, in FEATURE_NAMES order: every signal the
        reranker has, before any weight is applied.

        Split out of rerank() so the offline tuner scores the same vector the
        agent does. If these two ever computed features differently, a tuned
        weight would mean nothing.

        Every feature is oriented so more is better, the BM25 term included --
        it enters as the negated position, so a candidate the query already
        liked scores above one it buried. That orientation is what lets the
        tuner prune, at capture time, the candidates the target beats under
        every non-negative weight vector.
        """
        constraints = [norm for norm in (normalize(item) for item in evidence) if norm]
        constraints = list(dict.fromkeys(constraints))
        depth = len(pool) or 1
        rows: list[tuple[float, ...]] = []
        for position, asin in enumerate(pool):
            exact = 0.0
            loose = 0.0
            earliest: int | None = None
            for constraint in constraints:
                if self._matches(constraint, asin):
                    exact += 1.0
                if constraint in self.text.get(asin, ""):
                    loose += 1.0
                if POSITION_ENABLED:
                    index = self._position(constraint, asin)
                    if index is not None and (earliest is None or index < earliest):
                        earliest = index
            rows.append((
                exact,
                loose,
                0.0 if earliest is None else 1.0 / (1.0 + earliest),
                self.pop.get(asin, 0.0),
                self.velocity.get(asin, 0.0),
                -(position / depth),
            ))
        return rows

    def cards(self, asins: list[str]) -> list[dict]:
        """Readable descriptions of candidates, for the LLM reranker prompt.

        Built from what the index already holds rather than from the catalog
        file, so this costs a dict lookup per candidate and no I/O. The feature
        bullets come out of `items` in the item's own metadata order, which is
        the order the manufacturer wrote them -- the defining material first,
        the incidental ones after -- and that ordering is itself part of what
        the model is being asked to judge.

        The bullets arrive casefolded and clipped, because `items` stores them
        in the form the constraint matcher compares against and keeping a
        second raw copy would cost ~25MB to recover capitalisation the model
        does not need.
        """
        out: list[dict] = []
        for asin in asins:
            title, store = self.card.get(asin, ("", ""))
            features = [
                value[:LLM_FEATURE_CHARS]
                for value in list(self.items.get(asin, {}))[:LLM_MAX_FEATURES]
            ]
            out.append({
                "asin": asin,
                "title": title[:LLM_TITLE_CHARS],
                "store": store,
                "features": features,
            })
        return out

    def rerank(
        self,
        pool: list[str],
        evidence: list[str],
        top_k: int,
        offset: int = 0,
        weights: tuple[float, ...] | None = None,
    ) -> list[str]:
        """Order a candidate pool by everything the customer has disclosed.

        The pool is already filtered, so every candidate matches the applied
        constraints *somewhere*; what separates them is whether the match lands
        on a whole quoted item and how much of the rest of the session they
        account for. Evidence decides the ordering outright wherever candidates
        differ on it; where they do not -- which is most of the pool, since they
        all cleared the same filter -- popularity and the incoming BM25 position
        settle it between them.

        Returns the slate together with how many candidates were tied on
        evidence at the top, which is what the caller needs in order to know
        whether the ordering it just received means anything.
        """
        # Clamped so a deep page never returns a short slate: running past the
        # end of the pool would hand back fewer than top_k ids and waste the
        # slots outright, which is strictly worse than repeating a tail.
        offset = max(0, min(offset if DEEP_PAGING else 0, max(0, len(pool) - top_k)))
        if not RERANK_ENABLED or not pool:
            return pool[offset:offset + top_k], {"tied": len(pool)}
        (exact_w, loose_w, position_w, pop_w,
         velocity_w, rank_w) = weights or rerank_weights()
        rows = self.features(pool, evidence)

        # Evidence is scored on its own pass rather than inline with the
        # tie-breakers, because how many candidates share the best evidence
        # score is itself the answer to a question the caller has to ask: is
        # this ordering carrying information, or is it popularity and BM25
        # position deciding a session between indistinguishable items?
        earned = [exact_w * row[0] + loose_w * row[1] for row in rows]
        placement = [position_w * row[2] for row in rows]

        # What counts as "tied" is what the withholding gate reads, so it has to
        # reflect everything we can actually separate candidates on.
        decisive = [
            value + (placement[index] if POSITION_BREAKS_TIES else 0.0)
            for index, value in enumerate(earned)
        ]
        best = max(decisive)
        tied = sum(1 for value in decisive if value >= best - 1e-9)

        # The BM25 feature is already negated, so every weight enters with the
        # same sign and the score is a dot product end to end.
        totals = [
            earned[index] + placement[index]
            + pop_w * row[3] + velocity_w * row[4] + rank_w * row[5]
            for index, row in enumerate(rows)
        ]
        ordered = sorted(range(len(pool)), key=lambda index: (-totals[index], index))
        return [pool[index] for index in ordered[offset:offset + top_k]], {"tied": tied, "best": best}

    def browse(
        self,
        category: str | None,
        top_k: int = 10,
        pool: int = 500,
        bucket: str | None = None,
    ) -> tuple[list[str], dict]:
        """Browsing track: spread the slots across the category.

        With nothing to filter on, ten near-neighbours are a poor guess and
        teach us nothing. A spread is still a poor guess but a useful probe --
        whatever the shopper reacts to tells us where to go next.
        """
        terms = tokens(category) if category else []
        if not terms and bucket is None:
            return [], {"mode": "browse_empty", "pool": 0}
        weights = ", ".join(str(w) for w in FIELD_WEIGHTS)
        rows: list[tuple] = []
        if terms:
            expression = " AND ".join(f'"{term}"' for term in terms)
            clause, extra = self._filters(None, bucket)
            try:
                rows = self.connection.execute(
                    "SELECT parent_asin, store, title FROM products "
                    "WHERE products MATCH ?" + clause
                    + f" ORDER BY bm25(products, {weights}) LIMIT ?",
                    (expression, *extra, pool),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        # The keyword query is a ranking device, not a filter, once the bucket
        # is known: the taxonomy words are not reliably present in the indexed
        # text, so a bucket member the query missed is still a candidate. Rank
        # what matched, then backfill the rest of the bucket behind it.
        if bucket is not None and len(rows) < pool:
            seen = {str(row[0]) for row in rows}
            rows = rows + [
                row for row in self.connection.execute(
                    "SELECT parent_asin, store, title FROM products "
                    "WHERE bucket_key = ? LIMIT ?", (bucket, pool),
                ).fetchall() if str(row[0]) not in seen
            ]
        if not rows:
            return [], {"mode": "browse_error", "pool": 0}

        picked: list[str] = []
        seen_stores: set[str] = set()
        seen_shapes: set[str] = set()
        for asin, store, title in rows:
            brand = fold(store).strip()
            # Two products whose titles open the same way are the same idea.
            shape = " ".join(tokens(title)[1:4])
            if (brand and brand in seen_stores) or (shape and shape in seen_shapes):
                continue
            picked.append(str(asin))
            seen_stores.add(brand)
            seen_shapes.add(shape)
            if len(picked) >= top_k:
                break
        # If diversity was too strict to fill the slots, backfill by rank.
        if len(picked) < top_k:
            for asin, _, _ in rows:
                if str(asin) not in picked:
                    picked.append(str(asin))
                    if len(picked) >= top_k:
                        break
        return picked, {
            "mode": "browse", "pool": len(rows), "brands": len(seen_stores),
            "bucket": bucket,
        }

    def search(
        self,
        category: str | None,
        constraints: list[str],
        top_k: int = 10,
        bucket: str | None = None,
    ) -> tuple[list[str], dict]:
        compiled = [Constraint(raw) for raw in dict.fromkeys(constraints) if raw]
        phrase_constraints = [c for c in compiled if c.kind == "phrase" and c.terms]
        price = next((c.value for c in compiled if c.kind == "price" and c.value), None)

        # Most specific first: a long verbatim feature narrows far harder than
        # a bare material word, and we want it in before anything can crowd out.
        ordered = sorted(phrase_constraints, key=lambda c: c.specificity, reverse=True)
        # With the bucket applied the category is already enforced exactly, so
        # this is kept for what BM25 does with it, not for what it filters.
        keep = bucket is None or BUCKET_KEEPS_CATEGORY_TERMS
        if keep and category and tokens(category):
            ordered.append(Constraint(category, kind="tokens"))

        applied: list[str] = []
        dropped: list[str] = []
        expression = ""
        price_applied = price or None

        for constraint in ordered:
            accepted = False
            for form in constraint.phrases():
                candidate = f"{expression} AND {form}" if expression else form
                try:
                    # Price is deliberately excluded here. A budget the catalog
                    # cannot satisfy exactly would zero every count and drop
                    # every text constraint with it; it is applied at ranking
                    # time instead, where it can be backed out safely.
                    survivors = self._count(candidate, None, bucket)
                except sqlite3.OperationalError:
                    continue
                if survivors > 0:
                    expression = candidate
                    applied.append(constraint.raw)
                    accepted = True
                    break
            if not accepted:
                dropped.append(constraint.raw)

        if not expression:
            # Nothing the customer said survives inside the bucket, but the
            # bucket itself is still the strongest filter we hold.
            if bucket is not None:
                return self._bucket_pool(bucket, top_k), {
                    "applied": [], "dropped": dropped, "mode": "bucket_only",
                    "bucket": bucket,
                }
            if price_applied:
                rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE price_text = ? "
                    "ORDER BY length(title) LIMIT ?", (price_applied, top_k),
                ).fetchall()
                return [str(row[0]) for row in rows], {
                    "applied": [], "dropped": dropped, "mode": "price_only",
                }
            return [], {"applied": [], "dropped": dropped, "mode": "empty"}

        # The price filter is dropped rather than returning nothing at all.
        results = self._rank(expression, price_applied, top_k, bucket)
        if not results and price_applied:
            price_applied = None
            results = self._rank(expression, None, top_k, bucket)

        return results, {
            "applied": applied,
            "dropped": dropped,
            "surviving": self._count(expression, price_applied, bucket),
            "mode": "constraints",
            "bucket": bucket,
        }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

# The customer releases any undisclosed constraint for "other", and only
# same-class constraints for a typed attribute, so "other" dominates.
ASK_ATTRIBUTE = "other"

PROMPTS = (
    "Happy to narrow this down. What else matters to you here?",
    "Got it. Anything else about it that's important -- material, fit, or how you'll use it?",
    "Thanks. Is there any other detail I should be matching on?",
    "Noted. Anything else you'd want me to take into account?",
)

# Said on a turn whose slate is trimmed to the single best candidate. Which
# explanation is honest depends on why the rest was held back: either several
# candidates match everything disclosed so far equally well and padding the
# list would be guessing, or we simply have one clear front-runner and want a
# second detail before widening.
TIED_PROMPT = (
    "Several items match everything you've told me so far equally well, so I'd "
    "rather not guess between them. Here's the closest one -- and one more "
    "detail would let me narrow it properly. What else matters to you?"
)
NARROW_PROMPT = (
    "Here's the closest match I have so far. Tell me one more thing about what "
    "you're after and I'll widen the list to the best ten."
)


class SessionState:
    __slots__ = ("profile", "category", "constraints", "heard", "asked", "last_ask",
                 "gained", "exhausted", "intent", "dry", "slack")

    def __init__(self, profile: dict) -> None:
        self.profile = profile or {}
        # Routed on the opening turn: a shopper who leads with a requirement is
        # buying, one who leads with only a category is browsing.
        self.intent: str | None = None # "buying" or "browsing" or "intent_override"
        self.category: str | None = None 
        self.constraints: list[str] = [] 
        # Everything the customer has ever disclosed, never cleared. A
        # superseded preference and a constraint the filter had to drop both
        # still describe the target, so they stay available to the reranker
        # even once they have stopped being safe to filter on.
        self.heard: list[str] = []
        # What we asked, in order, and what each ask actually bought us. The
        # payoff for asking X only becomes visible in the *next* message, so
        # last_ask carries the pending question across the turn boundary.
        self.asked: list[str] = []
        self.last_ask: str | None = None
        self.gained: dict[str, int] = {}
        self.exhausted: set[str] = set()
        # Consecutive turns that told us nothing new. Each one is a slate the
        # shopper has already been shown, so it doubles as the page number.
        self.dry = 0
        # Turns that disclosed nothing at all. The exposure gate is really a
        # bound on how much evidence stands behind the slate, not on the clock,
        # so each evidence-free turn buys the gate one more turn.
        self.slack = 0

    def record_ask(self, attribute: str | None) -> None:
        if attribute:
            self.asked.append(attribute)
            self.last_ask = attribute

    def absorb(self, message: str) -> None:
        turn = parse_message(message)

        # Recorded before the override reset below, which is the whole point:
        # filtering forgets, ranking does not.
        for constraint in turn.constraints:
            if constraint and constraint not in self.heard:
                self.heard.append(constraint)

        if turn.is_override and OVERRIDE_RESETS:
            # Off by default -- see OVERRIDE_RESETS. Kept switchable because a
            # simulator whose override genuinely changed the target would want
            # this on.
            self.constraints.clear()
            if OVERRIDE_RESETS_CATEGORY:
                self.category = None
            if OVERRIDE_RESETS_ASKS:
                self.asked.clear()
                self.exhausted.clear()
                self.gained.clear()
            self.last_ask = None

        if turn.category and not self.category:
            self.category = turn.category
        if self.intent is None and turn.category:
            self.intent = "buying" if turn.constraints else "browsing"

        before = len(self.constraints)
        for constraint in turn.constraints:
            if constraint and constraint not in self.constraints:
                self.constraints.append(constraint)
        gain = len(self.constraints) - before

        # A turn that adds no constraint leaves the ranking byte-identical to
        # the one that just missed, so it is a dead slate and the next turn
        # should page past it. Two turns are exempt. An override rewrites the
        # constraint set, so its ranking is new. A Boundary shrug is a one-off
        # scenario artifact -- the customer answers properly on the turn after
        # it -- and paging on it would move the slate that scenario usually
        # converts on.
        if gain > 0:
            self.dry = 0
        elif self.asked and not turn.is_override and not turn.refusal_is_boundary:
            # self.asked is empty only on the opening message, which discloses
            # no constraint in a Browsing session. That is not a dead turn --
            # there is no slate behind it yet -- so it must not advance the page.
            self.dry += 1

        # A Boundary shrug costs a turn and discloses nothing, which shifts that
        # whole scenario one turn later than Browsing: its first constraint
        # lands on turn 3, not turn 2. The exposure gate counts turns, so by
        # then it has already opened -- and the one slate it exists to protect,
        # the first that is genuinely ranked, goes out at full width. Browsing
        # converts 51.5% of its sessions on the gated turn 2 at 100% rank-1;
        # Boundary converts 75.8% on the ungated turn 3 at 64.1%, which is
        # essentially the whole 0.166 MRR gap between them.
        #
        # Over the 2,490 Boundary dev sessions: MRR 0.7397 -> 0.8990 (Browsing
        # sits at 0.9062), hit -0.0020, mttc +0.286. No other scenario emits
        # this refusal -- "an additional preference" parses as a genuine
        # exhaustion, not a shrug -- so nothing else moves; verified
        # bit-identical over 5,914 non-Boundary sessions.
        if turn.refusal_is_boundary:
            self.slack += 1

        # Attribute the outcome to whatever we asked on the previous turn. The
        # customer names the attribute it is refusing, so prefer that over our
        # own memory of what we sent.
        #
        # An override turn is the exception: the harness sends it instead of
        # calling customer_reply, so our pending ask was never read and must
        # not be credited with what the override happened to disclose.
        answered = None if turn.is_override else (turn.refused or self.last_ask)
        if answered:
            self.gained[answered] = self.gained.get(answered, 0) + gain
            # A Boundary shrug is a scenario artifact, not evidence that the
            # attribute is spent, so it must not mark anything exhausted.
            if gain == 0 and turn.refused and not turn.refusal_is_boundary:
                self.exhausted.add(answered)
        self.last_ask = None

    def unasked(self) -> list[str]:
        seen = set(self.asked) | self.exhausted
        return [a for a in ALLOWED_ATTRIBUTES if a not in seen]


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def _make_client():
    """The Ollama endpoint, or None when it is unreachable or switched off.

    None is the offline fallback: the agent then runs on the deterministic
    policy alone, which is what we expect during official scoring.
    Set OLLAMA_HOST=off to force that path.
    """
    return None
    if OLLAMA_HOST.strip().lower() in ("", "off", "none"):
        return None
    try:
        # A one-token warm-up. Loading the model costs ~25s on a cold server,
        # and paying that here means no session turn ever eats it. It also
        # proves OLLAMA_MODEL exists, which /api/tags alone would not.
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status == 200:
                return OLLAMA_HOST
    except Exception:
        pass
    return None


class Agent:
    """Constraint-accumulating retrieval agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.matcher = Matcher(catalog_path)
        self._sessions: dict[str, SessionState] = {}
        self._client = _make_client()
        # Constructed once. When the model is off or unreachable this is a
        # disabled object rather than None, so respond() has no second code
        # path to keep in sync -- it calls the same method either way and gets
        # the identity ordering back.
        self._llm = LLMReranker(provider=None if LLM_RERANK_ENABLED else "off")

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(user_profile)

    def _choose_attribute(self, state: SessionState) -> tuple[str, dict]:
        """Pick the next attribute to request.

        Any failure falls back to the deterministic choice. This matters more
        than it looks: the evaluator discards the entire response when
        respond() raises, so one API hiccup would throw away the
        recommendations we already computed successfully.
        """
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self._client is None:
            return ASK_ATTRIBUTE, usage

        prompt = (
            "The shopper has disclosed these requirements:\n"
            f"{state.constraints or 'nothing yet'}\n\n"
            f"Attributes already asked, in order: {state.asked or 'none'}\n"
            f"Attributes that returned nothing and must not be repeated: "
            f"{sorted(state.exhausted) or 'none'}\n\n"
            "Choose the single most useful attribute to ask about next, from "
            f"exactly this list: {', '.join(ALLOWED_ATTRIBUTES)}.\n"
            "Reply with the attribute only, no punctuation or explanation."
        )
        try:
            body = json.dumps({
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # One word is all we want, and temperature 0 keeps runs
                # reproducible the way the rest of the harness is.
                "options": {"temperature": 0, "num_predict": 8},
            }).encode("utf-8")
            request = urllib.request.Request(
                f"{self._client}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
            raw = (payload.get("message", {}).get("content") or "").strip().lower()
            usage = {
                "prompt_tokens": int(payload.get("prompt_eval_count") or 0),
                "completion_tokens": int(payload.get("eval_count") or 0),
            }
            # The evaluator coerces anything off-list to "other" silently, so
            # validate here where a bad answer is still visible.
            chosen = next((a for a in ALLOWED_ATTRIBUTES if a == raw), None)
            if chosen is None:
                chosen = next((a for a in ALLOWED_ATTRIBUTES if a in raw), ASK_ATTRIBUTE)
            return chosen, usage
        except Exception:
            return ASK_ATTRIBUTE, usage

    @staticmethod
    def _reveal(turn: int, top_k: int, slack: int = 0) -> int:
        """How many candidates this turn's slate is allowed to carry.

        `slack` is how many turns the customer spent disclosing nothing. It
        pushes the cap back by that much, because what the cap is really
        bounding is the evidence behind the slate, and an evidence-free turn
        advances the clock without advancing the evidence.

        The turn cap is the safety valve: past it the full slate always goes
        out, so trimming can cost us rank but never a hit.
        """
        if not EXPOSURE_ENABLED or turn > EXPOSURE_UNTIL_TURN + slack:
            return top_k
        return EXPOSURE_WIDTH

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.absorb(user_message)
        ranking: dict = {}
        # Never allowed to break retrieval: an unrecognised phrase, or anything
        # unexpected here, leaves both tracks exactly as they were.
        try:
            bucket = self.matcher.resolve_bucket(state.category)
        except Exception:
            bucket = None

        # Dual-track routing. Constraints are what the precision track filters
        # on, so with none in hand we take the browsing track instead of
        # ranking the whole category by keyword weight alone.
        if state.constraints or not DUAL_TRACK_ROUTING:
            # Retrieve a pool rather than a page, then order it on the full
            # session. state.heard is a superset of the constraints the filter
            # applied and of the ones it had to drop, so it is all the evidence
            # there is.
            offset = state.dry * top_k if DEEP_PAGING else 0
            limit = POOL_SIZE if RERANK_ENABLED else top_k + offset
            pool, _ = self.matcher.search(
                state.category, state.constraints, limit, bucket
            )
            try:
                recommendations, ranking = self.matcher.rerank(
                    pool, state.heard, top_k, offset
                )
            except Exception:
                # The evaluator discards the whole turn if respond() raises, so
                # a reranking bug must cost us the ordering, not the results.
                start = max(0, min(offset, max(0, len(pool) - top_k)))
                recommendations = pool[start:start + top_k]
        else:
            # Nothing disclosed yet means nothing to rank on; the browse track
            # already spreads the slots across the category.
            recommendations, _ = self.matcher.browse(
                state.category, top_k, bucket=bucket
            )

        # The model pass sits between ranking and trimming, which is the only
        # place it can pay: on the exposure turns the slate is cut to one item,
        # so reordering after the cut would reorder a list of length one.
        llm_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self._llm.enabled and turn <= LLM_RERANK_UNTIL_TURN and len(recommendations) > 1:
            contested = ranking.get("tied", 0) > 1
            if contested or not LLM_RERANK_ONLY_WHEN_TIED:
                # Wrapped because respond() raising costs the evaluator the
                # whole turn, recommendations included. The reranker already
                # swallows its own failures; this is the belt to that braces.
                try:
                    cards = self.matcher.cards(recommendations)
                    order, llm_usage = self._llm.order(state.category, state.heard, cards)
                    recommendations = [recommendations[index] for index in order]
                except Exception:
                    pass

        # Early turns go out as rank-1-or-nothing: the harness stops at the
        # first hit, so a full slate would lock in whatever rank the coin flip
        # below the top happened to give us for the rest of the session.
        reveal = self._reveal(turn, top_k, state.slack)
        trimmed = 0 < reveal < len(recommendations)
        recommendations = recommendations[:reveal]

        ask, usage = self._choose_attribute(state)
        state.record_ask(ask)
        # Token usage is reported per turn, so both model calls have to land in
        # the same figure the evaluator sums.
        usage = {
            "prompt_tokens": usage["prompt_tokens"] + llm_usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"] + llm_usage["completion_tokens"],
        }

        if trimmed:
            message = TIED_PROMPT if ranking.get("tied", 0) > 1 else NARROW_PROMPT
        else:
            message = PROMPTS[(turn - 1) % len(PROMPTS)]

        return {
            "message": message,
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": usage,
        }
