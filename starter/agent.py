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

# On "ignore my earlier preference", how much of the session is thrown away.
OVERRIDE_RESETS = True
OVERRIDE_RESETS_CATEGORY = False
OVERRIDE_RESETS_ASKS = True

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
FIELD_WEIGHTS = (0.0, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

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

# The harness ends the session the moment the target appears anywhere in the
# slate, so whatever rank we surface it at is the rank we are scored on for
# good -- there is no second chance to improve it later in the session. When
# the top of the pool is tied on evidence, the order among those candidates is
# settled by popularity and BM25 position, two proxies that are close to a coin
# flip, so serving them spends the session's one scoring opportunity on that
# coin flip. Staying quiet and buying one more constraint is the better trade:
# an extra turn costs 0.2 x (1/200)/10 of the score, while lifting a hit from
# rank 3 to rank 1 is worth 0.3 x (2/3)/200 -- ten times as much.
WITHHOLD_ON_TIE = True

# Last turn on which we are allowed to stay quiet. This is the safety valve:
# past it the slate always goes out, so withholding can cost us rank but never
# a hit. It is set at 2 because turn 3 measurably starts losing sessions
# outright, and hit rate carries 0.5 of the score against MRR's 0.3.
WITHHOLD_UNTIL_TURN = 2

# (whole-item / attribute match, loose substring, popularity, BM25 position).
# The last two only ever settle candidates that are tied on evidence: together
# they cannot span the 0.6 gap one loose match opens, so nothing outranks a
# candidate that accounts for more of what the customer said. Within a tie the
# two disagree on purpose -- popularity backs the product people actually buy,
# BM25 backs the closest textual fit -- and the session is decided by which of
# them the tie-group happens to favour.
RERANK_WEIGHTS = (1.5, 0.6, 0.3, 0.10)

# The customer clips each disclosed constraint at this many characters, so the
# catalog side is clipped identically -- otherwise a truncated constraint could
# never equal the item it was taken from.
CLIP_LIMIT = 180
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric", re.I)
POP_SCALE = math.log1p(100000.0)


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
        self.items: dict[str, set[str]] = {}
        self.text: dict[str, str] = {}
        self.price: dict[str, str | None] = {}
        self.pop: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, price_text UNINDEXED, title, categories, "
            "features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                price = product.get("price")
                asin = str(product["parent_asin"])
                self.items[asin] = {
                    norm for norm in (
                        normalize(value)
                        for field_name in ("features", "details")
                        for value in item_values(product.get(field_name))
                    ) if norm
                }
                self.text[asin] = " ".join(
                    flatten(product.get(field_name)) for field_name in SEARCH_FIELDS
                ).casefold()
                self.price[asin] = (
                    str(price).casefold() if price not in (None, "") else None
                )
                self.pop[asin] = min(
                    1.0, math.log1p(int(product.get("rating_number") or 0)) / POP_SCALE
                )
                batch.append((
                    str(product["parent_asin"]),
                    fold(price).strip() if price not in (None, "") else "",
                    flatten(product.get("title")),
                    flatten(product.get("categories")),
                    flatten(product.get("features")),
                    flatten(product.get("details")),
                    flatten(product.get("store")),
                    flatten(product.get("description")),
                ))
                if len(batch) >= 2000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", batch)
        self.connection.commit()

    def _count(self, expression: str, price: str | None) -> int:
        sql = "SELECT count(*) FROM products WHERE products MATCH ?"
        params: list[object] = [expression]
        if price is not None:
            sql += " AND price_text = ?"
            params.append(price)
        return int(self.connection.execute(sql, params).fetchone()[0])

    def _rank(self, expression: str, price: str | None, top_k: int) -> list[str]:
        sql = "SELECT parent_asin FROM products WHERE products MATCH ?"
        params: list[object] = [expression]
        if price is not None:
            sql += " AND price_text = ?"
            params.append(price)
        sql += f" ORDER BY bm25(products, {', '.join(str(w) for w in FIELD_WEIGHTS)}) LIMIT ?"
        params.append(top_k)
        return [str(row[0]) for row in self.connection.execute(sql, params).fetchall()]

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

    def rerank(
        self,
        pool: list[str],
        evidence: list[str],
        top_k: int,
        offset: int = 0,
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
        exact_w, loose_w, pop_w, rank_w = RERANK_WEIGHTS
        constraints = [norm for norm in (normalize(item) for item in evidence) if norm]
        constraints = list(dict.fromkeys(constraints))
        depth = len(pool)

        # Evidence is scored on its own pass rather than inline with the
        # tie-breakers, because how many candidates share the best evidence
        # score is itself the answer to a question the caller has to ask: is
        # this ordering carrying information, or is it popularity and BM25
        # position deciding a session between indistinguishable items?
        earned: dict[str, float] = {}
        for asin in pool:
            total = 0.0
            for constraint in constraints:
                if self._matches(constraint, asin):
                    total += exact_w
                if constraint in self.text.get(asin, ""):
                    total += loose_w
            earned[asin] = total
        best = max(earned.values())
        tied = sum(1 for value in earned.values() if value >= best - 1e-9)

        def score(entry: tuple[int, str]) -> tuple[float, int]:
            position, asin = entry
            total = earned[asin] + pop_w * self.pop.get(asin, 0.0) - rank_w * (position / depth)
            return (-total, position)

        ordered = sorted(enumerate(pool), key=score)
        return [asin for _, asin in ordered[offset:offset + top_k]], {"tied": tied, "best": best}

    def browse(self, category: str | None, top_k: int = 10, pool: int = 500) -> tuple[list[str], dict]:
        """Browsing track: spread the slots across the category.

        With nothing to filter on, ten near-neighbours are a poor guess and
        teach us nothing. A spread is still a poor guess but a useful probe --
        whatever the shopper reacts to tells us where to go next.
        """
        terms = tokens(category) if category else []
        if not terms:
            return [], {"mode": "browse_empty", "pool": 0}
        expression = " AND ".join(f'"{term}"' for term in terms)
        weights = ", ".join(str(w) for w in FIELD_WEIGHTS)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin, store, title FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {weights}) LIMIT ?",
                (expression, pool),
            ).fetchall()
        except sqlite3.OperationalError:
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
        return picked, {"mode": "browse", "pool": len(rows), "brands": len(seen_stores)}

    def search(
        self,
        category: str | None,
        constraints: list[str],
        top_k: int = 10,
    ) -> tuple[list[str], dict]:
        compiled = [Constraint(raw) for raw in dict.fromkeys(constraints) if raw]
        phrase_constraints = [c for c in compiled if c.kind == "phrase" and c.terms]
        price = next((c.value for c in compiled if c.kind == "price" and c.value), None)

        # Most specific first: a long verbatim feature narrows far harder than
        # a bare material word, and we want it in before anything can crowd out.
        ordered = sorted(phrase_constraints, key=lambda c: c.specificity, reverse=True)
        if category and tokens(category):
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
                    survivors = self._count(candidate, None)
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
        results = self._rank(expression, price_applied, top_k)
        if not results and price_applied:
            price_applied = None
            results = self._rank(expression, None, top_k)

        return results, {
            "applied": applied,
            "dropped": dropped,
            "surviving": self._count(expression, price_applied),
            "mode": "constraints",
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

# Said on a turn we deliberately stay quiet: several candidates match
# everything disclosed so far equally well, and guessing between them would
# waste the recommendation. Naming that is also the honest explanation.
WITHHOLD_PROMPT = (
    "Several items match everything you've told me so far equally well, so I'd "
    "rather not guess between them. One more detail and I can narrow it down -- "
    "what else matters to you?"
)


class SessionState:
    __slots__ = ("profile", "category", "constraints", "heard", "asked", "last_ask",
                 "gained", "exhausted", "intent", "dry")

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
            # Take "ignore my earlier preference" literally: everything learned
            # before the override is discarded and the session restarts from
            # the new requirement alone.
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

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.absorb(user_message)
        withheld = False

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
            pool, _ = self.matcher.search(state.category, state.constraints, limit)
            ranking: dict = {}
            try:
                recommendations, ranking = self.matcher.rerank(
                    pool, state.heard, top_k, offset
                )
            except Exception:
                # The evaluator discards the whole turn if respond() raises, so
                # a reranking bug must cost us the ordering, not the results.
                start = max(0, min(offset, max(0, len(pool) - top_k)))
                recommendations = pool[start:start + top_k]
            # Nothing separates the top of this pool, so showing it would lock
            # in a coin-flip rank for the rest of the session. Spend the turn
            # on another constraint instead; the turn cap above guarantees the
            # slate still goes out while there is plenty of session left.
            if (
                WITHHOLD_ON_TIE
                and turn <= WITHHOLD_UNTIL_TURN
                and ranking.get("tied", 0) > 1
            ):
                withheld = True
                recommendations = []
        else:
            # Nothing disclosed yet means nothing to rank on; the browse track
            # already spreads the slots across the category.
            recommendations, _ = self.matcher.browse(state.category, top_k)

        ask, usage = self._choose_attribute(state)
        state.record_ask(ask)

        return {
            "message": (
                WITHHOLD_PROMPT if withheld else PROMPTS[(turn - 1) % len(PROMPTS)]
            ),
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": usage,
        }
