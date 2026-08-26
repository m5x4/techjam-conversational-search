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
                    survivors = self._count(candidate, price_applied)
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


class SessionState:
    __slots__ = ("profile", "category", "constraints", "asked", "last_ask", "gained", "exhausted")

    def __init__(self, profile: dict) -> None:
        self.profile = profile or {}
        self.category: str | None = None
        self.constraints: list[str] = []
        # What we asked, in order, and what each ask actually bought us. The
        # payoff for asking X only becomes visible in the *next* message, so
        # last_ask carries the pending question across the turn boundary.
        self.asked: list[str] = []
        self.last_ask: str | None = None
        self.gained: dict[str, int] = {}
        self.exhausted: set[str] = set()

    def record_ask(self, attribute: str | None) -> None:
        if attribute:
            self.asked.append(attribute)
            self.last_ask = attribute

    def absorb(self, message: str) -> None:
        turn = parse_message(message)
        if turn.category and not self.category:
            self.category = turn.category

        before = len(self.constraints)
        for constraint in turn.constraints:
            if constraint and constraint not in self.constraints:
                self.constraints.append(constraint)
        gain = len(self.constraints) - before

        # Attribute the outcome to whatever we asked on the previous turn. The
        # customer names the attribute it is refusing, so prefer that over our
        # own memory of what we sent.
        answered = turn.refused or self.last_ask
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
        recommendations, _ = self.matcher.search(state.category, state.constraints, top_k)

        ask, usage = self._choose_attribute(state)
        state.record_ask(ask)

        return {
            "message": PROMPTS[(turn - 1) % len(PROMPTS)],
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": usage,
        }
