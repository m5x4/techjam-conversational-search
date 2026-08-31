"""Phrase-filter retrieval over the frozen catalog.

The simulated customer describes the target using strings lifted verbatim
from that product's own catalog entry, so each disclosed phrase is treated as
an exact-phrase filter: retrieval is elimination first, ranking second.
Constraints are applied most-specific-first and kept only where they leave
the candidate set non-empty, so one unmatchable constraint degrades the
result rather than collapsing it to nothing.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

from .config import *  # noqa: F401,F403  -- see config.__all__
from .text import (
    bucket_key,
    coarse_category,
    flatten,
    fold,
    item_values,
    listed_year,
    normalize,
    tokens,
)

BUDGET_RE = re.compile(r"^budget around \$(?P<value>.+)$", re.I)
COLOR_RE = re.compile(r"^colou?r:\s*(?P<value>.+)$", re.I)
MATERIAL_RE = re.compile(
    r"cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric", re.I
)


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
                alias_text = " " + " ".join(aliased) if aliased else ""
                self.text[asin] = (
                    " ".join(
                        flatten(product.get(field_name)) for field_name in SEARCH_FIELDS
                    )
                    + alias_text
                ).casefold()
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
                batch.append(
                    (
                        str(product["parent_asin"]),
                        fold(price).strip() if price not in (None, "") else "",
                        flatten(product.get("title")),
                        flatten(product.get("categories")),
                        flatten(product.get("features")) + alias_text,
                        flatten(product.get("details")),
                        flatten(product.get("store")),
                        flatten(product.get("description")),
                        key,
                    )
                )
                if len(batch) >= 2000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", batch
                    )
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

    def _filters(
        self, price: str | None, bucket: str | None
    ) -> tuple[str, list[object]]:
        sql = ""
        params: list[object] = []
        if price is not None:
            sql += " AND price_text = ?"
            params.append(price)
        if bucket is not None:
            sql += " AND bucket_key = ?"
            params.append(bucket)
        return sql, params

    def _count(
        self, expression: str, price: str | None, bucket: str | None = None
    ) -> int:
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
            "ORDER BY length(title) LIMIT ?",
            (bucket, top_k),
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

    def features(self, pool: list[str], evidence: list[str]) -> list[tuple[float, ...]]:
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
            rows.append(
                (
                    exact,
                    loose,
                    0.0 if earliest is None else 1.0 / (1.0 + earliest),
                    self.pop.get(asin, 0.0),
                    self.velocity.get(asin, 0.0),
                    -(position / depth),
                )
            )
        return rows

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
            return pool[offset : offset + top_k], {"tied": len(pool)}
        (exact_w, loose_w, position_w, pop_w, velocity_w, rank_w) = (
            weights or rerank_weights()
        )
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
            earned[index]
            + placement[index]
            + pop_w * row[3]
            + velocity_w * row[4]
            + rank_w * row[5]
            for index, row in enumerate(rows)
        ]
        ordered = sorted(range(len(pool)), key=lambda index: (-totals[index], index))
        return [pool[index] for index in ordered[offset : offset + top_k]], {
            "tied": tied,
            "best": best,
        }

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
                    "WHERE products MATCH ?"
                    + clause
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
                row
                for row in self.connection.execute(
                    "SELECT parent_asin, store, title FROM products "
                    "WHERE bucket_key = ? LIMIT ?",
                    (bucket, pool),
                ).fetchall()
                if str(row[0]) not in seen
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
            "mode": "browse",
            "pool": len(rows),
            "brands": len(seen_stores),
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
                    "applied": [],
                    "dropped": dropped,
                    "mode": "bucket_only",
                    "bucket": bucket,
                }
            if price_applied:
                rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE price_text = ? "
                    "ORDER BY length(title) LIMIT ?",
                    (price_applied, top_k),
                ).fetchall()
                return [str(row[0]) for row in rows], {
                    "applied": [],
                    "dropped": dropped,
                    "mode": "price_only",
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
