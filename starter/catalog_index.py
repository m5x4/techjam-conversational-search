"""Catalog index: BM25 over SQLite FTS5, plus the per-product data the
linear rerank needs (flattened lower-cased blob, discrete feature/detail
value set, price/rating, a popularity prior) and the coarse-category bucket
map.

No ML / embedding dependency -- the agent runs fully offline. A dense
bi-encoder route and a local-LLM reranker were both prototyped on top of the
old fuse and empirically rejected (README: "Rejected approaches").
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from . import config
from .buckets import BucketIndex, coarse_category
from .config import AND_TERM_CAP, OR_TERM_CAP
from .text_utils import _field_value_strings, _norm, _segment, _text, _quote


def _seg_index(value: str) -> str:
    """Segmentation applied to the FTS5 column text at index time so the
    Unicode-aware query terms from ``_terms`` have something to match.

    Pure-ASCII text is passed through byte-for-byte -- the ``unicode61``
    token stream is provably unchanged, so every all-English product indexes
    exactly as before. A value with any non-ASCII character is folded and
    (for CJK / Kana / Hangul) split into overlapping character bigrams by
    ``_segment``, mirroring what ``_terms`` does to the disclosed
    constraint. This feeds ONLY the FTS5 columns; `blob` and `field_values`
    are built from the raw text and are left untouched (the rerank matches
    against them with its own Unicode-safe `_norm`)."""
    return value if value.isascii() else " ".join(_segment(value))


def _bm25_expr() -> str:
    """`bm25(products, w0, w1, ...)` built from config.BM25_COLUMN_WEIGHTS at
    call time -- so a sweep that reassigns starter.config.BM25_COLUMN_WEIGHTS
    is picked up on the next query without an index rebuild."""
    weights = ", ".join(repr(float(w)) for w in config.BM25_COLUMN_WEIGHTS)
    return f"bm25(products, {weights})"


class _CatalogIndex:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.category_text: dict[str, str] = {}
        # Numeric catalog fields (consumed by the popularity / hard-price
        # signals). price is stored ONLY when present and > 0 -- absence must
        # stay distinguishable from "$0".
        self.price: dict[str, float] = {}
        self.rating: dict[str, float] = {}
        self.rating_count: dict[str, int] = {}
        self.popularity: dict[str, float] = {}
        # Lower-cased flattened text per product, for verbatim phrase-match
        # (LOOSE term of Agent._rerank).
        self.blob: dict[str, str] = {}
        # Discrete (NOT flattened) feature/detail value strings per product,
        # each _norm-folded. The EXACT whole-field match term of Agent._rerank
        # tests constraint-equality against this set.
        self.field_values: dict[str, set[str]] = {}
        # coarse_category(...) -> parent_asin list. Populated by _build(),
        # wrapped in a BucketIndex below.
        self.coarse_category_by_id: dict[str, str] = {}
        self._bucket_map: dict[str, list[str]] = {}
        self._build()
        self.bucket_index = BucketIndex(self._bucket_map)

    # -- index construction -----------------------------------------------

    def _absorb_numeric(self, parent_asin: str, product: dict) -> None:
        """Pull price / average_rating / rating_number off a catalog row.

        price is kept only when it parses to a finite value > 0 -- a missing
        price must stay distinguishable from a genuine cheap item (only
        ~21% of the catalog carries a price at all). rating / rating_number
        default to 0 when absent.
        """
        raw_price = product.get("price")
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = 0.0
        if math.isfinite(price) and price > 0:
            self.price[parent_asin] = price

        try:
            rating = float(product.get("average_rating") or 0.0)
        except (TypeError, ValueError):
            rating = 0.0
        try:
            count = int(product.get("rating_number") or 0)
        except (TypeError, ValueError):
            count = 0
        self.rating[parent_asin] = rating
        self.rating_count[parent_asin] = count

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        bucket_map: dict[str, list[str]] = defaultdict(list)
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                coarse = coarse_category(product.get("categories")).lower()
                self.coarse_category_by_id[parent_asin] = coarse
                bucket_map[coarse].append(parent_asin)
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                # FTS5 columns get Unicode-segmented (no-op for ASCII rows);
                # category_text / blob / field_values below keep the raw text.
                batch.append(
                    (
                        parent_asin,
                        _seg_index(title),
                        _seg_index(categories),
                        _seg_index(features),
                        _seg_index(details),
                        _seg_index(store),
                        _seg_index(description),
                    )
                )
                self.category_text[parent_asin] = categories
                doc_text = " ".join(
                    (title, categories, features, details, store, description)
                )
                self.blob[parent_asin] = doc_text.lower()
                self.field_values[parent_asin] = {
                    norm
                    for norm in (
                        _norm(v)
                        for fld in ("features", "details")
                        for v in _field_value_strings(product.get(fld))
                    )
                    if norm
                }
                self._absorb_numeric(parent_asin, product)
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch
            )
        self.connection.commit()
        self._bucket_map = {key: ids for key, ids in bucket_map.items()}

        # Popularity prior: quality x volume. Real purchase targets skew
        # heavily-reviewed (catalog median rating_number ~12; the graded
        # targets' median is ~6.8k), so log1p(rating_number) is the dominant
        # term, lightly modulated by the mean rating.
        for parent_asin in self.blob:
            self.popularity[parent_asin] = self.rating.get(
                parent_asin, 0.0
            ) * math.log1p(self.rating_count.get(parent_asin, 0))

    # -- retrieval ------------------------------------------------------

    def bm25_search(
        self, and_terms: list[str], or_terms: list[str], limit: int
    ) -> list[str]:
        """Two-tier FTS5 search: a precise AND-phrase query, falling back to
        (and complemented by) a broad OR query. Returns a de-duplicated,
        rank-ordered id list (precise-query hits first)."""
        cursor = self.connection.cursor()
        results: list[str] = []
        seen: set[str] = set()

        if and_terms:
            phrase_expr = " ".join(_quote(t) for t in and_terms[:AND_TERM_CAP])
            try:
                rows = cursor.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    f"ORDER BY {_bm25_expr()} LIMIT ?",
                    (phrase_expr, limit),
                ).fetchall()
                for row in rows:
                    pid = str(row[0])
                    if pid not in seen:
                        seen.add(pid)
                        results.append(pid)
            except sqlite3.OperationalError:
                pass

        broad_terms = or_terms or and_terms
        unique_terms = list(dict.fromkeys(broad_terms))[:OR_TERM_CAP]
        if unique_terms:
            expression = " OR ".join(_quote(t) for t in unique_terms)
            try:
                rows = cursor.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    f"ORDER BY {_bm25_expr()} LIMIT ?",
                    (expression, limit),
                ).fetchall()
                for row in rows:
                    pid = str(row[0])
                    if pid not in seen:
                        seen.add(pid)
                        results.append(pid)
            except sqlite3.OperationalError:
                pass

        return results[:limit]

    def bm25_search_scoped(
        self,
        and_terms: list[str],
        or_terms: list[str],
        scope_ids: list[str],
        limit: int,
    ) -> list[str]:
        """Same two-tier FTS5 search as bm25_search, but with the MATCH run
        directly against `scope_ids` (a coarse-category bucket) via a joined
        temp table instead of a whole-catalog pass filtered down afterwards.

        Why this beats filter-after: the whole-catalog bm25_search caps at
        BM25_POOL (500) rows, so a bucket member ranked 500+ catalog-wide on a
        thin/generic query never enters the pool and has to be back-filled by
        a synthetic popularity tail with no term-relevance ordering. Scoping
        the query means every bucket member that shares a query term is ranked
        by its real in-bucket BM25 score, and `limit=len(scope_ids)` keeps the
        whole bucket in play -- no truncation, no popularity tail.
        """
        if not scope_ids:
            return []
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _scope (parent_asin TEXT PRIMARY KEY)"
        )
        cursor.execute("DELETE FROM _scope")
        cursor.executemany(
            "INSERT OR IGNORE INTO _scope(parent_asin) VALUES (?)",
            ((pid,) for pid in scope_ids),
        )

        results: list[str] = []
        seen: set[str] = set()

        def _run(match_expr: str) -> None:
            try:
                rows = cursor.execute(
                    "SELECT products.parent_asin FROM products "
                    "JOIN _scope ON _scope.parent_asin = products.parent_asin "
                    "WHERE products MATCH ? "
                    f"ORDER BY {_bm25_expr()} LIMIT ?",
                    (match_expr, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return
            for row in rows:
                pid = str(row[0])
                if pid not in seen:
                    seen.add(pid)
                    results.append(pid)

        if and_terms:
            _run(" ".join(_quote(t) for t in and_terms[:AND_TERM_CAP]))
        broad_terms = or_terms or and_terms
        unique_terms = list(dict.fromkeys(broad_terms))[:OR_TERM_CAP]
        if unique_terms:
            _run(" OR ".join(_quote(t) for t in unique_terms))

        return results[:limit]
