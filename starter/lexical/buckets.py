"""Coarse-category bucket resolution -- the pre-retrieval filter.

The customer simulator always opens with
    "I'm looking for {coarse_category(target.categories)} ..."
so the opening line names, verbatim on the public set, a catalog bucket
guaranteed to contain the target (median bucket ~180 vs the 50k catalog).
Resolving it up front turns retrieval into a small in-bucket ranking problem
and gives a hard recall floor.

BucketIndex.resolve() degrades gracefully (exact -> containment ->
token-overlap -> None); on None the agent falls back to the whole-catalog
BM25 pipeline unchanged. lexical-track.md: "Bucket pre-filter".
"""

from __future__ import annotations

import re

# Mirrors evaluator/local_evaluator.py::coarse_category EXACTLY. The public
# opener is built by that function, so byte-identical output here is what
# makes the cheap EXACT resolve rung fire. If the evaluator's copy changes,
# change this to match.
_STORE_CATEGORY_EXCLUDED = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}


def coarse_category(categories_field: object) -> str:
    """Coarse catalog bucket for a product's `categories` field: split every
    comma-separated segment, strip whitespace, drop the store-wide top-level
    category (case-insensitive), join the LAST TWO surviving segments with a
    space. Generic fallback when nothing survives. Accepts the raw field as a
    list (catalog shape) or a plain string."""
    if categories_field is None:
        values: list[str] = []
    elif isinstance(categories_field, str):
        values = [categories_field]
    elif isinstance(categories_field, (list, tuple)):
        values = [str(v) for v in categories_field]
    else:
        values = [str(categories_field)]
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in _STORE_CATEGORY_EXCLUDED:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


# Filler words stripped before the token-overlap resolve rung.
_BUCKET_FILLER = {
    "a",
    "an",
    "the",
    "some",
    "any",
    "new",
    "today",
    "please",
    "me",
    "i",
    "for",
    "of",
    "in",
    "and",
    "to",
    "my",
    "looking",
    "want",
    "need",
    "hi",
    "hello",
    "hey",
    "there",
    "recommendations",
    "recommendation",
    "suggestions",
    "on",
    "about",
    "something",
    "anything",
    "im",
    "am",
}
# Leading verb phrases stripped when carving the category fragment out of an
# opener. Longest / most specific first so "looking for" wins over "after".
_BUCKET_VERB_PREFIXES = (
    "looking for",
    "searching for",
    "hunting for",
    "shopping for",
    "interested in",
    "show me",
    "i want",
    "i need",
    "after",
)
_BUCKET_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OVERLAP_ACCEPT = 0.34  # min overlap-coefficient for the fuzzy resolve rung


def _singularize(token: str) -> str:
    """Cheap plural heuristic for bucket-key / fragment token matching."""
    if token.endswith("sses"):
        return token[:-2]  # "dresses" -> "dress"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]  # "sandals" -> "sandal"
    return token


def _bucket_tokens(text: str) -> set[str]:
    return {
        _singularize(tok)
        for tok in _BUCKET_TOKEN_RE.findall(text.lower())
        if tok not in _BUCKET_FILLER
    }


class BucketIndex:
    """Lowercased coarse_category(...) -> [parent_asin], built once at
    index-build time, plus a precomputed token set per bucket key for the
    fuzzy resolve rung."""

    def __init__(self, bucket_map: dict[str, list[str]]) -> None:
        # keys already lowercased by the caller (_CatalogIndex._build)
        self.buckets: dict[str, list[str]] = bucket_map
        self.key_tokens: dict[str, set[str]] = {
            key: _bucket_tokens(key) for key in bucket_map
        }

    def ids(self, key: str) -> list[str]:
        return self.buckets.get(key, [])

    def _fragment(self, message: str) -> str:
        """Carve the category fragment out of an opening message: strip a
        leading verb phrase, then cut at the first '.'/',' clause boundary.
        If no verb phrase matches at all, keep the whole message so a fully
        reworded opener still has something to fuzzy-match against."""
        low = " ".join(str(message).split()).lower()
        fragment: str | None = None
        for verb in _BUCKET_VERB_PREFIXES:
            idx = low.find(verb)
            if idx != -1:
                fragment = low[idx + len(verb) :]
                break
        if fragment is None:
            fragment = low
        fragment = re.split(r"[.,]", fragment, maxsplit=1)[0]
        return fragment.strip(" -\t'\"")

    def resolve(self, message: str) -> tuple[str | None, str]:
        """(bucket_key, how) via a degrading ladder; never raises. Returns
        (None, reason) when nothing matches -- caller then falls back to the
        existing whole-catalog retrieval."""
        try:
            fragment = self._fragment(message)
            if not fragment:
                return None, "unresolved"

            # (b) EXACT
            if fragment in self.buckets:
                return fragment, "exact"

            # (c) CONTAINMENT -- longest key that contains / is contained by
            # the fragment (longer key == more specific).
            best_key: str | None = None
            for key in self.buckets:
                if len(key) >= 3 and (key in fragment or fragment in key):
                    if best_key is None or len(key) > len(best_key):
                        best_key = key
            if best_key is not None:
                return best_key, "containment"

            # (d) TOKEN OVERLAP -- overlap coefficient (intersection over the
            # SMALLER token set, not Jaccard: the fragment legitimately
            # carries extra descriptive words a bucket key never will).
            fragment_tokens = _bucket_tokens(fragment)
            if fragment_tokens:
                best_score = 0.0
                best_key = None
                for key, key_tokens in self.key_tokens.items():
                    if not key_tokens:
                        continue
                    overlap = len(fragment_tokens & key_tokens)
                    if not overlap:
                        continue
                    score = overlap / min(len(fragment_tokens), len(key_tokens))
                    if score > best_score or (
                        score == best_score
                        and best_key is not None
                        and len(key) > len(best_key)
                    ):
                        best_score = score
                        best_key = key
                if best_key is not None and best_score >= _OVERLAP_ACCEPT:
                    return best_key, f"overlap:{best_score:.2f}"

            return None, "unresolved"
        except Exception as exc:  # resolution must never hard-fail retrieval
            return None, f"error:{type(exc).__name__}"
