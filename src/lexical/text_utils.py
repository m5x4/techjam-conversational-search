"""Text normalisation helpers shared across the agent.

Tokenisation, stopword stripping, FTS5 phrase quoting, and the two folding
functions the rerank depends on: ``_field_value_strings`` (keep discrete
feature/detail values separate -- the simulator quotes one whole bullet
verbatim) and ``_norm`` (fold a constraint or catalog value into one
comparable form).
"""

from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
    "still",
    "exploring",
    "not",
    "quite",
    "right",
    "yet",
    "those",
    "options",
}


def _text(value: object) -> str:
    """Flatten any catalog field (str/list/dict) into a single string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


# Scripts written without reliable inter-word spaces: FTS5's `unicode61`
# tokenizer hands a whole run back as ONE token, so a run in one of these is
# split into overlapping character bigrams instead (the classic
# no-dependency CJK IR trick). Every other non-ASCII script
# (Cyrillic / Greek / Arabic / Hebrew / ...) is space/punctuation-delimited
# and rides through `unicode61`'s own word-segmentation unchanged.
_NGRAM_RE = re.compile(
    "["
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "぀-ヿ"  # Hiragana + Katakana
    "ㇰ-ㇿ"  # Katakana Phonetic Extensions
    "가-힯"  # Hangul Syllables
    "ᄀ-ᇿ"  # Hangul Jamo
    "㄰-㆏"  # Hangul Compatibility Jamo
    "]"
)


def _segment(text: str) -> list[str]:
    """Tokenise a string that may contain non-ASCII scripts.

    Pure-ASCII callers never get here -- ``_terms`` and
    ``_CatalogIndex``'s ``_seg_index`` both guard on ``str.isascii()`` -- so
    this only runs on genuinely multilingual input. NFD-fold + drop of
    combining marks collapses accented Latin onto ASCII (``café`` ->
    ``cafe``) to line up with the FTS5 index's ``remove_diacritics``. NFD
    (not NFKD) is deliberate: NFKD would also rewrite compatibility
    characters (trademark sign -> "tm", vulgar fractions, super/subscripts)
    and perturb the token stream of near-ASCII English rows. The trailing
    NFC re-composes anything that decomposed but kept all its parts
    (notably Hangul syllables -> jamo -> syllables).
    """
    folded = unicodedata.normalize("NFD", str(text))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = unicodedata.normalize("NFC", folded).lower()
    out: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[^\x00-\x7f]+", folded):
        if chunk.isascii():
            out.append(chunk)  # identical to TOKEN_RE extraction on ASCII
        elif _NGRAM_RE.search(chunk):
            chars = [c for c in chunk if not c.isspace()]
            if len(chars) == 1:
                out.append(chars[0])
            else:
                out.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        elif any(unicodedata.category(c).startswith("L") for c in chunk):
            out.append(chunk)
    return out


def _terms(text: str) -> list[str]:
    if text.isascii():
        return [
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in STOPWORDS
        ]
    return [
        term
        for term in _segment(text)
        if (len(term) > 1 or not term.isascii()) and term not in STOPWORDS
    ]


def _quote(term: str) -> str:
    # Escape internal quotes for FTS5 phrase syntax.
    return '"' + term.replace('"', '""') + '"'


def _field_value_strings(value: object) -> list[str]:
    """The discrete strings a catalog field (features / details) contributes,
    kept SEPARATE rather than flattened into one blob. The simulator quotes
    one whole feature bullet or one whole detail value verbatim, so the
    boundary between entries is itself signal. Backs _CatalogIndex.field_values
    / the EXACT term of Agent._rerank."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _norm(value: object) -> str:
    """Fold a disclosed constraint or a catalog value into one comparable
    form: collapse internal whitespace, strip, lower-case. `.lower()` (not
    `.casefold()`) keeps it byte-identical with the substring target
    _CatalogIndex.blob, which is stored `.lower()`."""
    return re.sub(r"\s+", " ", str(value)).strip().lower()
