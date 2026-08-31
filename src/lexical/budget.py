"""Parse a disclosed budget constraint into (operator, amount).

The local evaluator only ever writes "budget around $X" (-> "~", the soft
proximity re-rank). The "<"/">" arms match a reworded private evaluator
("keep it under $30", "at least $50") and drive the hard price split in
Agent._price_rank_map. "<="/">=" style bounds collapse to "<"/">";
inclusivity is applied at split time because the disclosed amount is the
target's own price.
"""

from __future__ import annotations

import re

_BUDGET_AMOUNT_RE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
_BUDGET_LT_RE = re.compile(
    r"\b(?:under|below|less\s+than|no\s+more\s+than|not\s+more\s+than|at\s+most|"
    r"up\s+to|cheaper\s+than|max(?:imum)?|within|budget\s+of)\b",
    re.IGNORECASE,
)
_BUDGET_GT_RE = re.compile(
    r"\b(?:over|above|more\s+than|at\s+least|no\s+less\s+than|min(?:imum)?|"
    r"starting\s+at|upwards\s+of)\b",
    re.IGNORECASE,
)
_BUDGET_LT_SYM_RE = re.compile(r"<=?\s*\$?\s?\d")
_BUDGET_GT_SYM_RE = re.compile(r">=?\s*\$?\s?\d")


def _budget_amount(value: str) -> float | None:
    """Pull a dollar figure out of a disclosed budget constraint
    ("budget around $24.99" -> 24.99). The simulator builds that string from
    the target's own exact price, so when it surfaces the number is a very
    strong retrieval anchor."""
    match = _BUDGET_AMOUNT_RE.search(value)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _budget_constraint(value: str) -> tuple[str, float] | None:
    """(operator, amount) from a disclosed budget string, or None.

    operator is one of "<", ">", "~". "~" (the only shape the local
    evaluator emits) keeps the existing soft proximity re-rank; "<"/">"
    drive the hard price split in Agent._price_rank_map.
    """
    amount = _budget_amount(value)
    if amount is None:
        return None
    if _BUDGET_LT_RE.search(value) or _BUDGET_LT_SYM_RE.search(value):
        return "<", amount
    if _BUDGET_GT_RE.search(value) or _BUDGET_GT_SYM_RE.search(value):
        return ">", amount
    return "~", amount
