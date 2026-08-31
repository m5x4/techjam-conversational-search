"""Sweep acceptance policies over orderings already returned by a model.

`llm_bench.py replay --cache` writes what the model said. This reads it back and
asks a different question: given those answers, how much of the model's opinion
should the agent actually act on?

The policy that matters is the promotion ceiling. On a contested slate the
linear reranker is right about the top-1 roughly two thirds of the time, so a
model that promotes a candidate from rank 8 is usually overruling a better
guess than its own. Capping how deep a promotion may reach trades away the
model's rare deep saves for protection against its common deep mistakes, and
whether that trade is positive is an empirical question -- one answerable from
the cache for free, rather than by another few hundred calls per setting.

    python3 tools/llm_policy.py dev/orders.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def apply_ceiling(order: list[int], ceiling: int) -> list[int]:
    """The model's ordering, declined when its pick came from too deep.

    Mirrors `llm_rerank.parse_order` under MAX_PROMOTION_RANK: the promotion is
    the model's first choice, and a choice below the ceiling is dropped in
    favour of the incoming order.
    """
    if not order:
        return order
    if ceiling and order[0] >= ceiling:
        return sorted(order)
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache", help="orders written by `replay --cache`")
    args = parser.parse_args()

    rows = [
        row for row in json.loads(Path(args.cache).read_text())
        if row["target_index"] is not None
    ]
    total = len(rows) or 1
    linear_top1 = sum(1 for row in rows if row["target_index"] == 0)
    linear_mrr = sum(1.0 / (row["target_index"] + 1) for row in rows) / total

    print(f"{len(rows)} decisions with the target in the slate")
    print(f"linear                top-1 {linear_top1:>3}/{total} "
          f"({linear_top1 / total:.3f})  mrr {linear_mrr:.4f}")
    print()
    print("ceiling  top-1        mrr     fixed  broke   net")
    # 0 means no ceiling: whatever the model picked is taken.
    for ceiling in (0, 1, 2, 3, 4, 5):
        top1 = 0
        mrr = 0.0
        fixed = broke = 0
        for row in rows:
            order = apply_ceiling(list(row["order"]), ceiling)
            index = row["target_index"]
            new = order.index(index)
            top1 += new == 0
            mrr += 1.0 / (new + 1)
            fixed += index != 0 and new == 0
            broke += index == 0 and new != 0
        label = "none" if ceiling == 0 else str(ceiling)
        print(f"{label:>7}  {top1:>3}/{total} ({top1 / total:.3f})  "
              f"{mrr / total:.4f}  {fixed:>5}  {broke:>5}  {top1 - linear_top1:>+4}")


if __name__ == "__main__":
    main()
