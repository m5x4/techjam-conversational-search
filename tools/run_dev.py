"""Evaluate a dev set in parallel, and report both raw and public-comparable scores.

Two problems appear once the dev set gets large, and this tool exists for both.

Throughput. `evaluator.local_evaluator` runs sessions in one process at roughly
30ms each, so the 49,800-session exhaustive set takes ~25 minutes. The sessions
are independent -- the agent is reset between each one -- so they shard cleanly
across cores. Each worker pays the ~5s catalog index build once.

Comparability. This matters more. The public set's targets are drawn heavily
from the head of the popularity distribution: 24.5% of them have >=3000 ratings
*and* a price, and none at all have fewer than 10 ratings without a price. The
catalog cannot support that mix at scale -- there are only 180 such rows left
after the public set claims its share -- so an exhaustive dev set is 37% items
the public set never samples, and its raw score answers a different question.
Total variation distance from the public target distribution runs about 0.86,
against 0.27 for the stratified 800.

The fix is not to shrink the set. It is to score it twice: once raw, over
everything, which is the honest measure of how the agent does across the whole
catalog; and once reweighted so each stratum contributes its public-set share,
which is the number to compare against a public or leaderboard result. Kish
effective sample size is reported alongside the weighted figures, because
reweighting a 49,800-session run to a distribution concentrated in 180 rows
buys far less precision than the raw count suggests.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import MAX_TURNS, catalog_index, evaluate, load_jsonl, metric_summary
from tools.make_dev_set import RATING_BINS, stratum

_WORKER: dict = {}


def _init(catalog_path: str) -> None:
    from starter.agent import Agent
    identifiers, categories, products = catalog_index(catalog_path)
    _WORKER.update({
        "ids": identifiers,
        "categories": categories,
        "products": products,
        "agent": Agent(catalog_path=catalog_path),
    })


def _run_shard(samples: list[dict]) -> dict:
    result = evaluate(
        _WORKER["agent"], samples, _WORKER["ids"], _WORKER["categories"], _WORKER["products"]
    )
    return {"sessions": result["sessions"], "usage": result["reported_token_usage"]}


def score(summary: dict) -> tuple[float, float]:
    """(efficiency, technical score) from the official 0.50/0.30/0.20 weighting."""
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    technical = (
        0.50 * summary["hit_rate_at_10"] + 0.30 * summary["mrr"] + 0.20 * efficiency
    )
    return round(efficiency, 6), round(technical, 6)


def reweight(sessions: list[dict], strata: dict[str, tuple[int, bool]], public: list[dict],
             products: dict[str, dict]) -> dict:
    """Score the run as if its target mix matched the public set's.

    Each session's weight is the ratio of its stratum's public share to its
    share in this run, so strata the public set over-samples count for more and
    strata it never touches count for nothing.
    """
    want = Counter(stratum(products[str(s["ground_truth"]["parent_asin"])]) for s in public)
    got = Counter(strata[s["sample_id"]] for s in sessions)
    weights = {
        key: (want[key] / len(public)) / (got[key] / len(sessions))
        for key in got if want.get(key)
    }
    weighted = [(weights.get(strata[s["sample_id"]], 0.0), s) for s in sessions]
    total = sum(w for w, _ in weighted)
    if not total:
        return {"error": "no session falls in a stratum the public set samples"}

    def mean(field) -> float:
        return sum(w * field(s) for w, s in weighted) / total

    summary = {
        "hit_rate_at_10": round(mean(lambda s: float(s["hit"])), 6),
        "mrr": round(mean(lambda s: s["reciprocal_rank"]), 6),
        "mttc": round(mean(
            lambda s: s["first_hit_turn"] if s["first_hit_turn"] is not None else MAX_TURNS + 1
        ), 6),
    }
    efficiency, technical = score(summary)
    # Kish: how many equally-weighted sessions carry the same precision.
    effective = total ** 2 / sum(w * w for w, _ in weighted)
    covered = sum(1 for w, _ in weighted if w > 0)
    return {
        **summary,
        "efficiency": efficiency,
        "recommended_technical_score": technical,
        "sessions_in_public_strata": covered,
        # Kish. Read it as precision against scenario/profile noise only. On an
        # exhaustive set the targets are a census -- every catalog row in a
        # stratum is present -- so there is no target-sampling error left to
        # measure; what remains is that each target was scored under a single
        # random scenario and profile draw. Re-running the high-weight strata
        # under every scenario is what shrinks this, not more targets.
        "effective_sample_size": round(effective, 1),
        "strata_missing_from_run": sorted(
            f">={RATING_BINS[band - 1] if band else 0}{'/$' if priced else '/-'}"
            for band, priced in set(want) - set(got)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel dev-set evaluation with public reweighting")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="dev/dev_full.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--output", default="dev/dev_full_results.json")
    parser.add_argument("--workers", type=int, default=0, help="0 = one per core")
    parser.add_argument("--limit", type=int, default=0, help="first N samples only, for a smoke run")
    parser.add_argument("--sessions", action="store_true", help="keep per-session rows in the output")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    workers = args.workers or mp.cpu_count()
    # One shard per worker: sessions cost about the same, and fewer, larger
    # shards amortise the per-worker index build over more of them.
    size = -(-len(samples) // workers)
    shards = [samples[i:i + size] for i in range(0, len(samples), size)]

    started = time.time()
    if workers == 1:
        _init(args.catalog)
        results = [_run_shard(shard) for shard in shards]
    else:
        with mp.get_context("spawn").Pool(workers, _init, (args.catalog,)) as pool:
            results = pool.map(_run_shard, shards)
    elapsed = time.time() - started

    sessions = [row for result in results for row in result["sessions"]]
    usage = Counter()
    for result in results:
        usage.update(result["usage"])

    overall = metric_summary(sessions)
    efficiency, technical = score(overall)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)

    _, _, products = catalog_index(args.catalog)
    strata = {
        str(s["sample_id"]): stratum(products[str(s["ground_truth"]["parent_asin"])])
        for s in samples
    }
    payload = {
        **overall,
        "efficiency": efficiency,
        "recommended_technical_score": technical,
        "reported_token_usage": dict(usage),
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "public_reweighted": reweight(sessions, strata, load_jsonl(args.public), products),
        "runtime_seconds": round(elapsed, 1),
        "workers": workers,
    }
    if args.sessions:
        payload["sessions"] = sessions
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
