"""What the LLM's answers would have scored, under two architectures.

`llm_bench.py replay --cache` already recorded what the model said about each
gated decision. This asks what those same answers are worth end to end, and it
asks it twice:

  final reranker  the model's pick is promoted to rank 1 outright. This is what
                  `starter/agent.py` actually does when LLM_RERANK=1, so it is
                  checkable against a real evaluation run.

  7th feature     the pick becomes one more term in the reranker's dot product,
                  worth `w`, and has to out-score the linear margin to move
                  anything. At w=0 it is the deterministic agent; as w grows it
                  converges on the final-reranker policy. The sweep in between
                  is the question -- whether there is a weight at which the
                  model helps a little without being allowed to steamroll.

No model is called. The deterministic half is re-run to recover the linear
score of every candidate (the cache holds orderings, not margins), and the
model's half is read back off the cache.

Sessions are driven for all ten turns rather than stopped at the first hit: a
policy that breaks a turn-1 hit needs turn 2 to exist. The conversation itself
does not depend on the ranking -- `ask_attribute` is deterministic and the
customer replies to that, not to the slate -- so one capture serves every
policy.

    python3 tools/llm_bench.py replay --cache dev/orders.json
    python3 tools/llm_simulate.py --orders dev/orders.json
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter import agent as A
from starter.agent import Agent, Matcher

_CAPTURE: dict = {}


def install_probes() -> None:
    """Take the slate, its linear scores, and the exposure width off the live
    code rather than recomputing any of them.

    The scores in particular have to come from here. `features()` emits the
    BM25 term as `-(position / depth)` over the *pool*, so recomputing it on the
    ten-item slate would silently rescale that feature and change the margins
    this whole simulation turns on.
    """
    original_features = Matcher.features
    original_rerank = Matcher.rerank
    original_browse = Matcher.browse
    original_reveal = Agent._reveal

    def features(self, pool, evidence):
        rows = original_features(self, pool, evidence)
        weights = A.rerank_weights()
        _CAPTURE["totals"] = {
            asin: sum(weight * value for weight, value in zip(weights, row))
            for asin, row in zip(pool, rows)
        }
        return rows

    def rerank(self, pool, evidence, top_k, offset=0, weights=None):
        slate, info = original_rerank(self, pool, evidence, top_k, offset, weights)
        _CAPTURE["slate"] = list(slate)
        _CAPTURE["tied"] = info.get("tied", 0)
        return slate, info

    def browse(self, category, top_k=10, pool=500, bucket=None):
        picked, info = original_browse(self, category, top_k, pool, bucket)
        _CAPTURE["slate"] = list(picked)
        _CAPTURE["tied"] = 0
        return picked, info

    def reveal(*args, **kwargs):
        value = original_reveal(*args, **kwargs)
        _CAPTURE["reveal"] = value
        return value

    Matcher.features = features
    Matcher.rerank = rerank
    Matcher.browse = browse
    Agent._reveal = staticmethod(reveal)


def trace_session(agent: Agent, sample: dict, categories, products, catalog_ids) -> dict:
    """Every turn of one session, whether or not it would have ended earlier."""
    session_id = f"sim_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )

    turns: list[dict] = []
    for index in range(1, MAX_TURNS + 1):
        _CAPTURE.clear()
        try:
            response = agent.respond(session_id, message, index, TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        served = normalize_recommendations(response.get("recommendations"), catalog_ids)
        slate = _CAPTURE.get("slate", list(served))
        totals = _CAPTURE.get("totals", {})
        turns.append({
            "turn": index,
            "counts": override_applied,
            "slate": slate,
            # Absent on the browse track, where the ordering is not a dot
            # product and no weight can move it.
            "scores": [totals[asin] for asin in slate] if totals else None,
            "reveal": _CAPTURE.get("reveal", len(served)),
            "gated": (
                index <= A.LLM_RERANK_UNTIL_TURN and len(slate) > 1
                and (_CAPTURE.get("tied", 0) > 1 or not A.LLM_RERANK_ONLY_WHEN_TIED)
            ),
            "target_index": slate.index(target) if target in slate else None,
            "served_rank": served.index(target) + 1 if target in served else 0,
        })
        if index == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and index + 1 == int(override.get("turn", 3)):
            override_applied = True
            value = str(override.get("new_value", ""))
            if value:
                disclosed.add(value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )
    return {"sample_id": str(sample["sample_id"]), "turns": turns}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def apply_policy(turn: dict, pick: int | None, mode: str, weight: float) -> list[int]:
    """The slate's indices after the policy, best first."""
    size = len(turn["slate"])
    order = list(range(size))
    if pick is None or not turn["gated"] or mode == "identity":
        return order
    if mode == "rerank":
        # What the agent does today: the pick goes first, the rest hold.
        return [pick] + [index for index in order if index != pick]
    # 7th feature. The model's opinion is a binary indicator worth `weight`, so
    # the pick moves up only past candidates it now out-scores -- which is the
    # whole difference between the two architectures.
    scores = turn["scores"]
    if scores is None:
        return order
    adjusted = [value + (weight if index == pick else 0.0) for index, value in enumerate(scores)]
    return sorted(order, key=lambda index: (-adjusted[index], index))


def score_run(traces: list[dict], picks: dict, mode: str, weight: float) -> dict:
    ranks: list[float] = []
    hit_turns: list[int] = []
    for trace in traces:
        best_rank = None
        hit_turn = None
        for turn in trace["turns"]:
            pick = picks.get((trace["sample_id"], turn["turn"]))
            order = apply_policy(turn, pick, mode, weight)
            reveal = turn["reveal"]
            served = [turn["slate"][index] for index in order[:reveal]]
            index = turn["target_index"]
            if turn["counts"] and index is not None and turn["slate"][index] in served:
                best_rank = served.index(turn["slate"][index]) + 1
                hit_turn = turn["turn"]
                break
        ranks.append(0.0 if best_rank is None else 1.0 / best_rank)
        hit_turns.append(hit_turn if hit_turn else MAX_TURNS + 1)

    count = len(traces) or 1
    hit_rate = sum(1 for value in ranks if value > 0) / count
    mrr = sum(ranks) / count
    mttc = sum(hit_turns) / count
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit": hit_rate, "mrr": mrr, "mttc": mttc,
        "score": 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency,
        "turn1": sum(1 for value in hit_turns if value == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", required=True, help="cache from `replay --cache`")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--trace", default=None, help="reuse/write the deterministic trace")
    args = parser.parse_args()

    cached = Path(args.trace) if args.trace else None
    if cached and cached.exists():
        traces = json.loads(cached.read_text())
    else:
        install_probes()
        ids, categories, products = catalog_index(args.catalog)
        agent = Agent(catalog_path=args.catalog)
        samples = load_jsonl(args.dataset)
        traces = []
        for number, sample in enumerate(samples, 1):
            traces.append(trace_session(agent, sample, categories, products, ids))
            if number % 20 == 0:
                print(f"  traced {number}/{len(samples)}", file=sys.stderr, flush=True)
        if cached:
            cached.write_text(json.dumps(traces))

    picks = {
        (row["sample_id"], row["turn"]): (row["order"][0] if row["order"] else None)
        for row in json.loads(Path(args.orders).read_text())
    }

    base = score_run(traces, picks, "identity", 0.0)
    print(f"{len(traces)} sessions, {len(picks)} model answers\n")
    print("policy                  score      hit     mrr     mttc   turn-1 hits")
    print(f"{'linear only':<22}  {base['score']:.6f}  {base['hit']:.3f}  "
          f"{base['mrr']:.4f}  {base['mttc']:.3f}  {base['turn1']:>5}")

    final = score_run(traces, picks, "rerank", 0.0)
    print(f"{'LLM as reranker':<22}  {final['score']:.6f}  {final['hit']:.3f}  "
          f"{final['mrr']:.4f}  {final['mttc']:.3f}  {final['turn1']:>5}  "
          f"{final['score'] - base['score']:+.6f}")
    print()
    for weight in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 5.0):
        row = score_run(traces, picks, "feature", weight)
        print(f"{'LLM 7th feature w=' + format(weight, '.2f'):<22}  {row['score']:.6f}  "
              f"{row['hit']:.3f}  {row['mrr']:.4f}  {row['mttc']:.3f}  "
              f"{row['turn1']:>5}  {row['score'] - base['score']:+.6f}")


if __name__ == "__main__":
    main()
