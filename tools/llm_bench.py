"""Measure the LLM reranker against the linear one, without running sessions.

Two commands, split for the same reason `tune_reranker.py` splits trace from
replay: the expensive half should only run once.

`capture` drives the deterministic agent over a dataset and writes out every
decision the LLM gate would have fired on -- the shopper's category and quoted
requirements, the ten candidates in the order the linear reranker put them, and
which one is the target. No model is called and nothing about the agent is
reimplemented; the candidates are lifted off the live `rerank` call through the
same probe style `tune_reranker.py` uses.

`replay` calls a model on each captured decision and reports whether it puts the
target first more often than the linear ordering did. That is the whole question
-- on the exposure turns the slate is one item wide, so top-1 accuracy on this
set *is* the score difference, and it can be measured for the price of a few
hundred short prompts instead of a full evaluation run per prompt edit.

The number to beat is printed as `linear`. A model that ties it is not worth
shipping: it costs tokens, latency, and a network dependency at scoring time.

    python3 tools/llm_bench.py capture --dataset data/public_set.jsonl
    python3 tools/llm_bench.py replay --provider ollama --model llama3:latest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import local_evaluator as LE
from starter import agent as A
from starter.agent import Agent, Matcher
from starter.llm_rerank import LLMReranker, make_client

DEFAULT_SET = Path("dev/llm_decisions.json")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

_CAPTURE: dict = {}


def install_probes() -> None:
    """Record the slate the live reranker produced, before the exposure trim."""
    original = Matcher.rerank

    def rerank(self, pool, evidence, top_k, offset=0, weights=None):
        slate, info = original(self, pool, evidence, top_k, offset, weights)
        _CAPTURE["slate"] = list(slate)
        _CAPTURE["tied"] = info.get("tied", 0)
        _CAPTURE["evidence"] = list(evidence)
        return slate, info

    Matcher.rerank = rerank


def gated(turn: int, slate: list[str], tied: int) -> bool:
    """Exactly the condition `Agent.respond` applies before calling the model.

    Kept as one function so the bench cannot drift into measuring a population
    the agent would never send.
    """
    if turn > A.LLM_RERANK_UNTIL_TURN or len(slate) < 2:
        return False
    return tied > 1 or not A.LLM_RERANK_ONLY_WHEN_TIED


def command_capture(args) -> None:
    install_probes()
    ids, categories, products = LE.catalog_index(args.catalog)
    samples = LE.load_jsonl(args.dataset)
    agent = Agent(catalog_path=args.catalog)

    decisions: list[dict] = []
    original_respond = Agent.respond
    turns: list[dict] = []

    def respond(self, session_id, user_message, turn, top_k):
        _CAPTURE.clear()
        response = original_respond(self, session_id, user_message, turn, top_k)
        turns.append({
            "turn": turn,
            "message": user_message,
            "category": self._sessions[session_id].category,
            "slate": _CAPTURE.get("slate", []),
            "tied": _CAPTURE.get("tied", 0),
            "evidence": _CAPTURE.get("evidence", []),
        })
        return response

    Agent.respond = respond
    for sample in samples:
        turns.clear()
        LE.evaluate(agent, [sample], ids, categories, products)
        target = str(sample["ground_truth"]["parent_asin"])
        for record in turns:
            if not gated(record["turn"], record["slate"], record["tied"]):
                continue
            slate = record["slate"]
            decisions.append({
                "sample_id": sample["sample_id"],
                "scenario": sample["scenario_type"],
                "turn": record["turn"],
                "category": record["category"],
                "evidence": record["evidence"],
                "tied": record["tied"],
                "candidates": agent.matcher.cards(slate),
                # None when the target missed the slate entirely. Those rows are
                # kept rather than filtered: they are decisions the agent really
                # does send to the model, so the token count and the latency are
                # real even though no ordering of them can score.
                "target_index": slate.index(target) if target in slate else None,
            })
    Agent.respond = original_respond

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decisions, indent=1))
    reachable = [d for d in decisions if d["target_index"] is not None]
    print(f"{len(decisions)} decisions from {len(samples)} sessions -> {out}")
    print(f"  target in slate: {len(reachable)}")
    print(f"  linear puts it first: {sum(1 for d in reachable if d['target_index'] == 0)}")
    print("  by turn:", dict(sorted(Counter(d["turn"] for d in decisions).items())))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def command_replay(args) -> None:
    decisions = json.loads(Path(args.decisions).read_text())
    if args.limit:
        decisions = decisions[: args.limit]
    client = make_client(args.provider)
    if client is None:
        raise SystemExit(
            f"no {args.provider} client: check the server is up, or the key is set"
        )
    if args.model:
        client.model = args.model
    reranker = LLMReranker(client=client)

    reachable = [d for d in decisions if d["target_index"] is not None]
    linear_top1 = sum(1 for d in reachable if d["target_index"] == 0)
    linear_mrr = sum(1.0 / (d["target_index"] + 1) for d in reachable)

    llm_top1 = 0
    llm_mrr = 0.0
    moved_up = moved_down = 0
    broke = fixed = 0
    latencies: list[float] = []
    # Every ordering the model returned, keyed by decision. Written out so that
    # acceptance policies -- take the pick always, take it only when it lands
    # inside the linear top-3, and so on -- can be swept afterwards for free
    # rather than costing another few hundred calls each.
    cache: list[dict] = []
    for number, decision in enumerate(decisions, 1):
        start = time.time()
        order, _ = reranker.order(
            decision["category"], decision["evidence"], decision["candidates"]
        )
        latencies.append(time.time() - start)
        cache.append({
            "sample_id": decision["sample_id"],
            "turn": decision["turn"],
            "order": order,
            "target_index": decision["target_index"],
        })
        index = decision["target_index"]
        if index is None:
            continue
        new = order.index(index)
        llm_top1 += new == 0
        llm_mrr += 1.0 / (new + 1)
        if new < index:
            moved_up += 1
        elif new > index:
            moved_down += 1
        # The two that decide whether this ships: decisions the linear model got
        # right and the model broke, against decisions it got wrong and the
        # model fixed.
        if index == 0 and new != 0:
            broke += 1
        if index != 0 and new == 0:
            fixed += 1
        if args.verbose and index != new:
            arrow = "FIX " if new == 0 else ("BREAK" if index == 0 else "     ")
            print(f"  {arrow} {decision['sample_id']} t{decision['turn']} "
                  f"rank {index + 1} -> {new + 1}  {decision['evidence']}")

    if args.cache:
        Path(args.cache).write_text(json.dumps(cache))
        print(f"orders cached -> {args.cache}")

    total = len(reachable) or 1
    latencies.sort()
    print()
    print(f"decisions replayed : {len(decisions)}  (target reachable in {len(reachable)})")
    print(f"model              : {client.name}/{getattr(client, 'model', '?')}")
    print(f"calls / failures   : {reranker.calls} / {reranker.failures}")
    print(f"tokens             : {reranker.usage}")
    if latencies:
        print(f"latency p50/p95 s  : {latencies[len(latencies) // 2]:.2f} / "
              f"{latencies[int(len(latencies) * 0.95) - 1]:.2f}")
    print()
    print(f"top-1  linear {linear_top1:>4}/{total}  ({linear_top1 / total:.3f})")
    print(f"top-1  llm    {llm_top1:>4}/{total}  ({llm_top1 / total:.3f})")
    print(f"mrr    linear {linear_mrr / total:.4f}")
    print(f"mrr    llm    {llm_mrr / total:.4f}")
    print(f"fixed {fixed}  broke {broke}  (net top-1 {fixed - broke:+d})")
    print(f"moved up {moved_up}  moved down {moved_down}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    grab = sub.add_parser("capture", help="write the decision set (no model calls)")
    grab.add_argument("--dataset", default="data/public_set.jsonl")
    grab.add_argument("--catalog", default="data/catalog.jsonl")
    grab.add_argument("--output", default=str(DEFAULT_SET))
    grab.set_defaults(func=command_capture)

    run = sub.add_parser("replay", help="score a model against the decision set")
    run.add_argument("--decisions", default=str(DEFAULT_SET))
    run.add_argument("--provider", default="ollama")
    run.add_argument("--model", default=None)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--cache", default=None,
                     help="write each returned ordering here, for policy sweeps")
    run.set_defaults(func=command_replay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
