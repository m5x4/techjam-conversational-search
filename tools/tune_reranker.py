"""Offline weight tuning for the linear reranker.

The reranker scores each candidate as a dot product of five features and five
weights (`RERANK_WEIGHTS` with `POSITION_W` spliced in). Tuning them by
re-running the evaluator would cost ~20s per candidate vector, which buys a few
hundred trials a day. This tool gets the same answer in milliseconds, on two
observations about the harness.

Nothing upstream of the ranking depends on the ranking. `customer_reply` reads
only `ask_attribute` and what has already been disclosed, never the slate; and
`Agent._choose_attribute` reads only session state. So the constraint sequence,
the retrieved pool, the deep-paging offset and the exposure width are all
fixed before any weight is chosen -- weights decide *only* which slice of an
already-determined pool goes out. That makes the whole thing replayable: trace
the sessions once, then re-score the cached candidates offline.

Every feature is oriented so that more is better. So if candidate c is no
better than the target t on all five, no non-negative weight vector can put c
above t; if it is at least as good on all five, every one of them does. Only
the candidates in between can change places with the target, and at capture
time they are usually a small minority -- most turns collapse to "the target
wins regardless" or "the target loses regardless", and are stored as a
constant or dropped outright. This is exact, not an approximation, provided
every weight stays >= 0, which `replay` enforces.

Usage:
    python tools/tune_reranker.py trace  --dataset dev/dev_set.jsonl --output dev/trace.pkl
    python tools/tune_reranker.py replay --trace dev/trace.pkl --weights 1.5,0.6,0.3,0.3,0.10
    python tools/tune_reranker.py search --trace dev/trace.pkl --trials 3000

`trace` takes several comma-separated datasets and pools them. `search`
reports every result on a held-out split it never selected on, and prints the
paste-ready constants; it does not edit the agent. Re-running it under a few
`--split-salt` values re-draws that split -- and the training set with it --
which is the only cheap way to tell a real gain from one the search fitted:

    for salt in a b c d e; do
      python tools/tune_reranker.py search --trace dev/trace.pkl --split-salt $salt
    done
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import random
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import starter.agent as agent_module
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import DEEP_PAGING, FEATURE_NAMES, Agent, Matcher, rerank_weights

# The scale anchor. Multiplying every weight by the same positive constant
# leaves the ranking untouched, so one of the five is redundant and `exact`
# is the one held fixed -- searching it as well would just explore rescalings
# of vectors already tried.
ANCHOR = 0


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

# Filled by the probes below on every turn, read once the turn is over.
_CAPTURE: dict = {}


def install_probes() -> None:
    """Record what the agent's own code computed, rather than recomputing it.

    Three thin wrappers: `features` hands over the candidate rows it already
    built for the live ranking, `rerank` the paging offset, `_reveal` the
    exposure width. Nothing about the retrieval or the conversation is
    reimplemented here, so a change to any of it shows up in the trace instead
    of silently invalidating it.
    """
    original_features = Matcher.features
    original_rerank = Matcher.rerank
    original_reveal = Agent._reveal

    def features(self, pool, evidence, *args, **kwargs):
        rows = original_features(self, pool, evidence, *args, **kwargs)
        _CAPTURE["pool"] = list(pool)
        _CAPTURE["rows"] = rows
        return rows

    def rerank(self, pool, evidence, top_k, offset=0, weights=None, **kwargs):
        # Mirrors the clamp in rerank(). If that ever drifts, --verify fails on
        # the first session rather than quietly tuning against a fiction.
        _CAPTURE["offset"] = max(
            0, min(offset if DEEP_PAGING else 0, max(0, len(pool) - top_k))
        )
        return original_rerank(self, pool, evidence, top_k, offset, weights, **kwargs)

    def reveal(*args, **kwargs):
        value = original_reveal(*args, **kwargs)
        _CAPTURE["reveal"] = value
        return value

    Matcher.features = features
    Matcher.rerank = rerank
    Agent._reveal = staticmethod(reveal)


def build_turn(index: int, counts: bool, target: str) -> dict | None:
    """Reduce one captured turn to the smallest thing that can still be scored.

    Returns None when the turn can never produce a hit -- the target missed the
    pool, or is beaten by more candidates than the slate is wide however the
    weights fall. Those turns carry no information for any weight vector, and
    dropping them is most of why the trace stays small.
    """
    if not counts:
        # Before an Intent Override lands, the harness ignores the slate
        # entirely, so nothing that happens on this turn can be scored.
        return None
    if "rows" not in _CAPTURE:
        # Browse track, or the exception fallback: the slate is not a function
        # of the weights at all, so whatever it did is what it always does.
        rank = _CAPTURE.get("served", 0)
        return {"turn": index, "kind": "fixed", "rank": rank} if rank else None

    pool = _CAPTURE["pool"]
    rows = _CAPTURE["rows"]
    low = _CAPTURE["offset"]
    cap = low + _CAPTURE["reveal"]
    try:
        position = pool.index(target)
    except ValueError:
        return None
    anchor_row = rows[position]

    # Split the pool three ways against the target: candidates that beat it
    # under every non-negative weight vector, candidates that beat it under
    # none, and the rest. Only the rest need to be carried.
    floor = 0
    undecided: list[tuple[float, ...]] = []
    for other, row in enumerate(rows):
        if other == position:
            continue
        delta = tuple(value - anchor for value, anchor in zip(row, anchor_row))
        # The BM25 feature is a strictly decreasing function of pool position,
        # so it is never zero between two candidates: dominance on every other
        # feature is therefore dominance including the position tie-break. It
        # is the last coordinate, which is why new features are spliced in
        # ahead of it rather than appended.
        if all(value >= 0 for value in delta[:-1]) and delta[-1] > 0:
            floor += 1
        elif all(value <= 0 for value in delta[:-1]) and delta[-1] < 0:
            continue
        else:
            undecided.append(delta)

    if floor >= cap or floor + len(undecided) < low:
        return None
    if not undecided:
        return {"turn": index, "kind": "fixed", "rank": floor - low + 1}
    return {
        "turn": index,
        "kind": "ranked",
        "low": low,
        "cap": cap,
        "floor": floor,
        "rows": undecided,
    }


def certain_hit(turn: dict) -> bool:
    """Does this turn hit under every weight vector? Then the session ends here
    and no later turn is reachable."""
    if turn["kind"] == "fixed":
        return True
    return turn["floor"] >= turn["low"] and turn["floor"] + len(turn["rows"]) < turn["cap"]


def trace_session(agent: Agent, sample: dict, catalog_ids, categories, products) -> dict:
    """Drive one session to the end of its usable turns.

    The harness stops at the first hit; this does not, because a weight vector
    that misses turn 2 needs turn 3 to exist. It stops instead at the first
    turn that hits unconditionally, which is the earliest point no weight can
    reach past.
    """
    session_id = f"trace_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    turns: list[dict] = []
    checks = {"turns": 0, "mismatched": 0}
    for index in range(1, MAX_TURNS + 1):
        _CAPTURE.clear()
        try:
            response = agent.respond(session_id, user_message, index, TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        served = ranked.index(target) + 1 if target in ranked else 0
        _CAPTURE["served"] = served

        turn = build_turn(index, override_applied, target)
        checks["turns"] += 1
        if not verify_turn(turn, served, override_applied):
            checks["mismatched"] += 1
        if turn is not None:
            turns.append(turn)
            if certain_hit(turn):
                break
        if index == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and index + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    return {
        "sample_id": str(sample["sample_id"]),
        "dataset": str(sample.get("_dataset", "")),
        "scenario_type": sample["scenario_type"],
        "target": target,
        "turns": turns,
        "checks": checks,
    }


def verify_turn(turn: dict | None, served: int, counts: bool) -> bool:
    """Replay the turn under the live weights and insist it reproduces reality.

    This is the whole safety net. Every reduction above -- the dominance split,
    the dropped turns, the offset clamp mirrored out of rerank() -- is only
    sound as long as it predicts what the agent actually served, so it is
    checked on every turn of every session at capture time.
    """
    if not counts:
        return True
    weights = rerank_weights()
    if turn is None:
        return served == 0
    if turn["kind"] == "fixed":
        return turn["rank"] == served
    return rank_in_turn(turn, weights) == served


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def rank_in_turn(turn: dict, weights) -> int:
    """Where the target lands in the served slate, or 0 if it misses it.

    Counting beaters rather than sorting: the slate is at most ten wide, so the
    count can stop the moment it passes the far edge of the window.
    """
    cap = turn["cap"]
    beaters = turn["floor"]
    if beaters >= cap:
        return 0
    for row in turn["rows"]:
        total = sum(weight * value for weight, value in zip(weights, row))
        # Ties fall to the earlier pool position, which is exactly the sign of
        # the (negated, strictly decreasing) BM25 feature -- the last coordinate.
        if total > 0.0 or (total == 0.0 and row[-1] > 0.0):
            beaters += 1
            if beaters >= cap:
                return 0
    if beaters < turn["low"]:
        return 0
    return beaters - turn["low"] + 1


def replay(session: dict, weights) -> tuple[int | None, float]:
    """(first hitting turn, reciprocal rank) for one traced session."""
    for turn in session["turns"]:
        rank = turn["rank"] if turn["kind"] == "fixed" else rank_in_turn(turn, weights)
        if rank:
            return turn["turn"], 1.0 / rank
    return None, 0.0


def evaluate_weights(sessions: list[dict], weights, session_weights=None) -> dict:
    """The official 0.50/0.30/0.20 score over a traced set."""
    if min(weights) < 0.0:
        raise ValueError("negative weights break the dominance pruning in the trace")
    total = 0.0
    hits = 0.0
    reciprocal = 0.0
    turns_to_convert = 0.0
    for index, session in enumerate(sessions):
        weight = 1.0 if session_weights is None else session_weights[index]
        if not weight:
            continue
        first, rr = replay(session, weights)
        total += weight
        hits += weight * (1.0 if first else 0.0)
        reciprocal += weight * rr
        turns_to_convert += weight * (first if first else MAX_TURNS + 1)
    if not total:
        return {"hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": 0.0, "score": 0.0}
    hit_rate = hits / total
    mrr = reciprocal / total
    mttc = turns_to_convert / total
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": efficiency,
        "score": 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_POOL: dict = {}


def _init_pool(trace_path: str, objective: str, salt: str) -> None:
    trace = load_trace(trace_path)
    _POOL["sessions"] = trace["sessions"]
    _POOL["splits"] = build_splits(trace, objective, salt)


def _score_many(job: tuple[str, list[tuple]]) -> list[float]:
    split, candidates = job
    sessions, session_weights = _POOL["splits"][split]
    return [evaluate_weights(sessions, weights, session_weights)["score"] for weights in candidates]


def build_splits(trace: dict, objective: str, salt: str = "") -> dict:
    """train / holdout / all, each as (sessions, per-session weights or None).

    The split is a hash of the sample id, so it is the same on every run and
    across processes, and a trace captured once can be searched many times
    without the holdout leaking into selection.

    `salt` moves the cut without re-tracing, which is what makes a repeated
    random subsampling check affordable: the same search run against several
    salts says far more about whether a gain is real than any interval
    computed from a single holdout, because it re-draws the training set too.
    """
    sessions = trace["sessions"]
    weights = public_weights(trace) if objective == "public" else [1.0] * len(sessions)
    train_index, holdout_index = [], []
    for index, session in enumerate(sessions):
        key = f"{salt}:{session.get('dataset', '')}:{session['sample_id']}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        (holdout_index if int(digest[:8], 16) % 100 < trace["meta"]["holdout_pct"] else train_index).append(index)
    return {
        "all": (sessions, weights),
        "train": ([sessions[i] for i in train_index], [weights[i] for i in train_index]),
        "holdout": ([sessions[i] for i in holdout_index], [weights[i] for i in holdout_index]),
    }


def public_weights(trace: dict) -> list[float]:
    """Per-session weights that make the traced target mix match the public
    set's, using the same ratio-of-shares as tools/run_dev.py."""
    want = Counter(tuple(item) for item in trace["meta"].get("public_strata") or [])
    if not want:
        raise SystemExit("this trace has no public strata; re-run `trace` with --public")
    got = Counter(tuple(session["stratum"]) for session in trace["sessions"])
    total_public = sum(want.values())
    total_got = sum(got.values())
    ratio = {
        key: (want[key] / total_public) / (got[key] / total_got)
        for key in got if want.get(key)
    }
    return [ratio.get(tuple(session["stratum"]), 0.0) for session in trace["sessions"]]


def sample_weights(rng: random.Random, baseline) -> tuple:
    """One random vector, log-uniform over three decades either side of nothing.

    A coordinate is zeroed outright one time in twelve: "this signal should not
    be in the model at all" is a hypothesis the log scale can approach but
    never actually state.
    """
    values = list(baseline)
    for index in range(len(values)):
        if index == ANCHOR:
            continue
        values[index] = 0.0 if rng.random() < 1 / 12 else math.exp(rng.uniform(math.log(0.002), math.log(5.0)))
    return tuple(values)


LADDER = (0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1, 2.6, 3.2, 4.0)


def coordinate_descent(seed, evaluate_batch, rounds: int = 8) -> tuple[tuple, float]:
    """Sweep one weight at a time along a geometric ladder until nothing moves.

    The objective is a step function -- it only changes when a candidate and
    the target actually swap places -- so gradients and simplex methods have
    nothing to work with, but a ladder sweep reads the steps directly.
    """
    current = tuple(seed)
    best = evaluate_batch([current])[0]
    for _ in range(rounds):
        improved = False
        for index in range(len(current)):
            if index == ANCHOR:
                continue
            base = current[index] or 0.05
            candidates = []
            for factor in LADDER:
                value = base * factor
                candidate = current[:index] + (value,) + current[index + 1:]
                if candidate != current:
                    candidates.append(candidate)
            scores = evaluate_batch(candidates)
            top = max(range(len(candidates)), key=lambda i: scores[i])
            if scores[top] > best + 1e-12:
                best, current, improved = scores[top], candidates[top], True
        if not improved:
            break
    return current, best


def bootstrap(sessions, session_weights, left, right, draws: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap over sessions of (right - left) in technical score."""
    rng = random.Random(seed)
    size = len(sessions)
    outcomes = []
    for index, session in enumerate(sessions):
        weight = 1.0 if session_weights is None else session_weights[index]
        first_l, rr_l = replay(session, left)
        first_r, rr_r = replay(session, right)
        outcomes.append((weight, first_l, rr_l, first_r, rr_r))

    def technical(rows, pick) -> float:
        total = sum(row[0] for row in rows)
        if not total:
            return 0.0
        first_at, rr_at = (1, 2) if pick == "left" else (3, 4)
        hit = sum(row[0] * (1.0 if row[first_at] else 0.0) for row in rows) / total
        mrr = sum(row[0] * row[rr_at] for row in rows) / total
        mttc = sum(row[0] * (row[first_at] if row[first_at] else MAX_TURNS + 1) for row in rows) / total
        return 0.50 * hit + 0.30 * mrr + 0.20 * max(0.0, min(1.0, (11.0 - mttc) / 10.0))

    deltas = []
    for _ in range(draws):
        rows = [outcomes[rng.randrange(size)] for _ in range(size)]
        deltas.append(technical(rows, "right") - technical(rows, "left"))
    deltas.sort()
    return (
        technical(outcomes, "right") - technical(outcomes, "left"),
        deltas[int(0.025 * draws)],
        deltas[int(0.975 * draws) - 1],
    )


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------


def agent_fingerprint() -> str:
    return hashlib.md5(Path(agent_module.__file__).read_bytes()).hexdigest()


def load_trace(path: str) -> dict:
    with open(path, "rb") as handle:
        trace = pickle.load(handle)
    if trace["meta"]["agent_md5"] != agent_fingerprint():
        print(
            f"warning: {path} was captured from a different starter/agent.py. "
            "Retrieval or the conversation may have changed; re-run `trace`.",
            file=sys.stderr,
        )
    return trace


def by_dataset(sessions, session_weights, weights) -> dict:
    """The same score, computed over each traced set separately."""
    names = sorted({session.get("dataset", "") for session in sessions})
    report = {}
    for name in names:
        picked = [index for index, s in enumerate(sessions) if s.get("dataset", "") == name]
        subset = [sessions[index] for index in picked]
        subset_weights = None if session_weights is None else [session_weights[index] for index in picked]
        report[name or "unnamed"] = (len(subset), evaluate_weights(subset, weights, subset_weights)["score"])
    return report


def format_weights(weights) -> str:
    return ", ".join(f"{name}={value:.6g}" for name, value in zip(FEATURE_NAMES, weights))


def as_constants(weights) -> str:
    exact_w, loose_w, position_w, pop_w, velocity_w, rank_w = weights
    return (
        f"RERANK_WEIGHTS = ({exact_w:.6g}, {loose_w:.6g}, {pop_w:.6g}, {rank_w:.6g})\n"
        f"POSITION_W = {position_w:.6g}\n"
        f"VELOCITY_W = {velocity_w:.6g}"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_trace(catalog_path: str) -> None:
    install_probes()
    identifiers, categories, products = catalog_index(catalog_path)
    _WORKER.update({
        "ids": identifiers,
        "categories": categories,
        "products": products,
        "agent": Agent(catalog_path=catalog_path),
    })


def _trace_shard(samples: list[dict]) -> list[dict]:
    return [
        trace_session(_WORKER["agent"], sample, _WORKER["ids"], _WORKER["categories"], _WORKER["products"])
        for sample in samples
    ]


def command_trace(args) -> None:
    # More than one dataset can be traced into a single file. Sessions carry
    # the set they came from, which keeps the train/holdout split well defined
    # when two sets number their samples independently, and lets a search over
    # the pooled sessions still report what it did to each set on its own.
    samples: list[dict] = []
    for path in [item for item in args.dataset.split(",") if item.strip()]:
        rows = load_jsonl(path.strip())
        if args.limit:
            rows = rows[:args.limit]
        stem = Path(path.strip()).stem
        samples.extend({**row, "_dataset": stem} for row in rows)
    workers = args.workers or mp.cpu_count()
    size = -(-len(samples) // workers)
    shards = [samples[i:i + size] for i in range(0, len(samples), size)]

    started = time.time()
    if workers == 1:
        _init_trace(args.catalog)
        results = [_trace_shard(shard) for shard in shards]
    else:
        with mp.get_context("spawn").Pool(workers, _init_trace, (args.catalog,)) as pool:
            results = pool.map(_trace_shard, shards)
    sessions = [row for result in results for row in result]
    elapsed = time.time() - started

    public_strata = None
    if args.public:
        from tools.make_dev_set import stratum
        _, _, products = catalog_index(args.catalog)
        for session in sessions:
            session["stratum"] = list(stratum(products[session["target"]]))
        public_strata = [
            list(stratum(products[str(item["ground_truth"]["parent_asin"])]))
            for item in load_jsonl(args.public)
        ]

    checks = Counter()
    for session in sessions:
        checks.update(session.pop("checks"))
    rows = sum(len(turn.get("rows", ())) for session in sessions for turn in session["turns"])
    trace = {
        "meta": {
            "dataset": args.dataset,
            "catalog": args.catalog,
            "feature_names": list(FEATURE_NAMES),
            "captured_weights": list(rerank_weights()),
            "agent_md5": agent_fingerprint(),
            "holdout_pct": args.holdout,
            "public_strata": public_strata,
            "verified_turns": checks["turns"],
            "mismatched_turns": checks["mismatched"],
        },
        "sessions": sessions,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)

    live = evaluate_weights(sessions, rerank_weights())
    print(json.dumps({
        "sessions": len(sessions),
        "scored_turns": sum(len(session["turns"]) for session in sessions),
        "carried_candidate_rows": rows,
        "verified_turns": checks["turns"],
        "mismatched_turns": checks["mismatched"],
        "replayed_live_score": round(live["score"], 6),
        "trace_bytes": Path(args.output).stat().st_size,
        "runtime_seconds": round(elapsed, 1),
    }, indent=2))
    if checks["mismatched"]:
        raise SystemExit(
            "the trace does not reproduce what the agent served -- do not tune on it"
        )


def command_replay(args) -> None:
    trace = load_trace(args.trace)
    weights = tuple(float(value) for value in args.weights.split(",")) if args.weights else rerank_weights()
    splits = build_splits(trace, args.objective, args.split_salt)
    report = {
        "weights": dict(zip(FEATURE_NAMES, weights)),
        "objective": args.objective,
    }
    for name in ("all", "train", "holdout"):
        sessions, session_weights = splits[name]
        summary = evaluate_weights(sessions, weights, session_weights)
        report[name] = {
            "sessions": len(sessions),
            **{k: round(v, 6) for k, v in summary.items()},
            "by_dataset": {
                key: {"sessions": count, "score": round(score, 6)}
                for key, (count, score) in by_dataset(sessions, session_weights, weights).items()
            },
        }
    print(json.dumps(report, indent=2))


def command_search(args) -> None:
    trace = load_trace(args.trace)
    splits = build_splits(trace, args.objective, args.split_salt)
    baseline = rerank_weights()
    rng = random.Random(args.seed)

    workers = args.workers or mp.cpu_count()
    context = mp.get_context("spawn")
    pool = (context.Pool(workers, _init_pool, (args.trace, args.objective, args.split_salt))
            if workers > 1 else None)

    def batch(candidates, split="train"):
        if not candidates:
            return []
        if pool is None:
            sessions, session_weights = splits[split]
            return [evaluate_weights(sessions, w, session_weights)["score"] for w in candidates]
        size = max(1, -(-len(candidates) // workers))
        jobs = [(split, candidates[i:i + size]) for i in range(0, len(candidates), size)]
        return [score for chunk in pool.map(_score_many, jobs) for score in chunk]

    started = time.time()
    trials = [sample_weights(rng, baseline) for _ in range(args.trials)]
    scores = batch(trials)
    ranked = sorted(zip(scores, trials), key=lambda item: -item[0])
    seeds = [baseline] + [weights for _, weights in ranked[:args.refine]]

    refined = []
    for seed in seeds:
        weights, score = coordinate_descent(seed, batch, rounds=args.rounds)
        refined.append((score, weights))
    refined.sort(key=lambda item: -item[0])
    elapsed = time.time() - started

    train, train_weights = splits["train"]
    holdout, holdout_weights = splits["holdout"]
    base_train = evaluate_weights(train, baseline, train_weights)
    base_holdout = evaluate_weights(holdout, baseline, holdout_weights)

    print(f"\nobjective: {args.objective}   trace: {args.trace}   "
          f"train {len(train)} / holdout {len(holdout)} sessions")
    print(f"{args.trials} random trials + {len(seeds)} refinements in {elapsed:.1f}s "
          f"on {workers} worker(s)\n")
    print(f"{'':4} {'train':>9} {'holdout':>9}  weights")
    print(f"{'base':4} {base_train['score']:9.6f} {base_holdout['score']:9.6f}  {format_weights(baseline)}")

    unique, seen = [], set()
    for score, weights in refined:
        key = tuple(round(value, 9) for value in weights)
        if key in seen:
            continue
        seen.add(key)
        unique.append((score, weights))
    for index, (score, weights) in enumerate(unique[:args.show], start=1):
        holdout_score = evaluate_weights(holdout, weights, holdout_weights)["score"]
        print(f"{index:<4} {score:9.6f} {holdout_score:9.6f}  {format_weights(weights)}")

    if unique:
        best = unique[0][1]
        delta, low, high = bootstrap(holdout, holdout_weights, baseline, best, args.bootstrap, args.seed)
        summary = evaluate_weights(holdout, best, holdout_weights)
        print(f"\nbest candidate on the held-out split it was never selected on:")
        print(f"  hit {summary['hit_rate_at_10']:.6f}  mrr {summary['mrr']:.6f}  "
              f"mttc {summary['mttc']:.6f}  score {summary['score']:.6f}")
        print(f"  vs baseline: {delta:+.6f}  (95% paired bootstrap {low:+.6f} .. {high:+.6f})")
        base_split = by_dataset(holdout, holdout_weights, baseline)
        best_split = by_dataset(holdout, holdout_weights, best)
        for name in sorted(best_split):
            count, score = best_split[name]
            print(f"  {name:<20} {count:>5} sessions  "
                  f"base {base_split[name][1]:.6f} -> {score:.6f} "
                  f"({score - base_split[name][1]:+.6f})")
        if low <= 0.0:
            print("  the interval covers zero -- this is not yet evidence of a real gain.")
        print(f"\nnot applied. To adopt it, edit starter/agent.py:\n\n{as_constants(best)}\n")
    if pool is not None:
        pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("trace", help="cache the reranker's candidates for a dataset")
    capture.add_argument("--catalog", default="data/catalog.jsonl")
    capture.add_argument("--dataset", default="dev/dev_set.jsonl",
                         help="one path, or several comma-separated, traced into one file")
    capture.add_argument("--public", default="", help="public set, to store strata for reweighting")
    capture.add_argument("--output", default="dev/trace.pkl")
    capture.add_argument("--workers", type=int, default=0, help="0 = one per core")
    capture.add_argument("--limit", type=int, default=0)
    capture.add_argument("--holdout", type=int, default=30, help="percent of sessions held out")
    capture.set_defaults(func=command_trace)

    once = sub.add_parser("replay", help="score one weight vector against a trace")
    once.add_argument("--trace", default="dev/trace.pkl")
    once.add_argument("--weights", default="", help=f"comma-separated, in the order {','.join(FEATURE_NAMES)}")
    once.add_argument("--objective", choices=("raw", "public"), default="raw")
    once.add_argument("--split-salt", default="", help="move the train/holdout cut without re-tracing")
    once.set_defaults(func=command_replay)

    hunt = sub.add_parser("search", help="random search plus coordinate descent")
    hunt.add_argument("--trace", default="dev/trace.pkl")
    hunt.add_argument("--objective", choices=("raw", "public"), default="raw")
    hunt.add_argument("--split-salt", default="", help="move the train/holdout cut without re-tracing")
    hunt.add_argument("--trials", type=int, default=3000)
    hunt.add_argument("--refine", type=int, default=8, help="random seeds to refine")
    hunt.add_argument("--rounds", type=int, default=8, help="max coordinate sweeps")
    hunt.add_argument("--show", type=int, default=10)
    hunt.add_argument("--bootstrap", type=int, default=2000)
    hunt.add_argument("--workers", type=int, default=0)
    hunt.add_argument("--seed", type=int, default=7)
    hunt.set_defaults(func=command_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
