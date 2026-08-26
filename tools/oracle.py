"""Constraint oracle: replays the simulator's hidden side for a public session.

Dev tool only -- it reads ground truth, so it can never live inside the Agent.
"""
from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, behavior_for, catalog_index, classify_constraint, coarse_category,
    customer_reply, initial_message, intent_card, load_jsonl, searchable_text,
)


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def make_predicate(constraint, products, corpus):
    """Map a simulator constraint string onto a catalog filter."""
    cleaned = norm(constraint)
    budget = re.match(r"^budget around \$(.+)$", cleaned)
    if budget:
        target_price = budget.group(1)
        return f"price == {target_price}", lambda a: norm(products[a].get("price") or "") == target_price
    color = re.match(r"^color:\s*(.+)$", cleaned)
    if color:
        value = color.group(1)
        return f"color ~ {value}", lambda a: value in corpus[a]
    label = cleaned if len(cleaned) <= 46 else cleaned[:43] + "..."
    return f"text ~ {label}", lambda a: cleaned in corpus[a]


def narrow(candidates, disclosed, products, corpus):
    for constraint in disclosed:
        _, predicate = make_predicate(constraint, products, corpus)
        candidates = {a for a in candidates if predicate(a)}
    return candidates


class StaticPolicy:
    """A hypothetical asker. Answers 'what could any agent achieve?'."""

    stop_on_hit = False
    shows_rank = False

    def __init__(self, choose) -> None:
        self.choose = choose

    def start(self, sample) -> None:
        pass

    def turn(self, message, turn):
        return self.choose(turn), None


class AgentDriver:
    """Drives the real Agent. Answers 'what does mine actually do?'."""

    stop_on_hit = True
    shows_rank = True

    def __init__(self, catalog_path) -> None:
        from starter.agent import Agent

        self.agent = Agent(catalog_path)
        self.session_id = ""

    def start(self, sample) -> None:
        self.session_id = f"oracle_{sample['sample_id']}"
        self.agent.reset(self.session_id, sample["user_profile"])

    def turn(self, message, turn):
        response = self.agent.respond(self.session_id, message, turn, TOP_K)
        ranked = [
            str(item.get("parent_asin"))
            for item in response.get("recommendations") or []
            if isinstance(item, dict)
        ]
        return response.get("ask_attribute"), ranked


def trace_session(sample, products, corpus, categories, all_ids, driver):
    target = str(sample["ground_truth"]["parent_asin"])
    card = intent_card(products[target])
    rng = random.Random(f"{sample.get('sample_id','')}\0{sample.get('scenario_type','')}")
    behavior = behavior_for(str(sample["scenario_type"]), card, rng)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    every = [*card["hard_constraints"], *card["soft_preferences"]]
    every = list(dict.fromkeys(every))

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    driver.start(sample)

    turns = []
    candidates = set(all_ids)
    applied: set[str] = set()
    full_knowledge_turn = None
    hit_turn = None
    hit_rank = None
    for turn in range(1, MAX_TURNS + 1):
        candidates = narrow(candidates, disclosed - applied, products, corpus)
        applied = set(disclosed)
        known = len(disclosed)
        if known == len(every) and full_knowledge_turn is None:
            full_knowledge_turn = turn

        # One call per turn: the ask that gets recorded is the same one the
        # customer answers, which matters once a real agent is driving.
        ask, ranked = driver.turn(message, turn)
        rank = ranked.index(target) + 1 if ranked and target in ranked else None
        scored = override_applied and rank is not None

        turns.append({
            "turn": turn,
            "heard": message,
            "known": sorted(disclosed),
            "candidates": len(candidates),
            "target_survives": target in candidates,
            "hit_allowed": override_applied,
            "ask": ask,
            "rank": rank,
            "hit": scored,
        })
        if scored and driver.stop_on_hit:
            hit_turn, hit_rank = turn, rank
            break

        override = behavior.get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", ""))
        else:
            message, boundary_used = customer_reply(
                effective, ask, disclosed, boundary_used
            )
    return {
        "sample_id": sample["sample_id"],
        "scenario": sample["scenario_type"],
        "target": target,
        "constraints": [(c, classify_constraint(c)) for c in every],
        "turns": turns,
        "full_knowledge_turn": full_knowledge_turn,
        "hit_turn": hit_turn,
        "hit_rank": hit_rank,
        "shows_rank": driver.shows_rank,
    }


def render(report, products, corpus, limit_turns=6):
    lines = []
    lines.append(f"=== {report['sample_id']}  [{report['scenario']}]  target={report['target']}")
    title = norm(products[report["target"]].get("title"))[:88]
    lines.append(f"    {title}")
    lines.append("    hidden constraints the simulator will release:")
    for constraint, attribute in report["constraints"]:
        label, _ = make_predicate(constraint, products, corpus)
        lines.append(f"      [{attribute:<9}] {norm(constraint)[:66]:<66} -> {label}")
    lines.append("")
    show_rank = report.get("shows_rank")
    header = f"    {'turn':<5}{'ask':<10}{'known':<7}{'cands':>9}  {'alive':<6}{'gate':<7}"
    if show_rank:
        header += f"{'rank':<6}"
    lines.append(header + "heard")
    for row in report["turns"][:limit_turns]:
        heard = norm(row["heard"])
        heard = heard if len(heard) <= 52 else heard[:49] + "..."
        line = (
            f"    {row['turn']:<5}{str(row['ask']):<10}{len(row['known']):<7}{row['candidates']:>9}"
            f"  {'yes' if row['target_survives'] else 'NO':<6}"
            f"{'yes' if row['hit_allowed'] else 'gated':<7}"
        )
        if show_rank:
            if row["rank"] is None:
                mark = "-"
            else:
                mark = f"#{row['rank']}*" if row["hit"] else f"#{row['rank']}"
            line += f"{mark:<6}"
        lines.append(line + heard)
    lines.append(f"    -> all constraints known by turn: {report['full_knowledge_turn']}")
    if show_rank:
        if report["hit_turn"]:
            lines.append(f"    -> HIT on turn {report['hit_turn']} at rank {report['hit_rank']}")
        else:
            lines.append("    -> MISS: target never appeared in the returned top 10")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint oracle for public sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--show", default="public_0001,public_0002")
    parser.add_argument("--aggregate", type=int, default=0)
    parser.add_argument("--policy", default="other", choices=["other", "cycle", "agent"])
    args = parser.parse_args()

    all_ids, categories, products = catalog_index(args.catalog)
    corpus = {a: norm(searchable_text(p)) for a, p in products.items()}
    samples = load_jsonl(args.dataset)

    wheel = ["material", "color", "budget", "size", "style", "use_case", "feature", "other"]
    if args.policy == "agent":
        driver = AgentDriver(args.catalog)
    elif args.policy == "cycle":
        driver = StaticPolicy(lambda turn: wheel[(turn - 1) % len(wheel)])
    else:
        driver = StaticPolicy(lambda turn: "other")

    wanted = {s.strip() for s in args.show.split(",") if s.strip()}
    for sample in samples:
        if sample["sample_id"] in wanted:
            report = trace_session(sample, products, corpus, categories, all_ids, driver)
            print(render(report, products, corpus))
            print()

    if args.aggregate:
        by_scenario: dict[str, list[int]] = {}
        alive = 0
        hit_turns: list[int] = []
        for sample in samples[: args.aggregate]:
            report = trace_session(sample, products, corpus, categories, all_ids, driver)
            turn = report["full_knowledge_turn"] or 11
            by_scenario.setdefault(report["scenario"], []).append(turn)
            # An agent run stops at its hit, so the trace can be shorter than `turn`.
            final = report["turns"][min(turn, len(report["turns"])) - 1]
            alive += int(final["target_survives"])
            if report["hit_turn"]:
                hit_turns.append(report["hit_turn"])
        flat = [t for values in by_scenario.values() for t in values]
        print(f"--- ceiling over {len(flat)} sessions, ask policy = {args.policy}")
        for scenario in sorted(by_scenario):
            values = by_scenario[scenario]
            print(f"    {scenario:<16} n={len(values):<4} mean full-knowledge turn = {statistics.fmean(values):.2f}")
        print(f"    {'ALL':<16} n={len(flat):<4} mean full-knowledge turn = {statistics.fmean(flat):.2f}")
        print(f"    target survives its own constraint filter: {alive}/{len(flat)}")
        if driver.shows_rank:
            mean_turn = f"{statistics.fmean(hit_turns):.2f}" if hit_turns else "n/a"
            print(f"    agent hit {len(hit_turns)}/{len(flat)} sessions, mean hit turn {mean_turn}")


if __name__ == "__main__":
    main()
