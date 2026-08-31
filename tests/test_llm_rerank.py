"""The safety properties the LLM reranking layer is built on.

The layer is only defensible because of two invariants: whatever the model
returns is a permutation of the slate it was handed, and every failure path
returns the slate untouched. Everything else about the design -- that a bad
model costs rank but never a hit, that scoring with the network disabled is
identical to scoring without this file -- follows from those two, so they are
tested directly rather than inferred from an end-to-end score.
"""
from __future__ import annotations

import unittest

from starter import llm_rerank
from starter.llm_rerank import LLMReranker, build_prompt, parse_order


CARDS = [
    {"asin": f"B{index}", "title": f"product {index}", "store": "acme", "features": ["x"]}
    for index in range(5)
]


class ExplodingClient:
    name = "boom"
    model = "boom"

    def complete(self, system, user):
        raise RuntimeError("upstream is down")


class ScriptedClient:
    name = "scripted"
    model = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[str] = []

    def complete(self, system, user):
        self.seen.append(user)
        return self.reply, {"prompt_tokens": 7, "completion_tokens": 2}


class ParseOrderTest(unittest.TestCase):
    def test_rank_mode_returns_a_permutation(self) -> None:
        self.assertEqual(parse_order("4,1,7", 5, mode="rank"), [3, 0, 1, 2, 4])

    def test_rank_mode_drops_out_of_range_and_duplicate_indices(self) -> None:
        self.assertEqual(parse_order("99,2,2,1", 3, mode="rank"), [1, 0, 2])

    def test_garbage_falls_back_to_the_incoming_order(self) -> None:
        self.assertEqual(parse_order("no idea", 4, mode="rank"), [0, 1, 2, 3])
        self.assertEqual(parse_order("", 4, mode="pick"), [0, 1, 2, 3])

    def test_pick_mode_reads_only_the_first_number(self) -> None:
        # The trailing "5" is part of an explanation, not a second choice.
        self.assertEqual(
            parse_order("3 because it has 5 stars", 6, mode="pick"), [2, 0, 1, 3, 4, 5]
        )

    def test_every_mode_returns_a_permutation_for_any_reply(self) -> None:
        replies = ["", "1", "9,9,9", "the answer", "0", "3,2,1", "-1", "10"]
        for mode in ("pick", "rank"):
            for reply in replies:
                order = parse_order(reply, 5, mode=mode)
                self.assertEqual(sorted(order), list(range(5)), f"{mode}: {reply!r}")

    def test_promotion_ceiling_declines_a_deep_pick(self) -> None:
        original = llm_rerank.MAX_PROMOTION_RANK
        llm_rerank.MAX_PROMOTION_RANK = 3
        try:
            self.assertEqual(parse_order("7", 8, mode="pick"), list(range(8)))
            self.assertEqual(parse_order("2", 8, mode="pick"), [1, 0, 2, 3, 4, 5, 6, 7])
        finally:
            llm_rerank.MAX_PROMOTION_RANK = original


class RerankerTest(unittest.TestCase):
    def test_disabled_without_a_client(self) -> None:
        reranker = LLMReranker(provider="off")
        self.assertFalse(reranker.enabled)
        order, usage = reranker.order("belts", ["leather"], CARDS)
        self.assertEqual(order, list(range(len(CARDS))))
        self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})

    def test_a_raising_client_returns_the_incoming_order(self) -> None:
        reranker = LLMReranker(client=ExplodingClient())
        order, usage = reranker.order("belts", ["leather"], CARDS)
        self.assertEqual(order, list(range(len(CARDS))))
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(reranker.failures, 1)

    def test_a_short_slate_is_never_sent(self) -> None:
        client = ScriptedClient("1")
        reranker = LLMReranker(client=client)
        self.assertEqual(reranker.order("belts", [], CARDS[:1])[0], [0])
        self.assertEqual(reranker.order("belts", [], [])[0], [])
        self.assertEqual(client.seen, [])

    def test_usage_is_accumulated_across_calls(self) -> None:
        reranker = LLMReranker(client=ScriptedClient("2"))
        reranker.order("belts", ["leather"], CARDS)
        reranker.order("belts", ["leather"], CARDS)
        self.assertEqual(reranker.usage, {"prompt_tokens": 14, "completion_tokens": 4})
        self.assertEqual(reranker.calls, 2)

    def test_the_prompt_carries_the_evidence_verbatim(self) -> None:
        client = ScriptedClient("1")
        LLMReranker(client=client).order("Accessories Belts", ["100% Leather"], CARDS)
        self.assertIn("100% Leather", client.seen[0])
        self.assertIn("Accessories Belts", client.seen[0])

    def test_prompt_numbers_candidates_from_one(self) -> None:
        text = build_prompt("belts", ["leather"], CARDS)
        self.assertIn("1. product 0", text)
        self.assertIn("5. product 4", text)


if __name__ == "__main__":
    unittest.main()
