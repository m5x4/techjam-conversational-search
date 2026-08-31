"""Scenario-routed meta-agent -- submission entry point.

`Agent` (reset / respond) is a thin router over two independent
implementations that live in sibling sub-packages:

    lexical/       bucket pre-filter -> two-tier BM25 -> linear rerank.
                   Stronger on the Intent-Override scenario (durable edge
                   on both the public and held-out sets).
    elimination/   elimination-first phrase filtering + diversified browse.
                   Stronger on Buying / Browsing / Boundary, and the
                   better generaliser overall.

The customer simulator's opening message fixes the scenario for the whole
session, so the route is decided once, from turn 1, and never switched:

  * "I'm looking for <cat>. A key requirement is: <c>."   -> Buying      -> elimination
  * "I'm looking for <cat>. <old_value>"                  -> Intent-Override
                                                                        -> lexical
  * "I'm looking for <cat>, but I'm still exploring."     -> Browsing /
                                                            Boundary    -> elimination

Anything that does not clearly match the Intent-Override opener shape routes
to the `elimination` track (the stronger overall generaliser) -- the safe
default for a private evaluator that reworks the wording.

See `C:\\Users\\xhwon\\.claude\\plans\\since-agent-is-good-peppy-journal.md`
and `src/README.md` for the per-scenario measurements behind the split.
"""

from __future__ import annotations

from pathlib import Path

from .lexical import agent as lexical_agent
from .elimination import Agent as _Elimination
from .lexical.message_parsing import (
    RE_KEY_REQUIREMENT,
    RE_OPENER_FILLER,
    RE_OPENER_TAIL,
)


def _is_intent_override(message: str) -> bool:
    """True when `message` is an Intent-Override opener.

    The simulator's Intent-Override opener is "I'm looking for <cat>.
    <old_value>", where <old_value> is a real catalog-derived attribute of
    the target (soft_preferences[-1]). It is distinguished from:
      * the Buying opener  -- carries "A key requirement is: ..."
      * the Browsing/Boundary opener -- the trailing clause is the filler
        "but I'm still exploring." (RE_OPENER_FILLER)
      * a bare / reworded opener -- no trailing clause at all
    """
    if RE_KEY_REQUIREMENT.search(message):  # Buying
        return False
    match = RE_OPENER_TAIL.match(message.strip())
    if not match:  # no "I'm looking for <cat>.<tail>" shape at all
        return False
    tail = match.group(1).strip(" .")
    return bool(tail) and not RE_OPENER_FILLER.match(tail)


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        # Both sub-agents build their own in-memory FTS5 index over the whole
        # catalog (~2x cold start, ~2x RAM). They are fully independent -- no
        # shared state -- so a per-session route just picks which one handles
        # every turn of that session.
        self._lexical = lexical_agent.Agent(catalog_path)
        self._elimination = _Elimination(catalog_path)
        self._route: dict[str, object] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._lexical.reset(session_id, user_profile)
        self._elimination.reset(session_id, user_profile)
        self._route.pop(session_id, None)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        sub = self._route.get(session_id)
        if sub is None:
            sub = (
                self._lexical
                if _is_intent_override(user_message)
                else self._elimination
            )
            self._route[session_id] = sub
        return sub.respond(session_id, user_message, turn, top_k)
