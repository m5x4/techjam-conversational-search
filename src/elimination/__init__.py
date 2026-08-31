"""Elimination-first retrieval track.

Constraint-accumulating shopping agent: hard FTS5 phrase filtering, an
always-on diversified browse track, and dry-turn deep paging. One of the two
implementations behind the scenario router in ``src/agent.py``; the router
hands this one every Buying / Browsing / Boundary session.

Module map (mirrors the lexical twin's ``src/lexical/`` layout):

    config.py    every tuning constant, with rationale inline
    text.py      folding / tokenisation (folds differently -- not shared)
    parsing.py   simulator message templates -> Turn
    matcher.py   Constraint + Matcher (FTS5 index, filter, rerank, browse)
    session.py   SessionState -- accumulated dialogue state
    agent.py     Agent (reset / respond) + the customer-facing prompts
"""

from __future__ import annotations

from .agent import Agent

__all__ = ["Agent"]
