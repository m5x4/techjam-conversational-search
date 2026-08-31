"""Lexical retrieval track -- the Intent-Override route.

bucket pre-filter -> two-tier BM25 -> linear rerank -> window. One of the two
implementations behind the scenario router in ``src/agent.py``; the router
hands this one every turn of an Intent-Override session.

Module map:

    agent.py            the lexical ``Agent`` (reset / respond): retrieve ->
                        rerank -> window, THIN / CONF gates
    config.py           every swept tuning constant, one place
    text_utils.py       tokenisation + normalisation helpers
    budget.py           disclosed-budget -> (operator, amount) parsing
    message_parsing.py  simulator-message regexes + constraint classification
    session.py          _SessionState dialogue state machine
    buckets.py          coarse-category bucket resolution (pre-retrieval filter)
    catalog_index.py    BM25 (SQLite FTS5) + blob / field-value / popularity maps
"""

from __future__ import annotations

from .agent import Agent

__all__ = ["Agent"]
