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
and `starter/lexical-track.md` / `starter/elimination-track.md` for the per-scenario
measurements behind the split.
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


_SAFE_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}
_DEFAULT_TOP_K = 10
_TOP_K_CAP = 200


def _empty_response() -> dict:
    """A contract-valid ``turn_response`` that recommends nothing.

    Last-resort fallback: the evaluator discards the whole session if
    ``respond()`` raises, so a failure anywhere below must degrade to an empty
    slate, never an exception."""
    return {
        "message": "Let me keep looking based on what you've told me.",
        "ask_attribute": "other",
        "recommendations": [],
        "usage": dict(_SAFE_USAGE),
    }


def _looks_valid(value: object) -> bool:
    """True when a sub-agent's return value already satisfies the response
    contract, so the router can pass it straight through untouched (the
    happy path -- no coercion, no behaviour change)."""
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("message"), str):
        return False
    ask = value.get("ask_attribute", None)
    if ask is not None and not isinstance(ask, str):
        return False
    return isinstance(value.get("recommendations"), list)


def _coerce_response(value: object, top_k: int) -> dict:
    """Force a malformed / partial sub-agent return value into the response
    shape. Only ever runs on the failure path -- a valid response is returned
    verbatim by ``respond`` before this is reached.

    Invariant enforced here (not on the happy path): ``recommendations`` is a
    list of ``{"parent_asin": <str>}``, de-duplicated, no longer than
    ``top_k``."""
    out = _empty_response()
    if not isinstance(value, dict):
        return out
    msg = value.get("message")
    if isinstance(msg, str) and msg:
        out["message"] = msg
    ask = value.get("ask_attribute", "other")
    if ask is None or isinstance(ask, str):
        out["ask_attribute"] = ask
    recs = value.get("recommendations")
    if isinstance(recs, list):
        seen: set[str] = set()
        clean: list[dict] = []
        for entry in recs:
            pid = entry.get("parent_asin") if isinstance(entry, dict) else None
            if isinstance(pid, str) and pid not in seen:
                seen.add(pid)
                clean.append({"parent_asin": pid})
            if len(clean) >= max(0, top_k):
                break
        out["recommendations"] = clean
    usage = value.get("usage")
    if isinstance(usage, dict):
        out["usage"] = usage
    return out


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
        # every turn of that session. If one track fails to build (a bad
        # catalog row, an FTS5 quirk on a reworded private catalog), the other
        # still serves every session rather than the whole agent being dead.
        self._lexical: object | None = None
        self._elimination: object | None = None
        errors: list[Exception] = []
        try:
            self._lexical = lexical_agent.Agent(catalog_path)
        except Exception as exc:  # noqa: BLE001 -- degrade to the other track
            errors.append(exc)
        try:
            self._elimination = _Elimination(catalog_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        if self._lexical is None and self._elimination is None:
            raise RuntimeError(
                f"both retrieval tracks failed to initialise: {errors!r}"
            )
        self._route: dict[str, object] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        session_id = str(session_id)
        if not isinstance(user_profile, dict):
            user_profile = {}
        for sub in (self._lexical, self._elimination):
            if sub is None:
                continue
            try:
                sub.reset(session_id, user_profile)
            except Exception:  # noqa: BLE001 -- a reset must never raise
                pass
        self._route.pop(session_id, None)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        session_id = str(session_id)
        if not isinstance(user_message, str):
            user_message = "" if user_message is None else str(user_message)
        try:
            turn = int(turn)
        except (TypeError, ValueError):
            turn = 1
        if turn < 1:
            turn = 1
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = _DEFAULT_TOP_K
        if top_k <= 0:
            top_k = _DEFAULT_TOP_K
        top_k = min(top_k, _TOP_K_CAP)

        sub = self._route.get(session_id)
        if sub is None:
            try:
                intent = _is_intent_override(user_message)
            except Exception:  # noqa: BLE001 -- classification is best-effort
                intent = False
            preferred = self._lexical if intent else self._elimination
            sub = preferred or self._lexical or self._elimination
            self._route[session_id] = sub

        # Route to the chosen track; on an exception or a malformed return,
        # fall back to the other track once, then to an empty slate. The
        # evaluator zeroes any session whose respond() raises, so nothing
        # below this point is allowed to propagate.
        fallback: object | None = None
        tried: list[object] = []
        for candidate in (sub, self._elimination, self._lexical):
            if candidate is None or any(candidate is t for t in tried):
                continue
            tried.append(candidate)
            try:
                result = candidate.respond(session_id, user_message, turn, top_k)
            except Exception:  # noqa: BLE001 -- try the other track, then give up
                continue
            if _looks_valid(result):
                return result
            fallback = result  # keep the best-effort junk for coercion
        return _coerce_response(fallback, top_k)
