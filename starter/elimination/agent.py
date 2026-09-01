"""The elimination agent: constraint-accumulating, elimination-first retrieval.

Each turn is evidence. Disclosed phrases are parsed out of the message,
accumulated across the session, and used as exact-phrase filters over the
catalog; ranking is a second pass over what survives. Buying and Browsing run
down separate tracks, early slates are trimmed to a single best candidate,
and a dry turn pages one slate deeper into the pool.

Self-contained: standard library only, no network, no model calls.
"""

from __future__ import annotations

from pathlib import Path

from .config import *  # noqa: F401,F403  -- see config.__all__
from .matcher import Matcher
from .session import SessionState

# The customer releases any undisclosed constraint for "other", and only
# same-class constraints for a typed attribute, so "other" dominates.
ASK_ATTRIBUTE = "other"

PROMPTS = (
    "Happy to narrow this down. What else matters to you here?",
    "Got it. Anything else about it that's important -- material, fit, or how you'll use it?",
    "Thanks. Is there any other detail I should be matching on?",
    "Noted. Anything else you'd want me to take into account?",
)

# Said on a turn whose slate is trimmed to the single best candidate. Which
# explanation is honest depends on why the rest was held back: either several
# candidates match everything disclosed so far equally well and padding the
# list would be guessing, or we simply have one clear front-runner and want a
# second detail before widening.
TIED_PROMPT = (
    "Several items match everything you've told me so far equally well, so I'd "
    "rather not guess between them. Here's the closest one -- and one more "
    "detail would let me narrow it properly. What else matters to you?"
)
NARROW_PROMPT = (
    "Here's the closest match I have so far. Tell me one more thing about what "
    "you're after and I'll widen the list to the best ten."
)


class Agent:
    """Constraint-accumulating retrieval agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.matcher = Matcher(catalog_path)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(user_profile, dict):
            user_profile = {}
        self._sessions[session_id] = SessionState(user_profile)

    def _choose_attribute(self, state: SessionState) -> tuple[str, dict]:
        """The next attribute to request. Always "other" -- the customer
        releases any undisclosed constraint for it, only same-class ones for a
        typed attribute (see ASK_ATTRIBUTE). Kept as a method (not inlined) so
        a smarter policy has one place to land."""
        return ASK_ATTRIBUTE, {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def _reveal(turn: int, top_k: int, slack: int = 0) -> int:
        """How many candidates this turn's slate is allowed to carry.

        `slack` is how many turns the customer spent disclosing nothing. It
        pushes the cap back by that much, because what the cap is really
        bounding is the evidence behind the slate, and an evidence-free turn
        advances the clock without advancing the evidence.

        The turn cap is the safety valve: past it the full slate always goes
        out, so trimming can cost us rank but never a hit.
        """
        if not EXPOSURE_ENABLED or turn > EXPOSURE_UNTIL_TURN + slack:
            return top_k
        return EXPOSURE_WIDTH

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 10
        if top_k <= 0:
            top_k = 10
        try:
            turn = int(turn)
        except (TypeError, ValueError):
            turn = 1
        if turn < 1:
            turn = 1
        if not isinstance(user_message, str):
            user_message = "" if user_message is None else str(user_message)

        state = self._sessions.get(session_id)
        if state is None:
            # respond() before reset() is a harness contract violation, but
            # recovering with a fresh state keeps the session alive.
            self.reset(session_id, {})
            state = self._sessions[session_id]
        try:
            state.absorb(user_message)
        except Exception:  # noqa: BLE001 -- a pathological message must not
            # break the turn; constraints heard on earlier turns still stand.
            pass
        ranking: dict = {}
        # Never allowed to break retrieval: an unrecognised phrase, or anything
        # unexpected here, leaves both tracks exactly as they were.
        try:
            bucket = self.matcher.resolve_bucket(state.category)
        except Exception:
            bucket = None

        # Dual-track routing. Constraints are what the precision track filters
        # on, so with none in hand we take the browsing track instead of
        # ranking the whole category by keyword weight alone.
        if state.constraints or not DUAL_TRACK_ROUTING:
            # Retrieve a pool rather than a page, then order it on the full
            # session. state.heard is a superset of the constraints the filter
            # applied and of the ones it had to drop, so it is all the evidence
            # there is.
            offset = state.dry * top_k if DEEP_PAGING else 0
            limit = POOL_SIZE if RERANK_ENABLED else top_k + offset
            pool, _ = self.matcher.search(
                state.category, state.constraints, limit, bucket
            )
            try:
                recommendations, ranking = self.matcher.rerank(
                    pool, state.heard, top_k, offset
                )
            except Exception:
                # The evaluator discards the whole turn if respond() raises, so
                # a reranking bug must cost us the ordering, not the results.
                start = max(0, min(offset, max(0, len(pool) - top_k)))
                recommendations = pool[start : start + top_k]
        else:
            # Nothing disclosed yet means nothing to rank on; the browse track
            # already spreads the slots across the category.
            recommendations, _ = self.matcher.browse(
                state.category, top_k, bucket=bucket
            )

        # Early turns go out as rank-1-or-nothing: the harness stops at the
        # first hit, so a full slate would lock in whatever rank the coin flip
        # below the top happened to give us for the rest of the session.
        reveal = self._reveal(turn, top_k, state.slack)
        trimmed = 0 < reveal < len(recommendations)
        recommendations = recommendations[:reveal]

        ask, usage = self._choose_attribute(state)
        state.record_ask(ask)

        if trimmed:
            message = TIED_PROMPT if ranking.get("tied", 0) > 1 else NARROW_PROMPT
        else:
            message = PROMPTS[(turn - 1) % len(PROMPTS)]

        return {
            "message": message,
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": usage,
        }
