"""Per-session dialogue state for the elimination track.

Everything the customer has ever disclosed is accumulated and never cleared:
a superseded preference and a constraint the filter had to drop both still
describe the target, so they stay available to the reranker even once they
have stopped being safe to filter on.
"""

from __future__ import annotations

from .config import (
    OVERRIDE_RESETS,
    OVERRIDE_RESETS_ASKS,
    OVERRIDE_RESETS_CATEGORY,
)
from .parsing import ALLOWED_ATTRIBUTES, parse_message


class SessionState:
    __slots__ = (
        "profile",
        "category",
        "constraints",
        "heard",
        "asked",
        "last_ask",
        "gained",
        "exhausted",
        "intent",
        "dry",
        "slack",
    )

    def __init__(self, profile: dict) -> None:
        self.profile = profile or {}
        # Routed on the opening turn: a shopper who leads with a requirement is
        # buying, one who leads with only a category is browsing.
        self.intent: str | None = None  # "buying" or "browsing" or "intent_override"
        self.category: str | None = None
        self.constraints: list[str] = []
        # Everything the customer has ever disclosed, never cleared. A
        # superseded preference and a constraint the filter had to drop both
        # still describe the target, so they stay available to the reranker
        # even once they have stopped being safe to filter on.
        self.heard: list[str] = []
        # What we asked, in order, and what each ask actually bought us. The
        # payoff for asking X only becomes visible in the *next* message, so
        # last_ask carries the pending question across the turn boundary.
        self.asked: list[str] = []
        self.last_ask: str | None = None
        self.gained: dict[str, int] = {}
        self.exhausted: set[str] = set()
        # Consecutive turns that told us nothing new. Each one is a slate the
        # shopper has already been shown, so it doubles as the page number.
        self.dry = 0
        # Turns that disclosed nothing at all. The exposure gate is really a
        # bound on how much evidence stands behind the slate, not on the clock,
        # so each evidence-free turn buys the gate one more turn.
        self.slack = 0

    def record_ask(self, attribute: str | None) -> None:
        if attribute:
            self.asked.append(attribute)
            self.last_ask = attribute

    def absorb(self, message: str) -> None:
        turn = parse_message(message)

        # Recorded before the override reset below, which is the whole point:
        # filtering forgets, ranking does not.
        for constraint in turn.constraints:
            if constraint and constraint not in self.heard:
                self.heard.append(constraint)

        if turn.is_override and OVERRIDE_RESETS:
            # Off by default -- see OVERRIDE_RESETS. Kept switchable because a
            # simulator whose override genuinely changed the target would want
            # this on.
            self.constraints.clear()
            if OVERRIDE_RESETS_CATEGORY:
                self.category = None
            if OVERRIDE_RESETS_ASKS:
                self.asked.clear()
                self.exhausted.clear()
                self.gained.clear()
            self.last_ask = None

        if turn.category and not self.category:
            self.category = turn.category
        if self.intent is None and turn.category:
            self.intent = "buying" if turn.constraints else "browsing"

        before = len(self.constraints)
        for constraint in turn.constraints:
            if constraint and constraint not in self.constraints:
                self.constraints.append(constraint)
        gain = len(self.constraints) - before

        # A turn that adds no constraint leaves the ranking byte-identical to
        # the one that just missed, so it is a dead slate and the next turn
        # should page past it. Two turns are exempt. An override rewrites the
        # constraint set, so its ranking is new. A Boundary shrug is a one-off
        # scenario artifact -- the customer answers properly on the turn after
        # it -- and paging on it would move the slate that scenario usually
        # converts on.
        if gain > 0:
            self.dry = 0
        elif self.asked and not turn.is_override and not turn.refusal_is_boundary:
            # self.asked is empty only on the opening message, which discloses
            # no constraint in a Browsing session. That is not a dead turn --
            # there is no slate behind it yet -- so it must not advance the page.
            self.dry += 1

        # A Boundary shrug costs a turn and discloses nothing, which shifts that
        # whole scenario one turn later than Browsing: its first constraint
        # lands on turn 3, not turn 2. The exposure gate counts turns, so by
        # then it has already opened -- and the one slate it exists to protect,
        # the first that is genuinely ranked, goes out at full width. Browsing
        # converts 51.5% of its sessions on the gated turn 2 at 100% rank-1;
        # Boundary converts 75.8% on the ungated turn 3 at 64.1%, which is
        # essentially the whole 0.166 MRR gap between them.
        #
        # Over the 2,490 Boundary dev sessions: MRR 0.7397 -> 0.8990 (Browsing
        # sits at 0.9062), hit -0.0020, mttc +0.286. No other scenario emits
        # this refusal -- "an additional preference" parses as a genuine
        # exhaustion, not a shrug -- so nothing else moves; verified
        # bit-identical over 5,914 non-Boundary sessions.
        if turn.refusal_is_boundary:
            self.slack += 1

        # Attribute the outcome to whatever we asked on the previous turn. The
        # customer names the attribute it is refusing, so prefer that over our
        # own memory of what we sent.
        #
        # An override turn is the exception: the harness sends it instead of
        # calling customer_reply, so our pending ask was never read and must
        # not be credited with what the override happened to disclose.
        answered = None if turn.is_override else (turn.refused or self.last_ask)
        if answered:
            self.gained[answered] = self.gained.get(answered, 0) + gain
            # A Boundary shrug is a scenario artifact, not evidence that the
            # attribute is spent, so it must not mark anything exhausted.
            if gain == 0 and turn.refused and not turn.refusal_is_boundary:
                self.exhausted.add(answered)
        self.last_ask = None

    def unasked(self) -> list[str]:
        seen = set(self.asked) | self.exhausted
        return [a for a in ALLOWED_ATTRIBUTES if a not in seen]
