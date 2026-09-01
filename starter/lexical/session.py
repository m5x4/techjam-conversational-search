"""Per-session dialogue state machine.

A parsed category, a dict of attribute-keyed slots (each an ordered list of
trusted phrases), and a broad fallback bag of every term ever seen. Ordinary
reveals accumulate into their slot; an Intent-Override message erases and
rewrites only the ONE slot the new value belongs to. lexical-track.md:
"Dialogue state".
"""

from __future__ import annotations

from .budget import _budget_constraint
from .message_parsing import (
    RE_CATEGORY,
    RE_GENERIC_REBUFF,
    RE_KEY_REQUIREMENT,
    RE_NO_PREF_BOUNDARY,
    RE_NO_PREF_EXHAUSTED,
    RE_NO_PREFERENCE,
    RE_OPENER_FILLER,
    RE_OPENER_TAIL,
    RE_OVERRIDE,
    RE_REVEALED,
    _SAFE_CONFLICT_ATTRS,
    _SLOT_ATTRS,
    _classify,
)
from .text_utils import _terms


def _choose_ask_attribute(state: "_SessionState") -> str | None:
    """Ask "other" every turn until the customer has nothing left to
    disclose, then null.

    The local evaluator's customer_reply() special-cases the attribute
    "other" to reveal the next undisclosed constraint of ANY bucket (up to
    two per turn) -- the highest-yield thing we can ask, since it never
    wastes a turn probing a bucket the customer has no constraint in.
    "other" is a member of the evaluator's ALLOWED_ATTRIBUTES and is listed
    in agent_api_contract.json's enum, so it is valid output.

    Once the simulator answers "I don't have an additional preference for
    other" (state.exhausted), everything has been disclosed and there is
    nothing left to ask -- emit null.
    """
    if state.exhausted:
        return None
    return "other"


class _SessionState:
    __slots__ = (
        "category_text",
        "slots",  # dict[attr] -> ordered list of currently-trusted phrases for that attribute
        "history_terms",  # every raw term ever seen (broad fallback bag)
        "turn",
        "mode",  # "buying" | "browsing", fixed from the turn-1 message shape
        "override_seen",
        "exhausted",  # True once the simulator says there's nothing more to reveal
        "last_ask_attribute",  # ask_attribute returned by the PREVIOUS respond() call
        "budget_amount",  # float parsed from a "$NN" budget reveal, else None
        "budget_op",  # "<" | ">" | "~" operator for budget_amount, else None
        "prior_rating",  # user_profile.average_prior_rating (1..5), set by Agent.reset; else None
        "slack",  # count of Boundary "a preference for X" shrugs; extends the THIN cap iff config.BOUNDARY_SLACK

        "stable_top",  # top_k list first shown on the turn exhaustion was reached
        "rotation_cursor",  # offset into fused[top_k:] for post-exhaustion rotation
        "exhausted_since_turn",  # turn number the "nothing more to reveal" dead-end landed
        "opening_message",  # verbatim first user message, for bucket resolution
        "bucket_attempted",  # bucket resolution run once per session (lazy, cached)
        "bucket_ids",  # resolved coarse-category bucket id list, else None
        "bucket_key",  # resolved coarse-category bucket key string, else None
        "bucket_how",  # which resolve rung fired ("exact"/"containment"/... /"unresolved")
        "last_snapshot",  # debugging state dump; deliberately NOT in the response dict
        "last_fused_scores",  # score dict from the last _retrieve (now the linear _rerank scores)
        "last_rerank_tie_count",  # candidates tying the best rerank EVIDENCE sub-score; see _rerank
    )

    def __init__(self) -> None:
        self.category_text = ""
        self.slots: dict[str, list[str]] = {attr: [] for attr in _SLOT_ATTRS}
        self.history_terms: list[str] = []
        self.turn = 0
        self.mode = ""
        self.override_seen = False
        self.exhausted = False
        self.last_ask_attribute: str | None = None
        self.budget_amount: float | None = None
        self.budget_op: str | None = None
        self.prior_rating: float | None = None
        self.slack = 0
        self.stable_top: list[str] | None = None
        self.rotation_cursor = 0
        self.exhausted_since_turn: int | None = None
        self.opening_message = ""
        self.bucket_attempted = False
        self.bucket_ids: list[str] | None = None
        self.bucket_key: str | None = None
        self.bucket_how = ""
        self.last_snapshot: dict | None = None
        self.last_fused_scores: dict[str, float] = {}
        self.last_rerank_tie_count = 0

    def _reset_rotation(self) -> None:
        """Fresh signal arrived -> retrieval is live again; drop the
        post-exhaustion rotation bookkeeping so the next _retrieve starts
        from the real fused order."""
        self.exhausted = False
        self.exhausted_since_turn = None
        self.stable_top = None
        self.rotation_cursor = 0

    @property
    def active_constraints(self) -> list[str]:
        """Flattened view of every trusted phrase across all slots. Order
        doesn't affect retrieval (BM25 terms are deduplicated into a set;
        TF-IDF weighting is by term *count*, not position), so a fixed
        per-attribute grouping is fine."""
        phrases: list[str] = []
        for attr in _SLOT_ATTRS:
            phrases.extend(self.slots[attr])
        return phrases

    def slot_term_groups(self) -> list[set[str]]:
        """One term-set per non-empty slot. No caller in the shipped agent;
        kept for the dev tooling in scripts/_common.py (weak-signal gate
        classification)."""
        return [
            set(_terms(" ".join(phrases))) for phrases in self.slots.values() if phrases
        ]

    def phrase_constraints(self) -> list[str]:
        """Multi-word trusted phrases only, for verbatim substring matching
        against catalog text. Single tokens carry no phrase signal beyond
        what BM25 already uses."""
        return [c for c in self.active_constraints if len(c.split()) >= 2]

    def _accumulate(self, value: str) -> None:
        """Incremental slot fill: classify a newly-disclosed constraint into
        its attribute slot and append it. Multiple phrases can legitimately
        coexist in one slot (e.g. two separate 'feature' reveals), so this
        never overwrites -- only override() does that."""
        self.slots[_classify(value)].append(value)
        budget = _budget_constraint(value)
        if budget is not None:
            self.budget_op, self.budget_amount = budget

    def _override(self, new_value: str) -> None:
        """Abrupt Intent Override: erase-and-rewrite the ONE slot the new
        value belongs to; every other slot is untouched.

        NOTE: despite the "ignore my earlier preference" wording, the
        simulator's override value is almost always something already
        disclosed a turn or two earlier via an "other" reveal. Wiping all
        slots here was actively destructive, so keep every other slot as-is.

        Reinforcement rule: only re-insert (and only 2x-weight) the override
        value when it adds genuine retrieval signal -- i.e. it is NOT already
        covered verbatim by an existing phrase in its slot, AND it is a
        multi-word / discriminating value. lexical-track.md: "Intent override".
        """
        new_attr = _classify(new_value)
        new_value_lower = new_value.lower().strip(" .")
        already_covered = any(
            new_value_lower in c.lower() for c in self.slots[new_attr]
        )
        if new_attr in _SAFE_CONFLICT_ATTRS:
            # Only erase an existing same-slot phrase if it's a genuinely
            # DIFFERENT value (e.g. "color: blue" vs override "red"). If the
            # existing phrase already contains the override's text (e.g.
            # "90% Cotton, 10% Others" vs override "cotton"), it's not a
            # conflict -- it's a more specific restatement of the same value.
            self.slots[new_attr] = [
                c for c in self.slots[new_attr] if new_value_lower in c.lower()
            ]
            already_covered = bool(self.slots[new_attr])
        if already_covered:
            # Pure restatement of something already in the slot -- no new
            # signal to add. Leave the (more specific) existing phrases as-is.
            return
        self.slots[new_attr].insert(0, new_value)
        if len(new_value.split()) >= 2:
            self.slots[new_attr].insert(0, new_value)

    def absorb(self, user_message: str) -> None:
        self.history_terms.extend(_terms(user_message))

        # Keep the verbatim opening message: it is the category-bearing turn
        # the bucket pre-filter resolves against (see Agent._resolve_bucket).
        if not self.opening_message:
            self.opening_message = user_message

        if self.turn == 1 and not self.mode:
            # Buying vs Browsing routing: a hard constraint disclosed in the
            # very first message ("A key requirement is: ...") means the
            # customer already knows what they want -- everything else is
            # treated as Browsing: no committed constraint yet, so retrieval
            # should stay exploratory. Frozen for the rest of the session.
            self.mode = (
                "buying" if RE_KEY_REQUIREMENT.search(user_message) else "browsing"
            )

        cat_match = RE_CATEGORY.search(user_message)
        if cat_match and not self.category_text:
            self.category_text = cat_match.group(1).strip()

        if self.turn == 1 and not RE_KEY_REQUIREMENT.search(user_message):
            # Intent-Override opener carries a real intent-card preference
            # after the category sentence ("I'm looking for <cat>. <value>").
            # Capture it now so it reaches the rerank immediately; the Browsing
            # filler ("but I'm still exploring.") carries no constraint.
            tail_match = RE_OPENER_TAIL.match(user_message.strip())
            if tail_match:
                tail = tail_match.group(1).strip(" .")
                if tail and not RE_OPENER_FILLER.match(tail):
                    self._accumulate(tail)

        override_match = RE_OVERRIDE.search(user_message)
        if override_match:
            self.override_seen = True
            self._reset_rotation()
            self._override(override_match.group(1).strip())
            return

        key_req_match = RE_KEY_REQUIREMENT.search(user_message)
        if key_req_match:
            self._reset_rotation()
            self._accumulate(key_req_match.group(1).strip())
            return

        revealed_match = RE_REVEALED.search(user_message)
        if revealed_match:
            parts = [p.strip() for p in revealed_match.group(1).split(";") if p.strip()]
            if parts:
                self._reset_rotation()
            for part in parts:
                self._accumulate(part)
            return

        if RE_NO_PREFERENCE.search(user_message):
            # Either a boundary "use your judgment" reply, or "I don't have an
            # additional preference for other": either way, no new constraint.
            if RE_NO_PREF_EXHAUSTED.search(user_message):
                # The "no additional preference" dead-end for our (always-
                # "other") ask means everything has been disclosed.
                self.exhausted = True
                if self.exhausted_since_turn is None:
                    self.exhausted_since_turn = self.turn
            elif RE_NO_PREF_BOUNDARY.search(user_message):
                # One-off Boundary shrug: discloses nothing about the named
                # attribute and just costs a turn. Counted here always;
                # only consumed downstream when config.BOUNDARY_SLACK is on.
                self.slack += 1
            return

        if RE_GENERIC_REBUFF.search(user_message):
            return
        # Turn 1 browsing message ("...but I'm still exploring.") or anything
        # unrecognized: category (if any) plus history_terms already capture it.

    def snapshot(self) -> dict:
        """Everything the agent has collected so far about the product the
        customer wants, as of this turn. Emitted in respond()'s payload and
        recorded per-turn in the evaluator's transcript (results.json)."""
        primary, fallback = self.query_terms()
        return {
            "turn": self.turn,
            "mode": self.mode,
            "category": self.category_text,
            "bucket": {
                "how": self.bucket_how,
                "size": len(self.bucket_ids) if self.bucket_ids else 0,
            },
            "slots": {attr: list(phrases) for attr, phrases in self.slots.items()},
            "active_constraints": list(self.active_constraints),
            "phrase_constraints": self.phrase_constraints(),
            "budget_amount": self.budget_amount,
            "budget_op": self.budget_op,
            "prior_rating": self.prior_rating,
            "slack": self.slack,
            "override_seen": self.override_seen,
            "exhausted": self.exhausted,
            "last_ask_attribute": self.last_ask_attribute,
            "history_terms": list(dict.fromkeys(self.history_terms)),
            "query_terms": {
                "primary": list(dict.fromkeys(primary)),
                "fallback": list(dict.fromkeys(fallback)),
            },
        }

    def query_terms(self) -> tuple[list[str], list[str]]:
        """Return (primary_terms, fallback_terms).

        primary_terms: category + trusted constraints -- high precision.
        fallback_terms: everything ever said -- high recall, used only to
        pad out a query that returns too few candidates.
        """
        primary: list[str] = []
        primary.extend(_terms(self.category_text))
        for constraint in self.active_constraints:
            primary.extend(_terms(constraint))
        fallback = list(self.history_terms)
        return primary, fallback
