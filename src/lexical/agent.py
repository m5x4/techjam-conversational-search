"""Multi-turn conversational-search agent -- the lexical retrieval track.

One of the two implementations behind the scenario router in `agent.py`; the
router hands this one every turn of an Intent-Override session. `Agent`
(reset / respond) is the same interface the router and evaluator expect. The
implementation is split across sibling modules:

    config.py           tuning constants (swept; see README)
    text_utils.py       tokenisation + normalisation helpers
    budget.py           disclosed-budget -> (operator, amount) parsing
    message_parsing.py  simulator-message regexes + constraint classification
    session.py          _SessionState dialogue state machine
    buckets.py          coarse-category bucket resolution (pre-retrieval filter)
    catalog_index.py    BM25 (SQLite FTS5) + blob / field-value / popularity maps

Strategy, tuning history and rejected experiments are written up in
`src/README.md`.

Runtime overview:
  * ask_attribute emits "other" every turn until the simulator has nothing
    left to disclose, then null (session._choose_ask_attribute).
  * Dialogue state is a small state machine: a parsed category, attribute-
    keyed slots of trusted phrases, and a broad fallback term bag. Ordinary
    reveals accumulate into their slot; an Intent-Override rewrites only the
    one slot the new value belongs to.
  * Retrieval: two-tier BM25 (precise AND + broad OR), scoped to the
    resolved coarse-category bucket when there is one, produces a candidate
    pool. The pool is ordered by a single linear rerank (_rerank): EXACT_W
    per disclosed constraint equal to a whole feature/detail value, LOOSE_W
    per disclosed constraint found as a substring, POP_W * pool-normalised
    popularity, minus a small RANK_W * incoming-BM25-position penalty. A
    hard "<"/">" budget split is applied last as a stable sort. Everything
    is computed from the local catalog -- no network / model download.
  * Buying vs Browsing routing: the turn-1 message shape fixes state.mode
    for the session. Buying (opener carries "A key requirement is: ...")
    trusts a thin-but-precise pool as-is; Browsing widens the pool
    proactively. ask_attribute strategy is unaffected by mode.
"""

from __future__ import annotations

from pathlib import Path

from .catalog_index import _CatalogIndex
from .config import *  # noqa: F401,F403  -- re-exported so sweep scripts can mutate them
from .message_parsing import _SYNTHETIC_PHRASE_RE
from .session import _SessionState, _choose_ask_attribute
from .text_utils import _norm, _terms


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.index = _CatalogIndex(catalog_path)
        self._sessions: dict[str, _SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = _SessionState()
        # Anonymized profile can lightly steer retrieval. `preference_tags` are
        # generic ("fit", "comfort", ...) so we fold them into the low-weight
        # fallback bag rather than the precise query. `average_prior_rating`
        # feeds the optional rerank rating-affinity term (config.PRIOR_RATING_*,
        # default OFF). The other profile fields are deliberately unused:
        # `purchase_frequency` is a dataset constant, `summary` restates
        # `preference_tags`, `rating_style` is a coarser `average_prior_rating`
        # (README: "User-profile rating affinity").
        tags = (
            user_profile.get("preference_tags")
            if isinstance(user_profile, dict)
            else None
        )
        if isinstance(tags, list):
            for tag in tags:
                state.history_terms.extend(_terms(str(tag)))
        apr = (
            user_profile.get("average_prior_rating")
            if isinstance(user_profile, dict)
            else None
        )
        if (
            isinstance(apr, (int, float))
            and not isinstance(apr, bool)
            and 0 < float(apr) <= 5
        ):
            state.prior_rating = float(apr)
        self._sessions[session_id] = state

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn = turn
        state.absorb(user_message)

        recommendations = self._retrieve(state, top_k)

        # Thin-signal guard (config.THIN_*): on an early turn with a still-
        # thin constraint signal and the simulator not yet exhausted, a
        # target at rank 4..10 is nearly always a popularity-prior
        # coincidence the next "other" reveal lifts to rank 1..3. Returning
        # only THIN_KEEP ids defers that weak hit one turn; retrieval is
        # monotonic so it re-enters at the same-or-better rank -- ~1 turn of
        # MTTC at worst, never a new miss. README: "Thin-signal guard".
        # config.BOUNDARY_SLACK (README: "Combined v1+v2 features"): a
        # Boundary-scenario "I don't have A preference for X" shrug discloses
        # nothing and shifts that session one turn later, so each shrug buys
        # the turn cap one more turn -- what the cap really bounds is the
        # evidence behind the slate, not the clock. Default OFF -> the cap is
        # exactly THIN_MAX_TURN.
        thin_cap = THIN_MAX_TURN + (state.slack if BOUNDARY_SLACK else 0)
        if (
            THIN_ENABLE
            and turn <= thin_cap
            and not state.exhausted
            and len(state.phrase_constraints()) <= THIN_PHRASE_MAX
            and len(recommendations) > THIN_KEEP
        ):
            recommendations = recommendations[:THIN_KEEP]

        # EXPERIMENTAL post-rerank confidence gate (config.CONF_GATE_*, default
        # OFF). Defer a weak early hit when the linear-rerank score margin is
        # thin. README: "Confidence gate".
        if (
            CONF_GATE_ENABLE
            and turn <= CONF_GATE_MAX_TURN
            and not state.exhausted
            and len(recommendations) > CONF_GATE_KEEP
            and len(recommendations) >= 2
        ):
            sc = state.last_fused_scores or {}
            top_score = sc.get(recommendations[0], 0.0)
            ref_idx = min(len(recommendations), top_k) - 1
            ref_score = sc.get(recommendations[ref_idx], 0.0)
            if ref_score > 0.0:
                ratio = top_score / ref_score
                if ratio < CONF_GATE_RATIO:
                    recommendations = recommendations[:CONF_GATE_KEEP]

        ask_attribute = _choose_ask_attribute(state)
        state.last_ask_attribute = ask_attribute

        message = self._compose_message(state, bool(recommendations), ask_attribute)

        # Intent snapshot kept for our own debugging only. The contract's
        # `turn_response` sets additionalProperties:false and does not list
        # "collected_intent", so emitting it fails validation. `usage` is
        # allowed and stays.
        state.last_snapshot = state.snapshot()

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": pid} for pid in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # -- internals ------------------------------------------------------

    def _resolve_bucket(self, state: _SessionState) -> list[str] | None:
        """Resolve the session's opening line to a coarse-category bucket,
        once per session (cached on state). Returns the bucket's id list, or
        None to fall through to the whole-catalog BM25 pipeline."""
        if not state.bucket_attempted:
            state.bucket_attempted = True
            source = state.opening_message or state.category_text
            key, how = self.index.bucket_index.resolve(source)
            state.bucket_how = how
            ids = self.index.bucket_index.ids(key) if key is not None else []
            state.bucket_ids = list(ids) if ids else None
            # Keep the resolved key: it scopes memoised per-bucket work (two
            # different buckets can share a `how` of "exact").
            state.bucket_key = key if (key is not None and ids) else None
        return state.bucket_ids

    def _retrieve(self, state: _SessionState, top_k: int) -> list[str]:
        primary_terms, fallback_terms = state.query_terms()

        and_terms = list(dict.fromkeys(primary_terms))
        or_terms = list(dict.fromkeys(primary_terms + fallback_terms))

        # Bucket pre-filter: when the opener resolves to a coarse-category
        # bucket, the candidate pool IS that bucket -- a category-pure,
        # median-~180-item set guaranteed (on the public set) to contain the
        # target. The two-tier BM25 runs SCOPED to the bucket and `limit` is
        # the full bucket size, so nothing is truncated. The only back-fill
        # is for bucket members that match no query term at all (near-always
        # none); it exists purely so the WHOLE bucket still rides through the
        # rerank -- the recall floor the 500-cap whole-catalog pool lacked.
        # Unresolved -> the whole-catalog pipeline runs verbatim (private-set
        # safety net). README: "Bucket pre-filter".
        bucket_ids = self._resolve_bucket(state)
        if bucket_ids is not None:
            pool = self.index.bm25_search_scoped(
                and_terms, or_terms, bucket_ids, limit=len(bucket_ids)
            )
            covered = set(pool)
            tail = [pid for pid in bucket_ids if pid not in covered]
            if tail:
                tail.sort(
                    key=lambda pid: self.index.popularity.get(pid, 0.0), reverse=True
                )
                pool = pool + tail
        else:
            pool = self.index.bm25_search(and_terms, or_terms, BM25_POOL)
            # Buying trusts a thin-but-precise pool as-is; Browsing widens
            # proactively (well above bare top_k) to give the rerank more
            # candidates to differentiate. README: "Buying vs Browsing".
            widen_threshold = (
                top_k if state.mode == "buying" else top_k * _BROWSING_WIDEN_FACTOR
            )
            if len(pool) < widen_threshold and fallback_terms:
                wide_pool = self.index.bm25_search(
                    [], or_terms + fallback_terms, BM25_POOL
                )
                pool = list(dict.fromkeys(pool + wide_pool))

        if not pool:
            return []

        fused = self._rerank(state, pool)
        # config.BROWSE_DIVERSITY (README: "Combined v1+v2 features"): a
        # no-constraint Browsing turn has nothing to rank on, so ten
        # near-neighbours all teach the same thing. Spread the head across
        # distinct stores / title-shapes before _window slices it, so the
        # slate probes different sub-types. Default OFF.
        if (
            BROWSE_DIVERSITY
            and state.mode == "browsing"
            and not state.active_constraints
        ):
            fused = self._diversify(fused, top_k)
        return self._window(state, fused, top_k)

    def _diversify(self, fused: list[str], top_k: int) -> list[str]:
        """Reorder `fused` so the first `top_k` cover distinct stores and
        distinct title-shapes (first-after-brand 3 title tokens), skipped
        duplicates appended behind them. Ported from agent_v2.py's browse
        track. Never drops an id -- `_window` still gets the full list."""
        picked: list[str] = []
        deferred: list[str] = []
        seen_store: set[str] = set()
        seen_shape: set[str] = set()
        for pid in fused:
            brand = _norm(self.index.store.get(pid, ""))
            shape = " ".join(_terms(self.index.title.get(pid, ""))[1:4])
            if (brand and brand in seen_store) or (shape and shape in seen_shape):
                deferred.append(pid)
                continue
            picked.append(pid)
            if brand:
                seen_store.add(brand)
            if shape:
                seen_shape.add(shape)
            if len(picked) >= top_k:
                break
        result = picked + deferred
        placed = set(result)
        result.extend(pid for pid in fused if pid not in placed)
        return result

    def _rerank(self, state: _SessionState, pool: list[str]) -> list[str]:
        """Single linear-scoring rerank of the BM25 / bucket `pool`.

            score(pid) = EXACT_W * (# disclosed constraints that equal one
                                    whole feature/detail value on this item)
                       + LOOSE_W * (# disclosed constraints occurring as a
                                    substring anywhere in the flattened blob)
                       + POP_W   * (popularity(pid) / max popularity in pool)
                       - RANK_W  * (pool_position / len(pool))

        "disclosed constraints" == state.active_constraints, _norm-folded and
        de-duplicated. Simulator-synthetic "<attr>: <value>" / "budget around
        $..." phrases (_SYNTHETIC_PHRASE_RE) are dropped from the EXACT term
        (not verbatim catalog text) but LEFT IN the LOOSE term (their colon
        shape practically never occurs in catalog text). The incoming BM25 /
        bucket order is folded in as the small RANK_W penalty and is also the
        tiebreak on an exact score tie.

        A known hard-budget ("<"/">") price violation still demotes a
        candidate regardless of score: _price_rank_map is applied last as a
        stable sort. README: "Linear rerank".
        """
        if not pool:
            state.last_rerank_tie_count = 0
            state.last_fused_scores = {}
            return pool

        idx = self.index
        depth = len(pool)

        constraints = list(
            dict.fromkeys(
                c for c in (_norm(x) for x in state.active_constraints) if c
            )
        )
        # A simulator-synthetic "<attr>: <value>" phrase is normally dropped
        # from EXACT (the catalog doesn't write attributes with a colon) --
        # but the evaluator also emits verbatim `details` entries in the same
        # "Key: value" shape, and those ARE real discrete field values that
        # the colon-free blob (LOOSE) never matches either. Rescue such a
        # phrase into EXACT only when it is BOTH a real field value of some
        # pool member AND discriminating -- carried by no more than
        # SYNTH_RESCUE_MAX_DF_FRAC of the pool. A near-universal value
        # ("color: black", "department: womens") stays dropped: crediting it
        # just shifts ties toward the popular decoys that also carry it.
        # README: "Synthetic phrase / real field value".
        rescued: set[str] = set()
        if SYNTH_RESCUE_MAX_DF_FRAC > 0.0:
            synth = [c for c in constraints if _SYNTHETIC_PHRASE_RE.match(c)]
            if synth:
                df_cap = max(1, int(SYNTH_RESCUE_MAX_DF_FRAC * len(pool)))
                df = {c: 0 for c in synth}
                for pid in pool:
                    fv = idx.field_values.get(pid)
                    if not fv:
                        continue
                    for c in synth:
                        if c in fv:
                            df[c] += 1
                rescued = {c for c, n in df.items() if 0 < n <= df_cap}
        exact_constraints = [
            c
            for c in constraints
            if not _SYNTHETIC_PHRASE_RE.match(c) or c in rescued
        ]
        loose_constraints = constraints

        pool_pops = [idx.popularity.get(pid, 0.0) for pid in pool]
        max_pop = max(pool_pops, default=0.0)
        pop_denom = max_pop if max_pop > 0.0 else 1.0

        # User-profile rating-affinity penalty (config.PRIOR_RATING_*;
        # PRIOR_RATING_W = 0.0 disables it). A pool candidate whose catalog
        # average_rating diverges from the user's average_prior_rating is
        # demoted by PRIOR_RATING_W * min(|cat_rating - prior| / 2.0, 1.0),
        # gated to priors below PRIOR_RATING_MAX (shipped 5.1 -> every rating;
        # 5.0 was tested and is worse -- see config). Unrated items (rating
        # 0.0) are neutral. Score-only: evidence / tie-count are untouched.
        # README: "User-profile rating affinity".
        prior_rating = state.prior_rating
        use_prior = (
            PRIOR_RATING_W > 0.0
            and prior_rating is not None
            and prior_rating < PRIOR_RATING_MAX
        )
        # Ported from agent_v2.py (README: "Combined v1+v2 features"). Both
        # are score-only additive terms -- evidence / tie-count untouched --
        # and both are inert at their config default of 0.0.
        #   POSITION_W: reward a disclosed constraint that lands on an early
        #     feature/detail entry (the defining attribute) over one buried
        #     deep (an incidental mention). 1 / (1 + earliest_entry_index).
        #   VELOCITY_W: rating_number / listing-age, added ALONGSIDE POP_W.
        use_position = POSITION_W > 0.0
        use_velocity = VELOCITY_W > 0.0

        evidence: dict[str, float] = {}
        score: dict[str, float] = {}
        for position, pid in enumerate(pool):
            fields = idx.field_values.get(pid) or frozenset()
            blob = idx.blob.get(pid, "")
            exact_hits = sum(1 for c in exact_constraints if c in fields)
            loose_hits = sum(1 for c in loose_constraints if c in blob)
            ev = EXACT_W * exact_hits + LOOSE_W * loose_hits
            evidence[pid] = ev
            score[pid] = (
                ev
                + POP_W * (idx.popularity.get(pid, 0.0) / pop_denom)
                - RANK_W * (position / depth)
            )
            if use_prior:
                cr = idx.rating.get(pid, 0.0)
                if cr > 0.0:
                    divergence = min(abs(cr - prior_rating) / 2.0, 1.0)
                    score[pid] -= PRIOR_RATING_W * divergence
            if use_position:
                fpos = idx.field_pos.get(pid)
                if fpos:
                    earliest: int | None = None
                    for c in constraints:
                        i = fpos.get(c)
                        if i is None:
                            i = next(
                                (j for k, j in fpos.items() if c in k), None
                            )
                        if i is not None and (earliest is None or i < earliest):
                            earliest = i
                    if earliest is not None:
                        score[pid] += POSITION_W / (1.0 + earliest)
            if use_velocity:
                score[pid] += VELOCITY_W * idx.velocity.get(pid, 0.0)

        best_ev = max(evidence.values(), default=0.0)
        state.last_rerank_tie_count = sum(
            1 for v in evidence.values() if abs(v - best_ev) <= 1e-9
        )
        state.last_fused_scores = score

        incoming = {pid: i for i, pid in enumerate(pool)}
        ranked = sorted(pool, key=lambda pid: (-score[pid], incoming[pid]))

        # Hard budget split, applied last as a stable sort: a known price that
        # violates a "<"/">" budget sorts after every satisfying / price-less
        # candidate, whatever its rerank score. No-op for "~" budgets and when
        # PRICE_HARD_SPLIT is off (_price_rank_map returns None).
        price_rank = self._price_rank_map(state, ranked)
        if price_rank is not None:
            ranked.sort(key=lambda pid: price_rank.get(pid, 0))

        return ranked

    def _price_rank_map(
        self, state: _SessionState, fused: list[str]
    ) -> dict[str, int] | None:
        """0 = satisfies the hard budget (or has no known price), 1 = a known
        price that VIOLATES it. None when there is no hard ("<"/">") budget in
        play. Comparisons are inclusive + PRICE_TOL: the disclosed amount is
        the target's own price, so a strict bound would demote the real
        target. README: "Structured price split"."""
        if not PRICE_HARD_SPLIT:
            return None
        op = state.budget_op
        amount = state.budget_amount
        if op not in ("<", ">") or not amount or amount <= 0:
            return None
        ranks: dict[str, int] = {}
        for pid in fused:
            price = self.index.price.get(pid)
            if price is None:  # absence is not a violation
                ranks[pid] = 0
                continue
            if op == "<":
                ok = price <= amount + PRICE_TOL
            else:
                ok = price >= amount - PRICE_TOL
            ranks[pid] = 0 if ok else 1
        return ranks

    def _window(self, state: _SessionState, fused: list[str], top_k: int) -> list[str]:
        """Normally return fused[:top_k]. Once the session has dead-ended
        (state.exhausted) and the list has been shown once unchanged, rotate
        deeper candidates through the tail slots so later turns keep probing
        instead of re-showing an identical miss. README: "Post-exhaustion
        rotation"."""
        if not state.exhausted:
            return fused[:top_k]

        if state.stable_top is None:
            # First turn with nothing left to reveal: lock the head, show the
            # list as-is one more time, start the rotation cursor past it.
            state.stable_top = fused[:top_k]
            state.rotation_cursor = max(top_k - ROTATE_WINDOW, 0)
            return list(state.stable_top)

        window = min(ROTATE_WINDOW, top_k)
        head = state.stable_top[: top_k - window]
        head_set = set(head)
        rotation_pool = [pid for pid in fused if pid not in head_set]
        if not rotation_pool:
            return fused[:top_k]
        if ROTATE_BY_POPULARITY:
            rotation_pool.sort(
                key=lambda pid: self.index.popularity.get(pid, 0.0), reverse=True
            )
        elif ROTATE_NOVELTY_FIRST:
            # Novelty-first: in a category-pure bucket where the disclosed
            # constraints do not separate the target from the pack, the real
            # purchase target tends to be a quirky, lightly-reviewed item that
            # the evidence+popularity fused order buries. Surfacing the
            # least-reviewed tied candidates first gives it earlier rotation
            # shots. (Descending popularity was tried and rejected -- README.)
            rotation_pool.sort(
                key=lambda pid: self.index.popularity.get(pid, 0.0)
            )

        cursor = state.rotation_cursor % len(rotation_pool)
        picks = rotation_pool[cursor : cursor + window]
        if len(picks) < window:  # wrapped past the end
            picks += rotation_pool[: window - len(picks)]
        state.rotation_cursor = (cursor + window) % len(rotation_pool)

        result = head + [pid for pid in picks if pid not in head_set]
        if len(result) < top_k:  # pad from anything not yet shown
            for pid in rotation_pool:
                if pid not in result:
                    result.append(pid)
                if len(result) >= top_k:
                    break
        return result[:top_k]

    def _compose_message(
        self, state: _SessionState, has_results: bool, ask_attribute: str | None
    ) -> str:
        if not has_results:
            return "I couldn't find a strong match yet -- tell me more and I'll keep looking."
        if ask_attribute:
            return f"Here are the closest matches so far -- do you have a {ask_attribute} preference?"
        if state.override_seen:
            return "Updating my search based on your new preference -- here's what I found."
        if state.active_constraints:
            return "Here are the closest matches based on what you've told me so far."
        if state.mode == "buying":
            return "Here's a close match for what you need -- let me know if I should refine it."
        return "Here are some options to start with -- let me know what matters most."
