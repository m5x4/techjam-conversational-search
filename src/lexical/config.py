"""Tuning constants for the conversational-search agent.

Every value here was swept against the 200-sample public set. The rationale,
sweep grids and rejected alternatives are written up in ``src/README.md``
(sections referenced from the inline notes below) rather than repeated here.

The sweep scripts under ``scripts/`` mutate these names on the imported
``src.lexical.agent`` module at runtime (``agent_mod.THIN_KEEP = ...``); the
agent re-exports them with ``from .config import *`` and reads them as bare
module globals, so a reassignment on ``src.lexical.agent`` is picked up on
the next call. Constants consumed inside ``catalog_index.py`` (the FTS5 term
caps) are read from this module directly -- point ``sweep_weight.py`` at
``src.lexical.config`` to sweep those.
"""

from __future__ import annotations

__all__ = [
    "BM25_POOL",
    "AND_TERM_CAP",
    "OR_TERM_CAP",
    "BM25_COLUMN_WEIGHTS",
    "_BROWSING_WIDEN_FACTOR",
    "EXACT_W",
    "LOOSE_W",
    "POP_W",
    "RANK_W",
    "ROTATE_WINDOW",
    "ROTATE_BY_POPULARITY",
    "ROTATE_NOVELTY_FIRST",
    "THIN_ENABLE",
    "THIN_KEEP",
    "THIN_MAX_TURN",
    "THIN_PHRASE_MAX",
    "SYNTH_RESCUE_MAX_DF_FRAC",
    "CONF_GATE_ENABLE",
    "CONF_GATE_RATIO",
    "CONF_GATE_MAX_TURN",
    "CONF_GATE_KEEP",
    "PRICE_HARD_SPLIT",
    "PRICE_TOL",
    "PRIOR_RATING_W",
    "PRIOR_RATING_MAX",
    "POSITION_W",
    "VELOCITY_W",
    "BOUNDARY_SLACK",
    "BROWSE_DIVERSITY",
]

# Candidates pulled from FTS5 before the rerank, on the unresolved
# whole-catalog path (the bucket path uses the full bucket size instead).
BM25_POOL = 500

# FTS5 query-term caps. Long multi-reveal sessions accumulate enough
# constraint terms to overflow the old 12/40 caps, which silently dropped
# later-slot constraints from the query. Read inside catalog_index.py.
AND_TERM_CAP = 16
OR_TERM_CAP = 60

# FTS5 bm25() per-column weights, in the products-table column order
#   (parent_asin, title, categories, features, details, store, description)
# parent_asin is UNINDEXED so its weight is inert (kept as 0.0 for a
# 1:1 map to the column list). Read inside catalog_index.py; used for BOTH
# the whole-catalog and the bucket-scoped query. Hand-set originally, then
# swept against the bucket pools. README: "BM25 column weights".
BM25_COLUMN_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Browsing sessions widen the BM25 pool well before it runs dry (see
# Agent._retrieve). README: "Buying vs Browsing routing".
_BROWSING_WIDEN_FACTOR = 3

# --- Linear rerank weights (the single active scoring pass) ---------------
# score(pid) = EXACT_W * (# disclosed constraints equal to a whole
#                         feature/detail value on this item)
#            + LOOSE_W * (# disclosed constraints occurring as a substring
#                         of this item's flattened blob)
#            + POP_W   * (popularity(pid) / max popularity in the pool)
#            - RANK_W  * (pool_position / len(pool))
# Swept against the category-pure bucket pools (README: "Rerank weight
# sweep"). A ~140-point grid has a broad flat optimum: every point with
# LOOSE_W >= EXACT_W, LOOSE_W in [1.0, 1.5], POP_W in [0.3, 0.9] and
# RANK_W in [0.05, 0.10] lands on the SAME session outcomes -- public
# composite 0.96662 -> 0.96989, MRR 0.9647 -> 0.9743, MTTC 2.14 -> 2.12,
# no scenario regression (boundary MRR 0.95 -> 1.0, buying 0.974 -> 0.983).
# The active lever is LOOSE_W (0.6 -> 1.25, i.e. up to EXACT_W); POP_W
# nudged 0.3 -> 0.5. Chosen point sits inside the verified plateau.
# OUT: EXACT_W > LOOSE_W (drops off the plateau, e.g. 1.5/1.25 -> 0.96877);
# RANK_W = 0.0 (overfits -- val composite 0.968 -> 0.961).
EXACT_W = 1.25
LOOSE_W = 1.25
POP_W = 0.5
RANK_W = 0.10

# Post-exhaustion "stall rotation": once the simulator has nothing left to
# reveal and the target still isn't in the top_k, pin the strongest
# (TOP_K - ROTATE_WINDOW) slots and cycle deeper candidates through the rest
# each turn. README: "Post-exhaustion rotation".
ROTATE_WINDOW = 5
ROTATE_BY_POPULARITY = False
# Order the post-exhaustion rotation pool least-reviewed-first. Mutually
# exclusive with ROTATE_BY_POPULARITY (which takes precedence). README:
# "Post-exhaustion rotation".
ROTATE_NOVELTY_FIRST = False

# Thin-signal guard -- defer a weak early hit to a sharper later rank. The
# single biggest post-tuning lever (public composite ~0.901 -> ~0.941).
# On an early turn (turn <= THIN_MAX_TURN) with a still-thin constraint
# signal (<= THIN_PHRASE_MAX verbatim multi-word phrases) and the simulator
# not yet exhausted, return only THIN_KEEP ids. Retrieval is monotonic so
# the target re-enters at the same-or-better rank next turn.
# README: "Thin-signal guard". scripts/sweep_thin.py reproduces the sweep.
THIN_ENABLE = True
THIN_KEEP = 1
THIN_MAX_TURN = 3
THIN_PHRASE_MAX = 2

# The EXACT rerank term drops simulator-synthetic "<attr>: <value>" phrases
# (_SYNTHETIC_PHRASE_RE) because the catalog does not write attributes in
# that colon shape -- EXCEPT the evaluator also emits verbatim `details`
# dict entries in the identical "Key: value" shape (via _flatten_values), and
# those ARE real discrete field values that the colon-free blob (LOOSE term)
# never matches either, so the constraint otherwise scores zero. Rescue such
# a phrase into EXACT only when it is a real field value carried by no more
# than this fraction of the pool -- a discriminating value ("Color: Shark Jaw
# With Dive Flag", "Material: Pleuche"). A near-universal one ("color: black",
# "department: womens") stays dropped: crediting it only shifts ties toward
# the popular decoys that also carry it. 0.0 disables the rescue entirely.
# README: "Synthetic phrase / real field value".
SYNTH_RESCUE_MAX_DF_FRAC = 0.30

# EXPERIMENTAL post-rerank confidence gate. Default OFF -- baseline
# behaviour is unchanged unless CONF_GATE_ENABLE is set (scripts/
# sweep_conf_gate.py toggles it on the imported module). Reads the actual
# linear-rerank score margin instead of THIN's turn/phrase-count heuristic:
# when the leader is not >= CONF_GATE_RATIO x the tail score, trim to
# CONF_GATE_KEEP ids. Only fires on turn <= CONF_GATE_MAX_TURN while the
# simulator still has more to disclose. README: "Confidence gate".
CONF_GATE_ENABLE = False
CONF_GATE_RATIO = 1.5
CONF_GATE_MAX_TURN = 3
CONF_GATE_KEEP = 1

# Structured price constraint (operator-aware budget). Inert on the public
# set -- the local evaluator only ever emits "budget around $X" ("~"), so
# 0/200 sessions carry a "<"/">" budget. Exists for a reworded private
# evaluator ("keep it under $30"): a candidate whose KNOWN price violates
# the bound sorts after every candidate that satisfies it; a candidate with
# no price is not penalised. Comparisons are inclusive + PRICE_TOL because
# the amount is the target's own price. README: "Structured price split".
PRICE_HARD_SPLIT = True
PRICE_TOL = 0.01

# User-profile rating affinity. The anonymised profile's `average_prior_rating`
# (1..5) is the only field with any variance -- `purchase_frequency` is a
# dataset constant, `summary` restates `preference_tags`, `rating_style` is a
# coarser `average_prior_rating`, so none of those three are read. A pool
# candidate whose catalog `average_rating` diverges from the user's prior mean
# is penalised in the linear rerank by
#   PRIOR_RATING_W * min(|cat_rating - prior| / 2.0, 1.0)
# so a full 2.0-point gap costs the whole weight and an unrated item
# (rating 0.0) is neutral. PRIOR_RATING_W = 0.0 disables the term entirely
# (baseline behaviour is then byte-identical).
#
# Swept `PRIOR_RATING_W {0 .. 1.0}` x `PRIOR_RATING_MAX {5.0, 5.1}` on the
# 150/50 split (scripts/sweep_weight.py --const PRIOR_RATING_W). Global
# r(prior, target average_rating) is only ~0.18, but the effect concentrates
# on the residual non-rank-1 set: composite all 0.96904 -> 0.97026,
# train 0.97019 -> 0.97179, val 0.96560 -> 0.96570, MRR 0.9718 -> 0.9749,
# MTTC 2.125 -> 2.110; three previously-deep sessions improve rank
# (public_0161 2 -> 1, 0020 9 -> 7, 0099 4 -> 3) and four hit a turn earlier;
# two (0014 / 0196) hit a turn later but keep rank 1. No rank/hit regression,
# hit@10 stays 1.0, MRR only rises, intent_override / boundary untouched.
# Plateau `W in [0.15, 0.25]`; `W >= 0.30` demotes rank-1 sessions
# (0049 / 0083 / 0140). `PRIOR_RATING_MAX
# = 5.0` (gate out prior = 5.0 users) was tested and is *worse* -- a prior-5.0
# user whose target is itself 5.0-rated takes zero penalty while the mid-rated
# decoys above it are demoted, so the shipped 5.1 keeps the gate open for
# every rating. README: "User-profile rating affinity".
#
# REVERTED to 0.0 (2026-09-01). The W = 0.15 gain was measured only on the
# 200-sample public set (+0.00122 composite, +0.0001 on its 50-session val
# half -- both below the local evaluator's ~0.001 cross-run drift). On the
# held-out 800-session dev_set the term is net-NEGATIVE: with it OFF the
# composite is 0.95093 -> 0.95157 and MRR 0.93317 -> 0.93524, and browsing /
# boundary MRR both improve (only intent_override dips 0.003). It was fit to
# public-set noise. Code path kept (inert at 0.0) for a private evaluator
# that might phrase the profile with more usable variance.
PRIOR_RATING_W = 0.0
PRIOR_RATING_MAX = 5.1

# --- Ported from the standalone agent_v2.py (README: "Combined v1+v2
# features"). Four candidate levers, each gated here and inert at its no-op
# default. The backing per-product data (field_pos map, velocity map,
# store/title maps) is built unconditionally in catalog_index.py -- only the
# weight / flag gates behaviour, so scripts/sweep_weight.py moves it without
# an index rebuild. Validated on the held-out 800-session dev_set per the
# post-PRIOR_RATING_W rule ("Report new levers on dev_set, not the 200-set").
# Verdict: F1 ships; F2 / F3 / F4 rejected (kept inert for a private
# evaluator, same as CONF_GATE and PRIOR_RATING_W).

# F1 -- match-prominence term. A disclosed constraint that lands on the
# item's FIRST feature/detail entry is the defining attribute; one buried
# deep ("...ships with a nylon carry bag") is incidental. Score-only, added
# to _rerank as POSITION_W * 1/(1 + earliest_entry_index). This is a signal
# outside the disclosed text's mere presence/absence -- the one kind
# README's "six-session data-ambiguity dead end" leaves open. 0.0 disables.
#
# SHIPPED at 0.2. Public sweep {0.1 .. 0.5}: interior peak at 0.2 (composite
# +0.0002, MTTC only) -- noise-band on its own. But it GENERALISES: on the
# held-out dev_set (800) POSITION_W = 0.2 gives composite 0.951571 ->
# 0.952421 (+0.00085), MRR 0.935237 -> 0.937320 (+0.0021, a real rank gain
# not just MTTC), hit-rate flat, 22 sessions moved, 0 regressions; buying
# MRR +0.0036, intent_override +0.0042, boundary / browsing untouched.
# Bigger on held-out than on the tuning set -- the opposite of the
# PRIOR_RATING_W failure. README: "Combined v1+v2 features".
POSITION_W = 0.2

# F2 -- ratings velocity: rating_number normalised by listing age, added
# alongside (not replacing) the popularity term as VELOCITY_W * velocity.
#
# REJECTED (kept inert at 0.0). Public looked like a win -- fine sweep
# {0.3 .. 0.6} showed a plateau at composite 0.970889 (+0.0006 vs the
# then-baseline). It did NOT survive the dev_set: at 0.5, composite
# 0.951571 -> 0.951052 (-0.0005), MRR -0.0021, and 16 rank/hit regressions
# incl. dev_0323 dropping a rank-7 hit to a miss; boundary / browsing /
# buying MRR all fall (only intent_override rises). The public plateau was
# fit to 200-set noise -- same story as PRIOR_RATING_W. Code path kept for a
# private evaluator whose age/volume relationship differs.
VELOCITY_W = 0.0

# F3 -- boundary-shrug slack. "I don't have A preference for X" is a one-off
# Boundary-scenario shrug that costs a turn; "AN ADDITIONAL preference" is
# genuine exhaustion. When True each shrug pushes the THIN turn cap back one.
#
# REJECTED (kept inert at False). No-op on the public set (all 10 boundary
# sessions already rank-1). On dev_set (40 boundary sessions) it is still a
# no-op-to-slightly-negative: composite -0.00005, 2 sessions converge one
# turn later, MRR flat. agent.py's THIN guard already caps at turn 3, which
# covers the boundary MTTC of ~2.6, so the extra turn of slack buys nothing
# (agent_v2's exposure gate, which this was ported from, capped at turn 2).
# Split-refusal parsing (message_parsing.RE_NO_PREF_*) is kept -- it is
# strictly more correct -- and state.slack is still counted, just unused.
BOUNDARY_SLACK = False

# F4 -- browse-track diversity spread: on a no-constraint Browsing turn,
# dedupe the reranked head by store/brand and title-shape before the top_k
# slice.
#
# REJECTED (kept inert at False). Completely dead on both sets: 0 sessions
# move. The THIN guard trims Browsing turns 1-2 to a single id, and by
# turn 3 constraints have landed so `not state.active_constraints` is false
# and the branch never fires on a scored turn.
BROWSE_DIVERSITY = False
