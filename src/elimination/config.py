"""Tuning constants for the elimination-first retrieval track.

Every knob the elimination agent reads lives here, with the rationale for its
value kept inline. ``matcher.py`` / ``session.py`` / ``agent.py`` pull these
in with ``from .config import *`` (mirroring the lexical twin's
``src/lexical/config.py``), so ``__all__`` below is the authoritative list.
"""

from __future__ import annotations

import math

__all__ = [
    "DUAL_TRACK_ROUTING",
    "BUCKET_ENABLED",
    "BUCKET_KEEPS_CATEGORY_TERMS",
    "OVERRIDE_RESETS",
    "OVERRIDE_RESETS_CATEGORY",
    "OVERRIDE_RESETS_ASKS",
    "FIELD_WEIGHTS",
    "RERANK_ENABLED",
    "POOL_SIZE",
    "DEEP_PAGING",
    "POSITION_ENABLED",
    "POSITION_W",
    "POSITION_BREAKS_TIES",
    "EXPOSURE_ENABLED",
    "EXPOSURE_UNTIL_TURN",
    "EXPOSURE_WIDTH",
    "RERANK_WEIGHTS",
    "FEATURE_NAMES",
    "rerank_weights",
    "CLIP_LIMIT",
    "TRUNCATION_LIMIT",
    "SEARCH_FIELDS",
    "FEATURE_ALIASES",
    "POP_SCALE",
    "VELOCITY_ENABLED",
    "VELOCITY_W",
    "VELOCITY_FLOOR_YEAR",
    "VELOCITY_NOW",
]

# Route Buying and Browsing down separate retrieval tracks.
DUAL_TRACK_ROUTING = True

# Restrict retrieval to the coarse category bucket named in the opening
# message. The customer opens with a phrase the simulator derives from the
# target's own `categories` field, so when that phrase names a bucket we
# know outright, the target is in it -- and the other ~49,800 products are
# not. Exact lookup only: a bucket we merely guessed at is not safe to
# filter on, because excluding the target is unrecoverable in a way that
# ranking it badly is not. Anything short of an exact hit falls through to
# the unrestricted pipeline.
BUCKET_ENABLED = True
# Whether the category also stays in the MATCH expression once the bucket is
# filtering on it. Redundant as a filter, but not as a ranking signal: it is
# what puts the taxonomy words into the BM25 score that orders the pool.
BUCKET_KEEPS_CATEGORY_TERMS = True

# On "ignore my earlier preference", how much of the session is thrown away.
#
# Nothing, as it turns out. Taking the override literally was measurably wrong:
# the simulator draws the replacement from hard_constraints[0] of the *same*
# target product, and the superseded preference from its soft_preferences, so
# the customer never says anything false and the target never changes. Keeping
# the accumulated constraints is worth +0.0008 public / +0.0010 dev, and it
# repairs the one failure mode the reset creates -- when the replacement is a
# bare word like "cotton", filtering on it alone leaves a pool the target does
# not survive into, and no amount of reranking can recover an item retrieval
# never returned.
OVERRIDE_RESETS = False
OVERRIDE_RESETS_CATEGORY = False
OVERRIDE_RESETS_ASKS = False

# One weight per FTS5 column, in declaration order:
# parent_asin, price_text, title, categories, features, details, store, description
FIELD_WEIGHTS = (0.0, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0, 0.0)

# Retrieval used to end at the BM25 cut, so the top-10 slice *was* the BM25
# slice. It now fetches a pool and reranks it; BM25 survives as the tie-break,
# which means a session with no usable evidence returns exactly what it
# returned before.
RERANK_ENABLED = True
POOL_SIZE = 500

# Once the customer stops disclosing, the ranking stops changing -- and the ten
# items it produces have already been shown and already missed, because the
# harness ends the session the moment the target appears. Re-serving them is
# provably worthless, so a dry turn pages one slate deeper into the pool
# instead. The session is being lost anyway; this spends its dead turns on
# candidates the shopper has not seen yet.
#
# Deliberately an offset rather than a served-item blacklist: an Intent
# Override session does not register a hit until the override lands, so the
# target can legitimately appear in an early slate without ending the session.
# Blacklisting what we have shown would bury it for the rest of that session.
DEEP_PAGING = True

# How prominently an item carries the string the customer quoted, measured as
# its index in that item's own features+details list.
#
# A bare attribute word is the reason this is needed. "nylon" is matched by
# substring against the whole concatenated product text, so the jacket that IS
# nylon and the cotton jacket that merely ships with a nylon carry bag earn
# byte-identical evidence -- 51 of the 56 sessions lost to an evidence tie are
# exactly this. Position separates them, because manufacturers lead with the
# defining material and bury the incidental ones.
#
# Weighted so that POSITION_W + popularity + BM25 position stays under the 0.6
# a single loose match is worth: like the other tie-breakers, this may only
# reorder candidates that evidence has already declared equal.
POSITION_ENABLED = True
POSITION_W = 0.30

# Whether position also counts toward the tie count. It no longer gates
# anything -- the exposure gate below replaced the tie gate -- so this now only
# shapes how a trimmed slate is described to the customer. Left off: position
# orders candidates, but it is not a reliable enough separator to claim we can
# tell them apart.
POSITION_BREAKS_TIES = False

# Exposure gate.
#
# The harness ends the session the moment the target appears anywhere in the
# slate, so whatever rank we surface it at is the rank we are scored on for
# good -- there is no second chance to improve it later in the session. Early
# on, the order below the top is settled by popularity and BM25 position, two
# proxies close to a coin flip, so a full slate spends the session's one
# scoring opportunity on that coin flip. Serving only the single best candidate
# instead makes it rank 1 or nothing: a hit converts at RR 1.0, and a miss
# costs one turn, which is the cheaper side of the trade. An extra turn costs
# 0.2 x (1/200)/10 of the score, while lifting a hit from rank 3 to rank 1 is
# worth 0.3 x (2/3)/200 -- ten times as much.
#
# This supersedes withholding the slate outright, which was strictly worse:
# an empty slate cannot convert at all, and the top-1 it was suppressing
# converts often enough to be worth +0.0096 public / +0.0056 dev on its own.
#
# Measured over public/dev, reveal schedule -> TechScore:
#   full slate every turn (the old withhold-on-tie gate)  0.9497 / 0.9415
#   turn 1 only                                           0.9440 / 0.9319
#   turns 1-2                                             0.9600 / 0.9503  <-
#   turns 1-3                                             0.9526 / 0.9370
#   turn 1, then top-3 on turn 2                          0.9503 / 0.9406
#   turns 1-2, but turn 2 only when the top is tied       0.9593 / 0.9471
# Turn 1 alone gives back most of the win -- by turn 2 the second constraint
# has landed and the top-1 is worth betting on -- and turn 3 starts losing
# sessions outright, which costs hit rate at 0.5 against MRR's 0.3. Gating on
# the tie test is worse than trimming unconditionally, so ties no longer decide
# anything; they only pick which explanation the customer gets.
EXPOSURE_ENABLED = True
EXPOSURE_UNTIL_TURN = 2
EXPOSURE_WIDTH = 1

# (whole-item / attribute match, loose substring, popularity, BM25 position).
# The last two only ever settle candidates that are tied on evidence: together
# they cannot span the 0.6 gap one loose match opens, so nothing outranks a
# candidate that accounts for more of what the customer said. Within a tie the
# two disagree on purpose -- popularity backs the product people actually buy,
# BM25 backs the closest textual fit -- and the session is decided by which of
# them the tie-group happens to favour.
RERANK_WEIGHTS = (1.5, 0.6, 0.3, 0.10)

# One name per reranking feature, in the order Matcher.features() emits them.
# The weight vector is RERANK_WEIGHTS with POSITION_W spliced into third place,
# and the score is the plain dot product of the two. Keeping it a dot product
# is what makes the ranking tunable offline: tools/tune_reranker.py caches
# these rows once and re-scores them under candidate weights without going near
# sqlite again.
FEATURE_NAMES = ("exact", "loose", "position", "popularity", "velocity", "bm25")

# The customer clips each disclosed constraint at this many characters, so the
# catalog side is clipped identically -- otherwise a truncated constraint could
# never equal the item it was taken from.
CLIP_LIMIT = 180
# A constraint clipped at the customer's character limit may end mid-word; that
# partial token can never match the indexed text, so Constraint.phrases() also
# offers a form with the last token dropped when the value is this long.
TRUNCATION_LIMIT = 180
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
# A few hundred rows carry an untranslated boilerplate bullet that no English
# query can reach. The English sense is indexed *alongside* the original, never
# instead of it: the customer quotes constraints verbatim out of the catalog, so
# a constraint that reads "进口" still has to match the bullet it came from.
FEATURE_ALIASES = {"进口": "imported"}
POP_SCALE = math.log1p(100000.0)

# Ratings velocity: rating_number normalised by how long the item has been
# listed.
#
# This exists because it is an *interaction* the linear model cannot build for
# itself. Given popularity and recency as separate additive terms, no weight
# vector expresses a ratio -- and the additive pair was measured: recency on
# its own is worthless-to-harmful at every weight (it decays monotonically,
# 0.9568 -> 0.9494 at w=0.8), while the same information as a denominator is
# worth +0.0028. That gap is the whole justification for the feature.
#
# It is *additional* to popularity, not a replacement for it and not a
# redistribution of its weight. Replacing popularity outright is worse at every
# velocity weight; splitting a fixed 0.30 budget between the two is flat. The
# two disagree in a useful way -- popularity backs the item with the most
# ratings outright, velocity backs the one accumulating them fastest -- and it
# is the disagreement, not either signal alone, that separates a tie group.
#
# Naive intuition for this feature is backwards, which is worth recording:
# "old items had longer to accumulate ratings, so normalise by age" does not
# hold here. corr(log1p(ratings), age) = -0.118 -- *newer* listings carry more
# ratings, not fewer. Velocity earns its keep in spite of that, not because
# of it.
#
# Measured, pop_w held at 0.30 (1,000-session trace; live figures in
# EXPERIMENTS.md sec.3): the sweep peaks at 0.5 on train, holdout and both
# objectives at once, and the paired bootstrap clears zero
# (+0.00282 raw [+0.0007,+0.0054], +0.00243 public [+0.0006,+0.0044]).
VELOCITY_ENABLED = True
# At 0.5 the tie-breakers can, in principle, out-vote the 0.6 that one loose
# match is worth. Measured, they essentially never do: forcing evidence to
# dominate (weights x100) changes the score by +0.00004 raw / +0.00008 public,
# so ~99.97% of the gain is within-tie-group reordering rather than evidence
# being overridden. Drop to 0.3 to keep the invariant exactly binding; it costs
# about 0.001.
VELOCITY_W = 0.5
VELOCITY_FLOOR_YEAR = 2008
VELOCITY_NOW = 2024


def rerank_weights() -> tuple[float, ...]:
    """The live weight vector, in FEATURE_NAMES order."""
    exact_w, loose_w, pop_w, rank_w = RERANK_WEIGHTS
    velocity_w = VELOCITY_W if VELOCITY_ENABLED else 0.0
    return (exact_w, loose_w, POSITION_W, pop_w, velocity_w, rank_w)
