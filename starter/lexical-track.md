# Lexical retrieval track

Design rationale and full tuning history for the **lexical** track
(`starter/lexical/`) and the scenario router (`starter/agent.py`). Its sibling,
`starter/elimination-track.md`, covers the **elimination** track. Top-level
overview: `../README.md`.

Multi-turn shopping agent for the TechJam conversational-search task. Exports
`Agent` (`reset` / `respond`) from `agent.py`. Pure Python standard library +
SQLite FTS5 — **no network access, no model download** at scoring time.

## Architecture: scenario router (`agent.py`)

`agent.py` is a thin **router** over two independent implementations. The
customer simulator's opening message fixes the scenario for the whole session,
so the route is decided once, from turn 1, and never switched:

| opener shape | scenario | route |
|---|---|---|
| `… A key requirement is: <c>.` | Buying | `elimination` |
| `I'm looking for <cat>. <old_value>` (real attribute tail) | Intent-Override | `lexical` |
| `…, but I'm still exploring.` | Browsing / Boundary | `elimination` |
| anything else (reworded) | — | `elimination` (safe default) |

Why: on the held-out `dev_set` (800), `elimination` wins Buying (MRR 0.950 vs
0.934), Browsing (0.954 vs 0.917) and is the better generaliser overall
(composite 0.9592 vs 0.9518), while `lexical` wins Intent-Override
(0.979 vs 0.953) on *both* the public and held-out sets. Boundary is a wash and
shares a byte-identical opener with Browsing, so it rides with `elimination`.
(See `../../.claude/plans/since-agent-is-good-peppy-journal.md` and
`memory/techjam-score-improvement-wip.md`.)

Composite = `0.5·hit_rate@10 + 0.3·MRR + 0.2·efficiency`,
`efficiency = clip((11 − MTTC)/10, 0, 1)`.

`python -m evaluator.local_evaluator` scores the router. Composites:

| | public_set (200) | dev_set (800, held out) |
|---|---|---|
| `lexical` standalone | 0.96914 | 0.95242 |
| `elimination` standalone | 0.96695 | 0.95921 |
| **router (shipped)** | **0.96758** | **0.95968** |

The router is `elimination` on Buying / Browsing / Boundary — per-scenario
MRR / MTTC **byte-identical** to `elimination` standalone on both sets — plus
`lexical` on Intent-Override, where it lifts MRR from `elimination`'s 0.953
to 0.970 (dev). Net vs the stronger standalone (`elimination`): +0.0006 public,
+0.0005 dev — small but positive on both independent sets, with 100 %
route-correctness and zero regression on the other three scenarios. Route is
decided once from the opener (`starter/agent.py::_is_intent_override`).

---

## Module map

| module | responsibility |
|---|---|
| `agent.py` | **`Agent` — the scenario router.** Classifies the opener once, delegates every turn to the `lexical/` or `elimination/` package. **Submission entry point.** |

**`lexical/` package** — bucket pre-filter → two-tier BM25 → linear rerank → window. Handles Intent-Override sessions.

| module | responsibility |
|---|---|
| `lexical/agent.py` | The lexical `Agent` (was top-level `agent.py`, then `agent_lexical.py`): retrieve → rerank → window, THIN / CONF gates. |
| `lexical/config.py` | Every swept tuning constant for the lexical track, one place. Re-exported by `lexical/agent.py` so the sweep scripts can mutate them on the imported module. |
| `lexical/text_utils.py` | Tokenisation, stopwords, FTS5 phrase quoting, the two folding functions the rerank needs (`_field_value_strings`, `_norm`). |
| `lexical/budget.py` | Disclosed budget string → `(operator, amount)`. |
| `lexical/message_parsing.py` | Customer-simulator message templates (regexes) + constraint→attribute classification. The router reuses `RE_KEY_REQUIREMENT` / `RE_OPENER_TAIL` / `RE_OPENER_FILLER` for opener classification. |
| `lexical/session.py` | `_SessionState` — the per-session dialogue state machine + the `ask_attribute` policy. |
| `lexical/buckets.py` | `coarse_category` + `BucketIndex` — the pre-retrieval coarse-category filter. |
| `lexical/catalog_index.py` | `_CatalogIndex` — BM25 (FTS5) index, flattened blob / discrete field-value / price / popularity maps, the bucket map. |

**`elimination/` package** — elimination-first: hard FTS phrase filtering, always-on diversified browse, dry-turn deep paging. Self-contained (stdlib only). Handles Buying / Browsing / Boundary. Mirrors the lexical layout:

| module | responsibility |
|---|---|
| `elimination/agent.py` | The elimination `Agent` (reset / respond) + the customer-facing prompt strings. |
| `elimination/config.py` | Every elimination-track tuning constant, rationale inline; consumed via `from .config import *`. |
| `elimination/text.py` | Folding / tokenisation (`normalize`, `fold`, `tokens`, `coarse_category`, …). Not shared with `lexical/text_utils.py` — this track folds with `casefold` + a 180-char clip. |
| `elimination/parsing.py` | Simulator message templates → `Turn` (`parse_message`). |
| `elimination/matcher.py` | `Constraint` + `Matcher`: the FTS5 index, phrase filter, rerank, and browse track. |
| `elimination/session.py` | `SessionState` — accumulated dialogue state (never cleared). |

`lexical/` data flow for one `respond()` call:

```
user_message
  └─ _SessionState.absorb        parse templates → fill attribute slots / category / budget / exhausted
  └─ Agent._retrieve
       ├─ _resolve_bucket        opener → coarse-category bucket id list (once per session, cached)
       ├─ bm25_search[_scoped]   two-tier FTS5 (precise AND + broad OR), scoped to the bucket if resolved
       ├─ _rerank                single linear score over the pool  (see "Linear rerank")
       └─ _window                normally pool[:top_k]; post-exhaustion tail rotation
  └─ THIN guard / CONF gate      maybe trim the slate to defer a weak early hit
  └─ _choose_ask_attribute       "other" until exhausted, then null
  └─ _compose_message
```

The `elimination` track runs its own elimination-first pipeline (`parse_message` →
`Matcher.search` / `Matcher.browse` → `Matcher.rerank` → exposure gate); see the
package docstring in `elimination/__init__.py`.

---

## Held-out validation & overfitting audit (2026-09-01)

All `lexical` tuning turned on the 200-sample public set;
`dev_set.jsonl` (800) / `dev_full.jsonl` (49,800) are materialised by the same
evaluator and were never tuned against. `lexical` on `dev_set` shows a
~0.018 composite drop vs public — genuine distribution shift (`dev_set` carries
an explicit easy/medium/hard difficulty mix), not a tuning artefact. Ablations
on `dev_set` confirm THIN (+0.071 composite there), the rerank weight plateau,
and the opener trailing-clause capture all generalise. The one lever that did
**not**: `PRIOR_RATING_W`, reverted to 0.0 (see "User-profile rating affinity").
`elimination` generalises better on Buying/Browsing, which is why the router exists.
**Report new levers on `dev_set` / a `dev_full` sample, not the 200-set** — the
150/50 public split has been selected against too many times to still measure
generalisation.

---

## Design decisions & tuning history

All numbers below are against the 200-sample public set unless noted. Sweep
scripts that reproduce them live in `../scripts/`.

### Bucket pre-filter

The customer simulator always opens with
`"I'm looking for {coarse_category(target.categories)} ..."`, where
`coarse_category` is the target's last two comma-split `categories` segments
with the store-wide `"Clothing, Shoes & Jewelry"` stripped. That line names,
**verbatim on the public set, a catalog bucket guaranteed to contain the
target** (median bucket ~184 items vs the 50k catalog). All 200 public
sessions resolve EXACT to the correct bucket.

`buckets.coarse_category` mirrors the evaluator's copy byte-for-byte — if the
evaluator's changes, change this to match. `BucketIndex.resolve` degrades
gracefully: **exact → containment → token-overlap (≥ 0.34 overlap
coefficient) → None**. On `None`, `_retrieve` falls back to the unchanged
whole-catalog BM25 pipeline (the private-set safety net for a reworded
opener).

On resolve, the pool is built by `_CatalogIndex.bm25_search_scoped` — the
two-tier FTS5 query run against a joined `_scope` temp table with
`limit = len(bucket_ids)` — not a 500-cap whole-catalog search filtered down
afterwards. Every bucket member that shares a query term is ranked by its
real in-bucket BM25 score; the popularity tail then only ever orders
zero-match members (near-always none, since the broad OR query carries the
category terms every member shares). This also gives a hard recall floor:
the target can no longer fall out of a 500-cap pool on a bad term match
(that was the one public hit-rate miss, `public_0020`).

Score progression: hit_rate `0.995 → 1.0`, MRR `0.9215 → 0.9262`, composite
`0.9408 → 0.9450` (adding the filter), then `→ 0.9449` (scoped-BM25 variant —
byte-identical hit/MRR, one cold-start turn-convergence flip).

**Caveat (mostly closed):** every constant was originally swept against the
500-item whole-catalog pool. The bucket pool is smaller and category-pure, so
the optima could have shifted. Re-swept against bucket pools: the rerank
weights **did** move (see "Rerank weight sweep", composite `0.9667 → 0.9699`);
`THIN_*`, `CONF_GATE` and the BM25 column weights were all re-swept and held
at their existing values (see "Rejected approaches").

### Linear rerank

The single scoring pass (`Agent._rerank`) that **replaced** an earlier
multi-signal RRF fuse + a separate verbatim-phrase-count promotion pass. For
each candidate `pid` in the pool:

```
score(pid) = EXACT_W · (# disclosed constraints equal to one whole
                        feature/detail value on this item)          EXACT_W = 1.25
           + LOOSE_W · (# disclosed constraints occurring as a
                        substring of this item's flattened blob)    LOOSE_W = 1.25
           + POP_W   · (popularity(pid) / max popularity in pool)   POP_W   = 0.5
           − RANK_W  · (pool_position / len(pool))                  RANK_W  = 0.10
```

* "disclosed constraints" = `state.active_constraints`, `_norm`-folded
  (whitespace-collapsed + lower-cased) and de-duplicated.
* `field_values` (the EXACT target) is built from **discrete**
  `features`/`details` entries via `_field_value_strings` — kept separate,
  not flattened, because the simulator quotes one whole feature bullet /
  detail value verbatim, so the boundary between entries is itself signal.
* Simulator-synthetic `"<attr>: <value>"` / `"budget around $..."` phrases
  (`_SYNTHETIC_PHRASE_RE`) are dropped from EXACT (not verbatim catalog
  text) but left in LOOSE (their colon shape practically never occurs in
  catalog text, so it's a no-op that is strictly more correct on the rare
  turn one does line up).
* The incoming BM25/bucket order is folded in as the small `RANK_W` penalty
  and is also the tiebreak on an exact score tie.
* `state.last_rerank_tie_count` records how many candidates tie the best
  EVIDENCE sub-score (EXACT + LOOSE only). Computed and stored, **not wired
  to any behaviour** — left for a possible future withhold-on-tie feature.
* A known hard-budget (`<` / `>`) price violation still demotes a candidate
  regardless of score: `_price_rank_map` is applied last as a stable sort.

**Why RRF was wrong here:** there is one BM25 index and one pool — the six
"fused" lists were all re-sortings of the same candidate set, not
independent retrievers. RRF's `1/(k+rank)` also compresses magnitude
("3 phrases matched" vs "1" ≈ `1/61` vs `1/63`), which is exactly why the
old pipeline needed a hard phrase-count override bolted on top. A linear
weighted sum keeps the magnitude and needs no override.

Result vs the old RRF + phrase-priority pipeline: hit `1.0` (unchanged),
MRR `0.9312 → 0.9647`, MTTC `2.625 → 2.140`, composite `0.9469 → 0.9667`.

### Rerank weight sweep  (`config.EXACT_W / LOOSE_W / POP_W / RANK_W`)

The four weights above shipped verbatim from the reference implementation and
were swept — a ~140-point grid on a 150/50 train/val split — only after the
bucket pre-filter changed the pool from "500-cap whole catalog" to
"whole category-pure bucket".

The optimum is a **broad flat plateau**: every point with `LOOSE_W ≥ EXACT_W`,
`LOOSE_W ∈ [1.0, 1.5]`, `POP_W ∈ [0.3, 0.9]`, `RANK_W ∈ [0.05, 0.10]` produces
the **same 200 session outcomes**. The one active lever is `LOOSE_W`
(`0.6 → 1.25`, i.e. raised to parity with `EXACT_W`) with `POP_W` nudged
`0.3 → 0.5`; `EXACT_W` and `RANK_W` keep their reference values. Chosen point
`(1.25, 1.25, 0.5, 0.10)` sits in the interior of the plateau.

Result: hit `1.0` (unchanged), MRR `0.9647 → 0.9743`, MTTC `2.140 → 2.120`,
composite **`0.9667 → 0.9699`**. Three previously-deep sessions
(`public_0035 / 0049 / 0163`, rank 2–4) move to rank 1; none regress.
Scenario MRR: boundary `0.95 → 1.0`, buying `0.974 → 0.983`, browsing
`0.953 → 0.963`, `intent_override` flat.

Two edges were rejected. `EXACT_W > LOOSE_W` falls off the plateau
(`1.5 / 1.25 → 0.96877`). `RANK_W = 0.0` scores highest on *train* (`0.9707`)
but overfits — one held-out val session drops from rank 1, val composite
`0.968 → 0.961`; `RANK_W ≥ 0.05` is stable both ways.

### Thin-signal guard  (`config.THIN_*`, `scripts/sweep_thin.py`)

The single biggest post-tuning lever found: composite `0.90103 → 0.94085`,
MRR `0.767 → 0.921`, on the **same** hit set — no new miss, only earlier
turns' weak hits deferred.

On an early turn (`turn ≤ THIN_MAX_TURN = 3`) where the constraint signal is
still thin (`≤ THIN_PHRASE_MAX = 2` verbatim multi-word phrase constraints)
and the simulator still has more to disclose (`not state.exhausted`), a
target sitting at rank 4..10 is nearly always a popularity-prior coincidence
that the next "other" reveal lifts to rank 1..3 — not a genuine best-case
hit. Returning only `THIN_KEEP = 1` id defers that weak hit one turn.
Retrieval is monotonic (constraints only accumulate into their slots), so
the target re-enters at the same-or-better rank next turn; worst case is
~1 turn of MTTC on a session whose deep hit really was as good as it gets
(MTTC `2.325 → 2.655`, a ~0.007 efficiency cost against a ~0.046 MRR gain).

Swept `THIN_KEEP {1,2,3,4}` × `THIN_MAX_TURN {1..6}` × `THIN_PHRASE_MAX
{0..6}`: monotonic in every knob toward `(1, 3, 2)`, then flat (past turn 3
every remaining session is already hit or exhausted; > 2 phrase constraints
by turn 3 is rare). `THIN_KEEP = 1` beat 2 clearly (`0.9409` vs `0.9291`)
with no extra miss. Supersedes an earlier turn-1-buying-only version.

### Confidence gate  (`config.CONF_GATE_*`, default OFF, `scripts/sweep_conf_gate.py`)

Experimental alternative trigger for the same "defer a weak early hit"
action: instead of THIN's turn/phrase-count heuristic, read the actual
linear-rerank score margin — top-1 score vs the score at rank
`min(top_k, len)`. When the leader is not at least `CONF_GATE_RATIO ×` the
tail score the ranking is "not confident", so trim to `CONF_GATE_KEEP` ids.
Only fires on `turn ≤ CONF_GATE_MAX_TURN` while the simulator still has more
to disclose, so it can never convert a hit into a miss.

Left OFF: an earlier confidence-gated thin variant, instrumented against the
public set, fired on 68 turns but every one already had the target inside
the kept head — it never changed a scored outcome at any threshold in a
`{1.3..3.0} × keep {2,3,4}` sweep. The productive trigger is the opposite:
trim while the signal is still *thin* and a reveal is still coming (the THIN
guard).

### Buying vs Browsing routing

The turn-1 message shape fixes `state.mode` once for the whole session. A
**Buying** opener carries `"A key requirement is: ..."` — the customer
already handed us a hard constraint, so a thin-but-precise BM25 pool is
trusted as-is. Everything else (a vague opener, or the Intent-Override
scenario's bare old preference) is **Browsing**: no committed constraint
yet, so on the unresolved whole-catalog path the pool is widened
proactively — `top_k × _BROWSING_WIDEN_FACTOR` (= 3) rather than waiting
until it nearly runs dry — to give the rerank more candidates to
differentiate. `ask_attribute` strategy is unaffected by mode. (On the
bucket-resolved path the pool is already the whole bucket, so widening is
moot.)

### Cold start

A cold-start turn (no slot filled, no multi-word phrase disclosed — almost
always turn 1) used to double-count: the old fuse's passthrough lists
returned the raw pool unchanged and then fused those passthrough copies,
stacking ~2.6× weight on raw BM25 order. Dropping the passthrough
double-count cut cold-start mean rank-at-hit `5.9 → 2.2` while holding
overall MRR. The linear rerank has no passthrough lists, so this is
structurally gone; popularity contributes only its normal `POP_W` term.

### Post-exhaustion rotation  (`config.ROTATE_*`)

Once the simulator has nothing left to reveal and the target still isn't in
the top-`k`, the slate would otherwise sit frozen for the rest of the
session. Instead `_window` pins the strongest `top_k − ROTATE_WINDOW` (= 5)
slots and cycles deeper candidates through the remaining slots each turn —
extra shots on goal at zero risk (a target that belonged in the pinned head
would already have hit and ended the session).

`ROTATE_BY_POPULARITY = False`: ordering the rotation pool by the popularity
prior instead of raw fused depth turned 3 depth-rotation recoveries
(`public_0076/0087/0144`, rank 7–9 on turn 6–7) into misses — popularity
surfaces heavily-reviewed non-targets and buries the less-reviewed real
target the depth rotation was already cycling toward.

### Structured price split  (`config.PRICE_HARD_SPLIT`)

Operator-aware budget. `budget.py` parses a disclosed budget string to
`(op, amount)` with `op ∈ {"<", ">", "~"}`. The local evaluator **only ever
emits `"budget around $X"` (→ `"~"`)**, so `_price_rank_map` /
`PRICE_HARD_SPLIT` is **inert on the public set — 0/200 sessions carry a
`<`/`>` budget**, and metrics are byte-identical with it on or off (verified).

It exists purely for a reworded private evaluator that phrases a budget as
"keep it under $30" / "at least $50": there, a candidate whose **known**
price violates the bound sorts after every candidate that satisfies it. A
candidate with no price is **not** penalised (absence ≠ violation).
Comparisons are inclusive + `PRICE_TOL`, because the amount is the target's
own price and a strict `<` would demote the real target. `"~"` budgets keep
no special handling (the old soft proximity re-rank was dropped with the
RRF fuse).

### FTS5 term caps  (`config.AND_TERM_CAP` / `OR_TERM_CAP`)

Long multi-reveal sessions (mostly `intent_override`) accumulate enough
constraint terms to overflow the old `12` / `40` caps, which **silently
dropped the later-slot constraints from the query**. Raised to `16` / `60`.

---

### Multilingual constraint retrieval  (`text_utils._segment` / `catalog_index._seg_index`)

The catalog carries non-English text in `features` bullets, `details`
values, and `description` (CJK / Cyrillic / Greek / Arabic / Hebrew /
Hangul / Kana; ~0.2–0.9% of products per field; `categories` and `details`
keys are 100% ASCII). The customer simulator splices constraint phrases
**verbatim** out of the target's own `features` / `details` text into an
English template (`"A key requirement is: {c}."` etc.), so a non-English
value does reach `absorb()` and is parsed into an attribute slot.

The gap was in retrieval, not scoring. `_terms()` was
`TOKEN_RE = [a-z0-9]+` — it returned `[]` for a CJK/other-script value and a
truncated fragment for accented Latin (`"café"` → `"caf"`), so
`query_terms()` produced no BM25 term for that constraint and
`bm25_search[_scoped]` could not pull the target into the pool. (The rerank
already handled it: `_norm` is a plain `.lower()`, Unicode-safe, and matches
the raw `blob` / `field_values`.)

Fix — retrieval only:

* **`text_utils._segment(text)`** — NFD-fold + drop combining marks
  (`"café"` → `"cafe"`, lining up with the index's `remove_diacritics 2`),
  then split each run: ASCII stays as `[a-z0-9]+` tokens; a CJK / Kana /
  Hangul run becomes overlapping character bigrams (`"舒适透气"` →
  `舒适 适透 透气`); a space-delimited non-Latin run (Cyrillic, …) passes
  through whole for `unicode61` to tokenise. NFD (not NFKD) is deliberate —
  NFKD also rewrites `™`→`tm`, `½`→`1/2` etc. and would perturb near-ASCII
  English rows.
* **`_terms()`** gets a `str.isascii()` fast-path returning the **exact**
  old comprehension; only non-ASCII input reaches `_segment`.
* **`catalog_index._seg_index()`** applies the same segmentation to the six
  FTS5 column strings at index time (so bigram query terms have bigram
  index tokens to match), again guarded by `str.isascii()`.

Non-regression is structural: an all-ASCII session never enters the new
path and an all-ASCII column is inserted unchanged. Verified — `_terms`
output is byte-identical to the old function across **all 50,000** catalog
rows; the public-set run is unchanged (composite **0.969042**, hit_rate
1.0, MRR / MTTC identical to 6 dp; `results.json` differs only by a single
rank-7↔8 tail swap in one session whose target is already rank 1); the
800-row `dev_set.jsonl` run is unchanged to 6 dp (5/800 sessions see a
tail reorder, 0 target-rank changes). The payoff is correctness /
robustness on constraint text the pipeline used to silently ignore, not a
score bump on the visible sets.

Left untouched on purpose: `_norm` / `blob` / `field_values` / the rerank
(already Unicode-safe), and `buckets.py` (`categories` is ASCII, so the
opener and bucket resolution never see non-English input). Rejected: a
secondary substring scan of `blob` for non-ASCII constraints — the
bucket path already carries the whole bucket into the rerank, and the
common CJK content (`进口` "imported", in 213 products) is
non-discriminative.

---

### User-profile rating affinity  (`config.PRIOR_RATING_*`)

> **REVERTED to `PRIOR_RATING_W = 0.0` (2026-09-01).** The `W = 0.15` win below
> was measured only on the 200-sample public set (+0.00122 composite, +0.0001
> on its 50-session val half — both under the evaluator's ~0.001 cross-run
> drift). On the held-out 800-session `dev_set` the term is net-**negative**:
> `composite 0.95093 → 0.95157`, `MRR 0.93317 → 0.93524` with it OFF, browsing
> and boundary MRR both improve, only `intent_override` dips 0.003. Public with
> the term off is `0.97026 → 0.96894` (i.e. it only ever bought public-set
> noise). The mechanism and code path are kept (inert at `W = 0.0`) for a
> private evaluator that might phrase the profile with more usable variance;
> the analysis below is retained as the record of why it does not ship.

`Agent.reset` receives an anonymised `user_profile`. Field by field:

| field | on the public set | used? |
|---|---|---|
| `preference_tags` | generic (`fit` / `comfort` / `material` …) | yes — folded into the low-weight fallback OR bag (`history_terms`) |
| `average_prior_rating` | `1..5`; varies (5.0×134, 3.0×22, 4.0×21, 1.0×14, 2.0×9) | yes — the rating-affinity term below |
| `purchase_frequency` | the constant string `"3-4 prior purchases"` for **all 200** | no — zero variance |
| `summary` | a template restatement of `preference_tags` + `rating_style` | no — adds only stopwords + the rating words |
| `rating_style` | 3-bucket coarsening of `average_prior_rating` | no — strictly less information |

**The term.** In `_rerank`, a pool candidate whose catalog `average_rating`
diverges from the user's `average_prior_rating` is penalised:
`PRIOR_RATING_W · min(|cat_rating − prior| / 2.0, 1.0)` (a full 2.0-point gap
costs the whole weight; an unrated item is neutral). Score-only — `evidence`
and the tie-count are untouched — and the hard-budget split still sorts last.
`PRIOR_RATING_W = 0.0` makes it byte-identical to baseline.

**Sweep.** `PRIOR_RATING_W {0 … 1.0}` × `PRIOR_RATING_MAX {5.0, 5.1}` on the
150/50 split. Global `r(prior, target average_rating)` is only **0.18**, but
the effect concentrates on the residual non-rank-1 set (`r ≈ 0.56` there):

| | baseline | `W = 0.15` |
|---|---|---|
| composite (all) | 0.96904 | **0.97026** |
| composite (train) | 0.97019 | 0.97179 |
| composite (val) | 0.96560 | 0.96570 |
| MRR / MTTC | 0.9718 / 2.125 | 0.9749 / 2.110 |

Three previously-deep sessions improve their rank (`public_0161` 2 → 1,
`0020` 9 → 7, `0099` 4 → 3) and four more hit a turn earlier; two
(`public_0014` / `0196`) hit a turn later but keep rank 1. No session loses
rank or drops a hit, hit@10 stays 1.0, MRR only rises, and `intent_override`
/ `boundary` are untouched (no-op). Plateau `W ∈ [0.15, 0.25]`; `W ≥ 0.30`
demotes rank-1 sessions (`0049 / 0083 / 0140`) and the composite falls below
baseline. Shipped at `W = 0.15`.

`PRIOR_RATING_MAX = 5.0` (gate the term out for `prior = 5.0` users) was
tested and is **worse**: a prior-5.0 user whose target is itself 5.0-rated
takes zero penalty while the mid-rated decoys ranked above it are demoted, so
`MAX = 5.1` (open for every rating) ships. The knob is retained for a private
evaluator that might phrase the profile differently.

The gain is small — comparable to the cross-run drift of the local evaluator
— but it is positive on all three splits with a clean mechanism and no
regression, and it is inert unless `PRIOR_RATING_W > 0`.

---

### Combined v1+v2 features  (`config.POSITION_W` / `VELOCITY_W` / `BOUNDARY_SLACK` / `BROWSE_DIVERSITY`)

Four levers ported from an earlier parallel monolithic prototype (a file then
called `agent_v2.py`, unrelated to today's `elimination/` package), each
gated by a `lexical/config.py` constant that is **inert at its default**, the backing
per-product data (`_CatalogIndex.field_pos` / `.velocity` / `.store` /
`.title`) built unconditionally. All four were A/B'd against the flags-off
baseline on the **held-out 800-session `dev_set`** — the post-`PRIOR_RATING_W`
bar ("report new levers on `dev_set`, not the 200-set"). One shipped.

#### F1 — match-prominence term  (`POSITION_W = 0.2`, SHIPPED)

The linear rerank's EXACT term tests each disclosed constraint against a
**set** of the item's discrete `features`/`details` values — it knows *whether*
the constraint is one of them, not *where*. A bare attribute word is the case
that needs "where": `"nylon"` substring-matches the jacket that leads with
`"Shell: 100% nylon"` and the cotton jacket whose bullet 7 mentions a `"nylon
carry bag"` identically. `_CatalogIndex.field_pos` now maps each folded value
to its first-occurrence index in the item's own metadata order, and `_rerank`
adds a score-only term

```
+ POSITION_W · 1 / (1 + earliest index over disclosed constraints)
```

exact-key first, then a containment scan (a bare `"nylon"` is never a whole
value). Evidence / tie-count are untouched, so it only reorders within an
evidence tie — the same discipline as the popularity and rating terms.

On the public set it is noise-band: interior peak at `0.2`, composite
`0.969042 → 0.969142` (`+0.0001`), one session converges a turn sooner. It
earns its place on **`dev_set`**, where it *generalises upward*:

| | baseline | `POSITION_W = 0.2` |
|---|---|---|
| composite | 0.951571 | **0.952421** (+0.00085) |
| MRR | 0.935237 | 0.937320 (+0.00208) |
| MTTC | 2.35625 | 2.34500 |
| hit@10 | 0.99625 | 0.99625 |

22 sessions move, **0 rank/hit regressions**; `buying` MRR `+0.0036`,
`intent_override` MRR `+0.0042`, `boundary` / `browsing` untouched. A real MRR
(rank) gain, not just MTTC, and larger on the never-tuned set than on the
tuning set — the opposite of the `PRIOR_RATING_W` failure mode. Public sweep:
`0.1` flat, interior peak `0.2`, `0.5` back to baseline.

#### F2 — ratings velocity  (`VELOCITY_W`, REJECTED, kept at 0.0)

`rating_number / listing-age` (year parsed from `details["Date First
Available"]`, age clamped `[1, 2024−2008]`, `log1p` then `/log1p(1e5)`), added
**alongside** `POP_W` — distinct from the rejected "age-adjusted popularity"
which *replaced* volume with an age residual. On the public set it looked
solid: a fine sweep `{0.3 … 0.6}` showed a plateau at composite `0.970889`
(`+0.0006`). It **failed `dev_set`**: at `0.5`, composite `0.951571 →
0.951052` (`−0.0005`), MRR `−0.0021`, **16 rank/hit regressions** including
`dev_0323` dropping a rank-7 hit to a miss; `boundary` / `browsing` / `buying`
MRR all fall. Stacked with F1 it drags the pair to `−0.0006` MRR / 15
regressions on `dev_set`. The public plateau was fit to 200-set noise — the
`PRIOR_RATING_W` story again. Code kept inert for a private evaluator whose
age/volume relationship differs (catalog `corr(log_age, log1p(count)) ≈
−0.106`).

#### F3 — boundary-shrug slack  (`BOUNDARY_SLACK`, REJECTED, kept `False`)

`message_parsing` now splits the two refusals — `RE_NO_PREF_BOUNDARY` ("I
don't have **a** preference for X", a one-off Boundary artefact) vs
`RE_NO_PREF_EXHAUSTED` ("**an additional** preference", true exhaustion) — and
`_SessionState.slack` counts the shrugs. `BOUNDARY_SLACK` would push the THIN
turn cap back one per shrug. No-op on the public set (all 10 boundary sessions
already rank-1); on `dev_set` (40 boundary sessions) it is no-op-to-slightly-
negative (composite `−0.00005`, 2 sessions a turn later). `agent.py`'s THIN
guard already caps at turn 3, which covers the boundary MTTC ~2.6 — the extra
slack buys nothing (the old prototype's exposure gate it was ported from
capped at turn 2). The split-refusal regexes are kept (strictly more correct);
`slack` is recorded but unread.

#### F4 — browse-track diversity spread  (`BROWSE_DIVERSITY`, REJECTED, kept `False`)

`Agent._diversify` dedupes a no-constraint Browsing slate by store and
title-shape before the `top_k` cut. **Zero sessions move on either set** — the
THIN guard trims Browsing turns 1–2 to a single id, and by turn 3
`state.active_constraints` is non-empty so the branch never fires on a scored
turn.

---

### Per-scenario `elimination` config split  (buying vs browsing, REJECTED, reverted 2026-09-01)

The router already routes Buying and Browsing/Boundary to the same `elimination`
instance. Tested giving each its own `EliminationConfig` (a frozen dataclass
threaded into `elimination.Agent` only, so one shared `Matcher`/FTS5 index) so
the two tracks could be tuned apart. The refactor was verified byte-identical to
the shared config on both sets (public 0.967579, dev\_set 0.959678) before any
divergence. Two divergences measured from that control, one field at a time:

| divergence | public | dev\_set(800) | note |
|---|---|---|---|
| control (both `DEFAULT`) | 0.967579 | 0.959678 | — |
| **a.** BUYING `exact` rerank weight 1.5 → {2.0, 2.5, 3.0, 4.0} | 0.967579 | 0.959678 | **byte-identical at every value.** Exact-match already outranks anything a loose match + tie-breakers can span (the rerank invariant, see *Rerank weight sweep*); scaling a decisive term reorders nothing. |
| **b.** BROWSING keep the diversified browse track through turn 2 (`browse_until_turn=2`) | 0.9611 | 0.9537 | browsing MTTC 2.19 → 3.00 pub / 2.29 → 3.02 dev; MRR flat. Delaying the first *ranked* slate by a turn just pushes every browsing conversion one turn later. |
| **b.** …through turn 3 (`browse_until_turn=3`) | 0.9366 | 0.9228 | browsing `hit_rate` falls to 0.988 pub / 0.959 dev — loses targets outright. |

`elimination`'s existing `DUAL_TRACK_ROUTING` (browse with no constraint in hand,
flip to the precision track the moment the first one lands) is already the right
split — the discriminating signal between the two scenarios is entirely the
turn-1 opener (`"A key requirement is: …"` vs `"…still exploring."`), and
`state.constraints` truthiness already captures it. Over the whole session the
two scenarios disclose about the same amount (~2.4 constraint phrases each);
Buying just front-loads one and converges faster. Nothing cleared the ship
gate (improve on **both** dev\_set and public); `starter/agent.py` + `starter/elimination/`
are reverted to the shared-config state and the `EliminationConfig` scaffold
removed — it has no inert default to keep (unlike the `lexical/config.py` flags).

---

## Rejected approaches

Everything here was implemented and measured, then removed — no constant, no
code path survives.

### Dense bi-encoder retrieval route

`sentence-transformers`, catalog embedded once at index build, cosine fused
into the (then-RRF) pipeline at a swept weight `W_DENSE`. Tried
`all-MiniLM-L6-v2` (384-d) at `W_DENSE ∈ {0.1, 0.2, 0.3, 0.5}` against a
150/50 train/val split and the browsing bucket (weakest-MRR, the one place a
semantic prior on weak-signal turns could plausibly beat popularity):

```
val      MRR 0.796 → 0.752 / 0.769 / 0.746 / 0.702   (hit_rate flat 1.0)
browsing MRR 0.689 → 0.641 / 0.661 / 0.633 / 0.622   (hit_rate flat 1.0)
```

`hit_rate@10` never moved off `1.0`, so the bi-encoder recovered **zero**
targets the lexical route had missed; its only effect was demoting
already-top-10 targets, monotonically worse with more weight, no interior
peak. The simulator's reveals are verbatim substrings of the target's own
catalog text, so retrieval here is an exact-match problem the lexical route
already nails; a 384-d cosine is a lower-resolution view of the same token
overlap and just adds tie-zone noise.

### Local-LLM reranking pass

Ollama, `qwen2.5:0.5b`, tested end-to-end against a real server on the full
public set. `hit_rate` and MTTC unaffected (the model never changed
whether/when the target was found), but MRR dropped **~26% overall** and in
every scenario bucket (boundary −44%, browsing −27%, buying −24%,
intent_override −20%), at real latency and token cost. A 494M-param model
working from bare titles was consistently a worse reranking signal than the
tuned lexical + slot-coverage pipeline.

### Title-term-overlap signal

Same outcome as the two above — demoted already-top-10 targets, recovered
none. Dropped.

### Structured-attribute exact-match with department-conflict penalty

Parse each row's `details` dict into a normalised `{dept, color, material,
closure, …}` map, match the revealed `Key: value` constraints against it,
fuse the order in — with a real DEPARTMENT conflict penalty (asked "Women" →
demote Mens rows), the one thing BM25's additive boost cannot do. At weight
1.0 it lost `0.9408 → 0.9368` (12 sessions worse, 0 better); no low weight
cleared baseline. Two causes: (a) the signal is sparse, so fusing it as a
full ranked list mostly re-weights raw BM25 order and dilutes the tuned
weights; (b) the evaluator's `coarse_category()` strips the gender segment,
so the customer's category message is leaf-only ("Accessories Belts", never
"Men Accessories Belts") and the dept penalty — the part with unique value —
almost never has a query-side dept to act on. What's left duplicates the
EXACT/LOOSE terms.

### Multi-signal RRF fuse + verbatim-phrase-count override  (the previous pipeline)

Six ranked lists (category / tf-idf / pool / slot-coverage / popularity /
phrase) fused by weighted reciprocal-rank fusion, then a `_phrase_priority`
pass that promoted candidates by verbatim disclosed-phrase-match count
(raw, or per-bucket IDF-rarity-weighted). Long tuning history:
`slot_coverage` swept `{0.3, 0.5, 1.0} → 1.0`; `category` / `tf-idf` weights
pinned at 0 (any non-zero tf-idf weight lowered the score);
`W_POP × W_PHRASE` grid flat at ~0.889 over `[0.3,0.5] × [0.6,0.9]`;
`PHRASE_PRIORITY_MIN_CONSTRAINTS` swept `{1, 2} → 1` (MIN=1 moved 5 sessions
earlier/higher, 0 regressions). Rarity-weighting the phrase score was
**byte-identical to raw count on all 200 sessions** — wherever the sort
engaged, the tied candidates matched the *identical* disclosed phrase set,
so any function of "which phrases matched" is equal for all of them.

Superseded wholesale by the linear rerank (composite `0.947 → 0.967`). The
phrase signal lives on as the rerank's EXACT + LOOSE terms.

### The six-session data-ambiguity dead end  (do not re-chase)

`public_0058 / 0083 / 0087 / 0126 / 0161 / 0174` sit at rank 3/7/10/3/3/10
and moved under **nothing** tried (RRF count, rarity weight, promote-tier,
and structurally unchanged under the linear key). At the hit turn the target
verbatim-matches every disclosed non-synthetic phrase and no candidate
outscores it — but 6–90 bucket-mates match the *exact same* phrase set (e.g.
in the 1354-item "shirts t-shirts" bucket, 914 members contain "pull on
closure"; "95% Polyester, 5% Spandex" is 24/681, shared by 13 others). The
disclosed constraints genuinely do not separate the target from the pack;
the tiebreak (target already ~rank 3) is the ceiling. Any further gain needs
a signal *outside* the disclosed text, and popularity / dense / LLM
rerankers were already shown to only demote correct targets here.

After the rerank weight sweep the surviving non-rank-1 set is
`public_0020 / 0076 / 0099 / 0120 / 0144 / 0161 / 0172 / 0175` (all in the
`phrase` weak-signal subset, mean best rank 1.15). Same story — the disclosed
constraints (`cotton`, `100% Cotton`, `Imported`, `Zipper closure`,
`Pull On closure`, `Rubber sole`) are near-universal in the bucket.

### IDF-weighted rerank hits  (`IDF_W`, removed)

Replace the flat EXACT/LOOSE hit *count* with an in-pool IDF-weighted sum, so
a disclosed constraint carried by only a few pool members outweighs a
near-universal one. Swept `IDF_W ∈ {0.1 … 1.5}` on the split: `0` (off) is the
optimum, every non-zero value is worse and monotone (`0.9699 → 0.9698` at
`0.2`, `→ 0.9689` at `0.7`, `→ 0.9681` at `1.5`). Down-weighting the common
matches also down-weights them for the *target*, which carries the same
common values as its bucket-mates; nothing is gained and the rare-value
signal is too sparse to separate the pack. Code and constant removed.

### Unexplained-attribute penalty  (`UNEXPL_W`, removed)

Per candidate, penalise each discrete feature/detail value that no disclosed
constraint accounts for — prefer the item whose discriminating attributes the
customer has actually spoken to. Swept `UNEXPL_W ∈ {0.05 … 0.75}`: strictly
worse on the full set at every value (`0.9699 → 0.9687` at `0.05`, tanking
val — `0.969 → 0.965` — by `0.1`), no interior peak. Real targets routinely
carry extra unmentioned bullets, so the penalty demotes them. Code and
constant removed.

### CONF_GATE re-evaluated under the linear rerank

The experimental score-margin gate (`CONF_GATE_*`, still present, still OFF)
was re-swept on top of the new weights. On top of THIN it is byte-identical
to baseline for `CONF_GATE_RATIO ≤ 1.5` and slightly worse above. *Replacing*
THIN it is far worse (best `0.9639` vs THIN's `0.9699`) — the turn/phrase-count
trigger beats the score-margin trigger, as originally found. Left OFF.

### BM25 column-weight re-sweep

`bm25(products, w_title, w_categories, w_features, w_details, …)` lifted into
`config.BM25_COLUMN_WEIGHTS` and coordinate-descended. With the pool now equal
to the whole bucket, the FTS5 order only feeds the small `RANK_W` penalty and
the exact-tie tiebreak, so it cannot move a target the evidence score already
separates: `val` composite is dead flat at `0.9690` across every value tried,
`train` wiggles < 0.0005. Held at the reference `(0.0, 6.0, 4.0, 2.5, 2.5,
1.5, 1.0)`; the constant stays only as documentation of the swept values.

### Age-adjusted popularity  (`scripts/sweep_age_popularity.py`)

Idea: `rating_number` is inflated by how long a listing has existed, so the
popularity prior should use an *age-adjusted* volume — the residual of
`log1p(rating_number)` regressed on `log(age)` from `details["Date First
Available"]` — instead of raw volume. **Rejected at the premise.** Catalog
`corr(log_age, log1p(rating_number))` is only **−0.106** — weak *and*
negative (newer listings carry slightly more ratings: review activity grew
over time and Amazon "Date First Available" resets on relist). `corr(age_days,
rating_number)` is −0.01. No coarse-category bucket exceeds ~−0.28…+0.10.
6.2% of rows have no parseable date. Ranking each public target within its
bucket by `average_rating · age_residual` instead of `average_rating ·
log1p(count)` *loses* top-1 (68→64), top-5 (142→136) and top-20 (175→170).
Full evaluator: pure age-adjusted volume `0.970264 → 0.967739` (MRR
−0.005); a `raw + w · ar·residual` blend is a no-op at `w ≤ 0.25` and
monotonically worse above. There is no age inflation to correct for.

---

## `ask_attribute` policy

Emit `"other"` every turn until the simulator has nothing left to disclose,
then `null`. The local evaluator's `customer_reply()` special-cases `"other"`
to reveal the next undisclosed constraint of **any** bucket (up to two per
turn) — the highest-yield thing to ask, since it never wastes a turn probing
a bucket the customer has no constraint in. `"other"` is in the evaluator's
`ALLOWED_ATTRIBUTES` and the contract enum, so it is valid output. Once the
simulator answers "I don't have an additional preference for other"
(`state.exhausted`), emit `null`.

## Intent override

Despite the "ignore my earlier preference" wording, the simulator's override
value is almost always something already disclosed a turn or two earlier via
an "other" reveal (it is drawn from `hard_constraints[0]`, which "other"
surfaces first). Wiping all slots on override was actively destructive — it
threw away accumulated working signal for one short duplicate phrase. So
`_override` **rewrites only the one slot the new value belongs to**, and
only re-inserts (at 2× weight) the value when it adds genuine signal: not
already covered verbatim in its slot, and multi-word / discriminating.
Same-bucket conflicts are erased only for the narrow, unambiguously
classifiable slots (`material`, `color`, `size`, `budget`); `feature` /
`style` / `use_case` accumulate.

### Opener trailing clause  (`RE_OPENER_TAIL` / `RE_OPENER_FILLER`)

The Intent-Override opener is `"I'm looking for <cat>. <old_value>"` where
`<old_value>` is `soft_preferences[-1]` — a real, catalog-derived attribute
of the target. The old parser kept only the category from this turn and
dropped the trailing clause into the low-weight fallback bag, so that
discriminating phrase did not reach the rerank until the simulator
re-disclosed it via an `"other"` reveal 2–3 turns later. `absorb` now
`_accumulate`s the turn-1 trailing clause (skipping the Buying
`"A key requirement is: …"` tail, already handled, and the Browsing
`"but I'm still exploring."` filler). Marginal on the public set
(`0.969892 → 0.969992`, `public_0068` turn 4→3, zero regressions): most
Intent-Override targets still can't reach rank 1 before the *hard* constraint
lands, and the thin-signal guard holds a rank-2 early hit back anyway — but
it is strictly more signal, earlier, at no cost. `public_0002` is
unchanged: its target ties leather+buckle decoys on evidence and only
separates once `"Imported"` is disclosed on turn 4; forcing it earlier means
a turn-3 rank-2 hit (RR 0.5) in place of a turn-4 rank-1 (RR 1.0), and
relaxing `THIN_KEEP` to allow it costs `mrr 0.974 → 0.889`.

---

## Dev tooling (`../scripts/`, not part of the submission bundle)

| script | what it does |
|---|---|
| `sweep_thin.py` | reproduce the `THIN_*` sweep (composite `0.901 → 0.941`) |
| `sweep_conf_gate.py` | evaluate the experimental confidence gate three ways (baseline / added / replacing THIN) |
| `sweep_weight.py` | sweep one module-level constant on `starter.lexical.agent` over a value grid / sample subset |
| `sweep_early_keep1.py` | blunt "show only top-1 for the first N turns" comparison |
| `eval_subsets.py` / `catalog_diag.py` / `_common.py` | weak-signal subset reporting, catalog stats, shared harness helpers |

The sweep scripts mutate a constant on the imported `starter.lexical.agent` module
(`agent_mod.THIN_KEEP = ...`); `lexical/agent.py` does `from .config import *`
and reads the names as bare module globals, so a reassignment is picked up on
the next call. The FTS5 term caps (`AND_TERM_CAP` / `OR_TERM_CAP`) are consumed
inside `catalog_index.py` — to sweep those, target `starter.lexical.config` instead.

Run the evaluator: `python -m evaluator.local_evaluator` (writes
`results.json`). Unit tests: `python -m unittest discover -s tests`.
