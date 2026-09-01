# Elimination retrieval track — experiment log

What was tried on the **elimination** track (`starter/elimination/`), and what it did
to the score — the code carries the rationale, this file carries the record. Its
sibling, `starter/lexical-track.md`, covers the lexical track and the scenario
router; top-level overview in `../README.md`.

> **Path note.** This log was written against the development workspace, whose
> layout differs from the submission bundle. Read `starter/agent.py` as
> `starter/elimination/` + `starter/agent.py`, `tools/` as `scripts/` (only a subset is
> shipped), and `dev/dev_set.jsonl` as `data/dev_set.jsonl`. The measurements
> are unaffected.

## How to read the numbers

`TechnicalScore = 0.50*HitRate@10 + 0.30*MRR + 0.20*Efficiency`, where
`Efficiency = clip((11 - MTTC)/10, 0, 1)`.

Three evaluation sets are used, and a delta is only comparable within one of them:

| Set | Size | Command | What it is |
| --- | --- | --- | --- |
| **public** | 200 | `python3 -m evaluator.local_evaluator` | The released labeled sessions. The leaderboard-comparable number. |
| **dev-800** | 800 | `python3 -m evaluator.local_evaluator --dataset dev/dev_set.jsonl --output dev/dev_results.json` | Stratified to match the public target distribution (TVD ≈ 0.27). The tie-breaker when public is too small to separate two variants. |
| **dev-full** | 2,000+ | `python3 tools/run_dev.py` | Broad sweep over the catalog. Scores lower by construction — 37% of it is items the public set never samples — so read it for direction, never as a leaderboard estimate. |

Public is 200 sessions, so one session is worth ~0.0025 of hit rate. Anything under
about +0.002 on public alone is noise; that is why most decisions below were made on
dev-800 and confirmed on public.

## Score timeline

| Stage | Public score | Hit@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| Shipped weak-BM25 baseline (`docs/baseline_results.json`) | 0.1067 | 0.125 | 0.068 | 9.81 |
| Constraint-accumulating rewrite (`60aad56`) | — | — | — | — |
| Linear reranker added (`325d5d6`, commit msg) | 0.9063 | — | — | — |
| (`a788ece`, commit msg) | 0.906 | — | — | — |
| Exposure gate at turns 1–2 | 0.9600 | — | — | — |
| Bucket prefilter (`results.json`) | 0.9635 | 1.000 | 0.9609 | 2.24 |
| Ratings-velocity feature | 0.9663 | 1.000 | 0.9666 | 2.185 |
| `OVERRIDE_RESETS = False` (current tree) | **0.9670** | 1.000 | 0.9688 | 2.185 |

Current dev-800: 0.9579 raw / 0.9617 public-reweighted (was 0.9551 / 0.9592).
Current dev-full 2,000: 0.9291 (hit 0.9845, MRR 0.9025, MTTC 2.694).

---

## 1. Reading the customer

**Constraint accumulation instead of query rewriting** — *landed, the whole win.*
The simulator quotes the target's own catalog strings verbatim, so every turn is
evidence rather than a paraphrase. Parse the disclosed phrases out, accumulate them
across the session, and use them as exact-phrase filters: eliminate first, rank
second. This is the step that moved the score from 0.107 to ~0.90.

**Splitting reveals on `; `** — *landed.* A constraint may itself contain a semicolon,
so splitting is ambiguous, but it is safe under both readings: if the real constraint
was "A; B" then A and B are each still phrases of the target. The joined form matched
the wrong product often enough to evict the right one.

**Distinguishing the two refusal phrasings** — *landed.* "I don't have **a**
preference for X" is a one-off Boundary-scenario shrug and says nothing about X;
"**an additional** preference for X" means X is genuinely exhausted. Treating them
alike either burned attributes that still had constraints left, or kept re-asking
spent ones.

**Unrecognised phrasings fall through to a weak whole-line phrase** — *landed.*
Rewording degrades the signal instead of erasing it.

## 2. Retrieval

**Coarse-category bucket prefilter** — *landed, the last big win.*
The opening message embeds the evaluator's own `coarse_category(...)` output verbatim,
so the agent mirrors that function, buckets the 50k catalog, and hard-restricts
retrieval to the target's bucket.

- public 0.9592 → **0.9635**, hit rate 0.995 → **1.000**; dev-800 0.9500 → **0.9551**.
- Before it, **40 of 200** public sessions returned a turn-1 top-10 with *zero* items
  from the target's bucket, even though the category was parsed correctly — it was
  only a soft token constraint.
- Cost: roughly 2× session latency (~24ms → ~50ms), from the UNINDEXED `bucket_key`
  filter and the bucket-only backfill scans. Not optimized.
- Not a recall backbone — recall@10 was already 0.995 without it. Its value is turn-1
  precision, i.e. MTTC.

**Exact bucket lookup only, no fuzzy ladder** — *landed, deliberate.*
The opener fragment matches a bucket key byte-for-byte in **200/200** public sessions,
so containment / token-overlap / singularization fallbacks would be dead code. It is
also the safe side of the trade: a *guessed* bucket hard-excludes the target, which
ranking can never recover from, while a missed bucket only falls back to the
unrestricted pipeline.

**Keeping the category terms in the MATCH expression once the bucket filters**
(`BUCKET_KEEPS_CATEGORY_TERMS`) — *landed.* Redundant as a filter, not as a ranking
signal: it is what puts the taxonomy words into the BM25 score that orders the pool.
Dev-800: keep **+0.0051** vs drop +0.0025.

**Constraint backoff ladder** — *landed.* Apply the most specific constraint first (a
long verbatim feature narrows far harder than a bare material word); any form that
would zero the count is dropped rather than allowed to empty the pool. A clipped
constraint also retries without its trailing partial token, because the customer clips
at 180 chars and that can land mid-word.

**Price excluded from the filter stage** — *landed.* A budget the catalog cannot
satisfy exactly zeroed every count and dropped every text constraint with it. Applied
at ranking time instead, where it can be backed out safely, and dropped entirely if it
would return nothing.

**Non-English boilerplate aliasing** (`进口` → `imported`) — *landed deliberately, and
it costs a little.* 323 catalog rows carry non-Latin text; in 212 of them the only
non-English content is a lone `进口` ("imported") feature bullet, unreachable by any
English query. The alias indexes the English sense *alongside* the original, never
instead of it, and sharing the original bullet's position slot — the customer quotes
verbatim, so a constraint that reads "进口" still has to match the bullet it came from.
Applied on a second pass, so the one row carrying *both* `进口` and a real `Imported`
bullet (`B0BD3W6QR7`) keeps its genuine bullet's slot. Verified over all 50,000 rows:
212 gain the alias, no existing key's position moves.

Measured on dev-800 (`BUCKET_ENABLED=True`, only the alias table toggled, in-process):

| | alias off | alias on | delta |
| --- | --- | --- | --- |
| TechnicalScore | 0.955379 | **0.955129** | **−0.000250** |
| MRR | 0.947012 | 0.946179 | −0.000833 |
| Hit@10 / MTTC | 0.996250 / 2.343 | 0.996250 / 2.343 | unchanged |

Two sessions move, both down: `dev_0281` rank 1 → 2, `dev_0356` rank 2 → 3. Nothing
improves, and nothing can — `intent_card()` builds constraints verbatim from the
target's own bullets, so a `进口` row's constraint *is* `进口`, which already matches
exactly through `self.items`. The alias sits on a branch no session reaches.

The cost is not dilution in the vague sense. `Imported` is boilerplate on 13,846 of
50,000 rows, which makes it a near-random partition uncorrelated with everything else
in a query — an unusually good separator of near-duplicates. `dev_0281`'s target is
`B07ZNLTF7X` (VIJIV sequin flapper dress), whose four disclosed constraints are
`polyester` / `100% polyester` / `imported` / `zipper closure` — all boilerplate, none
describing a flapper dress. The only thing keeping `B09TFHCSKT` ("Women 1920s Gatsby
Flapper Dresses Sequin ...", shares *dress*, *flapper*, *sequin*, same category, same
`100% Polyester` + `Zipper closure`) out of the results was that it says `进口` where
the target says `Imported`. Aliasing removes exactly that separation, so the rows the
alias admits are drawn from the target's closest lookalikes rather than from random
junk. The benefit and the harm are one mechanism: making a row reachable by "imported"
*is* making it a competitor on "imported", and no split of the index surfaces gets one
without the other.

Kept anyway, with open eyes: −0.00025 buys catalog correctness that this evaluator
cannot reward but a real shopper would expect. The generalizable warning is the
inverse — low-information boilerplate (`Imported`, `Machine Wash`, `Pull On closure`,
department strings) can be load-bearing *because* it is meaningless and therefore
uncorrelated with the query. Normalizing it away is not free.

**Dual-track routing** — *landed.* A shopper who leads with a requirement goes down the
precision/filter track; one who leads with only a category goes down a browse track
that spreads the ten slots across the category (deduped by store and by title shape),
because with nothing to filter on, ten near-neighbours are a poor guess that teaches
nothing.

**Deep paging on dry turns instead of a served-item blacklist** — *landed, and the
blacklist was rejected.* Once the customer stops disclosing, the ranking stops changing
and the ten items produced have already been shown and missed, so a dry turn pages one
slate deeper. It is an offset, not a blacklist: an Intent Override session does not
register a hit until the override lands, so the target can legitimately appear in an
early slate without ending the session — blacklisting what was shown would bury it for
the rest of that session.

## 3. Ranking

**Linear reranker over a 500-candidate pool** — *landed* (`325d5d6`, 0.9063).
Five features — exact match, loose substring, position, popularity, BM25 rank — scored
as a plain dot product. BM25 survives as the tie-break, so a session with no usable
evidence returns exactly what it returned before the reranker existed.

**Position feature** (`POSITION_W = 0.30`) — *landed.* A bare attribute word matches by
substring against the whole product text, so the jacket that *is* nylon and the cotton
jacket that ships with a nylon carry bag earn byte-identical evidence — **51 of the 56**
sessions lost to an evidence tie were exactly this. Manufacturers lead with the defining
material and bury the incidental ones, so the index of the match separates them.
Weighted so position + popularity + BM25 stays under the 0.6 that one loose match is
worth: tie-breakers may only reorder candidates evidence has already declared equal.

**Position counting toward the tie count** (`POSITION_BREAKS_TIES`) — *tried, left off.*
It no longer gates anything (the exposure gate replaced the tie gate), so it would only
change how a trimmed slate is described to the customer, and position is not a reliable
enough separator to claim the items are distinguishable.

**Offline weight tuning** (`tools/tune_reranker.py`) — *landed as tooling.*
Re-running the evaluator per candidate vector costs ~20s, i.e. a few hundred trials a
day. Nothing upstream of the ranking depends on the ranking, so sessions can be traced
once and the cached candidates re-scored offline in milliseconds. Every feature is
oriented so more is better, which makes the pruning exact rather than approximate:
candidates dominated by the target in all five features can never overtake it under any
non-negative weights. `search` reports on a held-out split, and re-running under several
`--split-salt` values is the cheap test for whether a gain is real or fitted.

**Popularity weight, and whether "nearly tied" should count as tied** — *swept, nothing
changed, and two of the three questions turned out to be structurally closed.* Measured
on a fresh 1,000-session trace (public + dev, 0 mismatched turns, live 0.9568); the
earlier traces were stale against `agent.py` and would have measured the pre-bucket
retrieval.

Sweeping `pop_w` alone, paired bootstrap (2,000 draws) against the shipped 0.30:

| `pop_w` | raw delta | 95% CI | public delta | 95% CI |
| --- | --- | --- | --- | --- |
| 0.00 | −0.02058 | [−0.0251, −0.0165] | −0.02520 | [−0.0308, −0.0201] |
| 0.20 | −0.00280 | [−0.0050, −0.0011] | −0.00389 | [−0.0067, −0.0018] |
| **0.30** | — | shipped | — | shipped |
| 0.35 | +0.00089 | [−0.0007, +0.0030] | +0.00028 | [−0.0007, +0.0014] |
| 0.50 | +0.00145 | [−0.0001, +0.0033] | +0.00111 | [−0.0001, +0.0023] |
| 1.00 | +0.00164 | [−0.0005, +0.0040] | +0.00164 | [−0.0001, +0.0035] |

0.30 sits exactly on the knee: every step down is significant, every step up has a CI
straddling zero. Popularity is not a small feature — deleting it costs **−0.021 raw /
−0.025 public**, which is more than the bucket prefilter won — it is simply already
tuned.

*Popularity is already a pure tie-break, not merely weighted like one.* Scaling the
evidence weights 100x (`150, 60, 0.30, 0.30, 0.10`), which makes it arithmetically
impossible for a tie-break to cross an evidence boundary, scores **identically to the
shipped vector: delta +0.00000, zero-width CI.** The invariant §3 documents — tie-breakers
may only reorder candidates evidence has declared equal — is binding in practice on all
1,000 sessions, not just by construction. There is no lexicographic variant to build.

*A tie **tolerance** cannot do anything.* Evidence is `1.5*exact + 0.6*loose` over integer
counts, so it is quantized. Across all 52,971 carried candidate rows the target-to-rival
evidence delta takes **24 distinct values, and the smallest nonzero one is 0.6**:

| delta (rival − target) | rows | share |
| --- | --- | --- |
| **0.00** | 21,959 | 41.5% |
| −1.50 | 10,873 | 20.5% |
| −4.20 | 4,944 | 9.3% |
| −2.10 | 4,722 | 8.9% |

So any epsilon below 0.6 merges nothing at all, and any epsilon at or above 0.6 merges
candidates separated by a full loose match — the atomic unit of evidence, not a rounding
artifact. There is no "nearly tied" band between the two. (`tied` has exactly one
consumer, the `TIED_PROMPT`/`NARROW_PROMPT` choice, so widening it would change wording
rather than ranking, and would need a full dev run rather than a trace replay to score.)

Ties are common and the tie-break is load-bearing — 41.5% of carried rows are exact
evidence ties, median 3 tied rivals per ranked turn, max 315 — which is why popularity
is worth 2 points. The lever that is exhausted is *reweighting* it; splitting those
groups needs a new signal, not a bigger `pop_w`.

**Provenance features: `Date First Available`, `average_rating`, store name** — *built,
measured, reverted.* The follow-on to the popularity sweep above: if reweighting the
tie-break is exhausted, try a new signal. Three were added as *separate* features
(rather than one hand-blended composite) so the tuner could price each independently
and say which, if any, carried its weight. Spliced in ahead of BM25, which must stay
last — the dominance pruning relies on the final coordinate being strictly decreasing
in pool position. `tools/tune_reranker.py` was generalised from a hardcoded 5-tuple to
an arbitrary-length vector to support this; **that generalisation was kept.**

The catalog-side priors looked genuinely promising, which is the point of the entry:

- **Recency survives conditioning on `rating_number`** — the test `average_rating`
  fails. Inside a fixed rating band, an item first listed in 2019+ is 2.54x likelier to
  be the target in the 0–100 band and 2.56x in 100–1,000. It is not popularity in
  disguise: a newer item has *fewer* ratings, so the two signals partly oppose.
- Marginal year lift runs from 0.26x (2014) to **5.56x (2023)**. Coverage is good:
  93.8% of the catalog, 97.5% of targets.
- `Generic` covers 197 rows for **0 targets**, exactly the "no-name store" pattern one
  would predict — but the prior only expects ~0.8, so P(0) ≈ 0.45. Not significant.
  Nor is it a *generic* pattern: Nike has 564 rows and 0 targets, PUMA 265 and 0.

Every one of them died on contact with the reranker (1,000-session trace, provenance-off
baseline 0.95680 raw / 0.96081 public):

| feature | best weight | raw delta | 95% CI | verdict |
| --- | --- | --- | --- | --- |
| recency | 0.05 | +0.00024 | [−0.0008, +0.0013] | **harmful above 0.1**, monotonically |
| quality (`average_rating`) | 0.10 | +0.00059 | [−0.0006, +0.0022] | noise |
| brand (store not anonymous) | any | +0.00015 | [+0.0000, +0.0005] | **perfectly flat** |

`recency` is the instructive failure. It decays monotonically — 0.95680 → 0.95189 at
w=0.5 → 0.94942 at w=0.8 — despite a real, conditioning-proof marginal lift. The lift is
a *catalog-wide* prior, and by the time the reranker sees a tie group the pool is already
filtered to one category and ordered by popularity; within that group the newer rival is
not the likelier target. A prior that is real over 50,000 rows can be worthless over the
three rows that are actually tied.

`brand` scores *identically* at every weight from 0.05 to 0.8. Its deltas among carried
candidates are all zero: a target and its close rivals sit in the same category and are
almost always both real brands, so the feature never discriminates where discrimination
is needed. A 0-target store list is not a tie-break.

The joint search overfits exactly as the split is designed to catch: **+0.0037 on train
(0.9601 → 0.9638), −0.0004 on holdout**, CI covering zero, and it gets there by pushing
`popularity` to 2.9 and crushing `loose` to 0.17 — i.e. by breaking the evidence-dominance
invariant that §3 exists to protect. Reverted: the features nearly doubled carried
candidate rows (52,971 → 97,553) by weakening the dominance pruning, which taxes every
future tuning run, and they bought nothing. Post-revert trace reproduces the baseline
exactly (52,971 rows, 0.956797).

The generalisable lesson matches the `进口` entry from §2: a signal's marginal lift over
the whole catalog says nothing about its value *inside a tie group*, because retrieval
has already conditioned the pool on most of what made the lift look real.

**Ratings velocity — `rating_number` normalised by listing age** — *landed, +0.0028.*
The one thing in this line of work that survived. `VELOCITY_W = 0.5`, alongside an
unchanged `pop_w = 0.30`.

It exists because it is an **interaction the linear model cannot build for itself**.
Popularity and recency were both already available as additive terms, and no weight
vector over the two expresses a ratio. The additive form had already been measured and
rejected — recency on its own decays monotonically, 0.9568 to 0.9494 at w=0.8 — so the
*same two fields*, combined as a denominator rather than a summand, going from harmful
to +0.0028 is the entire result. Nothing was added to the catalog; only the formula
changed.

Measured three ways, all agreeing:

| | baseline | velocity 0.5 | delta |
| --- | --- | --- | --- |
| trace replay, raw | 0.95680 | 0.95961 | **+0.00282** (95% CI [+0.0007, +0.0054]) |
| trace replay, public-weighted | 0.96081 | 0.96324 | **+0.00243** (95% CI [+0.0006, +0.0044]) |
| live public-200 | 0.963471 | **0.966279** | +0.002808 (MRR 0.9609 → 0.9666, MTTC 2.24 → 2.185) |
| live dev-800 raw | 0.955129 | **0.957947** | +0.002818 |
| live dev-800 public-reweighted | 0.959192 | **0.961748** | +0.002556 |

Three findings worth keeping:

- **The naive intuition for this feature is backwards.** "Old listings had longer to
  accumulate ratings, so normalise by age" does not hold in this catalog:
  `corr(log1p(ratings), age) = −0.118`. *Newer* listings carry more ratings, not fewer
  (mean ~14 for 2015 vintage, ~33 for 2020). Velocity earns its keep in spite of the
  correlation running the wrong way for it, not because of it.
- **It is additional to popularity, not a redistribution of it.** Replacing popularity
  outright is worse at every velocity weight (0.9459 at v=0.1, still below baseline at
  v=0.5); splitting a fixed 0.30 budget across the two is flat. The gain requires new
  weight. Popularity backs the item with the most ratings outright, velocity backs the
  one accumulating them fastest, and it is the *disagreement* that splits a tie group.
- **It is still a tie-break in practice, despite exceeding the 0.6 bound.** At v=0.5 the
  tie-breakers can arithmetically out-vote one loose match. Forcing evidence to dominate
  (weights x100) moves the score by **+0.00004 raw / +0.00008 public**, so ~99.97% of
  the gain is within-tie-group reordering. `VELOCITY_W = 0.3` keeps the §3 invariant
  exactly binding for about 0.001; 0.5 was kept with that trade recorded.

Holdout stability across six independent `--split-salt` cuts: four clearly positive
(+0.0015 to +0.0056), one flat, one slightly negative (−0.0003). Real, and noisy at this
sample size — the effect is roughly the size of one dev-800 stratum.

**Per scenario, the aggregate replicates but the attribution does not.** Overall the two
sets agree to five decimals (+0.00281 public, +0.00282 dev); which scenario pays for it
they disagree about outright.

Public-200:

| scenario | n | Hit@10 | MRR | MTTC | score delta |
| --- | --- | --- | --- | --- | --- |
| buying | 80 | 1.000 → 1.000 | 0.9587 → 0.9764 (+0.0177) | 1.637 → 1.538 | **+0.00731** |
| intent_override | 30 | 1.000 → 1.000 | 0.9370 → 0.9417 (+0.0046) | 3.600 → 3.600 | +0.00139 |
| boundary | 10 | 1.000 → 1.000 | 1.0000 → 1.0000 | 3.100 → 3.100 | 0.00000 |
| browsing | 80 | 1.000 → 1.000 | 0.9672 → 0.9620 (−0.0052) | 2.225 → 2.188 | **−0.00081** |
| **overall** | 200 | 1.000 → 1.000 | 0.9609 → 0.9666 (+0.0057) | 2.240 → 2.185 | **+0.00281** |

Dev-800:

| scenario | n | Hit@10 | MRR | MTTC | score delta |
| --- | --- | --- | --- | --- | --- |
| boundary | 40 | 0.975 → **1.000** | 0.9208 → 0.9458 (+0.0250) | 3.425 → 3.275 | **+0.02300** |
| intent_override | 120 | 0.9917 → 0.9917 | 0.9302 → 0.9436 (+0.0134) | 3.517 → 3.517 | +0.00402 |
| browsing | 320 | 1.000 → 1.000 | 0.9478 → 0.9535 (+0.0057) | 2.341 → 2.291 | +0.00270 |
| buying | 320 | 0.9969 → 0.9969 | 0.9537 → 0.9504 (−0.0033) | 1.769 → 1.722 | −0.00004 |
| **overall** | 800 | 0.9962 → 0.9975 | 0.9462 → 0.9504 (+0.0042) | 2.342 → 2.296 | **+0.00282** |

Public says buying wins (+0.0073) and browsing pays (−0.0008); dev says browsing wins
(+0.0027) and buying is flat-negative. Both cannot be the mechanism. With 80 buying
sessions on the public set the per-scenario MRR moves are within their own noise, so the
decomposition should not be read as a finding — only the aggregate replicates.

**The one thing that is consistent across every scenario on both sets is MTTC: it
improves or holds, and never worsens** (public 1.637→1.538, 2.225→2.188; dev 3.425→3.275,
2.341→2.291, 1.769→1.722). MRR moves in both directions depending on the slice. That
matches §9 — the remaining headroom was Efficiency, and this feature converts earlier
rather than ranking better, which is where the points actually were.

The generalisable point, against the reverted provenance features directly above: those
failed as *additive priors* over the whole catalog. The same fields succeed as a *ratio*,
because retrieval has already conditioned the pool on category and popularity, and what
survives that conditioning is not a level but a rate.

## 4. Turn policy — where the remaining points actually were

**Exposure gate: serve only the single best candidate on early turns** — *landed, the
second-biggest win.* The harness ends the session the moment the target appears
anywhere in the slate, so whatever rank it surfaces at is the rank it is scored on
forever. Early on, the order below the top is settled by popularity and BM25 position —
close to a coin flip — so a full slate spends the session's one scoring opportunity on
that coin flip. Serving top-1 makes it rank 1 or nothing; a miss costs one turn, which
is the cheap side: an extra turn costs `0.2 × (1/200)/10`, while lifting a hit from rank
3 to rank 1 is worth `0.3 × (2/3)/200` — roughly ten times as much.

Measured, public / dev:

| Reveal schedule | Public | Dev |
| --- | --- | --- |
| Full slate every turn (the old withhold-on-tie gate) | 0.9497 | 0.9415 |
| Turn 1 only | 0.9440 | 0.9319 |
| **Turns 1–2 (shipped)** | **0.9600** | **0.9503** |
| Turns 1–3 | 0.9526 | 0.9370 |
| Turn 1, then top-3 on turn 2 | 0.9503 | 0.9406 |
| Turns 1–2, but turn 2 only when the top is tied | 0.9593 | 0.9471 |

Turn 1 alone gives back most of the win — by turn 2 the second constraint has landed and
the top-1 is worth betting on — and turn 3 starts losing sessions outright, which costs
hit rate at 0.5 against MRR's 0.3. Gating on the tie test is worse than trimming
unconditionally, so ties no longer decide anything; they only pick which explanation the
customer is given.

**Withholding the slate entirely** — *tried, strictly worse.* An empty slate cannot
convert at all, and the top-1 it was suppressing converts often enough to be worth
**+0.0096 public / +0.0056 dev** on its own.

**Boundary slack: an evidence-free turn buys the gate one more turn** — *landed.*
A Boundary shrug costs a turn and discloses nothing, shifting that whole scenario one
turn later than Browsing — its first constraint lands on turn 3, not turn 2. The gate
counts turns, so by then it had already opened, and the first genuinely-ranked slate
went out at full width. Browsing converts 51.5% of its sessions on the gated turn 2 at
100% rank-1; Boundary converted 75.8% on the ungated turn 3 at 64.1% — essentially the
whole 0.166 MRR gap between them. Over 2,490 Boundary dev sessions: MRR 0.7397 →
**0.8990** (Browsing sits at 0.9062), hit −0.0020, MTTC +0.286. No other scenario emits
that refusal, and the change was verified bit-identical over 5,914 non-Boundary
sessions.

**Asking `other` every turn** — *landed by measurement, not by laziness.* The customer
releases any undisclosed constraint for `other`, but only same-class constraints for a
typed attribute, so `other` strictly dominates the typed asks.

## 5. Intent override — taking the override literally

Tried: on "actually, ignore my earlier preference", clear the accumulated constraints
(and optionally the category and the ask history). It was measurably wrong. The
simulator draws the replacement from `hard_constraints[0]` of the **same** target and
the superseded preference from that same target's `soft_preferences`, so the customer
never says anything false and the target never changes. Resetting also creates its own
failure mode: when the replacement is a bare word like "cotton", filtering on it alone
leaves a pool the target does not survive into, and no reranking recovers an item
retrieval never returned.

Re-measured on the current tree (2026-08-31), everything else held fixed:

| `OVERRIDE_RESETS` | Public | Dev-800 | Override-scenario MRR (public / dev) |
| --- | --- | --- | --- |
| `True` — reset (**current setting**) | 0.963471 | 0.955129 | 0.9370 / 0.9302 |
| `False` — keep constraints | **0.964242** | **0.956320** | **0.9542 / 0.9383** |

Keeping is worth **+0.0008 public / +0.0012 dev**, concentrated entirely in the
Intent Override scenario, and it also lifts dev hit rate 0.99625 → 0.99750.

> **Resolved (2026-09-01).** `OVERRIDE_RESETS = False` is now what the tree runs, which
> is what the comment above it and the inline comment in `SessionState.absorb` always
> described. Public moved 0.9663 → 0.9670, override-scenario MRR 0.9417 → 0.9567. Note
> this is separate from `heard`, which is never cleared — ranking always sees the
> superseded preference even when filtering forgets it.

## 6. LLM reranking over the top 10

*Built, measured across four local models and two prompt modes and two architectures,
shipped **off** at **−0.0058**.* `LLM_RERANK_ENABLED` is hard-wired to `False` in
`starter/agent.py` — the env-var read above it is commented out deliberately, so no
environment can switch this on during official scoring. Restore that line to re-enable
it. The shipped path makes no model calls and scores exactly what it scored before the
layer existed (0.9670, 0 tokens).

### The ceiling that motivated it

§9 says the remaining headroom is Efficiency — turn-1 conversion — so the question
worth asking is how much of it a better top-1 could reach. Instrumenting `rerank` to
record the untrimmed slate on every turn answers it exactly. Over the public set, the
earliest turn the target appears anywhere in the reranked top 10:

| Turn it first appears | At rank 1 | At rank 2–10 |
| --- | --- | --- |
| 1 | 65 | 32 |
| 2 | 72 | 21 |
| 3–4 | 10 | 0 |

An oracle that always pulled the target to rank 1 of the slate the linear reranker
already produced would score **0.9886** against 0.9670 — **+0.0216**, almost all of it
Efficiency (MTTC 2.185 → 1.57, MRR → 1.000). Recall is not the constraint; ordering the
ten items we already have is.

### What the layer is allowed to do

Reorder, and nothing else. `starter/llm_rerank.py` returns a permutation of the slate
it was handed — indices the model omits are appended in their incoming order, indices
it invents or repeats are dropped. It cannot introduce a product, cannot drop one, and
therefore **cannot turn a hit into a miss**; the worst it can do is order a slate badly,
which is the risk the linear model already carries. Every failure path — no client,
unreachable server, timeout, malformed reply, exception — returns the incoming order,
so scoring with the network disabled (`docs/submission_rules.md`) is identical to
scoring without the file.

It sits between ranking and the exposure trim, which is the only place it can pay: on
turns 1–2 the slate is cut to one item, so reordering after the cut would reorder a
list of length one. It is gated to those turns and to contested slates
(`tied > 1`), which is 158 decisions over the 200 public sessions.

Three levels of subordination, each measurable independently:

| Level | What it means |
| --- | --- |
| `rank` mode | the model's ordering replaces the linear ordering within the top 10 |
| `pick` mode | the model promotes **one** candidate; the other nine keep linear order |
| promotion ceiling | the promotion is honoured only if the candidate came from rank ≤ N |

`ceiling = 1` reproduces the linear baseline exactly, which is the sanity check that the
policy plumbing is wired correctly.

### Measurements

The decision set is 158 gated decisions; the target is in the slate for 145 of them,
and the linear reranker puts it first in **90 (0.621)**, MRR 0.7559. That is the number
to beat, and no configuration beat it.

| Model | Mode | Decisions | Linear top-1 | LLM top-1 | Fixed | Broke |
| --- | --- | --- | --- | --- | --- | --- |
| llama3.2 3B | rank | 15 | 6 | 1 | 0 | 5 |
| llama3 8B | rank | 38 | 20 (0.526) | 17 (0.447) | **0** | 3 |
| llama3.2 3B | pick | 145 | 90 (0.621) | 17 (0.117) | 15 | 88 |

llama3-8B moved the target *up* zero times in 38 decisions. Sweeping the promotion
ceiling over the cached answers (`tools/llm_policy.py`, free — no re-calling):

| Ceiling | top-1 | MRR | Fixed | Broke | Net |
| --- | --- | --- | --- | --- | --- |
| linear | 90/145 (0.621) | 0.7559 | — | — | — |
| none | 17/145 (0.117) | 0.4909 | 15 | 88 | **−73** |
| ≤2 | 56/145 (0.386) | 0.6386 | 14 | 48 | −34 |
| ≤3 | 36/145 (0.248) | 0.5651 | 15 | 69 | −54 |
| ≤5 | 23/145 (0.159) | 0.5139 | 15 | 82 | −67 |

> **Sample-size warning, recorded because it nearly cost a wrong conclusion.** On the
> first 40 decisions, `ceiling ≤2` measured **+1** and looked like a marginal win. On
> the full 145 the same policy is **−34**. Forty gated decisions is roughly a fifth of
> the public set and separates nothing; the +1 was noise of exactly the size §"How to
> read the numbers" warns about.

mistral-7B was started and abandoned: it needs **26 s to emit an 8-token reply** to a
trivial prompt on this machine, so it is CPU-bound here and a 158-decision run cannot
finish in a useful time. Two 7–8B models resident at once made Ollama queue requests
badly enough to stall a run for hours; the client's own timeout was verified to fire
correctly (5.1 s at a 5 s setting), so that was contention, not a defect.

### End to end, with it switched on

The decision-set numbers above are a proxy. This is the score with the layer enabled
(measured before it was hard-wired off), llama3.2-3B in `pick` mode, full public set:

| Metric | LLM off | LLM on | Delta |
| --- | --- | --- | --- |
| Hit Rate@10 | 1.000000 | 1.000000 | **+0.000000** |
| MRR | 0.968847 | 0.972181 | +0.003334 |
| MTTC | 2.185 | 2.525 | **+0.340** |
| Efficiency | 0.881500 | 0.847500 | −0.034000 |
| **TechnicalScore** | **0.966954** | **0.961154** | **−0.005800** |

167,409 tokens and ~40 minutes of wall clock, against ~6 seconds deterministic.

Three things are worth reading out of this, and only the first is bad news.

**Hit rate is unchanged at exactly 1.000.** The permutation invariant was a design
claim; this is the measurement of it. A model that got 88 of 103 decisions wrong lost
zero sessions.

**MRR went *up* while the score went down.** Breaking a turn-1 hit does not fail the
session — it survives to turn 3, where the exposure gate opens, the full slate goes
out, and the linear ordering puts the target high anyway. The damage converts into
extra turns rather than lost sessions, so it lands on Efficiency (20% of the score)
instead of hit rate (50%).

**The damage is confined to where the gate fires.** Buying MTTC 1.538 → 2.150 and
Browsing 2.188 → 2.425; Boundary (3.100) and Intent Override (3.600) are untouched,
because their turn-1/2 slates are not contested in the way the gate tests for.

Together these are the argument for the guard rails rather than against them: a badly
performing model cost **−0.0058**, and a pure-LLM reranker with no permutation
constraint and no exposure gate would have taken hit rate down with it.

### Final reranker vs. 7th feature

The obvious objection to the architecture above is that the model *overrides* the
linear ordering instead of contributing to it. The alternative is to make its pick one
more term in the dot product, worth `w`, so it has to out-score the linear margin to
move anything. `tools/llm_simulate.py` answers this off the cached answers — the model
is not called again — by recovering each candidate's linear score and replaying both
policies. The `identity` row reproducing 0.966954 exactly is what says the replay is
faithful.

| Policy | Score | MRR | MTTC | Turn-1 hits | Delta |
| --- | --- | --- | --- | --- | --- |
| linear only | 0.966954 | 0.9688 | 2.185 | **44** | — |
| LLM as final reranker | 0.962154 | 0.9722 | 2.475 | 13 | −0.0048 |
| 7th feature, `w=0.01` | 0.967054 | 0.9688 | 2.180 | 44 | +0.0001 |
| 7th feature, `w=0.05` | 0.966754 | 0.9688 | 2.195 | 42 | −0.0002 |
| 7th feature, `w=0.10` | 0.967154 | 0.9722 | 2.225 | 37 | +0.0002 |
| 7th feature, `w=0.20` | 0.965154 | 0.9722 | 2.325 | 27 | −0.0018 |
| 7th feature, `w=0.30` | 0.963754 | 0.9722 | 2.395 | 19 | −0.0032 |
| 7th feature, `w=0.50` | 0.962754 | 0.9722 | 2.445 | 16 | −0.0042 |
| 7th feature, `w=1.20` | 0.962254 | 0.9722 | 2.470 | 14 | −0.0047 |
| 7th feature, `w≥2.00` | 0.962154 | 0.9722 | 2.475 | 13 | −0.0048 |

**The feature form strictly dominates the reranker form, and its optimum is zero.** It
is a generalisation of the reranker — at `w≥2` the two are identical, same score and
same 13 turn-1 hits — and every smaller weight is better. So the feature form is the
one to reach for if this is ever retried. But there is no peak: the two positive blips
(`w=0.01`, `w=0.10`) are ±1 session against a ~0.002 noise floor, and from `w=0.20` the
curve decays monotonically. A tuner maximising it lands at `w≈0`.

The turn-1 hit column is the mechanism, and it is the column to watch: 44 → 37 → 27 →
13 as the weight lets the model overcome progressively larger linear margins. Every
weight that buys the model influence spends turn-1 conversions to get it.

> **Limitation, in the direction that flatters the model.** Simulated reranker −0.0048
> against the measured −0.0058. The gap is accounted for: when the LLM breaks a turn-1
> hit the session reaches turn 2 and makes a second call that is not in the cache,
> because the cache was captured from the baseline run where those sessions had already
> ended at turn 1. The simulator falls back to identity there. So every LLM row above is
> optimistic by roughly 0.001, more so at large `w`. The −0.0058 is a measurement; the
> sweep is an estimate.

### Why it failed — and why prompting was not the fix

Not a prompt-engineering problem. The turn-1 decision is close to **information-free**.
The shopper quotes one phrase lifted verbatim from the target's own listing, and every
candidate in the tie group contains that same phrase:

| Session | Evidence | Tied | Linear #1 | Target |
| --- | --- | --- | --- | --- |
| `public_0002` | `Buckle closure` | 108 | BULLIANT Men's Belt | Hide & Drink Men's Belt |
| `public_0003` | `Stainless Steel Band` | 4 | Casio **Ladies** Watch | Casio **Men's** Watch |
| `public_0005` | `leather` | 86 | Columbia 100% Leather Boot | GLOBALWIN boot, no leather bullet |

Nothing in "Buckle closure" separates one leather belt from 107 others, and no amount of
reasoning recovers what the message does not contain. The oracle ceiling is real but it
requires *knowing* the answer, not deriving it.

Worse, the system prompt told the model *"popularity is not a reason to rank a product
higher"* — and popularity is the single most predictive signal available here, because
the public set oversamples popular targets (`tools/run_dev.py` docstring: 24.5% have
≥3000 ratings). The linear reranker already exploits that prior. The model was
instructed to discard it and given nothing to replace it with.

**The generalisable lesson.** A tie group is not a ranking problem waiting for a smarter
ranker. It is a set of candidates the evidence genuinely does not separate, and the
right move inside one is to bet the best available *prior* — which is what
popularity + velocity + BM25 position already are — not to spend a model call
re-deriving a coin flip. This is the same finding as the provenance-features entry in
§3, arrived at from the opposite direction.

### Cost, had it shipped

167,409 tokens and **~40 minutes per public-set evaluation** against ~6 seconds for the
deterministic path — in exchange for −0.0058. The run costs more than the decision-set
benchmark suggested (158 calls, p50 3.80 s) because breaking turn-1 hits lengthens
sessions, which produces more gated turns, which produces more calls.

### What was kept

The layer stays in the tree, off, because the harness is what makes the negative result
reusable: a stronger model can be dropped in and measured against the same 158
decisions in minutes. `starter/llm_rerank.py` also carries an Anthropic backend that
was written but never exercised — no key was available, and the local-only path was the
one chosen. `tests/test_llm_rerank.py` pins the two invariants the design rests on
(permutation, and fallback on every failure) so they cannot rot while the layer is dark.

## 7. Tried and abandoned

**A local LLM (Ollama) choosing `ask_attribute`** — *abandoned, path is dead code.*
`_make_client()` returns `None` unconditionally before its body ever runs, so the agent
is standard-library-only and deterministic, which is what official scoring expects.
The deterministic answer (`other`, see §4) dominates anyway, so the model had nothing to
add; the plumbing (warm-up call, on-list validation, exception fallback, token
accounting) is left in place but inert.

**Fuzzy bucket matching** — never built; the exact lookup hits 200/200 (§2).

**Tie-gated exposure, top-3 on turn 2, turns 1–3** — all measured, all worse (§4 table).

**Served-item blacklist** — rejected in favour of a paging offset (§2).

**Provenance features** (`Date First Available`, `average_rating`, store name) — built,
measured, reverted (§3). Real catalog-wide priors, no value inside a tie group.

**User-profile rating affinity** (`|average_rating − average_prior_rating|`) — rejected
without building (§8). On the 134/200 sessions with a prior of 5.0 it is ordinally
identical to the `average_rating` feature already reverted above; elsewhere it is noise.

**Preference tags as a low-weight OR bag** — built, measured, shipped off (§8). A real
67th-percentile catalog-wide prior that is a coin flip inside the pool.

**LLM reranking of the top 10** — built, measured, shipped off (§6). End to end it costs
**−0.0058** (0.9670 → 0.9611), all of it Efficiency; hit rate holds at exactly 1.000,
which is the permutation invariant doing its job. Blending the model in as a weighted
7th feature instead of an override is strictly the better architecture and its optimum
weight is zero. The tie groups it was meant to break are information-free, not
mis-ranked.

**Per-scenario reranker weights** — measured, rejected (§10). Fitting one weight vector
per scenario and routing each session to its own is worth +0.0004 across four holdout
cuts, and every refit — routed or global — scores below the shipped weights out of
sample.

## 8. Preference tags as a low-weight OR bag — *built, measured, off*

`user_profile` reaches `reset()` and, until this experiment, nothing read a single field
of it. A field-by-field audit says only two carry information at all: `purchase_frequency`
is the string `"3-4 prior purchases"` on 200/200 public sessions, `rating_style` is a
strict function of `average_prior_rating` (1/2/3 → critical, 4 → mixed, 5 → usually
positive, verified on all 200), and `summary` restates both. That leaves
`average_prior_rating` and `preference_tags`.

**`average_prior_rating` was rejected without building it.** The proposal is to penalise
a candidate whose catalog `average_rating` diverges from the shopper's prior. 134 of the
200 public sessions have a prior of exactly 5.0 and no catalog row is rated above 5.0, so
on two-thirds of the set `−|r − 5.0|` is *monotone increasing in r* — ordinally identical
to the `quality` feature already measured at +0.00059 (CI [−0.0006, +0.0022], noise) and
reverted in sec.3. On the other 66 sessions it carries nothing: mean catalog percentile of
the target is 0.471 / 0.471 / 0.450 / 0.569 for priors 1/2/3/4, against 0.500 for a
feature with no signal. `r(prior, target average_rating) = 0.182` but the per-prior target
means are non-monotone (prior 1.0 → 4.393, prior 3.0 → 4.300, prior 5.0 → 4.413, catalog
mean 4.087), so there is no affinity to exploit — only the weak "targets are rated above
catalog average" prior that sec.3 already priced at zero.

**`preference_tags` was built.** `TAGS_ENABLED` / `TAGS_W` / `TAGS_NORMALIZE`, spliced
into `FEATURE_NAMES` ahead of BM25 (which must stay last for the dominance pruning). The
feature is the fraction of the shopper's tags present in the candidate's text, score-only,
excluded from `decisive` so the tie count the exposure gate reads is untouched. Tags the
shopper has already quoted as a constraint are dropped from the bag — evidence already
owns those, and counting them twice would let the bag amplify a signal it did not earn.

The marginal prior is real, and better than the `average_rating` one that motivated sec.3's
provenance entry:

| | target | average catalog row |
| --- | --- | --- |
| fraction of the shopper's tags present | 0.445 | 0.265 |

which puts the target at the **67th percentile** of the catalog. Coverage is broad —
`fit` appears in 16,930 of 50,000 rows, `material` 13,008, `comfort` 15,873, `style`
14,260; `durability` 1,963, `performance` 2,572, `weather` 1,644, `warmth` 875.

It is worth nothing in the pool, and the trace says why before any weight is chosen.
Rival-minus-target deltas over the 51,008 candidates that survive dominance pruning
(1,000 sessions = dev-800 + public-200):

| feature | mean delta | rival beats target | target beats rival | equal |
| --- | --- | --- | --- | --- |
| popularity | −0.3441 | 6.8% | 93.1% | 0.1% |
| velocity | −0.3194 | 6.4% | 93.5% | 0.0% |
| **tags** | **+0.0038** | **35.3%** | **29.0%** | 35.8% |

A coin flip leaning the wrong way. The sweep follows, monotone from the smallest weight
tested, on every split and both objectives:

| `TAGS_W` | raw all | raw train | raw holdout | public all | public train | public holdout |
| --- | --- | --- | --- | --- | --- | --- |
| **0.00** | **0.960758** | **0.959394** | **0.963768** | **0.964440** | **0.963851** | **0.965723** |
| 0.02 | 0.960099 | 0.958682 | 0.963223 | 0.963862 | 0.963236 | 0.965226 |
| 0.05 | 0.959660 | 0.958426 | 0.962383 | 0.963439 | 0.962870 | 0.964680 |
| 0.10 | 0.959357 | 0.957713 | 0.962983 | 0.963280 | 0.962327 | 0.965356 |
| 0.20 | 0.958950 | 0.957492 | 0.962166 | 0.963147 | 0.962512 | 0.964532 |
| 0.40 | 0.956428 | 0.954756 | 0.960113 | 0.960317 | 0.959505 | 0.962088 |
| 1.00 | 0.947817 | 0.947502 | 0.948511 | 0.951945 | 0.951734 | 0.952406 |

**Word-boundary matching does not rescue it.** Substring lets `fit` collect `outfit` and
`benefit`, so `TAGS_WORD_BOUNDARY` was added and the whole trace recaptured. It cleans the
delta up — +0.0038 → −0.0071, i.e. marginally in the target's favour — and changes nothing:
every weight still loses on `all` and on `train` (0.960758 → 0.960114 at 0.02, 0.959415 at
0.15, 0.951139 at 1.00). Holdout shows a lone +0.0007 bump at w = 0.20 with train −0.0018
underneath it, which is what holdout noise looks like.

**Why the public 200 disagrees, and why it is wrong.** Run end to end on the public set
alone, `TAGS_W = 0.05` scores 0.96778 against a 0.96695 baseline — +0.0008, all of it MRR,
MTTC unchanged. Five sessions move at that weight and the delta is one of them:
`public_0145` goes rank 2 → 1, worth 0.3 × 0.5/200 = 0.00075. `public_0144` contributes
5 → 4 (0.00004); `public_0032`, `public_0106` and `public_0174` are turn churn that nets
to zero. The neighbouring grid points are +0.00008 (0.02), −0.00005 (0.08) and −0.0008
(0.10), and by 0.08 it has started giving sessions back (`public_0052` rank 1 → 2,
`public_0099` 3 → 4). A single unflanked bump on 200 sessions against a monotone decay on
1,000 is a coin landing. **This is the diagnostic worth keeping from the whole
experiment**: any candidate feature should be asked for its rival-minus-target delta
distribution on the trace *before* it is scored, because that table predicted the sweep
and the public set did not.

**Shipped `TAGS_ENABLED = False`, not deleted.** The public set populates
`preference_tags` with eight generic words; a private evaluator that put something specific
there would be a different feature on the same plumbing. Turning the flag off empties the
bag, which makes the column constant — restoring the ranking exactly (per-session
identical to `results.json`, 0.966954) *and* the trace's dominance pruning. That second
part is not free: a live bag at `TAGS_W = 0.0` still carries 51,008 candidate rows against
the baseline's 35,296, a 45% tax on every future tuning run for a weight of zero. Same
reasoning that reverted the provenance features in sec.3, same fix as the sec.6 LLM
reranker — the code stays, the flag is off.

## 9. Where the remaining headroom is

At 0.9670 public with hit rate 1.000 and MRR 0.969, ~0.033 remains and **~0.024 of it is
Efficiency** — converting on turn 1 instead of turns 2–3. Recall is done; ranking is
nearly done. It is easy to keep tuning the reranker for MRR and buy almost nothing,
because MRR is only 30% of the score and already at 0.97. The diagnostic worth printing
for any new idea is the turn distribution (how many sessions convert on turn 1 vs 2 vs
3), not just hit rate.

**That headroom is now bounded, and most of it is unreachable.** §6 measures the oracle:
pulling the target to rank 1 of the slate the reranker already produces is worth
+0.0216, and *everything else* — better retrieval, more features, more turns — is worth
at most the remaining ~0.011. But the oracle is an oracle. The 55 decisions the linear
reranker gets wrong are dominated by tie groups where the shopper's one quoted phrase
appears identically in every candidate, so the information to order them is not present
in the session at all. Before building anything aimed at this headroom, check on the
decision set (`tools/llm_bench.py capture`) whether the sessions it targets are actually
separable; §6 is the record of what happens when they are not.

**The weights themselves are bounded far more tightly than that.** §10 prices a
per-session oracle over weight vectors — every session ordered as well as any
non-negative vector could order it — at **+0.0091** over the pooled 1,000 sessions
(+0.0076 on public), with hit rate unchanged. Any reranker change that only moves
weights, including one vector per scenario, is competing for less than a point; the
remaining headroom is in turn policy and in what the slate is built from, not in how it
is scored.

**Caveat when comparing any two runs.** `Matcher.features()` emits the incoming BM25
rank as `-(position / depth)` with `depth = len(pool)`, so that feature is scaled by how
large the candidate pool happens to be. Any retrieval change that shrinks the pool — the
bucket prefilter cut the median dev pool from 2 to 1 and the mean from 43 to 27 —
silently changes that feature's effective weight even though no weight was touched.
Re-run `tools/tune_reranker.py` after a retrieval change before treating the measured
delta as final. (The bucket prefilter's +0.0051 on dev-800 was measured against weights
tuned *without* it, so it is probably an underestimate.)

## 10. Per-scenario reranker weights — *measured, rejected*

One weight vector per `scenario_type`, each session routed to its own, instead of the
single global vector. Traced fresh off the current tree (`trace` over public-200 +
dev-800 = 1,000 sessions, 2,793 verified turns, **0 mismatched**, 35,296 carried
candidate rows, live replay 0.960758 raw), then fitted on the train split and scored on
the holdout it never selected on, under four `--split-salt` cuts. The fit is
`tune_reranker`'s own machinery — `build_splits`, then 500 random probes plus three
coordinate-descent refinements — run once over all training sessions and once over each
scenario's slice of them, so the two arms differ only in what they were fitted on:

| salt | live weights | one global refit | per-scenario routed | routed − global |
| --- | --- | --- | --- | --- |
| a | **0.965254** | 0.960806 | 0.959740 | −0.00107 |
| b | **0.967087** | 0.966183 | 0.964153 | −0.00203 |
| c | **0.964937** | 0.960069 | 0.964899 | +0.00483 |
| d | **0.965187** | 0.964017 | 0.963876 | −0.00014 |

Routing is +0.0004 on average and positive on one salt in four. The louder result is the
first column: **the shipped weights beat both refits on every cut.** The search already
overfits with six free weights at this sample size, and four vectors is four times the
parameters against the same evidence. Salt c's apparent win is mostly Boundary, fitted
on ~15 training sessions and scored on 6.

Three reasons there was nothing to find, each worth reusing before the next variant of
this idea:

**The whole weight axis is nearly exhausted.** A *per-session* oracle over weights —
each session ordered as well as any non-negative vector could order it, which is
strictly more powerful than any router and not reachable by one — is worth +0.0091:

| slice | n | live | per-session oracle | ceiling |
| --- | --- | --- | --- | --- |
| all | 1,000 | 0.960758 | 0.969867 | **+0.00911** |
| buying | 400 | 0.971736 | 0.983950 | +0.01221 |
| browsing | 400 | 0.961160 | 0.969127 | +0.00797 |
| intent_override | 150 | 0.936600 | 0.942152 | +0.00555 |
| boundary | 50 | 0.942200 | 0.946267 | +0.00407 |
| public-200 | 200 | 0.966954 | 0.974575 | +0.00762 |
| dev-800 | 800 | 0.959210 | 0.968690 | +0.00948 |

Hit rate is 0.9990 under the oracle too — identical to live. It is MRR 0.9553 → 0.9748
and MTTC 2.27 → 2.10, and no scenario has a materially larger reachable gap than the
others. This is a much tighter bound than §9's: it prices *every remaining weight
change*, global or routed, at under a point.

**The effective sample size collapses per scenario.** Only 500 of the 1,000 sessions
contain a turn any weight can decide at all; the rest are traced as constants because
the target wins or loses regardless.

| scenario | sessions | weight-decidable | contested rows |
| --- | --- | --- | --- |
| buying | 400 | 261 (65%) | 13,725 |
| browsing | 400 | 185 (46%) | 18,008 |
| intent_override | 150 | 35 (23%) | 585 |
| boundary | 50 | 19 (38%) | 2,978 |

After the 80/20 split that is ~28 Override and ~15 Boundary sessions to fit six weights
against a step-function objective. (Override's count fell from 123 to 35 when
`OVERRIDE_RESETS = False` landed in §5 — keeping the constraints means the target now
usually wins outright.)

**A feature that is inert in a scenario constrains nothing, so one vector already serves
both.** Share of contested rival rows where each feature separates rival from target
(rival better / target better):

| scenario | exact | loose | position | popularity | velocity | tags | bm25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| buying | 0.0 / 18.3 | 0.1 / 4.9 | 7.8 / 14.3 | 9.6 / 90.3 | 8.0 / 91.9 | 0 / 0 | 86.2 / 13.8 |
| browsing | 0.0 / 44.0 | 0.1 / 0.6 | 0.3 / 1.4 | 6.9 / 93.0 | 7.1 / 92.8 | 0 / 0 | 93.0 / 7.0 |
| intent_override | 0.0 / 29.2 | 0.5 / 1.2 | 0.5 / 6.5 | 15.0 / 85.0 | 18.6 / 81.4 | 0 / 0 | 87.0 / 13.0 |
| boundary | 0.0 / 28.2 | 0.3 / 1.1 | 0.3 / 1.9 | 4.3 / 95.7 | 5.6 / 94.4 | 0 / 0 | 97.0 / 3.0 |

`exact` is never worse for the target on any contested row in any scenario — it can only
help, which is why it is the tuner's scale anchor. `loose` and `position` separate almost
nothing outside Buying, so their global weight is free to serve the sessions that do use
them. Routing can only pay where two scenarios want *opposite* values of a feature live
in both, and the ones live everywhere — `popularity`, `velocity`, `bm25` — are already
tie-breaks (§3). `tags` is identically zero everywhere because §8 shipped it off; the
search still assigned it 3.267 for Buying on salt b, which is the overfitting visible in
a coordinate that provably cannot matter.

**Separately, the label is not observable.** The agent is never told `scenario_type`; it
infers intent from the simulator's templated opening (§1), Override is not knowable until
turn 3–4 and Boundary not until the shrug. The spec reserves the right to paraphrase
those messages, so a weight table keyed on that inference would multiply the exposure of
a parse that is already template-locked.

## Tooling built along the way

| Tool | Why it exists |
| --- | --- |
| `tools/run_dev.py` | Shards dev sessions across cores (~25 min → minutes), and scores twice: raw, and reweighted so each stratum contributes its public-set share. Reports Kish effective sample size, because reweighting a large run to a distribution concentrated in ~180 catalog rows buys far less precision than the raw count suggests. |
| `tools/make_dev_set.py` | Builds the stratified dev set. An exhaustive set is 37% items the public set never samples (TVD ≈ 0.86 vs 0.27 for the stratified 800), so its raw score answers a different question. |
| `tools/tune_reranker.py` | `trace` / `replay` / `search` for offline weight tuning (§3). |
| `tools/llm_bench.py` | `capture` writes the decision set the LLM gate would fire on (no model calls); `replay` scores a model against the linear top-1 on it. On the exposure turns top-1 accuracy *is* the score difference, so a prompt change costs a few hundred short prompts instead of a full evaluation run. |
| `tools/llm_simulate.py` | Replays cached model answers under a different architecture — pick-as-override against pick-as-a-weighted-feature — without calling the model again. Recovers each candidate's linear score from the live `features()` call rather than recomputing it, because the BM25 term is normalised by pool size. The identity policy reproducing the real score is the check that the replay is faithful. |
| `tools/llm_policy.py` | Sweeps acceptance policies over answers `replay --cache` already collected, so testing a new promotion ceiling costs zero calls. |
| `tools/oracle.py` | Single-session inspection: `python3 tools/oracle.py --policy agent --dataset dev/dev_set.jsonl --show dev_0020`. |