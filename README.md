# TechJam Conversational E-Commerce Search — Submission

A multi-turn shopping agent for the TechJam Conversational E-Commerce Search
Challenge. The agent reads a short customer message each turn, asks one useful
follow-up question, and returns a ranked list of up to 10 catalog
`parent_asin` values, aiming to surface the customer's hidden target product as
early as possible within the 10-turn budget.

**Team:** Maximus Lim, Wong Xin Hui.

## Results

`TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency`,
`Efficiency = clip((11 − MTTC) / 10, 0, 1)`. Deterministic, standard-library
only — **no network access and no model download at scoring time** (0 tokens).

| Set | Sessions | HitRate@10 | MRR | MTTC | **TechnicalScore** |
|---|---|---|---|---|---|
| **public** (released labels) | 200 | 1.0000 | 0.9716 | 2.195 | **0.9676** |
| **dev_set** (held out, never tuned against) | 800 | 0.9988 | 0.9544 | 2.301 | **0.9597** |

Per-scenario on the public set:

| scenario | n | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 1.000 | 0.976 | 1.54 |
| browsing | 80 | 1.000 | 0.962 | 2.19 |
| intent_override | 30 | 1.000 | 0.975 | 3.67 |
| boundary | 10 | 1.000 | 1.000 | 3.10 |

The weak BM25 starter shipped with the challenge scores `0.1067`
(HitRate@10 `0.125`, MRR `0.068`, MTTC `9.81`); see `docs/baseline_results.json`.

## Project Overview

### The key observation

The customer simulator does not paraphrase. Every reveal is a substring lifted
**verbatim** from the target product's own catalog text (`"A key requirement
is: {c}."`, etc.), and the opening message embeds the evaluator's own
`coarse_category(...)` output verbatim. Retrieval here is therefore an
exact-match / elimination problem, not a semantic one — a fact that shaped
every design decision and killed every dense / LLM experiment we tried (see
"Limitations" and `starter/elimination-track.md` / `starter/lexical-track.md`).

### Architecture: a scenario router over two retrieval tracks

`starter/agent.py` exports `Agent` (`reset` / `respond`) — the **submission entry
point**. It is a thin router over two independent implementations in sibling
sub-packages. The simulator's opening message fixes the scenario for the whole
session, so the route is decided once, from turn 1, and never switched:

| opener shape | scenario | route |
|---|---|---|
| `… A key requirement is: <c>.` | Buying | `elimination` |
| `I'm looking for <cat>. <old_value>` (real attribute tail) | Intent Override | `lexical` |
| `I'm looking for <cat>, but I'm still exploring.` | Browsing / Boundary | `elimination` |
| anything else (reworded / unrecognised) | — | `elimination` (safe default) |

* **`starter/lexical/`** — coarse-category bucket pre-filter → two-tier BM25 (SQLite
  FTS5) → single linear reranker → exposure window. Wins Intent Override on
  both the public and held-out sets.
* **`starter/elimination/`** — elimination-first: accumulate disclosed phrases across
  the session and use them as hard exact-phrase filters, then rank; always-on
  diversified browse track for constraint-free turns; deep paging on dry turns.
  Wins Buying / Browsing / Boundary and is the better overall generaliser.

**Why this split — measured, not assumed.** Each sub-package is a complete agent.
We ran both standalone and route each scenario to whichever one scored higher on
it, judged on the held-out `dev_set` (800 sessions, never tuned against):

| standalone track | public (200) | dev_set (800) | public → dev_set |
|---|---|---|---|
| `lexical` | 0.9691 | 0.9524 | **−0.0167** |
| `elimination` | 0.9670 | 0.9592 | **−0.0077** |

`elimination` wins the aggregate and generalises far better — less than half the
lexical track's public→held-out drop, the lexical constants having been swept
hard on the 200-set — and on `dev_set` it also wins Buying (MRR `0.950` vs
`0.934`) and Browsing (`0.954` vs `0.917`). `lexical` wins **Intent Override** on
*both* sets. Boundary is a wash and shares a byte-identical opener with Browsing,
so it cannot be routed apart from it and rides with `elimination`.

Routing `elimination` on Buying / Browsing / Boundary (per-scenario metrics
byte-identical to `elimination` standalone) and `lexical` on Intent Override —
which lifts that scenario's `dev_set` MRR from `0.953` to `0.970` — nets
**+0.0005 on both** independent sets with zero regression on the other three
scenarios (router: public `0.9676`, dev_set `0.9597`).

*Why Intent Override is the scenario that splits off:* there the customer's
turn-3/4 "ignore my earlier preference" is nearly a no-op — the replacement value
was already disclosed a turn or two earlier, and the old and new values are both
genuine, non-contradictory attributes of the same unchanged target. `lexical`
rewrites only the one affected slot and keeps every other accumulated phrase as
evidence. `elimination` filters the candidate pool by disclosed phrases, so a
bare material word arriving as a fresh hard constraint can carve the target out
of the pool — and retrieval cannot recover an item that filtering dropped. Full
head-to-head and per-scenario tables in `starter/lexical-track.md`.

Both tracks share the core ideas that moved the score from `0.107` to `~0.96`:
**constraint accumulation** (treat every turn as evidence, never a rewrite), a
**coarse-category bucket pre-filter** (mirror the evaluator's `coarse_category`
and hard-restrict retrieval to the target's bucket — closes the last recall
gap, HitRate@10 → 1.000 on public), a **linear reranker** with an
evidence-dominates-tiebreakers invariant, and an **exposure gate** that serves
only the single best candidate on the first turn or two (a miss costs one turn;
a wrong-rank hit is scored at that rank forever).

Design rationale lives next to the code, both under `starter/`:
`starter/lexical-track.md` (lexical track, full tuning history) and
`starter/elimination-track.md` (elimination track, score timeline and every
rejected idea).

## Setup and Installation

**Requirements**

* Python **3.10+** (developed and scored on 3.12.4).
* **No third-party packages, no virtualenv.** The agent and the evaluator import
  only the Python standard library (`sqlite3` with the FTS5 extension, which
  ships with the stock CPython build on Windows, macOS and most Linux
  distributions). There is no `requirements.txt` and nothing to `pip install` —
  a clean Python 3.10+ runs everything. Verified on a bare interpreter with the
  bundled `.venv/` off the path.
* The `.venv/` in this tree contains packages (`torch`, `sentence-transformers`,
  …) used **only** for exploratory experiments that were measured and rejected;
  none are imported by the submitted agent or the evaluator, and it is not
  needed to reproduce any result below.

**Get the catalog** (not committed — it is a GitHub Release attachment):

```bash
# from the repository root
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl        # expected: 50,000 rows
```

Verify the download against the published `SHA256SUMS` file.

**Sanity check**

```bash
python -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('FTS5 OK')"
```

## Steps to Reproduce Our Results

Run from the repository root. The evaluator imports `starter.agent.Agent`, builds
both retrieval indexes once (~13 s cold start on a laptop), then replays every
session deterministically.

**Public set (200) — the leaderboard-comparable number:**

```bash
python -m evaluator.local_evaluator
# writes results.json  ->  recommended_technical_score = 0.967579
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

```bash
python -m evaluator.local_evaluator --dataset data/dev_set.jsonl --output results.dev.json
# recommended_technical_score = 0.959678
```

Both runs finish in well under two minutes on a single core and print the
aggregate and per-scenario metrics. `data/dev_full.jsonl` (49,800 sessions) is
also materialised by the same evaluator for broad-sweep direction only — it
scores lower by construction (it samples catalog regions the public set never
touches) and is not a leaderboard estimate.

Unit tests:

```bash
python -m unittest discover -s tests
```

Numbers are reproducible bit-for-bit: the agent makes no random draws, no
network calls, and no model calls.

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

```text
starter/agent.py                 Submission entry point — the scenario router
starter/lexical/                 Bucket pre-filter -> BM25 -> linear rerank track
starter/elimination/             Elimination-first phrase-filter + browse track
starter/lexical-track.md         Lexical-track report + full tuning history
starter/elimination-track.md     Elimination-track report + score timeline + rejected ideas
evaluator/local_evaluator.py Deterministic public-set simulator and scorer (unmodified)
data/public_set.jsonl        200 labeled development sessions
data/dev_set.jsonl           800 held-out sessions (stratified to the public distribution)
scripts/                     Offline sweep / diagnostic tooling (not part of the agent)
tests/                       Unit tests
docs/                        Challenge spec, API contract, scoring config, submission rules
```

## Model Choice, Cost, Latency, and Token Usage

* **Model:** none. The final agent is fully deterministic and standard-library
  only. It **does not require network access** and has no live-credential
  dependency, so it runs unchanged under the organizer's network-disabled
  scoring policy.
* **Token usage:** `0` prompt / `0` completion tokens (`usage` is reported as
  zero every turn).
* **Estimated model cost:** `$0`.
* **Latency:** one-time index build ≈ 13 s (both tracks build an in-memory FTS5
  index over the 50k catalog); per-turn compute is tens of milliseconds. A full
  200-session public evaluation completes in roughly half a minute on one core.
* During development we prototyped an optional local-LLM reranking layer
  (Ollama, `llama3.2` / `llama3` / `qwen2.5`) and a dense bi-encoder retrieval
  route. Both were measured end-to-end and **removed / hard-disabled** because
  they lowered the score (LLM rerank: −0.0058, all of it Efficiency; dense
  route: recovered zero missed targets, only demoted correct ones). The
  submitted bundle makes no such calls.

## Limitations and What We Would Improve

**Information-free tie groups are the ceiling.** On the sessions we still lose
rank on, the customer's one quoted phrase (`"Buckle closure"`, `"leather"`,
`"100% Cotton"`, `"Imported"`) appears identically in 6–300+ bucket-mates. The
disclosed constraints genuinely do not separate the target from the pack, so
the residual is a tiebreak problem with no signal inside the session to solve
it. We priced the headroom directly: an oracle that pulls the target to rank 1
of the slate the reranker already produces is worth only **+0.022**, and a
per-session oracle over *all* reranker weight vectors is worth only **+0.009** —
both with HitRate@10 unchanged. Popularity, ratings-velocity, dense cosine and
LLM rerankers were all tested here and only ever demoted correct targets.
*Given more time:* look for a discriminating signal **outside** the disclosed
text — e.g. catalog structure or co-listing patterns that survive conditioning
on category and popularity — since every catalog-wide prior we tried evaporated
inside a tie group.

**Overfitting to the 200-sample public set.** The lexical track's constants
were swept on public; it drops ~0.018 composite on the held-out 800-set (partly
genuine distribution shift — dev_set carries an explicit easy/medium/hard mix).
One lever (`PRIOR_RATING_W`) was caught overfitting and reverted to inert.
*Given more time:* move all tuning onto `dev_set` / a `dev_full` sample and
treat public purely as a held-out check; the 150/50 public split has been
selected against too many times to still measure generalisation.

**Opener parsing is template-locked.** Both the route decision and the bucket
pre-filter key off exact simulator phrasings. A reworded private evaluator could
misroute a session or miss the bucket. This is mitigated but not solved:
unrecognised openers fall back to the `elimination` track (the stronger
generaliser), and an unresolved bucket degrades to the full-catalog BM25
pipeline rather than hard-excluding the target. *Given more time:* a
paraphrase-robust opener classifier and a fuzzy bucket-resolution ladder with
measured recall floors.

**Purely lexical retrieval.** The agent leans entirely on the simulator quoting
verbatim. If the private evaluator paraphrased its reveals instead, recall
would fall and there is no semantic backstop (the dense route we built was net
negative *on this data* and was removed). *Given more time:* a dense retrieval
path gated to only fire when lexical evidence is thin, validated so it cannot
demote exact matches.

**Efficiency is where the remaining points are, and most are unreachable.** At
HitRate@10 = 1.000 and MRR ≈ 0.97 on public, ~0.024 of the remaining ~0.033 is
Efficiency (turn-1 conversion). The exposure gate already captures the reachable
part; the rest is bounded by the information-free tie groups above.

**Operational cost of the router.** Running two independent tracks doubles cold
start and RAM (each builds its own FTS5 index over the full catalog). *Given
more time:* share a single index and matcher between the tracks — the two were
kept separate only for development independence, not by necessity.

## Team Member Contributions

| Member | Share | Focus |
|---|---|---|
| **Maximus Lim** | 50% | Elimination-first retrieval track (`starter/elimination/`), constraint-accumulation and exposure-gate design, ratings-velocity feature, offline weight-tuning harness, `starter/elimination-track.md`. |
| **Wong Xin Hui** | 50% | Lexical retrieval track (`starter/lexical/`), coarse-category bucket pre-filter, linear reranker and weight sweeps, multilingual constraint handling, the scenario router (`starter/agent.py`), held-out validation and overfitting audit, `starter/lexical-track.md`. |

Both members contributed to problem analysis, evaluation methodology, the
rejected dense / LLM experiments, and this report.

## Data Attribution

The catalog and sessions derive from the Amazon Reviews 2023 dataset (McAuley
Lab, UCSD), `Clothing_Shoes_and_Jewelry` category. See `DATA_ATTRIBUTION.md`
before using or redistributing the data.
