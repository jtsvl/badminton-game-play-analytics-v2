# Phase-1 measurements — M1 & M3 results (M2 handed off)

**Datum:** 2026-07-13 · Status: measured, throwaway scripts under `worker/scratch/stroke_v2/`
(not production code, not wired into the CLI). Governing spec: `stroke-classification-v2.md`
§10 (measurement plan), §9 (gate stack G0–G5). Scripts read data via three symlinks into the
main repo checkout (`videos/`, `worker/worker/.labeling/`, `worker/worker/.cache/` — gitignored,
per-worktree data doesn't exist in `.claude/worktrees/stroke-classification-v2` by default).

## Spec wording fix

§10's M1 row reads "Audio onsets × 738 known hits (**1 game**)" — but 738 is the sum across all
5 Session-2 games (151+207+167+144+69), confirmed directly from `tasks_Session2_Game_{1..5}.json`
and cross-checked against `stroke_labels.parquet` + `hit_detection_eval.parquet` (554+141+43=738).
No document pins which single game "(1 game)" was meant to mean. **Ran against all 5 games** (your
call) — the spec's "(1 game)" parenthetical should be corrected to "(5 games)" or dropped.

---

## M1 — audio onsets vs. 695 known hits, all 5 games

**Question:** does the hall's audio give usable hit-timing, and how often does a neighbor
court's impact land inside our detection window? (Biggest open unknown per spec §13 risk 1;
gates the whole Stage-1 hit-detector design.)

**Method:** audio extracted directly from the raw 4K sources (`videos/Session2_Game_{1..5}.mp4`,
not the proxy, to avoid a second lossy AAC pass); onset candidates via `librosa.onset.onset_detect`
(standard spectral-flux, backtrack=True — the "SuperFlux class" the spec names, though not a
byte-for-byte SuperFlux/madmom implementation — see caveat below). For each of the 695 known hits
(`stroke_labels.parquet`, both confident and "Unsicher" rows — a hit happened either way), searched
the intended ±150ms production window plus a ±400ms diagnostic window (since `t_s` itself is
heuristic-derived with known ±0.2s drift). Also ran the same window check against the 43 `no_hit`
false-positive candidates (`hit_detection_eval.parquet`), and computed onset rate during
rally-idle vs. rally-active time per game (idle-time onsets are provably foreign — our own court
has no play there).

**Result — pooled across 5 games, n=695 hits:**

| Metric | Measured | G0 target | Read |
|---|---|---|---|
| Median \|Δt\| | **61 ms** | ≤ 25 ms | 2.4× over |
| Recall @ ±50ms | **26%** | ≥ 0.90 | far short |
| Precision @ ±50ms | **40%** | ≥ 0.90 | far short |
| Recall @ ±150ms window (any onset at all) | **65%** | — (first recall estimate) | 1 in 3 hits has *no* onset candidate nearby at all |
| Coincidence probability (≥2 candidates in window) | **20%** | reported separately | real, not rare |
| False-onset rate in `no_hit` candidate windows | **56%** | — ("falscher Onset gewählt" proxy) | over half the video heuristic's false positives still *sound* like a hit |
| Idle-court onset rate | 1.0–1.9 onsets/s (per game) | — | ~40–65% of the in-rally rate — the hall is loud even when our court rests |
| Signed offset (onset − t_s) | +16 to +34 ms mean, per game | — (M9 territory) | consistently positive, physically plausible (sound travel + backtrack latency), not a bug |

**Sanity checks:** clip filenames/timestamps line up with `t_s` exactly (spot-checked game 1's
first 8 hits); signed offsets are small and consistently positive across all 5 games rather than
scattered noise, which is what you'd expect from real sound-travel-time + onset-detector latency,
not a pipeline bug.

**Read against the gate:** this **triggers the spec's own contingency** (§13 risk 1: "if M1 finds
impacts inaudible, the video detector stands alone... and G0's ±50ms is renegotiated honestly").
It's not that impacts are silent — 65% of hits have *some* onset nearby — but generic spectral-flux
onset detection is nowhere near precise or selective enough: median timing error is 2.4× the
target, and over half of the video heuristic's own false-positive windows still contain a
plausible-looking onset. Two honest caveats before treating this as final: (1) this is a
best-effort spectral-flux detector, not literally SuperFlux/madmom's log-frequency variant with
vibrato suppression — a proper implementation might close some of the precision gap, though the
65% recall ceiling (hits with *zero* candidate nearby) is a harder SNR floor that better picking
alone won't fix; (2) `t_s` itself carries known ±0.2s heuristic drift, which inflates the measured
spread somewhat independent of audio quality — M2's hand-counted precise hit times (once you do
them) would let this be re-measured more fairly.

**Flag for you:** G0's ±50ms/±25ms bar is not clearing on this evidence. I'd recommend treating
audio as validated-not-viable for now rather than sinking time into a full SuperFlux
reimplementation before M2 gives precise reference times to re-test against — your call on
priority.

---

## M3 — GBDT learning curve (140/280/554)

**Question:** how much does macro-recall improve per label-doubling, to replace the speculative
accuracy band in spec §12 with a measured slope before G2's target is frozen?

**Method:** reused `worker.stroke.gbdt`/`worker.stroke.train` unmodified; added a thin
`run_logo_capped` wrapper that subsamples each LOGO fold's training pool to a fixed cap
(stratified by class, fixed seed) before calling the same `xgb_fit_predict`. Same hyperparameters
as the existing 2026-07-12 spike run (`max_depth=4, eta=0.05, n_estimators=400, ppos=True,
seed=42`). Per-video confident counts (121/51/88/125/169) mean every fold's other-4-videos pool is
≥385 rows, so cap=554 never actually subsamples — I re-measured it anyway (rather than reusing the
old report) so all three points share one consistent protocol; it landed within ~1pt of the
original 0.536 run (0.524 vs 0.536 — small residual delta is expected, this run's per-fold xgboost
seed derivation differs slightly from the original, not a discrepancy in the data or method).

**Important correction while building this:** the table's "points per label-doubling" question is
about **macro-recall** (G2's actual gate metric — the mean of per-class recalls), not the overall
pooled accuracy the harness prints as its headline number. Under class imbalance (drop/lift/serve
dominate at n=554: 165/153/55 vs net/drive at 20/22) those two metrics diverge a lot, so I computed
both.

**Result:**

| n (labels) | mean accuracy | macro-recall | train–eval gap |
|---|---|---|---|
| 140 | 0.475 ± 0.052 | 0.374 | +52.5 pts |
| 280 | 0.487 ± 0.029 | 0.408 | +50.7 pts |
| 554 | 0.524 ± 0.035 | 0.434 | +47.7 pts |

Deltas: 140→280 +3.4pt macro-recall, 280→554 +2.6pt. Log2(labels)-linear extrapolation:

| | @ ~1400 labels (Session 2) | @ ~2100 labels (Session 3) | spec §12 band (mean accuracy) |
|---|---|---|---|
| mean accuracy | 0.553 | 0.568 | 0.60–0.70 / 0.65–0.75 |
| **macro-recall** | **0.476** | **0.493** | not stated as a separate target in §12 |

Per-class recall @ n=554 (G2 needs every class with n≥30: recall ≥0.40): clear 0.58, smash 0.15,
drop 0.60, **net 0.35 (n=20, below the n≥30 threshold so not yet gate-relevant)**, **drive 0.14
(n=22, same)**, lift 0.61, serve 0.62. Smash and drive are the two classes actively failing the
per-class floor once they cross n=30 — worth watching, not yet gate-triggering at this sample size.

**Read against the gate — two separate findings, don't conflate them:**

1. **Train–eval gap (+48 to +53 pts) blows through G2's ≤15pt memorization-alarm at every size
   tested.** This is expected and out-of-scope by spec's own design: GBDT is explicitly
   "labeling-assist + baseline only, never pipeline" (§5 Stage 3) and its harness runs "no early
   stopping — no honest validation set inside a LOGO fold" by design (its own docstring). The
   model that actually has to pass G2 is BST4 + ShuttleSet transfer + the imbalance/abstention
   levers, not this spike. **This number doesn't gate anything by itself** — it's a property of
   the deliberately-simple baseline, not new evidence about G2's reachability.
2. **Macro-recall from labels alone is a slow, sub-linear climb — 0.43 @554 → an extrapolated
   ~0.48–0.49 even at full Session-3 scale (~2100 labels), still below the 0.55 gate.** This is
   the GBDT-alone floor (no ShuttleSet pretraining, no FineBadminton transfer, no imbalance
   handling beyond class weights). Read charitably, it **validates rather than undercuts** the
   spec's existing Stage-3 design: "data strategy dominates model choice" already assumes the
   transfer/pretrain levers carry the real weight, not raw Session-3 label volume — this measured
   curve says that assumption is *necessary*, since volume alone plateaus well short of the bar.

**Flag for you:** I don't think this is grounds to lower G2's 0.55 macro-recall target — we
haven't yet measured what BST4+transfer actually does (that needs M4 plus real BST4 runs under the
new taxonomy, out of this round's scope). But I'd flag one spec-clarity gap: **§12's accuracy
table doesn't say whether its 0.55–0.75 band is mean accuracy or macro-recall** — under this
measured ~9pt gap between the two metrics at n=554, that ambiguity matters. Worth pinning explicitly
when the spec next gets touched.

---

## M2 — manual hit count (trainer-completed, analyzed)

15 rallies sampled (3 per game × 5 games, fixed seed, biased toward rallies with ≥3 shots),
watched and hand-counted by the trainer, hit timing recorded as a frame number per hit
(`manual_hit_frame`, counted from the start of each exported clip). Converted to absolute game
time via `frame / (60000/1001 fps) + (t_start_s - 1.0s pad)` — frame identity survives the 0.25×
export unchanged (`setpts` doesn't duplicate frames, only stretches how long each is shown).

**Method:** each manual hit was matched against the heuristic's own candidate hit times for that
rally (`stroke_labels.parquet` + `hit_detection_eval.parquet`, both already carry a precise `t_s`)
using an order-preserving two-pointer match within a ±400ms window (generous — chosen because the
heuristic's own known drift is up to ±0.2s, so a tight window would undercount legitimate matches).
Sanity-checked by hand on one short rally before trusting the aggregate (`Session2_Game_1_rally004`:
the algorithm correctly matched a manual hit at 32.028s to a heuristic "drop" at 32.032s — 4ms apart
— and correctly left an isolated manual hit at 31.344s unmatched, since the nearest heuristic
candidate was 521ms away, outside any reasonable window).

**Result — pooled across 15 rallies, 96 manually-confirmed hits, 103 heuristic candidates:**

| Metric | Measured | Read |
|---|---|---|
| Recall (real hits the heuristic actually found) | **65%** | far below the ≥0.90 G0 target |
| Precision (heuristic candidates that were real) | **60%** | far below the ≥0.90 G0 target |
| Missed real hits | 34 / 96 | heuristic silently drops over a third of real hits |
| Invented candidates | 41 / 103 | 4 in 10 heuristic "hits" weren't real |
| Median timing offset, matched pairs | 187ms | large, but see caveat below |

This is a **materially worse picture than the original v1 label review implied** (695/738 = 94%
of exported candidates were confirmed as real hits back then). The difference: that review only
ever asked "is *this* candidate real?" one isolated clip at a time — it could never catch hits the
heuristic never proposed a candidate for in the first place. Watching the whole rally fresh and
counting from scratch is a much stricter test, and it surfaces the miss rate the original process
was structurally blind to. This is a real, broad pattern — not one bad rally: 11 of the 15 rallies
show at least one invented candidate, and 13 of 15 show at least one missed hit.

**Caveat on the 187ms timing number:** unlike M1 (machine vs. machine), this comparison is
heuristic-vs-human, so the offset mixes two things — the heuristic's own timing drift, *and*
however many frames a person's eye+reaction take to register "that's the hit" while frame-stepping
a video. The mean signed offset (heuristic ≈133ms *before* the human-marked frame) is directionally
consistent with the smaller, cleaner offset M1 found (heuristic ≈20–30ms before the actual audio
onset) — same direction, larger magnitude here, most likely because part of it is human
perceptual lag rather than pure heuristic error. Don't read 187ms as the heuristic's "true" drift on
its own; the recall/precision numbers above are the more robust finding from this measurement.

**Shot types among the hits the heuristic DID catch** (not full per-type recall — see caveat
below): drop 20, lift 17, serve 6, clear 5, smash 5, drive 2, unknown 7.

**Important limitation:** true recall *per stroke type* (what M2 was originally meant to answer)
needs to know what type each *missed* hit would have been — and the trainer recorded hit timing,
not shot type, so that's not knowable from this data. What we get instead is honest and still
useful: an overall recall/precision number, plus the type mix of what the heuristic *does* catch
(skewed toward its own biases, not the true population — exactly the "detector-biased class
counts" the spec already warned about).

**Flag for you:** this is the sharpest finding of the whole round. 65%/60% recall/precision is a
long way from G0's ≥0.90/≥0.90 bar, and it's a direct, independent confirmation (this time from a
human, not a microphone) that the current trajectory-reversal heuristic is not good enough to
build on as-is — reinforcing M1's conclusion and the spec's own plan to replace it with a learned
detector (Stage 1). I'd treat "the heuristic needs a real rebuild, not incremental gate-tuning" as
the headline takeaway from this round, going into M4+.

**One bug caught and fixed while preparing the clips:** my first export attempt truncated every
clip to its first quarter — `-t` landed as an *output*-duration limit applied after the 4×
`setpts` slow-down instead of an *input* trim before it. Fixed by moving `-ss`/`-t` ahead of `-i`;
re-exported and spot-checked durations before handing off.
