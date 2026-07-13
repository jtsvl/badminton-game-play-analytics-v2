# Feature — Pose/swing fusion for rally segmentation (FR-35 follow-up, step 2)

**Status (2026-07-03): veto DROPPED (measured T6 verdict below); T8 bridge PROMOTED ON**
(`RALLY_SWING_BRIDGE_MIN_STRENGTH=4.0`, ~80% measured split recovery) **with a known, deliberately
accepted gap: the false-merge safety check had zero real test cases on the 5-game corpus** — see
the T8 verdict below before touching this default. Rules over the existing heuristic, **not a
learned segmenter** (settled 2026-07-02, deep-research verdict). The single biggest measured
weakness of stage 6 is its single input signal (shuttle continuity); this fuses a **player
swing-activity cue** — derived from the RTMPose keypoints the pipeline already caches — into the
rule-based segmenter, plus the eval harness that proves the gain on the trainer's own reviewed
footage. No new model, no web changes.

## Failure-mode inventory (measured on trainer-reviewed footage, 2026-07-02)

The trainer's rally reviews are the evidence base (`rally_review` on Session-2 games 1 and 5):

- **game1** (`2026-06-30-session2-game1`): machine 52 rallies → trainer kept **32**. Four merge
  survivors (idx 0 absorbed 1,2; idx 13 absorbed 14,15,16 → now 13.3 s, conf 0.42; idx 9; idx 44).
  ~14 removals are pure **drops**: phantom "rallies" 0.5–2.5 s, 1–4 shots, often conf 1.00 —
  dense short noise (pickup tosses, walk-backs, neighbour-court leaks).
- **game5**: 23 → 21 on the first (light) pass; re-reviewed strictly the same day → **14** kept.
- Games 2/3/4 reviewed 2026-07-02 evening: 61→36, 61→36, 49→31 — same signature everywhere
  (~40 % of machine rallies are noise; full per-game numbers in the baseline table below). The
  four review sessions on the capture-enabled build wrote the first **90 `rally_corrections`
  rows** (60 drops, 15 merge pairs), end-to-end-verifying the FR-35 capture in production.

Two failure classes, by measured frequency:

1. **Phantom short rallies** between real ones — the dominant class (~16 of 52 in game1). The
   original hypothesis: no swing happens near them → swing **veto**. **Measured false 2026-07-03**
   (see T6 verdict below) — real short rallies' swing strength overlaps phantoms' too much; this
   class stays a manual one-click drop in rally-review.
2. **Over-splits** at shuttle-detection dropouts longer than `RALLY_BRIDGE_S`=1.5 s (4+1 trainer
   merges). Players keep swinging through the dropout → swing **bridge** (T8, mechanism built,
   threshold tuning pending).

## GT regime — what is measurable today (and what is not)

The rally-review UI only **drops and merges** — it cannot move boundaries. So the ground-truth
boundaries of every kept rally are the machine's own boundaries. Consequences:

- The baseline boundary-F1 deficit comes *exactly* from phantoms + splits, so veto/bridge are fairly
  measurable (game1 baseline ≈ 0.76: recall 1.0, precision ≈ 64/104).
- Boundary-**moving** improvements (serve-snap, rally-end-trim) would always score as regressions
  under this GT → **deferred**. Now: a confidence-only serve corroboration marker. The timing work
  is future work, unlocked by a one-off hand-labeled boundary session (~20–30 rallies).
- The literature's +0.18 F1 for pose fusion (Sensors 2024) is **not comparable** to this GT regime;
  this spec sets its own targets below.

## Signal definition (swing-activity events)

`worker/worker/rally/swing.py` (pure, dev-extra testable — no torch/cv2/polars):

- Per (frame, kept track): **arm-tip speed** = displacement/dt of the most distal *confident* joint
  per arm — wrist (Halpe kp 9/10) at `kp_score ≥ SWING_KP_CONF`, else elbow (kp 7/8) ×
  `SWING_ELBOW_GAIN`; take the max over both arms.
- If a middle frame's joints are unconfident, take displacement across the gap over the true 2·dt
  (a motion-blurred swing peak still shows as one big step) — but never across dt >
  `SWING_MAX_DT_S` (tracker dropouts must not fabricate speed).
- Normalize by a **per-track rolling-median torso length** (‖kp17−kp19‖ over
  `SWING_TORSO_WINDOW_S` ≈ 2 s). Instantaneous torso collapses in lunges and would spike exactly at
  athletic moments; bbox height is worse; court-projecting an airborne wrist is geometrically
  meaningless. Unit: **torso-lengths/second (tl/s)**.
- A **swing event** = a maximal run of consecutive samples above the emission floor
  `SWING_MIN_TLPS`; emit `{"t": t_peak, "track": id, "strength": max}`; runs closer than
  `SWING_MERGE_S`=0.3 s coalesce. No local-max logic — at 12 fps a swing is 1–2 samples, peak
  detection is degenerate.

**Honest limit — 12 fps:** the analysis cache is 12 fps; a swing's forward phase is 1–2 frames.
This signal is an **activity cue** ("racket-like arm motion in this window"), not a hit timer —
±83 ms timing is fine for veto/bridge and is one more reason hit/serve *timing* stays stage-7 /
future work.

**One operating threshold, one dropped:** the veto would have filtered at a low bar (any plausible
arm activity saves a rally) — measured 2026-07-03 not to separate real short rallies from
phantoms, so it was never implemented. The bridge filters at `RALLY_SWING_BRIDGE_MIN_STRENGTH`
(high — only unambiguous swings justify merging), picked from measured gap-interior strength
distributions (`--bridge-hist`), not guessed.

## Fusion rules (each env-gated, **default OFF** until the measured gate passes)

Cost asymmetry drives both rules' shape: the review UI can drop a surviving phantom in one click,
but a falsely vetoed real rally and a falsely bridged pair are **unrecoverable** (no restore/split
op). Hence: veto = low threshold + span guard; bridge = high threshold + serve-exclusion zone.

- **Veto — DROPPED (2026-07-03 measured verdict, see T6 below).** Never implemented in
  `segment.py`. Design retained here only as the record of what was tried: discard a candidate
  rally iff **no** swing ≥ veto-strength within `[t_start − RALLY_SWING_VETO_MARGIN_S, t_end +
  margin]` AND span < `RALLY_SWING_VETO_MAX_S`. Measured fused (swing-OR-settled) kill rate at the
  largest shared threshold keeping `missed == 0` on all 5 games: **2/76 (2.6%)**, nowhere near the
  ≥50% target — phantom removal stays the trainer's one-click drop in rally-review.
- **Bridge (T8 — PROMOTED ON 2026-07-03)** (inside `merge_runs_into_rallies`): gaps in
  `(RALLY_BRIDGE_S, RALLY_SWING_BRIDGE_S]` merge iff a swing ≥ `RALLY_SWING_BRIDGE_MIN_STRENGTH`
  (4.0 tl/s) lies in the gap **interior**, `t ∈ [gap_start, gap_end − RALLY_SERVE_EXCLUSION_S]`. The
  exclusion zone exists because a quick re-serve puts a genuine swing at a gap's *end* — bridging
  across it would glue two real rallies. Cap set from the measured gap histogram of the
  trainer-merged pairs (18/20 ≤ 2.34s). Threshold picked from `--bridge-hist` (~80% split recovery,
  see the T8 verdict below) — **but the false-merge side of that measurement had zero test cases**,
  a deliberately accepted gap, not a proven-safe result.
- **Serve corroboration** (confidence-only, **times never move**): `find_serve_frame` marks
  `payload.swing_corroborated` and raises confidence to `RALLY_SERVE_CONF_CORROBORATED` when
  settled AND a swing falls within ±`RALLY_SERVE_SWING_WINDOW_S` of the first detection. Feeds
  graduated-autonomy certainty surfacing; not scored by the harness.

**Wiring (built 2026-07-03):** `rally_hit.py` computes swings unconditionally when `pose.parquet`
exists (join pose `track` → `tracks.parquet` kept track ids; `source="interp"` rows have no pose —
harmless) and passes them to `build_rallies(dets, player_energy, swings)`. Gating lives ONLY
inside the segment.py rules (`swing_bridge` flag), so the eval harness flips it via env without
rewiring. (Second `pose.parquet` consumer since 2026-07-12: the `overlay` stage persists in-rally
skeleton keyframes to `pose_keyframes` — `features/pose-skeleton-overlay.md`; tuning `SWING_*`
knobs does not affect it.)

## Eval harness

`worker/worker/eval_rally.py`, mirroring `eval_corrections.py` (pure scoring top, lazy-import CLI
bottom):

- **GT** = current DB rallies of matches with `rally_edited_at IS NOT NULL` (the post-review rally
  list IS trainer-approved ground truth; `build_rallies` is deterministic, so re-running it on the
  cached `shuttle.parquet` reproduces the machine's pre-review output exactly — no per-edit
  backfill needed).
- **Prediction** = `build_rallies` re-run from cached parquets through rally_hit's own loaders
  (extracted, not duplicated — else the "baseline reproduces pre-review output" claim silently
  breaks).
- **Metrics:** boundary F1 @ ±0.5 s with starts matched to starts and ends to ends (separately,
  greedy 1:1 — else a phantom's end matches a real start and inflates precision), plus a confusion
  in the trainer's own vocabulary via max-overlap assignment (NOT IoU≥0.5 — split fragments fail
  IoU and would miscount): `false_rallies` (=drops), `splits` (=merges), `missed` (false-veto
  alarm), `merges` (false-bridge alarm).
- **`--swing-hist` diagnostics** (T6): per-sample in/between strength split (near/far), the
  serve-window report, per-rally max GT-vs-phantom distributions with the weakest-GT listing,
  settled-before-start counts, and the veto sweep with swing-only vs fused (swing-OR-settled)
  columns; `--json` carries the raw per-rally records.
- **`--bridge-hist` diagnostics** (T8, added 2026-07-03): swing-strength distributions over
  `split_gap_spans` (interior gaps of trainer-merged rallies — should bridge) vs `real_gap_spans`
  restricted to the same candidate range (genuine gaps between distinct real rallies — must NOT
  bridge), plus an operating-point sweep (`bridge_operating_points`) reporting `splits_bridged` and
  the hard-gate `false_merges` per threshold; `--json` carries the raw per-gap records.

## Numeric targets (updated 2026-07-03 — veto dropped, T8 bridge-only scope)

| Metric (all 5 reviewed games) | Baseline (measured) | Target |
| --- | --- | --- |
| boundary-F1 @ ±0.5 s | 0.742–0.775 per game | **up on every game** — smaller gain now that only splits (not phantoms) are addressable |
| `false_rallies` | 76 aggregate (7–20/game) | **ABANDONED** — the veto that targeted this (≤38, ≥50% cut) measured 2/76 (2.6%); phantom removal stays manual (rally-review one-click drop) |
| `splits` | 20 aggregate (2–5/game) | **PROMOTED 2026-07-03** — thr=4.0 recovers 17/21 (81%) measured splits |
| `missed` (falsely vetoed real rally) | 0 | **N/A** — no veto code exists in `segment.py`, so this can't regress |
| `merges` (falsely bridged pair) | 0 | **untestable on this corpus** (0 real gaps in the 1.5–2.5s candidate range) — promoted anyway on a deliberate safety margin, see T8 verdict; **the tripwire is future rally-review data**, not this gate |

`RALLY_SWING_BRIDGE` was promoted ON 2026-07-03 despite the `false_merges` gate being untestable
(vacuously 0 over an empty population) rather than measured-safe — a deliberate, surfaced
exception to the "flip on only when the hard gate passes" rule, justified by the conservative
threshold margin and the unrecoverable-but-rare failure mode. If future rally-review data ever
shows a falsely-glued pair, raise `RALLY_SWING_BRIDGE_MIN_STRENGTH` or set `RALLY_SWING_BRIDGE=0`
— both are env-only reverts, no code change needed.

### Baseline table (measured 2026-07-02 after the trainer reviewed ALL 5 games, flags off, tol ±0.5 s)

| game | machine | kept | boundary-F1 | precision | recall | false_rallies | splits | missed | merges | min/median kept span |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| game1 | 52 | 32 | **0.762** | 0.615 | 1.000 | 15 | 4 | 0 | 0 | 0.50 / 3.13 s |
| game2 | 61 | 36 | **0.742** | 0.590 | 1.000 | 20 | 5 | 0 | 0 | 1.00 / 3.04 s |
| game3 | 61 | 36 | **0.742** | 0.590 | 1.000 | 20 | 5 | 0 | 0 | 0.58 / 3.46 s |
| game4 | 49 | 31 | **0.775** | 0.633 | 1.000 | 14 | 4 | 0 | 0 | 0.83 / 3.42 s |
| game5 | 23 | 14 | **0.757** | 0.609 | 1.000 | 7 | 2 | 0 | 0 | 1.00 / 2.71 s |
| **Σ** | **246** | **149** | — | — | 1.000 | **76** | **20** | **0** | **0** | min 0.50 s |

Every game reproduces the pre-review machine output exactly (recall 1.0, `missed=0`,
`merges=0`) — the flags-off determinism check passes on all 5. The picture is remarkably
uniform: F1 0.74–0.78 everywhere, ~40 % of machine rallies are noise. (game5 was re-reviewed
more strictly on 2026-07-02 18:33 — its GT tightened from 21 to 14 kept rallies, so its earlier
provisional F1 of 0.955 was an artifact of the lighter first pass.)

**Merged-pair gap histogram** (interior gaps of trainer-merged rallies, 20 pairs): game1
`[1.58, 1.58, 1.58, 1.67, 1.92]`, game2 `[1.50, 1.50, 1.58, 1.84, 2.34]`, game3
`[1.58, 1.58, 1.67, 1.84, 2.09]`, game4 `[1.50, 1.58, 2.17, 3.59]`, game5 `[1.92, 1.92]` (s).
18/20 gaps ≤ 2.34 s; one outlier at **3.59 s** (game4). → `RALLY_SWING_BRIDGE_S` choice at
T6/T11: **≈2.5 s** covers 18/20 splits with a tight cap, **≈3.7 s** covers all 20 but widens
the false-bridge surface (mitigated by the interior-swing + serve-exclusion conditions). Start
at 2.5, revisit against the measured swing distributions.

**Kept-rally span distribution — CONTRADICTS the plan's veto-guard expectation, on all 5
games.** The plan assumed the shortest kept rally ≈ 3 s, so `RALLY_SWING_VETO_MAX_S` could sit
"just below the shortest kept rally" and make every real rally veto-ineligible by span alone.
Measured minima: 0.50 / 1.00 / 0.58 / 0.83 / 1.00 s; medians 2.7–3.5 s — roughly half of all
kept rallies sit under 3 s. Real short rallies (serve faults, quick kills) overlap the phantom
band (0.5–2.5 s) almost completely — **no span guard can separate them**. If the veto ships,
the span guard can only cap eligibility (damage control for pose dropouts); the swing signal
alone must protect real short rallies, and the hard `missed == 0` gate is the arbiter.
*Decision pending at the T4 trainer-review checkpoint.*

### Swing-strength distributions (T6)

**First pass (measured 2026-07-02, per-sample):** the naive veto — save a candidate iff any
swing sample ≥ thr within span ±0.5 s — kills only **4/76** phantoms aggregate at the largest
threshold that keeps `missed == 0` on all 5 games, far under the ≥50 % target; the per-sample
in-rally vs between-rally distributions overlap heavily, and a run-length variant did not
separate either. Back-to-camera check on game1 (all in-rally samples): near p50 = 1.35 vs far
p50 = 1.14 tl/s — **inconclusive**, because it lumps every shot type; serve-instant swings were
never isolated.

**T6 extension (2026-07-02): decision-shaped diagnostics**, all emitted by ONE `--swing-hist`
run per game (`--json` additionally carries the raw per-rally records for offline re-analysis):

- **Serve-window report** — per GT rally, the strongest swing in `t_start ± 0.5 s`, attributed
  to its court half. The server is unlabeled: attribution = strongest swing in the window, and
  serves alternate ≈50/50, so a strong far-skew (or systematically weaker near-attributed
  maxima) is the back-to-camera signal; `no_sample_rallies` separately counts serve instants
  with no pose signal at all.
- **Per-rally max** over span ±0.5 s — the exact quantity the veto thresholds on — GT vs
  phantom distributions plus the weakest-GT listing (the rallies that pin the missed==0
  threshold down; `max=None` = unsavable by any swing threshold, i.e. a pose dropout).
- **Settled fusion** — `players_settled_before(t_start)` counts at 3/4/6 km/h and a fused
  sweep column `swing ≥ thr OR settled` (production serve-cue defaults): the measured phantom
  classes (pickup toss, walk-back) should lack the settled-before-start pattern real serves
  have, letting thr rise past the naive pin point.

**Pre-registered decision rule** (set before seeing the numbers): fused sweep reaches ≥38/76
kills with fused-missed == 0 on every game → the T7 veto condition is swing-OR-settled. The
serve-window report showing near-serve blindness (strong far-skew / high no-sample count) is a
T7 design input, not a reason to lower thresholds. Neither separates → drop the veto (bridge
T8 only); phantom removal stays the trainer's one-click drop in rally-review.

**Per-game results (measured 2026-07-03, all 5 games, `--swing-hist`)** — the largest shared
threshold at which the FUSED (swing-OR-settled) rule keeps `missed == 0` on every game is
**thr=3.00 tl/s**, killing only **2/76 (2.6%)** phantoms aggregate (game2: 2/20; all other games:
0):

| game | fused OK up to (tl/s) | kills at that thr | phantom_total |
| --- | --- | --- | --- |
| game1 | 4.00 | 1 | 15 |
| game2 | 5.00 | 2 | 20 |
| game3 | **3.00** (breaks at 4.00: 34/36, 2 real rallies falsely vetoed) | 0 | 20 |
| game4 | 5.00 | 0 | 14 |
| game5 | 6.00 | 0 | 7 |

game3 is the binding constraint: its settled-before-start rate is far lower than the other games
(15–20/36 across 3/4/6 km/h vs 24–30+/32–36 elsewhere) — players there don't reset between points
as reliably, so the settled cue can't rescue it, and real short-rally swing strength overlaps
phantom swing strength too much for the swing signal alone to separate them at any shared
threshold. Even discounting the missed-rally violations entirely and pushing to the highest
tested threshold (8.0 tl/s), aggregate kills only reach ~15/76 (20%) — still far under the ≥50%
target. **Not a borderline call.**

Serve-window (near/far) data: near-attributed serves skew mildly stronger than far in 4/5 games
(game4 is the exception, small n), but every rally had *some* swing sample at the serve instant
(`no_sample_rallies = 0` everywhere) — no severe back-to-camera blindness to explain the gap away.

**Verdict: the veto is dropped**, per the pre-registered rule's "neither separates" branch. T7
does not happen. Phantom-rally removal stays the trainer's one-click drop in rally-review
(already shipped, FR-35) — this is not a regression, it's confirming the existing manual step is
the right tool for this failure class. Work moves to **T8 (bridge-only)**.

### T8 — bridge mechanism (built 2026-07-03)

`merge_runs_into_rallies` and `build_rallies` (`worker/worker/rally/segment.py`) now accept an
optional `swings` sequence and a `swing_bridge` flag (default `RALLY_SWING_BRIDGE`, env "0"/off):
when on, a gap in `(RALLY_BRIDGE_S, RALLY_SWING_BRIDGE_S]` additionally merges if a swing ≥
`RALLY_SWING_BRIDGE_MIN_STRENGTH` falls in `[gap_start, gap_end − RALLY_SERVE_EXCLUSION_S]` (10
new unit tests, TDD, `worker/tests/test_rally.py::TestSwingBridge` +
`TestBuildRallies::test_swings_threaded_through_to_the_bridge_rule`). `rally_hit.py` computes
swings unconditionally whenever `pose.parquet` exists (`_swings_from_frames`, joined to
`tracks.parquet`'s kept track ids) and passes them through — the flag alone gates behavior, so no
rewiring is needed once a real threshold is set. `eval_rally.predict_from_cache` was updated to
load and pass swings identically (`load_swings`, reusing `rally_hit._swings_from_frames` — not a
duplicate path), so the plain scorecard stays truthful once the flag flips.

**Per-game results (measured 2026-07-03, all 5 games, `--bridge-hist`):**

| game | split gaps (should-bridge) | at thr=4.0: bridged | real gaps in (1.5, 2.5]s range (must-NOT-bridge) |
| --- | --- | --- | --- |
| game1 | 5 | 5/5 | 0 |
| game2 | 5 | 3/5 | 0 |
| game3 | 5 | 3/5 | 0 |
| game4 | 4 | 4/4 | 0 |
| game5 | 2 | 2/2 | 0 |
| **Σ** | **21** | **17/21 (81%)** | **0** |

At thr=2.5 tl/s every measured split gap bridges correctly (21/21) on all 5 games; recovery
degrades unevenly above that (game3 first, at thr=3.0). thr=4.0 was chosen as a **deliberate safety
margin** above the point of 100% recovery, at the cost of ~19% of the measured splits, precisely
*because* the false-merge side of this measurement is empty (see below) — there was no
data-driven way to justify staying at the lowest working threshold.

**The honest gap: `real_gap_spans` restricted to the candidate range (1.5, 2.5]s was EMPTY on
all 5 games — 0 samples, not 0 false-merges.** A `false_merges == 0` reading over an empty
population is vacuously true, not evidence of safety — unlike the veto's `missed == 0` gate, which
was tested against 149 real GT rallies per threshold. The plausible explanation: a real break
between points is either a quick re-serve (well under 1.5s, already handled by the unconditional
`RALLY_BRIDGE_S` path) or a proper reset pause (well over 2.5s) — gaps landing exactly in the
1.5–2.5s band between two genuinely distinct rallies may simply be rare in this sport's rhythm.
Plausible, **not proven** with 5 games.

**Why promoted anyway (user decision, 2026-07-03):** a falsely-bridged pair is unrecoverable —
the rally-review UI has no split/un-merge operation — so this is not a decision to take lightly.
Weighed against that: the conservative threshold (well above the ~0.7–1.4 tl/s between-rally
ambient-motion baseline measured in T6), the modest scope (rare 1.5–2.5s gaps only), and the
existing rally-review capture (FR-35) as an ongoing observation channel. Accepted with eyes open,
not defaulted into. **If a future review ever reveals a falsely-glued rally pair, this is the
first thing to revisit** — either raise `RALLY_SWING_BRIDGE_MIN_STRENGTH` further or turn
`RALLY_SWING_BRIDGE` back off; both are one env var away, no code change needed.

## Env knobs (defaults; pattern = module constant + default kwarg, `segment.py` style)

| Knob | Default | Meaning |
| --- | --- | --- |
| `SWING_KP_CONF` | 0.3 | min kp_score for a joint to count |
| `SWING_MIN_TLPS` | TBD (T6; placeholder 3.0) | emission floor, torso-lengths/s |
| `SWING_MERGE_S` | 0.3 | coalesce adjacent above-floor runs |
| `SWING_MAX_DT_S` | 0.25 | never differentiate across a longer gap |
| `SWING_ELBOW_GAIN` | 1.8 | elbow-speed → wrist-speed proxy factor |
| `SWING_TORSO_WINDOW_S` | 2.0 | rolling-median torso window |
| ~~`RALLY_SWING_VETO`~~ | DROPPED 2026-07-03 — never implemented in `segment.py` | phantom-rally veto master switch |
| ~~`RALLY_SWING_VETO_MAX_S`~~ | DROPPED | rallies ≥ this can never be vetoed |
| ~~`RALLY_SWING_VETO_MARGIN_S`~~ | DROPPED | swing search margin around the rally span |
| ~~`RALLY_SWING_VETO_MIN_STRENGTH`~~ | DROPPED | low bar — any plausible activity saves |
| `RALLY_SWING_BRIDGE` | **"1"** (on) — PROMOTED 2026-07-03, see T8 verdict | swing-conditional gap bridging master switch |
| `RALLY_SWING_BRIDGE_S` | 2.5 (T6-measured gap histogram: 18/20 trainer-merged gaps ≤ 2.34s) | max gap the swing bridge may close |
| `RALLY_SWING_BRIDGE_MIN_STRENGTH` | 4.0 — measured `--bridge-hist` (81% split recovery); deliberate margin above the 2.5 tl/s 100%-recovery point since false-merge safety was untestable (0 real-gap samples in range) | high bar — unambiguous swings only |
| `RALLY_SERVE_EXCLUSION_S` | 0.75 | no bridging on a swing this close to gap end |
| `RALLY_SERVE_CONF_CORROBORATED` | 0.85 | serve confidence when swing-corroborated |
| `RALLY_SERVE_SWING_WINDOW_S` | 0.5 | ± window around first detection |

## Honest limits (surface, don't hide)

- **12 fps aliasing:** a very compact swing can land between analysis frames; the cue misses it.
  Mitigated (not solved) by the 2·dt gap bridging over blurred/unconfident peak frames.
- **The swing signal doesn't separate real short rallies from swing-free phantoms** (measured T6
  verdict, 2026-07-03): even the fused (swing-OR-settled) rule only kills 2/76 phantoms at the
  largest shared threshold that never falsely vetoes a real rally — real short-rally swing
  strength overlaps phantom swing strength too much, and one game's players don't settle between
  points as reliably as the others. This is *why* the veto was dropped rather than shipped with a
  weak threshold — a plausible-but-barely-working auto-delete is worse than none.
- **Pose exists only for tracked boxes:** `gate_track` failures cascade — where the tracker lost a
  player, there are no keypoints, no swing events, and a swing-bridge gap with a real dropout-swing
  could go unbridged (false negative — the rally still stays a `split`, no correctness risk, just
  a missed improvement). The `false_merges == 0` hard gate is the correctness tripwire, not this.
- **`RALLY_SWING_BRIDGE` was promoted ON with an untested safety gate** (2026-07-03): the 5-game
  corpus had zero real inter-rally gaps in the 1.5–2.5s candidate range, so `false_merges == 0` is
  vacuous, not measured-safe. A falsely bridged pair is unrecoverable (no split/un-merge op in
  rally-review). Mitigated by a deliberate margin above the 100%-recovery threshold, not proven
  safe — future rally-review data is the real tripwire; see the T8 verdict for the reasoning.
- **GT is drop/merge-only:** kept-rally boundary correctness is *assumed*, not measured — serve/end
  timing quality is invisible to this harness (see GT regime above).

## Out of scope (explicit)

- Audio hit-detection spike (agreed next step after this feature), tiny GRU/BiLSTM
  sub-classifiers, any auto-accept of review gates (graduated-autonomy stays manual).
- Boundary-moving serve-snap / rally-end-trim — future work; unlock = hand-labeled boundary set
  (~20–30 rallies).
- Hit-event refinement / stroke classification (stage 7 — `stroke-classification.md` note #1 owns
  hit-level pose fusion).
- Web/UI changes, doubles, new models, per-op scoring of the `rally_corrections` stream (the
  harness scores rally lists; op-level scoring waits for corpus volume).
