# Feature — Pose/swing fusion for rally segmentation (FR-35 follow-up, step 2)

**Status: SPEC ONLY — no code yet.** Rules over the existing heuristic, **not a learned segmenter**
(settled 2026-07-02, deep-research verdict). The single biggest measured weakness of stage 6 is its
single input signal (shuttle continuity); this fuses a **player swing-activity cue** — derived from the
RTMPose keypoints the pipeline already caches — into the rule-based segmenter, plus the eval harness
that proves the gain on the trainer's own reviewed footage. No new model, no web changes.

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

1. **Phantom short rallies** between real ones — the dominant class (~16 of 52 in game1). No swing
   happens near them → swing **veto**.
2. **Over-splits** at shuttle-detection dropouts longer than `RALLY_BRIDGE_S`=1.5 s (4+1 trainer
   merges). Players keep swinging through the dropout → swing **bridge**.

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

**Two operating thresholds** on the same event stream: the veto filters at
`RALLY_SWING_VETO_MIN_STRENGTH` (low — any plausible arm activity saves a rally), the bridge at
`RALLY_SWING_BRIDGE_MIN_STRENGTH` (high — only unambiguous swings justify merging). Both picked
from measured in-rally vs between-rally strength distributions, not guessed.

## Fusion rules (each env-gated, **default OFF** until the measured gate passes)

Cost asymmetry drives both rules' shape: the review UI can drop a surviving phantom in one click,
but a falsely vetoed real rally and a falsely bridged pair are **unrecoverable** (no restore/split
op). Hence: veto = low threshold + span guard; bridge = high threshold + serve-exclusion zone.

- **Veto** (after `merge_runs_into_rallies`, before the emit loop): discard a candidate rally iff
  **no** swing ≥ veto-strength within `[t_start − RALLY_SWING_VETO_MARGIN_S, t_end + margin]` AND
  span < `RALLY_SWING_VETO_MAX_S` (long rallies can never be vetoed; guard set from the measured
  kept-span distribution — no kept rally may fall under it). No extra shuttle-conf/det-count
  conditions: measured phantoms carry conf 1.00 and ≥5 dets, so those gates are pure overfitting
  surface. Honest target: kills swing-free phantoms (≥50% of game1's ~16); phantoms with genuine
  nearby arm motion (pickup tosses) survive **by design**.
- **Bridge** (inside `merge_runs_into_rallies`): gaps in `(RALLY_BRIDGE_S, RALLY_SWING_BRIDGE_S]`
  merge iff a swing ≥ bridge-strength lies in the gap **interior**,
  `t ∈ [gap_start, gap_end − RALLY_SERVE_EXCLUSION_S]`. The exclusion zone exists because a quick
  re-serve puts a genuine swing at a gap's *end* — bridging across it would glue two real rallies.
  Cap set from the measured gap histogram of the trainer-merged pairs, not guessed.
- **Serve corroboration** (confidence-only, **times never move**): `find_serve_frame` marks
  `payload.swing_corroborated` and raises confidence to `RALLY_SERVE_CONF_CORROBORATED` when
  settled AND a swing falls within ±`RALLY_SERVE_SWING_WINDOW_S` of the first detection. Feeds
  graduated-autonomy certainty surfacing; not scored by the harness.

**Wiring:** `rally_hit.py` computes swings unconditionally when `pose.parquet` exists (join pose
`track` → `tracks.parquet` kept track ids; `source="interp"` rows have no pose — harmless) and
passes them to `build_rallies(dets, player_energy, swings)`. Gating lives ONLY inside the
segment.py rules, so the eval harness flips flags via env without rewiring.

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

## Numeric targets (provisional until the baseline table below is filled)

| Metric (all 5 reviewed games) | Baseline (measured) | Target |
| --- | --- | --- |
| boundary-F1 @ ±0.5 s | 0.742–0.775 per game | **up on every game** (game1 ≥ 0.85 aspirational) |
| `false_rallies` | 76 aggregate (7–20/game) | **≤ 38 aggregate** (≥50% cut) |
| `splits` | 20 aggregate (2–5/game) | **reduced on every game** |
| `missed` (falsely vetoed real rally) | 0 | **hard 0, every reviewed game** |
| `merges` (falsely bridged pair) | 0 | **hard 0, every reviewed game** |

Flags flip ON by default **only** when all hard gates pass on every reviewed game (mirrors the
promoted-checkpoint precedent). If a criterion fails: re-tune from the measured histograms only —
no new rule conditions without a new measured failure class.

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

**Per-game results: pending** the Mac runs
(`uv run python -m worker.eval_rally --slug <slug> --swing-hist [--json]`).

**Pre-registered decision rule** (set before seeing the numbers): fused sweep reaches ≥38/76
kills with fused-missed == 0 on every game → the T7 veto condition is swing-OR-settled. The
serve-window report showing near-serve blindness (strong far-skew / high no-sample count) is a
T7 design input, not a reason to lower thresholds. Neither separates → drop the veto (bridge
T8 only); phantom removal stays the trainer's one-click drop in rally-review.

## Env knobs (defaults; pattern = module constant + default kwarg, `segment.py` style)

| Knob | Default | Meaning |
| --- | --- | --- |
| `SWING_KP_CONF` | 0.3 | min kp_score for a joint to count |
| `SWING_MIN_TLPS` | TBD (T6; placeholder 3.0) | emission floor, torso-lengths/s |
| `SWING_MERGE_S` | 0.3 | coalesce adjacent above-floor runs |
| `SWING_MAX_DT_S` | 0.25 | never differentiate across a longer gap |
| `SWING_ELBOW_GAIN` | 1.8 | elbow-speed → wrist-speed proxy factor |
| `SWING_TORSO_WINDOW_S` | 2.0 | rolling-median torso window |
| `RALLY_SWING_VETO` | **"0"** (off) | phantom-rally veto master switch |
| `RALLY_SWING_VETO_MAX_S` | TBD — T3 contradicted the ≈3 s plan (see baseline table); user decision pending | rallies ≥ this can never be vetoed |
| `RALLY_SWING_VETO_MARGIN_S` | 0.5 | swing search margin around the rally span |
| `RALLY_SWING_VETO_MIN_STRENGTH` | TBD (T6) | low bar — any plausible activity saves |
| `RALLY_SWING_BRIDGE` | **"0"** (off) | swing-conditional gap bridging master switch |
| `RALLY_SWING_BRIDGE_S` | ≈2.2 (T3-measured; max trainer-merged gap 1.92 s) | max gap the swing bridge may close |
| `RALLY_SWING_BRIDGE_MIN_STRENGTH` | TBD (T6) | high bar — unambiguous swings only |
| `RALLY_SERVE_EXCLUSION_S` | 0.75 | no bridging on a swing this close to gap end |
| `RALLY_SERVE_CONF_CORROBORATED` | 0.85 | serve confidence when swing-corroborated |
| `RALLY_SERVE_SWING_WINDOW_S` | 0.5 | ± window around first detection |

## Honest limits (surface, don't hide)

- **12 fps aliasing:** a very compact swing can land between analysis frames; the cue misses it.
  Mitigated (not solved) by the 2·dt gap bridging over blurred/unconfident peak frames.
- **Backhand short serve** is the lowest-amplitude real swing → the false-veto risk. Mitigations:
  the veto threshold is deliberately low, the elbow fallback catches wrist misses, and the ±margin
  window catches slightly-off timing. The eval's hard `missed == 0` gate must confirm this on
  every reviewed game before any default flips.
- **Pose exists only for tracked boxes:** `gate_track` failures cascade — where the tracker lost a
  player, there are no keypoints, no swing events, and the veto could fire on a real rally. The
  span guard (`RALLY_SWING_VETO_MAX_S`) bounds the damage to short rallies; `missed == 0` is the
  tripwire.
- **Arm-motion phantoms survive the veto by design** (pickup tosses, racket fidgeting near the
  net). The veto targets swing-*free* phantoms only; expect `false_rallies` to halve, not vanish.
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
