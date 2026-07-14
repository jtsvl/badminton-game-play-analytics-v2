# Feature — Stroke classification v2 (fresh rebuild)

**Status: DESIGN SIGNED OFF (2026-07-13) — nothing below is built.** FR-21/22/23 (+ FR-24 input).
**Supersedes `specs/features/stroke-classification.md`** (v1). v1 stays on disk as measured
history — its Phase-2 numbers (LOGO-CV 0.400, gate FAILED, pipeline never wired) and its
Phase-1 slot-coverage measurements remain valid reference; its design decisions do NOT govern
new work. Governing brief: `docs/prompts/2026-07-13-stroke-v2-kickoff.md`. Research basis
(all claims cited + adversarially verified there): `docs/research/2026-07-stroke-v2/`
(`memo.md`, `taxonomy-crosswalk.md`, `sources.md`, `finebadminton-eval.md`).

**Next step after this spec: Phase-1 measurements M1–M10 (§10) — user checkpoint first.**
No code, no training, no labeling before that checkpoint.

## 1. Goal

Turn every rally into a per-stroke record: **when** each stroke happened (learned hit
detection, audio-refined timing), **who** hit it (4 doubles roles / 2 singles roles, with an
honest „Unsicher"), **what** it was (coarse-8 German taxonomy, fine-17 labels), and **how it
ended** (winner/error, Netz/Aus, forced/unforced as Schätzung) — each stage shipping only
after its own gate (§9), with abstention as a first-class output everywhere. Plausible-but-
wrong is worse than absent.

## 2. Correction to the v1 spec (taxonomy claim)

v1 (`stroke-classification.md:46`) claimed *"ShuttleSet is 35 classes / 25 merged;
BadmintonDB is the 18-class set"*. **Refuted 0:3 in verification**: ShuttleSet's native
vocabulary IS the 18 fine stroke types (+`unknown`) — confirmed against the KDD-2023 paper,
the repo CSVs (36 484 rows counted directly), and 3 further primary sources. The 35/25
numbers are the *checkpoint head sizes* of the published BST weights (35 plain / 25
`_merged`), a different axis entirely. Same correction applies to
`specs/30-ml-pipeline.md` stage 7. Consequence: our fine labels are natively
checkpoint-compatible — the transfer path is open.

## 3. Reused as verified inputs (do NOT rebuild)

Court calibration + homography · fine-tuned TrackNetV3 shuttle tracking (F1 0.953) ·
player detection/tracking (`gate_track`) · RTMPose Halpe-26 pose (`kp[:17]` = COCO-17) ·
rally segmentation · frame cache · Supabase migrations flow · overlay/report architecture.
From v1 specifically: the **BST4 model code** (4-quadrant-slot generalization of BST_CG_AP,
89 % of checkpoint parameters map, measured +7–12 pts transfer), the **clip-export +
Label-Studio tooling** (`stroke_clips.py`, `import_stroke_labels.py`, port-8010 instance),
and the GBDT training harness (learning-curve + baseline duty, §5 stage 3).

## 4. Taxonomy (signed off 2026-07-13 — Decision 1)

ShuttleSet-18 fine granularity for labeling, coarse-8 for training/reporting. Five
corrections vs. the kickoff table, all DBV-verified (`taxonomy-crosswalk.md` has evidence
per row): Stick-Smash · Töten · hoher Aufschlag · Abwehr regrouped (Block/hohe
Abwehr/Konter) · driven flight merged into Drive (68/36 484 ShuttleSet strokes; German name
collides with „Treibball" = Drive). Result: **17 fine labels + „Unsicher" + „Kein Schlag"**
(the last two are labeling/eval outputs, never trained classes).

### Fine labels (Label Studio)

| # | ShuttleSet | n (SSet) | German label |
|---|---|---|---|
| 1 | short service | 2 051 | kurzer Aufschlag |
| 2 | long service | 373 | **hoher Aufschlag** |
| 3 | clear | 2 922 | Clear |
| 4 | smash | 2 586 | Smash |
| 5 | wrist smash | 1 648 | **Stick-Smash** |
| 6 | drop | 2 144 | Drop |
| 7 | passive drop | 1 356 | passiver Drop |
| 8 | drive (+ driven flight, 68) | 700+68 | Drive |
| 9 | back-court drive | 473 | Hinterfeld-Drive |
| 10 | net shot | 6 290 | Netzdrop |
| 11 | return net | 3 620 | **Block** |
| 12 | cross-court net shot | 1 371 | Cross-Netzdrop |
| 13 | rush | 512 | **Töten** |
| 14 | push | 2 925 | Push |
| 15 | lob | 5 331 | Lob (Lift) |
| 16 | defensive return lob | 301 | **hohe Abwehr** |
| 17 | defensive return drive | 406 | **Konter** |
| — | unknown | 1 407 | Unsicher (never trained) |
| — | (no stroke) | — | Kein Schlag (never trained) |

### Coarse-8 (training classes)

| Coarse | Fine members |
|---|---|
| Aufschlag | kurzer Aufschlag · hoher Aufschlag |
| Clear | Clear |
| Smash | Smash · Stick-Smash |
| Drop | Drop · passiver Drop |
| Drive | Drive (inkl. driven flight) · Hinterfeld-Drive |
| Netzspiel | Netzdrop · Cross-Netzdrop · Töten · Push |
| Lift | Lob (Lift) |
| **Abwehr** | **Konter · hohe Abwehr · Block** |

Notes: the Abwehr group is DBV's own umbrella („Doppelabwehr"); it also fixes v1's
single-member Abwehr class. Known hard pairs (drop↔net, clear↔lift) are unchanged by the
regrouping. FineBadminton independently confirms „stick smash" as a subtype and „block" as
its own class. **Open before labels freeze (Phase 1, <1 h):** pin the exact ShuttleSet
definitions of *passive drop* and *return net/Block* from Wang et al. 2022 (ACM TIST).
Fine labels are the label/reporting level; per-class stats stay coarse until per-class data
supports finer. Rationale for coarse-8 is accuracy, not license: fine-18 SOTA is 0.65 even
WITH leakage.

## 5. Architecture (signed off — Decision 2): four stages, data strategy first

Principle from the v1 post-mortem + research: **data strategy dominates model choice**
(BST vs TemPose < 1 pt); every stage has its own gate; abstention is first-class.

### Stage 0 — Data foundation
- **30-fps re-ingest** (source is 60 fps; 30 is the clean 2:1 pick, 60 stays available for
  hit-frame refinement). M8 verifies fps sensitivity + checkpoint-fps expectations FIRST.
- **Relabel all 738 Session-2 events** under the §4 taxonomy incl. 4-way hitter +
  Unsicher/Kein Schlag, reusing v1 tooling. The old 695 coarse labels become a
  cross-check/agreement signal only.
- **Burned-in hit marker (user requirement 2026-07-13):** v1 clips (±0.7 s) often contained
  2–3 strokes in fast doubles exchanges with no way to tell which was meant. Every exported
  clip gets a marker at the exact hit time — red frame flash + circle at the shuttle pixel
  position (px,py from the shuttle track) — plus „Schlag N von M im Ballwechsel" in the
  Label-Studio task. **M7 validates the marker design on 30–50 clips BEFORE the 738 are
  exported.** The same marker serves the VLM pilot prompt („klassifiziere den markierten
  Schlag").
- **~200 Session-1 strokes labeled early** (`Game_1..6.mp4`, on disk, unlabeled) → true
  cross-session eval exists from day 1 of model work.

### Stage 1 — Hit detection v2 (gates everything: G0)
Learned detector after the HitNet pattern (MonoTrack, CVPRW 2022): small GRU/TCN (~16k
params) over 12-frame windows of pose (all players) + shuttle + court features @30 fps →
hit candidates. Replaces the trajectory-reversal heuristic (±0.2 s drift, misses flat
strokes structurally — v1's class counts are presumed detector-biased). HitNet honest
numbers: raw F1 0.866; its published 0.946 needs a postprocessing rule ("alternating
players") that is **singles logic** — in doubles only "alternating sides" holds. No public
weights → rebuild the pattern, don't import. FineBadminton's heavier VideoMAE hit detector
independently confirms *learned* hit detection; we keep the light GRU (our verified inputs,
MPS-friendly). Output per hit: a single **`hit_frame`** (not a window) — externally
confirmed as the right granularity (FineBadminton carries one on 100 % of demo hits).

**Audio = timing refinement only, never a standalone detector.** Onset detection
(SuperFlux class) inside ±150 ms video-derived candidate windows; A/V-mux offset and
per-hit sound travel time (~40 ms across the court > 1 frame @30 fps) calibrated via M9.
Literature basis: the only far-microphone precedent (broadcast tennis, AVSP 2011) shows
audio-only F 0.41–0.50 but A/V fusion 0.77–0.84 with exactly this pattern. **Hard
constraint (user, 2026-07-13): Mehrfeld-Betrieb is the NORMAL case** — neighboring courts
produce spectrally identical impacts; even inside a candidate window a foreign onset can
land. „Falscher Onset gewählt" is therefore its own G0 error mode (M1 quantifies its rate);
disambiguation cues by strength: coincidence with our shuttle/pose signal > loudness >
stereo left/right imbalance. If M1 finds impacts inaudible, the video detector stands alone
(still better than the heuristic) and G0's ±50 ms is renegotiated honestly.

### Stage 2 — Hitter attribution (G1)
5-way head on the same model (4 quadrant slots + no-hit) — hit detection and attribution
jointly; „Unsicher" below a defined margin is a first-class output, net situations reported
separately (occlusion is the known hard case — forced guessing is prohibited). Baseline
first: **M6** nearest-wrist-to-shuttle (EventAnchor pattern, neck fallback under occlusion)
against the ~554 banked hitter labels. Reference bar: the ONLY published monocular doubles
number is 0.67 accuracy (zero-shot, arXiv 2508.13507) — our baseline may already beat it.
Mind the v1 caveat: label „links/rechts" is viewer-relative, slots are court-u-relative —
verify axis orientation per half before labels become ground truth.

### Stage 3 — Stroke classification (G2 + G3)
**BST-4-slot** (v1 code) initialized from the **ShuttleSet-25 `_merged` checkpoint** (MIT;
weights verified downloadable 2026-07-13), trained on **coarse-8**. Levers, in spec order:
- logit-adjusted loss for thin classes; conservative augmentation only (side-mirror +
  slight temporal jitter — "extreme" bundles hurt per literature);
- temperature scaling on OOF + max-softmax abstention; **marginal** split-conformal for
  coverage statements (class-conditional is not viable at n=20–30);
- optional, literature-backed: SkeletonMAE-style SSL pretraining on our thousands of
  unlabeled strokes, stacked on the ShuttleSet init;
- secondary pretrain sources: **FineBadminton-20K** (F1, §6 — strongest new lever),
  VideoBadminton + BadmintonDB-BST checkpoints (usable per the CLAUDE.md license override
  2026-07-13; BadmintonDB checkpoints are second-rank: leakage-trained, smaller base);
- **GBDT stays as baseline in every report + labeling assist** (`labeling_assist_only`
  carve-out, never pipeline). v1's "trees beat transformer" was measured under ±0.2 s hit
  drift — M4 re-ranks once timing is precise.
- COCO-17 vs Halpe-26 layout is not in the BST paper → local smoke test before training.

### Stage 4 — Outcomes (G4)
Rally-ending stroke × landing (in/aus/unsicher, `shuttle-landing-inout.md` — improving its
coverage is in scope) × `rallies.winner_side` × score-progression consistency (M5
baseline). **Netzroller (net-cord): descoped to a review tag unless M10 finds a real
signal.** Forced/unforced: permanently „Schätzung", review-overridable, never gated.
Human-entered `matches.score` stays authoritative, never overwritten.

## 6. Data & pretraining plan (F1 adopted)

- **FineBadminton-20K** (Apache-2.0 — real grant, no override needed; 70 shipped match
  videos, 43 GB + 20 757 hit annotations, 2 066 rallies, singles-only, 25-fps BWF
  broadcast): run **our own pipeline** (ft-TrackNetV3 + RTMPose + homography) over the 70
  matches → 20 757 labeled strokes **in our feature space** — pretraining for both the hit
  detector (start/end frame windows per hit) and the coarse-8 head, without the
  checkpoint-normalization/fps mismatch that throttled v1 transfer. Mapping per
  `finebadminton-eval.md` §2.2 is near-lossless; the one fuzzy edge: their „push shot"
  mixes our Push and flat Lifts → **subtype-aware mapping required** („flat lift" subtype →
  Lift). Round frame-ranges replace rally segmentation for these videos. Domain gap stays
  (broadcast singles vs our fixed-camera amateur doubles) — FineBadminton improves the DATA
  side, not the domain side.
- **VideoBadminton** (7 822 uni-training clips, closer to amateur conditions): optional
  additional pretrain/intermediate domain. No license grant — usable per user override
  2026-07-13, replace if the app ever distributes. Open question: does the zip contain
  audio tracks (would be the only public audio source)?
- Everything trains/evals **cross-session once Session-1 labels exist**; until then LOGO-CV
  by video with the explicit caveat that same-session results overstate.

## 7. Per-stroke data shapes (F2/F3/F5/F8 adopted)

Every stroke row carries (shapes follow `specs/20-data-model.md` conventions; exact DDL at
build time via migration):
- `hit_frame` (single frame) + refined `t_s`, timing source (video-only / A/V), confidence;
- hitter slot (4-way / 2-way) + attribution confidence, „Unsicher" explicit;
- fine label (17) + coarse class (8) + model confidence; abstention below τ;
- **ball-area 3×3 zone** (derived from shuttle track / landing — cheap, F2);
- **geometric shot characteristics**: straight/cross-court from the trajectory vector (F3);
- **`player_actions`** (technique, e.g. Vorhand/Rückhand — later, F7 radar) **and
  `strategies`** (tactical intent, e.g. passiv/defensiv/Antizipation) as TWO separate
  fields, never a mixed array; intent values are Schätzung tier (F3, schema pattern
  confirmed against FineBadminton's demo schema);
- **quality stars** (F5, user-opted): optional 1–5 „Schätzung" column, review-correctable,
  never gate-relevant, never a training target;
- outcome: `rally_continued` | winner | error{Netz, Aus} attributed to the correct player;
  forced/unforced estimate.
- Corrections: append-only `stroke_corrections` (pattern: `0011_rally_corrections.sql`),
  feeding retraining + G5 edit-rate measurement.

## 8. VLM pre-labeling pilot (signed off — Decision 3; pre-registered)

Runs only now that memo AND spec are signed off. ~100 stratified clips over the 8 coarse
classes (net/drive oversampled), burned-in marker, prompt = „klassifiziere den markierten
Schlag", reference = human fine labels.
- **Stack: mlx-vlm ≥ 0.6.3** (DeepStack fix), Qwen3-VL 30B-A3B 4-bit or 8B. **NOT
  llama.cpp** (video hardcoded to 4 fps — reproduces the 0.267 failure). First pilot cell
  verifies frame/timestamp behavior explicitly (mlx-vlm silently drops fps metadata).
- **16–32 frames per clip, explicitly controlled, hit-centered sampling** (F6 —
  FineBadminton's hit-centric keyframe selection improved every model they tested).
  Qwen3-VL's own defaults (fps=2, min 4 frames) are the same trap — verify frames arrive.
- **Gate: macro agreement ≥ 0.60 AND no coarse class < 0.30 AND measured time savings > 0**
  (20-clip timing test with/without assist). **Abort: < 0.5 on the first 30 clips** — no
  prompt-tuning marathon. Always reported next to the 0.267 Qwen2.5-VL prior.
- Expectation per literature: fails (VLM ceiling on fine-grained badminton: 0.32–0.42; even
  a FineBadminton-finetuned Qwen2.5-VL-7B reaches only 0.421). Pass ⇒ labeling
  triage/pre-sort ONLY — never pipeline. Optional extra cell: Qwen3-Omni audio+video
  (uncharted, not a prerequisite).

## 9. Gate stack G0–G5 (replaces v1's single 0.70 gate)

v1's gate was structurally broken: drop+lift+clear = 71 % of labels — head-class hits pass
0.70 with net/drive recall 0.00. "Passed" now means: **a trainer is not systematically
misled by the table.** Final G2 target number is fixed after M3 (learning curve); gates
freeze only AFTER M2 (true hit recall — class priors shift with a better detector).

| Gate | Criterion |
|---|---|
| **G0 Hit-Timing** (gates everything) | Recall ≥ 0.90 AND Precision ≥ 0.90 at ±50 ms on a hand-annotated eval set (≥150 hits, ≥2 games); median \|Δt\| ≤ 25 ms; „falscher Onset gewählt" (neighbor-court coincidence in the window) reported separately |
| **G1 Zuordnung** | Coverage ≥ 0.80 at accuracy ≥ 0.85; net situations reported separately |
| **G2 Schlagklasse** | Macro-recall ≥ 0.55 AND every class with n ≥ 30: recall ≥ 0.40 AND train–eval gap ≤ 15 pts (memorization alarm). Cross-session eval mandatory from Session 2 |
| **G3 Abstention** | Coverage ≥ 0.60, covered-accuracy ≥ 0.80, abstention informative (accuracy on abstained strokes ≥ 15 pts below covered-accuracy), ECE ≤ 0.10 after temperature scaling; coverage statements via marginal conformal (n≈550–700 ⇒ ±3–5 pts realized swing — reported) |
| **G4 Outcomes** | Winner/error ≥ 0.80 on rally-ending strokes (with abstention); Netz/Aus only above the landing-confidence floor; forced/unforced never gated (permanent Schätzung); per-game score consistency as a standing regression signal |
| **G5 Trainer-Nutzen** | Measured correction rate in review on a fresh game before EVERY auto-accept step (graduated-autonomy contract) |

**Partial shipping:** hit marker + timeline from G0 · labels below G2 = labeling assist
only (`labeling_assist_only`, never pipeline) · outcomes below G4 = `winner_side` only.

## 10. Phase 1 — measurement plan (NEXT STEP; user checkpoint before starting)

Throwaway measurements (scratchpad, no production code), from memo §11:

| # | Measurement | Answers | Effort |
|---|---|---|---|
| M1 | Audio onsets × 738 known hits (1 game) | timing precision, neighbor-court onset rate + in-window coincidence probability, first recall estimate | ~3 h |
| M2 | Manual hit count, 10–15 rallies @0.25× (trainer) | true heuristic recall per stroke type; seeds the G0 eval set | 1–2 h |
| M3 | GBDT learning curve (140/280/554) | points per label-doubling → G2 target number | 2–3 h |
| M4 | Hit-time jitter sensitivity | value of precise hit times; GBDT-vs-BST re-ranking | 1–2 h |
| M5 | winner_side vs score (1 game) | outcome baseline | ~1 h |
| M6 | Hitter baseline (nearest wrist) vs ~554 hitter labels | does attribution need a model? | 1–2 h |
| M7 | Fine-labeling trial 30–50 clips, timed (trainer) | clips/h, Unsicher rate, taxonomy practicality, **hit-marker validation** | ~2 h |
| M8 | Smash @12 vs 60 fps + checkpoint fps | 30-vs-60 decision; fps mismatch as v1 transfer brake? | ~2 h |
| M9 | A/V sync calibration (with M1) | mux offset + sound travel time | ~1 h |
| M10 | Netzroller evidence (5–10 known cases) | scope or descope | 1–2 h |

## 11. UI (built only after the relevant gates; German, du-form, no chart libs)

- **Confidence-aware event-marker overlay**: 4th SVG layer beside `SkeletonOverlay` —
  stroke label + hitter at each hit moment; confident = solid, low-confidence = muted
  „Schätzung" styling, consistent with `OverlayChips.tsx` / `VideoOverlay.tsx`; discrete
  event markers are new UI (existing tracks are continuous `[{t,u,v}]` / `trackAt()`).
- **Rally timeline strip** under the video: one tick per stroke colored by coarse class,
  click-to-seek, on the existing `vp.time`/`seek` contract. Ships from G0 (ticks without
  labels are honest earlier than labels).
- **Stroke review**: clone of the RallyReview pattern — click a stroke, fix
  label/hitter/outcome/stars; append-only `stroke_corrections`; corrections feed
  retraining and the G5 edit-rate metric. Label Studio stays for bulk labeling.
- **Active-learning loop**: model uncertainty + rare-class score drive the labeling queue;
  accept rates tracked per model version; one-command retraining cycle.
- **Statistics**: per-player per-coarse-class counts, winner/error splits,
  tactical-pattern aggregates (Aufschlag → Return → 3. Schlag …, „Punkte nach kurzem
  Aufschlag") via `ShotMixCard`/`SHOT_DE`; `shots` = `[[type,count]]` per the data model.

### Post-gate feature (F4, signed off): German rally commentary
After **G2 AND G4 pass** (not before — otherwise eloquent prose about wrong facts):
per-rally German text (Scoring/Losing/Overall reason) generated from stroke sequences +
outcomes — **structure-to-text, no video understanding** (FineBadminton's own pipeline
works exactly this way: CV delivers facts, LLM phrases, human/confidence gates).
**LLM: local Qwen3 on the Mac Studio (user decision 2026-07-13).** Always marked as
KI-Text, review-correctable, never a source of stats.

## 12. Honest accuracy expectation (estimate with verified anchors)

Anchors: SOTA ceiling 0.83–0.85 at ~33k pro singles labels · BST at ~6–8k labels: 0.69 ·
fine-18 with leakage: 0.65 · v1 local: 0.49 (pretrain, 554 labels, under hit drift).

| Stage | coarse-8, realistic | Reading |
|---|---|---|
| Session 1 (~700 labels) | ~0.55–0.65 (LOGO-CV) | below gate; labeling-assist carve-out only |
| Session 2 (~1 400, true cross-session) | ~0.60–0.70 | gate reach ONLY with all levers (G0 timing, pretrain, imbalance handling, maybe SSL) |
| Session 3 (~2 100) | ~0.65–0.75 | gate pass plausible, not guaranteed |

Same-session LOGO-CV overstates by construction — cross-session is the mandatory yardstick
from Session 2. M3 replaces this estimate band with a measured slope before numbers freeze.
Thin classes stay the bottleneck; a better hit detector likely GROWS exactly those classes
(v1's net n=20 / drive n=22 are presumed detector artifacts — M2 settles it).

## 13. Risks & open questions

1. **Hall audio SNR unmeasured** — M1 first; if impacts are inaudible, video-only stands
   and G0's ±50 ms is renegotiated. Neighbor courts ALWAYS play (user review) — „falscher
   Onset" rate is quantified, not assumed away.
2. **Doubles ambiguity rate unknown** — nowhere measured in the literature; if > 20 % of
   doubles hits are fundamentally ambiguous from one camera, G1's coverage target drops
   (honest, not guessed). Our 738-clip measurement is novel evidence either way.
3. **Class-prior shift**: v1 class counts are detector-biased; stratification and gates
   freeze only after M2.
4. **fps mismatch as silent transfer brake** (M8) — if checkpoints expect 30 fps, v1's
   +7–12 pt transfer was a lower bound.
5. **License-override residual risk** (BadmintonDB, VideoBadminton — no license grant,
   user-approved 2026-07-13): fine while non-distributed; both (and models trained on
   them) must be replaced if the app ever distributes. FineBadminton needs no override
   (Apache-2.0).
6. **Wang et al. 2022 definitions** not yet pinned (passive drop, Block) — Phase-1 task
   before labels freeze.
7. **FineBadminton paper↔release gap**: HF ships a ~62 % subset (70/120 matches); minor
   annotation noise (27 empty hit types, a few Chinese fragments) — filter on ingest.
