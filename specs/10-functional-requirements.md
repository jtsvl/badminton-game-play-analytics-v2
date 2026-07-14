# 10 — Functional Requirements

IDs are stable and referenced from code, the data model, and the design handoffs. Status: `todo` until
implemented + verified. Three IDs are pinned to match the handoff READMEs: **FR-5** (homography),
**FR-11** (heatmap grid), **FR-22** (unknown shot type).

## Ingestion & matches
- **FR-1** Trainer uploads a video clip (iPhone/DJI; H.264/HEVC) to Supabase Storage and creates a `match`.
- **FR-2** A match has discipline (`Einzel`/`Doppel`), date, court number, and 2 or 4 named players assigned to sides A/B.
- **FR-3** A match may have multiple videos (`match_videos`), each tagged `angle` = `overhead` | `side`.
- **FR-4** Upload creates a `processing_jobs` row; the worker claims and advances it through stages with live status.

## Court calibration
- **FR-5** Each video is calibrated to a **homography** mapping image pixels → metric BWF court coordinates
  (doubles 13.40 m × 6.10 m; singles width 5.18 m). Stored in the **`court_calibrations`** table as a review
  state machine (`proposed→needs_review→accepted→finalized→rejected`) holding `proposed_points`/`confirmed_points`,
  `homography`, `mask_polygon_px` (the in-court gating polygon, FR-10), `reprojection_error_m`, and `auto_accepted`.
  Court coords are normalized `u,v ∈ [0,1]` for the frontend overlay.
- **FR-6** The system **auto-proposes** the target court's 4 corners; the trainer can **drag to correct** them and
  **pick the court** when multiple are visible (decision #4). Auto-suggest may be empty/declined → fully manual.
- **FR-7** Calibration is per video/setup (camera position varies); never assume a fixed rig.
- **FR-8** DJI ultrawide footage is lens-undistorted before any geometric computation.

## Player detection, tracking & movement insights
- **FR-9** Detect persons per frame and track them with stable IDs across the clip.
- **FR-10** **Court gating**: keep only tracks whose foot point projects inside the calibrated court polygon
  (+ margin, temporally smoothed) — this is how adjacent-court players and spectators are excluded.
- **FR-11** Produce a **player position heatmap** on a metric court grid (**0.25 m cells**); stored both as the
  frontend's gaussian-kernel form `heat:[{x,y,w,s}]` and (optionally) a downsampled raster `heatmap_uri`.
- **FR-12** Per player: `distance_m`, `avg_kmh`, `top_kmh`, `sprints` (speed-threshold bursts) from court-space motion.
- **FR-13** Per player: `zones` = % active time in front/mid/rear and left/right (sums ≈100 within rounding).
- **FR-14** Per player: `timeline` = distance per 5-minute bucket (N buckets ≈ `durationMin/5`), excluding rest.

## Shuttle tracking
- **FR-15** Detect & track the shuttle per frame → 2D trajectory `[{t,u,v,visibility}]`.
- **FR-16** **White-shuttle-vs-white-wall** robustness is required: background-median subtraction, motion cues,
  trajectory inpainting, and targeted fine-tuning (ML spec §5). Missed-frame gaps are interpolated within a
  ballistic arc only, never across a racket contact.
- **FR-17** Shuttle speed (`fastest`, per-shot speeds) is computed via homography + fps and **labeled an estimate**
  with uncertainty; side-angle clips give better speed.

## Rally segmentation & hit detection
- **FR-18** Segment the clip into **rallies** (serve → point) from shuttle continuity + player motion + serve cue.
- **FR-19** Detect **hit frames** (shuttle direction reversal/speed spike near a player, fused with a swing cue).
- **FR-20** Aggregate rally stats: `count`, `avgShots`, `avgSec`, longest (`longShots`,`longSec`), and the
  rally-ending shot summary (`endLine`, e.g. „Der Smash beendete 31 % aller Ballwechsel.").

## Stroke / shot classification
- **FR-21** Classify each hit into a shot type from the BWF-derived taxonomy, output per-player `shots:[[type,count]]`.
- **FR-22** Hits that cannot be confidently classified are bucketed as **`unknown`**, surfaced in the UI as
  „Sonstige" **only when count > 0**.
- **FR-23** Ship a coarse taxonomy first for reliability; finer classes are a later refinement.
  Coarse-8 as signed off 2026-07-13 (`features/stroke-classification-v2.md` §4): Aufschlag, Clear,
  Smash, Drop, Drive, Netzspiel, Lift, Abwehr. *(Earlier revision listed a coarse-7 without Abwehr
  and with „Netzdrop" as the coarse name — superseded by the DBV-checked taxonomy.)*

## Pose / footwork (P3)
- **FR-24** Estimate 2D pose for each in-court player (top-down), feeding stroke classification and footwork metrics.
- **FR-25** Derive footwork/gesture signals (lunge, recovery) — surfaced cautiously; absolutes flagged as estimates.

## Doubles (P3+)
- **FR-26** For doubles, produce `pair_match_stats`: front-back % (`fb`), side-by-side % (`ss`), rotation count (`rot`).

## Overlays
- **FR-27** Produce downsampled (~10 Hz) court-coord keyframes per player and for the shuttle, plus hit events,
  to drive the video overlays (tracking boxes, movement trails, shuttle path, zone tint) via the real homography.

## Web app — reports
- **FR-28** Render a per-match report from one data fetch in two layouts: mobile stacked (<1200 px) and desktop
  dashboard (≥1200 px), per the handoffs; player tabs switch the active player and re-render all stats.
- **FR-29** All UI is German, du-form; numbers `de-DE`-formatted; brand tokens exact.
- **FR-30** The video card renders only when the match has a processed video; overlays toggle (tracking/Laufwege/
  shuttle/zonen); playback position persists per match, overlay prefs per device.
- **FR-31** Match list shows processed matches newest-first with date, opponents, discipline, result chip, score.

## Auth, sharing, review
- **FR-32** Trainer/admin authenticates (Supabase Auth) to upload, calibrate, review, publish. Admin identity is
  the `admin_users` allowlist; RLS write/read for admins is gated by `public.is_admin()` (not "any authenticated user").
- **FR-33** Each player opens their report via a **personal link** (magic-link or signed slug); tabs order so they are „Du".
- **FR-34** RLS: players read only their own match reports; only admin/service-role writes.
- **FR-35** **Review/repair UI**: before publish, fix player→identity assignment, drop/merge a rally, relabel a shot.
  The player→identity mapping slice is built (`/match/:slug/review`, spec `features/player-identity-review.md`):
  the trainer confirms/corrects each `match_players` (side,slot)→(`player_id`,`track_label`) binding after
  the worker's role-based estimate, then a `remap` job recomputes stats/overlays. Rally drop/merge is built
  (`/match/:slug/rallies`, spec `features/rally-shot-review.md`, `rally_review` job slice); shot relabel
  stays `todo` until stage 7 (`stroke`) exists.
- **FR-36** A player can request data deletion (footer copy promises this); requests are tracked in `deletion_log`.
- **FR-38** Player consent is recorded (`players.consent_date`); processing/sharing respects it (GDPR, minors).

## Fine-tuning loop (P4, supporting)
- **FR-37** A self-hosted labeling loop (CVAT/Label Studio) exports labels to fine-tune shuttle/detector/stroke
  models on the club's own footage, incl. white-wall cases. Footage stays local.
