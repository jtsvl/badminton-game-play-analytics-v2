# Feature — Player skeleton/posture overlay (visualization)

**Status (2026-07-13): BUILT + coverage-fixed + real-footage-verified** — worker + web + backfill
shipped 2026-07-12; the 2026-07-13 coverage pass (see §Coverage) replaced the per-player join
(label→role→whole role timeline), made the export whole-video, rebuilt tracking identity
(framewise re-ID v2 + half-aware primaries) and re-backfilled all Session-2 games: skeleton
coverage went from 5–16 % of playback per player to 82–100 %, and the far court — previously
**absent from the shipped identity entirely** — now has skeletons, Laufwege and stats.
Browser-verified on Session-2 footage (4 distinct skeletons, near AND far, in and between
rallies; lazy fetch confirmed). Visualization only; derived metrics (joint angles, swing quality,
injury risk) are explicitly out of scope for this pass, but the stored shape takes them later
additively. Post-review hardening 2026-07-12: lazy web fetch, stale-row cleanup, never-fail pose
block, return_status validation.

## What it is

While a processed match plays in the report, each identified player's skeleton — joints connected
by bone lines, MediaPipe-style — animates in sync with the video, toggled by a new „Skelett" chip
(default off). The data already exists: the `pose` stage extracts RTMPose-m **Halpe-26** keypoints
per tracked player per analysis frame into `pose.parquet` (consumed today by foot grounding and
the swing cue, `features/pose-swing-fusion.md`). This feature persists the in-rally slice of it to
a web-facing table and renders it. No new model, no training, no labeling.

## Data — `pose_keyframes` (migration 0012)

`id · video_id · subject ('player:<player_id>') · format ('halpe26') · data jsonb · created_at`
— unique `(video_id, subject)`, mirroring `track_keyframes`.

`data` = `[{t, kp: [[x, y, score] × 26]}]`, one entry per kept analysis frame (whole video;
native ~12 fps in-rally, thinned to ~6 Hz between rallies — see §Gating & payload):

- `t` seconds (3 decimals) from the **frames.json manifest** (decode PTS), *never* `frame ÷ fps`
  arithmetic — pose.parquet carries only the cached-frame index.
- `kp[i] = [x, y, score]` (4 decimals): **frame-normalized image coords** `x,y ∈ [0,1]`, keypoint
  pixels divided by the frames.json manifest's native width/height (the pose stage reads
  full-resolution cached frames), **clamped to [0,1]** — RTMPose places edge-cropped joints slightly
  outside the frame. `score` = RTMPose per-joint confidence, so the renderer can fade/hide weak bones.
- **Why image coords, not court (u,v):** elevated joints (shoulders/elbows/wrists/hips) sit above
  the floor plane, where the ground homography is geometrically wrong. Follows the `rally_end
  {x,y}` / `landing {px,py}` precedent — normalized image coords on web-facing rows, never raw
  pixels (`20-data-model.md` § Conventions).
- Alignment holds against the 720p proxy the browser plays: the proxy is an aspect-preserving
  `scale=-2:720` of the same source, and undistortion is a no-op in P1. **Caveat:** if real DJI
  undistortion ever lands, the proxy must be undistorted too, or frame-space overlays drift.
- Halpe-26 index map (verified against rtmlib's `halpe26.py`): 0 nose · 1/2 eyes · 3/4 ears ·
  5/6 shoulders · 7/8 elbows · 9/10 wrists · 11/12 hips · 13/14 knees · 15/16 ankles · 17 head ·
  18 neck · 19 pelvis · 20/21 big toes · 22/23 small toes · 24/25 heels (L/R throughout).

**Gating & payload:** export is **whole-video** (user decision 2026-07-13 — skeletons between
rallies too, matching the position overlay), with a **hybrid rate**: native ~12 fps inside rally
spans (fast swings deserve full temporal detail), thinned to `POSE_KEYFRAME_HZ` (default 6 Hz)
between rallies — `poseAt`'s 0.3 s hide threshold tolerates 1/6 s spacing, so rendering never
drops out. Measured on the Session-2 re-backfill: 0.8–2.6 MB per player, 3.7–9.5 MB per doubles
match (fetched lazily on chip toggle only; Kong currently serves REST **without gzip** — noted in
`fix-later.md`). `POSE_MAX_KEYFRAMES_PER_PLAYER` (10 000) is the safety valve for very long
recordings: over it, keyframes thin by a uniform stride — **thin, never skip** (this replaced the
old `POSE_UNGATED_MAX_FRAMES` skip-cap, which is gone along with rally gating).

## Worker wiring

`overlay._pose_keyframes` (next to `_shuttle_keyframes`, same graceful-absent pattern): returns
`None` when `pose.parquet` **or** `frames.json` is missing after trying the artifact-bucket
restore (`remap.resolve_artifact`), so pre-pose matches skip silently instead of failing the
stage. **Identity join (the 2026-07-13 coverage fix):** `match_players.track_label` names ONE raw
tracker id, but a player's timeline in tracks.parquet spans many ids (re-ID fragments keep their
own `track_id`) — so consumers bind label → that tid's **role** → every row of that role
(`pipeline/identity.py: role_players`), and pose rows ship iff their `(frame, track)` is a
non-interp row of a bound role's timeline (`role_frame_index` — the frame-level gate doubles as
the court gate; interp rows have no detection behind them, so the join can't fabricate a
skeleton). The same binding fixed the position overlay and `aggregate`'s movement stats, which
had silently dropped every re-ID row too. Writes via `store.upsert_pose_keyframes`
(`on_conflict=video_id,subject`).

Hardening (post-review, 2026-07-12): the whole pose block is wrapped so **a pose failure can never
fail the stage** (skeletons are optional; tracks/stats are already written). After a successful
write, **stale subjects are deleted** (`store.delete_pose_keyframes` keep-list) — a re-label can
move a player to a track without pose rows, and the web would otherwise keep drawing the old
skeleton. On a **remap with missing pose inputs** (evicted cache) all pose rows for the video are
cleared: better no skeleton than one that keeps the old, wrong track→player binding.

**Remap slice change:** the `remap` slice now honors `params.return_status` (default `done`,
matching `rally_review`'s existing convention) so the backfill can re-run overlay on `review`
matches without flipping them past the PlayerReview gate. Validated: only `review`/`done` may be
written (a typo'd status would hide the match from every status-gated view; invalid → `done` +
warning); `unchanged` leaves the match status untouched — for backfills that could race a trainer
confirmation.

**Backfill:** 2026-07-12 — one `remap` job per Session-2 match populated the first skeletons.
2026-07-13 — one **`retrack`** job each (new slice: `[gate_track, rally_sync, aggregate,
overlay]`, `return_status: unchanged`) rebuilt tracking identity and rewrote stats + both
overlays + skeletons; gate_track resolves `detections.parquet` from cache or the artifact bucket
(`remap.resolve_artifact`), so no re-detect. Excluded: the mock seed match and the 30 s test clip
without a local cache (skips gracefully).

## Web wiring

- `lib/pose.ts`: `poseAt(keys, t)` — **nearest-frame lookup**, no interpolation (blended limbs can
  be anatomically impossible); hides the skeleton when the nearest keyframe is > 0.3 s away
  (out-of-rally / track gap) instead of freezing it.
- `SkeletonOverlay.tsx` (sibling of `VideoOverlay.tsx`, same SVG shell — viewBox `0 0 192 108`,
  `preserveAspectRatio="none"`): renders `x*192, y*108` directly, **no homography**. Bones with an
  endpoint `score < 0.3` are hidden; opacity fades with confidence. Player colors follow the
  existing overlay assignment (brand orange / opponent blue). Face points (0–4) are not drawn;
  feet are (ankle→big toe/small toe/heel), per the trainer's choice of a full-foot skeleton.
- Plumbing: **lazy fetch** via `usePoseTracks(videoId, ov.skelett)` — pose data (hundreds of KB)
  is only downloaded when the „Skelett" chip is first toggled on, never eagerly with the report;
  fetched once per video, kept for the session. `PoseKeyframeRow` in `types.ts`,
  `MatchReport.video.id` carries the video id to the hook; demo matches have no pose rows — the
  chip shows nothing. `skelett` in `OverlayToggles` (localStorage merge backfills old clients),
  one chip in `OverlayChips.tsx`, hook + overlay mounted at **both** sites (mobile
  `VideoAnalysisCard`, desktop `MatchReportDashboard`).

## Coverage (measured, 2026-07-13 — `worker.eval_pose_coverage`)

Metric: % of analysis frames with a renderable skeleton per player (a keyframe within 0.3 s —
exactly `poseAt`'s hide threshold), **wall-clock** basis with the in-rally split alongside.
Harness: `uv run python -m worker.eval_pose_coverage --slug <slug>` (columns: DB / would-export /
role-timeline ceiling; `--recompute` rebuilds identity from cached detections with the current
gate_track code to measure a tracking change offline, before any pipeline run).

Root cause of the original 5–16 % coverage (three stacked problems, all fixed):
1. **Consumer join**: all three consumers filtered rows by `track_id == track_label`, dropping
   every re-ID fragment row (≈half of tracks.parquet) — each player kept exactly one contiguous
   window, so at any moment ~1 skeleton was visible.
2. **Stitching v1** rejected any fragment whose frame-span overlapped the primary's (in doubles
   nearly everything overlaps a sparse primary) and matched whole-track medians — most real
   detections stayed unattached.
3. **Primary selection** (the far-court killer): global top-N-by-count + relative role_assign
   picked 18 of 20 Session-2 role slots from NEAR-half tracks — games 1–3 had **all four** roles
   on near tracks, i.e. the far players were absent from every shipped stat/overlay/skeleton and
   "B1/B2" data was near-court people. Fixed by half-aware coexisting-pair primaries
   (`geometry.select_role_primaries`) + framewise re-ID v2 (`stitch_fragments_framewise`:
   per-frame hole filling, mutual-exclusion collisions, velocity-bounded median boundaries,
   corpse-anchor discard, identity-by-elimination, fixpoint iteration).

Session-2 after the retrack backfill (wall-clock % / in-rally %, DB = shipped):

| game | A1 | A2 | B1 (far) | B2 (far) |
|---|---|---|---|---|
| game1 | 99.5 / 99.7 | 97.6 / 95.1 | 91.9 / 94.3 | 96.0 / 97.6 |
| game2 | 81.7 / 86.4 | 92.2 / 90.2 | 86.2 / 91.9 | 89.2 / 93.4 |
| game3 | 99.5 / 100 | 96.5 / 95.6 | 89.2 / 93.3 | 90.7 / 91.3 |
| game4 | 90.1 / 90.7 | 93.8 / 93.3 | 85.2 / 90.9 | 85.6 / 95.4 |
| game5 | 98.0 / 100 | 100 / 100 | 94.4 / 96.5 | 88.0 / 83.2 |

Baseline before the fix: 5.3–16.4 % wall / 16.4–53.8 % in-rally, one window per player. Measured
detection ceilings (raw in-court detections): ~97 % near / ~90 % far per player — far court is
detection-recall-capped (deferred, `fix-later.md`); targets agreed with the trainer 2026-07-13:
near > 90 %, far ≥ 75 % committed with best effort toward the ~90 % ceiling.

## Honest limits

- A skeleton exists only where a **real detection** was identity-resolved — occlusions, missed
  far-court detections and honestly-ambiguous fragments (see below) show as dropouts,
  deliberately (a frozen or interpolated skeleton would lie).
- **Doubles near-pair ambiguity:** when both near roles are simultaneously untracked, a lone
  center-court fragment can be either partner — geometry alone can't tell, so it is dropped
  (honest hole) rather than coin-flipped; a wrong-color skeleton would also corrupt that
  player's stats. This is the main residual coverage cost (e.g. A1 game2 81.7 %).
- **Retrack re-seeds `track_label`s** (`_auto_map`): after a tracking-identity rebuild the
  A/B-side placeholder binding is the near=A / far=B convention again — the trainer's
  PlayerReview (FR-35) remains the identity gate and must be re-confirmed per match.
- Keypoints are **2D image projections**. Any future joint angle derived from them is a 2D
  estimate, not a true 3D angle — same „Schätzung" framing as all monocular metrics.
- Fingers/grip are not tracked (wrist is the last point); the **racket is not tracked at all**.
- Per-joint `score` is RTMPose confidence, not ground truth; low-contrast/far-court frames degrade.
- Desktop layout crops to 16:9 (`objectFit: cover`) — exact alignment relies on 16:9 sources,
  which all current footage is.

## Out of scope (explicit)

- Joint angles, swing-quality scores, footwork metrics, injury-risk indicators (future: additive
  `angles` key per frame or nullable jsonb column — no rework of `kp`/`format`).
- Racket detection, grip analysis.
- Any change to stroke classification (separate in-flight work) or to the pose stage itself.

## Verification

Unit: synthetic pose rows + manifest + tracks → identity join/normalized/rounded/thinned shape
(worker, `test_overlay_pose.py`, `test_identity.py`; stitching + primaries in
`test_geometry.py`, `test_gate_track.py`); `poseAt` fixtures incl. max-gap hide (web,
`pose.test.ts`). Coverage: `eval_pose_coverage` per game vs the §Coverage table. E2E
(2026-07-13): retrack all 5 games → DB ≡ export ≡ ceiling columns; browser playback with
„Skelett" on shows 4 distinct skeletons — near AND far court, during and between rallies,
identity stable across the game (checked 0:34 / 1:40 / 5:00).
