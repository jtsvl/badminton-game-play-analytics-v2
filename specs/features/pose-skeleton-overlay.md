# Feature — Player skeleton/posture overlay (visualization)

**Status (2026-07-12): BUILT + real-footage-verified** — worker + web + backfill shipped in one
pass; browser-verified on Session-2 footage (skeletons track limbs in both layouts; lazy fetch
confirmed: 0 downloads until the chip is toggled). Visualization only; derived metrics (joint
angles, swing quality, injury risk) are explicitly out of scope for this pass, but the stored
shape takes them later additively. Requested by the trainer as an independent workstream next to
stroke classification (which this feature does **not** touch or depend on). Post-review hardening
same day: ungated-payload cap, lazy web fetch, stale-row cleanup, never-fail pose block,
return_status validation.

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

`data` = `[{t, kp: [[x, y, score] × 26]}]`, one entry per in-rally analysis frame:

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

**Gating:** in-rally spans only (`aggregate.filter_to_spans` on the manifest-mapped `t`), at the
native analysis rate (~12 fps) — rallies are short; measured on the real Session-2 backfill:
343–601 KB per doubles match total. **Ungated pose (no rallies / gating emptied) ships only for
short clips** (`POSE_UNGATED_MAX_FRAMES`, ≈2.5 min of analysis frames): unlike the stats'
whole-clip fallback, an ungated whole-match skeleton row is a multi-MB payload hazard, not a
safety net — over the cap the stage skips pose with a warning. Thinning to ~6 Hz stays the volume
knob if a future match proves too heavy — measured, not preempted.

## Worker wiring

`overlay._pose_keyframes` (next to `_shuttle_keyframes`, same graceful-absent pattern): returns
`None` when `pose.parquet` **or** `frames.json` is missing, so pre-pose matches and evicted caches
skip silently instead of failing the stage. Join `pose.track → match_players.track_label`
(`by_label`, the same dict the player keyframes use): only the **primary** track per role carries
a label, so re-ID fragment rows drop out — identical to the existing player-overlay behavior — and
the join doubles as the court gate (pose.parquet contains every tracked person in the hall).
Writes via `store.upsert_pose_keyframes` (`on_conflict=video_id,subject`).

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

**Backfill (done in this pass):** the five fully-processed Session-2 matches all have
`pose.parquet` + `frames.json` cached; one `remap` job each (with `return_status` = current
status) populates their skeletons. Excluded: the mock seed match and the 30 s test clip without a
local cache (skips gracefully).

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

## Honest limits

- Skeleton exists only where the **primary track** was detected — re-ID-bridged gaps and occlusions
  show as skeleton dropouts, deliberately (a frozen or interpolated skeleton would lie).
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

Unit: synthetic pose rows + manifest + spans → gated/normalized/rounded shape (worker,
`test_overlay_pose.py`); `poseAt` fixtures incl. max-gap hide (web, `pose.test.ts`). E2E: remap
one real match → row lands with coords ∈ [0,1] and ~MB-scale payload, status preserved; browser
playback with „Skelett" on tracks the visible players' limbs in both layouts.
