# Fix-later / backlog

A running list of known issues and improvements we've consciously deferred, so they don't get lost.

## Far-court top-speed inflation (P1 movement tier)
The two players furthest from the camera show an inflated **top speed** (~30–34 km/h on the
`2026-06-14-doppel` clip); near players are believable (~7–25 km/h). Cause: the far baseline is only
~610 px wide → very few pixels per metre, so ankle-point jitter is amplified there, and
`top_kmh = max(step speeds)` is the most outlier-sensitive statistic possible (a single bad frame-pair
sets it). It's the inherent monocular far-court noise, not a calibration error.

**Fix:** (1) robust top speed — use a high percentile (e.g. 98th) of the smoothed step-speeds instead of
`max`; (2) stronger trajectory smoothing (`worker/worker/geometry.smooth_track` window 5 → ~9, ~0.75 s).
Both TDD-able in `worker/tests/`. Won't flatten genuine bursts.

_Raised 2026-06-17 while reviewing the trainer's real calibration._

## Automatic court detection (deferred — build-plan P4)
Today calibration is fully **manual**: the calibrate screen drops a generic starting trapezoid and the
trainer drags the 4 corners onto the court (the magnifier helps place them precisely). There is **no
automatic court detection yet** — the "far off" the trainer noticed is just that generic starting guess.
Auto-detect-with-manual-correction is locked decision #4, scheduled for P4. Likely approaches when we get
to it: CourtKeyNet (MIT) corner proposal, and/or reuse the player foot-point cloud we already compute to
pick the *right* court in the multi-court hall, then snap corners to nearby lines.
Deferred per the trainer 2026-06-17 — revisit later.
