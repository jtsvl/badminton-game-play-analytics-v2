# Feature — Frontend Port (mobile Spielbericht + desktop Video-Dashboard)

Status: in progress · Branch: `feat/frontend-port` · Depends on: P0 foundations, `specs/50-web-app.md`.

## Goal
Port the two pixel-final Claude Design handoffs to TypeScript React, **pixel-close**, driven by the
**live seeded Supabase data** (no CV/ML). One route `/match/:slug`, **one data fetch**, layout branched
on viewport via `useMediaQuery('(min-width:1200px)')`:
- `<MatchReportMobile>` — the mobile **Spielbericht** stacked column (`handovers/design_handoff_match_report_app`, theme **A** only).
- `<MatchReportDashboard>` — the desktop **Video-Dashboard** (`handovers/design_handoff_desktop_dashboard`, **only** `DeskVideoDashScreen`).

## Sources of truth
- Mobile: `handovers/design_handoff_match_report_app/{viz,screens,video}.jsx`. Use theme A values.
- Desktop: `handovers/design_handoff_desktop_dashboard/{desktop-screens,desktop-shared}.jsx` → only `DeskVideoDashScreen` + its building blocks.
- Data shapes: `specs/20-data-model.md` / `web/src/types.ts`. Localize canonical values via `web/src/lib/labels.ts`.

## Components (all under `web/src/components/report/` unless noted)
Primitives: `Icon` (inline SVG set — no lucide), `CountUp` (de-DE, settle-exact, reduced-motion), `Reveal` (IO + 900 ms fallback + reduced-motion).
Viz: `CourtHeatmap` (+ `heatGrid`) + `HeatLegend`, `CourtCoverageCard`, `ZoneCard`, `ActivityCard`, `ShotMixCard` (+ `compact`), `FormationCard`, `RallyCard`, `Endstand`, `PlayerTabs`, `SafariFrame`. Extend `KeyStats` with a `variant` ('cards' | 'bare') and `SectionCard` already has `dense`.
Video: `useVideoPlayer(slug)` (hook), `VideoOverlay` (shared SVG engine), `OverlayChips`, `VideoControls`, `VideoAnalysisCard` (mobile chrome). Desktop composes the surface inside `SafariFrame`.
Layout: `MatchReportMobile`, `MatchReportDashboard`, `ReportHero`; `pages/MatchReport.tsx` becomes a thin branch with lifted `tab` state.

## Libs (`web/src/lib/`)
- `homography.ts` — **production projection engine** replacing the prototype `vidProject`/`VID_QUAD`:
  `solveHomography(srcQuad, dstQuad)` (4-point DLT) + `projectUV(H, u, v) → {x, y}` (court u,v∈[0,1] → frame fraction 0..1).
- `overlay.ts` — `trackAt(keys, p)` (smoothstep over `[{t,u,v}]`) + `shuttleAt(p, near, far)` (alternating hits).
- `demoOverlay.ts` — the **fallback for uncalibrated (seeded) matches** (clearly labeled): court-corner↔frame
  correspondences derived from the prototype baseline quad → `H_DEMO = solveHomography(...)`; demo near/far
  tracks + `DEMO_LOOP`. The demo feeds the *same* `projectUV` engine production uses. **Done (P1, commit
  22fc7e7):** the live path is `calibration.ts` `overlayInputs` — it derives the court→frame homography from
  `court_calibrations.confirmed_points` and uses real `track_keyframes` whenever a calibration exists. (We do
  not read the stored `court_calibrations.homography`: it's the inverse direction, image-px→u,v, for worker
  gating; the web re-derives court→frame from the 4 corners.)
- `storage.ts` — `resolveVideoSrc(path)`. **Done (P1, commit 22fc7e7):** returns a signed URL via
  `supabase.storage.from('match-videos').createSignedUrl(path, ttl)` (1 h TTL), or `''` when no footage exists
  → the player then falls back to the synthetic-clock placeholder. Dev anon storage read is granted (test
  footage only) by migration `0003_dev_calibration_write.sql`.

## Decisions / deviations from the prototype (keep in sync with `specs/50-web-app.md`)
1. **Heatmap default style = `raster`** (12×13), per `specs/50-web-app.md` — overrides the prototype theme-A `heat:'blobs'`.
2. **Projection = a single planar homography** (the correct court model). It can drift slightly from the
   prototype's hand-tuned depth `k`; this is intentional. We never ship `VID_QUAD` as the engine.
3. **Video = real-or-placeholder.** With footage, the player loads the signed-URL `<video>` and reads its
   clock; without it (seeded demo matches) it falls back to a black surface + **synthetic 16 s clock** so the
   demo overlays still animate. Real signed-URL footage + real `track_keyframes` landed in P1 (commit 22fc7e7);
   the synthetic clock + demo tracks remain only as the uncalibrated fallback.
4. **`ShotMix compact` and `KeyStats variant='bare'` are net-new** (the prototype passes `compact` but the
   component ignores it). Designed against the desktop dashboard tile sizing.
5. **Doubles renders as-is**: no shots → empty-state Schlagverteilung tile (keeps the 4-tile desktop row);
   no video → a "keine Videoanalyse" placeholder fills the span-8 region so the grid doesn't collapse. No new seed data.

## Correctness guards (our hook's shapes differ from the prototype mock)
- `report.pairs` is `[]` not `null` → gate doubles UI on `pairs?.length`.
- `report.rallies` can be `null` → guard the Ballwechsel block + Endstand rally extra line.
- `stats.shots` can be `[]` → `ShotMixCard` early-returns an empty state (no `sorted[0]`/`NaN`).
- `meta.result` is `'win'`/`'loss'` (not `'S'`/`'N'`) → `ResultChip` takes the code and computes `won`.
- HTML entities (`&thinsp;`, `&prime;`) → unicode (` `, `′`). Salt `CourtHeatmap` SVG ids via `useId`.

## State
- `tab`/`setTab` lifted to `pages/MatchReport.tsx` (survives a viewport-crossing remount). Both layouts take it as props.
- Video/overlay state in `useVideoPlayer(slug)`, one instance per mounted layout. Position persists in
  `localStorage['bca-vt-'+slug]`; overlay toggles in `localStorage['bca-ov-v1']` (defaults tracking+wege on).

## Tests
- TDD: `homography` (identity + known quad + round-trip), `overlay` (`trackAt`/`shuttleAt`), `heatGrid` (counts/normalization/peak/suppression).
- One render smoke test per layout against a fixture report (catches the null/empty-shape crashes above).

## Verification
- `cd web && npm run typecheck && npm test && npm run build` green.
- `npm run dev`; open `/match/2026-06-10` (Einzel) and `/match/2026-06-03` (Doppel); compare mobile (<1200px)
  and desktop (≥1200px) to `handovers/**/Spielanalyse*.html`. Confirm anon read works (migration `0002_dev_anon_read`).
