# 50 — Web App

Vite + React 18 + TypeScript + react-router-dom + `@supabase/supabase-js`. Ports the `handovers/` designs
(pixel-final). Custom SVG viz (no chart lib). German, du-form, light mode, BC Rheinbach tokens.

## Routes
- `/` — **Spielliste** (match list; processed matches newest-first).
- `/match/:slug` — **Spielbericht** (one data fetch → mobile stacked OR desktop dashboard by viewport).
- `/upload` — admin upload (auth) → create match + video + job.
- `/match/:slug/calibrate` — calibration corner-drag UI (auth) (FR-6).
- `/match/:slug/review` — review/repair UI (auth) (FR-35).
- Personal report link → `/match/:slug?p=<token>` orders tabs so the linked player is „Du".

## Layout branch (critical)
One route, **one data fetch**, branch layout on viewport (`matchMedia('(min-width:1200px)')`):
- `useMatchReport(slug)` hook → loads rows, runs `buildPlayers`, derived values, returns `{data, tab, setTab, ...}`.
- `<MatchReportMobile/>` (stacked column, design width 390) — the mobile handoff.
- `<MatchReportDashboard/>` (12-col grid, video-dominant, Safari frame) — the desktop handoff (`DeskVideoDashScreen`).
Do **not** duplicate the data layer or add a second route.

## Component port map (from handovers/)
Shared viz → `src/components/report/`:
`CourtHeatmap`, `ZoneBlock/ZoneCard`, `TimelineChart/ActivityCard`, `ShotMix/ShotMixCard` (+`dense`/`compact` prop),
`FormationBlock/FormationCard`, `KeyStats`, `RallyCard`, `CourtCoverageCard`, `VideoAnalysisCard` (player + SVG
overlays + controls), `SectionCard`, `CountUp`, `Reveal`, `Icon`.
New for desktop: top bar, **Endstand (scoreboard)** card, **SafariFrame**, the dashboard grid.
**Do NOT port:** tweaks panel, dropped desktop directions, design-canvas, the hand-tuned `VID_QUAD` (use real homography).

## Libs
- `src/types.ts` — TS types from `20-data-model.md` (the report object + row types).
- `src/lib/supabase.ts` — client (anon key; signed URLs for video).
- `src/lib/format.ts` — `de-DE` number/date formatting (decimal comma, thousands dot, `M:SS` time).
- `src/lib/labels.ts` — canonical→German display maps (`discipline` Einzel/Doppel, `shot_type` Clear/Netzdrop/…, result Sieg/Niederlage). DB stores canonical English; localize only at render.
- `src/lib/score.ts` — `parseScore`, `formatGames`, `resultForSideA` (Endstand cells, result chip).
- `src/lib/homography.ts` — project court `u,v` → frame px for overlays (replaces prototype `vidProject`/`VID_QUAD`).
- `src/lib/overlay.ts` — `vidTrackAt` interpolation over `track_keyframes` (`[{t,u,v}]`, smoothstep).

## Viz data shapes (consumed as-is)
`CourtHeatmap(spots=[{x,y,w,s}])` · `ZoneBlock(zones={front,mid,rear,left,right})` ·
`TimelineChart(points=[n,...])` · `ShotMix(shots=[[name,count]])` · `FormationBlock(pairs=[{label,fb,ss,rot}])`.
Heatmap raster = 12×13 cells, hottest (>85 %) → `#C0392B`; default style `raster`.

## Video & overlays (FR-27/30)
`<video muted playsInline preload="metadata">`; restore playback position from `localStorage['bca-vt-'+slug]`
(else seek 1 s poster); persist every second. Overlay = absolute SVG `viewBox 0 0 192 108`,
`preserveAspectRatio=none`; geometry computed in court coords and projected via the **real homography** from
`match_videos.calibration`. Toggles (Tracking/Laufwege/Shuttle/Zonen) persist in `localStorage['bca-ov-v1']`
(defaults: tracking+Laufwege on, Shuttle+Zonen off). Player colors: viewer/near `#FA8832`, opponent/far `#1758A6`.

## Brand tokens (exact)
primary `#FA8832` · bubble `#C66C27` · dark-on-orange `#1A0500` · beige `#FEF4EA` · ink `#0A0A0B` ·
muted `hsl(215.4 16.3% 46.9%)` · hairline `hsl(214.3 31.8% 91.4%)` · opponent `#1758A6` · heat-max `#C0392B`.
Fonts: Space Grotesk (headings/numerals) + Inter (body) via `@fontsource-variable`. Radii cards 12 / Endstand 16 /
pills 99. Shadows: rest `0 1px 2px rgb(0 0 0/.05)`; bubble `0 2-3px 0 #C66C27` (orange only).
Animations „dezent": Reveal (opacity+translateY, ~.5s, 60 ms stagger, IO + ~900 ms fallback), CountUp (700 ms,
settle exact), respect `prefers-reduced-motion`.

## Auth (FR-32/33/34)
Supabase Auth for admin (upload/calibrate/review/publish). Player report links = signed token granting read of
one match; tab order derived from the token's player. RLS enforces read scope; only admin/service-role writes.

## Implementation notes (frontend-port slice — `specs/features/frontend-port.md`)
Done on `feat/frontend-port`, driven by the live seeded DB. Deviations from the prototype:
- **Heatmap default = `raster`** (12×13), overriding the prototype theme-A `heat:'blobs'`.
- **Projection engine** `lib/homography.ts` (`solveHomography` 4-pt DLT + `projectUV`) replaces
  `vidProject`/`VID_QUAD`. `lib/overlay.ts` = `trackAt`/`shuttleAt`. `lib/demoOverlay.ts` holds demo
  corner↔frame correspondences (→ `H_DEMO`) + demo tracks, clearly labelled; **TODO P1**: swap to
  `court_calibrations.homography` + `track_keyframes`. A planar homography is the correct court model and
  may drift slightly from the prototype's hand-tuned depth `k` (intentional).
- **Video = placeholder backdrop**: `lib/storage.resolveVideoSrc` returns no playable URL this slice
  (buckets private, demo clip absent); `useVideoPlayer` runs a **synthetic 16 s clock** so overlays animate
  over a black surface. **TODO P1**: signed URL via Storage. The shared overlay engine is `VideoOverlay`
  (consumed by the mobile card and the desktop `SafariFrame` surface).
- **New variants**: `ShotMixCard` `compact` (tighter rows, no per-row %) and `KeyStats` `variant='bare'`
  (desktop Kennzahlen tile) — net-new, not in the prototype. `SectionCard` gained `foot`/`style`/`dense`.
- **Empty/`null` shapes** the prototype assumed away are guarded: `pairs:[]` (gate on `.length`),
  `rallies:null`, `shots:[]` (empty-state tile), `result` is `'win'`/`'loss'`. Doubles with no video shows a
  "keine Videoanalyse" placeholder in the desktop span-8 so the grid doesn't collapse.
- **Data-layer fix**: `useMatchReport` queried a non-existent `rallies.ending_shot_type`; the column is
  `end_shot_type` (fixed), and the rally `endLine` shot name is localised via `shotDE`.

## Verification
Seed the DB from `seed.sql` (mirrors `mock-data.js`) and compare against `Spielanalyse App.html` /
`Spielanalyse Desktop.html` for pixel fidelity; check German formatting and the layout breakpoint.
Verified `2026-06-10` (Einzel) + `2026-06-03` (Doppel) at <1200px and ≥1200px against the live DB:
`npm run typecheck` + `npm test` + `npm run build` green; anon read OK (migration `0002_dev_anon_read`).
