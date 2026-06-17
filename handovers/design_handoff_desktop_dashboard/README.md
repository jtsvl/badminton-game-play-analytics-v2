# Handoff: Spielbericht — Desktop-Layout („Video-Dashboard") · BC Rheinbach

## Overview

This is the **desktop UX for the match-report page** — the large-screen counterpart to the
mobile "Spielanalyse" app that was already handed off and (partly) built in `web/`. Same
data, same brand, same components — **a different layout** tuned for a wide screen.

The chosen direction is the **"Video-Dashboard"**: a single, non-scrolling screen (designed at
**1440 × 900**) where the **gameplay video dominates** inside a Safari-style browser frame on the
left, with the final score (Endstand) and a large court heatmap stacked to its right, and a row
of four compact stat tiles underneath. Everything a player wants after a match is visible "at a
glance" without scrolling.

It is **one of five** desktop directions that were explored; the other four (Matchcenter
Split-View, Vereinsportal editorial, Analyse-Studio, plain Dashboard) were dropped. Only the
Video-Dashboard ships. The prototype source still contains the dropped screens (`desktop-screens.jsx`)
— **implement only `DeskVideoDashScreen`**.

## About the Design Files

The files in this bundle are **design references created in HTML/JSX** — an interactive prototype
showing the intended look and behavior, **not production code to copy verbatim**. The task is to
**recreate this layout in the existing target codebase**: the `web/` app of the
`badminton-game-play-analytics` repo (Vite + React 18 + TypeScript + Supabase + react-router-dom).

Crucially, **this desktop view is the same report as the existing mobile `MatchReport`**, just laid
out for a wide viewport. Almost every building block already exists in
`web/src/components/report/*` and should be **reused, not rebuilt**:

| Mobile component (exists) | Reused in desktop dashboard as |
|---|---|
| `report/VideoAnalysisCard.tsx` (video + SVG overlays + controls) | the big left video — but **re-skinned into a Safari frame** (see §Safari frame) |
| `report/KeyStats.tsx` | the "Kennzahlen" tile (2×2), **denser** variant |
| `report/ZoneCard.tsx` | "Spielfeld-Zonen" tile |
| `report/ActivityCard.tsx` | "Aktivität" tile |
| `report/ShotMixCard.tsx` | "Schlagverteilung" tile (**compact** variant) |
| `report/CourtCoverageCard.tsx` | "Feldabdeckung" (large heatmap, right column) |
| `report/RallyCard.tsx` | folded into the Endstand card's sub-line (or its own tile) |
| `report/FormationCard.tsx` | doubles only — add as a tile/row when `pairStats` present |
| `components/SectionCard.tsx`, `CountUp.tsx`, `Reveal.tsx`, `Icon.tsx` | unchanged |

**New pieces to build** (don't exist yet): the **desktop top bar**, the **Endstand (scoreboard)
card**, the **Safari browser frame** wrapper, and the **dashboard grid layout** itself.

## Fidelity

**High-fidelity.** Colors, typography, spacing, radii, copy and interactions are final — recreate
pixel-perfectly using the brand tokens already in `web/src/styles.css` /
`@fontsource-variable/{inter,space-grotesk}`. The prototype's `mock-data.js` shows the exact target
shapes/formatting (German locale, units) but must be fed by the real Supabase queries that
`MatchReport.tsx` already runs.

## Responsive strategy — desktop sits NEXT TO mobile

This is the **most important architectural note.** The mobile report (stacked column, design width
390 px) already exists and must keep working. The desktop dashboard is an **additional layout for
wide viewports**, not a replacement. Recommended approach:

- Keep one route (`/match/:slug`, `MatchReport`) and **one data-fetch** (the existing
  `useEffect` Supabase block — do not duplicate it).
- Branch the **layout** on viewport width. Suggested breakpoint: **≥ 1200 px → dashboard**,
  below → the existing stacked mobile report. Implement with a `matchMedia('(min-width: 1200px)')`
  hook (or CSS container/Grid) so both share the fetched `data`, `tab`, and player-tab state.
- Extract the data-loading + `buildPlayers` + derived values out of `MatchReport` into a hook
  (e.g. `useMatchReport(slug)`), then render either `<MatchReportMobile …>` (today's JSX) or
  `<MatchReportDashboard …>` (this design) from it.
- The dashboard is designed at a fixed **1440 × 900** canvas. On real desktops, **let it be fluid**:
  the 12-column grid and the height-filling video already adapt. The "no-scroll at 1440×900" goal is
  a target, not a hard constraint — on shorter windows the page may scroll; that's acceptable. Do
  **not** hard-code 900 px height in production; use `min-height: 100vh` / `100dvh` and let the grid
  rows size to content, keeping the video height-capped so it never pushes the bottom row off-screen.

## Screen: Video-Dashboard

Reference: `DeskVideoDashScreen` in `desktop-screens.jsx` + the building blocks in
`desktop-shared.jsx`. Page background **`#FEF4EA`** (beige), text ink **`#0A0A0B`**, body font Inter,
headings Space Grotesk.

### A. Top bar
Full-width, **height 62 px**, `border-bottom: 1px solid hsl(214.3 31.8% 91.4%)`, `padding: 0 20px`,
flex row, `gap: 14px`, vertically centered:
- **Logo mark** (`logo-mark.svg`, height 30). *(The club-name text "Spielanalyse" was intentionally
  removed — do not add a title here.)*
- **Match picker** — a white pill button: `background:#fff`, hairline border, `border-radius:10px`,
  `padding:7px 12px`, `box-shadow: shadow-sm`, `margin-left:12px`, `cursor:pointer`. Contents: text
  `{dd}. · {Disziplin} gegen {Gegner}` (Inter 600, 13 px, `white-space:nowrap`) + chevron-down icon
  (14 px, muted). In production this opens a match switcher (dropdown to other processed matches).
- **Spacer**, then **player tabs** pushed to the far right (`margin-left:auto`, width ~240 px) — the
  existing segmented control (`.bca-tabs`): "Du" for the viewer, first names for others. Switching
  tabs swaps every stat in the grid (same behavior as mobile).
- *(Note: the result chip and the set-by-set score were intentionally removed from this bar — the
  full result lives in the Endstand card. Do not re-add them to the top bar.)*

### B. Body grid
Below the bar: `flex:1`, `padding:16px`, `display:grid`, `grid-template-columns: repeat(12, 1fr)`,
`grid-template-rows: minmax(0, 1fr) auto`, `gap:14px`.

**Row 1, left — `grid-column: span 8`: the video (dominant).**
- A **Safari browser frame** (see §Safari frame) filling the column width, `display:flex;
  flex-direction:column`, `flex:1; min-height:0` so it owns the row-1 height.
- Inside, top→bottom: the browser toolbar (fixed height), then the **video surface** (fills remaining
  height as a centered **16:9** box on a black backdrop — `flex:1; min-height:0`, inner box
  `height:100%; aspect-ratio:16/9; max-width:100%`), then a **controls row** docked at the bottom of
  the frame body (white, `border-top` hairline, `padding:6px 12px`, flex `gap:12`): play/pause button
  + scrubber + time on the left (`flex:1`) and the four overlay toggle chips on the right.
- The 16:9 box keeps the SVG overlay perfectly aligned (overlay is `position:absolute; inset:0` over
  that box). Because the video is **height-capped**, it letterboxes against the black frame rather
  than overflowing — this is what guarantees no overlap with Row 2.

**Row 1, right — `grid-column: span 4`:** a vertical stack (`gap:14`):
- **Endstand card** (see §Endstand) — dense.
- **Feldabdeckung card** — `SectionCard`-style, `flex:1`; large court heatmap (`CourtCoverageCard`),
  centered, ~200–250 px wide, with the wenig→viel legend bar. Sub: "wo du gestanden hast" / "wo
  {Name} gestanden hat".

**Row 2 — four compact tiles, each `grid-column: span 3`** (all `SectionCard` "dense" — see §Tiles):
1. **Kennzahlen** (sub "aktive Spielzeit") — 2×2 grid of bare KPIs (`KeyStats`, denser): Distanz (km,
   2 dec), Sprints (int), Ø Tempo (km/h, 1 dec), Top-Tempo (km/h, 1 dec). Each: icon chip (24 px,
   radius 8, `bg rgba(250,136,50,.13)`, orange icon 13 px) + label (11.5 px muted) over a value
   (Space Grotesk 700, **22 px**) + unit suffix (12 px 600 muted). Count up on reveal.
2. **Spielfeld-Zonen** (sub "Anteil aktiver Spielzeit") — `ZoneCard` (front/mid/rear bars + L/R split).
3. **Aktivität** (sub "je 5 Minuten") — `ActivityCard` bar chart.
4. **Schlagverteilung** (sub "{total} Schläge") — `ShotMixCard`, **compact** variant: tighter row
   gap (~6.5 px), label column 70 px, value 12 px. Renders only when shot data exists.

For doubles, add a **Doppel-Formation** tile/row (`FormationCard`) when `pairStats` is present.

### C. Safari frame
A light macOS-Safari window chrome (reference: `SafariFrame` in `desktop-shared.jsx`). Wraps the
video so it reads as "the report, viewed on the club website."
- Container: `border-radius:12px`, hairline border, `background:#fff`, `box-shadow: shadow-md`,
  `overflow:hidden`.
- **Toolbar:** height **34–38 px**, `flex:none`, `background:#F4F1ED`, `border-bottom` hairline,
  `padding:0 12px`, flex `gap:10`, centered:
  - **Traffic lights** — three 12 px circles, `gap:7`: `#FF5F57`, `#FEBC2E`, `#28C840`.
  - **Back / forward chevrons** — 15 px, stroke `rgba(10,10,11,.35)`.
  - **URL pill** (centered, `flex:1` wrapper with `justify-content:center`): height 28, `radius:8`,
    `background: rgba(10,10,11,.055)`, `padding:0 12px`, `max-width:380`, contents = lock glyph +
    URL text (12 px, `rgba(10,10,11,.6)`, ellipsis). Prototype URL:
    `bc-rheinbach.de/spielanalyse/2026-06-10` → in production use the real public report URL/slug.
  - **Share + plus icons** on the right — 15 px, dim stroke.
- The video + controls live in the frame body beneath the toolbar.

### D. Endstand (scoreboard) card — NEW component
Reference: `DeskScoreboard` in `desktop-shared.jsx`. White card, hairline, `border-radius:16`,
`box-shadow: shadow-sm` (dense) / `shadow-md`, padding ~14–20 px.
- **Head row:** eyebrow "Endstand" (`.bca-eyebrow`, orange, 11 px) on the left; `{Disziplin} · Feld
  {N}` (12 px muted) on the right.
- **Two team rows** (side A first/bold ink, side B muted), each: team/player name (`flex:1`, Space
  Grotesk, ellipsis) + a row of **per-game score cells** (`flex:none`, `gap:6`). Each cell ~28×25 px,
  `radius:7`, Space Grotesk 700 ~13 px, tabular-nums. Won game → `background:#FA8832`, text
  `#1A0500`; lost game → `background: rgba(10,10,11,.06)`, text muted.
- **Foot row:** result chip (`.bca-chip-result`, win → orange tint) + `{durationMin} Min Spielzeit ·
  {restMin} Min Pause` (muted, right).
- **Optional extra line** (used here): a hairline-topped line with rally summary —
  `{count} Ballwechsel · Ø {avg} Schläge · längster {n} in {sec} s`. (You can instead keep the full
  `RallyCard` as a separate tile; the dashboard folds the headline numbers in here to save space.)

### E. Tiles ("dense" SectionCard)
The bottom-row cards are the same `SectionCard` but tighter: `padding ~14px 16px 16px`, title Space
Grotesk 700 14 px, sub 11.5 px muted right-aligned. Add a `dense`/`compact` prop to `SectionCard`
(and to `ShotMixCard`/`KeyStats`) rather than forking the components.

## Interactions & Behavior
- **Player tabs** switch the active player → every stat in the grid updates (re-mount to replay
  entrance animations, exactly like mobile). The Endstand and Feldabdeckung sub-labels follow the
  active player ("Du" vs first name).
- **Match picker** (top bar) opens a switcher to other processed matches → navigates to that slug.
- **Video:** play/pause, scrub via `input[type=range]` (`accent-color:#FA8832`), 4 overlay toggles
  (Tracking, Laufwege, Shuttle, Zonen). Same engine as the existing `VideoAnalysisCard` — overlay
  geometry through the real per-court homography (`calibration/court<N>.json`), movement from the
  pipeline's downsampled tracks (mock `lib/demoOverlay.ts` until real tracks publish). Persist
  playback position per match + overlay prefs per device in `localStorage`
  (`bca-vt-<slug>`, `bca-ov-v1`).
- **Overlay chips:** pill, 12 px 600, `padding:5px 12px`. On → `bg:#FA8832`, text `#1A0500`, border
  `#C66C27`, `box-shadow:0 2px 0 #C66C27`. Off → white, muted text, hairline border.
- **Entrance animations** ("dezent"): each tile wrapped in `Reveal` (opacity 0 + `translateY(10px)`
  → visible, ~.5 s, stagger ~60 ms by index, IntersectionObserver + ~900 ms timer fallback). Count-ups
  on KPIs (0 → value, 700 ms, `de-DE` formatting, settle on exact value). Respect
  `prefers-reduced-motion: reduce` → render end states immediately.
- **Card hover** (optional on desktop): `-2px` lift + shadow. Bubble buttons (orange) translate down
  and flatten their hard shadow on press — the brand pattern.

## State Management
- Reuse the existing `MatchReport` data fetch (matches, match_players, player_match_stats incl.
  `timeline` + `heatmap_uri`, player_shot_stats, rallies, pair_match_stats). **No new queries** for the
  layout itself.
- Local UI state: `tab` (active player index), optional `replayKey`, video `{playing, time, duration,
  overlays}`. A `isDesktop` boolean from a `matchMedia` hook selects the layout.
- The Endstand card needs per-game scores → parse from `match.score` with the existing
  `lib/score.ts` (`parseScore`, `formatGames`, `resultForSideA`). Duration/Pause: if not in the row
  yet, hide that part of the foot line (don't fabricate).

## Design Tokens
Identical to the mobile handoff / BC Rheinbach system (already in `web/src/styles.css`):
- **Colors:** primary `#FA8832`; bubble-shadow `#C66C27`; dark-on-orange `#1A0500`; page beige
  `#FEF4EA`; ink `#0A0A0B`; muted `hsl(215.4 16.3% 46.9%)`; hairline `hsl(214.3 31.8% 91.4%)`;
  opponent blue `#1758A6`; heat-max `#C0392B`; card white `#fff`. Safari toolbar `#F4F1ED`; traffic
  lights `#FF5F57 / #FEBC2E / #28C840`; toolbar glyphs `rgba(10,10,11,.35)`.
- **Typography:** Space Grotesk (headings/numerals) + Inter (body). Desktop scale: 30 (page H1 if
  used), 22 (KPI values), 15.5/14 (card titles), 13.5/13 (body/picker), 12–11.5 (labels/meta).
- **Radii:** Safari frame 12, video box inherits frame, cards 12 / Endstand 16, score cells 7, pills
  99, icon chips 8, buttons 8.
- **Shadows:** `shadow-sm` (cards rest), `shadow-md` (Safari frame / scoreboard), hard bubble
  `0 2–3px 0 #C66C27` (orange controls only).
- **Spacing:** body padding 16; grid gap 14; tile padding ~14–16; control row padding 6–7 px 12 px.

## Assets
- `assets/logo-mark.svg` — club logo mark (top bar).
- `assets/colors_and_type.css` — brand tokens + base type + `.btn-bubble` (loads Inter + Space
  Grotesk). In the repo these tokens already live in `web/src/styles.css` / fontsource — prefer the
  repo's existing setup.
- **Video:** not bundled (size). Prototype referenced `assets/match-clip.mp4`; the repo's demo clip
  is `videos/Badminton_Test_Clip.mp4`, and production uses per-match clips from Supabase Storage via
  the existing `VideoAnalysisCard`/`lib/demoOverlay.ts`.
- **Icons:** tiny inline stroke SVGs in `viz.jsx` (`ICONS`: route, gauge, bolt, clock, chevron, …) +
  the Safari glyphs in `desktop-shared.jsx`. The repo already standardizes on its `Icon` component /
  lucide equivalents — prefer those (chevron-down, lock, share, plus, arrow-left, play, pause).

## Files in this bundle
| File | What it is |
|---|---|
| `Spielanalyse Desktop.html` | Entry point. Open in a browser. (Renders a design-canvas shell — the actual screen is the single artboard.) |
| `desktop-screens.jsx` | **`DeskVideoDashScreen` is the shipped layout.** The other `Desk*Screen` functions are dropped explorations — ignore them. |
| `desktop-shared.jsx` | Desktop building blocks: `DeskCard`, `DeskScoreboard` (Endstand), `DeskKpiBare`/`DeskKpiTile`, `DeskRallies`, `SafariFrame`, and the `useDeskVideo` hook + `DeskVideoSurface`/`DeskVideoControls`/`DeskOverlayChips`. |
| `video.jsx` | `VideoOverlay` — the SVG overlay (tracking/trails/shuttle/zones) drawn over the video. Production uses the repo's real homography instead of the prototype quad. |
| `viz.jsx` | Shared chart/animation primitives reused from the mobile handoff (`CountUp`, `Reveal`, `CourtHeatmap`, `ZoneBlock`, `TimelineChart`, `ShotMix`, icons). |
| `screens.jsx` | Mobile screens + `THEMES`/`PlayerTabs`/`ResultChip`/`SectionCard` helpers the desktop reuses. **Theme key 'A' (Vereinsfarben) is the shipped direction — ignore B/C/D.** |
| `mock-data.js` | Mock data with exact German label/format conventions. Replace with the real Supabase queries. |
| `assets/` | Brand CSS + logo mark. |
| **Ignore:** `design-canvas.jsx`, `tweaks-panel.jsx` | Prototype scaffolding (the pan/zoom canvas + the tweak panel). **Not part of the product** — do not port. |

## What NOT to port
- The design-canvas wrapper, tweaks panel, the four dropped desktop directions, and the prototype's
  hand-tuned perspective quad (`VID_QUAD` in `video.jsx`) — production already has the real homography.
- Do not duplicate the data layer or create a second route; this is a layout branch of the existing
  report.
