# Handoff: Spielbericht-App („Spielanalyse") — BC Rheinbach

## Overview

Mobile-first match-report app for the club's video-analytics pipeline. Players receive a link after a recorded match and open it on their phone to see personal stats: distance, sprints, speeds, court heatmap, zone distribution, activity timeline, shot mix, rally stats, doubles formation — plus the gameplay video with toggleable analysis overlays (tracking boxes, movement trails, shuttle path, zone tint).

Two screens:
1. **Spielliste** — list of processed matches, each row tappable
2. **Spielbericht** — per-player report with tabs for every player in the match

Visual direction: **„Vereinsfarben"** — the BC Rheinbach brand system (warm beige, white cards, Crusta orange, Space Grotesk headings), light mode only, German UI in du-form, design width 390 px.

## About the Design Files

The files in this bundle are **design references created in HTML** — interactive prototypes showing intended look and behavior, **not production code to copy directly**. The task is to **recreate these designs in the existing target codebase**: the `web/` app of the `badminton-game-play-analytics` repo (Vite + React + TypeScript + Supabase, react-router). Use its established patterns — `src/pages/MatchList.tsx`, `src/pages/MatchReport.tsx`, `src/components/*`, `src/types.ts`, `src/lib/format.ts` — and replace/extend them rather than adding a parallel UI. The prototype's React component structure (screens.jsx, viz.jsx, video.jsx) maps closely and can guide the port, but rewrite in TypeScript against the real row types.

## Fidelity

**High-fidelity.** Colors, typography, spacing, radii, copy and interactions are final. Recreate pixel-perfectly. The mock data (`mock-data.js`) shows the exact target shapes of formatted output (German locale, units) but must be replaced by the real Supabase queries that already exist in the codebase.

## Screens / Views

### 1. Spielliste (match list)

- **Purpose**: entry point; player sees all processed matches, newest first, taps one to open its report.
- **Page**: full-height, background `#FEF4EA` (beige). Content column = full width of a 390 px viewport, cards inset with `padding: 0 14px`, vertical `gap: 10px`.
- **Header** (`padding: 6px 18px 4px`):
  - Club logo mark (SVG, height 28). No club-name text next to it (intentionally removed).
  - H1 „Spielberichte" — Space Grotesk 700, 30 px, line-height 1.05, letter-spacing −0.01em, color `#0A0A0B`.
  - Greeting line „Hallo Lukas — deine Auswertungen nach jedem Spiel." — Inter 13.5 px, color `hsl(215.4 16.3% 46.9%)`. Replace name with the authenticated/linked player.
- **Match row card** (one per match, tappable):
  - White card, `border: 1px solid hsl(214.3 31.8% 91.4%)`, radius 12, `box-shadow: 0 1px 2px rgb(0 0 0 / .05)`, padding `13px 14px`.
  - Grid `46px 1fr 18px`, gap 12, vertically centered.
  - Date block: day numeral (Space Grotesk 700, 21 px) over month (10.5 px uppercase, letter-spacing .08em, muted).
  - Main: line 1 = own side names (Inter 600, 14 px, ellipsis); line 2 = „gegen {opponents}" (12.5 px muted, ellipsis); line 3 = chip row (flex, gap 6): discipline chip („Einzel"/„Doppel": 11 px 600, bg `rgba(10,10,11,.06)`, pill radius), result chip („Sieg" bg `rgba(250,136,50,.16)` text `#C66C27` / „Niederlage" bg `rgba(10,10,11,.07)` text muted, 11 px 700, pill), score text (12 px muted, tabular-nums, format `21:18 · 19:21 · 21:15` — viewer's side first).
  - Chevron-right icon, orange `#FA8832`, 17 px.
  - Newest match: „Neu" badge — absolute, top −7 right 12, bg `#FA8832`, text `#1A0500`, 10.5 px 700, pill, `box-shadow: 0 2px 0 #C66C27` (mini bubble).
  - Hover: `translateY(-2px)` + shadow-lg; active: `scale(.985)`; transition .18s ease.
- **Footer**: centered 11 px muted, „Automatisch aus Videoaufnahmen ausgewertet. / Nur für Mitglieder sichtbar."

### 2. Spielbericht (match report)

- **Purpose**: the post-match reward moment + full personal stats; switch player via tabs.
- **Page**: bg `#FEF4EA`; sections in a column, `padding: 0 14px`, `gap: 12px`, bottom padding 28.
- **Top bar**: back button — 34 px circle, white, hairline border, arrow-left icon; navigates to list.
- **Hero** (`padding: 4px 18px 0`):
  - Eyebrow „Spielbericht" — Inter 600 12 px, orange `hsl(26 95% 59%)`.
  - H2 — Space Grotesk 700 27 px, line-height 1.08: „Stark gespielt, {Vorname}." on win, „Gut gekämpft, {Vorname}." on loss. Always addresses the report viewer.
  - Meta line 13 px muted: „{Wochentag}, {Datum} · {Disziplin} · Feld {N}".
  - Chip row: result chip („Sieg 2:1" / „Niederlage 0:2", same chip style as list, nowrap) + score (13 px muted, tabular-nums).
  - Replay button top-right: 34 px ghost circle, replay icon — remounts report content to replay animations (prototype nicety; optional in production).
- **Player tabs**: segmented control — container bg `rgba(10,10,11,.05)`, hairline border, radius 12, padding 4; buttons flex 1, radius 9, padding 8px 4px, Inter 600 13 px. Active: white bg + shadow-sm + ink text; inactive: muted text. Labels: „Du" for the viewer, first names for others (2 tabs singles, 4 doubles). Switching tabs swaps all stat sections (remount → entrance animations re-run).
- **Key numbers** — 2×2 grid, gap 10. Each cell: white card (border/radius-12/shadow-sm, padding 13px 14px) with icon chip (28 px square, radius 9, bg `rgba(250,136,50,.13)`, orange stroke icon 15 px) + label (11.5 px muted, nowrap) on one row; below, value — Space Grotesk 700 24 px with unit suffix (12.5 px 600 muted). Cells: Distanz (km, 2 decimals, German comma), Sprints (int), Ø Tempo (km/h, 1 dec), Top-Tempo (km/h, 1 dec). Values count up on first reveal (see Animations).
- **Video-Analyse card** (only when the match has a processed video) — see dedicated section below.
- **Feldabdeckung card** — section card (white, radius 12, hairline, shadow-sm, padding 15px 16px 18px; title Space Grotesk 700 15 px; sub right-aligned 11.5 px muted „wo du gestanden hast" / „wo {Name} gestanden hat"):
  - Court heatmap, centered, 222 px wide. Half-court top-down, net at top. SVG viewBox 244×268 (1 m = 40 px; court 6.1×6.7 m). Court lines: outer rect, singles sidelines at 0.46 m, short service line at 1.98 m, doubles long service line 0.76 m from back, center line from service line to back — stroke `#DDD5CC`, 2 px; net = solid 4 px `#B9AFA4` at top edge.
  - **Raster style (chosen)**: 12×13 grid of rounded cells (rx 3, 1.6 px gutter) over the court; fill `#FA8832`, hottest cells (>85 % of max) `#C0392B`; cell opacity `0.07 + value × 0.88`, cells < 5 % hidden. Cells fade in staggered by `(col+row) × 22ms`. In production the grid comes from FR-11 (0.25 m cells → downsample or render native resolution).
  - Legend below: „wenig" — gradient bar (transparent-orange → `#FA8832` → `#C0392B`, 6 px, radius 3) — „viel", 11 px muted.
- **Spielfeld-Zonen card** (sub „Anteil aktiver Spielzeit"):
  - Three rows (Vorderfeld/Mittelfeld/Hinterfeld): grid `86px 1fr 44px`; label 12.5 px muted; bar track 12 px tall radius 6 bg `rgba(250,136,50,.12)`; fill width = value/max, dominant zone solid `#FA8832`, others `rgba(250,136,50,.42)`; value right-aligned Space Grotesk 700 14 px „44 %".
  - Below: Links/Rechts split — labels row 12 px muted, then a single 12 px track split into two segments (gap 2): left segment solid orange, right segment the track tint; widths = percentages. Bars grow from their outer edges (scaleX, .7s, stagger).
- **Aktivität card** (sub „Distanz je 5 Minuten"): bar chart, SVG viewBox 330×120, baseline y = 96. One bar per 5-min bucket, width = 330/n − 6, rx 4. Max bucket solid `#FA8832` with value label above („340 m", Space Grotesk 700 10.5 px); others `rgba(250,136,50,.42)`. X labels every 2nd bucket („0′, 10′ …", 9.5 px muted). Bars grow bottom-up (scaleY, .55s, 55 ms stagger). Baseline hairline.
- **Schlagverteilung card** (sub „{total} Schläge", foot „Schnellster Smash ≈ {N} km/h — Schätzung aus der Shuttle-Verfolgung."):
  - One row per shot type, sorted desc: grid `76px 1fr 58px`; label 12.5 px (top type 600 ink, rest muted); bar 10 px radius 5, track `rgba(250,136,50,.12)`, fill = count/max (top type solid orange, others tint); right: count Space Grotesk 700 13 px + „· {share}%" 11 px muted. Bars draw in left-to-right, 70 ms stagger.
  - Shot types (German): Clear, Netzdrop, Drop, Lift, Drive, Aufschlag, Smash (+ `unknown` from FR-22 — show as „Sonstige" only if > 0).
- **Ballwechsel card**: 3-column centered stats — gesamt (count), Ø Schläge (1 dec), längster (shot count) — value Space Grotesk 700 22 px, label 11.5 px muted. Below, centered 12.5 px muted: „Längster Ballwechsel: 34 Schläge in 52 s. Der Smash beendete 31 % aller Ballwechsel."
- **Doppel-Formation card** (doubles only, sub „Anteil aktiver Spielzeit"): per pair — header row (pair label Space Grotesk 13.5 px 700 left, „{N} Rotationen" 12 px muted right), split bar 14 px (front-back segment solid `#FA8832` from left, side-by-side segment `rgba(10,10,11,.16)` from right, gap 2, radius 7), caption row „Vorne/Hinten {x} %" / „Nebeneinander {y} %" 11.5 px muted.
- **Footer**: centered logo mark (26 px, opacity .85) + 11 px muted: „Automatisch aus der Videoaufnahme ausgewertet. / Du möchtest deine Daten löschen? Schreib uns kurz."

### Video-Analyse card (in report)

- Section card titled „Video-Analyse", sub = clip label (e.g. „Demo-Aufnahme"), foot „Overlays aus der automatischen Auswertung — Demo-Daten auf Beispiel-Footage." (adjust copy when real data lands).
- **Video**: full card width, radius 10, clipped, black backdrop; `muted playsInline preload="metadata"`. On metadata load, restore last playback position from localStorage (key per match), else seek to 1 s for a poster frame. Persist position every second.
- **Overlay**: SVG absolutely positioned over the video, `viewBox 0 0 192 108`, `preserveAspectRatio="none"`, `pointer-events: none`. All geometry is computed in **court coordinates** (u ∈ [0,1] left→right, v ∈ [0,1] far baseline→near baseline) and projected to frame coordinates.
  - Prototype uses an eyeballed perspective quad + depth factor (see `video.jsx`: `VID_QUAD`, `vidProject`). **In production, replace this with the real per-court homography** the pipeline already stores (`calibration/court*.json`, FR-5) — project court coords through the homography instead.
  - **Tracking** (default on): per player a box anchored at the foot point — width `4.5 + t×4.5`, height `8 + t×7` viewBox units (t = perspective depth 0 far → 1 near), rx 1, stroke 0.45 + 8 % tinted fill; name chip above (height 3.6, fill = player color, white 2.7 px 700 text; viewer labeled „Du"); ground ellipse shadow under feet (rx = w×0.62, ry 1.1, 35 % opacity). Colors: viewer/near `#FA8832`, opponent/far `#1758A6`.
  - **Laufwege** (default on): last 5 s of each player's foot positions as a polyline (two segments: older half stroke 0.55/opacity .25, newer 0.8/.65, round caps), player colors.
  - **Shuttle** (default off): white dot (r 1, dark outline) with vertical arc offset along each hop + dashed white line from last hit point (0.4 stroke, dash 1.3/1, 70 % opacity).
  - **Zonen** (default off): the near half's three depth bands (v 0.5–0.65, 0.65–0.82, 0.82–1.0) as projected quads, fill `#FA8832` with opacity `0.10 + pct/100 × 0.5`, thin white separators; white 3 px 700 percentage labels at the right edge. Percentages = the viewer's front/mid/rear shares.
- **Overlay toggles**: chip row under the video (flex, gap 6, wrap). Chip: pill, 12 px 600, padding 5px 12px. On: bg `#FA8832`, text `#1A0500`, border `#C66C27`, `box-shadow: 0 2px 0 #C66C27`. Off: white bg, muted text, hairline border. Toggle state persists (localStorage in the prototype; user preference per device is fine).
- **Controls row**: orange play/pause button (38 px circle, bg `#FA8832`, icon `#1A0500`, border + 3 px bubble shadow `#C66C27`), native `input[type=range]` scrubber with `accent-color: #FA8832` (flex 1), time „m:ss / m:ss" (11.5 px, mono stack, tabular).
- Overlay positions re-render via rAF while playing; when paused, `timeupdate`/seek events drive the state.

## Interactions & Behavior

- List row tap → report route (`/match/:slug` in the existing router). Back button → list. Scroll resets to top on navigation; page content fades/slides in (opacity 0 + translateX(26px) → none, .3s, cubic-bezier(.22,.8,.3,1)).
- Player tab switch remounts the stat sections so entrance animations replay.
- **Entrance animations** („dezent" level — the chosen default): each section wrapped in a reveal — opacity 0 + translateY(10px) → visible, .5–.55s, ease/cubic-bezier(.22,.8,.3,1), staggered 60 ms per section index, triggered by IntersectionObserver when scrolled into view (threshold ~0.1) **with a ~900 ms timer fallback so content is never stuck hidden** (hidden tabs, prerender, IO edge cases).
- **Count-ups**: key numbers animate 0 → value over 700 ms, cubic ease-out, German formatting (`toLocaleString('de-DE')`), triggered on reveal; always settle on the exact final value (timer fallback — rAF can be throttled).
- **Chart draw-ins**: bars scaleX/scaleY from 0 with origin at their base, heat cells staggered opacity (see per-card specs).
- All animations respect `prefers-reduced-motion: reduce` → render end states immediately.
- Hover/active states: cards lift (−2px + shadow-lg), buttons press (scale .92–.985); bubble buttons translate down flattening their hard shadow (brand pattern).
- Video: play/pause toggle, scrub via range input, position + overlay toggles persisted. Video card only renders when the match has a video asset.

## State Management

- Route state: list vs. report (existing react-router setup covers this).
- Report: `tab` (selected player index), `replayKey` (optional remount counter).
- Video: `playing`, `time`, `duration`, `overlays {tracking, wege, shuttle, zonen}`; persistence keys: playback position per match, overlay prefs per device.
- Data: the existing Supabase queries in `MatchList.tsx` / `MatchReport.tsx` already fetch everything needed (`matches`, `match_players`, `player_match_stats` incl. `timeline` + `heatmap_uri`, `player_shot_stats`, `rallies`, `pair_match_stats`). New requirement for overlays: per-frame (or downsampled, e.g. 10 Hz) player court positions + shuttle positions + hit events for the clip being shown — exposed as JSON keyframes `{t, u, v}` per player; the prototype's `vidTrackAt` interpolation shows the consumption pattern.
- Derived viewer: the report is opened via a personal link — order tabs so the linked player is first and labeled „Du".

## Design Tokens

From the BC Rheinbach design system (`colors_and_type.css` included):

- **Colors**: primary orange `#FA8832` (Crusta); orange shade (bubble shadows) `#C66C27`; dark text on orange `#1A0500`; beige page bg `#FEF4EA`; ink `#0A0A0B`; muted text `hsl(215.4 16.3% 46.9%)`; hairline border `hsl(214.3 31.8% 91.4%)`; secondary blue (opponent overlay) `#1758A6`; heat-max red `#C0392B`; card white `#FFFFFF`.
- **Tints**: orange 12–16 % for chip/track fills (`rgba(250,136,50,.12/.13/.16/.42)`), ink 5–7 % for neutral chips.
- **Typography**: headings Space Grotesk 400–700; body Inter 400–700; numerals tabular where scores/times appear. Scale used: 30/27 (H1/H2), 24 (stat values), 21 (date numerals), 15 (card titles), 14/13.5/13 (body), 12.5–11 (meta/labels), 10.5 (badges).
- **Radii**: cards 12, video 10, segmented control 12/9, chips/pills 99, icon chips 9, buttons 8 (brand `--btn-radius`).
- **Shadows**: `shadow-sm 0 1px 2px rgb(0 0 0/.05)` (cards at rest), `shadow-lg 0 10px 15px -3px rgb(0 0 0/.1), 0 4px 6px -4px rgb(0 0 0/.1)` (hover), hard bubble shadows `0 2-3px 0 #C66C27` (orange elements only).
- **Spacing**: page gutter 14 px (cards) / 18 px (hero text); card padding 15–16 px; section gap 12 px; chip gaps 6 px.

## Assets

- `assets/logo-mark.svg` — club logo mark (from the BC Rheinbach design system).
- `assets/colors_and_type.css` — brand tokens + base type + `.btn-bubble` (load Inter + Space Grotesk from Google Fonts, see file header).
- Demo video: **not bundled** (size) — it is `videos/Badminton_Test_Clip.mp4` in the target repo; the prototype referenced it as `assets/match-clip.mp4`. Production uses per-match clips from Supabase Storage.
- Icons: tiny inline stroke SVGs (2 px, 24-grid) defined in `viz.jsx` (`ICONS`) — route, gauge, bolt, clock, chevron, arrow-left, replay, play/pause. lucide-react equivalents exist in the repo's dependency set (route, gauge, zap, timer, chevron-right, arrow-left, rotate-ccw, play, pause) — prefer those.

## Files

| File | What it is |
|---|---|
| `Spielanalyse App.html` | Entry point of the final design (open in a browser) |
| `app-final.jsx` | App shell: routing list↔report, phone frame, tweaks (ignore tweaks in production) |
| `screens.jsx` | `MatchListScreen`, `ReportScreen`, `SectionCard`, `KeyStats`, `PlayerTabs`, hero, chips — **theme key 'A' is the shipped direction; ignore the B/C/D branches** |
| `viz.jsx` | `CountUp`, `Reveal` (entrance animation), `CourtHeatmap`, `ZoneBlock`, `TimelineChart`, `ShotMix`, `FormationBlock`, icons |
| `video.jsx` | `VideoAnalysisCard` — overlays, projection, toggles, controls |
| `mock-data.js` | Mock data incl. exact German label/format conventions |
| `assets/` | Brand CSS + logo mark |

Not bundled but part of the design history (in the design project): `Spielanalyse Explorationen.html` (4-direction exploration canvas), `Prototyp.html` (direction switcher).
