# Feature — Shuttle landing + in/out call (estimate-grade "Aus-Schläge")

**Status: BUILT + real-footage-validated (2026-07-03) — v1: rally-ending out; singles full, doubles
partial.** Validated on a disposable re-run of Session2_Game_5 (doubles, 23 rallies): invariants hold
(one landing event per rally, degraded events carry reason + no position), but the located-rate was 1/23
— and that one was a **false positive** (a between-points racket catch "at rest" in mid-air, projected
2.85 m out by the ground homography; trainer-verified on video). That finding added the
`LANDING_MAX_OUT_CM` plausibility guard (see below); after the guard the run yields 0 located landings —
honest, coverage-limited (21 % shuttle coverage on the "12 fps" cache — really 10 fps; the 30 fps re-measure
is `features/analysis-30fps.md`). The match **score stays
human-entered**; nothing here ever computes or overrides it. All `LANDING_*` thresholds remain **untuned
first guesses** — no located-rate or precision claim before the landing answer key exists (see accept
criterion below).

## Goal
For each rally that ends with the shuttle landing on the floor, estimate **where it landed** and call it
**in / out / uncertain** against the singles court, and roll rally-ending *out* landings into a per-player,
per-shot-type **„Aus-Schläge"** (unforced-error) count. This is an **estimate** („Schätzung"), surfaced with
an uncertainty band and a confidence — never as an umpire's line call, never as the scoreboard.

## The load-bearing insight (why this is even possible in 2D)
Everywhere else the shuttle tracker fights the same wall: the **ground homography can't place an airborne
shuttle** — a pixel above the court's ground-plane horizon projects toward infinity, so court (u,v) is
unusable in flight (see `30-ml-pipeline.md` stage 5, `shuttle-tracking.md` "Honest limits"). **A landing is
the one moment that escapes this**: at floor contact the shuttle *is* on the ground plane, so the homography
is geometrically valid there. The hard problem therefore **reduces to one question: which pixel is the
landing pixel, and how sure are we?** Get that pixel and the in/out test is sound (modulo lens bow).

## Dependencies (both now exist)
- **Shuttle fine-tune** (`shuttle-finetune-plan.md`) — the white-wall recall that makes a landing visible at
  all (full-clip F1 0.483→0.958). Still one-session; a 2nd session is needed for cross-session trust.
- **Stage 6 rally-end** (`rally_hit`, BUILT) — `find_rally_end` already returns the last reliable in-flight
  point's **image (x,y) AND court (u,v)** and emits a `rally_end` event carrying them. That is the seed the
  landing-frame search refines.

## Approach (all image-space until the floor-contact pixel is found) — as built
Pure logic in `worker/rally/landing.py` (unit-tested, `tests/test_landing.py`), wired into
`build_rallies` (one `landing` event per rally) — **no new pipeline stage**, `replace_rallies` stays
the single writer.
1. **Landing-frame selection (image space).** `detect_landing`: over the conf-filtered terminal window
   (`LANDING_WINDOW_S` 0.5 s, ≤ `LANDING_MAX_WINDOW_DETS` 8), per-step vertical velocity flattens
   (below `max(LANDING_FLAT_VY_ABS 0.25/s, LANDING_FLAT_VY_FRAC 0.35 × peak descent)`) and stays flat
   → the first detection of that resting tail is the contact. Inputs are the same normalized
   image-space points stage 6 uses (`px/W, py/H`) — **not** (u,v), which is still wrong in the descent.
2. **Graceful degradation — never fabricate.** `located=false` with a `reason`, and NO position, when:
   `sparse` (< `LANDING_MIN_DETS_S` 0.4 s of terminal dets — 4 at the real 10 fps, 12 at 30 fps), `straddle`
   (terminal gap > `LANDING_MAX_END_GAP_S` 0.2 s — the sampling straddled the contact; checked before the
   window-density test since 2026-10-25, `features/analysis-30fps.md`), `edge_cutoff` (last det
   within `LANDING_EDGE_MARGIN` 0.04 of a frame border — descent left the frame/airspace mask),
   `no_contact` (track ends still falling — covers occlusion at contact, the labeler's never-click-a-guess
   rule), `no_descent` (nothing fell), `implausible` (the contact's court projection is >
   `LANDING_MAX_OUT_CM` 100 outside every line — an **above-floor rest** like a racket catch, net cord or
   between-points handover, which the image-space flatness test cannot tell from floor contact; the
   ground homography projects such an elevated pixel meters past the lines. Found on real footage
   2026-07-03: a shuttle handover caught mid-air minted a confident „out 2.85 m" + wrong `winner_side`).
   **Never extrapolate the arc to the floor** — a fabricated landing pixel is worse than an absent one
   (CLAUDE.md persona).
3. **Contact pixel → (u,v)** comes for free: each shuttle detection already carries the ground-homography
   projection of its pixel (`shuttle/track.py`), which is exactly valid at floor contact — the landing
   detection's stored (u,v) IS the court position; no re-projection.
4. **`classify_landing(u, v, band_cm, discipline)`** against the discipline's boundary with an
   uncertainty band:
   - Singles sidelines `u ∈ [0.0754, 0.9246]` — the **0.46 m** inset from each doubles sideline normalised by
     the 6.10 m doubles width (`0.46/6.10 = 0.0754`; now the shared `geometry.SINGLES_INSET_M`, reused by the
     `verify.py` minimap). **Doubles: the full width `u ∈ [0, 1]`.** Long service / baseline lines
     `v ∈ [0, 1]` (full length in both disciplines during rally play).
   - **Band:** flat **±45 cm** (`LANDING_BAND_CM`, mid of the spec'd 40–50), distance-weighted later
     (far-court pixels carry more court-error — same amplification as the foot-position work in
     `fix-later.md`). Note the band covers the whole 46 cm doubles alley — near-alley calls saturate to
     `uncertain` by design.
   - `call = out` only when the point is outside the line by **more than** the band; `in` when inside by more
     than the band; **`uncertain`** in the band — and **forced `uncertain` near the frame edges**
     (`LANDING_UNCERTAIN_EDGE` 0.08; DJI ultrawide barrel bow biases near-line calls; undistortion is
     still a no-op). Confidence = `min(1, |margin|/(2·band)) ×` mean terminal-window detection conf;
     flat 0.2 for `uncertain`; null when unlocated.
5. **Attribute & aggregate.** The landing half names the winner (`winner_half`: the receiver's half — the
   opponent hit out); landings within `LANDING_NET_MARGIN_M` (1.0 m) of the net are ambiguous (may not have
   crossed) → no attribution. `rally_hit` maps half→side via the track_label↔role binding
   (`_side_by_role`, robust under trainer remaps; conflicting bindings are skipped, not guessed) and sets
   `rallies.winner_side`. `aggregate` then counts out-landings against the hitting side
   (`out_error_counts`) and writes `player_shot_stats.errors` under **`shot_type='unknown'`** until
   stage 7 attributes real shot types (`Store.upsert_player_shot_errors`; a 0 overwrites stale counts
   after trainer drops — and because `rally_sync` rewrites the parquets from the edited DB first, a
   dropped rally loses its error automatically). **Singles only** — who of a doubles pair hit needs
   pose/stroke (P3). The web sums `errors` into the „Aus-Schläge" foot line of the Schlagverteilung card
   (`lib/landing.sumErrors`; hidden when the feature never ran, shown as „Schätzung, Untergrenze").

## Data model (no DDL — columns already exist / nullable)
- **`events.type = 'landing'`** (new type on the existing generic `events` stream), payload:
  ```
  {located:bool, call:'in'|'out'|'uncertain', line:'side'|'baseline'|null,
   u?,v?, px?,py?, margin_cm?, band_cm, confidence, reason?}
  ```
  `px, py` are **normalized** image coordinates (same convention as `rally_end`'s x, y); `reason` only
  on degraded landings (sparse|straddle|edge_cutoff|no_contact|no_descent|implausible). Exactly one
  landing event per rally, degraded or not. Reserve `'bounce'` for any raw mid-rally floor touches;
  `'landing'` is the **rally-ending** contact.
- **`rallies.winner_side` / `rallies.end_shot_type`** are **estimates decoupled from `matches.score`** — a
  landing-derived `winner_side` is a guess, never reconciled against the human-entered score.
- **`player_shot_stats.errors`** (already in the schema) is **sourced** here from rally-ending out-landings
  per shot_type — extends the column's meaning, no schema change.
- **`match_rally_stats`** gains an **app-derived `outLine`** summary alongside the existing app-derived
  `endLine` (both computed in the web from rally/landing data; neither is a stored column).

## Honest limits (surface, don't hide)
- **It's an estimate, not Hawk-Eye.** Monocular, single 30 fps camera, ultrawide lens (undistortion a no-op).
  Near-line and near-edge calls are biased → `uncertain` by design, not by accident.
- **Occlusion & sampling lose landings.** Many contacts are hidden or fall between frames → `located=false`.
  Coverage will be partial; „Aus-Schläge" is a **lower bound**, labelled as such.
- **Foreign-court landings** can still leak in (the wider-box / rally-continuity work *reduces* but can't
  eliminate cross-court intrusion — see `shuttle-tracking.md`).
- **Above-floor rests that project INSIDE the court are undetectable in v1.** The `LANDING_MAX_OUT_CM`
  guard only catches elevated rests whose projection lands far outside the lines (the common case —
  elevated pixels project away from the camera). A low racket catch mid-court can still read as an „in"
  landing; only physics/net-band modeling or stage-7 stroke context can close that. Phantom rallies from
  between-points shuttle handovers are a **rally-segmentation** problem (see pose/swing-fusion work),
  not a landing one — trainer rally-drops remove their landings via `rally_sync`.
- **Doubles is partial** (user decision 2026-07-03, scope extension over the original singles-only v1):
  landing events + in/out against the doubles boundary + side-level `winner_side` work; the per-player
  „Aus-Schläge" stays **singles-only** until stage 7 can attribute a shot within the pair. 4 players ⇒
  more occlusion ⇒ expect a lower located-rate.
- **Accept criterion needs labels:** a landing answer-key (a few labeled landings per session, ≥2 sessions)
  before any precision/recall claim. Until then: spot-check only, and the UI says „Schätzung".

## Defaults taken (noted, not asked) + user decisions (2026-07-03)
Flat ±45 cm band first; „Aus-Schläge" derived in the web by summing `player_shot_stats.errors` (no
new column), written under **`shot_type='unknown'`** until the stroke stage exists (user decision — the
sum is stable when attribution later refines); UI = one additive foot line in the Schlagverteilung card
(user decision; the RallyCard `outLine` sentence is **deferred**, same decision); dedicated
`events.type='landing'` for the rally-ending contact (`'bounce'` reserved for raw mid-rally touches);
**rally-ending out only** in v1 (mid-rally in/out and won-point landings come later). All `LANDING_*`
knobs env-overridable (`40-worker-contract.md`).

## Validation — RUN 2026-07-03 (Mac Studio, disposable match `landing-v1-smoketest`)
Session2_Game_5.mp4 re-enqueued as a fresh doubles match (reviewed matches untouched); full
calibrate-resume slice; trainer spot-checked the video. Results:
- **Invariants hold:** 23 landings for 23 rallies (exactly one each); every degraded event carries a
  `reason` and no position; no `errors` rows for doubles; web shows no „Sonstige 0" bar, Aus-Schläge
  foot line hidden for doubles and rendered correctly for a synthetic singles errors row.
- **Located-rate 1/23** (10 no_descent, 6 no_contact, 6 sparse) — ceiling set by 21 % shuttle coverage
  at ~12 fps analysis, not by the landing logic (spot-checked misses were honest: detections died
  before floor contact on real, human-visible landings).
- **The 1 located landing was a false positive** (trainer-verified): a between-points shuttle handover
  caught mid-air with the racket → flat-vy fired above the floor → projected „out 2.85 m", wrong
  `winner_side`. Fixed in the same change with the `LANDING_MAX_OUT_CM` plausibility guard
  (degrades to `reason=implausible`); re-run confirms the guard catches it.
Still to do before trusting numbers: the landing answer key (≥2 sessions of labeled landings), then
tune `LANDING_*` (esp. window/min-dets for ~12 fps sampling) against measured recall/precision.
