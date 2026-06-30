# Feature — Shuttle landing + in/out call (estimate-grade "Aus-Schläge")

**Status: SPEC ONLY — no code yet.** Captured honestly so it is build-ready once it's prioritised. The
match **score stays human-entered**; nothing here ever computes or overrides it.

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

## Approach (all image-space until the floor-contact pixel is found)
1. **Landing-frame selection (image space).** From the rally's terminal detections, find the floor-contact
   frame: the shuttle decelerates and its image-space vertical travel flattens/stops near the floor. Inputs
   are the same normalized image-space points stage 6 uses (`px/W, py/H`) — **not** (u,v), which is still
   wrong in the descent.
2. **Graceful degradation — never fabricate.** Mark the landing `located=false` / `uncertain` instead of
   guessing when: the shuttle is **occluded** at contact (behind a player/body — the same rule as the
   labeler: never click a guessed position); the descent is **cut off** by the airspace mask edge (a just-out
   landing can leave the box — see the wider-box A/B in `shuttle-tracking.md`); **30 fps sampling** straddles
   the contact frame; or the trajectory is too sparse. **Never extrapolate the arc to the floor** — a
   fabricated landing pixel is worse than an absent one (CLAUDE.md persona).
3. **Project the contact pixel → (u,v)** via the per-video homography (valid here, at floor contact).
4. **`classify_landing(u,v, band)`** against the **singles** boundary with an uncertainty band:
   - Singles sidelines `u ∈ [0.0754, 0.9246]` — the **0.46 m** inset from each doubles sideline normalised by
     the 6.10 m doubles width (`0.46/6.10 = 0.0754`; today the 0.46 m inset lives only in the `verify.py`
     minimap, not as a shared constant — introduce `SINGLES_INSET_M = 0.46`). Long service / baseline lines
     `v ∈ [0, 1]` (full length in singles).
   - **Band:** a flat ±~40–50 cm first (one number, honest and simple), distance-weighted later (far-court
     pixels carry more court-error — same amplification as the foot-position work in `fix-later.md`).
   - `call = out` only when the point is outside the line by **more than** the band; `in` when inside by more
     than the band; **`uncertain`** in the band — and **forced `uncertain` near the frame edges** (DJI
     ultrawide barrel bow biases near-line calls; undistortion is still a no-op).
5. **Aggregate.** A rally-ending **out** landing attributable to a player's last shot increments that player's
   `player_shot_stats.errors` for that shot_type. The web sums these into a „Aus-Schläge" figure (the report
   already surfaces `player_shot_stats`; no new column).

## Data model (no DDL — columns already exist / nullable)
- **`events.type = 'landing'`** (new type on the existing generic `events` stream), payload:
  ```
  {located:bool, call:'in'|'out'|'uncertain', line:'side'|'baseline'|null,
   u?,v?, px?,py?, margin_cm?, band_cm, confidence}
  ```
  Reserve `'bounce'` for any raw mid-rally floor touches; `'landing'` is the **rally-ending** contact.
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
- **Singles-first** (doubles deferred — wider lines, more occlusion; CLAUDE.md decision 3).
- **Accept criterion needs labels:** a landing answer-key (a few labeled landings per session, ≥2 sessions)
  before any precision/recall claim. Until then: spot-check only, and the UI says „Schätzung".

## Defaults taken (noted, not asked)
Singles-first; flat ±band first; „Aus-Schläge" derived in the web by summing `player_shot_stats.errors` (no
new column); dedicated `events.type='landing'` for the rally-ending contact (`'bounce'` reserved for raw
mid-rally touches); **rally-ending out only** in v1 (mid-rally in/out and won-point landings come later).
