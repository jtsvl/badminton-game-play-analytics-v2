# Feature — Graduated autonomy (review becomes labeling + spot-check)

**Status: SPEC ONLY — target vision, phased.** Written 2026-07-02 after the first real rally-review
session, from the trainer's direct feedback: mandatory manual review per video is not an acceptable
long-term UX. This spec turns "less manual work over time" from tribal knowledge into the documented
target. It does not change any built behavior yet.

## Target UX (the trainer's own words, lightly edited)
> I have an upload process, the identified players are shown, and I give them a name label.

That's the whole intended interaction per match: **upload → wait → name the detected players → report.**
Calibration and rally checks happen automatically; correction UIs remain available but stop being
mandatory gates. Player identity **deliberately stays human** (decision 2026-07-02) — it is reframed
from a review/repair chore into the one natural, quick labeling step.

## Principle
**Confidence-gated auto-accept per gate, earned with data, honest in provenance.**
- Each gate flips from mandatory-review to auto-accept only when its measured error rate on
  trainer-reviewed games clears a bar — not when it "feels good".
- Every auto-accepted result is provenance-marked (auto vs. human-confirmed) and reversible; the
  existing review screens (`Calibrate.tsx`, `PlayerReview.tsx`, `RallyReview.tsx`) become correction
  paths reachable from the report, not steps in front of it.
- The trainer's review edits are exactly the ground truth that earns each gate its autonomy
  (`edited_at` / `rally_edited_at` provenance from `rally-shot-review.md`; calibration corner deltas).

## The three gates and their paths to autonomy

| Gate | Path | When |
|---|---|---|
| 1. Court calibration | Automate fully | P4 (planned) |
| 2. Rally quality | Auto-accept above confidence bar + spot-checks | After rule tuning proves itself |
| 3. Player identity | **Stays human** — becomes a lightweight labeling step | UX rework, no ML needed |

### Gate 1 — Calibration: automate fully (P4, already planned)
Auto-detect the court (CourtKeyNet direction, `fix-later.md` / build-plan P4), accept when detection
confidence ≥ threshold **and** the implied homography passes sanity checks (court dimensions plausible,
corners inside frame). Below the bar → today's manual confirm. The camera barely moves between games in
the same session, so reusing the previous game's accepted calibration as the prior makes this the
easiest gate.

### Gate 2 — Rally quality: earn auto-accept from the trainer's own edits
The review edits accumulate as an evaluation set: which detected rallies were false (dropped), which
were split (merged). Known recurring false positive already named by the trainer: **shuttle pushed back
to the other side after a rally** — signature: very short rally (1–2 shots) starting shortly after a
real rally ends. Steps, in order:
1. **Export/preserve edits** (cheap, do first): trainer edits are clobbered by a full pipeline re-run
   (loud warning only, `replace_rallies`). Snapshot reviewed rally lists per match so the eval set
   survives re-runs.
2. **Metric**: edits-per-game on reviewed games (drops + merges). Today's baseline: ~2 edits / ~23
   rallies (Session-2 game 5).
3. **Tune the segmentation rules** (`rally-seg` / in-rally gating are heuristics over the shuttle
   track, not a neural net) against the eval set — e.g. a post-rally cooldown or min-shots filter for
   the push-back pattern.
4. **Flip the default** when edits-per-game ≤ target (proposal: ≤1 across the trailing 10 reviewed
   games): matches skip the mandatory rally check; „Ballwechsel prüfen" stays linked from the report
   for spot-checks. Per-rally `confidence` (already stored) flags low-confidence rallies in the UI so
   spot-checking is guided, not random.

### Gate 3 — Player identity: keep the human, remove the friction
**Not automated — explicit decision.** Cross-session person re-identification would carry GDPR/consent
weight (FR-38, minors) for marginal benefit. Instead the gate becomes the trainer's one labeling moment:
- After processing, show each detected track as a **video crop** (frame + track bbox — both already in
  the cache/artifacts) instead of an abstract `track_label` in a `<select>`.
- The trainer types a name or picks a registered player per crop. Done — that binding is what
  `match_players` already models (`player-identity-review.md` contract spine, unchanged).
- **Upload-flow delta**: the upload form (`upload-flow.md` target step 2) no longer needs player
  selection up front — the placeholder roster stays until labeling. Selecting players at upload becomes
  an optional shortcut for known lineups, not a requirement.

## Status-flow sketch (mechanism, details when built)
Today: `… → review → done`, with `review` a hard human gate. Target: the pipeline records per-gate
confidence; when calibration + rally gates auto-accept, the match lands at a lighter **`labeling`**
state (waiting only for player names), or directly at `done` if the roster was pre-selected at upload.
`review` remains the state when any gate misses its bar. New provenance columns (per-gate
`auto_accepted` + confidence) keep reports honest about what a human actually verified.

## Sequencing / dependencies
- Gate-2 step 1 (edits export) is cheap and protects data being generated **right now** — do it near-term.
- Gate-2 tuning needs a few more reviewed sessions' worth of edits to be honest.
- Gate 1 rides on P4 court auto-detection; gate 3's crop-labeling UI can be built independently and
  improves UX even while review stays mandatory.
- The full "upload from the phone" story additionally needs `upload-flow.md` (auth-gated, P4 sharing).

## Honest limits
- Auto-accept claims must be **measured on reviewed games**; until the numbers exist, every gate stays
  manual. No silent flips.
- Zero-click is explicitly not the target: naming players stays human, by the trainer's own choice.
- Labeling binds names to tracks; it cannot fix tracking/detection errors (same limit as
  `player-identity-review.md`). Doubles identity through occlusion stays an estimate.
- A rally the detector never saw at all is invisible to review and to auto-accept alike — missed-rally
  recall can only be estimated against fully hand-labeled games (the FR-37 labeling loop), not from
  drop/merge edits.
