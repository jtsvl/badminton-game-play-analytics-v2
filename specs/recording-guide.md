# Aufnahme-Guide — Checkliste vor jeder Session

One-pager for recording match footage the pipeline can digest well. Consolidates the capture knowledge
from `fix-later.md` and the shuttle specs (as of 2026-07-02). German checklist first, English rationale after.

## Checkliste (vor dem Start)

- [ ] **FOV-Modus: „Standard (Dewarp)"** an der Osmo Action 6 — *nicht* Wide/155°.
- [ ] **Stativ** hinter der Grundlinie, **erhöht** (~3–4 m), Kamera fest — nicht aus der Hand.
- [ ] **Bildausschnitt:** das ganze Zielfeld **plus Luftraum darüber** (hohe Clears!); Nachbarfelder so
      wenig wie möglich im Bild.
- [ ] **4K, ~60 fps** (wie bisher) — nicht reduzieren.
- [ ] **Belichtung prüfen:** die weiße Wand darf nicht ausbrennen (Probeaufnahme: ist ein hoch
      geschlagener Federball vor der Wand erkennbar?). Ggf. Belichtung leicht runterregeln (−0.3…−0.7 EV).
- [ ] **Eine Datei pro Spiel** (Aufnahme zwischen den Spielen stoppen) — das erspart das Schneiden.
- [ ] Nach der Session: Dateien nach `videos/` kopieren, Spiele wie gehabt registrieren.

## Warum (rationale, honest caveats included)

- **Standard (Dewarp) statt Wide:** the Wide/155° mode has strong barrel distortion — the trainer visually
  confirmed (2026-06-24) that painted court lines **bow** under the straight calibration overlay. A single
  4-corner homography can't fit curved lines: corners sit right but mid-line/edges drift, biasing player
  positions, heatmaps and court-space shuttle geometry. Dewarp does the rectilinear correction in-camera →
  straight lines → clean homography, no code needed. It **crops the FOV**, so mount slightly further
  back/higher than before to keep the full court + airspace in frame.
  **Honest caveat:** the shuttle model was fine-tuned on *distorted wide* frames — switching FOV changes its
  pixel domain. Expect somewhat lower shuttle detection on the first Dewarp session and include Dewarp clips
  in the next fine-tune round (the labeling pass is pending anyway, see `fix-later.md`).
- **~60 fps 4K source:** the 30 fps analysis ingest divides cleanly out of 60, and the 60 fps original stays
  re-ingestable for smash analysis later. Storage is cheap; missed frames are gone forever.
- **Placement (elevated, behind the baseline):** the homography error grows toward the far court — the higher
  and steadier the camera, the less the far half is compressed. Hand-held/standing drift shifts the court
  anchor over the clip (a single static homography assumes a fixed camera).
- **Minimize neighbour courts:** foreign-shuttle intrusion is only *mitigated* by the airspace mask + rally
  continuity, not solved (`fix-later.md`). Every neighbour court out of frame is noise that never happens.
- **Exposure / white wall:** white-shuttle-on-white-wall is the pipeline's hardest case (locked decision #5).
  A blown-out wall destroys the little contrast there is. Slightly under-exposed footage is better for the
  tracker than clipped highlights. Halls/lighting we haven't recorded yet are worth capturing for fine-tune
  diversity — variety is training data.
