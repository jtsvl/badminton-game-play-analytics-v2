# 00 — Product Spec

## Vision
Give a club player and their trainer an automatic, video-derived **match report** after every recorded
game: where they moved, how much, how fast, what shots they hit, how rallies went — with the gameplay
video annotated by analysis overlays. No manual tagging; upload a clip, get insights.

## Personas
- **Admin / Trainer** (primary operator). Records matches, uploads clips, confirms the court calibration,
  reviews/repairs auto-detected data, publishes reports, shares links. Tech-comfortable, time-poor.
- **Club player** (consumer). Opens a personal link on their phone after a match, sees their own report
  (addressed as „du"), switches between players via tabs. Not logged into anything heavy.

## Core use cases
1. **Upload & process**: trainer uploads a clip for a match (singles or doubles), tags discipline/court/date,
   the system processes it offline and produces per-player stats + an overlay-ready timeline.
2. **Calibrate**: the system proposes the court's 4 corners; the trainer confirms or drags to fix, and
   selects the correct court when several are visible. One calibration per video/setup.
3. **Review**: before publishing, the trainer fixes obvious mistakes (player identity assignment, a
   mis-segmented rally, a wrong shot label). Low-friction, optional.
4. **View report**: player/trainer opens the report — mobile (stacked) or desktop (dashboard) — with KPIs,
   heatmap, zones, activity timeline, shot mix, rally stats, doubles formation, and the annotated video.
5. **Share**: each player gets a personal report link that orders tabs so they are „Du".

## Recording conditions (designed for)
- Cameras: iPhone or DJI Osmo Action 6. Tripod ~3–4 m, position **varies** between recordings.
- Angle: usually elevated back-of-court; **sometimes side** (then a reduced pipeline: shuttle + pose/gesture).
- Venue: public school halls — **multiple adjacent courts** in frame, **overlapping multi-sport lines**.
- **White/bright walls & ceiling** make the white shuttle low-contrast (first-class problem; see ML spec §5).
- Offline batch processing on a Mac Studio M4 Max is fine — **no real-time requirement**.

## In scope (target, phased)
Player detection/tracking, court auto-calibration with manual correction, movement insights (heatmap, zones,
distance, speed, sprints, timeline), shuttle tracking, rally segmentation, hit detection, stroke
classification, pose/footwork, doubles formation, the two report layouts, upload/calibration/review UX,
auth + personal sharing links, a self-hosted fine-tuning/labeling loop.

## Out of scope (for now)
Measurement-grade 3D/speed (Hawk-Eye needs many calibrated cameras), live/real-time analysis, multi-club
tenancy, natural-language tactical coaching (defer), mobile native apps (web only).

## Success criteria
- A trainer can go upload → calibrate → published report with no manual data entry beyond corner confirmation.
- Movement-tier insights (heatmap/zones/distance/speed) are trustworthy on the club's own footage after P1.
- Reports match the `handovers/` designs pixel-closely and read naturally in German du-form.
- ML outputs that are estimates are clearly labeled; users can correct identity/rally/shot errors.
- Everything runs locally on the Mac Studio against the Coolify Supabase; minors' footage never leaves the club.
