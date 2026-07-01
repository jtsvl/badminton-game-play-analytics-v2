-- 0007_calibration_frames.sql — multiple candidate stills for the calibration screen
--
-- The calibration UI showed a single still (`frame_uri`, grabbed at ~5 s in ingest). When a player
-- happens to stand in front of a court corner in that one frame the trainer can't place the handle.
-- Ingest now uploads N evenly-spaced snapshots (analysis-artifacts/<video_id>/calib_frames/NNN.jpg)
-- and records them here so the web can step between them (labeler-style ◀/slider/▶). The court is
-- static, so switching the background never moves the corners. Backward compatible: null keeps the
-- old single-frame behavior (web falls back to `frame_uri`). (specs/30-ml-pipeline.md stage 0.)
--
-- No RLS change needed: the existing anon calibration policy (0003) covers new columns, and the web
-- already signs objects in analysis-artifacts (the current single still loads fine).

alter table court_calibrations
  add column if not exists frame_uris jsonb;   -- [{ "uri": text, "s": number }, …], time-ordered
