-- 0005_shuttle_labels.sql — persist verify-tool SHUTTLE labels (the fine-tune answer key + scoreboard GT)
--
-- The `worker verify --shuttle` tool lets the trainer click the shuttle pixel per frame on a processed clip
-- (sparse anchors + segment boundaries) and writes them to a cache `shuttle_labels.json`. The importer
-- expands those anchors to a DENSE per-frame answer key (linear fill within a segment, gaps → visibility=0)
-- and stores it here. This is the durable ground truth that (a) the TrackNetV3 fine-tune trains against and
-- (b) eval_shuttle.py scores raw-pixel precision/recall against — clip f35d0984 is the first saved key.
-- Mirrors 0004_detection_corrections.sql (specs/features/shuttle-finetune-plan.md §1, stage 5).
--
-- One row per frame: `source_id` carries the frame index so `unique(video_id, source_id)` makes re-import
-- idempotent per frame (re-labeling replaces). px,py are NORMALIZED 0..1 image coords (the labeler's click
-- model); a row with visibility=0 (occluded / out of frame) leaves px,py null. `interpolated` distinguishes
-- a hand-clicked anchor (false) from an interpolation-filled frame (true).
--
-- Worker-only: written by `python -m worker import-shuttle-labels <video_id>`, read by the exporter +
-- scoreboard, all over the service-role key (which bypasses RLS). RLS enabled with NO anon policy — unlike
-- calibration (0003) the web never reads or writes this; it is an offline evaluation/training artifact.

create table if not exists shuttle_labels (
  id            uuid primary key default gen_random_uuid(),
  video_id      uuid not null references match_videos(id) on delete cascade,
  source_id     int,                                  -- = frame_idx (the per-frame idempotent upsert key)
  frame_idx     int  not null,                        -- frame position in the clip's frames.json manifest
  t_s           double precision,                     -- timestamp (s) of the frame
  px            double precision,                     -- shuttle x, normalized 0..1 (image space); null if invisible
  py            double precision,                     -- shuttle y, normalized 0..1 (image space); null if invisible
  visibility    int,                                  -- 1 = visible, 0 = occluded / out of frame
  interpolated  boolean,                              -- false = hand-clicked anchor, true = interpolation-filled
  note          text,
  created_at    timestamptz not null default now(),
  unique (video_id, source_id)                        -- idempotent re-import (upsert key)
);

create index if not exists shuttle_labels_video_idx on shuttle_labels(video_id);
create index if not exists shuttle_labels_video_frame_idx on shuttle_labels(video_id, frame_idx);

-- Lock it down: service-role (worker + exporter + scoreboard) bypasses RLS; no anon/authenticated policy is
-- granted, so the web never sees this offline training/evaluation data.
alter table shuttle_labels enable row level security;
