-- 0004_detection_corrections.sql — persist verify-tool detection corrections (the scorecard answer key)
--
-- The `worker verify` tool lets the trainer fix player boxes / feet / roles on a processed clip and writes
-- them to a cache `corrections.json` (write-only today). This table is the durable home for that review so
-- the accuracy scorecard (worker/eval_corrections.py full mode) and every later detection fix re-run against
-- a persisted ground truth — clip f35d0984 is the first saved answer key. Mirrors the verify.py record schema
-- (specs/30-ml-pipeline.md stage 4, the detection-accuracy work in specs/fix-later.md).
--
-- Worker-only: written by `python -m worker import-corrections <video_id>` and read by the scorecard, both
-- over the service-role key (which bypasses RLS). RLS is enabled with NO anon policy — unlike calibration
-- (0003) the web never reads or writes this; it is an offline evaluation artifact.

create table if not exists detection_corrections (
  id            uuid primary key default gen_random_uuid(),
  video_id      uuid not null references match_videos(id) on delete cascade,
  source_id     int,                                  -- verify tool's per-correction `_id` (unique within a video)
  frame_idx     int  not null,
  t_s           double precision,                     -- timestamp (s) of the corrected frame
  role          text,                                 -- player label A1/A2/B1/B2 (A* = near, B* = far)
  type          text,                                 -- bbox | position | added | spurious | wrong_role
  bbox          jsonb,                                -- corrected [x,y,w,h] normalized 0..1 (image space)
  orig_bbox     jsonb,                                -- pipeline's original box (for the before/after delta)
  foot_norm     jsonb,                                -- corrected foot [nx,ny] normalized 0..1
  court_uv      jsonb,                                -- corrected court [u,v] in [0,1]
  orig_foot     jsonb,                                -- pipeline's original foot [nx,ny]
  orig_uv       jsonb,                                -- pipeline's original court [u,v]
  correct_role  text,                                 -- wrong_role: the label it should be
  edited_in     text,                                 -- 'video' | 'map' (which view the trainer edited)
  note          text,
  created_at    timestamptz not null default now(),
  unique (video_id, source_id)                        -- idempotent re-import (upsert key)
);

create index if not exists detection_corrections_video_idx on detection_corrections(video_id);
create index if not exists detection_corrections_video_frame_idx on detection_corrections(video_id, frame_idx);

-- Lock it down: service-role (worker + scorecard) bypasses RLS; no anon/authenticated policy is granted,
-- so the web never sees this offline evaluation data.
alter table detection_corrections enable row level security;
