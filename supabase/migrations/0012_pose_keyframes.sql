-- 0012_pose_keyframes.sql — per-player skeleton overlay keyframes (specs/features/pose-skeleton-overlay.md).
-- One row per (video, player): the in-rally slice of pose.parquet (RTMPose Halpe-26), written by the
-- worker `overlay` stage. data = [{t, kp:[[x,y,score] × 26]}] — x,y are FRAME-NORMALIZED image coords
-- (keypoint px ÷ match_videos native width/height), NOT court (u,v): elevated joints (shoulders/elbows/
-- wrists) sit above the floor plane where the ground homography is geometrically wrong. Follows the
-- rally_end {x,y} / landing {px,py} precedent: normalized [0,1] image coords, never raw pixels.
-- `format` names the joint-index dictionary so a future pose-model swap leaves old rows self-describing.
-- Visualization only for now — future derived metrics (joint angles) land as an additive `angles` key
-- per frame or a nullable jsonb column, never a reshape of `kp`.

create table if not exists pose_keyframes (
  id         uuid primary key default gen_random_uuid(),
  video_id   uuid not null references match_videos(id) on delete cascade,
  subject    text not null,                    -- 'player:<player_id>' (mirrors track_keyframes)
  format     text not null default 'halpe26',  -- joint-index dictionary version
  data       jsonb not null,                   -- [{t, kp:[[x,y,score] × 26]}]
  created_at timestamptz not null default now(),
  unique (video_id, subject)
);

create index if not exists pose_keyframes_video_idx on pose_keyframes(video_id);

-- Web-visible table → needs BOTH house policies (RLS with no policies is this repo's idiom for
-- web-INVISIBLE tables, e.g. 0011 rally_corrections). No table GRANTs needed: the schema's default
-- ACL already gives anon SELECT / authenticated ALL to new tables.
alter table pose_keyframes enable row level security;

drop policy if exists admin_all on pose_keyframes;
create policy admin_all on pose_keyframes
  for all to authenticated using (public.is_admin()) with check (public.is_admin());

-- ⚠️ DEV ONLY (mirrors 0002_dev_anon_read — remove together with it before real sharing / P4 token RLS)
drop policy if exists dev_anon_read on pose_keyframes;
create policy dev_anon_read on pose_keyframes
  for select to anon using (true);
