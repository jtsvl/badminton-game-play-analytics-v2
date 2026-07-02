-- 720p playback proxy (specs/features/video-proxy.md). Storage path within analysis-artifacts
-- (`<video_id>/proxy_720p.mp4`), written by the worker `proxy` stage. null = not generated.
-- No new RLS: anon read on match_videos exists (0002); the analysis-artifacts storage-read
-- policy (0003) already covers `<video_id>/%` objects for is_test matches.
alter table match_videos add column if not exists proxy_path text;
