-- 0003_dev_calibration_write.sql — ⚠️ DEV ONLY ⚠️
-- The no-auth calibration UI (P1, before the admin/auth login exists) needs to READ the calibration
-- row + still and WRITE the confirmed corners, flip match status, and enqueue the resume job. 0002
-- granted anon SELECT on most tables but OMITTED court_calibrations and processing_jobs.
--
-- Every WRITE grant and the storage READ are SCOPED to `is_test = true` matches/footage, so anon can
-- only ever touch test clips — never future real (non-test) matches or their private footage. Reads of
-- the (non-sensitive) calibration corners + job status stay broad to keep the report page simple.
--
-- REMOVE before real sharing: replace with token-scoped player-link RLS (specs/features/sharing.md)
-- and route writes through an authenticated admin / service_role edge function.

-- ── table grants: anon has only SELECT here by default; the operation must be granted for the RLS
--    policies below to apply. Row scoping to is_test stays enforced by those policies. ───────────
grant update on court_calibrations to anon;
grant update on matches to anon;
grant select, insert, update on processing_jobs to anon;

-- ── reads 0002 missed (corners + job status are not sensitive) ───────────────
drop policy if exists dev_anon_read on court_calibrations;
create policy dev_anon_read on court_calibrations for select to anon using (true);

drop policy if exists dev_anon_read on processing_jobs;
create policy dev_anon_read on processing_jobs for select to anon using (true);

-- ── writes the calibrate "Speichern" performs — TEST MATCHES ONLY ───────────
drop policy if exists dev_anon_calib_update on court_calibrations;
create policy dev_anon_calib_update on court_calibrations for update to anon
  using (exists (select 1 from matches m where m.id = court_calibrations.match_id and m.is_test))
  with check (true);

drop policy if exists dev_anon_match_update on matches;
create policy dev_anon_match_update on matches for update to anon
  using (is_test) with check (is_test);

drop policy if exists dev_anon_jobs_insert on processing_jobs;
create policy dev_anon_jobs_insert on processing_jobs for insert to anon
  with check (exists (select 1 from matches m where m.id = match_id and m.is_test));

drop policy if exists dev_anon_jobs_update on processing_jobs;
create policy dev_anon_jobs_update on processing_jobs for update to anon
  using (exists (select 1 from matches m where m.id = processing_jobs.match_id and m.is_test));

-- ── anon storage reads for signed URLs — TEST FOOTAGE ONLY ──────────────────
-- match-videos: only source videos of is_test matches. analysis-artifacts: only objects keyed by
-- an is_test match's video_id (e.g. "<video_id>/calib_frame.jpg"). Future real footage stays private.
drop policy if exists dev_anon_storage_read on storage.objects;
create policy dev_anon_storage_read on storage.objects for select to anon
using (
  (bucket_id = 'match-videos' and exists (
    select 1 from match_videos mv join matches m on m.id = mv.match_id
    where m.is_test and mv.storage_path = storage.objects.name))
  or
  (bucket_id = 'analysis-artifacts' and exists (
    select 1 from match_videos mv join matches m on m.id = mv.match_id
    where m.is_test and storage.objects.name like mv.id::text || '/%'))
);
