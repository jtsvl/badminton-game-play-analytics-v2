-- 0006_dev_review_write.sql — ⚠️ DEV ONLY ⚠️
-- The no-auth player-identity review UI (/match/:slug/review, FR-35) needs to WRITE the corrected
-- track→player mapping: update match_players (player_id / track_label), flip match status, and enqueue
-- the cheap `remap` recompute job. 0003 already granted anon is_test-scoped UPDATE on matches
-- (dev_anon_match_update) and INSERT/UPDATE on processing_jobs (dev_anon_jobs_insert/update); the only
-- gap is match_players, which anon has SELECT on (0002) but not UPDATE.
--
-- The WRITE grant + policy are SCOPED to `is_test = true` matches, so anon can only ever touch test
-- clips — never future real (non-test) matches.
--
-- REMOVE before real sharing: route writes through an authenticated admin / service_role edge function
-- and replace with token-scoped player-link RLS (specs/features/sharing.md), like 0002/0003.

-- ── table grant: anon has only SELECT on match_players (0002); the operation must be granted for the
--    RLS policy below to apply. Row scoping to is_test stays enforced by the policy. ────────────────
grant update on match_players to anon;

-- ── the write the review "Speichern" performs — TEST MATCHES ONLY ───────────
drop policy if exists dev_anon_review_update on match_players;
create policy dev_anon_review_update on match_players for update to anon
  using (exists (select 1 from matches m where m.id = match_players.match_id and m.is_test))
  with check (exists (select 1 from matches m where m.id = match_players.match_id and m.is_test));
