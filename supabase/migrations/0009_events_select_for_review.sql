-- 0009_events_select_for_review.sql — ⚠️ DEV ONLY ⚠️ anon SELECT policy on events
-- Fixes a silent no-op in the 0008 rally-review write path, found in the first real E2E run:
-- Postgres RLS requires rows referenced by an UPDATE/DELETE *WHERE clause* to ALSO pass a SELECT
-- policy — the UPDATE/DELETE policy's USING alone is not enough. 0008 assumed "the review UI reads
-- only `rallies`, so no anon SELECT on events" — wrong: `update events set rally_id=… where
-- rally_id=…` and `delete from events where rally_id=…` both READ events.rally_id, and with no anon
-- SELECT policy they matched 0 rows without an error (the rally deletes then left orphaned events
-- via the FK's ON DELETE SET NULL).
-- anon already holds the table-level SELECT privilege (0002); only the policy was missing.
-- REMOVE before real sharing, together with 0008 (see 0006 header).

drop policy if exists dev_anon_event_select on events;
create policy dev_anon_event_select on events for select to anon
  using (exists (select 1 from matches m where m.id = events.match_id and m.is_test));
