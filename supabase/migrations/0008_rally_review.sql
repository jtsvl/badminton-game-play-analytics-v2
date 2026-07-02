-- 0008_rally_review.sql — rally-review provenance + ⚠️ DEV ONLY ⚠️ anon write path
-- The no-auth rally-review UI (/match/:slug/rallies, FR-35) lets the trainer DROP a false rally
-- (neighbour-court leak) and MERGE two rallies a dropout split. Edits are human-authoritative
-- overrides (specs/features/rally-shot-review.md); a `rally_review` job then re-gates the stats.

-- ── provenance ──────────────────────────────────────────────────────────────
-- rallies.edited_at: null = machine-written (stage 6); set by the review UI on merge-modified rows.
alter table rallies add column if not exists edited_at timestamptz;
-- matches.rally_edited_at: set on EVERY rally-review save — a dropped rally leaves no row behind,
-- so drop provenance must live at match level. `replace_rallies` warns before clobbering either.
alter table matches add column if not exists rally_edited_at timestamptz;

-- ── DEV ONLY: anon writes for /match/:slug/rallies, SCOPED to is_test (matches 0003/0006). ──
-- Merge re-parents events.rally_id and updates the kept rally; drop deletes the rally's events
-- then the rally row (events.rally_id is ON DELETE SET NULL, so the delete must be explicit).
-- matches UPDATE + processing_jobs INSERT are already granted by 0003.
-- NOTE (corrected by 0009): this migration originally assumed no anon SELECT policy on events was
-- needed ("the review UI reads only rallies") — wrong: UPDATE/DELETE with a WHERE clause require the
-- rows to also pass a SELECT policy, so the writes below silently matched 0 rows. 0009 adds it.
-- REMOVE before real sharing: route writes through an authenticated admin / service_role edge
-- function and replace with token-scoped player-link RLS (see 0006 header).
grant update, delete on rallies to anon;
grant update, delete on events  to anon;

drop policy if exists dev_anon_rally_update on rallies;
create policy dev_anon_rally_update on rallies for update to anon
  using (exists (select 1 from matches m where m.id = rallies.match_id and m.is_test))
  with check (exists (select 1 from matches m where m.id = rallies.match_id and m.is_test));

drop policy if exists dev_anon_rally_delete on rallies;
create policy dev_anon_rally_delete on rallies for delete to anon
  using (exists (select 1 from matches m where m.id = rallies.match_id and m.is_test));

drop policy if exists dev_anon_event_update on events;
create policy dev_anon_event_update on events for update to anon
  using (exists (select 1 from matches m where m.id = events.match_id and m.is_test))
  with check (exists (select 1 from matches m where m.id = events.match_id and m.is_test));

drop policy if exists dev_anon_event_delete on events;
create policy dev_anon_event_delete on events for delete to anon
  using (exists (select 1 from matches m where m.id = events.match_id and m.is_test));
