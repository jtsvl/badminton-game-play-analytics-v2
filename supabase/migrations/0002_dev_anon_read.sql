-- 0002_dev_anon_read.sql — ⚠️ DEV ONLY ⚠️
-- Grants anonymous READ so the seeded demo report renders during P0 before the admin/auth UI exists.
-- REMOVE this before any real (non-test) footage is stored. In P4 this is replaced by token-scoped
-- player-link RLS (specs/features/sharing.md). Writes remain restricted to authenticated/service_role.

do $$
declare t text;
begin
  foreach t in array array[
    'players','matches','match_videos','match_players','player_match_stats',
    'player_shot_stats','pair_match_stats','rallies','track_keyframes'
  ] loop
    execute format('drop policy if exists dev_anon_read on %I;', t);
    execute format('create policy dev_anon_read on %I for select to anon using (true);', t);
  end loop;
end $$;
