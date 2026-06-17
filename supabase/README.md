# Supabase (self-hosted on Coolify)

Authoritative database for the app. Schema is versioned as plain SQL migrations here and applied to the
**Coolify-hosted** Postgres (not supabase.com). See `specs/20-data-model.md` for the model.

## Files
- `migrations/0001_init.sql` — full initial schema + `match_rally_stats` view + RLS (P0 stance).
- `seed.sql` — dev seed mirroring `handovers/**/mock-data.js` (Match 1 Einzel + Match 2 Doppel full).

## Apply

With `SUPABASE_DB_URL` from `.env` (the Coolify Postgres connection string):

```bash
psql "$SUPABASE_DB_URL" -f supabase/migrations/0001_init.sql
psql "$SUPABASE_DB_URL" -f supabase/seed.sql   # dev only
```

Or via the Supabase CLI if linked to the self-hosted project (`supabase db push`).

## Notes
- **RLS (P0):** `service_role` (worker) bypasses RLS; authenticated admin has full read/write. Anonymous /
  player-link token-scoped READ is **not** enabled yet — that's a P4 feature (`specs/features/sharing.md`).
  During web dev, authenticate as an admin user to read. Do not loosen to public read (minors' footage).
- **Storage buckets** (create in Supabase Studio / API): `match-videos` (raw uploads, private),
  `analysis-artifacts` (Parquet intermediates, private). Names are in `.env`.
- Migrations are forward-only and numbered; add `0002_*.sql`, etc. Keep them in sync with `20-data-model.md`.
- The rally seed rows are **generated** to plausible aggregates, not pixel-identical to the mock.
