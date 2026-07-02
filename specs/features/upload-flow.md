# Feature — Upload flow (web)

**Status: SPEC ONLY — the web page is a P0 stub.** FR-1/2/4. Until this is built, the **worker CLI is the
supported upload path** — state that plainly wherever the flow is documented.

## Current supported path (document, don't hide)
`python -m worker enqueue` (`worker/worker/enqueue.py`):

```
uv run python -m worker enqueue --path <storage-or-local path> --slug <unique-slug>
  [--discipline singles|doubles] [--label ...] [--angle overhead|side]
  [--played-on YYYY-MM-DD] [--analysis-fps 30] [--max-frames N]
```

It creates: the `matches` row (`status='uploaded'`, `is_test`), the `match_videos` row, a **placeholder**
roster (`players` "Spieler A1…" + `match_players` A/B slots), and one `processing_jobs
{stage:'ingest', status:'queued'}` row — all via `store.py` helpers. The polling worker picks it up;
after ingest the match sits at `calibrating` until the trainer confirms corners (`Calibrate.tsx`, which
`MatchList` links via „Feld kalibrieren →"). With `WORKER_LOCAL_VIDEO_DIR` set, a file already in
`videos/` is decoded in place — no storage download.

## Target (FR-1/2/4) — make `web/src/pages/Upload.tsx` real
1. **Upload** the clip to the `match-videos` Storage bucket. Multi-GB 4K clips need **resumable upload**
   (Supabase TUS endpoint) with progress UI — a plain POST will fail on hall Wi-Fi; treat this as a
   hard requirement, not polish.
2. **Form**: discipline, played-on date, court, label/angle, and the players — picked from the
   registered `players` table (no more placeholder roster). Duplicate-slug guard (mirrors the CLI's
   `get_match_by_slug` check) with an inline error.
3. **Insert the same rows the CLI creates** (matches / match_videos / match_players / processing_jobs
   ingest job) so the worker path is identical from there on: `uploaded → calibrating` gate →
   `Calibrate.tsx` → calibrate slice → `review`.
4. **Live status** via Supabase Realtime on `processing_jobs.progress`/`stage` and `matches.status`
   (the contract already names Realtime as the status channel — `40-worker-contract.md`).

## Dependency / sequencing
- **Auth-gated → depends on the deferred sharing/auth spec** (`features/sharing.md`, P4 — not yet
  written). Inserting rows and uploading to Storage from the browser needs an authenticated, RLS-scoped
  role; the current web app reads with the anon key under public-read policies (fine for reports, wrong
  for writes). **Until sharing/auth lands, the CLI stays the supported path.**
- The worker itself needs no changes — it already treats `stage='ingest'` jobs uniformly.

## Honest limits
- No client-side transcode/trim: whatever the camera wrote is what's uploaded (a 15-GB exercise clip
  included). Pre-trimming stays a manual step; the guide (`specs/recording-guide.md`) asks for one file
  per game at record time instead.
- Upload duration on club Wi-Fi for a ~5-GB game is minutes-to-tens-of-minutes; the TUS resume + a
  visible progress bar is the mitigation, not a speedup.
