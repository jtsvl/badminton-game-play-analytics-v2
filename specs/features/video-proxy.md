# Feature — 720p playback proxy

**Status: BUILT (2026-07-02).** Serves FR-35 rally review and the report's Video-Analyse card.
Companion change to `features/rally-shot-review.md` (its "no video scrubber (v1)" scope line is
retired). One deviation from the spec'd design, found by the spike: **uploads go via TUS resumable
chunks** (`Store.upload_resumable`, 6 MiB chunks), not a plain POST — the Kong gateway kills any
single request at ~60 s (HTTP 499), which the larger proxies (~70–105 MB at ~9 Mbps) would hit.

## Problem (root cause, verified on the live instance)
The web app cannot show footage for **any real game**: ingest decodes the 4K originals **in place**
from the Mac Studio's local `videos/` dir (`WORKER_LOCAL_VIDEO_DIR`) and never uploads them — the
`match-videos` bucket holds only the 30 s test clip. The report player resolves a signed URL for an
object that doesn't exist, gets `''`, and silently falls back to footage-less mode. Uploading the 4K
originals is not the fix: ~5 GB/game is too large for the bucket, and the browser doesn't need
analysis-grade pixels. The trainer verifying rallies (drop/merge, FR-35) needs to *see* the video —
the numeric rows alone can't distinguish a neighbour-court leak from a real rally.

## Design: worker-generated proxy, view-only
A new worker stage **`proxy`** transcodes the already-resolved source (same resolution logic as
`ingest`: local `videos/` first, else the cached download) to a small streamable MP4 and uploads it:

- **ffmpeg CLI** (host dependency, Homebrew; **not** in `pyproject.toml`), hardware-encoded:
  `-vf scale=-2:720 -r 30 -c:v h264_videotoolbox -b:v 2500k -maxrate 3000k -bufsize 5000k -g 60
  -map 0:v:0 -map 0:a:0? -c:a aac -b:a 96k -movflags +faststart`.
  Audio is kept when present (hit sounds help verify rally boundaries). ~70–210 MB per game
  (source durations 220–650 s).
- Upload to **`analysis-artifacts`** at **`<video_id>/proxy_720p.mp4`** via **TUS resumable chunks**
  (`Store.upload_resumable`, exactly 6 MiB per chunk, `x-upsert` — re-runs are idempotent). Plain
  POSTs die at the gateway's ~60 s request cap (spiked: 39 MB passed at ~34 s, 60 MB got HTTP 499 at
  60.1 s; TUS chunks each take ~5 s). The dev anon storage-read policy (0003, `<video_id>/%` for
  `is_test` matches) already covers it: **no new RLS**.
- Record it in **`match_videos.proxy_path`** (migration `0010_video_proxy.sql`, one nullable text
  column; anon read exists via 0002). `null` = not generated. Skip transcode when `proxy_path` is
  already set unless `params.force`.
- **The stage never raises.** Any failure (ffmpeg missing, transcode error, upload rejected) is
  caught internally and logged as a loud warning. Load-bearing twice: the ingest slice must not fail
  over a supplementary artifact, and `process_job`'s error path sets `matches.status='failed'` — a
  backfill hiccup must not knock a match out of `review`.

### Wiring
- Register `proxy` in the pipeline registry (`_REAL` — an unregistered stage name is a `KeyError`
  at claim time).
- `ingest` slice becomes `["ingest", "proxy"] → 'calibrating'` — every new upload gets a proxy.
- **Backfill** = plain `processing_jobs` inserts with `stage='proxy'`; the `_slice_for` fallback runs
  a single stage with **no end-status change** (match status preserved). Scope: the 5 Session-2 games
  (status `review`); other games on demand.
- All ML stages keep reading the 4K frame cache. The proxy is playback-only, never analysis input.

## Web
- `lib/storage.ts`: resolve `proxy_path` via `signedUrl('analysis-artifacts', …)`.
- `useMatchReport.ts`: prefer the proxy; fall back to the raw source URL (keeps the test clip
  working). The Video-Analyse card and overlays are untouched — proxy duration/timeline equals the
  source's, and `track_keyframes`/`t_start_s` are in source seconds, so overlays and rally spans
  stay aligned.
- `RallyReview.tsx` (container): fetch `proxy_path`, render a **sticky** video panel above the board
  reusing `useVideoPlayer` + `VideoControls` (unmuted). Position-persistence key is
  `slug + '/rallies'` — reusing the report's per-slug key would overwrite its saved position.
- `RallyReviewBoard.tsx` stays presentational: new optional props `onPlayRally(row)` (per-row „▶"
  seeks to ~1 s before `tStart` and plays) and `activeRallyId` (highlights the row whose span
  contains the playhead). No proxy → props absent → today's list-only board.
- **Deliberately no auto-stop at rally end:** for a suspected dropout split, the evidence *for*
  merging is that play visibly continues past the detected end.

## Verification (done — the risky bit first)
1. **Spike (2026-07-02):** game 5 transcoded (2.07 GB HEVC → 39 MB H.264 High/yuv420p in 45 s,
   ~5× realtime), uploaded, signed URL serves HTTP 206 `video/mp4` (seeking works). Found the real
   limit: not a body-size cap but a **~60 s gateway timeout** → TUS chunked upload (proven with a
   60 MB probe, ten ~5 s chunks).
2. Stage + board TDD'd; backfill ran as standalone `stage='proxy'` jobs; E2E in the browser on
   `/match/2026-06-30-session2-game5/rallies`.

## Spec touches in the same change
`40-worker-contract.md` (stage table + slices), `20-data-model.md` (`match_videos.proxy_path`),
`features/rally-shot-review.md` (video panel replaces the "no video scrubber" line).

## Honest limits
- 30 fps / 720p is for *boundary* verification, not stroke analysis — fast smashes lose frames and
  detail vs the 60 fps 4K original, which stays the only analysis input.
- ffmpeg is a host dependency; a machine without it just gets no proxy (warning, no failure).
- Signed URLs live 1 h; an unusually long review session may need a reload.
- A full pipeline re-run does not regenerate the proxy (it's outside `RESUME_STAGES`) — it also
  doesn't need to: the source video, and thus the proxy, is immutable per `video_id`.
