# Feature — Rally & shot review (FR-35)

**Status: drop/merge BUILT (2026-07-02, `feat/rally-review`); shot relabel deferred** until stage 7
(`stroke`) exists. FR-35 names three repairs: player→identity assignment (**built** — `PlayerReview.tsx` +
`remap`), **drop/merge a rally** (**built** — `RallyReview.tsx` + `rally_sync`), and **relabel a shot**
(deferred, spec'd below).

## Goal
Rallies and shot labels are estimates; the trainer is the authority. Before publishing a report, let them
**drop** a false rally (e.g. a neighbour-court leak), **merge** two rallies a dropout split, and — once
stage 7 exists — **relabel** a shot's type. Edits are human-authoritative overrides; a re-aggregate then
propagates them into the stats.

## Built: the flow (mirrors PlayerReview → remap)
`/match/:slug/rallies` (`RallyReview.tsx`, linked „Ballwechsel prüfen →" from the report's rally card /
dashboard at status `review`/`done`) → flags per row (drop-checkbox, merge-with-previous toggle) →
`rallyPlan.ts` folds them into a save plan (pure, unit-tested) → `onSave` executes **merges → event
re-parents → deletes** (order is load-bearing, see below), sets `matches.status='processing'` +
`rally_edited_at`, enqueues `processing_jobs` `stage='rally_review'` with
`params.return_status` = the status before the edit → worker runs the slice → match returns to its
prior status (`review` must not skip the PlayerReview confirmation).

Every web update/delete chains `.select('id')` and **throws on zero affected rows** — PostgREST reports
RLS-blocked writes as success-with-zero-rows, which must not look like a saved edit (0008 grants the dev
anon write path, `is_test`-scoped). **Lesson from the first real E2E run (0009):** Postgres RLS requires
rows referenced by an UPDATE/DELETE **WHERE clause** to also pass a **SELECT policy** — 0008's event
update/delete policies alone silently matched 0 rows (no error), and the rally deletes then orphaned the
events via the FK's ON DELETE SET NULL. 0009 adds the anon SELECT policy on `events`; the row-count
assert doesn't cover event writes (a rally may legitimately have zero events), so policy changes there
must be E2E-tested, not assumed.

## The load-bearing design point (resolved by `rally_sync`)
**`aggregate` reads rally spans from `rallies.parquet`** (ctx/cache fallback, `_rally_spans` in
`aggregate.py`), **not from the DB** — a DB-only rally edit would be silently invisible to
re-aggregation. Resolution (built): **`pipeline/rally_sync.py`** reads the *edited* `rallies` + `events`
rows from the DB (new `Store.get_rallies`/`get_events`) and **rewrites `rallies.parquet`/`events.parquet`**
in the cache (stage-6 schemas, events' `rally_id`→`idx` remap, orphans skipped with a logged count) before
`aggregate` runs, then re-uploads both artifacts. `aggregate` stays untouched.

Job slice: `rally_review → [remap, rally_sync, aggregate] → params.return_status`. Deviations from the
original sketch, both verified in code: `remap` is **in** (aggregate hard-requires `ctx["tracks_path"]`,
which only remap resolves) and `overlay` is **out** (it never reads rallies — nothing it produces changes).
`rally_sync` also runs in the `remap` slice (`[remap, rally_sync, aggregate, overlay]`) so identity
recomputes keep the DB-authoritative spans — a fresh worker cache used to silently fall back to
whole-clip gating.

## Data & operations
Backing rows: `rallies` (spans, `winner_side`, `end_shot_type`, `shot_count`) + `events`
(serve/hit/rally_end, FK `rally_id`). The worker writes them **delete-then-insert** per match via
`store.replace_rallies` (`store.py`) — trainer edits therefore live in the same tables and would be
wiped by a full pipeline re-run. **Provenance (0008):** `rallies.edited_at` marks merge-modified rows;
`matches.rally_edited_at` is set on **every** review save (a dropped rally leaves no row behind, so drop
provenance must live at match level). `replace_rallies` checks `rally_edited_at` before its delete and
**logs a loud warning** when a full re-run clobbers trainer edits — the "at minimum surface it" bar; full
edit preservation across re-runs stays out of scope. **Never touch `matches.score`** — it stays
human-entered, and rally-derived fields stay estimates decoupled from it (`40-worker-contract.md`).

- **Drop rally** (built): delete the rally's `events` explicitly, then the `rallies` row —
  `events.rally_id` is **ON DELETE SET NULL**, not cascade, so skipping the event delete would leave
  orphans. The rally's time span returns to "dead time" for movement gating.
- **Merge rallies** (built): keep the first row — `t_end_s` = last member's, `shot_count` = sum (the
  spec-sanctioned first cut), `winner_side`/`end_shot_type` = last member's (end attributes travel with
  the end), `confidence` = **min** (the merged span includes the dropout gap), `edited_at` = now.
  **Re-parent the absorbed rallies' events BEFORE deleting their rows** (same SET-NULL trap). Consecutive
  merge toggles fold into one group; `idx` is never renumbered (gaps are safe — `unique(match_id,idx)`
  holds, consumers order by idx; the UI shows positional numbers).
- **Relabel shot** (deferred, needs stage 7): update the `events` hit row's `shot_type` in place;
  recompute `player_shot_stats` for the affected players. Meaningless before `stroke` exists.

## Correction corpus (eval answer key, built alongside drop/merge)
Every review save also persists **what the trainer judged wrong** as append-only `rally_corrections`
rows (0011) — the eval/threshold-tuning corpus for stage 6's segmentation **heuristic**
(`worker/rally/segment.py` constants — not a trained model, so this is an answer key like
`detection_corrections` (0004), not NN supervision; it doubles as bootstrap labels if learned
sub-classifiers ever land).

- **Trigger is automatic** — no manual CLI step, unlike `detection_corrections`' offline
  `worker verify` → `corrections.json` → `import-corrections` flow. The web builds the rows with
  `rallyCorrections.ts` (pure mapper: pre-edit snapshot + `RallySavePlan`) and packs them into
  `params.corrections` on the `rally_review` job it already enqueues on every save; the worker's
  `rally_sync` stage inserts them server-side (`store.insert_rally_corrections`). Access posture
  mirrors `detection_corrections`: service-role only, RLS with no anon policy — the dev-only anon
  surface on `rallies`/`events` does not grow.
- **Three `op` values** (the client plan's `rallyDeletes.deleteEvents` flag conflates the first two):
  `drop` (rally judged noise), `merge_absorbed` (folded into a neighbour; `merged_into` = surviving
  id at edit time), `merge_keep` (the survivor; `patch` = the exact `RallyPatch` written). All rows
  carry the pre-edit `orig_*` snapshot.
- **`rally_id`/`merged_into` are deliberately not FKs** — dropped/absorbed ids are gone by insert
  time, and even surviving ids go stale on the next full stage-6 re-run (`replace_rallies` is
  delete-then-insert). A correction must outlive the rally it describes. Same reasoning drives
  `match_id`/`video_id` **ON DELETE SET NULL** (the `job_runs` audit precedent, not
  `detection_corrections`' cascade): the signal stays meaningful after dev cleanup deletes.
- **Failure isolation:** the insert is wrapped in its own try/except inside `rally_sync.run` — a
  broken write logs, but never blocks the rallies/stats resync the trainer is waiting on.
- **Out of scope for now:** the scorecard (`eval_rally_corrections.py`, mirroring
  `eval_corrections.py`) and anything consuming the corpus — deferred until enough corrections
  accumulate to be meaningful, like `detection_corrections` before it (`specs/fix-later.md`).

## UI (built — list + video panel)
`RallyReview.tsx` (container, mirrors `PlayerReview.tsx`) + `RallyReviewBoard.tsx` (presentational,
tested) + `rallyPlan.ts` (pure save-plan builder + validation, tested). Rows: Nr., start–end (`fmtTime`),
Dauer, Schläge, Konfidenz (de-DE), „bearbeitet"-badge for already-edited rows; per-row „Entfernen"
checkbox; „Mit vorherigem zusammenführen" toggle between rows (not on the first). Validation blocks
merging next to a dropped rally. German du-form, brand tokens.

**Video (built with `features/video-proxy.md`):** the container renders a sticky player over the 720p
proxy (`resolvePlaybackSrc`, unmuted — hit sounds help judge boundaries) reusing `useVideoPlayer` (position
key `slug + '/rallies'`) + `VideoControls`. Each row gets a „▶ Ansehen" button (board prop `onPlayRally`)
seeking to ~1 s before the rally start; the row under the playhead is highlighted (`activeRallyId`,
`aria-current`). **Deliberately no auto-stop at the rally end** — play visibly continuing past the
detected end is the evidence for a merge. No proxy → list-only fallback (the original v1 UI).

## Honest limits
- Boundary *adjustment* (drag start/end) is deferred — drop/merge covers the observed failure modes
  (foreign-court leaks, dropout splits) without a timeline UI. Revisit after the rally labeler exists.
- Merging recomputes `shot_count` provisionally (sum); until stage 7, counts stay `count_hits` estimates.
- An edit marks human truth for *this* processing run; a later full re-run regenerates rallies from
  scratch — surfaced only via the `replace_rallies` warning, and the regenerated rows carry no
  `edited_at`, so they are honestly presented as machine output again.
- Dropping **all** rallies empties `match_rally_stats` (a view over `rallies`) and returns `aggregate`
  to whole-clip gating — intended fallback, not an error.
