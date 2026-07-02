# Feature — Rally & shot review (the unbuilt half of FR-35)

**Status: SPEC ONLY — no code yet.** FR-35 names three repairs: player→identity assignment (**built** —
`PlayerReview.tsx` + `remap`), **drop/merge a rally**, and **relabel a shot** (both unbuilt, spec'd here).

## Goal
Rallies and shot labels are estimates; the trainer is the authority. Before publishing a report, let them
**drop** a false rally (e.g. a neighbour-court leak), **merge** two rallies a dropout split, and — once
stage 7 exists — **relabel** a shot's type. Edits are human-authoritative overrides; a re-aggregate then
propagates them into the stats.

## Mirror the built pattern (PlayerReview → remap)
The identity-review flow is the template, verified in code (`PlayerReview.tsx` `onSave`):
diff-only writes → set `matches.status='processing'` → insert a `processing_jobs` row for a **cheap
resume slice** → navigate back; the worker recomputes without re-detecting. Rally review copies that
page structure, diff-save, and job-enqueue verbatim — only the table and the slice differ.

## The load-bearing design point (verified — resolve it, don't trip on it)
**`aggregate` reads rally spans from `rallies.parquet`** (ctx/cache fallback, `_rally_spans` in
`aggregate.py`), **not from the DB** — so a DB-only rally edit would be silently invisible to
re-aggregation. Resolution (recommended): a new **`rally_sync` recompute stage** at the head of the
review slice — it reads the *edited* `rallies` (+ `events`) rows from the DB and **rewrites
`rallies.parquet`** (and `events.parquet`) in the cache before `aggregate` runs. `aggregate` stays
untouched; this is exactly how `remap.py` resolves `tracks.parquet` for the identity flow.
Job slice: `rally_review → [rally_sync, aggregate, overlay] → done`.

## Data & operations
Backing rows: `rallies` (spans, `winner_side`, `end_shot_type`, `shot_count`) + `events`
(serve/hit/rally_end, FK `rally_id`). The worker writes them **delete-then-insert** per match via
`store.replace_rallies` (`store.py`) — trainer edits therefore live in the same tables and would be
wiped by a full pipeline re-run: record each edit's provenance (`payload {edited: true}` or an
`edited_by` column, small DDL) so a re-run can either preserve confirmed edits or at minimum surface
that it clobbered them. **Never touch `matches.score`** — it stays human-entered, and rally-derived
fields stay estimates decoupled from it (`40-worker-contract.md` contract).

- **Drop rally**: delete the `rallies` row + its `events` (FK cascade or explicit) → enqueue recompute.
  The rally's time span simply returns to "dead time" for movement gating.
- **Merge rallies**: replace two adjacent rows with one spanning `[t_start_1, t_end_2]`; re-parent both
  rallies' events; recount `shot_count` (provisional `count_hits` over the merged span, or sum as a
  first cut) → enqueue recompute.
- **Relabel shot** (needs stage 7): update the `events` hit row's `shot_type` in place; recompute
  `player_shot_stats` for the affected players. Meaningless before `stroke` exists — **sequence rally
  drop/merge first**; ship shot relabel with stage 7.

## UI sketch (small, list-first — no video scrubber in v1)
A `RallyReview` page linked from the report at `review` status (like „Feld kalibrieren" on `MatchList`):
rallies as rows (start–end, duration, shot count, confidence flags), overlay-thumbnail per rally if cheap,
checkboxes → drop, adjacent-pair select → merge, save = diff → jobs flow above. German, du-form, per UI
conventions.

## Honest limits
- Boundary *adjustment* (drag start/end) is deferred — drop/merge covers the observed failure modes
  (foreign-court leaks, dropout splits) without a timeline UI. Revisit after the rally labeler exists.
- Merging recomputes `shot_count` provisionally; until stage 7, counts stay `count_hits` estimates.
- An edit marks human truth for *this* processing run; a later full re-run regenerates rallies from
  scratch (provenance note above) — the worker must not present regenerated rows as still-reviewed.
