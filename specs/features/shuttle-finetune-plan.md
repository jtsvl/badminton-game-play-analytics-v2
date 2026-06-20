# P1 Stage-5 Step 2 — Shuttle LABELING + FINE-TUNING (TrackNetV3)

> Implementation plan / runbook. Authored in a cloud planning session; **executed in a local
> Claude Code session on the Mac Studio**. Pick this up there.

## Context
Stage-5 slice 1 (committed `9bec142`) proved the **airspace mask solves multi-court isolation**
(on-court 7→26% TrackNetV3 / 8→20% WASB, cross-court scatter gone) but the broadcast-trained
trackers **don't lock onto our tiny low-contrast shuttle** in the wide overhead gym shot (coverage
46%/33%, detections cluster on far net/players). The spec's #1 ML ROI is therefore **domain
fine-tuning on our own footage**. This slice builds the *answer key + scoreboard* so we can label
our shuttle, fine-tune TrackNetV3 on it, and **measure** raw-pixel precision/recall on a held-out
clip — not eyeball it.

### Decisions locked with the user
- **Fine-tune TrackNetV3 only.** It has working `train.py` + a documented per-rally CSV
  (`Frame,Visibility,X,Y`), median-background is the built-in white-wall lever, and it was the
  stronger baseline. **WASB stays frozen** as the before/after benchmark (its public repo is
  eval-only).
- **Labeling = sparse-click + linear interpolation with a live path preview.** Click anchor frames
  on each arc, mark contact/occlusion boundaries, fill every frame by interpolation, mark gaps
  `visibility=0`. The labeler shows the interpolated path live so the user adds anchors at the arc
  apex where curvature is highest. (Parabolic-per-arc fit = deferred refinement.)
- **Fine-tune runs LOCALLY on the Mac Studio M4 Max (MPS).** This intentionally **overrides
  CLAUDE.md's "cloud NVIDIA for fine-tuning" convention** per the user's explicit choice; the
  fine-tune is small. Caveats honored: fp32, `PYTORCH_ENABLE_MPS_FALLBACK=1`, one CPU NaN-validation
  pass; flag any TrackNet train op that falls back to slow CPU.
- **Held-out split = whole clip.** We ingest ≥1 more white-wall/over-exposure clip; the test set is
  an entire clip never seen in training (adjacent 12fps frames are near-duplicates, so random-frame
  splits leak).

### Execution location (decided with user)
**All of this runs in a local Claude Code session on the Mac Studio**, not the cloud container —
the cloud box has no frame cache, no vendored repos, no weights (all git-ignored). Authoring 2a on
the Mac keeps the whole loop (build → label → fine-tune → benchmark) in one place against real
frames/weights, no PR round-trip. Slices 2a and 2b below describe that local work.

---

## Slice 2a — Answer-key + scoreboard infrastructure (code)

Mirror the existing player-correction stack at every layer:
`verify.py` (browser tool) → `corrections.json` → `import_corrections.py` → `Store.upsert_*` →
migration `0004`. New shuttle equivalents:

### 1. Migration `supabase/migrations/0005_shuttle_labels.sql`
Mirror `0004_detection_corrections.sql`. Table `shuttle_labels`:
`id uuid pk`, `video_id uuid → match_videos`, `source_id int`, `frame_idx int`, `t_s double`,
`px double`, `py double` (image-pixel click, or normalized 0..1 — match labeler output),
`visibility int` (0/1, or 0/1/2 if we keep occluded vs out-of-frame), `interpolated bool`
(anchor vs filled), `note text`, `created_at`, `unique(video_id, source_id)` for idempotent
re-import. Indexes on `(video_id)` and `(video_id, frame_idx)`. RLS enabled (worker uses
service-role). Apply via the `mcp__Supbase_Badminton_Analytics__apply_migration` MCP.

### 2. Shuttle-click mode in `worker/worker/verify.py`
Add a `--shuttle` flag to `verify_cli` (dispatched in `worker/worker/__main__.py` next to the
existing `verify`/`import-corrections` blocks). Reuse the ThreadingHTTPServer + HTML/JS canvas, PIL
frame rendering, `frames.json`/`frames/` cache loading, Cmd+Z undo, ← → frame nav, and the magnifier.
New behavior:
- **Click places the shuttle pixel** on the current frame (anchor). Shift/right-click marks a frame
  `visibility=0` (occluded / out of frame). A key marks a **contact/segment boundary** (interpolation
  never crosses it).
- **Live interpolated path**: between consecutive anchors within a visible segment, draw the filled
  per-frame positions so the user sees error and adds anchors at the apex.
- Persist to `{frame_cache_dir}/{video_id}/shuttle_labels.json` (same `_id`/`_save`/`_load`/dedup
  pattern as corrections). One label per `frame_idx`; re-clicking replaces.
- **TDD the pure logic** (no UI): `interpolate_anchors(anchors, segments) -> per-frame [(frame, px,
  py, visibility, interpolated)]` — linear fill within a segment, gaps→`visibility=0`, never cross a
  boundary. New module e.g. `worker/worker/shuttle/label.py`; tests in
  `worker/tests/test_shuttle_label.py`.

### 3. Importer `worker/worker/import_shuttle_labels.py`
Mirror `import_corrections.py`: `rows_from_shuttle_labels(video_id, labels) -> list[dict]`
(DB-free, unit-tested in `worker/tests/test_import_shuttle_labels.py`) + `import_shuttle_labels_cli`.
Add `Store.upsert_shuttle_labels(rows)` in `worker/worker/store.py` (mirror
`upsert_detection_corrections`, `on_conflict="video_id,source_id"`). Wire `import-shuttle-labels`
subcommand in `__main__.py`.

### 4. Label → TrackNetV3 dataset converter
`worker/worker/shuttle/dataset_export.py`: read `shuttle_labels` (from DB or the local JSON) for a
clip, emit TrackNetV3's training layout — per-rally **frame folders + label CSV with columns
`Frame,Visibility,X,Y`** (X,Y in original-frame pixels; confirm against the vendored repo's
`dataset.py` once cloned on the Mac). Group frames into rallies by the labeled segment boundaries.
**TDD the CSV row mapping + rally grouping** (pure) in `worker/tests/test_dataset_export.py`. Actual
frame-image export is thin I/O, verified on the Mac.

### 5. Ground-truth precision/recall in `worker/worker/eval_shuttle.py`
Today `stats()` is proxy-only (coverage/displacement/inside-court). Add a GT metric — the scoreboard:
- `score_against_labels(detections, labels, tol_px) -> dict` (precision, recall, F1, median pixel
  error). A detection is **TP if within `tol_px`** of the labeled shuttle on a visibility=1 frame;
  measured in **image/crop pixel space** (the fair metric — *not* court projection). **TDD the math**
  in `worker/tests/test_eval_shuttle_score.py`.
- New CLI path `--labels` (pull GT for `--video-id`) so `--compare` reports
  precision/recall/F1 per backend → the before/after table.

### 6. TrackNetV3 fine-tune config + local-MPS runbook
`worker/training/tracknetv3_finetune.md` (runbook) + a thin wrapper/config that invokes the vendored
`worker/.cache/_vendor/TrackNetV3/train.py` on the exported dataset: MPS device, fp32,
`PYTORCH_ENABLE_MPS_FALLBACK=1`, load `TrackNet_best.pt` as the init checkpoint, small LR, early-stop
on the val rallies. Runbook covers: clone vendored repo if absent, export dataset, train, copy new
weights to `models/weights/shuttle/ckpts/`, re-benchmark.

### 7. Spec updates (same change)
`specs/features/shuttle-tracking.md` and `specs/30-ml-pipeline.md`: mark the labeler/scoreboard built,
record the TrackNetV3-only + local-MPS + held-out-clip decisions, and add the before/after
precision/recall table (filled once measured on the Mac).

---

## Slice 2b — Hands-on, user-driven on the Mac
1. Ingest ≥1 more white-wall/over-exposure clip (reuse the existing `ingest` stage / frame cache).
2. Label ~15–30 rallies across clips (white-wall + over-exposure first) with the new tool;
   `import-shuttle-labels` to Postgres.
3. Export dataset; fine-tune TrackNetV3 on MPS per the runbook.
4. Re-benchmark **raw pixel** precision/recall on the **held-out clip** via
   `eval_shuttle --compare --labels`; record the fair winner in the spec. Don't fabricate metrics.

---

## Reuse (don't reinvent)
- `worker/worker/verify.py` — server, canvas, frame cache, undo, magnifier, `_save`/`_load`/dedup.
- `worker/worker/import_corrections.py` + `Store.upsert_detection_corrections` — importer pattern.
- `supabase/migrations/0004_detection_corrections.sql` — table/migration template.
- `worker/worker/geometry.py::apply_homography` — if we want a court-space sanity column (optional).
- `worker/worker/eval_shuttle.py::stats/format_stats/main` — extend, don't replace.
- `worker/worker/shuttle/{base,tracknetv3,wasb}.py` — backends unchanged; we only feed new weights.

## Verification
- **Code (CI-able):** `cd worker && uv run pytest` — new tests for interpolation, label→CSV
  conversion, importer row-mapping, and the precision/recall metric all green; existing 116 pass.
- **On the Mac (manual, recorded in spec):**
  - `uv run python -m worker verify --shuttle <video_id>` → click a rally, confirm
    `shuttle_labels.json`, then `import-shuttle-labels <video_id>` and verify rows via the
    Supabase MCP.
  - Export → fine-tune (MPS) → `eval_shuttle --video-id <held-out-id> --compare --labels` shows
    **fine-tuned TrackNetV3 precision/recall > frozen baseline**. That measured delta is the
    deliverable; if it doesn't beat baseline, report it honestly.

## Branch / commits
Continue on `claude/trusting-goodall-5xxige` (descends from the merged `feat/p1-shuttle`).
