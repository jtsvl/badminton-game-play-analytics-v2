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
  eval-only). V3-only is a deliberate scope choice for this slice; if fine-tuned V3 recall stays low
  on the white-wall test clip, the next lever is **TrackNetV4 motion-attention** (now MIT;
  `30-ml-pipeline.md:97`), which targets exactly this appearance-poor / motion-distinct failure mode.
- **Labeling = sparse-click + linear interpolation with a live path preview.** Click anchor frames
  on each arc, mark contact/occlusion boundaries, fill every frame by interpolation, mark gaps
  `visibility=0`. The labeler shows the interpolated path live so the user adds anchors at the arc
  apex where curvature is highest. **In practice this is not very "sparse": a drag-decelerated shuttle
  arc needs ~1 anchor every 2–4 frames, plus mandatory anchors on contact frames; the live preview
  should surface max-interpolation-error so sloppy GT can't slip in (interpolated frames feed BOTH the
  training target and the metric, so error compounds against the 4px tolerance).** (Parabolic-per-arc
  fit = deferred refinement — promote it if linear error exceeds the `tol_px=4` bar.)
- **Fine-tune runs LOCALLY on the Mac Studio M4 Max (MPS).** This intentionally **overrides
  `specs/30-ml-pipeline.md`'s "cloud NVIDIA only for fine-tuning runs" convention** (it is *not* in
  CLAUDE.md) per the user's explicit choice; the fine-tune is small. **This requires patching the
  vendored TrackNetV3 `train.py`/`test.py` first — they are hard-wired to `.cuda()` with no `--device`
  flag, so `PYTORCH_ENABLE_MPS_FALLBACK=1` alone will NOT make them run on MPS** (it only redirects
  unimplemented ops, not explicit `.cuda()` calls, which hard-fail on a CUDA-less Mac). See §6. Caveats:
  fp32, one 2-batch CPU NaN-validation pass before the full MPS run.
- **Three whole-clip splits (train / val / test), never random frames.** `train.py` *requires* a
  populated `val` split — it builds a val dataset every run and keeps the best checkpoint by val
  accuracy (so "early-stop on val" really means "keep best-val checkpoint"). So we need train **and**
  val **and** a held-out test, not just train+test. Adjacent frames are near-duplicates (even more so at the
  shuttle pipeline's 30fps — see §2b), so every split is a whole clip/rally, never a random-frame split
  (which would leak). **Data prerequisite (hard
  gate, see §2b): this needs ≥3–4 clips across ≥2 sessions/lighting. Only ONE ~30s clip exists today, so
  the fine-tune cannot run until more are recorded + ingested.**

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
service-role). Apply via the repo's documented flow (`psql "$SUPABASE_DB_URL" -f …` / `supabase db
push`, see `supabase/README.md`) or the current `mcp__claude_ai_Supabase_BC_Rheinbach__apply_migration`
MCP — the old `Supbase_Badminton_Analytics` name no longer exists.

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
`worker/worker/shuttle/dataset_export.py`. **This is the riskiest part, not "thin I/O" — the upstream
`dataset.py` enforces a strict on-disk contract and asserts on every deviation.** Read `shuttle_labels`
(DB or local JSON) for a clip and emit exactly:
- **Masked-crop frames, not original frames.** At inference the tracker only ever sees the airspace-
  **masked**, **cropped**, 512×288 frame (`pipeline/shuttle.py:82-96` → `tracknetv3.py:108-118`). Export
  the SAME masked crops (reuse `geometry.court_airspace_polygon`/`court_crop_box` + the
  `SHUTTLE_MASK/AIR_UP/AIR_SIDE/BBOX_MARGIN` env) so train ≈ inference, else the fine-tune re-opens the
  domain gap it's meant to close. Store each rally's `crop_box` so the full-frame label click transforms
  into crop-pixel space; X,Y in the CSV live in the **exported image's** pixel space (the dataset rescales
  by `Image.open(...).size`).
- **Directory layout:** `data/{split}/match{N}/frame/{rally_id}/{Frame}.png` — PNG, integer name = CSV
  `Frame` (**not** the cache's zero-padded `.jpg`) — plus the CSV `Frame,Visibility,X,Y` at
  `match{N}/csv/{rally_id}_ball.csv` for **train + val**, but `match{N}/corrected_csv/{rally_id}_ball.csv`
  for the **test** split (`dataset.py:223-226`). `match{N}` dirs must be integer-sortable.
- **Three splits.** Populate `train/`, `val/`, AND `test/` (carve whole rallies, never random frames) —
  an empty val breaks `train.py`'s best-checkpoint selection.
- **`median.npz` for the white-wall lever.** `bg_mode=concat` reads `match{N}/median.npz` and *asserts it
  exists* (`dataset.py:487/607`) — generate it per clip from the SAME masked crops (mirror
  `dataset.py:103-107` or call upstream `get_match_median`), else the concat channel is mismatched.
- **Clear stale caches.** The dataset writes `img_config_*.npz` + `data_l*.npz` once and reuses them
  (`dataset.py:117-134`); delete `data/*.npz` before each re-export or a relabel silently trains on old data.
**TDD the pure logic** in `worker/tests/test_dataset_export.py`: match-dir int-sortability, Frame↔filename
alignment, csv-vs-corrected_csv routing, and the full-frame→crop-pixel transform. Add a one-frame
assertion that the exported image is byte-identical to what `pipeline/shuttle.py` feeds the tracker.

### 5. Ground-truth precision/recall in `worker/worker/eval_shuttle.py`
Today `stats()` is proxy-only (coverage/displacement/inside-court). Add a GT metric — the scoreboard:
- `score_against_labels(detections, labels, tol_px) -> dict` (precision, recall, F1, median pixel
  error). A detection is **TP if within `tol_px`** of the labeled shuttle on a visibility=1 frame.
  **Pin ONE pixel space: 512×288 model-input space with `tol_px=4`** (the upstream default at
  `train.py:193`, ≈ the SIGMA=2.5 label-disk radius). Map both trackers' peaks AND the labels into that
  space before scoring — full-frame clicks, 512×288 model peaks, and the parquet's frame-mapped `px/py`
  differ by ~7×, so an unpinned `tol_px` is meaningless. Note WASB uses argmax-peak vs TrackNetV3's
  contour-center (a small systematic offset). This is the fair metric — *not* court projection. **TDD
  the math** in `worker/tests/test_eval_shuttle_score.py`.
- New CLI path `--labels` (pull GT for `--video-id`) so `--compare` reports precision/recall/F1 per
  backend → the before/after table. **Report `n` (visible-frame count per split) beside every number,
  and set a minimum meaningful delta up front (e.g. recall +10pp at tol=4px, else it's noise)** so a
  noisy +2% is never read as success.

### 6. TrackNetV3 fine-tune config + local-MPS runbook
`worker/training/tracknetv3_finetune.md` (runbook) + a **source patch** to the vendored
`worker/.cache/_vendor/TrackNetV3/train.py` (the repo is git-ignored — ship the diff under
`worker/training/`). The runbook, in order:
1. **Patch CUDA → MPS (prerequisite, not optional).** Swap `.cuda()`→`.to(DEVICE)` (DEVICE = mps→cpu,
   mirror the `tracknetv3.py:83-100` pattern) at `train.py:55,86,236` and guard `torch.cuda.manual_seed`
   (`:204`) behind `torch.cuda.is_available()`; same for the `eval_tracknet` path (`test.py:339`). Scope
   to the TrackNet train+eval path only (skip InpaintNet sites `149/152`). `PYTORCH_ENABLE_MPS_FALLBACK=1`,
   fp32.
2. **Load init weights explicitly.** `--resume_training` does NOT load `TrackNet_best.pt` — it reads
   `{save_dir}/{model}_cur.pt` and its `param_dict` overrides your LR. Add an `--init_ckpt` path that does
   `model.load_state_dict(torch.load(INIT_CKPT, map_location=DEVICE)['model'])` after model creation
   (`train.py:236`) with a **fresh optimizer**, so CLI LR/epochs stand.
3. **Hyperparameters (low-data):** LR `1e-4` (try `1e-5`) vs the `1e-3` default, 5–15 epochs,
   `--alpha 0.5` (mixup — the paper's headline lever, 87.72→97.51% accuracy), keep `WBCELoss` as-is.
   Small `--batch_size` (2–4); patch `pin_memory→False` and confirm `tensorboard` is installed (no-op /
   import friction on MPS). Run the 2-batch CPU NaN-probe first.
4. **Preserve the baseline (do not overwrite).** `train.py` saves its winner to
   `{save_dir}/TrackNet_best.pt` — the SAME filename the inference backend hard-loads (`tracknetv3.py:30`,
   git-ignored, no VCS recovery). First copy the baseline to `TrackNet_baseline.pt`; save the fine-tune as
   `TrackNet_ft_<clip>.pt` and point `TRACKNETV3_MODEL` at it via env; **promote to `TrackNet_best.pt` only
   after** the scoreboard proves it beats baseline.
5. Clone vendored repo if absent, export dataset (§4), train, re-benchmark.

### 7. Spec updates (same change)
`specs/features/shuttle-tracking.md` and `specs/30-ml-pipeline.md`: mark the labeler/scoreboard built,
record the TrackNetV3-only + local-MPS + held-out-clip decisions, and add the before/after
precision/recall table (filled once measured on the Mac). Also amend `30-ml-pipeline.md`'s "cloud NVIDIA
only for fine-tuning runs" line to record the local-MPS exception, so the spec and this plan stop
disagreeing (the convention lives there, not in CLAUDE.md).

---

## Slice 2b — Hands-on, user-driven on the Mac
0. **Data gate (do this first or nothing downstream runs).** Record + ingest **≥3–4 clips across ≥2
   sessions / lighting (incl. over-exposure)**, all ingested at **`--analysis-fps 30`** — ≥2 train, 1 val,
   1 held-out test, all distinct sessions so the metric measures generalization, not memorization. Only ONE
   ~30s clip exists today.
   **Framerate decision (RESOLVED 2026-06-21 → 30 fps).** The 12fps cache can't sample a smash (1–3 frames,
   blurred), so shuttle fine-tune/eval clips ingest at **`analysis_fps=30`** (`enqueue --analysis-fps 30`;
   a clean ÷2 of the ~60fps source, matches the public badminton TrackNet distribution, makes smashes
   trackable). The player-movement pipeline stays at the default **12 fps** (its `gate_track` smoothing/gap
   params are 12fps frame-counts — see the guard-rail in `shuttle-tracking.md`). **60 fps stays a
   re-ingestable comparison option**: the ~60fps source is retained in storage, and 30fps sample times are a
   subset of 60fps, so 30fps labels remap *exactly* onto a 60fps grid (top-up the smashes, no full re-label)
   for an `eval_shuttle --compare`. f35d0984 (slug `2026-06-14-doppel`) was re-ingested at 30 fps (890
   frames) and is being re-labeled from scratch.
1. Label the rallies with the new tool (white-wall + over-exposure first), then `import-shuttle-labels`
   to Postgres. **Effort reality:** ~15–30 rallies ≈ 750–1500 labeled frames (a full ~30s clip ≈ 890 frames
   at 30fps); airborne arcs need ~1 anchor every 2–4 frames (time-based, not fps-locked). **Recommended:
   start with a 5–8 rally pilot** to prove labeler → export →
   train → scoreboard end-to-end before scaling up.
2. Export dataset (3 splits + `median.npz`, §4); fine-tune TrackNetV3 on MPS per the runbook (§6).
3. Re-benchmark **raw pixel** precision/recall (tol=4px, 512×288 space) on the **held-out test clip** via
   `eval_shuttle --compare --labels`; record the fair winner + `n` in the spec. **The metric is void /
   downgraded-to-eyeball if no second independent clip exists.** Don't fabricate metrics; report a
   non-improvement honestly.

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
  - **Pre-flight before labeling:** after patching, `grep -n "\.cuda()" worker/.cache/_vendor/TrackNetV3/train.py`
    returns nothing on the TrackNet path, and a 2-batch CPU run completes without NaNs. A 1-rally synthetic
    export loads via `Shuttlecock_Trajectory_Dataset(split='train')` with no assert errors (match-dir
    sortable, `{Frame}.png` aligns, `median.npz` present).
  - `uv run python -m worker verify --shuttle <video_id>` → click a rally, confirm
    `shuttle_labels.json`, then `import-shuttle-labels <video_id>` and verify rows via the
    Supabase MCP.
  - Export → fine-tune (MPS) → `eval_shuttle --video-id <held-out-id> --compare --labels` shows
    **fine-tuned TrackNetV3 precision/recall > frozen baseline at tol=4px (512×288), with `n` reported**.
    That measured delta is the deliverable; if it doesn't beat baseline, report it honestly.

## Branch / commits
`claude/trusting-goodall-5xxige` is already merged (PR #5) — branch fresh from `main`, e.g.
`feat/p1-shuttle-finetune`, per CLAUDE.md's git rule.
