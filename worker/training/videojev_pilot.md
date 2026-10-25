# VideoJev-4B zero-shot stroke pilot — local runbook (Mac Studio M4 Max, 64 GB)

Spec: `specs/features/stroke-vlm-pilot.md`. All commands run from `worker/`. Unlike the BST
training, nothing is trained here: the model scores the labeled 1.4 s clips zero-shot, once on the
dev game to pick the prompt wording, once on the other games for the number of record. Every number
you copy into the spec carries its command, the model revision and the preset.

## 0. Prerequisites

```bash
uv lock && uv sync --extra ml --extra vlm --extra dev     # vlm = transformers==5.14.1 + friends; ml untouched
```
If `uv lock` cannot reconcile `transformers==5.14.1` with the `ml` extra (ultralytics/rtmlib), add
`[tool.uv] conflicts = [[{extra = "ml"}, {extra = "vlm"}]]` to `pyproject.toml` and sync the two
extras into separate envs — the pilot needs `ml` only for `av`, `numpy`, `polars`, `torch`.

```bash
uv run hf download ZhuYuhan/VideoJev-4B --local-dir ../models/weights/stroke_vlm/VideoJev-4B
```
Then, before anything runs:
1. **Pin the revision.** `hf download` prints the snapshot SHA (or read
   `../models/weights/stroke_vlm/VideoJev-4B/.cache/huggingface/` / the model page). Put it into
   `VIDEOJEV_REVISION` in `worker/stroke/vlm_eval.py` and into the results table below.
2. **Check the licence tag** on the model card (the GitHub repo is Apache-2.0; non-distributed use is
   unconstrained either way, CLAUDE.md).
3. **Read the remote code once** — `videojev_hf/{modeling,processing}_videojev.py` in the snapshot —
   it runs with `trust_remote_code=True`. Confirm `predict_features` still returns
   `{"probabilities", "answer_index", ...}` and that nothing imports flash-attn unconditionally.
4. Reference clone (read-only, never imported):
   ```bash
   git clone https://github.com/MCG-NJU/VideoJev .cache/_vendor/VideoJev && git -C .cache/_vendor/VideoJev log -1 --format=%H
   ```
5. `sysctl hw.memsize` — fp32 needs ≈ 18 GB weights + activations; the 64 GB box is fine. If an MPS
   allocator cap bites, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`; if memory is short, `--dtype bf16`
   (recorded in the report, probed against CPU fp32).

Inputs that must already exist (they do for Session 2): `worker/.labeling/stroke_labels.parquet`
(695 rows), `worker/.labeling/stroke/<game>/*.mp4` + `tasks_<game>.json` from `export-stroke-clips`.
The dev game is **G5** — its `match_videos.id` is `<G5>` below.

## 1. Tests

```bash
uv run pytest tests/test_stroke_vlm.py tests/test_stroke_train.py          # torch-free, always
STROKE_VLM_SMOKE=1 uv run pytest tests/test_stroke_vlm_smoke.py             # loads the real model on CPU (~16 GB)
```

## 2. CPU plumbing smoke (CLAUDE.md NaN probe) — fail fast before any sweep

```bash
uv run python -m worker eval-stroke-vlm --split dev --dev-video-id <G5> --preset plain_fps2 --device cpu --probe 0 --limit 2
```
Finite probabilities and a written `report.json` prove the plumbing; the numbers are meaningless at
n=2. Minutes per clip on CPU is expected.

## 3. Dev sweep — choose the preset here and only here

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker eval-stroke-vlm --split dev --dev-video-id <G5> --preset all --probe 5
```
Starts with the probe (5 clips, CPU fp32 vs MPS: abort unless finite, max |Δp| ≤ 0.05, all argmax
agree). Then 69 clips × 3 encodes (fps 2/10/30) × ≤ 4 scorings; one visual encoding per clip and fps
is shared by all prompts at that fps. Expect 1–2 h on MPS fp32 (30 fps dominates). The report's
`selected` block names the preset for step 4; `position_bias` compares `plain_fps10` with its
reversed-options twin. **The dev accuracy never counts.** `--resume <run dir>` continues an
interrupted sweep.

## 4. Frozen eval — the number of record

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m worker eval-stroke-vlm --split eval --dev-video-id <G5> --frozen-from .labeling/stroke_runs/<dev run dir>
```
Preset, fps, dtype, context and attention come from the dev report; `--preset/--dtype/--attn/--context`
are refused. The 4 other games (~626 hits, ~500 confident). Expect 0.5–2 h depending on the selected fps.
The report's `headline` carries pooled accuracy, per-game mean ± std, top-2, the decision tier and the
gate flag; `per_preset.<name>` has the full block (per-class recall, confusion, hard pairs, τ sweep,
Unsicher abstention, unknown-rate per match, confidence medians, prediction share).

## 5. Record

Copy the headline + per-game numbers + tier into the tables here and in the spec, **with the command**,
the model revision and the preset. Then apply the spec's decision rule — `closed` is a legitimate and
expected outcome; it still gets written down.

| date | split | revision | preset | n confident | pooled acc | per-game | top-2 | τ* (cov.) | Unsicher abst. | tier | command |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | dev | — | — | — | — | — | — | — | — | — | *not run yet* |
| — | eval | — | — | — | — | — | — | — | — | — | *not run yet* |

## 6. Caveats (report these with every number)

- **One session, one lighting, all doubles**: cross-game, not cross-session generalization.
- **Zero-shot, pixels only**: the labelers also heard the clip; the model cannot.
- **MPS is not bit-deterministic**; the CPU-fp32 probe is the reproducibility reference, and
  `--device cpu` reproduces any number at minutes per clip.
- **Dev selection on 69 hits is noisy** — it picks wording, it does not produce a number.
- **Seconds per clip** is not a pipeline budget; feasibility is out of scope.

## 7. If something breaks

- *Probe FAILED (MPS vs CPU disagree / NaN)*: Qwen3.5's hybrid layers have no MPS coverage upstream.
  Try `--attn eager`, then `--dtype bf16`; if still failing, run the pilot on `--device cpu` (slow but
  honest) and note it. Never `--allow-probe-fail` for the number of record.
- *`transformers` lock conflict*: see §0.
- *Missing clips*: `n_missing_clips > 0` blocks the run — re-run `export-stroke-clips` for that game
  (labels join by `event_id`, then same-video `t_s` ± 0.05 s, then the filename's 8-char id).
- *Remote code changed*: `--revision <SHA>` pins the snapshot; a different SHA is a different experiment.
