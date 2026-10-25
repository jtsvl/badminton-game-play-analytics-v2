# Feature — Stroke VLM pilot: VideoJev-4B zero-shot on the labeled clips (stage-7 side experiment)

**Status: BUILT, NOT RUN (2026-10-18).** Code, tests and protocol are in-tree; the model run needs
the Mac Studio (weights, clips and labels live only there) — runbook `worker/training/videojev_pilot.md`.
The results table below stays empty until then. **Nothing here wires anything into the pipeline**;
below-gate models are labeling-assist only, never pipeline (stage-7 policy, `30-ml-pipeline.md`).

## Why this exists
- Stage 7's BST4 (`features/stroke-classification.md`) failed its gate: LOGO-CV **0.400** vs 0.70, with
  train accuracy 0.945 → it memorizes 385 rows and does not transfer across games. The named levers were
  more data and better input signal; a third lever is a model that arrives with a **pretrained video
  representation** instead of learning everything from 554 clips.
- Origin: the question whether **TypeSafe's Jev** could beat our models. Verdict: no — it is a hosted,
  text-only decision API with no video modality (`stroke-classification.md` § Alternatives evaluated).
  The video-capable model that borrowed the "System One" name is **VideoJev** (Nanjing University), and
  the only honest way to know whether it carries signal on our footage is to measure it zero-shot.
- Cost: no labels, no training, one afternoon of Mac time. The clips (`stroke_clips.py`), the labels and
  the metric helpers (`stroke/train.py`) already exist.

## The model (facts verified from the GitHub source, 2026-10-18)
- **VideoJev-4B** — MCG-NJU, Apache-2.0, arXiv 2610.18615, weights `ZhuYuhan/VideoJev-4B` on Hugging
  Face (snapshot SHA to be pinned at first download → `VIDEOJEV_REVISION` + runbook). **Not a TypeSafe
  product.**
- A Qwen3.5-4B multimodal LLM whose output head is replaced by a candidate scorer: for a multiple-choice
  question it returns a **softmax over the text options** — the shape FR-22's τ machinery already consumes.
  One visual encoding per clip can be scored against several questions.
- Sampling is an exact `k/fps` grid at source resolution with a free `fps`: a 1.4 s clip yields
  **3 frames at 2 fps (upstream default), 14 at 10 fps, 42 at 30 fps**. Per-frame pixel cap ≈ 640²
  (720p proxy → ≈ 832×480 → ≈ 8.2k visual tokens at 30 fps). **Audio is ignored.**
- Why the 4B and not the 35B-A3B: the mixture-of-experts variant has ~3B *active* parameters but all
  35B must be resident — ≈ 70 GB bf16, which does not fit the **64 GB** Mac Studio; quantized paths on
  MPS would drop the custom scorer head or need MLX (ruled out). Its reported gains over the 4B are
  ≈ 5 points on general video-QA averages. The CLI takes `--model`, so the 35B can run the identical
  frozen protocol on a rented ≥ 80 GB NVIDIA box if the 4B result makes that worth buying.
- Upstream never ran this on fine-grained sports actions; its benchmarks are long-video QA and temporal
  grounding. **Expectation: somewhere between chance (1/7) and BST (0.40).** This pilot replaces the guess.

## Protocol (locked by this spec)
1. **Data:** the 695 labeled Session-2 hits (554 confident over the coarse 7, 141 Unsicher) as the
   existing ±0.7 s / 1.4 s proxy clips; labels join clips by `event_id` (fallbacks: same-video `t_s`
   within 0.05 s, then the clip filename's 8-char id; anything ambiguous is reported, never guessed).
   A missing clip blocks the run unless `--allow-missing-clips`.
2. **Zero-shot, pixels only.** No label, serve hint or rally index ever reaches the model. One question
   per pass (several questions in one call would share a causal sequence).
3. **Presets are the only knobs**, frozen in `stroke/vlm_eval.py`:

   | prompt | query | options (7, `CLASSES` order unless noted) |
   |---|---|---|
   | `plain` | "Which badminton stroke is hit at the center of this clip?" | `clear, smash, drop, net shot, drive, lift, serve` |
   | `described` | same | each option + one defining clause |
   | `context` | "Doubles badminton, elevated end-on camera. Exactly one stroke is hit in the middle of the clip. Which stroke type is it?" | as `described` |
   | `plain_rev` (fps 10 only) | as `plain` | **reversed order** — position-bias probe, dev only, never selectable |

   × fps ∈ {2, 10, 30} → 10 presets.
4. **Dev/eval split (anti prompt-fishing).** Dev = **G5** (69 hits, the smallest game): every preset runs,
   the one with the highest pooled dev accuracy is `selected` (ties → table order). Eval = the other 4
   games: the CLI **reads preset/fps/dtype/context/attn from the dev report** and refuses `--preset`,
   `--dtype`, `--attn`, `--context`. **Only the eval number counts** (dev: ~50 confident rows, 95 % CI
   ≈ ±0.14; eval: ~500 rows, CI ≈ ±0.045).
5. **Metrics** reuse `stroke/train.py`: pooled + per-game accuracy (mean ± std over games), per-class
   recall (NaN where unsupported, never averaged away), confusion + hard pairs (drop/net, clear/lift),
   FR-22 τ sweep on confident rows, Unsicher abstention at τ*, unknown-rate per match, top-2 accuracy,
   confidence medians (confident vs Unsicher), prediction share (collapse check), position-bias delta
   (`plain` vs `plain_rev`).
6. **Device honesty (CLAUDE.md):** fp32 default on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`; before any
   sweep, 5 evenly spread clips are scored on **CPU fp32 and on the device** — the run aborts unless
   every probability is finite, max |Δp| ≤ 0.05 and every argmax agrees (`--allow-probe-fail` records
   a failed probe instead). `--dtype bf16` is the recorded memory fallback.
7. **Outputs** per run dir `<labeling>/stroke_runs/<UTC>_videojev_zeroshot_<split>/`:
   `predictions.jsonl` (incremental, `--resume`-able), `predictions.parquet` (one row per event ×
   preset: label, preset, clip, join method, frames sampled, `prob_<class>`, `pred`, `p_max`, wall time),
   `report.json` (`per_preset` blocks in `run_logo_cv`'s shape, `selected` or `settings`, `probe`,
   model provenance incl. revision, labels sha256, timing, command, git SHA, caveats).

## Decision rule (on the eval split's pooled accuracy over confident rows; valid only with 0 missing clips and a passed probe)

| pooled eval accuracy | tier | what it means |
|---|---|---|
| < 0.35 | `closed` | indistinguishable from majority-class 0.30, below BST 0.40: no usable zero-shot signal on this footage. **No fine-tune** (554 one-session clips would meet the same cross-game gap at ~100× BST's compute). Record and move on. |
| 0.35 – 0.55, or top-2 ≥ 0.75 | `labeling_assist` | candidate for a Label-Studio *suggestion* (never pre-fill) for Session 3; optionally buy the 35B-A3B cloud row. No fine-tune yet. |
| ≥ 0.55 **and** recall > 0 for ≥ 6/7 classes **and** median confidence on Unsicher < on confident rows | `finetune_candidate` | clearly above BST with margin and an honest confidence signal → a separate LoRA/head fine-tune spec (LOGO protocol); user decision first. |
| ≥ 0.70 | `gate` | clears the locked gate zero-shot → next is a runtime-feasibility study (738 hits/session at seconds per clip), still no wiring from this pilot. |

Baselines carried in every report: chance 1/7, majority 165/554 ≈ 0.30, BST LOGO mean 0.400, gate 0.70.

## Results (no metric enters without the command that produced it)

| date | split | preset | n confident | pooled acc | per-game | top-2 | τ* (cov.) | Unsicher abst. | tier | command |
|---|---|---|---|---|---|---|---|---|---|---|
| — | dev | — | — | — | — | — | — | — | — | *not run yet* |
| — | eval | — | — | — | — | — | — | — | — | *not run yet* |

## Honest limits
- **One session, one lighting, all doubles**: cross-game, not cross-session generalization.
- **The model never hears the clip**; the human labelers used the sound (smash crack vs net tap). Any
  gap to human labels partly measures a missing modality.
- **Fine-grained 7-way strokes from an elevated end-on camera** are a harder target than VideoJev's
  benchmarks; a `closed` result is the expected outcome, not a failure of the pilot.
- **MPS is untested upstream** for Qwen3.5's hybrid layers — hence the CPU probe; `--device cpu` is
  the always-works escape at minutes per clip.
- **Seconds per clip** is far from pipeline budgets; runtime feasibility is a non-goal here.
- **Dev selection on 69 hits is noisy**; it only picks the wording, it never produces a number.

## Non-goals
No pipeline stage, no `_REAL` registration, no fine-tune, no 35B run on the Mac, no change to the
stage-7 taxonomy or gates. A `finetune_candidate` or `gate` result opens a *new* spec, nothing more.

## Build (what exists)
- `worker/worker/stroke/vlm_eval.py` — presets, label↔clip join, dev/eval split, selection, frozen
  settings, decision tier, report aggregation, run planning, probe comparison, `eval_cli`.
- `worker/worker/stroke/vlm_model.py` — lazy torch/transformers: `load_videojev`, `encode_clip`,
  `predict_from_features`, `score_clip`, `run_probe`.
- `worker/worker/__main__.py` — `python -m worker eval-stroke-vlm`.
- `worker/worker/stroke/train.py` — `hard_pair_block` lifted to module level (shared report format).
- `worker/pyproject.toml` — extra `vlm` (`transformers==5.14.1`, …; `ml` untouched).
- Tests: `tests/test_stroke_vlm.py` (pure helpers + CLI against a fake model, torch-free),
  `tests/test_stroke_vlm_smoke.py` (opt-in, real model, `STROKE_VLM_SMOKE=1`).
- Env: `STROKE_VLM_MODEL` (snapshot dir; default `models/weights/stroke_vlm/VideoJev-4B`, git-ignored).
