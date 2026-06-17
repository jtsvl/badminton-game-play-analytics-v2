# CV/ML Worker

Long-running Python process on the **Mac Studio (M4 Max)**. Polls `processing_jobs` on the Coolify
Supabase, downloads the clip from Storage, runs the staged pipeline (specs/30-ml-pipeline.md), and writes
results back. Offline/batch — no real-time SLA. PyTorch **MPS** primary; no MLX. See `specs/40-worker-contract.md`.

## Layout
```
worker/
  __main__.py      entry point — the poll/claim/process loop
  config.py        env-driven config
  db.py            JobStore — claims jobs (FOR UPDATE SKIP LOCKED), updates status
  pipeline/        stage registry (stubs in P0; real models in P1–P3)
  insights.py      PURE insight math (distance/speed/sprints/zones/timeline/heatmap) — no deps, tested
tests/             unittest/pytest tests (run with no ML deps)
```

## Setup
```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'        # core + test deps
pip install -e '.[ml]'         # heavy CV/ML stack (on the Mac Studio)
```

## Run
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m worker
```
Without DB env configured, the worker logs a warning and idles.

## Test
```bash
python -m unittest discover -s tests       # stdlib, no install
# or
pytest
```

## Status
P0: structure + job loop + tested pure insights math. Stage models (detect/track/pose/shuttle/stroke)
are stubs to be filled in P1–P3 per `specs/30-ml-pipeline.md` (benchmark on the club's own footage).
