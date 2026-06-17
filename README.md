# Badminton Game-Play Analytics (v2)

Self-hostable web app that turns badminton match recordings into per-player game insights
(court heatmaps, zone coverage, distance/speed, activity timeline, shot mix, rally stats,
doubles formation) plus a video player with analysis overlays. Built for the club **BC Rheinbach**
(German UI, du-form). Footage: iPhone / DJI Osmo Action 6 on a tripod (~3–4 m), school sports halls
with multiple adjacent courts and overlapping multi-sport lines.

> Full build plan: see the approved plan at `~/.claude/plans/help-me-build-an-atomic-lagoon.md`.
> Specs are the source of truth — start in [`/specs`](./specs).

## Architecture

```
Web app (React/TS, Vite)  ──upload──▶  Supabase (self-hosted on Coolify)  ◀──poll/claim──  CV/ML worker
   renders the two designs              Postgres + Auth + Storage + Realtime               (Python · PyTorch MPS,
   from structured stats   ◀──read──    stats · events · position timeseries ──write──▶     Mac Studio M4 Max)
```

The CV/ML worker runs **locally on the Mac Studio** (privacy + zero marginal cost; offline batch,
no real-time SLA). It claims jobs from `processing_jobs`, downloads the clip from Storage, runs the
staged pipeline (decode → court calibration → player detect → court-gating → track → pose → shuttle →
rally/hit → stroke → aggregate), and writes results back to Postgres.

## Monorepo layout

| Path | What |
|---|---|
| [`/specs`](./specs) | Product + functional spec, data model, ML pipeline, worker contract, web-app spec. **Read first.** |
| [`/web`](./web) | Vite + React 18 + TypeScript app (ports the `handovers/` designs). |
| [`/worker`](./worker) | Python CV/ML pipeline (PyTorch MPS on Apple Silicon) + tests + training configs. |
| [`/supabase`](./supabase) | Versioned SQL migrations + seed for the Coolify Postgres. |
| [`/models`](./models) | Model weights (git-ignored), fine-tune configs, dataset manifests. |
| [`/infra`](./infra) | Coolify / docker-compose notes, deployment. |
| [`/handovers`](./handovers) | Original Claude Design handoffs (pixel-final reference — do not edit). |

## Getting started

1. `cp .env.example .env` and fill in the Coolify Supabase URL + keys.
2. Apply the schema: `supabase/migrations/*.sql` (see [`/supabase/README.md`](./supabase)).
3. Web: `cd web && npm install && npm run dev`.
4. Worker: `cd worker && uv sync` (or `pip install -e .`), then `python -m worker`.

## Key constraints

- **German UI, du-form**, BC Rheinbach brand tokens (orange `#FA8832`, beige `#FEF4EA`, Space Grotesk + Inter).
- **Apple Silicon**: PyTorch MPS primary; CoreML for hot models; **not** MLX. Process locally.
- **Non-distributed / non-commercial** → models chosen by quality (see `/specs/30-ml-pipeline.md`).
- Honest about ML limits: monocular shuttle speed/3D are **estimates**, surfaced with uncertainty.
