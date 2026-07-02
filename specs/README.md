# Specs — source of truth

Read these before writing code. Keep them in sync with the implementation (update the spec in the
same change as the behavior). Functional requirements are referenced by ID (e.g. `FR-5`) from code,
the data model, and the design handoffs.

| File | Scope |
|---|---|
| [`00-product-spec.md`](./00-product-spec.md) | Vision, personas, use cases, scope, success criteria |
| [`10-functional-requirements.md`](./10-functional-requirements.md) | Numbered functional requirements (FR-*) |
| [`20-data-model.md`](./20-data-model.md) | Postgres schema + JSON shapes, mapped to the frontend |
| [`30-ml-pipeline.md`](./30-ml-pipeline.md) | Pipeline stages, model choices, datasets, fine-tuning, acceptance thresholds |
| [`40-worker-contract.md`](./40-worker-contract.md) | Job lifecycle, stage I/O, result-row contracts, Storage layout |
| [`50-web-app.md`](./50-web-app.md) | Routes, layout branch, component port map, brand tokens |
| [`features/`](./features) | One spec per implementation slice |

Phasing (build order): **P0** foundations · **P1** calibration + movement · **P2** shuttle + events ·
**P3** pose + strokes · **P4** polish/automation. Full plan: `~/.claude/plans/help-me-build-an-atomic-lagoon.md`.
Target end-state for the review workflow: [`features/graduated-autonomy.md`](./features/graduated-autonomy.md)
— review gates become confidence-gated auto-accepts; player naming stays the one human step.
