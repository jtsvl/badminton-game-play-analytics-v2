"""Pipeline stage registry. See specs/30-ml-pipeline.md and specs/40-worker-contract.md."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List

log = logging.getLogger("worker.pipeline")

# Ordered stage names (40-worker-contract.md). Side-angle videos run a reduced subset.
STAGE_ORDER: List[str] = [
    "ingest", "calibrate", "detect", "gate_track", "pose",
    "shuttle", "rally_hit", "stroke", "aggregate", "overlay",
]
SIDE_ANGLE_STAGES: List[str] = ["ingest", "calibrate", "detect", "pose", "shuttle", "rally_hit", "stroke", "overlay"]


class StageContext(dict):
    """Carries per-job artifacts/handles between stages (frame cache dir, db, config, ...)."""


def _todo(name: str) -> Callable[[StageContext], None]:
    def stage(ctx: StageContext) -> None:
        # P0: real implementation lands in P1-P3 (model wiring per specs/30-ml-pipeline.md).
        log.info("stage %s: not yet implemented (P0 stub)", name)
    stage.__name__ = name
    return stage


# In P1-P3 each entry is replaced by a real implementation module.
STAGES: Dict[str, Callable[[StageContext], None]] = {name: _todo(name) for name in STAGE_ORDER}


def run_pipeline(ctx: StageContext, stages: List[str], on_stage=None) -> None:
    """Run the given stages in order, reporting progress via on_stage(name, fraction)."""
    total = len(stages)
    for i, name in enumerate(stages):
        if on_stage:
            on_stage(name, i / total)
        STAGES[name](ctx)
    if on_stage:
        on_stage("done", 1.0)
