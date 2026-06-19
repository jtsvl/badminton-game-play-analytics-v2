"""Pipeline stage registry. See specs/30-ml-pipeline.md and specs/40-worker-contract.md."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List

log = logging.getLogger("worker.pipeline")

# Ordered stage names (40-worker-contract.md). Side-angle videos run a reduced subset.
# pose runs BEFORE gate_track: it derives the foot from the real ankle keypoint, which gate_track then
# projects through the homography instead of the bbox bottom (specs/30-ml-pipeline.md stage 4).
STAGE_ORDER: List[str] = [
    "ingest", "calibrate", "detect", "pose", "gate_track",
    "shuttle", "rally_hit", "stroke", "aggregate", "overlay",
]
SIDE_ANGLE_STAGES: List[str] = ["ingest", "calibrate", "detect", "pose", "shuttle", "rally_hit", "stroke", "overlay"]


class StageContext(dict):
    """Carries per-job artifacts/handles between stages (frame cache dir, db, config, ...)."""


def _todo(name: str) -> Callable[[StageContext], None]:
    def stage(ctx: StageContext) -> None:
        # rally_hit, stroke still stubbed for P2/P3; shuttle (stage 5) and pose are real.
        log.info("stage %s: not yet implemented (stub)", name)
    stage.__name__ = name
    return stage


def _lazy(name: str) -> Callable[[StageContext], None]:
    """Wrap a real stage module so heavy deps (torch/av/cv2) import only when the stage runs."""
    def stage(ctx: StageContext) -> None:
        from importlib import import_module

        import_module(f".{name}", __package__).run(ctx)
    stage.__name__ = name
    return stage


# Real stages. pose is the Phase-3 durable foot fix (RTMPose); shuttle is stage 5 (WASB/TrackNetV3
# behind shuttle.get_tracker). rally_hit/stroke stay stubbed for P2/P3. A name not listed here runs the
# no-op _todo stub. shuttle joins RESUME_STAGES (__main__) only once verified on real footage.
_REAL = ("ingest", "calibrate", "detect", "pose", "gate_track", "shuttle", "aggregate", "overlay")
STAGES: Dict[str, Callable[[StageContext], None]] = {name: _todo(name) for name in STAGE_ORDER}
STAGES.update({name: _lazy(name) for name in _REAL})


def run_pipeline(ctx: StageContext, stages: List[str], on_stage=None) -> None:
    """Run the given stages in order, reporting progress via on_stage(name, fraction)."""
    total = len(stages)
    for i, name in enumerate(stages):
        if on_stage:
            on_stage(name, i / total)
        STAGES[name](ctx)
    if on_stage:
        on_stage("done", 1.0)
