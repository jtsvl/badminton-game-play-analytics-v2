"""Shuttle tracking (pipeline stage 5).

The tracker backends (WASB, TrackNetV3, TrackNetV5) live behind the `ShuttleTracker` protocol in `base.py`
and are loaded lazily so the pure track assembly (`track.py`) and its tests import without torch/cv2. Pick
a backend with the `SHUTTLE_TRACKER` env (default `tracknetv3` — the fine-tuned own-footage winner,
F1 0.483→0.953; wasb stays available as the frozen baseline; tracknetv5 is under A/B — see
specs/features/shuttle-tracking.md and specs/features/shuttle-tracknetv5.md).
"""
from __future__ import annotations

import os

TRACKERS = ("wasb", "tracknetv3", "tracknetv5")


def get_tracker(name: str | None = None, **kwargs):
    """Construct a ShuttleTracker backend by name (env `SHUTTLE_TRACKER` when omitted). Backends are
    imported lazily so heavy deps load only when a real run needs them."""
    name = (name or os.environ.get("SHUTTLE_TRACKER", "tracknetv3")).lower()
    if name == "fake":
        from .base import FakeTracker

        return FakeTracker()
    if name == "wasb":
        from .wasb import WASBTracker

        return WASBTracker(**kwargs)
    if name == "tracknetv3":
        from .tracknetv3 import TrackNetV3Tracker

        return TrackNetV3Tracker(**kwargs)
    if name == "tracknetv5":
        from .tracknetv5 import TrackNetV5Tracker

        return TrackNetV5Tracker(**kwargs)
    raise ValueError(f"unknown SHUTTLE_TRACKER {name!r} (expected one of {TRACKERS})")
