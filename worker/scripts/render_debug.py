"""Render a few annotated frames to PNGs under /tmp (the static counterpart of `worker verify`).

Usage (from the worker/ dir):  uv run python scripts/render_debug.py [video_id] [frameIdx ...]
For an interactive scrubber instead, use:  uv run python -m worker verify [video_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

from worker.config import load_config
from worker.verify import annotate, load_data


def main() -> None:
    args = sys.argv[1:]
    video_id = next((a for a in args if not a.isdigit()), "f35d0984-207c-4ee1-8283-ef8ba58eca82")
    nums = [int(a) for a in args if a.isdigit()]
    cfg = load_config()
    data = load_data(video_id, cfg)
    idxs = nums or [len(data.frames) // 5, len(data.frames) // 2, 4 * len(data.frames) // 5]
    for idx in idxs:
        out = Path("/tmp") / f"validate-frame-{idx:03d}.png"
        annotate(data, idx).save(out)
        print(f"wrote {out}  (t={data.frames[idx]['t']:.1f}s)")


if __name__ == "__main__":
    main()
