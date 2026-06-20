"""Pure shuttle-label interpolation: sparse hand-clicked anchors -> dense per-frame answer key.

The `verify --shuttle` labeler stores only ANCHORS (a click every few frames on each arc) plus SEGMENT
BOUNDARIES (the 'a' key on a contact frame). This module expands that into one row per frame — the dense
ground truth that feeds BOTH the TrackNetV3 training CSV and the precision/recall metric. Because the same
interpolation feeds the target and the metric, error compounds against the 4px tolerance, so the rules are
deliberately conservative (mirrors geometry.fill_track_gaps):

  • Linear fill between two consecutive VISIBLE anchors, but only within a segment.
  • A boundary strictly between two anchors is a contact kink — never bridge it (those frames become a
    visibility=0 gap, forcing the user to anchor the contact). A boundary sitting ON an anchor is harmless:
    the two sub-segments meet there but don't cross.
  • A visibility=0 anchor (occluded / out of frame) is a hard local override: it emits no position and
    blocks fill on both sides (you can't interpolate a position to/from an unknown point).
  • Never extrapolate: nothing before the first or after the last anchor.

No torch/cv2 — TDD'd in tests/test_shuttle_label.py. (specs/features/shuttle-finetune-plan.md §2.)
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Set, Tuple

# (frame, px, py, visibility, interpolated) — px/py normalized 0..1, None when invisible;
# interpolated=False marks a hand-clicked anchor, True an interpolation- or gap-filled frame.
DenseRow = Tuple[int, Optional[float], Optional[float], int, bool]


def interpolate_anchors(
    anchors: Iterable[Mapping], boundaries: Iterable[int]
) -> List[DenseRow]:
    """Expand sparse anchors + segment boundaries to a dense per-frame track over the anchor span.

    `anchors`: items with keys frame_idx, px, py, visibility (visibility=1 carry px,py; visibility=0 mark
    occlusion). `boundaries`: frame indices where a contact/segment break was marked. Returns one
    DenseRow per integer frame in [first_anchor, last_anchor]."""
    items = sorted(anchors, key=lambda a: a["frame_idx"])
    if not items:
        return []

    bounds: Set[int] = {int(b) for b in boundaries}
    out: List[DenseRow] = []

    for cur, nxt in zip(items, items[1:]):
        fa = int(cur["frame_idx"])
        fb = int(nxt["frame_idx"])
        # the anchor itself
        out.append(_anchor_row(cur))
        # interior frames fa+1 .. fb-1
        bridgeable = (
            cur.get("visibility", 1) == 1
            and nxt.get("visibility", 1) == 1
            and not any(fa < b < fb for b in bounds)  # a boundary ON an anchor (==fa/==fb) does not block
        )
        for f in range(fa + 1, fb):
            if bridgeable:
                frac = (f - fa) / (fb - fa)
                px = cur["px"] + (nxt["px"] - cur["px"]) * frac
                py = cur["py"] + (nxt["py"] - cur["py"]) * frac
                out.append((f, px, py, 1, True))
            else:
                out.append((f, None, None, 0, True))  # gap — visibility=0, no fabricated position

    out.append(_anchor_row(items[-1]))
    return out


def _anchor_row(a: Mapping) -> DenseRow:
    """One DenseRow for a hand-clicked anchor (interpolated=False). A visibility=0 anchor carries no
    position regardless of any stray px/py."""
    if a.get("visibility", 1) == 0:
        return (int(a["frame_idx"]), None, None, 0, False)
    return (int(a["frame_idx"]), a["px"], a["py"], 1, False)
