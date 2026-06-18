"""Stage: calibrate (finalize) — confirmed corners → image-px→(u,v) homography + court mask.

Runs in the resume job after the trainer confirms the 4 corners in the web UI. Reuses the pure
geometry helpers (same DLT as the frontend) so worker and web agree. With exactly 4 corners the
reprojection residual is ~0 by construction — it is a degeneracy self-check, not an accuracy signal
(specs/30-ml-pipeline.md stage 1). Acceptance is metric-span sanity + the trainer's visual confirm.
"""
from __future__ import annotations

import logging

from ..geometry import (
    COURT_L_M,
    COURT_W_M,
    UNIT_SQUARE,
    apply_homography,
    homography_from_corners,
    reproject_error_m,
)

log = logging.getLogger("worker.pipeline.calibrate")


def run(ctx) -> None:
    store = ctx["store"]
    job = ctx["job"]
    video_id = job["video_id"]

    cal = store.get_calibration(video_id)
    if not cal or not cal.get("confirmed_points"):
        raise RuntimeError("calibrate: no confirmed_points — awaiting trainer corner placement")

    corners = [(float(x), float(y)) for x, y in cal["confirmed_points"]]  # px, FL,FR,NR,NL
    if len(corners) != 4:
        raise RuntimeError(f"calibrate: expected 4 confirmed corners, got {len(corners)}")

    H = homography_from_corners(corners)
    err = reproject_error_m(H, corners, UNIT_SQUARE)

    # Metric-span sanity: the unit-square dst means span IS 6.10 x 13.40 m by construction; we still
    # log the projected interior to catch a wildly degenerate quad the corner UI let through.
    cu, cv = apply_homography(H, *((sum(c[0] for c in corners) / 4, sum(c[1] for c in corners) / 4)))
    log.info(
        "calibrate: reproj_err=%.4fm, centroid->(u=%.2f,v=%.2f), court=%.2fx%.2fm",
        err, cu, cv, COURT_W_M, COURT_L_M,
    )

    store.update_calibration(
        cal["id"],
        {
            "homography": H,
            "mask_polygon_px": [list(c) for c in corners],
            "reprojection_error_m": round(err, 4),
            "status": "finalized",
        },
    )
    ctx["H"] = H
    ctx["mask_poly_px"] = corners
