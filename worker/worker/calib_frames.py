"""Pure + thin-I/O helpers for the multi-snapshot calibration frames (0007_calibration_frames.sql).

Ingest grabs a single still at ~5 s; when a player stands on a court corner in that one frame the
trainer can't place the handle. We instead upload N evenly-spaced snapshots so the calibration UI can
step between them (the court is static → switching the background never moves the corners). The index
math is pure (no cv2/torch/store) so it unit-tests under the dev extra; `upload_calib_frames` is the
shared I/O wrapper used by both `pipeline/ingest.py` and the `backfill-calib-frames` command.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Sequence


def evenly_spaced_indices(n_total: int, k: int) -> List[int]:
    """k evenly-spaced, strictly-increasing indices in [0, n_total).

    Includes the first and last frame when k >= 2; returns every frame when k >= n_total; the
    degenerate single-frame case (k == 1) returns the middle. Rounds half-up (not Python's banker's
    rounding) so spacing is deterministic, and deduplicates when rounding collides near k == n_total.
    """
    if n_total <= 0 or k <= 0:
        return []
    if k >= n_total:
        return list(range(n_total))
    if k == 1:
        return [(n_total - 1) // 2]
    out: List[int] = []
    for i in range(k):
        idx = int(i * (n_total - 1) / (k - 1) + 0.5)  # round half-up → monotonic, no banker's surprise
        if not out or idx > out[-1]:
            out.append(idx)
    return out


def select_calib_frames(manifest: Sequence[Mapping], k: int) -> List[dict]:
    """Pick k evenly-spaced manifest entries → [{"frame": int, "t": float}, …] in time order.

    `manifest` is frames.json's `frames` list ([{"frame","t"}, …]). The chosen `frame` is the cache
    JPEG's filename stem; `t` is its timestamp (surfaced to the switcher as the mm:ss label)."""
    return [
        {"frame": int(manifest[i]["frame"]), "t": float(manifest[i]["t"])}
        for i in evenly_spaced_indices(len(manifest), k)
    ]


def upload_calib_frames(store, bucket: str, video_id: str, frames_dir, picks: Sequence[Mapping]) -> List[dict]:
    """Upload the picked cache JPEGs to <video_id>/calib_frames/NNN.jpg; return the ordered
    `frame_uris` list [{"uri", "s"}, …] for the calibration row.

    Reads `frames_dir/{frame:06d}.jpg` exactly as ingest wrote it (`ingest.py:77`). Idempotent: the
    store uploads with upsert, and re-running overwrites the same NNN.jpg keys."""
    frames_dir = Path(frames_dir)
    frame_uris: List[dict] = []
    for n, pick in enumerate(picks):
        data = (frames_dir / f"{int(pick['frame']):06d}.jpg").read_bytes()
        uri = f"{video_id}/calib_frames/{n:03d}.jpg"
        store.upload(bucket, uri, data, "image/jpeg")
        frame_uris.append({"uri": uri, "s": round(float(pick["t"]), 3)})
    return frame_uris
