"""Export shuttle labels → a TrackNetV3 fine-tune dataset that is IDENTICAL to what inference feeds.

The vendored `dataset.py` enforces a strict on-disk contract and asserts on every deviation, and the
fine-tune only closes the white-wall domain gap if the training frames match the inference frames. So this
exporter emits the SAME airspace-masked, cropped, 512x288 frames the tracker sees at run time
(`pipeline/shuttle.py:80-96` → `tracknetv3.py:108-118`), with the label X,Y in that 512x288 pixel space.

Layout written (one clip → one match → one rally, assigned to one split):
    {out}/{split}/match{N}/frame/{rally}/{F}.png   PNG, F = 0-based contiguous frame (so 0.png exists)
    {out}/{split}/match{N}/csv/{rally}_ball.csv     Frame,Visibility,X,Y  (corrected_csv/ for the test split)
    {out}/{split}/match{N}/median.npz               key 'median', HWC (288,512,3) uint8 — the concat lever

Pure mapping (match-dir int-sortability, csv routing, 0-based Frame, full-frame→512x288) is TDD'd in
tests/test_dataset_export.py, alongside a byte-identity check that masked_model_crop equals the pipeline's
tracker input. (specs/features/shuttle-finetune-plan.md §4.)
"""
from __future__ import annotations

import csv as _csv
import json
import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from ..geometry import court_airspace_polygon, court_crop_box, frame_to_model_xy
# Reuse the pipeline's crop-geometry knobs verbatim so train == inference (no drift).
from ..pipeline.shuttle import AIR_DOWN, AIR_OUT, AIR_SIDE, AIR_UP, BBOX_MARGIN, MASK
from .label import interpolate_anchors

log = logging.getLogger("worker.shuttle.dataset_export")

MODEL_W, MODEL_H = 512, 288        # TrackNetV3 input (utils/general HEIGHT=288, WIDTH=512)
SEQ_LEN = 8                        # vendored train.py default; a rally shorter than this yields ZERO windows


# --- pure mapping helpers (TDD) -------------------------------------------------------------------

def match_name(n: int) -> str:
    """Match dir name. Must be integer-sortable: get_rally_dirs sorts by int(s.split('match')[-1])."""
    return f"match{int(n)}"


def csv_relpath(rally_id: str, split: str) -> str:
    """Per-rally label CSV path under the match dir. dataset.py:223 reads `corrected_csv/` for the test
    split (the rally path contains 'test') and `csv/` for train + val."""
    sub = "corrected_csv" if split == "test" else "csv"
    return f"{sub}/{rally_id}_ball.csv"


def rally_rows(
    dense: Sequence[Tuple], frame_w: int, frame_h: int,
    crop_box: Tuple[int, int, int, int], model_w: int = MODEL_W, model_h: int = MODEL_H,
) -> List[dict]:
    """Dense per-frame labels → CSV rows {Frame, Visibility, X, Y}.

    `dense` is interpolate_anchors output (frame_idx, px_norm, py_norm, visibility, interpolated). Frame is
    re-indexed 0-based contiguous (so `0.png` always exists). A visible label's normalized click is mapped
    full-frame → crop → 512x288 model pixels; an invisible/gap frame is written X=Y=0 (the TrackNetV3
    convention — `_get_heatmap` returns a zero target at (0,0))."""
    if not dense:
        return []
    first = int(dense[0][0])
    rows: List[dict] = []
    for frame, px, py, vis, _interp in dense:
        F = int(frame) - first
        if vis == 1 and px is not None and py is not None:
            X, Y = frame_to_model_xy(px * frame_w, py * frame_h, crop_box, model_w, model_h)
            rows.append({"Frame": F, "Visibility": 1, "X": X, "Y": Y})
        else:
            rows.append({"Frame": F, "Visibility": 0, "X": 0, "Y": 0})
    return rows


def crop_geometry(poly_px: Sequence, frame_w: int, frame_h: int):
    """Airspace polygon + crop box, identical to the pipeline (shuttle.run) so the masked crop matches —
    same AIR_UP/SIDE/DOWN/OUT knobs, so the scoring metric reflects the geometry under test."""
    air = court_airspace_polygon(poly_px, frame_w, frame_h, up=AIR_UP, side=AIR_SIDE, down=AIR_DOWN, out=AIR_OUT)
    crop_box = court_crop_box(air, frame_w, frame_h, up=0.0, side=BBOX_MARGIN, down=0.0)
    return air, crop_box


# --- crop helpers (cv2/numpy, no file I/O) --------------------------------------------------------

def build_airspace_mask(air: Sequence, crop_box: Tuple[int, int, int, int]):
    """Binary airspace mask in crop-local coords (mirrors pipeline/shuttle.py:83-85)."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = crop_box
    mask = np.zeros((y1 - y0, x1 - x0), dtype="uint8")
    local = np.array([[int(round(px - x0)), int(round(py - y0))] for px, py in air], dtype="int32")
    cv2.fillPoly(mask, [local], 255)
    return mask


def masked_model_crop(img_bgr, crop_box: Tuple[int, int, int, int], mask=None):
    """The exact array the tracker sees: crop → airspace-mask → resize(512x288) → BGR→RGB (288,512,3).

    Mirrors pipeline/shuttle.py:93-95 (crop then mask) + tracknetv3.py:113 (resize then BGR→RGB)."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = crop_box
    c = img_bgr[y0:y1, x0:x1]
    if mask is not None:
        c = cv2.bitwise_and(c, c, mask=mask)
    rgb = cv2.resize(c, (MODEL_W, MODEL_H))[:, :, ::-1]
    return np.ascontiguousarray(rgb)  # contiguous so Image.fromarray / np.savez are happy


# --- export shell (file I/O; smoke-tested on the Mac) ---------------------------------------------

def _clear_stale_split_caches(out: Path) -> None:
    """Delete the dataset's reusable input caches so a relabel never silently trains on old windows
    (dataset.py:117-134 writes img_config_*.npz + data_l*.npz once at the data root and reuses them)."""
    for p in list(out.glob("img_config_*.npz")) + list(out.glob("data_l*.npz")):
        p.unlink()
        log.info("cleared stale cache %s", p.name)


def export_clip(
    *, cache_dir, split: str, match_n: int, out_dir, poly_px: Sequence,
    rally_id: str, labels: Optional[Mapping] = None, mask: bool = MASK,
) -> dict:
    """Export one labeled clip into {out}/{split}/match{N}/ as a single rally. Returns a summary dict."""
    import cv2
    import numpy as np
    from PIL import Image

    cache = Path(cache_dir)
    manifest = json.loads((cache / "frames.json").read_text(encoding="utf-8"))
    frame_w, frame_h = int(manifest["width"]), int(manifest["height"])
    frames_meta = manifest["frames"]  # position → {"frame": int, "t": float}
    if labels is None:
        labels = json.loads((cache / "shuttle_labels.json").read_text(encoding="utf-8"))

    dense = interpolate_anchors(labels.get("anchors", []), labels.get("boundaries", []))
    if len(dense) < SEQ_LEN:
        raise ValueError(
            f"rally has {len(dense)} frames (< seq_len {SEQ_LEN}) — train.py would build ZERO windows "
            f"from it (sliding_step=1, drop_last=True). Label a longer span."
        )

    air, crop_box = crop_geometry(poly_px, frame_w, frame_h)
    x0, y0, x1, y1 = crop_box
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise RuntimeError(f"degenerate crop {crop_box} — check the calibration polygon")
    air_mask = build_airspace_mask(air, crop_box) if mask else None
    rows = rally_rows(dense, frame_w, frame_h, crop_box)

    out = Path(out_dir)
    _clear_stale_split_caches(out)
    match_dir = out / split / match_name(match_n)
    frame_dir = match_dir / "frame" / rally_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    rgb_frames = []
    for (frame_idx, *_), row in zip(dense, rows):
        if frame_idx >= len(frames_meta):
            raise IndexError(f"label frame_idx {frame_idx} past manifest ({len(frames_meta)} frames)")
        fnum = int(frames_meta[frame_idx]["frame"])
        img = cv2.imread(str(cache / "frames" / f"{fnum:06d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"missing cached frame {fnum:06d}.jpg for label frame_idx {frame_idx}")
        rgb = masked_model_crop(img, crop_box, air_mask)
        rgb_frames.append(rgb)
        Image.fromarray(rgb).save(frame_dir / f"{row['Frame']}.png")

    # median.npz per match dir: HWC (288,512,3) uint8 — what dataset.py:636 Image.fromarray consumes for
    # bg_mode=concat. Generated from the SAME masked crops (do NOT call upstream get_match_median: it reads
    # mp4s/per-rally npz we don't have).
    median = np.median(np.stack(rgb_frames), axis=0).astype("uint8")
    np.savez(match_dir / "median.npz", median=median)

    csv_path = match_dir / csv_relpath(rally_id, split)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        for r in rows:
            w.writerow([r["Frame"], r["Visibility"], round(float(r["X"]), 3), round(float(r["Y"]), 3)])

    n_vis = sum(1 for r in rows if r["Visibility"] == 1)
    summary = {
        "split": split, "match": match_name(match_n), "rally": rally_id,
        "frames": len(rows), "visible": n_vis, "crop_box": crop_box,
        "frame_dir": str(frame_dir), "csv": str(csv_path), "median": str(match_dir / "median.npz"),
    }
    log.info("exported %d frames (%d visible) → %s [crop=%s]", len(rows), n_vis, frame_dir, crop_box)
    return summary


def export_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser(prog="worker export-shuttle-dataset",
                                 description="Export shuttle_labels → a TrackNetV3 fine-tune dataset (one clip).")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--match", type=int, required=True, help="match{N} number (must be unique per split)")
    ap.add_argument("--out", default="data", help="dataset root (default: data/)")
    ap.add_argument("--rally", help="rally id (default: clip id prefix)")
    ap.add_argument("--file", help="shuttle_labels.json path (default: <frame_cache>/<video_id>/shuttle_labels.json)")
    args = ap.parse_args(argv)

    cfg = load_config()
    cache_dir = Path(cfg.frame_cache_dir).expanduser() / args.video_id
    labels = json.loads(Path(args.file).read_text(encoding="utf-8")) if args.file else None

    from ..__main__ import _make_store

    cal = _make_store(cfg).get_calibration(args.video_id)
    poly = (cal or {}).get("mask_polygon_px") or (cal or {}).get("confirmed_points")
    if not poly:
        raise SystemExit(f"no calibration polygon for {args.video_id} — calibrate must run first")

    rally_id = args.rally or args.video_id.split("-")[0]
    summary = export_clip(cache_dir=cache_dir, split=args.split, match_n=args.match,
                          out_dir=args.out, poly_px=poly, rally_id=rally_id, labels=labels)
    print(f"exported {summary['frames']} frames ({summary['visible']} visible) "
          f"→ {summary['split']}/{summary['match']}/frame/{summary['rally']}")
    print("  NOTE: train.py needs all THREE splits (train/val/test) populated; export each clip separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_cli())
