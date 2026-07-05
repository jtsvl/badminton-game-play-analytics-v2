"""Stage-7 dataset exporter — labeled strokes → BST training samples (features/stroke-classification.md).

Joins `stroke_labels.parquet` to the per-video cache artifacts via **video_id + t_s** (never
`event_id` — event uuids regenerate on every stage-6 re-run) and materializes one window sample
per labeled hit through the same `features.build_sample` the pipeline stage will call at
inference. Slots fill from time-aware role tracks plus pose recovery through the calibration
homography (fetched per video from the DB; recovery is skipped loudly without it). Emits
`stroke_dataset.npz` (stacked arrays) + `meta.parquet` (labels, fold key = video_id, slot fields)
and prints the Phase-1 gate report: is the labeled hitter's slot filled, and from which source.
The 513 hitter labels only MEASURE coverage here — they never feed the features.

    python -m worker export-stroke-dataset [--labels P] [--out DIR] [--cache DIR]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .features import SLOTS, build_sample, slot_of_role

log = logging.getLogger("worker.stroke.dataset")

FPS_EXPECTED = 12.0  # the ingest analysis cadence (ingest.DEFAULT_ANALYSIS_FPS)
FPS_TOLERANCE = 0.2  # warn when median frame dt deviates more than this fraction from 1/12


def slot_report(rows: Sequence[dict]) -> dict:
    """The Phase-1 gate numbers: over hitter-labeled rows, how often the labeled hitter's slot
    carries a player (split by source: court track vs pose recovery), plus per-slot fill rates
    over all rows. Hitter tokens are exact slot names, so the lookup is direct."""
    labeled = [r for r in rows if r.get("hitter")]
    filled = [r for r in labeled if r.get(f"track_{slot_of_role(r['hitter'])}") is not None]
    by_source: dict = {}
    for r in filled:
        src = r.get(f"source_{slot_of_role(r['hitter'])}")
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "n_hitter_labeled": len(labeled),
        "hitter_slot_filled": len(filled),
        "hitter_slot_filled_rate": len(filled) / len(labeled) if labeled else None,
        "hitter_slot_by_source": by_source,
        "slot_fill_rates": {
            slot: (sum(1 for r in rows if r.get(f"track_{slot}") is not None) / len(rows)
                   if rows else None)
            for slot in SLOTS
        },
    }


def _fps_guard(frames: List[dict], video_id: str) -> None:
    dts = sorted(frames[i + 1]["t"] - frames[i]["t"] for i in range(len(frames) - 1))
    if not dts:
        return
    median_dt = dts[len(dts) // 2]
    if median_dt > 0 and abs(median_dt - 1 / FPS_EXPECTED) / (1 / FPS_EXPECTED) > FPS_TOLERANCE:
        log.warning("stroke dataset: video %s median frame dt %.4fs deviates >%d%% from the "
                    "12 fps analysis cadence — window time-math is off (stage-7 assumption)",
                    video_id, median_dt, int(FPS_TOLERANCE * 100))


def build_video_samples(cache: Path, labels: Sequence[dict],
                        homography=None) -> Tuple[List[dict], List[dict]]:
    """All samples for one video from its cache artifacts. Missing artifacts degrade per
    `build_sample` (zeros / None meta); a missing frames.json skips the video loudly —
    there is no time base to anchor on."""
    import polars as pl  # lazy, matches the pipeline stages

    manifest_path = cache / "frames.json"
    if not manifest_path.exists():
        log.warning("stroke dataset: no frames.json under %s — skipping %d labels",
                    cache, len(labels))
        return [], []
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["frames"]
    frame_w, frame_h = float(manifest["width"]), float(manifest["height"])
    _fps_guard(frames, labels[0]["video_id"] if labels else "?")

    def _read(name: str) -> Optional["pl.DataFrame"]:
        p = cache / name
        return pl.read_parquet(p) if p.exists() else None

    pose_df, tracks_df, shuttle_df = _read("pose.parquet"), _read("tracks.parquet"), _read("shuttle.parquet")
    dets = ({int(r["frame"]): (float(r["px"]), float(r["py"]))
             for r in shuttle_df.iter_rows(named=True)} if shuttle_df is not None else {})
    track_rows = list(tracks_df.iter_rows(named=True)) if tracks_df is not None else []

    samples, meta = [], []
    for lab in labels:
        pose_rows: list = []
        if pose_df is not None:
            # cheap pre-filter: pose rows within ±1 s of the anchor cover any 16-frame window
            t = float(lab["t_s"])
            near = [f for f in frames if abs(f["t"] - t) <= 1.0]
            if near:
                pose_rows = list(pose_df.filter(
                    (pl.col("frame") >= near[0]["frame"])
                    & (pl.col("frame") <= near[-1]["frame"])).iter_rows(named=True))
        s = build_sample(pose_rows, track_rows, dets, frames, float(lab["t_s"]),
                         frame_w, frame_h, homography=homography)
        if s is None:
            log.warning("stroke dataset: no anchor for event %s (t=%.2fs) — skipped",
                        lab.get("event_id"), lab["t_s"])
            continue
        samples.append(s)
        meta.append({**lab, **s["meta"], "is_calibration": lab["shot_type"] == "unknown"})
    return samples, meta


_META_SCHEMA = {"event_id": "Utf8", "match_id": "Utf8", "video_id": "Utf8", "t_s": "Float64",
                "shot_type": "Utf8", "hitter": "Utf8", "rally_idx": "Int64", "hit_idx": "Int64",
                "anchor": "Int64", "shuttle_coverage": "Float64", "pad": "Int64",
                "track_near_left": "Int64", "track_near_right": "Int64",
                "track_far_left": "Int64", "track_far_right": "Int64",
                "source_near_left": "Utf8", "source_near_right": "Utf8",
                "source_far_left": "Utf8", "source_far_right": "Utf8",
                "is_calibration": "Boolean"}


def _fetch_homographies(video_ids: Sequence[str]) -> dict:
    """Per-video calibration homography from the DB (the cache stores none). Any failure
    degrades to no-recovery with a loud warning — the exporter must work offline too."""
    try:
        from ..config import load_config
        from ..store import Store

        cfg = load_config()
        store = Store(cfg.supabase_url, cfg.service_role_key)
        out = {}
        for vid in video_ids:
            cal = store.get_calibration(vid)
            out[vid] = cal.get("homography") if cal else None
            if out[vid] is None:
                log.warning("stroke dataset: no calibration for video %s — pose recovery off", vid)
        return out
    except Exception:
        log.exception("stroke dataset: calibration fetch failed — pose recovery off for all videos")
        return {}


def export_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    import numpy as np
    import polars as pl

    from ..config import load_config

    ap = argparse.ArgumentParser(prog="worker export-stroke-dataset",
                                 description="Materialize BST training samples from stroke labels.")
    ap.add_argument("--labels", help="stroke_labels.parquet (default: worker/.labeling next to the cache)")
    ap.add_argument("--out", help="output dir (default: <labels dir>/stroke_dataset)")
    ap.add_argument("--cache", help="frame cache root (default: WORKER_FRAME_CACHE_DIR)")
    ap.add_argument("--no-recovery", action="store_true",
                    help="skip pose recovery (no DB calibration fetch)")
    args = ap.parse_args(argv)

    cfg = load_config()
    cache_root = Path(args.cache or cfg.frame_cache_dir).expanduser()
    labels_path = (Path(args.labels).expanduser() if args.labels
                   else cache_root.parent / ".labeling" / "stroke_labels.parquet")
    out_dir = Path(args.out).expanduser() if args.out else labels_path.parent / "stroke_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(pl.read_parquet(labels_path).iter_rows(named=True))
    video_ids = sorted({lab["video_id"] for lab in labels})
    homographies = {} if args.no_recovery else _fetch_homographies(video_ids)

    all_samples, all_meta = [], []
    for video_id in video_ids:
        vid_labels = [lab for lab in labels if lab["video_id"] == video_id]
        samples, meta = build_video_samples(cache_root / str(video_id), vid_labels,
                                            homography=homographies.get(video_id))
        all_samples.extend(samples)
        all_meta.extend(meta)
        print(f"{video_id}: {len(samples)}/{len(vid_labels)} samples")

    if not all_samples:
        print("no samples built — check cache dirs")
        return 1

    np.savez_compressed(
        out_dir / "stroke_dataset.npz",
        joints=np.asarray([s["joints"] for s in all_samples], dtype=np.float32),
        shuttle=np.asarray([s["shuttle"] for s in all_samples], dtype=np.float32),
        positions=np.asarray([s["positions"] for s in all_samples], dtype=np.float32),
    )
    pl.DataFrame(all_meta, schema={k: getattr(pl, v) for k, v in _META_SCHEMA.items()},
                 strict=False).write_parquet(out_dir / "meta.parquet")

    train = [m for m in all_meta if not m["is_calibration"]]
    cov = sorted(m["shuttle_coverage"] for m in all_meta)
    rep = slot_report(all_meta)
    print(f"\n{len(all_meta)} samples -> {out_dir} "
          f"({len(train)} train, {len(all_meta) - len(train)} calibration)")
    print(f"shuttle coverage: median {cov[len(cov) // 2]:.2f}, "
          f"min {cov[0]:.2f}, zero-coverage {sum(1 for c in cov if c == 0)}")
    print(f"slot fill rates: " + ", ".join(
        f"{s} {rep['slot_fill_rates'][s]:.2f}" for s in SLOTS))
    print(f"GATE — hitter slot filled: {rep['hitter_slot_filled']}/{rep['n_hitter_labeled']} "
          f"({rep['hitter_slot_filled_rate']:.1%})  by source: {rep['hitter_slot_by_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_cli())
