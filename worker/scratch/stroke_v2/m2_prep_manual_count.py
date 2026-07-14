"""M2 prep — sample rallies + export 0.25x clips + a CSV template for the trainer's manual
hit count. This script only prepares materials; it does NOT do the counting (that's the
trainer, by hand, per spec `stroke-classification-v2.md` §10).

Samples 3 rallies per game x 5 games = 15 rallies (fixed seed), spanning every Session-2
game so the resulting hand-count spans >=2 games (G0's eval-set requirement) and a variety
of hall/neighbor-court conditions. Each rally is exported at 0.25x speed (video+audio, so
the trainer can also listen) via ffmpeg. A companion CSV lists each clip with the game/rally
metadata and the *current heuristic's* hit count for reference, plus blank columns for the
trainer to fill in by hand.

Usage: uv run python ../scratch/stroke_v2/m2_prep_manual_count.py
(run from worker/worker/ so the relative .cache/../../videos paths resolve)
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

RALLIES_PER_GAME = 3
SEED = 42

GAMES = {
    1: "9aaa599b-47b5-4487-bdc9-9ad316c05850",
    2: "ff0d8099-401a-49a0-b987-bd61bc87e874",
    3: "f9605876-b889-42cf-b92f-23dcedab3796",
    4: "e7458ace-c494-4d80-8804-0af0fe39a2d8",
    5: "91deb7ca-ff0b-4d11-a4d1-452530426519",
}

VIDEOS_DIR = Path("../../videos")
CACHE_DIR = Path(".cache")
LABELING_DIR = Path(".labeling")
OUT_DIR = LABELING_DIR / "phase1_measurements" / "m2_manual_count"
CLIPS_DIR = OUT_DIR / "clips"


def export_clip(video_path: Path, t_start: float, t_end: float, out_path: Path,
                pad_s: float = 1.0) -> None:
    start = max(t_start - pad_s, 0.0)
    dur = (t_end - t_start) + 2 * pad_s
    # -ss/-t as INPUT options (before -i): trims the source BEFORE setpts stretches the
    # timeline 4x. Placed after -i they'd apply to the already-stretched output and truncate
    # the clip to its first quarter (caught via ffprobe duration check on the first export).
    # setpts doesn't duplicate/interpolate frames, so "frame N of the clip" still maps 1:1 to
    # the same source frame regardless of playback speed — only the on-screen clip-relative
    # *time* is stretched 4x, not the frame count or frame identity.
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video_path),
         "-filter:v", "setpts=4.0*PTS", "-filter:a", "atempo=0.5,atempo=0.5",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac",
         str(out_path)],
        check=True, capture_output=True,
    )


def main() -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    stroke_labels = pl.read_parquet(LABELING_DIR / "stroke_labels.parquet")
    rng = np.random.default_rng(SEED)

    rows = []
    for game, video_id in GAMES.items():
        rallies = pl.read_parquet(CACHE_DIR / video_id / "rallies.parquet")
        # bias toward rallies with >=3 shots (a single-hit "rally" is a weak recall test)
        candidates = rallies.filter(pl.col("shot_count") >= 3)
        if candidates.height < RALLIES_PER_GAME:
            candidates = rallies
        idx = rng.choice(candidates.height, size=min(RALLIES_PER_GAME, candidates.height),
                         replace=False)
        picked = candidates[sorted(idx.tolist())]

        video_path = VIDEOS_DIR / f"Session2_Game_{game}.mp4"
        for r in picked.iter_rows(named=True):
            heuristic_hits = stroke_labels.filter(
                (pl.col("video_id") == video_id) & (pl.col("rally_idx") == r["idx"])
            ).height
            clip_name = f"Session2_Game_{game}_rally{r['idx']:03d}.mp4"
            clip_path = CLIPS_DIR / clip_name
            print(f"exporting {clip_name} (t={r['t_start_s']:.2f}-{r['t_end_s']:.2f}s, "
                 f"heuristic_hits={heuristic_hits}) ...")
            export_clip(video_path, r["t_start_s"], r["t_end_s"], clip_path)
            rows.append({
                "clip_file": clip_name, "game": f"Session2_Game_{game}", "rally_idx": r["idx"],
                "t_start_s": round(r["t_start_s"], 2), "t_end_s": round(r["t_end_s"], 2),
                "heuristic_hit_count": heuristic_hits,
                # manual_hit_frame: frame number counted from the start of the clip (frame 0
                # = first frame), if the trainer's player exposes one — clips are 60000/1001
                # (~59.94) fps, so frame_number / 59.94 + (t_start_s - pad_s) recovers the
                # absolute game time. Optional, alongside manual_hit_times_s.
                "manual_hit_count": "", "manual_hit_times_s": "", "manual_hit_frame": "",
                "notes": "",
            })

    csv_path = OUT_DIR / "manual_count_template.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} rallies exported to {CLIPS_DIR}/")
    print(f"CSV template: {csv_path}")


if __name__ == "__main__":
    main()
