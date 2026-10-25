"""M2 analysis — turn the trainer's filled-in `manual_count_template.csv` into the true
heuristic recall/precision numbers, per spec `stroke-classification-v2.md` §10. Throwaway,
scratchpad only.

The CSV uses `;` as the field delimiter (trainer's spreadsheet app exported that way — de-DE
locale default) and `manual_hit_frame` is itself a `;`-separated, quoted list of frame numbers
counted from the start of each exported clip (frame 0 = first frame). Clips run at a fixed
60000/1001 (~59.94) fps regardless of playback speed (quarter-speed only stretches how long
each frame is *shown*, it doesn't duplicate frames) and were exported with 1.0s of padding
before t_start_s, so:

    absolute_game_time_s = frame_number / (60000/1001) + (t_start_s - pad_s)

For each rally, the manual hit times are greedily matched against the heuristic's own hit
times (`stroke_labels.parquet` + `hit_detection_eval.parquet`, both already carry a precise
`t_s` per candidate) within a generous window — this both measures recall/precision AND, for
matched pairs, the heuristic's own timing drift (a second, independent read on the same
question M1 asked via audio, but against a human instead of a microphone).

Usage: uv run python ../scratch/stroke_v2/m2_analyze_manual_count.py
(run from worker/worker/ so the relative .labeling path resolves)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

FPS = 60000 / 1001
PAD_S = 1.0
MATCH_WINDOW_S = 0.400  # generous — same diagnostic window used in M1

LABELING_DIR = Path(".labeling")
CSV_PATH = LABELING_DIR / "phase1_measurements" / "m2_manual_count" / "manual_count_template.csv"
OUT_DIR = LABELING_DIR / "phase1_measurements" / "m2_manual_count"

GAME_TO_VIDEO_ID = {
    "Session2_Game_1": "9aaa599b-47b5-4487-bdc9-9ad316c05850",
    "Session2_Game_2": "ff0d8099-401a-49a0-b987-bd61bc87e874",
    "Session2_Game_3": "f9605876-b889-42cf-b92f-23dcedab3796",
    "Session2_Game_4": "e7458ace-c494-4d80-8804-0af0fe39a2d8",
    "Session2_Game_5": "91deb7ca-ff0b-4d11-a4d1-452530426519",
}


def parse_frames(cell: str) -> list[int]:
    cell = (cell or "").strip()
    if not cell:
        return []
    return [int(x) for x in cell.split(";") if x.strip() != ""]


def greedy_match(manual_times: list[float], heur_times: list[float], window_s: float):
    """Order-preserving two-pointer merge match (both lists pre-sorted ascending): advance
    whichever side is earlier when out of window, match-and-advance-both when within window.
    Avoids the crossing artifacts a "nearest available, any order" greedy match can produce
    in fast rallies where several candidates sit close together in time.
    Returns (matched_pairs, unmatched_manual, unmatched_heuristic)."""
    i, j = 0, 0
    pairs, unmatched_manual, unmatched_heur = [], [], []
    while i < len(manual_times) and j < len(heur_times):
        d = heur_times[j] - manual_times[i]
        if abs(d) <= window_s:
            pairs.append((manual_times[i], heur_times[j]))
            i += 1
            j += 1
        elif d < -window_s:  # heuristic candidate sits well before this manual hit
            unmatched_heur.append(heur_times[j])
            j += 1
        else:  # heuristic candidate sits well after this manual hit
            unmatched_manual.append(manual_times[i])
            i += 1
    unmatched_manual.extend(manual_times[i:])
    unmatched_heur.extend(heur_times[j:])
    return pairs, unmatched_manual, unmatched_heur


def main() -> None:
    stroke_labels = pl.read_parquet(LABELING_DIR / "stroke_labels.parquet")
    no_hit = pl.read_parquet(LABELING_DIR / "hit_detection_eval.parquet")
    heur_all = pl.concat([
        stroke_labels.select("video_id", "rally_idx", "t_s", "shot_type"),
        no_hit.select("video_id", "rally_idx", "t_s", "shot_type"),
    ])

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    per_rally = []
    all_pairs_dt = []
    all_tp_shot_types = []
    all_fn = 0
    all_fp = 0
    all_tp = 0
    for row in rows:
        game = row["game"]
        video_id = GAME_TO_VIDEO_ID[game]
        rally_idx = int(row["rally_idx"])
        t_start_s = float(row["t_start_s"])

        frames = parse_frames(row["manual_hit_frame"])
        manual_times = sorted(f / FPS + (t_start_s - PAD_S) for f in frames)

        heur_rows = heur_all.filter(
            (pl.col("video_id") == video_id) & (pl.col("rally_idx") == rally_idx)
        )
        heur_times = sorted(heur_rows["t_s"].to_list())
        shot_type_by_t = dict(zip(heur_rows["t_s"].to_list(), heur_rows["shot_type"].to_list()))

        pairs, unmatched_manual, unmatched_heur = greedy_match(manual_times, heur_times,
                                                                MATCH_WINDOW_S)
        dts = [ht - mt for mt, ht in pairs]
        all_pairs_dt.extend(dts)
        all_tp_shot_types.extend(shot_type_by_t[ht] for _, ht in pairs)
        all_tp += len(pairs)
        all_fn += len(unmatched_manual)
        all_fp += len(unmatched_heur)

        per_rally.append({
            "clip_file": row["clip_file"], "game": game, "rally_idx": rally_idx,
            "manual_hit_count_declared": int(row["manual_hit_count"]) if row["manual_hit_count"] else None,
            "manual_frames_given": len(frames),
            "heuristic_hit_count_col": int(row["heuristic_hit_count"]),
            "heuristic_candidates_in_rally": len(heur_times),
            "matched": len(pairs),
            "heuristic_missed_real_hits": len(unmatched_manual),
            "heuristic_invented_hits": len(unmatched_heur),
            "matched_dt_ms": [round(d * 1000, 1) for d in dts],
            "notes": row["notes"],
        })
        print(f"{row['clip_file']}: manual={len(frames)} heur={len(heur_times)} "
              f"matched={len(pairs)} missed={len(unmatched_manual)} invented={len(unmatched_heur)}")

    all_manual_n = all_tp + all_fn
    all_heur_n = all_tp + all_fp
    pooled = {
        "n_rallies": len(rows),
        "n_manual_hits_total": all_manual_n,
        "n_heuristic_candidates_total": all_heur_n,
        "matched": all_tp,
        "heuristic_missed_real_hits": all_fn,
        "heuristic_invented_hits": all_fp,
        "recall": all_tp / all_manual_n if all_manual_n else None,
        "precision": all_tp / all_heur_n if all_heur_n else None,
        "median_abs_dt_ms_matched": (float(np.median(np.abs(all_pairs_dt))) * 1000
                                     if all_pairs_dt else None),
        "p90_abs_dt_ms_matched": (float(np.percentile(np.abs(all_pairs_dt), 90)) * 1000
                                  if all_pairs_dt else None),
        "mean_signed_dt_ms_matched": (float(np.mean(all_pairs_dt)) * 1000
                                      if all_pairs_dt else None),
    }

    from collections import Counter
    shot_type_of_matched_hits = dict(Counter(all_tp_shot_types))

    report = {"match_window_s": MATCH_WINDOW_S, "per_rally": per_rally, "pooled": pooled,
              "shot_type_distribution_of_heuristic_matched_hits": shot_type_of_matched_hits,
              "caveat": ("per-stroke-type recall cannot be computed from this data — the "
                        "trainer recorded hit *timing*, not shot *type*, so only hits the "
                        "heuristic already found have a known type (via stroke_labels.parquet); "
                        "missed hits' types are unknown.")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "m2_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== POOLED (all 15 rallies) ===")
    for k, v in pooled.items():
        print(f"  {k}: {v}")
    print(f"\nshot types of hits the heuristic DID catch: {shot_type_of_matched_hits}")
    print(f"\nWrote {OUT_DIR / 'm2_report.json'}")


if __name__ == "__main__":
    main()
