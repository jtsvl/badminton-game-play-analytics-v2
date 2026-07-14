"""M8 — fps sensitivity for the hit-detector re-ingest (30 fps vs 60 fps), per spec
`stroke-classification-v2.md` §10. Throwaway, scratchpad only; run from worker/worker/ so the
relative `.labeling` / `.cache` paths resolve.

    uv run python ../scratch/stroke_v2/m8_fps_sensitivity.py

The decision it informs: the current 12 fps frame cache makes G0's hit-timing bar (median
|Δt| ≤ 25 ms; recall/precision ≥ 0.90 at ±50 ms) physically unreachable, so we must re-ingest
finer. Is 30 fps enough, or do we need 60 fps?

Component 1 (decisive, fully automated) — quantization floor: using the 96 M2 frame-accurate
manual hit times (59.94 fps ground truth) as truth, the timing error forced on us if frames
only existed at rate R is |t_true − nearest_grid_line(R)|, grid anchored at t=0. Reported per
rate, per game + pooled, against G0's ≤25 ms median.

Component 2a (automated risk signal) — smash visibility: for every smash in
stroke_labels.parquet, the actual gap in the *already-computed* production shuttle.parquet
(TrackNetV3 @ 12 fps) bracketing the hit `t_s`. A large gap around contact is a tracker
occlusion/confidence problem that more frames alone do NOT fix.

Component 3 (literature/config) — recorded as findings in the report, not computed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

FPS_TRUTH = 60000 / 1001  # ~59.94, the manual-clip frame rate (M2 ground truth)
PAD_S = 1.0
RATES = [12, 30, 60]
G0_MEDIAN_MS = 25.0
SMASH_GAP_ALERT_S = 0.250  # ≥3 missed 12 fps samples bracketing a hit

LABELING_DIR = Path(".labeling")
CACHE_DIR = Path(".cache")
CSV_PATH = LABELING_DIR / "phase1_measurements" / "m2_manual_count" / "manual_count_template.csv"
OUT_DIR = LABELING_DIR / "phase1_measurements" / "m8_fps_sensitivity"

GAME_TO_VIDEO_ID = {
    "Session2_Game_1": "9aaa599b-47b5-4487-bdc9-9ad316c05850",
    "Session2_Game_2": "ff0d8099-401a-49a0-b987-bd61bc87e874",
    "Session2_Game_3": "f9605876-b889-42cf-b92f-23dcedab3796",
    "Session2_Game_4": "e7458ace-c494-4d80-8804-0af0fe39a2d8",
    "Session2_Game_5": "91deb7ca-ff0b-4d11-a4d1-452530426519",
}


def parse_frames(cell: str) -> list[int]:
    cell = (cell or "").strip()
    return [int(x) for x in cell.split(";") if x.strip() != ""] if cell else []


def quant_error_s(times: np.ndarray, rate: int) -> np.ndarray:
    """|t − nearest grid line| for a frame grid at `rate` fps anchored at t=0."""
    spacing = 1.0 / rate
    return np.abs(times - np.round(times / spacing) * spacing)


def dist(errs_ms: np.ndarray) -> dict:
    return {
        "n": int(errs_ms.size),
        "median_ms": round(float(np.median(errs_ms)), 2),
        "p90_ms": round(float(np.percentile(errs_ms, 90)), 2),
        "mean_ms": round(float(np.mean(errs_ms)), 2),
        "max_ms": round(float(np.max(errs_ms)), 2),
    }


def component1_quantization() -> dict:
    import csv
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    times_by_game: dict[str, list[float]] = {g: [] for g in GAME_TO_VIDEO_ID}
    for row in rows:
        game = row["game"]
        t_start_s = float(row["t_start_s"])
        for fr in parse_frames(row["manual_hit_frame"]):
            times_by_game[game].append(fr / FPS_TRUTH + (t_start_s - PAD_S))

    all_times = np.array([t for ts in times_by_game.values() for t in ts])
    out = {"n_manual_hits": int(all_times.size), "per_rate": {}, "per_game": {}}
    for rate in RATES:
        pooled = dist(quant_error_s(all_times, rate) * 1000.0)
        pooled["clears_g0_median_pooled"] = pooled["median_ms"] <= G0_MEDIAN_MS
        out["per_rate"][rate] = pooled
    for game, ts in times_by_game.items():
        arr = np.array(ts)
        out["per_game"][game] = {
            rate: {"median_ms": round(float(np.median(quant_error_s(arr, rate) * 1000.0)), 2),
                   "clears_g0_median": bool(np.median(quant_error_s(arr, rate) * 1000.0) <= G0_MEDIAN_MS)}
            for rate in RATES
        }
    return out


def component2a_smash_gaps() -> dict:
    labels = pl.read_parquet(LABELING_DIR / "stroke_labels.parquet")
    smashes = labels.filter(pl.col("shot_type") == "smash")
    gaps_ms: list[float] = []
    per_video_cache: dict[str, pl.DataFrame] = {}
    n_no_track = 0
    for r in smashes.iter_rows(named=True):
        vid, t_s = r["video_id"], r["t_s"]
        if vid not in per_video_cache:
            p = CACHE_DIR / vid / "shuttle.parquet"
            per_video_cache[vid] = pl.read_parquet(p) if p.exists() else pl.DataFrame({"t": []})
        sh = per_video_cache[vid]
        if sh.height == 0:
            continue
        t = np.sort(sh["t"].to_numpy())
        before = t[t <= t_s]
        after = t[t >= t_s]
        if before.size == 0 or after.size == 0:
            n_no_track += 1
            continue
        gaps_ms.append(float(after[0] - before[-1]) * 1000.0)
    g = np.array(gaps_ms)
    return {
        "n_smashes": int(smashes.height),
        "n_measured": int(g.size),
        "n_no_bracketing_track": n_no_track,
        "median_gap_ms": round(float(np.median(g)), 1) if g.size else None,
        "p90_gap_ms": round(float(np.percentile(g, 90)), 1) if g.size else None,
        "max_gap_ms": round(float(np.max(g)), 1) if g.size else None,
        "frac_gap_over_250ms": round(float(np.mean(g > SMASH_GAP_ALERT_S * 1000.0)), 3) if g.size else None,
    }


COMPONENT3_FINDINGS = {
    "our_finetuned_tracknetv3": (
        "The deployed shuttle checkpoint was fine-tuned/evaluated at analysis_fps=30 "
        "(specs/features/shuttle-finetune-plan.md, shuttle-tracking.md) to match the public "
        "badminton TrackNet distribution. A 30 fps re-ingest is a ZERO-mismatch match to it; "
        "60 fps would introduce a new mismatch unless re-fine-tuned."),
    "upstream_tracknetv3": "fps-agnostic (seq_len=8 is a frame-count window, reads native fps at runtime).",
    "bst_shuttleset": (
        "BST uses a fixed seq_len=30 pose-sequence window, not raw video fps. ShuttleSet's "
        "source-video fps is not documented in the vendored files — unverified, not guessed."),
    "finebadminton": "documented 25 fps broadcast; 30 fps is proportionally closer than 60 fps.",
    "conclusion": "No evidence of an fps-mismatch transfer brake at 30 fps; the one concrete on-disk fact favors 30.",
}

RECOMMENDATION = (
    "Re-ingest at 30 fps. Quantization floor clears G0 with headroom at 30 fps in every game "
    "(median ~7.7 ms vs the ≤25 ms bar) and 60 fps improves it only ~3.6 ms; the real smash "
    "risk is a tracker occlusion/confidence gap near contact (~26% of smashes today) that more "
    "frames alone do not fix; and our deployed TrackNetV3 checkpoint expects 30 fps. Keep 60 fps "
    "available as a re-ingestable comparison if the real hit-detector's G0 eval later exposes a "
    "gap this analytic floor did not predict.")


def main() -> None:
    c1 = component1_quantization()
    c2 = component2a_smash_gaps()
    report = {
        "component1_quantization_floor": c1,
        "component2a_smash_track_gaps": c2,
        "component3_checkpoint_fps": COMPONENT3_FINDINGS,
        "recommendation": RECOMMENDATION,
        "caveats": (
            "Component 1 is the decisive, fully-automated result. Component 2a is an automated "
            "risk signal from the existing 12 fps track (not a 30/60 fps detector run). Component "
            "2b/2c visual smash case studies (hand-read pixels on ffmpeg-extracted 60 fps frames) "
            "are recorded in the session notes, not reproduced here — they need a trained detector."),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    print("=== M8 Component 1 — quantization floor (median |Δt|, ms) ===")
    for rate in RATES:
        d = c1["per_rate"][rate]
        print(f"  {rate:>2} fps: median={d['median_ms']:>5} p90={d['p90_ms']:>5} "
              f"max={d['max_ms']:>5}  clears G0 median (pooled): {d['clears_g0_median_pooled']}")
    print("  per-game median at 12 fps (the marginal rate):")
    for game, gd in c1["per_game"].items():
        print(f"    {game}: {gd[12]['median_ms']} ms  clears={gd[12]['clears_g0_median']}")
    print(f"\n=== M8 Component 2a — smash track gaps (n={c2['n_measured']}/{c2['n_smashes']}) ===")
    print(f"  median gap={c2['median_gap_ms']}ms  p90={c2['p90_gap_ms']}ms  "
          f"max={c2['max_gap_ms']}ms  frac>250ms={c2['frac_gap_over_250ms']}")
    print(f"\nRECOMMENDATION: {RECOMMENDATION}")
    print(f"\nWrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
