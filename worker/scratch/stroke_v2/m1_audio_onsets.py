"""M1 — audio onsets vs. known hits, all 5 Session-2 games.

Throwaway measurement (spec `stroke-classification-v2.md` §10, memo §11). Not wired into
the worker CLI, no tests — scratchpad only.

Answers, per hit event:
  - timing precision: signed offset (nearest audio onset - known hit t_s), inside +-150ms
    (the intended production candidate window) and a wider +-400ms diagnostic window (the
    video heuristic that produced t_s has +-0.2s drift per spec Stage 1).
  - first recall estimate: fraction of known hits with >=1 onset candidate in +-150ms.
  - coincidence probability: fraction of +-150ms windows containing >1 onset candidate
    (ambiguous — could be a foreign court's impact landing in-window).
  - false-onset-in-candidate-window rate: same check on the 43 `no_hit` heuristic false
    positives per game (a direct read on "falscher Onset gewaehlt" inside a candidate the
    video side already proposed).
  - neighbor-court onset rate: onset density (onsets/sec) during idle time (outside any
    rally span) vs during rally-active time, per game. Idle-time onsets are provably
    foreign since our own court has no play there.

Usage: uv run python scratch/stroke_v2/m1_audio_onsets.py
(run from worker/worker/ so the relative .labeling/.cache/../../videos paths resolve)
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

WINDOW_PROD_S = 0.150
WINDOW_DIAG_S = 0.400

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
OUT_DIR = LABELING_DIR / "phase1_measurements" / "m1_audio_onsets"


def extract_audio(video_path: Path, wav_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "2", "-ar", "48000",
         "-acodec", "pcm_s16le", str(wav_path)],
        check=True, capture_output=True,
    )


def detect_onsets(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    y, sr = librosa.load(wav_path, sr=48000, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    strengths = onset_env[onset_frames] if len(onset_frames) else np.array([])
    return onset_times, strengths


def window_stats(hit_times: np.ndarray, onset_times: np.ndarray, half_window_s: float) -> list[dict]:
    out = []
    for t in hit_times:
        lo, hi = t - half_window_s, t + half_window_s
        in_window = onset_times[(onset_times >= lo) & (onset_times <= hi)]
        if len(in_window):
            nearest = in_window[np.argmin(np.abs(in_window - t))]
            dt = float(nearest - t)
        else:
            dt = None
        out.append({"t_s": float(t), "n_candidates": int(len(in_window)), "dt": dt})
    return out


def idle_vs_active_rate(onset_times: np.ndarray, rallies: pl.DataFrame, duration_s: float) -> dict:
    spans = list(zip(rallies["t_start_s"].to_list(), rallies["t_end_s"].to_list()))
    active_dur = sum(e - s for s, e in spans)
    idle_dur = max(duration_s - active_dur, 0.0)

    def is_active(t: float) -> bool:
        return any(s <= t <= e for s, e in spans)

    n_active = sum(1 for t in onset_times if is_active(t))
    n_idle = len(onset_times) - n_active
    return {
        "active_duration_s": active_dur,
        "idle_duration_s": idle_dur,
        "n_onsets_active": n_active,
        "n_onsets_idle": n_idle,
        "onset_rate_active_per_s": n_active / active_dur if active_dur > 0 else None,
        "onset_rate_idle_per_s": n_idle / idle_dur if idle_dur > 0 else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stroke_labels = pl.read_parquet(LABELING_DIR / "stroke_labels.parquet")
    no_hit = pl.read_parquet(LABELING_DIR / "hit_detection_eval.parquet")

    per_game = {}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for game, video_id in GAMES.items():
            video_path = VIDEOS_DIR / f"Session2_Game_{game}.mp4"
            wav_path = tdp / f"game{game}.wav"
            print(f"[game {game}] extracting audio from {video_path} ...")
            extract_audio(video_path, wav_path)

            print(f"[game {game}] running onset detection ...")
            onset_times, strengths = detect_onsets(wav_path)

            hits = stroke_labels.filter(pl.col("video_id") == video_id)["t_s"].to_numpy()
            fps_candidates = no_hit.filter(pl.col("video_id") == video_id)["t_s"].to_numpy()

            prod = window_stats(hits, onset_times, WINDOW_PROD_S)
            diag = window_stats(hits, onset_times, WINDOW_DIAG_S)
            fp_prod = window_stats(fps_candidates, onset_times, WINDOW_PROD_S)

            dts = [w["dt"] for w in prod if w["dt"] is not None]
            recall_prod = sum(1 for w in prod if w["dt"] is not None) / len(prod) if prod else None
            coincidence_prod = (sum(1 for w in prod if w["n_candidates"] > 1) / len(prod)
                                if prod else None)
            fp_onset_rate = (sum(1 for w in fp_prod if w["n_candidates"] > 0) / len(fp_prod)
                             if fp_prod else None)

            rallies = pl.read_parquet(CACHE_DIR / video_id / "rallies.parquet")
            duration_s = float(onset_times.max()) if len(onset_times) else 0.0
            # use last rally end + margin as a duration proxy if no better source
            duration_s = max(duration_s, float(rallies["t_end_s"].max()) + 5.0)
            idle_stats = idle_vs_active_rate(onset_times, rallies, duration_s)

            per_game[game] = {
                "video_id": video_id,
                "n_hits": len(hits),
                "n_no_hit_candidates": len(fps_candidates),
                "n_onsets_total": int(len(onset_times)),
                "median_abs_dt_ms_prod_window": (float(np.median(np.abs(dts)) * 1000)
                                                 if dts else None),
                "n_hits_with_onset_prod_window": sum(1 for w in prod if w["dt"] is not None),
                "first_recall_estimate_prod_window": recall_prod,
                "coincidence_probability_prod_window": coincidence_prod,
                "diag_window_recall": (sum(1 for w in diag if w["dt"] is not None) / len(diag)
                                       if diag else None),
                "false_onset_rate_in_no_hit_windows": fp_onset_rate,
                "idle_vs_active": idle_stats,
                "hit_level_prod": prod,
                "hit_level_diag": diag,
                "no_hit_level_prod": fp_prod,
            }
            print(f"[game {game}] n_hits={len(hits)} median|dt|={per_game[game]['median_abs_dt_ms_prod_window']} "
                  f"recall={recall_prod} coincidence={coincidence_prod} fp_onset_rate={fp_onset_rate}")

    all_dts = [w["dt"] for g in per_game.values() for w in g["hit_level_prod"] if w["dt"] is not None]
    all_hits = sum(g["n_hits"] for g in per_game.values())
    all_recall_n = sum(g["n_hits_with_onset_prod_window"] for g in per_game.values())
    all_coincidence_n = sum(sum(1 for w in g["hit_level_prod"] if w["n_candidates"] > 1)
                            for g in per_game.values())
    all_fp = [w for g in per_game.values() for w in g["no_hit_level_prod"]]
    all_fp_onset_n = sum(1 for w in all_fp if w["n_candidates"] > 0)

    pooled = {
        "n_games": len(GAMES),
        "n_hits_total": all_hits,
        "median_abs_dt_ms": float(np.median(np.abs(all_dts)) * 1000) if all_dts else None,
        "p90_abs_dt_ms": float(np.percentile(np.abs(all_dts), 90) * 1000) if all_dts else None,
        "recall_at_prod_window": all_recall_n / all_hits if all_hits else None,
        "recall_within_50ms": (sum(1 for d in all_dts if abs(d) <= 0.050) / all_hits
                               if all_hits else None),
        "precision_within_50ms": (sum(1 for d in all_dts if abs(d) <= 0.050) / len(all_dts)
                                  if all_dts else None),
        "coincidence_probability": all_coincidence_n / all_hits if all_hits else None,
        "false_onset_rate_in_no_hit_windows": all_fp_onset_n / len(all_fp) if all_fp else None,
        "n_no_hit_candidates_total": len(all_fp),
    }

    report = {"window_prod_s": WINDOW_PROD_S, "window_diag_s": WINDOW_DIAG_S,
              "per_game": per_game, "pooled": pooled}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== POOLED (all 5 games, n=%d hits) ===" % all_hits)
    for k, v in pooled.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
