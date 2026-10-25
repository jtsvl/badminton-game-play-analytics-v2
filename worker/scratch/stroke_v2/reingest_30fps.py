"""Phase-0 NON-DESTRUCTIVE 30 fps re-ingest for the hit-detector rebuild
(spec stroke-classification-v2.md §5 Stage 0 / §10 M8 → 30 fps). Throwaway dev/ops script.

Why this exists and why it's safe: the 5 Session-2 games' 12 fps cache + Supabase artifacts
are LIVE production data. This driver re-runs only [ingest → detect → pose → gate_track →
shuttle] at analysis_fps=30 into a SEPARATE cache dir, through a STUB store whose every write
(DB + bucket) is a no-op and whose get_calibration returns the finalized homography (fetched
from Supabase via MCP, embedded below). Result: fresh 30 fps frames + shuttle/pose/tracks
parquets in an isolated dir, ZERO writes to production. calibrate + rally_hit are skipped
(homography is fps-independent; rally_hit is the trajectory-reversal heuristic we're replacing).

Run from the worker/ dir so the `worker` package imports:
    cd worker && uv run --no-sync python \
        ../.claude/worktrees/stroke-classification-v2/worker/scratch/stroke_v2/reingest_30fps.py \
        [--games 1,2] [--max-frames 60] [--cache-dir ABS]

Smoke-test first with --games 1 --max-frames 60, verify the parquets, THEN run the full sweep.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

REPO_ROOT = Path("/Users/jtsvl/Projects/badminton-game-play-analytics-v2")
PROD_CACHE = REPO_ROOT / "worker" / "worker" / ".cache"        # 12 fps LIVE cache — must NOT touch
DEFAULT_ISOLATED_CACHE = REPO_ROOT / "worker" / ".cache-30fps"  # separate, safe
ANALYSIS_FPS = 30.0
STAGES_TO_RUN = ["ingest", "detect", "pose", "gate_track", "shuttle"]

log = logging.getLogger("reingest30")

# video_id, storage_path, finalized homography (flat 9, verbatim from court_calibrations via the
# supabase MCP 2026-07-13). All doubles. storage_path matches videos/<file> → decode in place.
GAMES = [
    {"game": 1, "video_id": "9aaa599b-47b5-4487-bdc9-9ad316c05850", "storage_path": "Session2_Game_1.mp4",
     "H": [-0.00038677198867395596, -0.0004115877551034125, 1.1803781245209723, -0.000004270711135086751, -0.001184789690712836, 1.5005036199488786, -0.000006193455445944078, -0.0009472995195181635, 1],
     "poly": [[1710.7018457937938, 1260.3061270974108], [2245.283360523416, 1258.379166350315], [3480.1465702059327, 2135.6464960898943], [802.2699590867549, 2113.966137726824]]},
    {"game": 2, "video_id": "ff0d8099-401a-49a0-b987-bd61bc87e874", "storage_path": "Session2_Game_2.mp4",
     "H": [-0.0003866599926223732, -0.00040849176397954594, 1.1757068105206712, -0.0000039905352722796236, -0.0011889842745871342, 1.5085301667314315, -0.00000500136827184857, -0.0009489228409586528, 1],
     "poly": [[1706.3313887441598, 1263.0284716422443], [2246.267328339469, 1261.2163085566774], [3480.7041519683626, 2132.989750208792], [799.1786601979893, 2121.6985802141066]]},
    {"game": 3, "video_id": "f9605876-b889-42cf-b92f-23dcedab3796", "storage_path": "Session2_Game_3.mp4",
     "H": [-0.0003836744842313158, -0.00040796238231134046, 1.1718905805194877, 5.615396163212862e-07, -0.0011945430536191288, 1.510995248993574, -3.62129015697578e-07, -0.0009532256027155708, 1],
     "poly": [[1708.5453163302784, 1265.7180168552122], [2249.28482964203, 1265.9722116771695], [3479.5643892481016, 2130.8413939716042], [799.56404759261, 2120.583402930681]]},
    {"game": 4, "video_id": "e7458ace-c494-4d80-8804-0af0fe39a2d8", "storage_path": "Session2_Game_4.mp4",
     "H": [-0.0003858760868101645, -0.00040913837547212667, 1.1771139013929934, -0.0000012891513520245678, -0.0011914879761393056, 1.5077920312525297, -0.0000022770278829579876, -0.0009518920326107804, 1],
     "poly": [[1710.7018457937938, 1263.6188596158227], [2248.7846460022038, 1263.0366714752108], [3480.712351700163, 2133.7195353427983], [799.8674376692262, 2122.6661605041377]]},
    {"game": 5, "video_id": "91deb7ca-ff0b-4d11-a4d1-452530426519", "storage_path": "Session2_Game_5.mp4",
     "H": [-0.0003844866779153666, -0.00040382568680675493, 1.1576594446064943, -0.000009319835020812044, -0.001181101434148603, 1.5151271945141396, -0.000007753162952481746, -0.0009411347657861363, 1],
     "poly": [[1677.4929320020158, 1269.571938349404], [2223.053887612426, 1265.267026042062], [3448.020021011813, 2124.15033027105], [761.5090923067725, 2141.6897729861057]]},
]
DISCIPLINE = "doubles"


class StubStore:
    """Satisfies exactly the store methods [ingest, detect, pose, gate_track, shuttle] call.
    Every DB/bucket write is a no-op; get_calibration returns the finalized homography;
    get_match_players=[] makes gate_track's _auto_map a no-op (no track labels written)."""

    def __init__(self, games):
        self._path = {g["video_id"]: g["storage_path"] for g in games}
        self._H = {g["video_id"]: g["H"] for g in games}
        self._poly = {g["video_id"]: g["poly"] for g in games}

    def get_video(self, video_id):
        return {"storage_path": self._path[video_id]}

    def download(self, *a, **k):
        raise RuntimeError("stub: download called — expected local videos/ file in place")

    def get_calibration(self, video_id):
        return {"id": "stub", "homography": self._H[video_id],
                "mask_polygon_px": self._poly[video_id]}

    def get_match(self, *a, **k):
        return {"discipline": DISCIPLINE}

    def get_match_players(self, *a, **k):
        return []

    # all writes are no-ops
    def update_video_meta(self, *a, **k): pass
    def upload(self, *a, **k): pass
    def update_calibration(self, *a, **k): pass
    def insert_calibration(self, *a, **k): return {"id": "stub"}
    def set_track_label(self, *a, **k): pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="1,2,3,4,5", help="comma list of game numbers")
    ap.add_argument("--max-frames", type=int, default=None, help="dev cap per game (smoke test)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_ISOLATED_CACHE))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if cache_dir == PROD_CACHE.resolve() or cache_dir == (REPO_ROOT / "worker" / ".cache").resolve():
        raise SystemExit(f"REFUSING to write into a production cache path: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WORKER_FRAME_CACHE_DIR"] = str(cache_dir)  # before load_config

    from worker.config import load_config
    from worker.pipeline import STAGES, StageContext

    cfg = load_config()
    assert Path(cfg.frame_cache_dir).resolve() == cache_dir, cfg.frame_cache_dir
    log.info("isolated 30 fps cache: %s | prod cache untouched: %s", cache_dir, PROD_CACHE)

    want = {int(x) for x in args.games.split(",") if x.strip()}
    games = [g for g in GAMES if g["game"] in want]
    store = StubStore(games)

    t_all = time.time()
    for g in games:
        params = {"analysis_fps": ANALYSIS_FPS}
        if args.max_frames:
            params["max_frames"] = args.max_frames
        job = {"video_id": g["video_id"], "match_id": f"stub-{g['video_id']}", "params": params}
        ctx = StageContext(job=job, cfg=cfg, store=store)
        ctx["H"] = g["H"]  # belt-and-suspenders; get_calibration would also supply it
        log.info("=== Game %d (%s) @ %.0f fps%s ===", g["game"], g["storage_path"],
                 ANALYSIS_FPS, f" cap={args.max_frames}" if args.max_frames else "")
        t_game = time.time()
        for name in STAGES_TO_RUN:
            t0 = time.time()
            STAGES[name](ctx)
            log.info("  [%s] %s done in %.1fs", g["storage_path"], name, time.time() - t0)
        log.info("=== Game %d done in %.1f min ===", g["game"], (time.time() - t_game) / 60)
    log.info("ALL %d games done in %.1f min", len(games), (time.time() - t_all) / 60)


if __name__ == "__main__":
    main()
