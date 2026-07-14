"""overlay: _pose_keyframes — pose.parquet → per-player skeleton keyframes (pose-skeleton-overlay).

Reads synthetic pose.parquet + frames.json from a tmp cache; no model, no DB. Locks the contract:
t and width/height come from the frames.json manifest (never frame/fps — PTS drift; no DB roundtrip);
identity binds label → role → the WHOLE role timeline, so re-ID fragment rows export under the
role's player (the coverage fix) while bystanders/unrolled tracks and gap-fill interp frames drop
out (frame-level gate: only (frame, track) pairs present in tracks.parquet non-interp rows ship).
Export is whole-clip (skeletons between rallies too — user decision 2026-07-13); the payload
safety valve is POSE_MAX_KEYFRAMES_PER_PLAYER (thin, never skip). Frame-normalized
[[x,y,score]×26] clamped to [0,1], rounded to 3/4 decimals, sorted by t. Missing inputs return
None so old matches without the pose stage never break the overlay.
"""
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from worker.pipeline.overlay import POSE_MAX_KEYFRAMES_PER_PLAYER, _pose_keyframes

W, H = 1920, 1080


def _kp(x, y):
    """26 identical [x,y] joints — per-joint variety isn't what's under test."""
    return [[float(x), float(y)] for _ in range(26)]


def _write_pose(cache: Path, rows):
    """rows: (frame, track, x, y, score) — mirrors pose.parquet's schema (foot_* cols are ignored)."""
    pl.DataFrame(
        {
            "frame": [r[0] for r in rows],
            "track": [r[1] for r in rows],
            "kp": [_kp(r[2], r[3]) for r in rows],
            "kp_score": [[float(r[4])] * 26 for r in rows],
        }
    ).write_parquet(cache / "pose.parquet")


def _write_manifest(cache: Path, frame_ts: dict, width=W, height=H):
    (cache / "frames.json").write_text(
        json.dumps({"width": width, "height": height,
                    "frames": [{"frame": f, "t": t} for f, t in frame_ts.items()]})
    )


def _tracks(rows):
    """rows: (t, track_id, role, source) — the tracks.parquet slice identity consumes."""
    return pl.DataFrame(
        {
            "t": [r[0] for r in rows],
            "track_id": [r[1] for r in rows],
            "u": [0.5] * len(rows),
            "v": [0.5] * len(rows),
            "role": [r[2] for r in rows],
            "source": [r[3] for r in rows],
        }
    )


CTX = {}  # _pose_keyframes reads everything from the cache dir; ctx only carries pose_path overrides

MP = [{"player_id": "p-1", "side": "A", "slot": 0, "track_label": 1}]


class TestPoseKeyframes(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name)

    def test_missing_pose_parquet_returns_none(self):
        _write_manifest(self.cache, {0: 0.0})
        tracks = _tracks([(0.0, 1, "near", "detected")])
        self.assertIsNone(_pose_keyframes(CTX, self.cache, tracks, MP, None))

    def test_missing_frames_json_returns_none(self):
        _write_pose(self.cache, [(0, 1, 100.0, 200.0, 0.9)])
        tracks = _tracks([(0.0, 1, "near", "detected")])
        self.assertIsNone(_pose_keyframes(CTX, self.cache, tracks, MP, None))

    def test_happy_path_normalizes_rounds_and_sorts(self):
        # Rows deliberately out of t order; track 2 has no role; frame 9 is not in the manifest.
        _write_pose(self.cache, [
            (1, 1, 960.0, 540.0, 0.87654),  # t=0.0833333
            (0, 1, 100.0, 200.0, 0.5),      # t=0.0
            (9, 1, 400.0, 400.0, 0.9),      # missing from manifest → dropped (never t=0.0)
            (0, 2, 500.0, 500.0, 0.9),      # bystander: not in tracks.parquet → dropped
        ])
        _write_manifest(self.cache, {0: 0.0, 1: 0.0833333})
        tracks = _tracks([(0.0, 1, "near", "detected"), (0.0833333, 1, "near", "detected")])

        out = _pose_keyframes(CTX, self.cache, tracks, MP, [(0.0, 1.0)])  # in-rally: native rate

        self.assertEqual(set(out), {"player:p-1"})
        data = out["player:p-1"]
        self.assertEqual([d["t"] for d in data], [0.0, 0.083])  # sorted, t rounded to 3 decimals
        for d in data:
            self.assertEqual(len(d["kp"]), 26)
            self.assertTrue(all(len(joint) == 3 for joint in d["kp"]))
        # Frame-normalized by the manifest's native width/height, x/y/score rounded to 4 decimals.
        self.assertEqual(data[0]["kp"][0], [round(100.0 / W, 4), round(200.0 / H, 4), 0.5])
        self.assertEqual(data[1]["kp"][0], [0.5, 0.5, 0.8765])

    def test_reid_fragment_rows_export_under_the_roles_player(self):
        # THE coverage bug: player labeled track 1, but frames 1-2 belong to re-ID fragment 42.
        # Old code dropped them (42 != label); the role join must export all three frames.
        _write_pose(self.cache, [
            (0, 1, 96.0, 54.0, 0.9),
            (1, 42, 192.0, 108.0, 0.9),
            (2, 42, 288.0, 162.0, 0.9),
        ])
        _write_manifest(self.cache, {0: 0.0, 1: 1.0, 2: 2.0})
        tracks = _tracks([
            (0.0, 1, "near", "detected"),
            (1.0, 42, "near", "reid"),
            (2.0, 42, "near", "reid"),
        ])
        out = _pose_keyframes(CTX, self.cache, tracks, MP, None)
        self.assertEqual([d["t"] for d in out["player:p-1"]], [0.0, 1.0, 2.0])

    def test_pose_row_outside_role_timeline_is_dropped(self):
        # Track 1 has a pose row at frame 1, but tracks.parquet has no row there (court-gated
        # out, e.g. walked off court) — the frame-level gate drops it.
        _write_pose(self.cache, [(0, 1, 96.0, 54.0, 0.9), (1, 1, 192.0, 108.0, 0.9)])
        _write_manifest(self.cache, {0: 0.0, 1: 1.0})
        tracks = _tracks([(0.0, 1, "near", "detected")])
        out = _pose_keyframes(CTX, self.cache, tracks, MP, None)
        self.assertEqual([d["t"] for d in out["player:p-1"]], [0.0])

    def test_interp_frames_produce_no_pose_entries(self):
        # Gap-fill rows are synthetic positions — no detection, no pose row; the join must not
        # fabricate one even if a bystander pose row shares the (frame, tid) key by accident.
        _write_pose(self.cache, [(0, 1, 96.0, 54.0, 0.9), (1, 1, 192.0, 108.0, 0.9)])
        _write_manifest(self.cache, {0: 0.0, 1: 1.0})
        tracks = _tracks([(0.0, 1, "near", "detected"), (1.0, 1, "near", "interp")])
        out = _pose_keyframes(CTX, self.cache, tracks, MP, None)
        self.assertEqual([d["t"] for d in out["player:p-1"]], [0.0])

    def test_whole_clip_rows_outside_rally_spans_are_kept(self):
        # Whole-video export (user decision 2026-07-13): spans no longer gate rows out.
        _write_pose(self.cache, [(0, 1, 96.0, 54.0, 0.9), (2, 1, 192.0, 108.0, 0.9)])
        _write_manifest(self.cache, {0: 0.0, 2: 5.0})
        tracks = _tracks([(0.0, 1, "near", "detected"), (5.0, 1, "near", "detected")])
        out = _pose_keyframes(CTX, self.cache, tracks, MP, [(0.0, 1.0)])
        self.assertEqual([d["t"] for d in out["player:p-1"]], [0.0, 5.0])

    def test_edge_joints_clamped_to_unit_range(self):
        # RTMPose places edge-cropped joints outside the frame — the stored contract is [0,1].
        _write_pose(self.cache, [(0, 1, -25.0, 1100.0, 0.9)])
        _write_manifest(self.cache, {0: 0.0})
        tracks = _tracks([(0.0, 1, "near", "detected")])
        out = _pose_keyframes(CTX, self.cache, tracks, MP, None)
        self.assertEqual(out["player:p-1"][0]["kp"][0], [0.0, 1.0, 0.9])

    def test_over_cap_is_thinned_not_skipped(self):
        # The payload safety valve: very long clips thin uniformly instead of losing the feature.
        n = POSE_MAX_KEYFRAMES_PER_PLAYER + 40
        _write_pose(self.cache, [(i, 1, 96.0, 54.0, 0.9) for i in range(n)])
        _write_manifest(self.cache, {i: i / 12.0 for i in range(n)})
        tracks = _tracks([(i / 12.0, 1, "near", "detected") for i in range(n)])
        out = _pose_keyframes(CTX, self.cache, tracks, MP, None)
        data = out["player:p-1"]
        self.assertLessEqual(len(data), POSE_MAX_KEYFRAMES_PER_PLAYER)
        self.assertGreater(len(data), POSE_MAX_KEYFRAMES_PER_PLAYER // 2)  # thinned, not truncated
        self.assertEqual(data[0]["t"], 0.0)  # keeps the start, strides through the rest

    def test_missing_pose_parquet_restored_from_artifact_bucket(self):
        # Fresh worker / evicted cache on a retrack: pose.parquet comes back from the bucket
        # (pose.py uploaded it) instead of silently skipping the skeletons.
        other = Path(tempfile.mkdtemp())
        _write_pose(other, [(0, 1, 96.0, 54.0, 0.9)])
        payload = (other / "pose.parquet").read_bytes()

        class _Store:
            def download(self, bucket, key):
                assert key == "vid-9/pose.parquet"
                return payload

        class _Cfg:
            artifact_bucket = "analysis-artifacts"

        _write_manifest(self.cache, {0: 0.0})
        tracks = _tracks([(0.0, 1, "near", "detected")])
        ctx = {"store": _Store(), "cfg": _Cfg(), "job": {"video_id": "vid-9"}}
        out = _pose_keyframes(ctx, self.cache, tracks, MP, None)
        self.assertEqual([d["t"] for d in out["player:p-1"]], [0.0])

    def test_out_of_rally_keyframes_thin_to_pose_keyframe_hz(self):
        # Hybrid payload control: native 12 fps INSIDE rally spans (fast swings), ~6 Hz between
        # rallies (walking around). poseAt's 0.3 s max-gap tolerates 1/6 s spacing, so rendering
        # never drops out.
        n = 48  # 4 s @ 12 fps
        _write_pose(self.cache, [(i, 1, 96.0, 54.0, 0.9) for i in range(n)])
        _write_manifest(self.cache, {i: i / 12.0 for i in range(n)})
        tracks = _tracks([(i / 12.0, 1, "near", "detected") for i in range(n)])

        out = _pose_keyframes(CTX, self.cache, tracks, MP, [(1.0, 2.0)])

        ts = [d["t"] for d in out["player:p-1"]]
        in_rally = [t for t in ts if 1.0 <= t <= 2.0]
        out_rally = [t for t in ts if t < 1.0 or t > 2.0]
        self.assertEqual(len(in_rally), 13)  # every analysis frame 1.0..2.0 kept
        # out-of-rally: one keyframe per 1/6 s bucket → spacing ≥ ~1/6 s, roughly halved count
        self.assertLess(len(out_rally), 20)
        self.assertGreater(len(out_rally), 15)
        for a, b in zip(out_rally, out_rally[1:]):
            if b - a < 0.9:  # ignore the jump across the in-rally window
                self.assertGreaterEqual(round(b - a, 4), round(1 / 12, 4))

    def test_thinning_off_keeps_native_rate(self):
        from unittest.mock import patch

        n = 48
        _write_pose(self.cache, [(i, 1, 96.0, 54.0, 0.9) for i in range(n)])
        _write_manifest(self.cache, {i: i / 12.0 for i in range(n)})
        tracks = _tracks([(i / 12.0, 1, "near", "detected") for i in range(n)])
        with patch("worker.pipeline.overlay.POSE_KEYFRAME_HZ", 0):
            out = _pose_keyframes(CTX, self.cache, tracks, MP, [(1.0, 2.0)])
        self.assertEqual(len(out["player:p-1"]), n)

    def test_no_bound_players_returns_empty_dict(self):
        # Labels that bind nothing → {} (run() then clears stale skeleton rows), not None.
        _write_pose(self.cache, [(0, 1, 96.0, 54.0, 0.9)])
        _write_manifest(self.cache, {0: 0.0})
        tracks = _tracks([(0.0, 1, "near", "detected")])
        mps = [{"player_id": "p-1", "side": "A", "slot": 0, "track_label": 999}]
        self.assertEqual(_pose_keyframes(CTX, self.cache, tracks, mps, None), {})


if __name__ == "__main__":
    unittest.main()
