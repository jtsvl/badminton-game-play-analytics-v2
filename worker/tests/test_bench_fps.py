"""bench-fps — the non-destructive 12-vs-30 fps measurement driver (worker/worker/bench_fps.py).

Pure parts only: the stage plan (skip ingest when a matching frame cache can be reused), the read-only
store wrapper (every write is a counted no-op, reads delegate), the production-cache refusal and the
cache size walk. The stages themselves run only on the Mac (specs/features/analysis-30fps.md).
"""
import json
import tempfile
import unittest
from pathlib import Path

from worker import bench_fps as bf


class _RealStore:
    """Stands in for Store/JobStore: reads return data, writes would be destructive."""

    def __init__(self):
        self.writes = []

    def get_video(self, video_id):
        return {"id": video_id, "match_id": "m1", "storage_path": "x.mp4"}

    def get_calibration(self, video_id):
        return {"homography": [1] * 9, "mask_polygon_px": [[0, 0], [1, 0], [1, 1], [0, 1]]}

    def get_match(self, match_id):
        return {"id": match_id, "discipline": "doubles"}

    def get_match_players(self, match_id):
        return []

    def upload(self, *a, **k):
        self.writes.append("upload")

    def replace_rallies(self, *a, **k):
        self.writes.append("replace_rallies")
        return 0

    def set_track_label(self, *a, **k):
        self.writes.append("set_track_label")

    def update_calibration(self, *a, **k):
        self.writes.append("update_calibration")


class TestReadOnlyStore(unittest.TestCase):
    def test_reads_delegate(self):
        real = _RealStore()
        ro = bf.ReadOnlyStore(real)
        self.assertEqual(ro.get_video("v")["match_id"], "m1")
        self.assertEqual(ro.get_match("m1")["discipline"], "doubles")
        self.assertEqual(ro.get_match_players("m1"), [])

    def test_every_write_is_a_counted_noop(self):
        real = _RealStore()
        ro = bf.ReadOnlyStore(real)
        ro.upload("bucket", "path", b"...", "application/octet-stream")
        ro.update_video_meta("v", 30.0, 10.0, 3840, 2160)
        self.assertEqual(ro.insert_calibration({"video_id": "v"})["id"], "bench-stub")
        ro.update_calibration("cal", {})
        self.assertEqual(ro.replace_rallies("m1", [], []), 0)
        ro.set_track_label("mp", 1)
        ro.upsert_player_match_stats({})
        ro.update_match_status("m1", "done")
        self.assertEqual(real.writes, [])                     # nothing reached the real store
        self.assertEqual(ro.blocked["upload"], 1)
        self.assertEqual(ro.blocked["replace_rallies"], 1)
        self.assertEqual(sum(ro.blocked.values()), 8)

    def test_unknown_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            bf.ReadOnlyStore(_RealStore()).not_a_store_method()

    def test_write_names_cover_every_mutating_store_method(self):
        # the real Store's mutating surface — a new write method must be added here, never slip through
        from worker import store as st

        mutating = {n for n in dir(st.Store) if not n.startswith("_") and n.split("_")[0] in
                    ("upload", "update", "insert", "upsert", "delete", "replace", "set", "create",
                     "enqueue", "finish", "log", "claim")}
        self.assertTrue(mutating <= bf.WRITE_METHODS, mutating - bf.WRITE_METHODS)


class TestPlanStages(unittest.TestCase):
    def test_default_plan_runs_the_cadence_dependent_stages(self):
        self.assertEqual(bf.plan_stages(bf.DEFAULT_STAGES, reuse_frames=False, frames_ok=False),
                         ["ingest", "detect", "pose", "gate_track", "shuttle", "rally_hit"])

    def test_reuse_skips_ingest_only_when_a_matching_cache_exists(self):
        self.assertEqual(bf.plan_stages(bf.DEFAULT_STAGES, reuse_frames=True, frames_ok=True)[0], "detect")
        self.assertEqual(bf.plan_stages(bf.DEFAULT_STAGES, reuse_frames=True, frames_ok=False)[0], "ingest")

    def test_explicit_stage_list_is_validated(self):
        self.assertEqual(bf.plan_stages(["shuttle", "rally_hit"], reuse_frames=False, frames_ok=False),
                         ["shuttle", "rally_hit"])
        with self.assertRaises(ValueError):
            bf.plan_stages(["ingest", "aggregate"], reuse_frames=False, frames_ok=False)  # writes stats → no


class TestFramesMatch(unittest.TestCase):
    def test_matching_cache_is_detected_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "vid"
            cache.mkdir()
            (cache / "frames.json").write_text(json.dumps({"analysis_fps": 30.0, "frames": [{"frame": 0, "t": 0.0}]}))
            self.assertTrue(bf.frames_match(cache, 30.0))
            self.assertFalse(bf.frames_match(cache, 12.0))
            self.assertFalse(bf.frames_match(Path(d) / "missing", 30.0))


class TestProductionGuard(unittest.TestCase):
    def test_refuses_the_configured_and_default_cache_paths(self):
        with tempfile.TemporaryDirectory() as d:
            prod = Path(d) / "worker" / ".cache"
            prod.mkdir(parents=True)
            self.assertTrue(bf.is_production_cache(prod, prod))
            self.assertTrue(bf.is_production_cache(Path(d) / "worker" / ".cache", Path(d) / "elsewhere"))
            self.assertFalse(bf.is_production_cache(Path(d) / "worker" / ".cache-30fps", prod))


class TestDirSize(unittest.TestCase):
    def test_counts_all_bytes_below_the_root(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a").mkdir()
            (Path(d) / "a" / "x.bin").write_bytes(b"x" * 10)
            (Path(d) / "y.bin").write_bytes(b"y" * 5)
            self.assertEqual(bf.dir_size_bytes(Path(d)), 15)
            self.assertEqual(bf.dir_size_bytes(Path(d) / "nope"), 0)


class TestBenchRecord(unittest.TestCase):
    def test_record_shape(self):
        rec = bf.bench_record("vid", 30.0, ["detect"], {"detect": 12.5}, cache_bytes=10_240_000_000,
                              n_frames=10, cadence_fps=29.97)
        for key in ("video_id", "analysis_fps", "cadence_fps", "stages", "stage_seconds", "total_seconds",
                    "cache_bytes", "cache_gb", "n_frames", "recorded_at"):
            self.assertIn(key, rec)
        self.assertAlmostEqual(rec["total_seconds"], 12.5)
        self.assertAlmostEqual(rec["cache_gb"], 10.24)


if __name__ == "__main__":
    unittest.main()
