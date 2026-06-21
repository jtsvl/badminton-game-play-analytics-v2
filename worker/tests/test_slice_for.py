"""Tests for __main__._slice_for — the job.stage → (stages, end_status) gate map.

Guards the FR-35 `remap` slice (cheap identity-recompute: [remap, aggregate, overlay] → 'done') and
proves the existing ingest/calibrate slices are unchanged. Also covers remap.run resolving the cached
tracks.parquet (cache hit + S3-download fallback) without re-detecting.
"""
import tempfile
import unittest
from pathlib import Path

from worker.__main__ import RESUME_STAGES, _slice_for
from worker.pipeline import STAGES
from worker.pipeline import remap


class TestSliceFor(unittest.TestCase):
    def test_remap_runs_aggregate_overlay_then_done(self):
        self.assertEqual(_slice_for({"stage": "remap"}), (["remap", "aggregate", "overlay"], "done"))

    def test_ingest_slice_unchanged(self):
        self.assertEqual(_slice_for({"stage": "ingest"}), (["ingest"], "calibrating"))

    def test_default_stage_is_ingest(self):
        self.assertEqual(_slice_for({}), (["ingest"], "calibrating"))

    def test_calibrate_slice_unchanged(self):
        self.assertEqual(_slice_for({"stage": "calibrate"}), (RESUME_STAGES, "review"))
        self.assertEqual(_slice_for({"stage": "resume_after_calibration"}), (RESUME_STAGES, "review"))

    def test_unknown_stage_runs_only_itself(self):
        self.assertEqual(_slice_for({"stage": "detect"}), (["detect"], None))

    def test_remap_is_a_registered_runnable_stage(self):
        self.assertIn("remap", STAGES)


class _Cfg:
    def __init__(self, cache_dir: str, artifact_bucket: str = "analysis-artifacts"):
        self.frame_cache_dir = cache_dir
        self.artifact_bucket = artifact_bucket


class _Store:
    """Minimal fake: records the download it's asked for and returns canned bytes (or None)."""

    def __init__(self, payload: bytes | None):
        self._payload = payload
        self.downloaded: tuple[str, str] | None = None

    def download(self, bucket: str, path: str) -> bytes | None:
        self.downloaded = (bucket, path)
        return self._payload


class TestRemapRun(unittest.TestCase):
    def test_uses_cached_parquet_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_id = "vid-1"
            cache = Path(tmp) / video_id
            cache.mkdir(parents=True)
            (cache / "tracks.parquet").write_bytes(b"cached")
            store = _Store(payload=None)
            ctx = {"store": store, "cfg": _Cfg(tmp), "job": {"video_id": video_id}}
            remap.run(ctx)
            self.assertEqual(ctx["tracks_path"], str(cache / "tracks.parquet"))
            self.assertIsNone(store.downloaded)  # cache hit → no S3 call

    def test_downloads_from_artifact_bucket_on_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_id = "vid-2"
            store = _Store(payload=b"from-s3")
            ctx = {"store": store, "cfg": _Cfg(tmp), "job": {"video_id": video_id}}
            remap.run(ctx)
            restored = Path(tmp) / video_id / "tracks.parquet"
            self.assertEqual(ctx["tracks_path"], str(restored))
            self.assertEqual(restored.read_bytes(), b"from-s3")
            self.assertEqual(store.downloaded, ("analysis-artifacts", f"{video_id}/tracks.parquet"))

    def test_raises_when_no_parquet_anywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(payload=None)
            ctx = {"store": store, "cfg": _Cfg(tmp), "job": {"video_id": "vid-3"}}
            with self.assertRaises(RuntimeError):
                remap.run(ctx)


if __name__ == "__main__":
    unittest.main()
