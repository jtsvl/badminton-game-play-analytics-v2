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
    def test_remap_runs_rally_sync_aggregate_overlay_then_done(self):
        # rally_sync in the remap slice: identity recomputes keep the DB-authoritative rally spans
        # (fresh-cache runs used to silently fall back to whole-clip gating in aggregate).
        self.assertEqual(
            _slice_for({"stage": "remap"}),
            (["remap", "rally_sync", "aggregate", "overlay"], "done"),
        )

    def test_remap_returns_to_prior_status(self):
        # Backfill re-runs (e.g. pose_keyframes) pass the match's current status so a 'review'
        # match is not silently flipped past the PlayerReview gate.
        self.assertEqual(
            _slice_for({"stage": "remap", "params": {"return_status": "review"}}),
            (["remap", "rally_sync", "aggregate", "overlay"], "review"),
        )

    def test_remap_ignores_null_params(self):
        self.assertEqual(
            _slice_for({"stage": "remap", "params": None}),
            (["remap", "rally_sync", "aggregate", "overlay"], "done"),
        )

    def test_rally_review_slice_defaults_to_done(self):
        # No overlay: rally edits change nothing overlay produces (it never reads rallies).
        self.assertEqual(
            _slice_for({"stage": "rally_review"}),
            (["remap", "rally_sync", "aggregate"], "done"),
        )

    def test_rally_review_returns_to_prior_status(self):
        # A trainer editing rallies at status 'review' must land back at 'review',
        # not skip the PlayerReview confirmation.
        self.assertEqual(
            _slice_for({"stage": "rally_review", "params": {"return_status": "review"}}),
            (["remap", "rally_sync", "aggregate"], "review"),
        )

    def test_rally_review_ignores_null_params(self):
        self.assertEqual(
            _slice_for({"stage": "rally_review", "params": None}),
            (["remap", "rally_sync", "aggregate"], "done"),
        )

    def test_return_status_unchanged_leaves_match_status_untouched(self):
        # 'unchanged' → end None (like standalone proxy jobs): a backfill that races a trainer
        # confirmation must not overwrite the newer status with its enqueue-time snapshot.
        stages, end = _slice_for({"stage": "remap", "params": {"return_status": "unchanged"}})
        self.assertEqual(stages, ["remap", "rally_sync", "aggregate", "overlay"])
        self.assertIsNone(end)

    def test_return_status_rejects_unknown_values(self):
        # A typo'd status would hide the match from every status-gated view — fall back to 'done'.
        for bogus in ("reviewed", "calibrating", "failed", ""):
            with self.subTest(bogus=bogus):
                _, end = _slice_for({"stage": "remap", "params": {"return_status": bogus}})
                self.assertEqual(end, "done")

    def test_retrack_reruns_gate_track_and_downstream(self):
        # Stitching improvements need gate_track to re-run (tracks.parquet is rebuilt), then the
        # same downstream as remap. No re-detect: detections.parquet resolves from cache/bucket.
        self.assertEqual(
            _slice_for({"stage": "retrack"}),
            (["gate_track", "rally_sync", "aggregate", "overlay"], "done"),
        )

    def test_retrack_returns_to_prior_status(self):
        self.assertEqual(
            _slice_for({"stage": "retrack", "params": {"return_status": "review"}}),
            (["gate_track", "rally_sync", "aggregate", "overlay"], "review"),
        )

    def test_ingest_slice_includes_proxy(self):
        # proxy rides along on ingest so every new upload gets a playback proxy; it never
        # raises, so it cannot fail the ingest job (specs/features/video-proxy.md).
        self.assertEqual(_slice_for({"stage": "ingest"}), (["ingest", "proxy"], "calibrating"))

    def test_default_stage_is_ingest(self):
        self.assertEqual(_slice_for({}), (["ingest", "proxy"], "calibrating"))

    def test_proxy_backfill_runs_standalone_with_status_untouched(self):
        # Backfill jobs (stage='proxy') must not move match.status — end_status None.
        self.assertEqual(_slice_for({"stage": "proxy"}), (["proxy"], None))

    def test_calibrate_slice_unchanged(self):
        self.assertEqual(_slice_for({"stage": "calibrate"}), (RESUME_STAGES, "review"))
        self.assertEqual(_slice_for({"stage": "resume_after_calibration"}), (RESUME_STAGES, "review"))

    def test_unknown_stage_runs_only_itself(self):
        self.assertEqual(_slice_for({"stage": "detect"}), (["detect"], None))

    def test_remap_is_a_registered_runnable_stage(self):
        self.assertIn("remap", STAGES)

    def test_rally_sync_is_a_registered_runnable_stage(self):
        self.assertIn("rally_sync", STAGES)

    def test_proxy_is_a_registered_runnable_stage(self):
        # An unregistered stage name is a KeyError at claim time — backfill jobs need this.
        self.assertIn("proxy", STAGES)


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


class TestResolveArtifact(unittest.TestCase):
    """remap.resolve_artifact — the shared cache-hit / bucket-download / None resolver that
    gate_track (detections.parquet, retrack) and overlay (pose.parquet) reuse."""

    def test_cache_hit_skips_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "vid-1"
            cache.mkdir(parents=True)
            (cache / "detections.parquet").write_bytes(b"cached")
            store = _Store(payload=None)
            path = remap.resolve_artifact(store, _Cfg(tmp), "vid-1", cache, "detections.parquet")
            self.assertEqual(path, str(cache / "detections.parquet"))
            self.assertIsNone(store.downloaded)

    def test_downloads_on_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "vid-2"
            store = _Store(payload=b"from-s3")
            path = remap.resolve_artifact(store, _Cfg(tmp), "vid-2", cache, "pose.parquet")
            self.assertEqual(path, str(cache / "pose.parquet"))
            self.assertEqual(Path(path).read_bytes(), b"from-s3")
            self.assertEqual(store.downloaded, ("analysis-artifacts", "vid-2/pose.parquet"))

    def test_missing_everywhere_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(payload=None)
            cache = Path(tmp) / "vid-3"
            self.assertIsNone(
                remap.resolve_artifact(store, _Cfg(tmp), "vid-3", cache, "pose.parquet"))


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
