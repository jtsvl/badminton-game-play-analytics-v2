"""Bounded-memory chunked shuttle prediction (specs/features/shuttle-tracking.md).

Both backends window their input WITHOUT overlap (TrackNetV3: sliding_step=seq_len=8; WASB:
frames_in=3), so chunking the clip at window-aligned boundaries reproduces the full-clip window
grouping exactly — same predictions, a fraction of the memory. `chunk_spans` is the pure helper that
guarantees the alignment; FakeTracker proves the per-clip state (begin_clip) survives chunking; the
stage-level test proves chunked and single-pass runs write identical shuttle.parquet rows (needs
cv2 + polars, so it skips outside the ml env — it runs on the Mac Studio).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from worker.shuttle.base import FakeTracker, chunk_spans


class TestChunkSpans(unittest.TestCase):
    def test_covers_everything_without_overlap(self):
        spans = chunk_spans(1000, 240, 8)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 1000)
        for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
            self.assertEqual(b0, a1)  # contiguous, no gap, no overlap
            self.assertLess(a0, b0)

    def test_boundaries_are_window_aligned(self):
        for align in (3, 8):
            for a, _b in chunk_spans(1000, 250, align)[1:]:
                self.assertEqual(a % align, 0)

    def test_chunk_size_rounds_down_to_align_multiple(self):
        # 250 % 8 = 2 -> effective 248
        self.assertEqual(chunk_spans(500, 250, 8)[0], (0, 248))

    def test_chunk_smaller_than_align_still_gets_one_window(self):
        self.assertEqual(chunk_spans(20, 2, 8), [(0, 8), (8, 16), (16, 20)])

    def test_zero_or_oversized_chunk_is_a_single_span(self):
        self.assertEqual(chunk_spans(100, 0, 8), [(0, 100)])
        self.assertEqual(chunk_spans(100, -1, 8), [(0, 100)])
        self.assertEqual(chunk_spans(100, 100, 8), [(0, 100)])
        self.assertEqual(chunk_spans(100, 5000, 8), [(0, 100)])

    def test_align_one_ragged_tail(self):
        self.assertEqual(chunk_spans(10, 4, 1), [(0, 4), (4, 8), (8, 10)])

    def test_empty_clip(self):
        self.assertEqual(chunk_spans(0, 240, 8), [])


class _Loader:
    """Just enough loader surface for begin_clip (frame ids + length)."""

    def __init__(self, ids):
        self.frame_ids = list(ids)

    def __len__(self):
        return len(self.frame_ids)


class TestFakeTrackerChunked(unittest.TestCase):
    def test_chunked_equals_unchunked_after_begin_clip(self):
        ids = list(range(100, 140))  # frame ids need not start at 0
        crops = [(i, None) for i in ids]

        whole = FakeTracker()
        whole.begin_clip(_Loader(ids))
        full = whole.predict(crops)

        chunked = FakeTracker()
        chunked.begin_clip(_Loader(ids))
        parts = []
        for a, b in chunk_spans(len(crops), 16, chunked.window):
            parts.extend(chunked.predict(crops[a:b]))

        self.assertEqual(full, parts)  # dataclass equality: frame, x, y, visibility

    def test_without_begin_clip_keeps_legacy_per_call_behavior(self):
        crops = [(i, None) for i in range(40)]
        a = FakeTracker().predict(crops)
        b = FakeTracker().predict(crops)
        self.assertEqual(a, b)
        # phase spans the whole call: last x well right of first
        self.assertGreater(a[-1].x, a[0].x)

    def test_backends_declare_their_window(self):
        self.assertEqual(FakeTracker().window, 1)


class _Cfg:
    def __init__(self, cache_dir: str):
        self.frame_cache_dir = cache_dir
        self.artifact_bucket = "analysis-artifacts"


class TestStageChunkedIdentity(unittest.TestCase):
    """Chunked vs single-pass stage runs must write identical parquet rows (FakeTracker, no weights)."""

    N_FRAMES = 40
    W, H = 320, 180

    def _make_cache(self, root: str, video_id: str) -> None:
        import cv2
        import numpy as np

        frames_dir = Path(root) / video_id / "frames"
        frames_dir.mkdir(parents=True)
        for i in range(self.N_FRAMES):
            img = np.full((self.H, self.W, 3), (i * 3) % 255, dtype="uint8")
            cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), img)
        manifest = {"width": self.W, "height": self.H,
                    "frames": [{"frame": i, "t": i / 12.0} for i in range(self.N_FRAMES)]}
        (Path(root) / video_id / "frames.json").write_text(json.dumps(manifest))

    def _run_stage(self, root: str, video_id: str, chunk_frames: int) -> list:
        import polars as pl

        from worker import geometry as g
        from worker.pipeline import shuttle as stage

        corners = [(60.0, 40.0), (260.0, 40.0), (300.0, 160.0), (20.0, 160.0)]  # FL,FR,NR,NL
        ctx = {"cfg": _Cfg(root), "job": {"video_id": video_id},
               "H": g.homography_from_corners(corners), "mask_poly_px": corners}
        old_chunk, old_tracker = stage.CHUNK_FRAMES, os.environ.get("SHUTTLE_TRACKER")
        stage.CHUNK_FRAMES = chunk_frames
        os.environ["SHUTTLE_TRACKER"] = "fake"
        try:
            stage.run(ctx)
        finally:
            stage.CHUNK_FRAMES = old_chunk
            if old_tracker is None:
                os.environ.pop("SHUTTLE_TRACKER", None)
            else:
                os.environ["SHUTTLE_TRACKER"] = old_tracker
        return pl.read_parquet(ctx["shuttle_path"]).to_dicts()

    def test_chunked_parquet_equals_single_pass(self):
        try:
            import cv2  # noqa: F401
            import polars  # noqa: F401
        except ImportError:
            self.skipTest("needs cv2 + polars (ml extra)")
        with tempfile.TemporaryDirectory() as root:
            self._make_cache(root, "vid-chunk")
            chunked = self._run_stage(root, "vid-chunk", chunk_frames=16)
            single = self._run_stage(root, "vid-chunk", chunk_frames=0)
        self.assertGreater(len(single), 0)
        self.assertEqual(chunked, single)


if __name__ == "__main__":
    unittest.main()
