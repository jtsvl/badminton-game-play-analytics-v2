"""Tests for the --frames range parser (worker/eval_shuttle.parse_frame_range).

Pure parsing: "A:B" → (A, B) with inclusive non-negative bounds and A <= B. Garbage raises ValueError so
the CLI fails loudly before any heavy render work; absent input is None (full clip). No torch needed.
"""
import tempfile
import unittest
from pathlib import Path

from worker.eval_shuttle import clip_to_range, parse_frame_range


class TestClipToRange(unittest.TestCase):
    """--frames must scope the bench/--labels run, not just --render: clip rows (detections or labels)
    to the inclusive [a,b] span so a label outside the scoped tracker run is not counted as a miss the
    tracker never had a chance to find."""

    ROWS = [{"frame": 5}, {"frame": 10}, {"frame": 15}, {"frame": 20}]

    def test_clips_inclusive(self):
        self.assertEqual(clip_to_range(self.ROWS, (10, 15)), [{"frame": 10}, {"frame": 15}])

    def test_none_range_passes_through(self):
        self.assertEqual(clip_to_range(self.ROWS, None), self.ROWS)

    def test_empty_when_disjoint(self):
        self.assertEqual(clip_to_range(self.ROWS, (100, 200)), [])

    def test_handles_string_frame_keys(self):
        rows = [{"frame": "5"}, {"frame": "10"}]
        self.assertEqual(clip_to_range(rows, (8, 12)), [{"frame": "10"}])


class TestCompareCopyName(unittest.TestCase):
    """`--compare` keeps every backend's detections as shuttle.<provenance>.parquet (the parquet's
    `tracker` value, e.g. tracknetv5-lite@c7ca025e) so eval_rally --shuttle-parquet can A/B them and the
    last backend (fake) no longer replaces the cache's shuttle.parquet."""

    def _write(self, path, tracker_values):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars (ml extra) not installed")
        rows = [{"frame": i, "tracker": t} for i, t in enumerate(tracker_values)]
        pl.DataFrame(rows, schema={"frame": pl.Int64, "tracker": pl.Utf8}).write_parquet(path)

    def test_uses_the_parquets_provenance(self):
        from worker.eval_shuttle import compare_copy_name

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shuttle.parquet"
            self._write(p, ["tracknetv5-lite@c7ca025e", "tracknetv5-lite@c7ca025e"])
            self.assertEqual(compare_copy_name(p, "tracknetv5"), "shuttle.tracknetv5-lite@c7ca025e.parquet")

    def test_falls_back_to_the_backend_key_when_empty_or_missing(self):
        from worker.eval_shuttle import compare_copy_name

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shuttle.parquet"
            self._write(p, [])
            self.assertEqual(compare_copy_name(p, "wasb"), "shuttle.wasb.parquet")
            self.assertEqual(compare_copy_name(Path(d) / "nope.parquet", "fake"), "shuttle.fake.parquet")

    def test_sanitises_odd_characters(self):
        from worker.eval_shuttle import compare_copy_name

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shuttle.parquet"
            self._write(p, ["weird name/with spaces"])
            self.assertEqual(compare_copy_name(p, "x"), "shuttle.weird_name_with_spaces.parquet")


class TestParseFrameRange(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_frame_range("100:200"), (100, 200))

    def test_equal_bounds_ok(self):
        self.assertEqual(parse_frame_range("42:42"), (42, 42))

    def test_zero_start(self):
        self.assertEqual(parse_frame_range("0:5"), (0, 5))

    def test_absent_is_none(self):
        self.assertIsNone(parse_frame_range(None))
        self.assertIsNone(parse_frame_range(""))

    def test_start_after_end_rejected(self):
        with self.assertRaises(ValueError):
            parse_frame_range("200:100")

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            parse_frame_range("-1:100")

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            parse_frame_range("abc")

    def test_wrong_arity_rejected(self):
        with self.assertRaises(ValueError):
            parse_frame_range("1:2:3")
        with self.assertRaises(ValueError):
            parse_frame_range("100")


if __name__ == "__main__":
    unittest.main()
