"""Tests for the --frames range parser (worker/eval_shuttle.parse_frame_range).

Pure parsing: "A:B" → (A, B) with inclusive non-negative bounds and A <= B. Garbage raises ValueError so
the CLI fails loudly before any heavy render work; absent input is None (full clip). No torch needed.
"""
import unittest

from worker.eval_shuttle import parse_frame_range


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
