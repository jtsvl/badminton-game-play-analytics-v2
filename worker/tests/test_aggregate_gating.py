"""aggregate: gating player movement to in-rally spans (Workstream C).

The pure piece is filter_to_spans — keep only track rows whose seconds timestamp falls inside any rally
[t_start_s, t_end_s] span (inclusive). The whole-clip fallback (no rallies → unfiltered) lives in run() and
is exercised by the stage run; here we lock the span filter.
"""
import unittest

import polars as pl

from worker.pipeline.aggregate import filter_to_spans


def _tracks(ts):
    return pl.DataFrame({"t": ts, "track_id": [1] * len(ts),
                         "u": [0.5] * len(ts), "v": [0.5] * len(ts)})


class TestFilterToSpans(unittest.TestCase):
    def test_keeps_only_rows_inside_any_span(self):
        df = _tracks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 8.5])
        out = filter_to_spans(df, [(2.0, 4.0), (7.0, 8.0)])
        self.assertEqual(out["t"].to_list(), [2.0, 3.0, 4.0, 7.5])

    def test_bounds_are_inclusive(self):
        df = _tracks([1.9, 2.0, 4.0, 4.1])
        out = filter_to_spans(df, [(2.0, 4.0)])
        self.assertEqual(out["t"].to_list(), [2.0, 4.0])

    def test_no_overlap_yields_empty(self):
        df = _tracks([1.0, 2.0])
        out = filter_to_spans(df, [(10.0, 20.0)])
        self.assertEqual(out.height, 0)


if __name__ == "__main__":
    unittest.main()
