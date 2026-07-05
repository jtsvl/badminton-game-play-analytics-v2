"""stroke.dataset — label→sample assembly for the stage-7 training set.

TDD'd: the pure slot-coverage report (the Phase-1 gate numbers: is the labeled hitter's slot
filled, and from which source) and the per-video assembly against synthetic parquets in a temp
cache dir (schema-drift guard, mirroring test_slice_for's remap round-trip). The CLI itself is
smoke-tested on the real 5-game export (features/stroke-classification.md Phase 1).
"""
import json
import tempfile
import unittest
from pathlib import Path

from worker.stroke.dataset import slot_report

IDENTITY_H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


class TestSlotReport(unittest.TestCase):
    """Gate numbers: labeled hitter's slot filled (split by track/recovery source) +
    per-slot fill rates over all rows."""

    @staticmethod
    def _row(hitter="near_left", **slots):
        base = {f"track_{s}": None for s in
                ("near_left", "near_right", "far_left", "far_right")}
        base.update({f"source_{s}": None for s in
                     ("near_left", "near_right", "far_left", "far_right")})
        base["hitter"] = hitter
        base.update(slots)
        return base

    def test_hitter_slot_filled_split_by_source(self):
        rows = [
            self._row(track_near_left=47, source_near_left="track"),      # filled via track
            self._row(track_near_left=26, source_near_left="recovery"),   # filled via recovery
            self._row(),                                                   # hitter slot empty
            self._row(hitter=None, track_far_left=9, source_far_left="track"),  # unlabeled
        ]
        rep = slot_report(rows)
        self.assertEqual(rep["n_hitter_labeled"], 3)
        self.assertEqual(rep["hitter_slot_filled"], 2)
        self.assertAlmostEqual(rep["hitter_slot_filled_rate"], 2 / 3)
        self.assertEqual(rep["hitter_slot_by_source"], {"track": 1, "recovery": 1})

    def test_per_slot_fill_rates_cover_all_rows(self):
        rows = [self._row(track_near_left=1, source_near_left="track"),
                self._row(hitter=None)]
        rep = slot_report(rows)
        self.assertAlmostEqual(rep["slot_fill_rates"]["near_left"], 0.5)
        self.assertAlmostEqual(rep["slot_fill_rates"]["far_right"], 0.0)

    def test_empty_rows_report_zero_not_crash(self):
        rep = slot_report([])
        self.assertEqual(rep["n_hitter_labeled"], 0)
        self.assertIsNone(rep["hitter_slot_filled_rate"])


class TestBuildVideoSamples(unittest.TestCase):
    def test_assembles_samples_with_recovery(self):
        import polars as pl

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            frames = [{"frame": i, "t": i / 12.0} for i in range(240)]
            (cache / "frames.json").write_text(json.dumps(
                {"width": 3840, "height": 2160, "frames": frames}))
            kp = [[900.0, 1800.0]] * 26
            pose_rows = (
                # near_left player, carried by a court track
                [{"frame": f, "track": 47, "kp": kp, "kp_score": [0.9] * 26,
                  "foot_x": 0.3, "foot_y": 0.8, "foot_conf": 0.9, "foot_source": "ankle"}
                 for f in range(90, 111)]
                # far_right player, lost by the court tracks — pose recovery only
                + [{"frame": f, "track": 26, "kp": kp, "kp_score": [0.9] * 26,
                    "foot_x": 0.7, "foot_y": 0.2, "foot_conf": 0.9, "foot_source": "ankle"}
                   for f in range(90, 111)])
            pl.DataFrame(pose_rows, schema={
                "frame": pl.Int64, "track": pl.Int64, "kp": pl.List(pl.List(pl.Float64)),
                "kp_score": pl.List(pl.Float64), "foot_x": pl.Float64, "foot_y": pl.Float64,
                "foot_conf": pl.Float64, "foot_source": pl.Utf8,
            }).write_parquet(cache / "pose.parquet")
            pl.DataFrame(
                [{"t": f / 12.0, "track_id": 47, "u": 0.3, "v": 0.8, "role": "near_left",
                  "source": "detected"} for f in range(90, 111)],
                schema={"t": pl.Float64, "track_id": pl.Int64, "u": pl.Float64,
                        "v": pl.Float64, "role": pl.Utf8, "source": pl.Utf8},
            ).write_parquet(cache / "tracks.parquet")
            pl.DataFrame(
                [{"t": f / 12.0, "u": 0.5, "v": 0.5, "visibility": 0.9, "frame": f,
                  "px": 900.0, "py": 500.0, "conf": 0.9, "tracker": "tracknetv3"}
                 for f in range(93, 108)],
                schema={"t": pl.Float64, "u": pl.Float64, "v": pl.Float64,
                        "visibility": pl.Float64, "frame": pl.Int64, "px": pl.Float64,
                        "py": pl.Float64, "conf": pl.Float64, "tracker": pl.Utf8},
            ).write_parquet(cache / "shuttle.parquet")

            from worker.stroke.dataset import build_video_samples
            labels = [{"event_id": "e1", "match_id": "m1", "video_id": "v1",
                       "t_s": 100 / 12.0, "shot_type": "drop", "hitter": "far_right",
                       "rally_idx": 0, "hit_idx": 1}]
            samples, meta = build_video_samples(cache, labels, homography=IDENTITY_H)

            self.assertEqual(len(samples), 1)
            self.assertEqual(len(samples[0]["joints"]), 4)
            self.assertEqual(meta[0]["shot_type"], "drop")
            self.assertEqual(meta[0]["track_near_left"], 47)
            self.assertEqual(meta[0]["source_near_left"], "track")
            self.assertEqual(meta[0]["track_far_right"], 26)   # recovered
            self.assertEqual(meta[0]["source_far_right"], "recovery")
            self.assertFalse(meta[0]["is_calibration"])
            self.assertEqual(meta[0]["anchor"], 100)


if __name__ == "__main__":
    unittest.main()
