"""stroke.traj — hit-to-hit trajectory-geometry features for the GBDT spike.

Torch-free by design (spec: "Alternative candidate: trajectory-geometry GBDT"): every feature is
plain stdlib math on row dicts, so these tests run on light installs; polars appears only inside
the exporter-assembly test (lazy import, precedent test_stroke_dataset.py). The contract under
test: per labeled hit one feature row — 24 core + 3 ppos columns, degradation per-feature (never
per-row), no interpolation, court coords only at anchor points, no hitter attribution anywhere.
"""
import json
import tempfile
import unittest
from pathlib import Path

from worker.stroke.traj import (
    FEATURE_COLS,
    PPOS_COLS,
    build_feature_row,
    build_video_rows,
    rows_to_frame,
)

IDENTITY_H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _label(t_s, shot_type="drop", **over):
    d = {"event_id": "e1", "match_id": "m1", "video_id": "v1", "t_s": t_s,
         "shot_type": shot_type, "hitter": None, "rally_idx": 0, "hit_idx": 1}
    d.update(over)
    return d


def _hit(t):
    return {"rally_idx": 0, "t_s": t, "type": "hit", "payload": "{}", "confidence": 0.3}


def _rally(idx=0, start=0.0, end=10.0):
    return {"idx": idx, "t_start_s": start, "t_end_s": end}


def _det(t, px, py, conf=0.9):
    return {"t": t, "frame": round(t * 12), "px": px, "py": py, "conf": conf}


def _row(label, *, events, rallies, shuttle, tracks=None, frame_w=1.0, frame_h=1.0,
         homography=IDENTITY_H):
    return build_feature_row(label, events=events, rallies=rallies, shuttle=shuttle,
                             tracks=tracks, frame_w=frame_w, frame_h=frame_h,
                             homography=homography)


class TestConstants(unittest.TestCase):
    def test_column_groups_are_locked(self):
        self.assertEqual(len(FEATURE_COLS), 24)
        self.assertEqual(len(PPOS_COLS), 3)
        self.assertTrue(all(c.startswith("ppos_") for c in PPOS_COLS))
        self.assertFalse(set(FEATURE_COLS) & set(PPOS_COLS))


class TestBoundaries(unittest.TestCase):
    """Segment boundaries come from detected hits within the rally span (hits_in_rallies
    semantics), falling back to rally_end — labeled videos have no landing events."""

    def test_boundary_prefers_next_hit_over_rally_end(self):
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.6)], rallies=[_rally(end=3.0)],
                 shuttle=[_det(1.0, 0.3, 0.3)])
        self.assertAlmostEqual(r["flight_time_out"], 0.6)
        self.assertEqual(r["prov_out_boundary"], "hit")
        self.assertEqual(r["out_end_is_rally_end"], 0.0)
        self.assertEqual(r["first_of_rally"], 1.0)
        self.assertEqual(r["hit_idx_in_rally"], 1.0)
        self.assertEqual(r["prov_hit_idx_recomputed"], 1)

    def test_last_hit_boundary_is_rally_end(self):
        r = _row(_label(1.0), events=[_hit(1.0)], rallies=[_rally(end=3.0)],
                 shuttle=[_det(1.0, 0.3, 0.3)])
        self.assertAlmostEqual(r["flight_time_out"], 2.0)
        self.assertEqual(r["prov_out_boundary"], "rally_end")
        self.assertEqual(r["out_end_is_rally_end"], 1.0)

    def test_first_hit_incoming_features_absent(self):
        events = [_hit(1.0), _hit(1.6)]
        first = _row(_label(1.0), events=events, rallies=[_rally()],
                     shuttle=[_det(1.0, 0.3, 0.3)])
        for col in ("flight_time_in", "in_dv_abs", "in_speed_uv", "in_apex_rise", "n_dets_in"):
            self.assertIsNone(first[col], col)
        second = _row(_label(1.6, hit_idx=2), events=events, rallies=[_rally()],
                      shuttle=[_det(1.0, 0.3, 0.3), _det(1.6, 0.3, 0.7)])
        self.assertEqual(second["first_of_rally"], 0.0)
        self.assertAlmostEqual(second["flight_time_in"], 0.6)

    def test_rally_not_found_row_still_emitted(self):
        r = _row(_label(20.0), events=[_hit(1.0)], rallies=[_rally(end=10.0)],
                 shuttle=[_det(20.0, 0.3, 0.3)])
        self.assertFalse(r["prov_rally_found"])
        self.assertIsNone(r["prov_hit_idx_recomputed"])
        for col in ("first_of_rally", "flight_time_out", "n_dets_out"):
            self.assertIsNone(r[col], col)
        for col in FEATURE_COLS + PPOS_COLS:  # degradation is per-feature, never per-row
            self.assertIn(col, r)

    def test_zero_duration_segment_never_divides(self):
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.0)], rallies=[_rally()],
                 shuttle=[_det(1.0, 0.3, 0.3)])
        self.assertEqual(r["flight_time_out"], 0.0)
        self.assertIsNone(r["out_speed_uv"])
        self.assertIsNone(r["out_apex_t_frac"])
        self.assertIsNone(r["out_py_slope_early"])


class TestAnchor(unittest.TestCase):
    def test_anchor_match_within_tolerance(self):
        events, rallies = [_hit(1.0), _hit(1.6)], [_rally()]
        near = _row(_label(1.0), events=events, rallies=rallies,
                    shuttle=[_det(1.09, 0.3, 0.3)])
        self.assertIsNotNone(near["hit_pxn"])
        self.assertAlmostEqual(near["prov_anchor_dt"], 0.09)
        far = _row(_label(1.0), events=events, rallies=rallies,
                   shuttle=[_det(1.11, 0.3, 0.3)])
        self.assertIsNone(far["hit_pxn"])
        self.assertIsNone(far["hit_dist_net"])
        self.assertIsNone(far["out_apex_rise"])
        self.assertAlmostEqual(far["flight_time_out"], 0.6)  # timing survives without an anchor
        self.assertEqual(far["n_dets_out"], 1.0)             # pure time-window count

    def test_duplicate_t_dedup_keeps_max_conf(self):
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.6)], rallies=[_rally()],
                 shuttle=[_det(1.0, 100.0, 100.0, conf=0.2), _det(1.0, 200.0, 150.0, conf=0.9)],
                 frame_w=1000.0, frame_h=1000.0, homography=None)
        self.assertAlmostEqual(r["hit_pxn"], 0.2)
        self.assertAlmostEqual(r["hit_pyn"], 0.15)


class TestCourtGeometry(unittest.TestCase):
    def test_displacement_and_crossing_identity_h(self):
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()],
                 shuttle=[_det(1.0, 0.3, 0.3), _det(1.5, 0.35, 0.7)])
        self.assertAlmostEqual(r["hit_v_raw"], 0.3)
        self.assertAlmostEqual(r["hit_dist_net"], 0.2)
        self.assertAlmostEqual(r["hit_u_center_dist"], 0.2)
        self.assertAlmostEqual(r["out_end_dist_net"], 0.2)
        self.assertAlmostEqual(r["out_dv_abs"], 0.4)
        self.assertAlmostEqual(r["out_du_abs"], 0.05)
        self.assertEqual(r["out_crossed_net"], 1.0)
        self.assertAlmostEqual(r["out_speed_uv"], ((0.05 ** 2 + 0.4 ** 2) ** 0.5) / 0.5)

    def test_missing_homography_court_none_image_present(self):
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()],
                 shuttle=[_det(1.0, 0.3, 0.3), _det(1.5, 0.35, 0.7)], homography=None)
        self.assertFalse(r["prov_homography_ok"])
        for col in ("hit_v_raw", "hit_dist_net", "out_dv_abs", "out_speed_uv", "in_dv_abs"):
            self.assertIsNone(r[col], col)
        self.assertAlmostEqual(r["hit_pxn"], 0.3)
        self.assertIsNotNone(r["out_apex_rise"])
        self.assertAlmostEqual(r["flight_time_out"], 0.5)

    def test_horizon_endpoint_degrades_not_raises(self):
        # denom = px - 2 -> zero exactly at the endpoint det, fine at the hit det
        h = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -2.0]
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()],
                 shuttle=[_det(1.0, 0.3, 0.3), _det(1.5, 2.0, 0.7)], homography=h)
        self.assertIsNotNone(r["hit_v_raw"])
        for col in ("out_dv_abs", "out_du_abs", "out_crossed_net", "out_speed_uv",
                    "out_end_dist_net"):
            self.assertIsNone(r[col], col)

    def test_out_of_range_endpoint_projection_rejected(self):
        # identity H: endpoint u = 5.0 is outside the [-1, 2] blow-up guard
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()],
                 shuttle=[_det(1.0, 0.3, 0.3), _det(1.5, 5.0, 0.7)])
        self.assertIsNotNone(r["hit_v_raw"])
        self.assertIsNone(r["out_dv_abs"])
        self.assertIsNotNone(r["out_apex_rise"])  # image-space arc survives


class TestArc(unittest.TestCase):
    def test_synthetic_parabola_apex(self):
        # py(t) = 0.8 - 2.4*(t-1)*(2-t): min 0.2 exactly at t=1.5 (a 12 fps sample point)
        dets = [_det(1.0 + i / 12.0, 0.5, 0.8 - 2.4 * (i / 12.0) * (1.0 - i / 12.0))
                for i in range(13)]
        r = _row(_label(1.0), events=[_hit(1.0), _hit(2.0)], rallies=[_rally()], shuttle=dets)
        self.assertAlmostEqual(r["out_apex_rise"], 0.6, places=6)
        self.assertAlmostEqual(r["out_apex_t_frac"], 0.5, places=6)
        self.assertAlmostEqual(r["out_end_drop"], 0.6, places=6)
        self.assertEqual(r["n_dets_out"], 13.0)

    def test_descending_arc_zero_rise_first_argmin(self):
        dets = [_det(1.0, 0.5, 0.3), _det(1.25, 0.5, 0.5), _det(1.5, 0.5, 0.7)]
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()], shuttle=dets)
        self.assertEqual(r["out_apex_rise"], 0.0)   # anchor is the highest point
        self.assertEqual(r["out_apex_t_frac"], 0.0)  # first-argmin tie rule
        self.assertAlmostEqual(r["out_end_drop"], 0.4)

    def test_segment_excludes_out_of_bounds_detections(self):
        dets = [_det(0.5, 0.5, 0.01),   # before the anchor — a different arc
                _det(1.0, 0.5, 0.8), _det(1.25, 0.5, 0.6), _det(1.5, 0.5, 0.7),
                _det(1.9, 0.5, 0.01)]   # after the boundary
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally(end=3.0)],
                 shuttle=dets)
        self.assertAlmostEqual(r["out_apex_rise"], 0.2)  # min inside [1.0, 1.5] is 0.6
        self.assertEqual(r["n_dets_out"], 3.0)

    def test_py_slope_early_from_first_interior_detection(self):
        dets = [_det(1.0, 0.5, 0.5), _det(1.0 + 1 / 12.0, 0.5, 0.4), _det(1.5, 0.5, 0.45)]
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()], shuttle=dets)
        self.assertAlmostEqual(r["out_py_slope_early"], -1.2, places=6)

    def test_incoming_arc_rise_above_previous_hit(self):
        # previous hit at 1.0 (py 0.7), apex 0.2 between, this hit at 1.6
        dets = [_det(1.0, 0.5, 0.7), _det(1.25, 0.5, 0.2), _det(1.6, 0.5, 0.6)]
        r = _row(_label(1.6, hit_idx=2), events=[_hit(1.0), _hit(1.6)], rallies=[_rally()],
                 shuttle=dets)
        self.assertAlmostEqual(r["in_apex_rise"], 0.5)
        self.assertEqual(r["n_dets_in"], 2.0)  # rows with t_prev <= t < t0


class TestPlayerPositions(unittest.TestCase):
    """Label-free scene geometry: distances from the projected contact point to the nearest
    tracked player per half. No who-hit decision anywhere (measured ~50/50 on this footage)."""

    @staticmethod
    def _track(t, u, v, role, track_id=1):
        return {"t": t, "track_id": track_id, "u": u, "v": v, "role": role,
                "source": "detected"}

    def test_ppos_nearest_within_tolerance(self):
        tracks = [self._track(1.0, 0.3, 0.25, "far_left"),      # same half as hit (v0=0.3 < 0.5)
                  self._track(1.05, 0.6, 0.2, "far_right"),
                  self._track(1.0, 0.4, 0.8, "near_left"),      # other half
                  self._track(1.15, 0.1, 0.1, "far_right", 9)]  # outside 0.1 s — ignored
        r = _row(_label(1.0), events=[_hit(1.0), _hit(1.5)], rallies=[_rally()],
                 shuttle=[_det(1.0, 0.3, 0.3)], tracks=tracks)
        self.assertAlmostEqual(r["ppos_min_dist_same_half"], 0.05)   # (0.3,0.25) vs (0.3,0.3)
        self.assertAlmostEqual(r["ppos_min_dist_other_half"],
                               ((0.1 ** 2) + (0.5 ** 2)) ** 0.5)     # (0.4,0.8)
        self.assertEqual(r["ppos_n_slots_filled"], 3.0)

    def test_ppos_absent_file_vs_empty_tracks(self):
        common = {"events": [_hit(1.0), _hit(1.5)], "rallies": [_rally()],
                  "shuttle": [_det(1.0, 0.3, 0.3)]}
        absent = _row(_label(1.0), tracks=None, **common)
        for col in PPOS_COLS:
            self.assertIsNone(absent[col], col)
        empty = _row(_label(1.0), tracks=[], **common)
        self.assertIsNone(empty["ppos_min_dist_same_half"])
        self.assertEqual(empty["ppos_n_slots_filled"], 0.0)


class TestExporterAssembly(unittest.TestCase):
    def test_build_video_rows_from_synthetic_cache(self):
        import polars as pl

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            (cache / "frames.json").write_text(json.dumps(
                {"width": 3840, "height": 2160,
                 "frames": [{"frame": i, "t": i / 12.0} for i in range(120)]}))
            pl.DataFrame(
                [{"rally_idx": 0, "t_s": 5.0, "type": "hit", "payload": "{}", "confidence": 0.3},
                 {"rally_idx": 0, "t_s": 5.6, "type": "hit", "payload": "{}", "confidence": 0.3}],
                schema={"rally_idx": pl.Int64, "t_s": pl.Float64, "type": pl.Utf8,
                        "payload": pl.Utf8, "confidence": pl.Float64},
            ).write_parquet(cache / "events.parquet")
            pl.DataFrame(
                [{"idx": 0, "t_start_s": 4.5, "t_end_s": 8.0}],
                schema={"idx": pl.Int64, "t_start_s": pl.Float64, "t_end_s": pl.Float64},
            ).write_parquet(cache / "rallies.parquet")
            pl.DataFrame(
                [{"t": t, "u": 0.0, "v": 0.0, "visibility": 0.9, "frame": round(t * 12),
                  "px": px, "py": py, "conf": 0.9, "tracker": "tracknetv3"}
                 for t, px, py in ((5.0, 1900.0, 1000.0), (5.3, 1900.0, 700.0),
                                   (5.6, 1900.0, 1100.0))],
                schema={"t": pl.Float64, "u": pl.Float64, "v": pl.Float64,
                        "visibility": pl.Float64, "frame": pl.Int64, "px": pl.Float64,
                        "py": pl.Float64, "conf": pl.Float64, "tracker": pl.Utf8},
            ).write_parquet(cache / "shuttle.parquet")
            pl.DataFrame(
                [{"t": 5.0, "track_id": 7, "u": 0.5, "v": 0.4, "role": "far_left",
                  "source": "detected"}],
                schema={"t": pl.Float64, "track_id": pl.Int64, "u": pl.Float64,
                        "v": pl.Float64, "role": pl.Utf8, "source": pl.Utf8},
            ).write_parquet(cache / "tracks.parquet")

            unit_h = [1 / 3840, 0.0, 0.0, 0.0, 1 / 2160, 0.0, 0.0, 0.0, 1.0]
            labels = [_label(5.0), _label(5.6, shot_type="unknown", event_id="e2", hit_idx=2),
                      _label(20.0, event_id="e3")]
            rows = build_video_rows(cache, labels, homography=unit_h)

            self.assertEqual(len(rows), 3)  # never drop a label row
            self.assertFalse(rows[0]["is_calibration"])
            self.assertTrue(rows[1]["is_calibration"])
            self.assertFalse(rows[2]["prov_rally_found"])
            self.assertEqual(rows[0]["prov_hit_idx_recomputed"], rows[0]["hit_idx"])
            self.assertAlmostEqual(rows[0]["flight_time_out"], 0.6)
            self.assertIsNotNone(rows[0]["hit_dist_net"])
            self.assertIsNotNone(rows[0]["ppos_min_dist_same_half"])

            df = rows_to_frame(rows)
            for col in FEATURE_COLS + PPOS_COLS:
                self.assertEqual(df.schema[col], pl.Float64, col)
            self.assertEqual(df.schema["is_calibration"], pl.Boolean)
            self.assertEqual(df.schema["prov_out_boundary"], pl.Utf8)


if __name__ == "__main__":
    unittest.main()
