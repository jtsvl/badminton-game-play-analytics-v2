"""Tests for the shuttle-label importer (worker/import_shuttle_labels.py).

Pure mapper only (no DB): turns a shuttle_labels.json record (sparse anchors + boundaries) into the dense
per-frame rows for the `shuttle_labels` table, expanding via interpolate_anchors. The DB upsert itself is
exercised on the Mac Studio (integration). source_id carries the frame index so re-import is idempotent
per frame (specs/features/shuttle-finetune-plan.md §3).
"""
import unittest

from worker import import_shuttle_labels as isl

VIDEO = "f35d0984-207c-4ee1-8283-ef8ba58eca82"

# two visible anchors + a boundary on neither → linear fill across frames 10..12.
LABELS = {
    "video_id": VIDEO,
    "anchors": [
        {"frame_idx": 10, "t_s": 0.83, "px": 0.50, "py": 0.40, "visibility": 1, "_id": 0},
        {"frame_idx": 12, "t_s": 1.00, "px": 0.60, "py": 0.20, "visibility": 1, "_id": 1},
    ],
    "boundaries": [],
}
T_BY_FRAME = {10: 0.83, 11: 0.915, 12: 1.00}


class TestRowsFromShuttleLabels(unittest.TestCase):
    def test_expands_anchors_to_dense_rows(self):
        rows = isl.rows_from_shuttle_labels(VIDEO, LABELS, T_BY_FRAME)
        self.assertEqual([r["frame_idx"] for r in rows], [10, 11, 12])
        # anchors not interpolated; the in-between frame is the linear fill.
        self.assertEqual([r["interpolated"] for r in rows], [False, True, False])
        self.assertEqual([r["visibility"] for r in rows], [1, 1, 1])
        self.assertAlmostEqual(rows[1]["px"], 0.55)
        self.assertAlmostEqual(rows[1]["py"], 0.30)

    def test_source_id_equals_frame_idx_and_keys_present(self):
        # idempotent re-import relies on unique (video_id, source_id) == per-frame.
        for r in isl.rows_from_shuttle_labels(VIDEO, LABELS, T_BY_FRAME):
            self.assertIn("video_id", r)
            self.assertEqual(r["video_id"], VIDEO)
            self.assertEqual(r["source_id"], r["frame_idx"])

    def test_t_s_pulled_from_manifest_not_interpolated(self):
        rows = isl.rows_from_shuttle_labels(VIDEO, LABELS, T_BY_FRAME)
        self.assertEqual([r["t_s"] for r in rows], [0.83, 0.915, 1.00])

    def test_invisible_anchor_and_gap_carry_null_position(self):
        labels = {
            "anchors": [
                {"frame_idx": 0, "px": 0.1, "py": 0.1, "visibility": 1},
                {"frame_idx": 2, "visibility": 0},
                {"frame_idx": 4, "px": 0.4, "py": 0.4, "visibility": 1},
            ],
            "boundaries": [],
        }
        rows = isl.rows_from_shuttle_labels(VIDEO, labels, {i: float(i) for i in range(5)})
        by_frame = {r["frame_idx"]: r for r in rows}
        self.assertEqual(by_frame[2]["visibility"], 0)
        self.assertIsNone(by_frame[2]["px"])
        self.assertIsNone(by_frame[2]["py"])
        self.assertFalse(by_frame[2]["interpolated"])  # hand-marked occlusion
        self.assertEqual(by_frame[1]["visibility"], 0)  # neighbour not bridged
        self.assertTrue(by_frame[1]["interpolated"])

    def test_empty_labels_give_no_rows(self):
        self.assertEqual(isl.rows_from_shuttle_labels(VIDEO, {"anchors": [], "boundaries": []}, {}), [])


if __name__ == "__main__":
    unittest.main()
