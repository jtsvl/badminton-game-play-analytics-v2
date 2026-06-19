"""Tests for the verify-tool corrections -> Postgres importer (worker/import_corrections.py).

Pure mapper only (no DB): the importer turns a corrections.json record into a row for the
`detection_corrections` table. The DB write itself is exercised on the Mac Studio (integration).
"""
import unittest

from worker import import_corrections as ic

VIDEO = "f35d0984-207c-4ee1-8283-ef8ba58eca82"


class TestRowsFromCorrections(unittest.TestCase):
    def test_maps_id_to_source_id_and_keeps_core_fields(self):
        cors = [{"role": "A1", "type": "bbox", "frame_idx": 2, "t_s": 0.17,
                 "bbox": [0.5604, 0.4602, 0.0475, 0.1857],
                 "orig_bbox": [0.5604, 0.4602, 0.0476, 0.1608],
                 "foot_norm": [0.5841, 0.6459], "court_uv": [0.7435, 0.695], "_id": 1}]
        rows = ic.rows_from_corrections(VIDEO, cors)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["video_id"], VIDEO)
        self.assertEqual(r["source_id"], 1)   # _id -> source_id (the unique-per-video key)
        self.assertNotIn("_id", r)
        self.assertEqual(r["frame_idx"], 2)
        self.assertEqual(r["t_s"], 0.17)
        self.assertEqual(r["role"], "A1")
        self.assertEqual(r["type"], "bbox")
        self.assertEqual(r["bbox"], [0.5604, 0.4602, 0.0475, 0.1857])
        self.assertEqual(r["court_uv"], [0.7435, 0.695])

    def test_drops_absent_optional_fields(self):
        # a 'position' correction carries no bbox/orig_bbox/correct_role — those keys must be absent
        # (so the DB column stays NULL) rather than written as None.
        cors = [{"role": "A1", "type": "position", "frame_idx": 181, "t_s": 15.1,
                 "edited_in": "video", "foot_norm": [0.584, 0.5869], "court_uv": [0.7979, 0.5552],
                 "orig_foot": [0.5834, 0.5412], "orig_uv": [0.8474, 0.4228], "_id": 64}]
        r = ic.rows_from_corrections(VIDEO, cors)[0]
        self.assertNotIn("bbox", r)
        self.assertNotIn("orig_bbox", r)
        self.assertNotIn("correct_role", r)
        self.assertEqual(r["edited_in"], "video")
        self.assertEqual(r["orig_uv"], [0.8474, 0.4228])

    def test_every_row_has_both_upsert_key_fields(self):
        # idempotent re-import relies on the unique (video_id, source_id) — both must always be present.
        cors = [{"role": "B1", "type": "added", "frame_idx": 284, "t_s": 23.69,
                 "bbox": [0.5026, 0.3953, 0.0162, 0.0744], "foot_norm": [0.5107, 0.4697],
                 "court_uv": [0.5612, 0.0], "_id": 131}]
        r = ic.rows_from_corrections(VIDEO, cors)[0]
        self.assertIn("video_id", r)
        self.assertIn("source_id", r)
        self.assertEqual(r["type"], "added")


if __name__ == "__main__":
    unittest.main()
