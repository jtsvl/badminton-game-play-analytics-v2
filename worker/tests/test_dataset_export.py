"""Tests for the shuttle label -> TrackNetV3 dataset exporter (worker/shuttle/dataset_export.py).

The exporter is the riskiest piece: the vendored dataset.py asserts on every deviation from its on-disk
contract, and the fine-tune only closes the domain gap if the exported frames are IDENTICAL to what
inference feeds the tracker. These tests pin the pure mapping (match-dir int-sortability, csv-vs-
corrected_csv routing, 0-based Frame alignment, full-frame->512x288 transform) and the byte-identity of
the exported crop vs. the pipeline's tracker input (specs/features/shuttle-finetune-plan.md §4).
"""
import unittest

from worker.shuttle import dataset_export as dx


class TestMatchName(unittest.TestCase):
    def test_match_name_is_integer_sortable(self):
        # the vendored get_rally_dirs sorts match dirs by int(s.split('match')[-1]) — names must parse.
        names = [dx.match_name(n) for n in (1, 2, 10)]
        self.assertEqual(names, ["match1", "match2", "match10"])
        self.assertEqual(sorted(names, key=lambda s: int(s.split("match")[-1])), ["match1", "match2", "match10"])


class TestCsvRelpath(unittest.TestCase):
    def test_train_and_val_use_csv(self):
        self.assertEqual(dx.csv_relpath("r1", "train"), "csv/r1_ball.csv")
        self.assertEqual(dx.csv_relpath("r1", "val"), "csv/r1_ball.csv")

    def test_test_split_uses_corrected_csv(self):
        # dataset.py:223 reads corrected_csv/ when 'test' is in the rally path.
        self.assertEqual(dx.csv_relpath("r1", "test"), "corrected_csv/r1_ball.csv")


class TestRallyRows(unittest.TestCase):
    # crop box (x0,y0,x1,y1): cw=200, ch=60 → model scale 512/200, 288/60.
    BOX = (100, 200, 300, 260)
    FW, FH = 1000, 1000

    def _dense(self):
        # visible anchors at the crop centre and a corner, plus a gap frame, starting at frame 100.
        return [
            (100, 0.20, 0.23, 1, False),   # full-frame (200,230) = crop centre → model (256,144)
            (101, None, None, 0, True),     # gap
            (102, 0.10, 0.20, 1, False),   # full-frame (100,200) = crop top-left → model (0,0)
        ]

    def test_frames_are_zero_based_contiguous(self):
        rows = dx.rally_rows(self._dense(), self.FW, self.FH, self.BOX)
        self.assertEqual([r["Frame"] for r in rows], [0, 1, 2])

    def test_visible_xy_in_model_space(self):
        rows = dx.rally_rows(self._dense(), self.FW, self.FH, self.BOX)
        self.assertAlmostEqual(rows[0]["X"], 256.0)
        self.assertAlmostEqual(rows[0]["Y"], 144.0)
        self.assertEqual(rows[0]["Visibility"], 1)
        self.assertAlmostEqual(rows[2]["X"], 0.0)
        self.assertAlmostEqual(rows[2]["Y"], 0.0)

    def test_invisible_frame_is_zeroed(self):
        # TrackNetV3 convention: Visibility=0 ⇒ X=Y=0 (dataset._get_heatmap returns a zero target).
        rows = dx.rally_rows(self._dense(), self.FW, self.FH, self.BOX)
        self.assertEqual(rows[1]["Visibility"], 0)
        self.assertEqual(rows[1]["X"], 0)
        self.assertEqual(rows[1]["Y"], 0)


class TestSeqLenAndRange(unittest.TestCase):
    """TrackNetV5 trains with 5-frame windows (V3: 8) and an explicit within-clip split is exported with
    --frames A:B (specs/features/shuttle-tracknetv5.md)."""

    def test_default_window_is_v3_and_v5_is_five(self):
        self.assertEqual(dx.SEQ_LEN, 8)
        dx.check_rally_length(8, 8)
        dx.check_rally_length(5, 5)
        with self.assertRaises(ValueError):
            dx.check_rally_length(7, 8)
        with self.assertRaises(ValueError):
            dx.check_rally_length(4, 5)
        with self.assertRaises(ValueError):
            dx.check_rally_length(10, 0)

    def test_select_dense_range_is_inclusive_on_manifest_positions(self):
        dense = [(i, 0.5, 0.5, 1, False) for i in range(100, 110)]
        self.assertEqual([d[0] for d in dx.select_dense_range(dense, (103, 105))], [103, 104, 105])
        self.assertEqual(dx.select_dense_range(dense, None), dense)
        self.assertEqual(dx.select_dense_range(dense, (200, 300)), [])


class TestFramesMedian(unittest.TestCase):
    def test_median_is_float64_not_truncated(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy (ml extra) not installed")
        a = np.full((2, 2, 3), 100, dtype=np.uint8)
        b = np.full((2, 2, 3), 103, dtype=np.uint8)
        med = dx.frames_median([a, b])
        self.assertEqual(med.dtype, np.float64)
        self.assertEqual(med.shape, (2, 2, 3))
        self.assertTrue((med == 101.5).all())   # the .5 survives — V3 casts to uint8 itself, V5 subtracts


class TestMaskedModelCrop(unittest.TestCase):
    """The exported PNG must be byte-identical to the array pipeline/shuttle.py feeds the tracker:
    crop → airspace-mask → resize(512x288) → BGR→RGB. This proves train ≈ inference."""

    def test_matches_pipeline_and_survives_png_roundtrip(self):
        import io

        import cv2
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)  # BGR full frame
        crop_box = (50, 60, 300, 260)  # x0,y0,x1,y1
        air = [(60, 80), (280, 80), (290, 250), (40, 250)]  # airspace polygon (full-frame px)

        mask = dx.build_airspace_mask(air, crop_box)
        got = dx.masked_model_crop(img, crop_box, mask)

        # manual replication of pipeline/shuttle.py:93-95 + tracknetv3.py:113
        x0, y0, x1, y1 = crop_box
        c = img[y0:y1, x0:x1]
        c = cv2.bitwise_and(c, c, mask=mask)
        expected = cv2.resize(c, (512, 288))[:, :, ::-1]
        self.assertTrue(np.array_equal(got, expected))
        self.assertEqual(got.shape, (288, 512, 3))

        # PNG round-trip is lossless → the on-disk frame decodes to the same RGB array.
        buf = io.BytesIO()
        Image.fromarray(got).save(buf, "PNG")
        buf.seek(0)
        self.assertTrue(np.array_equal(np.array(Image.open(buf)), got))


if __name__ == "__main__":
    unittest.main()
