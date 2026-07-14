"""stroke.train pure logic — folds, encoding, bones, jitter, metrics, τ sweep, dataset guard.

Torch-free by design: everything here runs on numpy/polars fixtures (imported lazily inside the
test methods, precedent test_dataset_export.py) so the suite collects green on light installs.
The torch side (BST4, LOGO loop) lives in test_stroke_model.py behind a skipUnless guard.
Locked eval protocol (spec stroke-classification.md): 7 trained classes, `unknown` is
abstention-only and must never reach the encoder; LOGO folds keyed by video_id; τ = smallest
grid value with covered-accuracy ≥ 0.75 subject to coverage ≥ 0.70, else the knee, flagged.
"""
import tempfile
import unittest
from pathlib import Path

from worker.stroke.train import (BONE_PAIRS, CLASSES, accuracy, class_weights, confusion,
                                 encode_labels, fold_split, jitter_shift, load_dataset,
                                 make_bones, per_class_recall, sweep_tau, to_model_arrays,
                                 unknown_rate_per_match, unsicher_abstention)


class TestClassesAndEncoding(unittest.TestCase):
    def test_classes_frozen_order(self):
        self.assertEqual(CLASSES, ("clear", "smash", "drop", "net", "drive", "lift", "serve"))

    def test_encode_round_trip(self):
        import numpy as np

        y = encode_labels(["drop", "clear", "serve", "drop"])
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual([CLASSES[i] for i in y], ["drop", "clear", "serve", "drop"])

    def test_unknown_is_never_a_class(self):
        with self.assertRaises(ValueError):
            encode_labels(["clear", "unknown"])

    def test_alien_token_raises(self):
        with self.assertRaises(ValueError):
            encode_labels(["no_hit"])


class TestFoldSplit(unittest.TestCase):
    def test_sorted_unique_video_ids(self):
        self.assertEqual(fold_split(["b", "a", "b", "c", "a"]), ["a", "b", "c"])


class TestClassWeights(unittest.TestCase):
    def test_balanced_inverse_frequency(self):
        import numpy as np

        w = class_weights(np.array([0, 0, 0, 1]), n_class=3)
        # n=4, k=2 present classes: n/(k*count) -> [4/6, 4/2, 0 for the absent class]
        self.assertAlmostEqual(w[0], 4 / 6)
        self.assertAlmostEqual(w[1], 2.0)
        self.assertEqual(w[2], 0.0)
        self.assertEqual(len(w), 3)


class TestMakeBones(unittest.TestCase):
    def test_nineteen_coco_pairs(self):
        self.assertEqual(len(BONE_PAIRS), 19)
        self.assertEqual(BONE_PAIRS[0], (0, 1))
        self.assertEqual(BONE_PAIRS[-1], (14, 16))

    def test_bone_is_end_minus_start(self):
        import numpy as np

        joints = np.zeros((1, 1, 1, 17, 2), dtype=np.float32)
        joints[0, 0, 0, 0] = (0.1, 0.2)
        joints[0, 0, 0, 1] = (0.4, 0.9)
        bones = make_bones(joints)
        self.assertEqual(bones.shape, (1, 1, 1, 19, 2))
        np.testing.assert_allclose(bones[0, 0, 0, 0], [0.3, 0.7], rtol=1e-6)

    def test_missing_joint_zero_guard_is_per_coordinate(self):
        # vendor create_bones: np.where((start != 0) & (end != 0), end - start, 0) elementwise
        import numpy as np

        joints = np.zeros((1, 1, 1, 17, 2), dtype=np.float32)
        joints[0, 0, 0, 0] = (0.0, 0.5)   # x missing, y present
        joints[0, 0, 0, 1] = (0.3, 0.7)
        bones = make_bones(joints)
        np.testing.assert_allclose(bones[0, 0, 0, 0], [0.0, 0.2], rtol=1e-6)

    def test_fully_missing_joint_gives_zero_bone(self):
        import numpy as np

        joints = np.zeros((1, 1, 1, 17, 2), dtype=np.float32)
        joints[0, 0, 0, 1] = (0.3, 0.7)  # joint 0 stays missing
        self.assertTrue((make_bones(joints)[0, 0, 0, 0] == 0).all())


class TestToModelArrays(unittest.TestCase):
    def test_j_only_shapes_and_placement(self):
        import numpy as np

        joints = np.zeros((2, 4, 16, 17, 2), dtype=np.float32)
        positions = np.zeros((2, 4, 16, 2), dtype=np.float32)
        joints[1, 2, 5, 3, 1] = 0.77          # (sample, slot, frame, joint, y)
        positions[1, 2, 5] = (0.25, 0.5)
        J, P = to_model_arrays(joints, positions, pose_style="J_only")
        self.assertEqual(J.shape, (2, 16, 4, 34))
        self.assertEqual(P.shape, (2, 16, 4, 2))
        self.assertEqual(J[1, 5, 2, 3 * 2 + 1], np.float32(0.77))
        np.testing.assert_allclose(P[1, 5, 2], [0.25, 0.5])

    def test_jnb_bone_appends_bones_after_joints(self):
        import numpy as np

        joints = np.zeros((1, 4, 16, 17, 2), dtype=np.float32)
        positions = np.zeros((1, 4, 16, 2), dtype=np.float32)
        joints[0, 0, 0, 0] = (0.1, 0.2)
        joints[0, 0, 0, 1] = (0.4, 0.9)
        J, _ = to_model_arrays(joints, positions, pose_style="JnB_bone")
        self.assertEqual(J.shape, (1, 16, 4, 72))  # (17 joints + 19 bones) * 2
        self.assertEqual(J[0, 0, 0, 0], np.float32(0.1))            # joint block first
        # bone (0,1) = end - start lands right after the 34 joint dims
        np.testing.assert_allclose(J[0, 0, 0, 34:36], [0.3, 0.7], rtol=1e-6)


class TestJitterShift(unittest.TestCase):
    @staticmethod
    def _arrays():
        import numpy as np

        J = np.arange(2 * 4 * 2 * 3, dtype=np.float32).reshape(2, 4, 2, 3) + 1
        S = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2) + 1
        P = np.arange(2 * 4 * 2 * 2, dtype=np.float32).reshape(2, 4, 2, 2) + 1
        return J, S, P

    def test_zero_shift_is_identity(self):
        import numpy as np

        J, S, P = self._arrays()
        out = jitter_shift((J, S, P), np.array([0, 0]))
        for a, b in zip(out, (J, S, P)):
            np.testing.assert_array_equal(a, b)

    def test_positive_shift_delays_content_and_zeros_first_frame(self):
        import numpy as np

        J, S, P = self._arrays()
        Jj, Sj, Pj = jitter_shift((J, S, P), np.array([1, 0]))
        self.assertTrue((Jj[0, 0] == 0).all())
        np.testing.assert_array_equal(Jj[0, 1:], J[0, :-1])
        self.assertTrue((Sj[0, 0] == 0).all())           # same shift across all arrays
        np.testing.assert_array_equal(Pj[0, 1:], P[0, :-1])
        np.testing.assert_array_equal(Jj[1], J[1])       # per-sample shifts

    def test_negative_shift_advances_content_and_zeros_last_frame(self):
        import numpy as np

        J, S, P = self._arrays()
        Jj, _, _ = jitter_shift((J, S, P), np.array([-1, -1]))
        self.assertTrue((Jj[0, -1] == 0).all())
        np.testing.assert_array_equal(Jj[0, :-1], J[0, 1:])

    def test_shapes_and_dtypes_preserved(self):
        import numpy as np

        J, S, P = self._arrays()
        out = jitter_shift((J, S, P), np.array([1, -1]))
        for a, b in zip(out, (J, S, P)):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        import numpy as np

        self.assertAlmostEqual(
            accuracy(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1])), 0.75)

    def test_per_class_recall_nan_at_zero_support(self):
        import numpy as np

        r = per_class_recall(np.array([0, 0, 1]), np.array([0, 1, 1]), n_class=3)
        self.assertAlmostEqual(r[0], 0.5)
        self.assertAlmostEqual(r[1], 1.0)
        self.assertTrue(np.isnan(r[2]))  # class 2 never occurs — NaN, not 0 (fold 9aaa599b/net)

    def test_confusion_rows_true_cols_pred(self):
        import numpy as np

        c = confusion(np.array([0, 0, 1]), np.array([0, 1, 1]), n_class=2)
        np.testing.assert_array_equal(c, [[1, 1], [0, 1]])


class TestSweepTau(unittest.TestCase):
    def test_smallest_feasible_tau(self):
        import numpy as np

        p_max = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([True, True, True, False, False])
        out = sweep_tau(p_max, correct, acc_target=0.75, cov_floor=0.6)
        # τ=0.6: covered 4/5 (cov 0.8), acc 3/4 = 0.75 — feasible and smaller than τ=0.7
        self.assertAlmostEqual(out["tau"], 0.6)
        self.assertAlmostEqual(out["covered_accuracy"], 0.75)
        self.assertAlmostEqual(out["coverage"], 0.8)
        self.assertFalse(out["knee_fallback"])

    def test_unsatisfiable_falls_back_to_knee_and_says_so(self):
        import numpy as np

        p_max = np.array([0.9, 0.8, 0.7])
        correct = np.array([False, False, False])
        out = sweep_tau(p_max, correct, acc_target=0.75, cov_floor=0.7)
        self.assertTrue(out["knee_fallback"])
        self.assertAlmostEqual(out["tau"], 0.0)  # acc stuck at 0 -> knee maximizes coverage

    def test_curve_includes_zero_tau(self):
        import numpy as np

        out = sweep_tau(np.array([0.5, 0.9]), np.array([True, True]))
        self.assertIn(0.0, [pt["tau"] for pt in out["curve"]])


class TestAbstentionRates(unittest.TestCase):
    def test_unknown_rate_per_match(self):
        import numpy as np

        rates = unknown_rate_per_match(
            np.array([0.9, 0.3, 0.3, 0.9]), ["g1", "g1", "g2", "g2"], tau=0.5)
        self.assertAlmostEqual(rates["g1"], 0.5)
        self.assertAlmostEqual(rates["g2"], 0.5)

    def test_unsicher_abstention_share_below_tau(self):
        import numpy as np

        self.assertAlmostEqual(
            unsicher_abstention(np.array([0.2, 0.6, 0.4]), tau=0.5), 2 / 3)


class TestLoadDataset(unittest.TestCase):
    """Row-alignment guard: the NPZ carries no key column, so load_dataset must hard-fail
    when meta.parquet does not describe the same rows in the same order."""

    @staticmethod
    def _write(dir_path, coverages, meta_coverages=None, meta_rows=None):
        import numpy as np
        import polars as pl

        n = len(coverages)
        joints = np.zeros((n, 4, 16, 17, 2), dtype=np.float32)
        shuttle = np.zeros((n, 16, 2), dtype=np.float32)
        positions = np.zeros((n, 4, 16, 2), dtype=np.float32)
        for i, cov in enumerate(coverages):
            k = round(cov * 16)
            shuttle[i, :k] = 0.5
        np.savez_compressed(Path(dir_path) / "stroke_dataset.npz",
                            joints=joints, shuttle=shuttle, positions=positions)
        meta_coverages = coverages if meta_coverages is None else meta_coverages
        rows = meta_rows if meta_rows is not None else len(meta_coverages)
        pl.DataFrame({
            "video_id": ["g1"] * rows,
            "shot_type": ["clear"] * rows,
            "is_calibration": [False] * rows,
            "shuttle_coverage": list(meta_coverages)[:rows],
        }).write_parquet(Path(dir_path) / "meta.parquet")

    def test_aligned_dataset_loads(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, [0.5, 1.0, 0.0])
            data, meta = load_dataset(Path(d))
            self.assertEqual(data["joints"].shape[0], 3)
            self.assertEqual(meta.height, 3)

    def test_row_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, [0.5, 1.0, 0.0], meta_rows=2)
            with self.assertRaises(ValueError):
                load_dataset(Path(d))

    def test_shuffled_meta_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, [0.5, 1.0, 0.0], meta_coverages=[1.0, 0.5, 0.0])
            with self.assertRaises(ValueError):
                load_dataset(Path(d))


if __name__ == "__main__":
    unittest.main()
