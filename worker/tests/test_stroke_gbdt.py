"""stroke.gbdt — LOGO-CV protocol + XGBoost fit for the trajectory-geometry spike.

The protocol (folds, OOF coverage, masks, weights, report shape) is parameterized by a
`fit_predict` callable, so everything except the actual boosting tests with a stub on light
installs — the same split as test_stroke_train.py (pure) vs test_stroke_model.py (guarded).
The three xgboost-touching tests self-skip without the ml extra.
"""
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from worker.stroke.gbdt import (
    fit_full,
    load_features,
    mean_gain,
    run_logo,
    save_assist,
    train_cli,
)
from worker.stroke.train import CLASSES, class_weights

_XGB = bool(find_spec("xgboost"))
_SKIP = "needs xgboost (uv sync --extra ml)"

RUN_LOGO_KEYS = {
    "classes", "folds", "mean_accuracy", "std_accuracy", "pooled_accuracy",
    "per_class_recall", "confusion", "hard_pairs", "tau", "unsicher_abstention",
    "unknown_rate_per_match", "gate_passed", "device", "config", "feature_importance_gain",
}


def _write_features(dir_path, rows):
    import polars as pl

    path = Path(dir_path) / "features.parquet"
    pl.DataFrame(rows, strict=False).write_parquet(path)
    return path


def _rows(video_classes: dict, calib_per_video: int = 1):
    """Synthetic feature rows: per video one confident row per listed class (+ Unsicher rows).
    `flight_time_out` encodes the class id — the hook for the perfect/separable stubs;
    `in_speed_img` carries NaN (None) on first rows, like real first-of-rally hits."""
    rows = []
    for vid, classes in video_classes.items():
        for i, shot in enumerate(classes):
            rows.append({"event_id": f"{vid}-{i}", "match_id": "m", "video_id": vid,
                         "t_s": float(i), "shot_type": shot, "hitter": None,
                         "rally_idx": 0, "hit_idx": i + 1, "is_calibration": False,
                         "flight_time_out": float(CLASSES.index(shot)),
                         "hit_pyn": 0.1 * CLASSES.index(shot),
                         "out_speed_img": 0.5,
                         "in_speed_img": None if i == 0 else 0.3})
        for j in range(calib_per_video):
            rows.append({"event_id": f"{vid}-u{j}", "match_id": "m", "video_id": vid,
                         "t_s": 100.0 + j, "shot_type": "unknown", "hitter": None,
                         "rally_idx": 1, "hit_idx": j + 1, "is_calibration": True,
                         "flight_time_out": 0.0, "hit_pyn": 0.5, "out_speed_img": 0.5,
                         "in_speed_img": 0.3})
    return rows


def _two_video_fixture():
    all7 = list(CLASSES)
    return _rows({"vid-a": all7 + ["drop", "lift"], "vid-b": all7 + ["clear"]})


def _uniform_stub(X_tr, y_tr, sample_weight, X_te, seed):
    import numpy as np

    return np.full((X_te.shape[0], len(CLASSES)), 1.0 / len(CLASSES))


def _perfect_stub(X_tr, y_tr, sample_weight, X_te, seed):
    """One-hot from the class id planted in feature 0 (flight_time_out)."""
    import numpy as np

    probs = np.full((X_te.shape[0], len(CLASSES)), 0.001)
    ids = np.clip(np.nan_to_num(X_te[:, 0]).astype(int), 0, len(CLASSES) - 1)
    probs[np.arange(len(ids)), ids] = 1.0 - 0.001 * (len(CLASSES) - 1)
    return probs


class TestLoadFeatures(unittest.TestCase):
    def test_schema_guard_missing_required_column(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{k: v for k, v in r.items() if k != "is_calibration"}
                    for r in _two_video_fixture()]
            path = _write_features(d, rows)
            with self.assertRaises(ValueError):
                load_features(path)

    def test_happy_path_returns_present_feature_columns_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_features(d, _two_video_fixture())
            df, cols = load_features(path)
            self.assertEqual(cols, ["flight_time_out", "out_speed_img", "in_speed_img",
                                    "hit_pyn"])  # FEATURE_COLS order, present only
            self.assertEqual(df.height, len(_two_video_fixture()))

    def test_rejects_single_video(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_features(d, _rows({"only": list(CLASSES)}))
            with self.assertRaises(ValueError):
                load_features(path)

    def test_no_feature_columns_raises(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{k: r[k] for k in ("event_id", "video_id", "t_s", "shot_type",
                                       "is_calibration")} for r in _two_video_fixture()]
            path = _write_features(d, rows)
            with self.assertRaises(ValueError):
                load_features(path)


class TestRunLogoProtocol(unittest.TestCase):
    @staticmethod
    def _run(rows, fit_predict, **cfg_over):
        import polars as pl

        df = pl.DataFrame(rows, strict=False)
        cols = ["flight_time_out", "hit_pyn", "out_speed_img", "in_speed_img"]
        cfg = {"seed": 42, **cfg_over}
        return run_logo(df, cols, fit_predict, cfg)

    def test_oof_covers_every_row_exactly_once(self):
        rows = _two_video_fixture()
        oof, report = self._run(rows, _uniform_stub)
        self.assertEqual(oof.height, len(rows))
        folds = sorted({r["video_id"] for r in rows})
        for row in oof.iter_rows(named=True):
            self.assertEqual(row["fold"], folds.index(row["video_id"]))
            self.assertAlmostEqual(
                sum(row[f"prob_{c}"] for c in CLASSES), 1.0, places=6)

    def test_train_rows_exclude_heldout_and_calibration(self):
        rows = _two_video_fixture()
        seen = []

        def spy(X_tr, y_tr, sample_weight, X_te, seed):
            seen.append((X_tr.shape[0], X_te.shape[0]))
            return _uniform_stub(X_tr, y_tr, sample_weight, X_te, seed)

        self._run(rows, spy)
        n_conf = {v: sum(1 for r in rows if r["video_id"] == v and not r["is_calibration"])
                  for v in ("vid-a", "vid-b")}
        n_all = {v: sum(1 for r in rows if r["video_id"] == v) for v in ("vid-a", "vid-b")}
        # fold 0 holds out vid-a: trains on vid-b's confident rows, predicts ALL vid-a rows
        self.assertEqual(seen[0], (n_conf["vid-b"], n_all["vid-a"]))
        self.assertEqual(seen[1], (n_conf["vid-a"], n_all["vid-b"]))

    def test_nan_features_reach_fit_unimputed(self):
        import numpy as np

        saw_nan = []

        def spy(X_tr, y_tr, sample_weight, X_te, seed):
            saw_nan.append(bool(np.isnan(X_tr).any() or np.isnan(X_te).any()))
            return _uniform_stub(X_tr, y_tr, sample_weight, X_te, seed)

        self._run(_two_video_fixture(), spy)
        self.assertTrue(all(saw_nan))  # first-of-rally in_speed_img None -> NaN, not filled

    def test_sample_weight_is_class_weights_lookup(self):
        import numpy as np

        def spy(X_tr, y_tr, sample_weight, X_te, seed):
            np.testing.assert_allclose(sample_weight, class_weights(y_tr)[y_tr])
            return _uniform_stub(X_tr, y_tr, sample_weight, X_te, seed)

        self._run(_two_video_fixture(), spy)

    def test_report_key_parity_golden(self):
        _, report = self._run(_two_video_fixture(), _uniform_stub)
        self.assertEqual(set(report), RUN_LOGO_KEYS)
        self.assertNotIn("torch_version", report)
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(set(report["folds"][0]),
                         {"fold", "video_id", "n", "accuracy", "per_class_recall"})
        self.assertEqual(set(report["tau"]),
                         {"tau", "covered_accuracy", "coverage", "knee_fallback", "curve"})
        self.assertEqual(set(report["hard_pairs"]), {"drop/net", "clear/lift"})
        self.assertEqual(len(report["confusion"]), len(CLASSES))

    def test_gate_passed_thresholds(self):
        _, perfect = self._run(_two_video_fixture(), _perfect_stub)
        self.assertEqual(perfect["mean_accuracy"], 1.0)
        self.assertTrue(perfect["gate_passed"])
        _, uniform = self._run(_two_video_fixture(), _uniform_stub)
        self.assertLess(uniform["mean_accuracy"], 0.5)
        self.assertFalse(uniform["gate_passed"])

    def test_oof_parquet_schema(self):
        oof, _ = self._run(_two_video_fixture(), _uniform_stub)
        expected = (["event_id", "video_id", "t_s", "shot_type", "is_calibration", "fold"]
                    + [f"prob_{c}" for c in CLASSES] + ["pred", "p_max"])
        self.assertEqual(oof.columns, expected)
        for row in oof.iter_rows(named=True):
            self.assertIn(row["pred"], CLASSES)
            self.assertAlmostEqual(row["p_max"],
                                   max(row[f"prob_{c}"] for c in CLASSES), places=9)


class TestMeanGain(unittest.TestCase):
    class _FakeBooster:
        def __init__(self, scores):
            self._scores = scores

        def get_score(self, importance_type="gain"):
            assert importance_type == "gain"
            return self._scores

    def test_zero_filled_mean_sorted_desc(self):
        boosters = [self._FakeBooster({"a": 4.0, "b": 2.0}),
                    self._FakeBooster({"a": 2.0})]  # b unused in fold 2 -> counts as 0
        gain = mean_gain(boosters, ["a", "b", "c"])
        self.assertEqual(gain, {"a": 3.0, "b": 1.0, "c": 0.0})
        self.assertEqual(list(gain), ["a", "b", "c"])  # sorted by mean gain desc


@unittest.skipUnless(_XGB, _SKIP)
class TestXgbFitPredict(unittest.TestCase):
    def test_shape_rowsum_and_seed_determinism(self):
        import numpy as np

        from worker.stroke.gbdt import xgb_fit_predict

        rng = np.random.default_rng(0)
        X = rng.normal(size=(70, 5))
        X[::7, 2] = np.nan
        y = np.arange(70) % len(CLASSES)
        sw = class_weights(y)[y]
        params = {"max_depth": 3, "eta": 0.3, "n_estimators": 10,
                  "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1}
        a = xgb_fit_predict(X, y, sw, X[:9], 7, params=params)
        b = xgb_fit_predict(X, y, sw, X[:9], 7, params=params)
        self.assertEqual(a.shape, (9, len(CLASSES)))
        np.testing.assert_allclose(a.sum(axis=1), np.ones(9), atol=1e-5)
        np.testing.assert_array_equal(a, b)  # nthread=1 + fixed seed -> bit-identical

    def test_separable_toy_learns(self):
        import polars as pl

        df = pl.DataFrame(_two_video_fixture(), strict=False)
        cols = ["flight_time_out", "hit_pyn", "out_speed_img", "in_speed_img"]

        from worker.stroke.gbdt import xgb_fit_predict

        def fit(X_tr, y_tr, sw, X_te, seed):
            # min_child_weight 0.1: softmax hessians are ~0.12/row — the production default
            # of 1 would block every split at this 9-row toy scale (verified), while the
            # real 443-row folds train fine with it
            return xgb_fit_predict(X_tr, y_tr, sw, X_te, seed,
                                   params={"max_depth": 3, "eta": 0.3, "n_estimators": 50,
                                           "subsample": 1.0, "colsample_bytree": 1.0,
                                           "min_child_weight": 0.1})

        _, report = run_logo(df, cols, fit, {"seed": 42})
        self.assertGreater(report["pooled_accuracy"], 0.95)  # class id is a feature


class TestFitFull(unittest.TestCase):
    def test_trains_on_all_confident_rows_with_empty_test_matrix(self):
        import polars as pl

        rows = _two_video_fixture()
        df = pl.DataFrame(rows, strict=False)
        cols = ["flight_time_out", "hit_pyn", "out_speed_img", "in_speed_img"]
        seen = []

        def spy(X_tr, y_tr, sample_weight, X_te, seed):
            seen.append((X_tr.shape, len(y_tr), len(sample_weight), X_te.shape))
            return _uniform_stub(X_tr, y_tr, sample_weight, X_te, seed)

        n_train = fit_full(df, cols, spy, {"seed": 42})
        n_conf = sum(1 for r in rows if not r["is_calibration"])
        self.assertEqual(n_train, n_conf)
        (x_shape, n_y, n_w, te_shape), = seen
        self.assertEqual(x_shape, (n_conf, len(cols)))
        self.assertEqual((n_y, n_w), (n_conf, n_conf))
        self.assertEqual(te_shape[0], 0)  # nothing held out — assist model sees all labels


class TestSaveAssist(unittest.TestCase):
    class _FakeBooster:
        def save_model(self, path):
            Path(path).write_bytes(b"booster-bytes")

    def test_meta_contract_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "20260712T000000Z_gbdt_traj"
            out_dir.mkdir(parents=True)
            cols = ["b_feature", "a_feature"]  # order is the inference contract — no sorting
            path = save_assist(self._FakeBooster(), out_dir, feature_cols=cols,
                               params={"eta": 0.05}, seed=42, n_train_rows=554,
                               dataset_sha256="abc123", git_sha="deadbeef",
                               xgboost_version="3.2.0")
            self.assertEqual(path, out_dir / "assist_model.ubj")
            self.assertTrue(path.exists())
            meta = json.loads((out_dir / "assist_meta.json").read_text())
            self.assertEqual(meta["purpose"], "labeling_assist_only")
            self.assertIs(meta["never_promote"], True)
            self.assertEqual(meta["model_version"], "20260712T000000Z_gbdt_traj")
            self.assertEqual(meta["classes"], list(CLASSES))
            self.assertEqual(meta["feature_columns"], cols)
            self.assertEqual(meta["n_train_rows"], 554)
            self.assertEqual(meta["dataset_sha256"], "abc123")


@unittest.skipUnless(_XGB, _SKIP)
class TestFitFullCli(unittest.TestCase):
    def test_fit_full_writes_loadable_assist_artifact(self):
        import numpy as np
        import polars as pl
        import xgboost as xgb

        with tempfile.TemporaryDirectory() as d:
            path = _write_features(d, _two_video_fixture())
            out = Path(d) / "run"
            rc = train_cli(["--features", str(path), "--out", str(out),
                            "--n-estimators", "10", "--seed", "42", "--fit-full"])
            self.assertEqual(rc, 0)
            meta = json.loads((out / "assist_meta.json").read_text())
            self.assertEqual(meta["purpose"], "labeling_assist_only")
            self.assertEqual(meta["model_version"], "run")
            n_conf = sum(1 for r in _two_video_fixture() if not r["is_calibration"])
            self.assertEqual(meta["n_train_rows"], n_conf)

            booster = xgb.Booster(model_file=str(out / "assist_model.ubj"))
            df = pl.read_parquet(path)
            X = df.select(meta["feature_columns"]).to_numpy().astype(np.float64)
            probs = booster.predict(xgb.DMatrix(X, feature_names=meta["feature_columns"]))
            self.assertEqual(probs.shape, (df.height, len(CLASSES)))
            np.testing.assert_allclose(probs.sum(axis=1), np.ones(df.height), atol=1e-5)

    def test_default_writes_no_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_features(d, _two_video_fixture())
            out = Path(d) / "run"
            train_cli(["--features", str(path), "--out", str(out),
                       "--n-estimators", "10", "--seed", "42"])
            self.assertFalse((out / "assist_model.ubj").exists())
            self.assertFalse((out / "assist_meta.json").exists())


@unittest.skipUnless(_XGB, _SKIP)
class TestTrainCli(unittest.TestCase):
    def test_cli_end_to_end_writes_run_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_features(d, _two_video_fixture())
            out = Path(d) / "run"
            rc = train_cli(["--features", str(path), "--out", str(out),
                            "--n-estimators", "10", "--seed", "42"])
            self.assertEqual(rc, 0)
            report = json.loads((out / "report.json").read_text())
            self.assertTrue(RUN_LOGO_KEYS <= set(report))
            for key in ("command", "date", "git_sha", "xgboost_version", "dataset", "caveats"):
                self.assertIn(key, report)
            self.assertNotIn("torch_version", report)
            self.assertEqual(report["dataset"]["n_rows"], len(_two_video_fixture()))
            self.assertIn("features.parquet", report["dataset"]["sha256"])
            self.assertIsInstance(report["feature_importance_gain"], dict)
            self.assertEqual(report["config"]["n_estimators"], 10)

            import polars as pl

            oof = pl.read_parquet(out / "oof_predictions.parquet")
            self.assertEqual(oof.height, len(_two_video_fixture()))


if __name__ == "__main__":
    unittest.main()
