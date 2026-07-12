"""stroke.prelabel — GBDT pre-annotations for Label Studio task exports.

Pure mapping tests with a stubbed predict function (no xgboost, no Label Studio, no DB);
only the load_assist happy path trains a real 10-round booster and is guarded like the other
xgboost tests. (specs/features/stroke-labeling.md, AI-assisted pre-labeling)
"""
import json
import math
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from worker.stroke.prelabel import (
    AssistModel,
    attach_predictions,
    load_assist,
    prediction_entry,
    predict_rows,
    pseudo_labels,
)
from worker.stroke.train import CLASSES

_XGB = bool(find_spec("xgboost"))
_SKIP = "needs xgboost (uv sync --extra ml)"


class TestPredictionEntry(unittest.TestCase):
    def test_exact_label_studio_choices_schema(self):
        entry = prediction_entry("drop", 0.81237, model_version="run_x", result_id="e1")
        self.assertEqual(entry, {
            "model_version": "run_x",
            "score": 0.8124,
            "result": [{
                "id": "e1",
                "type": "choices",
                "from_name": "shot_type",   # must match the labeling config's name=
                "to_name": "video",         # ...and toName= (stroke-labeling.md)
                "value": {"choices": ["Drop"]},
            }],
        })

    def test_german_backmapping(self):
        self.assertEqual(prediction_entry("net", 0.9, model_version="v", result_id="x")
                         ["result"][0]["value"]["choices"], ["Netzschlag"])
        self.assertEqual(prediction_entry("serve", 0.9, model_version="v", result_id="x")
                         ["result"][0]["value"]["choices"], ["Aufschlag"])


class TestPseudoLabels(unittest.TestCase):
    def test_hits_become_traj_label_dicts(self):
        hits = [
            {"event": {"id": "e1", "t_s": 3.0}, "rally_idx": 1, "hit_idx": 1,
             "first_of_rally": True},
            {"event": {"id": "e2", "t_s": 4.5}, "rally_idx": 1, "hit_idx": 2,
             "first_of_rally": False},
        ]
        labels = pseudo_labels(hits, video_id="v1", match_id="m1")
        self.assertEqual(labels[0], {
            "event_id": "e1", "match_id": "m1", "video_id": "v1", "t_s": 3.0,
            "shot_type": None, "hitter": None, "rally_idx": 1, "hit_idx": 1,
        })
        self.assertEqual(labels[1]["event_id"], "e2")
        self.assertEqual(labels[1]["t_s"], 4.5)


def _probs_row(**named):
    row = [0.0] * len(CLASSES)
    for cls, p in named.items():
        row[CLASSES.index(cls)] = p
    return row


class TestPredictRows(unittest.TestCase):
    def _assist(self, matrix, seen):
        def predict_fn(X):
            seen.append(X)
            return matrix

        return AssistModel(predict_fn=predict_fn, feature_cols=["f_b", "f_a"],
                           classes=CLASSES, model_version="run_x")

    def test_feature_order_nan_passthrough_and_keys(self):
        import numpy as np

        seen = []
        assist = self._assist(np.array([_probs_row(drop=0.7, net=0.2, drive=0.1)]), seen)
        rows = [{"event_id": "e1", "f_a": 2.0, "f_b": None, "extra_meta": "ignored"}]
        preds = predict_rows(assist, rows)

        X, = seen
        self.assertEqual(X.shape, (1, 2))
        self.assertTrue(math.isnan(X[0, 0]))   # f_b first — assist.feature_cols order wins
        self.assertEqual(X[0, 1], 2.0)
        self.assertEqual(set(preds), {"e1"})
        self.assertEqual(preds["e1"]["pred"], "drop")
        self.assertAlmostEqual(preds["e1"]["score"], 0.7)
        self.assertAlmostEqual(preds["e1"]["prob_net"], 0.2)
        self.assertAlmostEqual(preds["e1"]["prob_drive"], 0.1)

    def test_empty_rows_no_predict_call(self):
        seen = []
        assist = self._assist(None, seen)
        self.assertEqual(predict_rows(assist, []), {})
        self.assertEqual(seen, [])


def _task(event_id, first_of_rally=False):
    return {"data": {"video": f"/data/local-files/?d=stroke/G1/{event_id}.mp4",
                     "game": "G1", "rally_idx": 1, "hit_idx": 2, "hit_t_s": 3.0,
                     "first_of_rally": first_of_rally, "serve_hint": "",
                     "event_id": event_id, "video_id": "v1", "match_id": "m1"}}


def _pred(pred="drop", score=0.9, prob_net=0.0, prob_drive=0.0):
    return {"pred": pred, "score": score, "prob_net": prob_net, "prob_drive": prob_drive}


class TestAttachPredictions(unittest.TestCase):
    def test_prefills_above_min_score_hints_below(self):
        tasks = [_task("hi"), _task("lo")]
        counts = attach_predictions(
            tasks, {"hi": _pred(score=0.85), "lo": _pred(pred="net", score=0.6)},
            model_version="run_x", min_score=0.8)
        self.assertIn("predictions", tasks[0])
        self.assertEqual(tasks[0]["predictions"][0]["result"][0]["value"]["choices"],
                         ["Drop"])
        self.assertEqual(tasks[0]["data"]["model_pred"], "Drop")
        self.assertNotIn("predictions", tasks[1])   # below threshold: display-only hint
        self.assertEqual(tasks[1]["data"]["model_pred"], "Netzschlag")
        self.assertAlmostEqual(tasks[1]["data"]["model_score"], 0.6)
        self.assertEqual(counts, {"prelabeled": 1, "skipped_low_score": 1,
                                  "skipped_first": 0, "no_features": 0})

    def test_boundary_score_prefills(self):
        tasks = [_task("edge")]
        attach_predictions(tasks, {"edge": _pred(score=0.8)},
                           model_version="v", min_score=0.8)
        self.assertIn("predictions", tasks[0])

    def test_skip_first_of_rally_optionally(self):
        tasks = [_task("serve1", first_of_rally=True)]
        counts = attach_predictions(tasks, {"serve1": _pred(pred="serve", score=0.95)},
                                    model_version="v", min_score=0.8,
                                    skip_first_of_rally=True)
        self.assertNotIn("predictions", tasks[0])
        self.assertEqual(tasks[0]["data"]["model_pred"], "Aufschlag")  # hint stays
        self.assertEqual(counts["skipped_first"], 1)

    def test_task_without_features_gets_empty_hint(self):
        tasks = [_task("orphan")]
        counts = attach_predictions(tasks, {}, model_version="v", min_score=0.8)
        self.assertNotIn("predictions", tasks[0])
        self.assertEqual(tasks[0]["data"]["model_pred"], "")
        self.assertIsNone(tasks[0]["data"]["model_score"])
        self.assertEqual(counts["no_features"], 1)

    def test_rare_score_is_max_of_net_and_drive(self):
        tasks = [_task("e1")]
        attach_predictions(tasks, {"e1": _pred(prob_net=0.12, prob_drive=0.348)},
                           model_version="v", min_score=0.8)
        self.assertAlmostEqual(tasks[0]["data"]["rare_score"], 0.348)


class TestLoadAssistValidation(unittest.TestCase):
    def _write_meta(self, d, **over):
        meta = {"purpose": "labeling_assist_only", "never_promote": True,
                "model_version": "run_x", "classes": list(CLASSES),
                "feature_columns": ["f_a"], "params": {}, "seed": 42,
                "n_train_rows": 10, "dataset_sha256": "x", "git_sha": None,
                "xgboost_version": "3.2.0"}
        meta.update(over)
        (Path(d) / "assist_meta.json").write_text(json.dumps(meta))

    def test_refuses_wrong_purpose_before_touching_xgboost(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_meta(d, purpose="pipeline")
            (Path(d) / "assist_model.ubj").write_bytes(b"x")
            with self.assertRaises(ValueError):
                load_assist(Path(d))

    def test_refuses_class_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_meta(d, classes=["clear", "smash"])
            (Path(d) / "assist_model.ubj").write_bytes(b"x")
            with self.assertRaises(ValueError):
                load_assist(Path(d))

    def test_missing_meta_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_assist(Path(d))


@unittest.skipUnless(_XGB, _SKIP)
class TestLoadAssistRoundTrip(unittest.TestCase):
    def test_loads_real_booster_and_predicts(self):
        import numpy as np
        import xgboost as xgb

        rng = np.random.default_rng(0)
        X = rng.normal(size=(70, 2))
        y = np.arange(70) % len(CLASSES)
        booster = xgb.train({"objective": "multi:softprob", "num_class": len(CLASSES),
                             "tree_method": "hist", "nthread": 1, "seed": 1},
                            xgb.DMatrix(X, label=y, feature_names=["f_b", "f_a"]),
                            num_boost_round=5)
        with tempfile.TemporaryDirectory() as d:
            booster.save_model(str(Path(d) / "assist_model.ubj"))
            TestLoadAssistValidation._write_meta(
                TestLoadAssistValidation(), d, feature_columns=["f_b", "f_a"])
            assist = load_assist(Path(d))
            self.assertEqual(assist.feature_cols, ["f_b", "f_a"])
            self.assertEqual(assist.model_version, "run_x")
            preds = predict_rows(assist, [{"event_id": "e1", "f_a": 0.5, "f_b": None}])
        self.assertIn(preds["e1"]["pred"], CLASSES)
        self.assertAlmostEqual(
            sum(v for k, v in preds["e1"].items() if k == "score") +
            0.0, preds["e1"]["score"])  # score present and numeric


if __name__ == "__main__":
    unittest.main()
