"""stroke.assist_report — labeling-assist stats + relabel queue over OOF predictions.

All pure functions on row dicts (the OOF parquet's to_dicts() shape) — known-answer fixtures,
no xgboost, no Label Studio. The CLI shell is exercised once end-to-end on a temp parquet.
(specs/features/stroke-labeling.md, AI-assisted pre-labeling)
"""
import json
import tempfile
import unittest
from pathlib import Path

from worker.import_stroke_labels import GERMAN_BY_CANONICAL, SHOT_TYPE
from worker.stroke.assist_report import (
    agreement,
    disagreements,
    enrichment,
    newest_oof,
    relabel_tasks,
    report_cli,
    threshold_table,
)
from worker.stroke.train import CLASSES


def _probs(**named):
    """Full prob_<class> dict, 0.0 unless named (e.g. _probs(drop=0.9, net=0.1))."""
    d = {f"prob_{c}": 0.0 for c in CLASSES}
    for k, v in named.items():
        d[f"prob_{k}"] = v
    return d


def _row(event_id, shot_type, pred, p_max, *, is_calibration=False,
         video_id="v1", t_s=1.0, probs=None):
    row = {"event_id": event_id, "video_id": video_id, "t_s": t_s, "fold": 0,
           "shot_type": shot_type, "is_calibration": is_calibration,
           "pred": pred, "p_max": p_max}
    row.update(probs if probs is not None else _probs(**{pred: p_max}))
    return row


def _agreement_fixture():
    return [
        # 2 top-1 correct
        _row("e1", "drop", "drop", 0.9),
        _row("e2", "lift", "lift", 0.8),
        # top-1 wrong, true class is the second-highest prob
        _row("e3", "drop", "net", 0.6, probs=_probs(net=0.6, drop=0.3)),
        # wrong entirely (true class not in top 2)
        _row("e4", "clear", "smash", 0.7, probs=_probs(smash=0.7, drive=0.2, clear=0.05)),
        # calibration row — excluded everywhere
        _row("u1", "unknown", "drop", 0.95, is_calibration=True),
    ]


class TestAgreement(unittest.TestCase):
    def test_known_answers_confident_rows_only(self):
        res = agreement(_agreement_fixture())
        self.assertEqual(res["n"], 4)
        self.assertAlmostEqual(res["top1"], 0.5)
        self.assertAlmostEqual(res["top2"], 0.75)

    def test_per_class_breakdown(self):
        res = agreement(_agreement_fixture())
        self.assertEqual(res["per_class"]["drop"], {"n": 2, "top1": 0.5})
        self.assertEqual(res["per_class"]["clear"], {"n": 1, "top1": 0.0})
        self.assertNotIn("unknown", res["per_class"])


class TestThresholdTable(unittest.TestCase):
    def test_coverage_and_covered_accuracy_per_tau(self):
        rows = [
            _row("e1", "drop", "drop", 0.9),    # covered @0.8, correct
            _row("e2", "drop", "net", 0.85),    # covered @0.8, wrong
            _row("e3", "lift", "lift", 0.6),    # covered @0.5 only, correct
            _row("e4", "net", "drop", 0.4),     # never covered
            _row("u1", "unknown", "drop", 0.99, is_calibration=True),  # excluded
        ]
        table = {t["tau"]: t for t in threshold_table(rows, taus=(0.5, 0.8))}
        self.assertAlmostEqual(table[0.8]["coverage"], 0.5)
        self.assertAlmostEqual(table[0.8]["covered_accuracy"], 0.5)
        self.assertEqual(table[0.8]["n"], 2)
        self.assertAlmostEqual(table[0.5]["coverage"], 0.75)
        self.assertAlmostEqual(table[0.5]["covered_accuracy"], 2 / 3)


class TestEnrichment(unittest.TestCase):
    def test_topk_hits_by_class_probability(self):
        rows = [
            _row("e1", "net", "net", 0.9, probs=_probs(net=0.9)),
            _row("e2", "drop", "drop", 0.8, probs=_probs(drop=0.1, net=0.8)),
            _row("e3", "net", "drop", 0.7, probs=_probs(net=0.7, drop=0.2)),
            _row("e4", "drop", "drop", 0.9, probs=_probs(drop=0.9, net=0.3)),
            _row("e5", "drop", "drop", 0.9, probs=_probs(drop=0.9, net=0.2)),
            _row("e6", "lift", "lift", 0.9, probs=_probs(lift=0.9, net=0.1)),
            _row("u1", "net", "net", 0.99, is_calibration=True),  # excluded
        ]
        res = {e["k"]: e for e in enrichment(rows, "net", ks=(2, 4))}
        self.assertEqual(res[2]["hits"], 1)   # top-2 by prob_net: e1 (net), e2 (drop)
        self.assertEqual(res[4]["hits"], 2)   # + e3 (net), e4 (drop)
        self.assertAlmostEqual(res[2]["base_rate"], 2 / 6)

    def test_k_capped_at_row_count(self):
        rows = [_row("e1", "net", "net", 0.9)]
        res = enrichment(rows, "net", ks=(5,))
        self.assertEqual(res[0]["k"], 1)


class TestDisagreements(unittest.TestCase):
    def test_filters_and_sorts_desc(self):
        rows = [
            _row("lo", "drop", "net", 0.79),                          # below min_p
            _row("hi", "drop", "net", 0.9),                           # in
            _row("edge", "lift", "clear", 0.8),                       # boundary: >= min_p
            _row("ok", "drop", "drop", 0.95),                         # correct — out
            _row("cal", "unknown", "drop", 0.99, is_calibration=True),  # calibration — out
        ]
        out = disagreements(rows, min_p=0.8)
        self.assertEqual([r["event_id"] for r in out], ["hi", "edge"])

    def test_keeps_exactly_the_review_columns(self):
        out = disagreements([_row("hi", "drop", "net", 0.9)], min_p=0.8)
        self.assertEqual(set(out[0]),
                         {"event_id", "video_id", "t_s", "shot_type", "pred", "p_max"})


def _task(event_id, **extra):
    data = {"video": f"/data/local-files/?d=stroke/G1/{event_id}.mp4", "game": "G1",
            "rally_idx": 1, "hit_idx": 2, "hit_t_s": 3.0, "first_of_rally": False,
            "serve_hint": "", "event_id": event_id, "video_id": "v1", "match_id": "m1"}
    data.update(extra)
    return {"data": data}


class TestRelabelTasks(unittest.TestCase):
    def test_joins_by_event_id_and_adds_display_fields(self):
        disags = disagreements([_row("e1", "net", "serve", 0.9)], min_p=0.8)
        queue, missing = relabel_tasks([_task("e1"), _task("e2")], disags)
        self.assertEqual(missing, 0)
        self.assertEqual(len(queue), 1)
        d = queue[0]["data"]
        self.assertEqual(d["prior_label"], "Netzschlag")   # German, via GERMAN_BY_CANONICAL
        self.assertEqual(d["model_pred"], "Aufschlag")
        self.assertAlmostEqual(d["model_score"], 0.9)
        self.assertEqual(d["video"], _task("e1")["data"]["video"])  # original fields intact

    def test_no_predictions_array_on_suspected_mislabels(self):
        disags = disagreements([_row("e1", "net", "serve", 0.9)], min_p=0.8)
        queue, _ = relabel_tasks([_task("e1")], disags)
        self.assertNotIn("predictions", queue[0])

    def test_counts_disagreements_without_matching_task(self):
        disags = disagreements(
            [_row("e1", "net", "serve", 0.9), _row("gone", "drop", "net", 0.85)], min_p=0.8)
        queue, missing = relabel_tasks([_task("e1")], disags)
        self.assertEqual(len(queue), 1)
        self.assertEqual(missing, 1)


class TestGermanByCanonical(unittest.TestCase):
    def test_inverse_of_import_map_and_bijective(self):
        self.assertEqual(len(GERMAN_BY_CANONICAL), len(SHOT_TYPE))
        for german, canonical in SHOT_TYPE.items():
            self.assertEqual(GERMAN_BY_CANONICAL[canonical], german)


class TestNewestOof(unittest.TestCase):
    def test_picks_lexicographically_newest_gbdt_run(self):
        with tempfile.TemporaryDirectory() as d:
            for stamp in ("20260101T000000Z_gbdt_traj", "20260301T000000Z_gbdt_traj",
                          "20260401T000000Z_cg_ap_J_only"):  # BST run — ignored
                run = Path(d) / "stroke_runs" / stamp
                run.mkdir(parents=True)
                (run / "oof_predictions.parquet").write_bytes(b"")
            found = newest_oof(Path(d))
            self.assertEqual(found.parent.name, "20260301T000000Z_gbdt_traj")

    def test_none_when_no_runs(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(newest_oof(Path(d)))


class TestReportCli(unittest.TestCase):
    def test_end_to_end_report_and_relabel_queue(self):
        import polars as pl

        rows = [
            _row("e1", "drop", "drop", 0.9),
            _row("e2", "net", "serve", 0.85),   # high-confidence disagreement
            _row("e3", "lift", "lift", 0.6),
            _row("u1", "unknown", "drop", 0.95, is_calibration=True),
        ]
        with tempfile.TemporaryDirectory() as d:
            oof = Path(d) / "oof_predictions.parquet"
            pl.DataFrame(rows).write_parquet(oof)
            tasks_json = Path(d) / "tasks_G1.json"
            tasks_json.write_text(json.dumps([_task("e1"), _task("e2")]), encoding="utf-8")
            queue_path = Path(d) / "relabel.json"

            rc = report_cli(["--oof", str(oof), "--relabel-queue", str(queue_path),
                             "--tasks", str(Path(d) / "tasks_*.json")])
            self.assertEqual(rc, 0)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            self.assertEqual([t["data"]["event_id"] for t in queue], ["e2"])
            self.assertEqual(queue[0]["data"]["prior_label"], "Netzschlag")

    def test_relabel_queue_requires_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                report_cli(["--oof", str(Path(d) / "x.parquet"),
                            "--relabel-queue", str(Path(d) / "q.json")])


if __name__ == "__main__":
    unittest.main()
