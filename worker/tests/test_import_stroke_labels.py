"""Tests for the stroke-label importer (worker/import_stroke_labels.py).

Pure mapping only (no I/O): turns a Label Studio JSON export (tasks with choice annotations) into
the training rows for the stroke_labels parquet, German UI values → canonical tokens per the table
in specs/features/stroke-labeling.md. `no_hit` rows split off to the hit-detection eval set;
`unknown` stays in training (the taxonomy's honesty option, not a labeling failure). Unmapped
values and duplicate event_ids are hard errors — never silently coerced into training data.
"""
import unittest

from worker import import_stroke_labels as isl


def _ann(shot=None, hitter=None, was_cancelled=False, updated_at="2026-07-04 10:00:00"):
    result = []
    if shot is not None:
        result.append({"from_name": "shot_type", "to_name": "video", "type": "choices",
                       "value": {"choices": [shot]}})
    if hitter is not None:
        result.append({"from_name": "hitter", "to_name": "video", "type": "choices",
                       "value": {"choices": [hitter]}})
    return {"id": 1, "result": result, "was_cancelled": was_cancelled,
            "created_at": updated_at, "updated_at": updated_at}


def _task(shot="Clear", hitter="vorne-links", event_id="e-1", annotations=None, **data_overrides):
    data = {"video": "/data/local-files/?d=stroke/g1/clip.mp4", "game": "Session2_Game_1",
            "rally_idx": 3, "hit_idx": 2, "hit_t_s": 41.27, "first_of_rally": False,
            "serve_hint": "", "event_id": event_id, "video_id": "v-1", "match_id": "m-1"}
    data.update(data_overrides)
    if annotations is None:
        annotations = [_ann(shot, hitter)]
    return {"id": 7, "data": data, "annotations": annotations}


class TestPickAnnotation(unittest.TestCase):
    def test_single_annotation_returned(self):
        ann = _ann("Clear", "vorne-links")
        self.assertIs(isl.pick_annotation({"annotations": [ann]}), ann)

    def test_cancelled_annotations_skipped(self):
        keep = _ann("Clear", None)
        cancelled = _ann("Smash", None, was_cancelled=True, updated_at="2026-07-04 12:00:00")
        self.assertIs(isl.pick_annotation({"annotations": [cancelled, keep]}), keep)

    def test_latest_updated_at_wins(self):
        old = _ann("Clear", None, updated_at="2026-07-04 10:00:00")
        new = _ann("Smash", None, updated_at="2026-07-04 11:30:00")
        self.assertIs(isl.pick_annotation({"annotations": [old, new]}), new)
        self.assertIs(isl.pick_annotation({"annotations": [new, old]}), new)

    def test_none_when_empty_or_all_cancelled(self):
        self.assertIsNone(isl.pick_annotation({"annotations": []}))
        self.assertIsNone(isl.pick_annotation(
            {"annotations": [_ann("Clear", None, was_cancelled=True)]}))


class TestRowsFromExport(unittest.TestCase):
    def test_maps_task_to_canonical_row(self):
        rows = isl.rows_from_export([_task("Aufschlag", "vorne-rechts")])
        self.assertEqual(rows, [{
            "event_id": "e-1", "match_id": "m-1", "video_id": "v-1", "t_s": 41.27,
            "shot_type": "serve", "hitter": "near_right", "rally_idx": 3, "hit_idx": 2,
        }])

    def test_missing_hitter_is_none(self):
        rows = isl.rows_from_export([_task("Unsicher", None)])
        self.assertEqual(rows[0]["shot_type"], "unknown")
        self.assertIsNone(rows[0]["hitter"])

    def test_kein_schlag_maps_to_no_hit(self):
        rows = isl.rows_from_export([_task("Kein Schlag", None)])
        self.assertEqual(rows[0]["shot_type"], "no_hit")
        self.assertIsNone(rows[0]["hitter"])

    def test_unlabeled_task_skipped(self):
        rows = isl.rows_from_export([_task(annotations=[]), _task(event_id="e-2")])
        self.assertEqual([r["event_id"] for r in rows], ["e-2"])

    def test_unmapped_shot_type_raises(self):
        with self.assertRaisesRegex(ValueError, "Schmetterball"):
            isl.rows_from_export([_task("Schmetterball", None)])

    def test_unmapped_hitter_raises(self):
        with self.assertRaisesRegex(ValueError, "mitte"):
            isl.rows_from_export([_task("Clear", "mitte")])

    def test_missing_shot_type_raises(self):
        # labeling config marked shot_type required — absence means a malformed export.
        with self.assertRaisesRegex(ValueError, "shot_type"):
            isl.rows_from_export([_task(None, "vorne-links")])

    def test_duplicate_event_id_raises(self):
        with self.assertRaisesRegex(ValueError, "e-1"):
            isl.rows_from_export([_task(event_id="e-1"), _task(event_id="e-1")])


class TestSplitRows(unittest.TestCase):
    def test_no_hit_goes_to_eval_unknown_stays_in_training(self):
        rows = isl.rows_from_export([
            _task("Clear", "vorne-links", event_id="e-1"),
            _task("Unsicher", None, event_id="e-2"),
            _task("Kein Schlag", None, event_id="e-3"),
        ])
        training, eval_rows = isl.split_rows(rows)
        self.assertEqual([r["shot_type"] for r in training], ["clear", "unknown"])
        self.assertEqual([r["shot_type"] for r in eval_rows], ["no_hit"])


class TestMergeRows(unittest.TestCase):
    @staticmethod
    def _row(event_id, shot_type, hitter=None):
        return {"event_id": event_id, "match_id": "m-1", "video_id": "v-1", "t_s": 1.0,
                "shot_type": shot_type, "hitter": hitter, "rally_idx": 1, "hit_idx": 1}

    def test_fixes_win_by_event_id_order_stable(self):
        base = [self._row("a", "drop"), self._row("b", "lift"), self._row("c", "net")]
        fixes = [self._row("b", "clear")]
        merged = isl.merge_rows(base, fixes)
        self.assertEqual([(r["event_id"], r["shot_type"]) for r in merged],
                         [("a", "drop"), ("b", "clear"), ("c", "net")])

    def test_novel_fixes_appended_in_order(self):
        base = [self._row("a", "drop")]
        fixes = [self._row("z", "serve"), self._row("a", "smash")]
        merged = isl.merge_rows(base, fixes)
        self.assertEqual([(r["event_id"], r["shot_type"]) for r in merged],
                         [("a", "smash"), ("z", "serve")])


class TestMergeIntoCli(unittest.TestCase):
    def test_relabel_flips_cross_the_training_no_hit_split(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            base_dir = Path(d)
            # existing labeling dir: e-1 drop + e-3 lift (training), e-2 no_hit (eval)
            isl.write_labels_parquet(
                [TestMergeRows._row("e-1", "drop"), TestMergeRows._row("e-3", "lift")],
                base_dir / "stroke_labels.parquet")
            isl.write_labels_parquet([TestMergeRows._row("e-2", "no_hit")],
                                     base_dir / "hit_detection_eval.parquet")
            # review export flips e-1 -> Kein Schlag and e-2 -> Drop
            export = base_dir / "review.json"
            export.write_text(json.dumps([
                _task("Kein Schlag", None, event_id="e-1"),
                _task("Drop", "vorne-links", event_id="e-2"),
            ]), encoding="utf-8")

            rc = isl.import_cli([str(export), "--merge-into", str(base_dir)])
            self.assertEqual(rc, 0)

            import polars as pl

            training = pl.read_parquet(base_dir / "stroke_labels.parquet")
            eval_rows = pl.read_parquet(base_dir / "hit_detection_eval.parquet")
            got_train = dict(zip(training["event_id"], training["shot_type"]))
            self.assertEqual(got_train, {"e-3": "lift", "e-2": "drop"})
            self.assertEqual(dict(zip(eval_rows["event_id"], eval_rows["shot_type"])),
                             {"e-1": "no_hit"})

    def test_merge_into_missing_parquet_fails_loudly(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            export = Path(d) / "review.json"
            export.write_text(json.dumps([_task("Drop", None)]), encoding="utf-8")
            with self.assertRaises(SystemExit):
                isl.import_cli([str(export), "--merge-into", str(Path(d) / "nope")])


if __name__ == "__main__":
    unittest.main()
