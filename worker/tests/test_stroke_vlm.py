"""stroke.vlm_eval pure logic — the VideoJev-4B zero-shot stroke pilot's deterministic half.

Torch-free by design (numpy/polars imported lazily inside the tests, precedent
test_stroke_train.py): presets are the only knobs that exist and are frozen in code; the
dev/eval split keeps prompt selection on one game; the eval run must read its settings from a
dev report and refuse overrides; the report mirrors run_logo_cv's keys where comparable so the
pilot number sits next to the BST LOGO number without translation.
(specs/features/stroke-vlm-pilot.md)
"""
import json
import tempfile
import unittest
from pathlib import Path

from worker.stroke import vlm_eval as ve
from worker.stroke.train import CLASSES


def _probs(pred: str, true: str | None = None, p: float = 0.6, q: float = 0.3):
    """7-vector in CLASSES order: p on `pred`, q on `true` (if different), rest spread evenly."""
    out = [0.0] * len(CLASSES)
    others = [c for c in CLASSES if c not in (pred, true)]
    if true is None or true == pred:
        rest = (1.0 - p) / len(others)
        for c in CLASSES:
            out[CLASSES.index(c)] = p if c == pred else rest
    else:
        rest = (1.0 - p - q) / len(others)
        for c in CLASSES:
            out[CLASSES.index(c)] = p if c == pred else q if c == true else rest
    return out


def _row(event_id: str, video_id: str, shot_type: str, probs, preset: str = "plain_fps10"):
    base = {"event_id": event_id, "match_id": "m1", "video_id": video_id, "t_s": 1.0,
            "shot_type": shot_type, "is_calibration": shot_type == "unknown",
            "clip_path": f"stroke/{video_id}/{event_id}.mp4", "join_method": "event_id"}
    pr = ve.PRESETS[preset]
    option_order = [probs[CLASSES.index(c)] for c in pr.option_classes]
    return ve.prediction_row(base, pr, option_order, n_frames_expected=15, wall_s=1.0)


class TestPresets(unittest.TestCase):
    def test_table_shape(self):
        names = list(ve.PRESETS)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 3 * len(ve.FPS_GRID) + 1)   # 3 prompts x fps grid + probe
        for name, p in ve.PRESETS.items():
            self.assertEqual(p.name, name)
            ve.validate_preset(p)   # 7 options, option_classes is a permutation of CLASSES
            self.assertEqual(len(p.options), len(CLASSES))
            self.assertEqual(sorted(p.option_classes), sorted(CLASSES))
            self.assertIn(p.fps, ve.FPS_GRID)
            self.assertIsNone(p.max_frames)

    def test_order_is_prompt_major_then_fps(self):
        self.assertEqual(list(ve.PRESETS)[:4],
                         ["plain_fps2", "plain_fps10", "plain_fps30", "described_fps2"])
        self.assertEqual(list(ve.PRESETS)[-1], "plain_rev_fps10")

    def test_reversed_probe_is_marked_and_reversed(self):
        rev, plain = ve.PRESETS["plain_rev_fps10"], ve.PRESETS["plain_fps10"]
        self.assertTrue(rev.probe)
        self.assertFalse(plain.probe)
        self.assertEqual(rev.option_classes, tuple(reversed(plain.option_classes)))
        self.assertEqual(rev.options, tuple(reversed(plain.options)))
        self.assertEqual(rev.query, plain.query)

    def test_validate_rejects_bad_tables(self):
        p = ve.PRESETS["plain_fps2"]
        with self.assertRaises(ValueError):
            ve.validate_preset(ve.Preset("x", p.query, p.options[:6], p.option_classes[:6], 2.0))
        with self.assertRaises(ValueError):
            ve.validate_preset(ve.Preset("x", p.query, p.options,
                                         ("clear",) * len(CLASSES), 2.0))
        with self.assertRaises(ValueError):
            ve.validate_preset(ve.Preset("x", p.query, p.options, p.option_classes, 0.0))

    def test_build_question_shape(self):
        q = ve.build_question(ve.PRESETS["described_fps10"])
        self.assertEqual(set(q), {"id", "task", "query", "options"})
        self.assertEqual(q["task"], "choice")
        self.assertEqual(len(q["options"]), 7)
        self.assertIsInstance(q["options"], list)
        self.assertIn("smash", q["options"][1])   # CLASSES order, smash is index 1

    def test_no_metadata_in_queries(self):
        # pixels only: no serve hint / rally index can leak into the prompt
        for p in ve.PRESETS.values():
            self.assertNotIn("Aufschlag", p.query)
            self.assertNotIn("rally", p.query.lower().replace("badminton", ""))


class TestProbsInClassOrder(unittest.TestCase):
    def test_reversed_preset_round_trips(self):
        import numpy as np

        probs = _probs("smash", "clear")
        rev = ve.PRESETS["plain_rev_fps10"]
        option_order = [probs[CLASSES.index(c)] for c in rev.option_classes]
        back = ve.probs_in_class_order(option_order, rev)
        np.testing.assert_allclose(back, probs)
        self.assertAlmostEqual(float(back.sum()), 1.0)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            ve.probs_in_class_order([0.5, 0.5], ve.PRESETS["plain_fps2"])

    def test_non_finite_raises(self):
        with self.assertRaises(ValueError):
            ve.probs_in_class_order([float("nan")] + [0.0] * 6, ve.PRESETS["plain_fps2"])


class TestExpectedFrames(unittest.TestCase):
    """Mirrors upstream videojev_temporal.time_grid: ceil(duration*fps) frames on a k/fps grid."""

    def test_grid_counts(self):
        self.assertEqual(ve.expected_frames(1.4, 2.0), 3)
        self.assertEqual(ve.expected_frames(2.0, 10.0), 20)
        self.assertEqual(ve.expected_frames(1.4, 10.0), 14)
        self.assertEqual(ve.expected_frames(1.4, 30.0), 42)
        # upstream ceils the float product; 1.4*fps happens to round exactly for 10 and 30

    def test_max_frames_caps(self):
        self.assertEqual(ve.expected_frames(2.0, 30.0, max_frames=16), 16)
        self.assertEqual(ve.expected_frames(2.0, 2.0, max_frames=16), 4)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            ve.expected_frames(0.0, 2.0)
        with self.assertRaises(ValueError):
            ve.expected_frames(1.4, 0.0)


class TestPredictionRow(unittest.TestCase):
    def test_row_carries_class_order_probs_pred_and_pmax(self):
        probs = _probs("drop", "net", p=0.55, q=0.35)
        row = _row("e1", "g1", "net", probs, preset="plain_rev_fps10")
        self.assertEqual(row["pred"], "drop")
        self.assertAlmostEqual(row["p_max"], 0.55)
        self.assertEqual(row["preset"], "plain_rev_fps10")
        self.assertEqual(row["n_frames_expected"], 15)
        self.assertAlmostEqual(row["probs"][CLASSES.index("net")], 0.35)


class TestLoadTasksAndClipIndex(unittest.TestCase):
    @staticmethod
    def _tasks(game, items):
        return [{"data": {"video": f"/data/local-files/?d=stroke/{game}/{fname}", "game": game,
                          "rally_idx": 1, "hit_idx": i + 1, "hit_t_s": t, "first_of_rally": i == 0,
                          "serve_hint": "", "event_id": eid, "video_id": vid, "match_id": "m1"}}
                for i, (fname, t, eid, vid) in enumerate(items)]

    def test_load_tasks_parses_relpath_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tasks_G1.json").write_text(json.dumps(self._tasks(
                "G1", [("G1_r01_h01_0001.00s_aaaaaaaa.mp4", 1.0, "aaaaaaaa-1", "v1")])))
            (root / "tasks_G2.json").write_text(json.dumps(self._tasks(
                "G2", [("G2_r01_h01_0002.00s_bbbbbbbb.mp4", 2.0, "bbbbbbbb-1", "v2")])))
            tasks = ve.load_tasks(root)
        self.assertEqual(set(tasks), {"aaaaaaaa-1", "bbbbbbbb-1"})
        self.assertEqual(tasks["aaaaaaaa-1"]["clip"], "stroke/G1/G1_r01_h01_0001.00s_aaaaaaaa.mp4")
        self.assertEqual(tasks["aaaaaaaa-1"]["video_id"], "v1")
        self.assertAlmostEqual(tasks["bbbbbbbb-1"]["hit_t_s"], 2.0)
        self.assertEqual(tasks["bbbbbbbb-1"]["game"], "G2")

    def test_clip_index_groups_by_event8(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for game, fname in (("G1", "G1_r01_h01_0001.00s_aaaaaaaa.mp4"),
                                ("G1", "G1_r02_h01_0009.00s_cccccccc.mp4"),
                                ("G2", "G2_r01_h01_0002.00s_cccccccc.mp4")):
                (root / "stroke" / game).mkdir(parents=True, exist_ok=True)
                (root / "stroke" / game / fname).write_bytes(b"")
            index = ve.clip_index_from_files(root)
        self.assertEqual(index["aaaaaaaa"], ["stroke/G1/G1_r01_h01_0001.00s_aaaaaaaa.mp4"])
        self.assertEqual(len(index["cccccccc"]), 2)   # ambiguous short id


class TestJoinLabelsToClips(unittest.TestCase):
    LABELS = [
        {"event_id": "aaaaaaaa-1", "match_id": "m1", "video_id": "v1", "t_s": 1.0, "shot_type": "clear", "hitter": None},
        {"event_id": "bbbbbbbb-2", "match_id": "m1", "video_id": "v1", "t_s": 5.02, "shot_type": "unknown", "hitter": None},
        {"event_id": "cccccccc-3", "match_id": "m1", "video_id": "v2", "t_s": 9.0, "shot_type": "drop", "hitter": None},
        {"event_id": "dddddddd-4", "match_id": "m1", "video_id": "v2", "t_s": 40.0, "shot_type": "lift", "hitter": None},
    ]
    TASKS = {
        "aaaaaaaa-1": {"clip": "stroke/G1/a.mp4", "video_id": "v1", "match_id": "m1", "hit_t_s": 1.0, "game": "G1"},
        "bbbbbbbb-9": {"clip": "stroke/G1/b.mp4", "video_id": "v1", "match_id": "m1", "hit_t_s": 5.0, "game": "G1"},
    }
    INDEX = {"cccccccc": ["stroke/G2/c.mp4"], "dddddddd": ["stroke/G2/d1.mp4", "stroke/G2/d2.mp4"]}

    def test_three_join_paths_and_missing_report(self):
        rows, missing = ve.join_labels_to_clips(self.LABELS, self.TASKS, self.INDEX)
        by_id = {r["event_id"]: r for r in rows}
        self.assertEqual(by_id["aaaaaaaa-1"]["join_method"], "event_id")
        self.assertEqual(by_id["aaaaaaaa-1"]["clip_path"], "stroke/G1/a.mp4")
        self.assertEqual(by_id["bbbbbbbb-2"]["join_method"], "t_s")      # re-exported ids, same hit
        self.assertEqual(by_id["bbbbbbbb-2"]["clip_path"], "stroke/G1/b.mp4")
        self.assertTrue(by_id["bbbbbbbb-2"]["is_calibration"])
        self.assertEqual(by_id["cccccccc-3"]["join_method"], "filename")
        self.assertFalse(by_id["cccccccc-3"]["is_calibration"])
        self.assertEqual([m["event_id"] for m in missing], ["dddddddd-4"])  # ambiguous, never guessed
        self.assertIn("reason", missing[0])

    def test_t_s_fallback_respects_tolerance(self):
        labels = [dict(self.LABELS[1], t_s=5.2)]   # 0.2 s off: too far for the 0.05 s window
        rows, missing = ve.join_labels_to_clips(labels, self.TASKS, {})
        self.assertEqual(rows, [])
        self.assertEqual(len(missing), 1)

    def test_alien_shot_type_raises(self):
        with self.assertRaises(ValueError):
            ve.join_labels_to_clips([dict(self.LABELS[0], shot_type="no_hit")], self.TASKS, {})


class TestSplitRows(unittest.TestCase):
    ROWS = [{"video_id": "g1"}, {"video_id": "g2"}, {"video_id": "g1"}, {"video_id": "g3"}]

    def test_dev_and_eval_partition(self):
        dev = ve.split_rows(self.ROWS, "g1", "dev")
        ev = ve.split_rows(self.ROWS, "g1", "eval")
        self.assertEqual(len(dev), 2)
        self.assertEqual({r["video_id"] for r in ev}, {"g2", "g3"})
        self.assertEqual(len(dev) + len(ev), len(self.ROWS))

    def test_unknown_dev_id_or_split_raises(self):
        with self.assertRaises(ValueError):
            ve.split_rows(self.ROWS, "nope", "dev")
        with self.assertRaises(ValueError):
            ve.split_rows(self.ROWS, "g1", "all")


class TestSelectPreset(unittest.TestCase):
    def test_argmax_with_ties_in_table_order(self):
        self.assertEqual(ve.select_preset({"plain_fps2": 0.3, "context_fps10": 0.5,
                                           "described_fps10": 0.5}), "described_fps10")

    def test_probe_presets_are_never_selected(self):
        self.assertEqual(ve.select_preset({"plain_rev_fps10": 0.9, "plain_fps2": 0.3}), "plain_fps2")

    def test_empty_or_unknown_raises(self):
        with self.assertRaises(ValueError):
            ve.select_preset({})
        with self.assertRaises(ValueError):
            ve.select_preset({"made_up": 0.9})


class TestFrozenSettings(unittest.TestCase):
    REPORT = {"split": "dev", "dev_video_id": "g5",
              "selected": {"preset": "described_fps10", "fps": 10.0, "max_frames": None,
                           "dtype": "fp32", "context": 32768, "attn": "sdpa", "dev_accuracy": 0.41}}

    def test_returns_exactly_the_frozen_keys(self):
        s = ve.frozen_settings(self.REPORT, "g5")
        self.assertEqual(set(s), {"preset", "fps", "max_frames", "dtype", "context", "attn"})
        self.assertEqual(s["preset"], "described_fps10")
        self.assertEqual(s["fps"], 10.0)

    def test_rejects_non_dev_report(self):
        with self.assertRaises(ValueError):
            ve.frozen_settings(dict(self.REPORT, split="eval"), "g5")

    def test_rejects_other_dev_game(self):
        with self.assertRaises(ValueError):
            ve.frozen_settings(self.REPORT, "g1")

    def test_rejects_unknown_or_probe_preset(self):
        bad = dict(self.REPORT, selected=dict(self.REPORT["selected"], preset="made_up"))
        with self.assertRaises(ValueError):
            ve.frozen_settings(bad, "g5")
        probe = dict(self.REPORT, selected=dict(self.REPORT["selected"], preset="plain_rev_fps10"))
        with self.assertRaises(ValueError):
            ve.frozen_settings(probe, "g5")

    def test_rejects_fps_inconsistent_with_preset(self):
        bad = dict(self.REPORT, selected=dict(self.REPORT["selected"], fps=2.0))
        with self.assertRaises(ValueError):
            ve.frozen_settings(bad, "g5")


class TestDecisionTier(unittest.TestCase):
    RECALL_ALL = {c: 0.5 for c in CLASSES}
    PMAX_OK = {"confident": 0.7, "unsicher": 0.4}

    def test_boundaries(self):
        self.assertEqual(ve.decision_tier(0.34, self.RECALL_ALL, self.PMAX_OK, 0.5), "closed")
        self.assertEqual(ve.decision_tier(0.35, self.RECALL_ALL, self.PMAX_OK, 0.5), "labeling_assist")
        self.assertEqual(ve.decision_tier(0.55, self.RECALL_ALL, self.PMAX_OK, 0.8), "finetune_candidate")
        self.assertEqual(ve.decision_tier(0.70, self.RECALL_ALL, self.PMAX_OK, 0.9), "gate")

    def test_top2_alone_can_lift_to_assist(self):
        self.assertEqual(ve.decision_tier(0.30, self.RECALL_ALL, self.PMAX_OK, 0.75), "labeling_assist")

    def test_finetune_needs_recall_breadth_and_honest_confidence(self):
        narrow = dict(self.RECALL_ALL, net=0.0, drive=0.0)   # only 5 classes with recall
        self.assertEqual(ve.decision_tier(0.60, narrow, self.PMAX_OK, 0.9), "labeling_assist")
        overconfident = {"confident": 0.6, "unsicher": 0.65}
        self.assertEqual(ve.decision_tier(0.60, self.RECALL_ALL, overconfident, 0.9), "labeling_assist")
        six = dict(self.RECALL_ALL, net=None)   # NaN recall (no support) does not count, 6 remain
        self.assertEqual(ve.decision_tier(0.60, six, self.PMAX_OK, 0.9), "finetune_candidate")

    def test_none_accuracy_gives_none(self):
        self.assertIsNone(ve.decision_tier(None, self.RECALL_ALL, self.PMAX_OK, None))


class TestAggregateReport(unittest.TestCase):
    def _rows(self):
        rows = []
        # g1: all 7 classes once; net->drop, drive->lift, serve->lift wrong (true class 2nd) -> 4/7
        wrong1 = {"net": "drop", "drive": "lift", "serve": "lift"}
        for i, c in enumerate(CLASSES):
            pred = wrong1.get(c, c)
            rows.append(_row(f"g1-{i}", "g1", c, _probs(pred, c, p=0.6, q=0.3)))
        # g2: smash->clear, drop->net wrong -> 5/7
        wrong2 = {"smash": "clear", "drop": "net"}
        for i, c in enumerate(CLASSES):
            pred = wrong2.get(c, c)
            rows.append(_row(f"g2-{i}", "g2", c, _probs(pred, c, p=0.8, q=0.1)))
        # two Unsicher rows, one per game, low confidence
        rows.append(_row("g1-u", "g1", "unknown", _probs("drop", p=0.2)))
        rows.append(_row("g2-u", "g2", "unknown", _probs("lift", p=0.2)))
        return rows

    def test_report_keys_and_numbers(self):
        rep = ve.aggregate_report(self._rows(), split="eval")
        for key in ("split", "n", "n_confident", "n_unknown", "per_game", "mean_accuracy",
                    "std_accuracy", "pooled_accuracy", "top2_accuracy", "per_class_recall",
                    "confusion", "hard_pairs", "tau", "unsicher_abstention",
                    "unknown_rate_per_match", "pmax_median", "pred_share", "pred_share_max",
                    "gate_passed", "decision_tier"):
            self.assertIn(key, rep, key)
        self.assertEqual(rep["n"], 16)
        self.assertEqual(rep["n_confident"], 14)
        self.assertEqual(rep["n_unknown"], 2)
        self.assertEqual([g["video_id"] for g in rep["per_game"]], ["g1", "g2"])
        self.assertEqual([g["n"] for g in rep["per_game"]], [7, 7])   # unknown rows excluded
        self.assertAlmostEqual(rep["per_game"][0]["accuracy"], 4 / 7)
        self.assertAlmostEqual(rep["per_game"][1]["accuracy"], 5 / 7)
        self.assertAlmostEqual(rep["mean_accuracy"], (4 / 7 + 5 / 7) / 2)
        self.assertAlmostEqual(rep["pooled_accuracy"], 9 / 14)       # 0.643: finetune band, below gate
        self.assertAlmostEqual(rep["top2_accuracy"], 1.0)            # true class always 1st or 2nd
        self.assertGreaterEqual(rep["top2_accuracy"], rep["pooled_accuracy"])
        self.assertAlmostEqual(rep["per_class_recall"]["net"], 0.5)   # 1 of 2 net rows right
        self.assertAlmostEqual(rep["per_class_recall"]["clear"], 1.0)
        self.assertEqual(rep["hard_pairs"]["drop/net"]["labels"], ["drop", "net"])
        self.assertEqual(rep["hard_pairs"]["drop/net"]["matrix"][1][0], 1)   # net predicted as drop once
        self.assertFalse(rep["gate_passed"])
        self.assertEqual(rep["decision_tier"], "finetune_candidate")
        # tau: sweep_tau on confident rows; the two Unsicher rows (p_max 0.2) abstain below any tau > 0.2
        self.assertIn("tau", rep["tau"])
        self.assertAlmostEqual(rep["pmax_median"]["unsicher"], 0.2)
        self.assertGreater(rep["pmax_median"]["confident"], rep["pmax_median"]["unsicher"])
        self.assertEqual(set(rep["unknown_rate_per_match"]), {"g1", "g2"})
        self.assertAlmostEqual(sum(rep["pred_share"].values()), 1.0)
        self.assertEqual(rep["split"], "eval")

    def test_nan_recall_becomes_none(self):
        rows = [_row("a", "g1", "clear", _probs("clear")), _row("b", "g1", "clear", _probs("clear"))]
        rep = ve.aggregate_report(rows, split="dev")
        self.assertIsNone(rep["per_class_recall"]["net"])
        self.assertEqual(rep["per_game"][0]["per_class_recall"][CLASSES.index("net")], None)
        self.assertIsNone(rep["unsicher_abstention"])   # no Unsicher rows in the split
        self.assertIsNone(rep["pmax_median"]["unsicher"])

    def test_no_confident_rows_is_reported_not_crashed(self):
        rep = ve.aggregate_report([_row("u", "g1", "unknown", _probs("drop", p=0.2))], split="dev")
        self.assertEqual(rep["n_confident"], 0)
        self.assertIsNone(rep["pooled_accuracy"])
        self.assertIsNone(rep["decision_tier"])
        self.assertFalse(rep["gate_passed"])


class TestPlanRun(unittest.TestCase):
    DEV_REPORT = TestFrozenSettings.REPORT

    def test_dev_defaults_and_all_presets(self):
        plan = ve.plan_run("dev", preset=None, frozen_report=None, dev_video_id="g5",
                           dtype=None, attn=None, context=None)
        self.assertEqual(plan["presets"], list(ve.PRESETS))
        self.assertEqual((plan["dtype"], plan["attn"], plan["context"], plan["frozen"]),
                         ("fp32", "sdpa", ve.DEFAULT_CONTEXT, False))

    def test_dev_single_preset_and_overrides(self):
        plan = ve.plan_run("dev", preset="context_fps30", frozen_report=None, dev_video_id="g5",
                           dtype="bf16", attn="eager", context=16384)
        self.assertEqual(plan["presets"], ["context_fps30"])
        self.assertEqual((plan["dtype"], plan["attn"], plan["context"]), ("bf16", "eager", 16384))
        with self.assertRaises(ValueError):
            ve.plan_run("dev", preset="nope", frozen_report=None, dev_video_id="g5",
                        dtype=None, attn=None, context=None)
        with self.assertRaises(ValueError):   # a dev run must not carry a frozen report
            ve.plan_run("dev", preset=None, frozen_report=self.DEV_REPORT, dev_video_id="g5",
                        dtype=None, attn=None, context=None)

    def test_eval_reads_everything_from_the_dev_report(self):
        plan = ve.plan_run("eval", preset=None, frozen_report=self.DEV_REPORT, dev_video_id="g5",
                           dtype=None, attn=None, context=None)
        self.assertEqual(plan["presets"], ["described_fps10"])
        self.assertEqual((plan["dtype"], plan["attn"], plan["context"], plan["frozen"]),
                         ("fp32", "sdpa", 32768, True))

    def test_eval_refuses_overrides_and_missing_report(self):
        for kw in ({"preset": "plain_fps2"}, {"dtype": "bf16"}, {"attn": "eager"}, {"context": 8192}):
            base = {"preset": None, "dtype": None, "attn": None, "context": None, **kw}
            with self.assertRaises(ValueError, msg=str(kw)):
                ve.plan_run("eval", frozen_report=self.DEV_REPORT, dev_video_id="g5", **base)
        with self.assertRaises(ValueError):
            ve.plan_run("eval", preset=None, frozen_report=None, dev_video_id="g5",
                        dtype=None, attn=None, context=None)


class TestProbeSummary(unittest.TestCase):
    REF = [[0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01], [0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.01]]

    def test_pass_within_tolerance(self):
        dev = [[0.58, 0.22, 0.1, 0.05, 0.03, 0.01, 0.01], [0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.01]]
        s = ve.probe_summary(self.REF, dev)
        self.assertTrue(s["passed"])
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["argmax_agreement"], 2)
        self.assertAlmostEqual(s["max_abs_prob_diff"], 0.02)
        self.assertEqual(s["nonfinite"], 0)

    def test_fail_on_argmax_flip_or_large_drift_or_nan(self):
        flipped = [[0.2, 0.6, 0.1, 0.05, 0.03, 0.01, 0.01], self.REF[1]]
        self.assertFalse(ve.probe_summary(self.REF, flipped)["passed"])
        drift = [[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01], self.REF[1]]   # 0.1 > tol
        self.assertFalse(ve.probe_summary(self.REF, drift)["passed"])
        nan = [[float("nan")] + [0.0] * 6, self.REF[1]]
        s = ve.probe_summary(self.REF, nan)
        self.assertFalse(s["passed"])
        self.assertEqual(s["nonfinite"], 1)
        self.assertIsNone(s["max_abs_prob_diff"])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ve.probe_summary(self.REF, self.REF[:1])
        with self.assertRaises(ValueError):
            ve.probe_summary([], [])


class TestPredictionsFrame(unittest.TestCase):
    def test_columns_and_types(self):
        import polars as pl

        rows = [_row("a", "g1", "clear", _probs("clear")), _row("u", "g1", "unknown", _probs("drop", p=0.2))]
        df = ve.predictions_frame(rows)
        self.assertEqual(df.height, 2)
        for col in ("event_id", "video_id", "match_id", "t_s", "shot_type", "is_calibration",
                    "preset", "clip_path", "join_method", "n_frames_expected", "pred", "p_max",
                    "wall_s", *(f"prob_{c}" for c in CLASSES)):
            self.assertIn(col, df.columns, col)
        self.assertNotIn("probs", df.columns)
        self.assertEqual(df.schema["is_calibration"], pl.Boolean)
        self.assertEqual(df.schema["n_frames_expected"], pl.Int64)
        self.assertAlmostEqual(df["prob_clear"][0], 0.6)


if __name__ == "__main__":
    unittest.main()


class TestEvalCliWithFakeModel(unittest.TestCase):
    """The I/O shell end to end — labels + tasks + clip files in, predictions.jsonl/.parquet +
    report.json out — against a fake model, so the loop, the dev→eval handshake, --resume and
    the missing-clip guard are exercised here, not for the first time on the Mac."""

    GAMES = {"g5": ["clear", "smash", "unknown"], "g1": ["drop", "net", "lift", "serve", "unknown"]}

    def _labeling_root(self, root: Path, *, drop_clip_for: str | None = None):
        import polars as pl

        (root / "snapshot").mkdir(parents=True, exist_ok=True)   # a "local model dir" for the CLI
        rows, tasks = [], {}
        for vid, shots in self.GAMES.items():
            game = f"G_{vid}"
            (root / "stroke" / game).mkdir(parents=True, exist_ok=True)
            tasks[game] = []
            for i, shot in enumerate(shots):
                eid = f"{vid}-{i:02d}-{'x' * 8}"
                t = 10.0 * (i + 1)
                fname = f"{game}_r01_h{i + 1:02d}_{t:07.2f}s_{eid[:8]}.mp4"
                if eid != drop_clip_for:
                    (root / "stroke" / game / fname).write_bytes(b"\x00")
                rows.append({"event_id": eid, "match_id": "m1", "video_id": vid, "t_s": t,
                             "shot_type": shot, "hitter": None})
                tasks[game].append({"data": {"video": f"/data/local-files/?d=stroke/{game}/{fname}",
                                             "game": game, "rally_idx": 1, "hit_idx": i + 1,
                                             "hit_t_s": t, "first_of_rally": i == 0, "serve_hint": "",
                                             "event_id": eid, "video_id": vid, "match_id": "m1"}})
            (root / f"tasks_{game}.json").write_text(json.dumps(tasks[game]))
        pl.DataFrame(rows).write_parquet(root / "stroke_labels.parquet")
        return {r["event_id"]: r["shot_type"] for r in rows}

    @staticmethod
    def _fakes(truth: dict):
        """described_fps10 answers correctly with 0.9; every other preset says 'drop' at 0.5."""
        def resolve_device(requested="auto"):
            return "cpu"

        def score_clip(model, processor, clip_path, presets, *, fps, max_frames, context):
            eid8 = Path(clip_path).stem.rsplit("_", 1)[-1]
            true = next((s for e, s in truth.items() if e.startswith(eid8)), "drop")
            out = []
            for p in presets:
                target = true if (p.name == "described_fps10" and true != "unknown") else "drop"
                conf = 0.9 if p.name == "described_fps10" else 0.5
                probs_cls = _probs(target, p=conf)
                out.append((p, [probs_cls[CLASSES.index(c)] for c in p.option_classes], 14, 0.01))
            return out

        def run_probe(path, clips, preset, *, device, dtype, attn, revision, context, tol=0.05):
            ref = [[1 / 7] * 7 for _ in clips]
            summary = ve.probe_summary(ref, ref, tol=tol)
            summary.update(device=device, dtype=dtype, reference="cpu/fp32", preset=preset.name)
            return summary, (object(), object(), {"id": path, "revision": revision, "dtype": dtype,
                                                  "attn": attn, "device": device, "model_type": "fake"})

        def load_videojev(path, *, device, dtype="fp32", attn="sdpa", revision=None):
            return object(), object(), {"id": path, "revision": revision, "dtype": dtype,
                                        "attn": attn, "device": device, "model_type": "fake"}

        return {"resolve_device": resolve_device, "score_clip": score_clip,
                "run_probe": run_probe, "load_videojev": load_videojev}

    def _run(self, truth, argv):
        from unittest import mock

        from worker.stroke import vlm_model as vm

        with mock.patch.multiple(vm, **self._fakes(truth)):
            return ve.eval_cli(argv)

    def test_dev_then_frozen_eval_then_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            truth = self._labeling_root(root)
            dev_dir, eval_dir = root / "runs" / "dev", root / "runs" / "eval"
            base = ["--dev-video-id", "g5", "--labeling-root", str(root), "--model", str(root / "snapshot")]

            # dev: all presets, probe on; the fake makes described_fps10 the clear winner
            self.assertEqual(self._run(truth, ["--split", "dev", *base, "--out", str(dev_dir)]), 0)
            rep = json.loads((dev_dir / "report.json").read_text())
            self.assertEqual(rep["split"], "dev")
            self.assertEqual(rep["selected"]["preset"], "described_fps10")
            self.assertAlmostEqual(rep["selected"]["dev_accuracy"], 1.0)
            self.assertEqual(rep["selected"]["dtype"], "fp32")
            self.assertTrue(rep["probe"]["passed"])
            self.assertEqual(rep["labels"]["n_missing_clips"], 0)
            self.assertEqual(rep["labels"]["n_split"], 3)                 # g5 rows only
            self.assertEqual(rep["per_preset"]["plain_fps2"]["n_confident"], 2)
            self.assertAlmostEqual(rep["per_preset"]["plain_fps2"]["pooled_accuracy"], 0.0)
            self.assertIn("position_bias", rep)
            self.assertEqual(len(rep["presets_run"]), len(ve.PRESETS))
            lines = (dev_dir / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3 * len(ve.PRESETS))
            import polars as pl
            self.assertEqual(pl.read_parquet(dev_dir / "predictions.parquet").height, len(lines))

            # eval: settings come from the dev report; only the frozen preset runs
            self.assertEqual(self._run(truth, ["--split", "eval", *base, "--frozen-from", str(dev_dir),
                                               "--out", str(eval_dir)]), 0)
            rep = json.loads((eval_dir / "report.json").read_text())
            self.assertEqual(rep["presets_run"], ["described_fps10"])
            self.assertEqual(rep["settings"]["fps"], 10.0)
            self.assertEqual(rep["labels"]["n_split"], 5)                 # the other game(s)
            self.assertEqual(rep["labels"]["n_confident"], 4)
            self.assertAlmostEqual(rep["headline"]["pooled_accuracy"], 1.0)
            self.assertEqual(rep["headline"]["decision_tier"], "gate")
            per = rep["per_preset"]["described_fps10"]
            self.assertIsNotNone(per["unsicher_abstention"])                # the Unsicher row was scored
            self.assertEqual([g["video_id"] for g in per["per_game"]], ["g1"])

            # resume: a run with --limit 1 continued without the limit scores the rest, once
            part = root / "runs" / "part"
            self.assertEqual(self._run(truth, ["--split", "dev", *base, "--preset", "plain_fps2",
                                               "--limit", "1", "--probe", "0", "--out", str(part)]), 0)
            self.assertEqual(len((part / "predictions.jsonl").read_text().splitlines()), 1)
            self.assertEqual(self._run(truth, ["--split", "dev", *base, "--preset", "plain_fps2",
                                               "--probe", "0", "--resume", str(part)]), 0)
            self.assertEqual(len((part / "predictions.jsonl").read_text().splitlines()), 3)
            rep = json.loads((part / "report.json").read_text())
            self.assertIsNone(rep["probe"])
            self.assertTrue(any("NO cpu-vs-device probe" in c for c in rep["caveats"]))

    def test_hf_id_without_revision_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            truth = self._labeling_root(root)
            with self.assertRaises(SystemExit) as cm:
                self._run(truth, ["--split", "dev", "--dev-video-id", "g5", "--labeling-root", str(root),
                                  "--model", "SomeOrg/SomeModel", "--probe", "0"])
            self.assertEqual(cm.exception.code, 2)   # argparse error, nothing scored

    def test_missing_clip_blocks_unless_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            truth = self._labeling_root(root, drop_clip_for="g1-01-xxxxxxxx")
            base = ["--dev-video-id", "g5", "--labeling-root", str(root), "--model", str(root / "snapshot"),
                    "--probe", "0", "--preset", "plain_fps2"]
            # the hole is in g1 (eval): the dev split is unaffected ...
            self.assertEqual(self._run(truth, ["--split", "dev", *base, "--out", str(root / "r1")]), 0)
            # ... a dev report is needed to even start eval; forge the handshake via a real dev run
            dev_dir = root / "r2"
            self.assertEqual(self._run(truth, ["--split", "dev", "--dev-video-id", "g5", "--labeling-root",
                                               str(root), "--model", str(root / "snapshot"), "--probe", "0",
                                               "--out", str(dev_dir)]), 0)
            ev = ["--split", "eval", "--dev-video-id", "g5", "--labeling-root", str(root), "--model",
                  str(root / "snapshot"), "--probe", "0", "--frozen-from", str(dev_dir)]
            self.assertEqual(self._run(truth, [*ev, "--out", str(root / "r3")]), 1)
            self.assertFalse((root / "r3" / "report.json").exists())
            self.assertEqual(self._run(truth, [*ev, "--allow-missing-clips", "--out", str(root / "r4")]), 0)
            rep = json.loads((root / "r4" / "report.json").read_text())
            self.assertEqual(rep["labels"]["n_missing_clips"], 1)
            self.assertEqual(rep["labels"]["n_split"], 4)
