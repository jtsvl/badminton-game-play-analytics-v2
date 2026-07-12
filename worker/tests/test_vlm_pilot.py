"""vlm_prelabel_pilot.py — the pure sampling/parsing helpers (the rest is measurement glue).

The script lives in worker/scripts (not the package); imported here by file path. All heavy
deps (mlx-vlm, ffmpeg) are lazy inside main(), so the import stays clean on any install.
"""
import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vlm_prelabel_pilot.py"
spec = importlib.util.spec_from_file_location("vlm_prelabel_pilot", _SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def _rows(counts: dict):
    rows = []
    for cls, n in counts.items():
        rows.extend({"event_id": f"{cls}-{i}", "shot_type": cls} for i in range(n))
    return rows


class TestStratifiedSample(unittest.TestCase):
    def test_min_per_class_and_determinism(self):
        rows = _rows({"drop": 40, "net": 4, "serve": 20})
        a = pilot.stratified_sample(rows, per_class=15, seed=7)
        b = pilot.stratified_sample(rows, per_class=15, seed=7)
        self.assertEqual(a, b)  # deterministic
        by_cls = {}
        for r in a:
            by_cls.setdefault(r["shot_type"], []).append(r)
        self.assertEqual(len(by_cls["drop"]), 15)
        self.assertEqual(len(by_cls["net"]), 4)      # min(15, n) — take all of a rare class
        self.assertEqual(len(by_cls["serve"]), 15)

    def test_different_seed_different_sample(self):
        rows = _rows({"drop": 40})
        a = pilot.stratified_sample(rows, per_class=5, seed=1)
        b = pilot.stratified_sample(rows, per_class=5, seed=2)
        self.assertNotEqual(a, b)


class TestParseVlmAnswer(unittest.TestCase):
    def test_parses_json_with_surrounding_prose(self):
        got = pilot.parse_vlm_answer(
            'Sure! Here is my answer:\n```json\n{"label": "Netzschlag", "confidence": 0.7}\n```')
        self.assertEqual(got, ("net", 0.7))

    def test_unmapped_label_or_garbage_returns_none(self):
        self.assertIsNone(pilot.parse_vlm_answer('{"label": "Schmetterball", "confidence": 1}'))
        self.assertIsNone(pilot.parse_vlm_answer("I am not sure about this clip."))

    def test_confidence_clamped_to_unit_interval(self):
        self.assertEqual(pilot.parse_vlm_answer('{"label": "Clear", "confidence": 7}'),
                         ("clear", 1.0))


if __name__ == "__main__":
    unittest.main()
