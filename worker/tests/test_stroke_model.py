"""stroke.model + stroke.train torch section — BST4, vendor import, LOGO-CV loop.

NEW PRECEDENT, deliberately: these tests import torch (no other test does). They self-skip
unless torch + positional-encodings are installed AND the vendored BST checkout exists, so the
suite still collects green on light installs. The vendored classes are upstream-frozen (SHA
pinned in worker/training/bst_stroke_train.md); what needs testing is OUR composition: the
presence-masked 4-way AimPlayer/CleanGate generalization, the sys.modules['model'] collision
handling (TrackNetV3 owns the same top-level name), and checkpoint round-trips.
"""
import os
import sys
import tempfile
import types
import unittest
from importlib.util import find_spec
from pathlib import Path

_VENDOR = Path(os.environ.get(
    "BST_REPO",
    str(Path(__file__).resolve().parents[2] / "worker/.cache/_vendor/BST/stroke_classification"),
))
_READY = bool(find_spec("torch")) and bool(find_spec("positional_encodings")) \
    and (_VENDOR / "model" / "bst.py").exists()
_SKIP = "needs torch + positional-encodings (uv sync --extra ml) and the vendored BST checkout"


@unittest.skipUnless(_READY, _SKIP)
class TestImportVendor(unittest.TestCase):
    def test_returns_classes_and_restores_colliding_module(self):
        from worker.stroke import model as m

        m._VENDOR_CACHE.clear()
        fake = types.ModuleType("model")  # stand-in for TrackNetV3's top-level `model`
        sys.modules["model"] = fake
        path_before = list(sys.path)
        try:
            classes = m.import_vendor()
            for name in ("TCN", "FeedForward", "MLP", "MLP_Head", "TransformerEncoder",
                         "CrossTransformerLayer", "MultiHeadCrossAttention"):
                self.assertIn(name, classes)
            self.assertIs(sys.modules.get("model"), fake)  # collision victim restored
            self.assertEqual(sys.path, path_before)
        finally:
            sys.modules.pop("model", None)
        # imported classes stay usable after the sys.modules cleanup
        import torch

        tcn = classes["TCN"](4, [8, 8], kernel_size=3, drop_p=0.0)
        out = tcn(torch.zeros(2, 4, 16))
        self.assertEqual(tuple(out.shape), (2, 8, 16))

    def test_missing_repo_raises_with_instructions(self):
        from worker.stroke import model as m

        with self.assertRaises(RuntimeError) as ctx:
            m.import_vendor(repo="/nonexistent/BST")
        self.assertIn("BST", str(ctx.exception))


@unittest.skipUnless(_READY, _SKIP)
class TestBST4Helpers(unittest.TestCase):
    def test_slot_presence_from_all_zero_joints(self):
        import torch

        from worker.stroke.model import BST4

        joints = torch.zeros(1, 16, 4, 34)
        joints[0, 3, 1, 5] = 0.2
        self.assertEqual(BST4.slot_presence(joints).tolist(), [[False, True, False, False]])

    def test_aim_weights_zero_absent_and_sum_to_one(self):
        import torch

        from worker.stroke.model import BST4

        sims = torch.tensor([[0.5, 0.5, 0.9, 0.5]])
        present = torch.tensor([[True, True, False, True]])
        w = BST4.aim_weights(sims, present)
        self.assertEqual(w[0, 2].item(), 0.0)  # absent slot: exactly 0, despite max sim
        self.assertAlmostEqual(w.sum().item(), 1.0, places=6)
        self.assertAlmostEqual(w[0, 0].item(), 1 / 3, places=6)

    def test_clean_input_min_over_present_slots_only(self):
        import torch

        from worker.stroke.model import BST4

        cross = torch.tensor([[[1.0, 5.0], [2.0, 3.0], [-9.0, -9.0], [4.0, 4.0]]])
        present = torch.tensor([[True, True, False, True]])
        out = BST4.clean_input(cross, present)
        self.assertEqual(out.tolist(), [[1.0, 3.0]])  # absent slot's -9s never win the min

    def test_clean_input_all_absent_guard(self):
        import torch

        from worker.stroke.model import BST4

        out = BST4.clean_input(torch.ones(1, 4, 3), torch.zeros(1, 4, dtype=torch.bool))
        self.assertEqual(out.tolist(), [[0.0, 0.0, 0.0]])


@unittest.skipUnless(_READY, _SKIP)
class TestBST4Forward(unittest.TestCase):
    @staticmethod
    def _inputs(b=2, in_dim=34, seed=0):
        import torch

        g = torch.Generator().manual_seed(seed)
        return (torch.randn(b, 16, 4, in_dim, generator=g),
                torch.randn(b, 16, 2, generator=g),
                torch.randn(b, 16, 4, 2, generator=g))

    def test_cg_ap_logits_shape_and_finite(self):
        import torch

        from worker.stroke.model import BST4

        model = BST4(in_dim=34, seq_len=16, n_class=7, variant="cg_ap")
        model.eval()
        with torch.no_grad():
            out = model(*self._inputs())
        self.assertEqual(tuple(out.shape), (2, 7))
        self.assertTrue(torch.isfinite(out).all())

    def test_bst0_has_no_ppf_or_clean_gate_params(self):
        from worker.stroke.model import BST4

        names = [n for n, _ in BST4(variant="bst0").named_parameters()]
        self.assertFalse(any("mlp_positions" in n or "mlp_clean" in n for n in names))

    def test_bst0_logits_invariant_to_positions(self):
        import torch

        from worker.stroke.model import BST4

        model = BST4(variant="bst0")
        model.eval()
        joints, shuttle, pos = self._inputs()
        with torch.no_grad():
            a = model(joints, shuttle, pos)
            b = model(joints, shuttle, pos * 0 + 9.9)
        self.assertTrue(torch.equal(a, b))

    def test_unknown_variant_raises(self):
        from worker.stroke.model import BST4

        with self.assertRaises(ValueError):
            BST4(variant="cg")


@unittest.skipUnless(_READY, _SKIP)
class TestCheckpointRoundTrip(unittest.TestCase):
    def test_identical_logits_after_reload(self):
        import torch

        from worker.stroke.model import BST4, load_checkpoint, save_checkpoint

        cfg = {"in_dim": 34, "seq_len": 16, "n_class": 7, "variant": "cg_ap"}
        model = BST4(**cfg)
        model.eval()
        inputs = TestBST4Forward._inputs()
        with torch.no_grad():
            want = model(*inputs)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bst.pt"
            save_checkpoint(path, model, cfg)
            loaded, cfg_back = load_checkpoint(path)
        self.assertEqual(cfg_back["variant"], "cg_ap")
        with torch.no_grad():
            got = loaded(*inputs)
        self.assertTrue(torch.equal(want, got))


@unittest.skipUnless(_READY, _SKIP)
class TestOverfitSmoke(unittest.TestCase):
    def test_memorizes_32_random_samples(self):
        """From-scratch trainability: the composed model must overfit a tiny random set.
        drop_p=0 so the check is about wiring, not regularization."""
        import torch

        from worker.stroke.model import BST4

        torch.manual_seed(7)
        joints = torch.randn(32, 16, 4, 34)
        shuttle = torch.randn(32, 16, 2)
        pos = torch.rand(32, 16, 4, 2)
        y = torch.arange(32) % 7
        model = BST4(in_dim=34, seq_len=16, n_class=7, variant="cg_ap", drop_p=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(150):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(joints, shuttle, pos), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(joints, shuttle, pos).argmax(1) == y).float().mean().item()
        self.assertGreaterEqual(acc, 0.9)


@unittest.skipUnless(_READY, _SKIP)
class TestPredictProbs(unittest.TestCase):
    def test_batch_independent_softmax(self):
        """predict_probs must run in eval mode — train-mode BatchNorm would make the OOF
        probabilities depend on batch composition."""
        import numpy as np
        import torch

        from worker.stroke.model import BST4
        from worker.stroke.train import predict_probs

        torch.manual_seed(3)
        model = BST4(in_dim=34, drop_p=0.0)
        model.train()  # deliberately wrong incoming mode; predict_probs must switch to eval
        rng = np.random.default_rng(3)
        J = rng.normal(size=(8, 16, 4, 34)).astype(np.float32)
        S = rng.normal(size=(8, 16, 2)).astype(np.float32)
        P = rng.random(size=(8, 16, 4, 2)).astype(np.float32)
        all_at_once = predict_probs(model, J, S, P, device="cpu", batch=8)
        one_by_one = predict_probs(model, J, S, P, device="cpu", batch=1)
        self.assertEqual(all_at_once.shape, (8, 7))
        np.testing.assert_allclose(all_at_once, one_by_one, atol=1e-5)
        np.testing.assert_allclose(all_at_once.sum(axis=1), np.ones(8), atol=1e-5)


@unittest.skipUnless(_READY, _SKIP)
class TestRunLogoCV(unittest.TestCase):
    """Synthetic 3-video / 45-row dataset, 2 CPU epochs: protocol correctness, not accuracy."""

    @staticmethod
    def _dataset():
        import numpy as np
        import polars as pl

        rng = np.random.default_rng(11)
        n = 45
        joints = rng.normal(size=(n, 4, 16, 17, 2)).astype(np.float32)
        shuttle = rng.random(size=(n, 16, 2)).astype(np.float32)
        positions = rng.random(size=(n, 4, 16, 2)).astype(np.float32)
        shot_types = (["clear", "smash", "drop", "net", "drive", "lift", "serve"] * 7)[:n]
        is_cal = [i % 9 == 0 for i in range(n)]  # 5 calibration rows
        shot_types = ["unknown" if c else s for s, c in zip(shot_types, is_cal)]
        meta = pl.DataFrame({
            "event_id": [f"e{i}" for i in range(n)],
            "video_id": [f"g{i % 3}" for i in range(n)],
            "t_s": [float(i) for i in range(n)],
            "shot_type": shot_types,
            "is_calibration": is_cal,
        })
        return {"joints": joints, "shuttle": shuttle, "positions": positions}, meta

    def test_oof_complete_and_report_consistent(self):
        import numpy as np

        from worker.stroke.train import GATE_MEAN_ACCURACY, run_logo_cv

        data, meta = self._dataset()
        cfg = {"variant": "cg_ap", "pose_style": "J_only", "lr": 1e-3,
               "epochs": 2, "batch_size": 8, "seed": 1, "device": "cpu"}
        oof, report = run_logo_cv(data, meta, cfg)

        # every row — confident AND calibration — gets exactly one OOF prediction
        self.assertEqual(oof.height, 45)
        probs = oof.select([f"prob_{c}" for c in
                            ("clear", "smash", "drop", "net", "drive", "lift", "serve")])
        np.testing.assert_allclose(probs.to_numpy().sum(axis=1), np.ones(45), atol=1e-5)
        self.assertEqual(sorted(oof["fold"].unique().to_list()), [0, 1, 2])
        # fold assignment matches video_id (holdout game = the fold's videos)
        for vid, fold in zip(oof["video_id"], oof["fold"]):
            self.assertEqual(f"g{fold}", vid)

        for key in ("classes", "folds", "mean_accuracy", "std_accuracy", "pooled_accuracy",
                    "per_class_recall", "confusion", "hard_pairs", "tau",
                    "unsicher_abstention", "unknown_rate_per_match", "gate_passed", "device"):
            self.assertIn(key, report)
        self.assertEqual(len(report["folds"]), 3)
        # per-fold n counts only confident rows (45 - 5 calibration = 40 across 3 games)
        self.assertEqual(sum(f["n"] for f in report["folds"]), 40)
        want_mean = float(np.mean([f["accuracy"] for f in report["folds"]]))
        self.assertAlmostEqual(report["mean_accuracy"], want_mean, places=9)
        self.assertEqual(report["gate_passed"], want_mean >= GATE_MEAN_ACCURACY)
        # abstention/unknown rates live in [0, 1]
        self.assertTrue(0.0 <= report["unsicher_abstention"] <= 1.0)
        self.assertEqual(len(report["unknown_rate_per_match"]), 3)


if __name__ == "__main__":
    unittest.main()
