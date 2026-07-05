"""stroke.model — BST4 (4-slot BST_CG_AP generalization) + collision-safe vendor import.

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


if __name__ == "__main__":
    unittest.main()
