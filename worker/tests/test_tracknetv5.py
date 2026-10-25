"""TrackNetV5 backend (worker/shuttle/tracknetv5.py) — the pure parts (specs/features/shuttle-tracknetv5.md).

Input construction and heatmap decoding are hand-rolled (the vendored repo's `dataset`/`utils`/`models`
module names collide with TrackNetV3's and WASB's inside one `eval_shuttle --compare` process), so they are
pinned here against the upstream formula read from the pinned commit 9116161:
  frame → [R, G, B, D] with D = uint8(Σ_rgb |frame − median|) (wraps mod 256 above 255), 5 frames → 20
  channels, /255; non-overlapping 5-frame windows padded with the last frame; blob decode at a threshold.
Model loading is torch-only and runs on the Mac (convert-tracknetv5 --probe); registry + helpers here.
"""
import hashlib
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from worker.shuttle import tracknetv5 as t5
from worker.shuttle.base import checkpoint_tag, decode_heatmap, largest_blob_bbox, vendor_commit

try:
    import numpy as np
except ImportError:  # dev extra without the ml stack
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


class TestVariants(unittest.TestCase):
    def test_published_operating_points(self):
        self.assertEqual(t5.VARIANTS["lite"]["alpha"], 0.5)
        self.assertEqual(t5.VARIANTS["full"]["alpha"], 1.0)
        self.assertEqual(t5.VARIANTS["lite"]["thr"], 0.35)
        self.assertEqual(t5.VARIANTS["full"]["thr"], 0.30)
        self.assertEqual(t5.VARIANTS["lite"]["upstream_sha256"],
                         "c7ca025e5117d292f9efa60b3132d401434ca1afe26aba9353133ad2823e7a07")
        self.assertEqual(t5.VARIANTS["full"]["upstream_sha256"],
                         "cf9025285cdcd3089229a45a19ed8ed4075f6a4f459cbf99e450b1f968f7198d")

    def test_pin_and_shapes(self):
        self.assertEqual(t5.TRACKNETV5_SHA, "9116161772183e841bd1b51d6747c70f269b8b1c")
        self.assertEqual((t5.SEQ_LEN, t5.IN_CHANNELS, t5.MODEL_W, t5.MODEL_H), (5, 20, 512, 288))

    def test_default_model_paths(self):
        self.assertTrue(t5.default_model_path("lite").endswith("models/weights/shuttle/tracknetv5_lite.pt"))
        self.assertTrue(t5.default_model_path("full").endswith("models/weights/shuttle/tracknetv5.pt"))
        with self.assertRaises(ValueError):
            t5.default_model_path("huge")

    def test_threshold_resolution_order(self):
        # explicit env > converted checkpoint config > the variant's published sweep value
        self.assertEqual(t5.resolve_thr("lite", env_value="0.5", config={"thr": 0.4}), 0.5)
        self.assertEqual(t5.resolve_thr("lite", env_value=None, config={"thr": 0.4}), 0.4)
        self.assertEqual(t5.resolve_thr("lite", env_value="", config={}), 0.35)
        self.assertEqual(t5.resolve_thr("full", env_value=None, config=None), 0.30)


@unittest.skipUnless(np is not None, "numpy (ml extra) not installed")
class TestBackgroundDiff(unittest.TestCase):
    def test_wrap_reproduces_upstreams_uint8_cast(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = (255, 45, 0)                 # Σ|rgb − 0| = 300 → uint8 wraps to 44
        median = np.zeros((1, 1, 3), dtype=np.float64)
        self.assertEqual(int(t5.background_diff(rgb, median, "wrap")[0, 0]), 44)
        self.assertEqual(int(t5.background_diff(rgb, median, "clip")[0, 0]), 255)

    def test_small_differences_are_exact_and_symmetric(self):
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        median = np.full((2, 2, 3), 103.5, dtype=np.float64)   # a float median, as np.median returns
        d = t5.background_diff(rgb, median, "wrap")
        self.assertEqual(d.dtype, np.uint8)
        self.assertTrue((d == 10).all())          # 3 × 3.5 = 10.5 → truncates to 10 like astype('uint8')

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            t5.background_diff(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3)), "mirror")


@unittest.skipUnless(np is not None, "numpy (ml extra) not installed")
class TestBuildInput(unittest.TestCase):
    def test_channel_layout_rgb_then_diff_per_frame(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, size=(5, 4, 6, 3), dtype=np.uint8)
        median = np.full((4, 6, 3), 50.0)
        x = t5.build_input(frames, median, "wrap")
        self.assertEqual(x.shape, (20, 4, 6))
        self.assertEqual(x.dtype, np.float32)
        for f in range(5):
            np.testing.assert_allclose(x[4 * f: 4 * f + 3], np.moveaxis(frames[f], -1, 0) / 255.0, atol=1e-6)
            np.testing.assert_allclose(x[4 * f + 3], t5.background_diff(frames[f], median, "wrap") / 255.0, atol=1e-6)
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)

    def test_requires_exactly_seq_len_frames(self):
        with self.assertRaises(ValueError):
            t5.build_input(np.zeros((4, 2, 2, 3), np.uint8), np.zeros((2, 2, 3)))


class TestWindows(unittest.TestCase):
    def test_non_overlapping_with_last_frame_padding(self):
        self.assertEqual(t5.windows(12), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 11, 11, 11]])
        self.assertEqual(t5.windows(5), [[0, 1, 2, 3, 4]])
        self.assertEqual(t5.windows(1), [[0, 0, 0, 0, 0]])
        self.assertEqual(t5.windows(0), [])


@unittest.skipUnless(np is not None and cv2 is not None, "numpy + cv2 (ml extra) not installed")
class TestDecodeHeatmap(unittest.TestCase):
    def _heatmap(self):
        hm = np.zeros((288, 512), dtype=np.float32)
        hm[20:23, 10:13] = 0.9     # the shuttle: a 3×3 blob
        hm[100, 400] = 0.6         # a single stray pixel above a low threshold
        return hm

    def test_largest_blob_centre_and_global_peak(self):
        x, y, conf = decode_heatmap(self._heatmap(), 0.5)
        self.assertAlmostEqual(x, 11.5)
        self.assertAlmostEqual(y, 21.5)
        self.assertAlmostEqual(conf, 0.9, places=6)

    def test_threshold_is_configurable(self):
        self.assertIsNone(decode_heatmap(self._heatmap(), 0.95))
        x, y, _ = decode_heatmap(self._heatmap(), 0.35)
        self.assertAlmostEqual(x, 11.5)   # the 3×3 blob still beats the 1-px stray at a lower threshold

    def test_nan_never_leaks_into_the_confidence(self):
        hm = self._heatmap()
        hm[0, 0] = np.nan
        _, _, conf = decode_heatmap(hm, 0.5)
        self.assertTrue(np.isfinite(conf))

    def test_largest_blob_bbox_matches_the_v3_contract(self):
        binary = (self._heatmap() > 0.5).astype("uint8")
        self.assertEqual(largest_blob_bbox(binary), (10, 20, 3, 3))
        self.assertEqual(largest_blob_bbox(np.zeros((4, 4), np.uint8)), (0, 0, 0, 0))


class TestProvenanceHelpers(unittest.TestCase):
    def test_checkpoint_tag_is_a_sha256_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.pt"
            p.write_bytes(b"weights" * 1000)
            self.assertEqual(checkpoint_tag(p), hashlib.sha256(b"weights" * 1000).hexdigest()[:8])
            self.assertEqual(len(checkpoint_tag(p, 12)), 12)

    def test_vendor_commit_reads_detached_and_symbolic_heads(self):
        sha = "9116161772183e841bd1b51d6747c70f269b8b1c"
        with tempfile.TemporaryDirectory() as d:
            git = Path(d) / ".git"
            git.mkdir()
            (git / "HEAD").write_text(sha + "\n")
            self.assertEqual(vendor_commit(d), sha)
            (git / "HEAD").write_text("ref: refs/heads/main\n")
            (git / "refs" / "heads").mkdir(parents=True)
            (git / "refs" / "heads" / "main").write_text(sha + "\n")
            self.assertEqual(vendor_commit(d), sha)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(vendor_commit(d))


class TestVendoredImport(unittest.TestCase):
    """The model class is loaded by FILE PATH under a private module name — no sys.path edits, and the
    bare `models` name (WASB's vendored package) never enters sys.modules."""

    def test_loads_class_without_touching_sys_path_or_models(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "models").mkdir()
            (Path(d) / "models" / "tracknetv5.py").write_text(textwrap.dedent("""
                class TrackNetV5:
                    def __init__(self, in_dim, out_dim, alpha=1.0):
                        self.dims = (in_dim, out_dim, alpha)
            """))
            before_path = list(sys.path)
            before_models = sys.modules.get("models")
            cls = t5.load_vendored_model_class(d)
            self.assertEqual(cls(20, 5, 0.5).dims, (20, 5, 0.5))
            self.assertEqual(sys.path, before_path)
            self.assertIs(sys.modules.get("models"), before_models)

    def test_missing_repo_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            t5.load_vendored_model_class("/nonexistent/TrackNetV5")


class TestConversionPayload(unittest.TestCase):
    def test_payload_is_primitives_plus_state_dict(self):
        sd = {"w": "tensor-placeholder"}
        args = {"seq_len": 5, "model_alpha": 0.5, "bg_mode": "subtract_concat", "save_dir": "runs/lite",
                "nested": {"not": "allowed"}}
        payload = t5.conversion_payload(sd, args, variant="lite", src_name="TrackNetV5_Lite_best.pt",
                                        src_sha256="c7ca" + "0" * 60, vendor_sha=None, val_f1=0.98363)
        self.assertIs(payload["model_state_dict"], sd)
        cfg = payload["config"]
        self.assertEqual((cfg["backend"], cfg["variant"], cfg["alpha"], cfg["seq_len"], cfg["bg_mode"]),
                         ("tracknetv5", "lite", 0.5, 5, "subtract_concat"))
        self.assertEqual((cfg["in_dim"], cfg["out_dim"], cfg["thr"]), (20, 5, 0.35))
        src = payload["source"]
        self.assertEqual(src["commit"], t5.TRACKNETV5_SHA)
        self.assertEqual(src["file"], "TrackNetV5_Lite_best.pt")
        self.assertNotIn("nested", src["upstream_args"])        # only primitives survive
        self.assertAlmostEqual(src["val_f1"], 0.98363)
        self.assertTrue(t5.only_primitives(cfg))
        self.assertTrue(t5.only_primitives(src))


class TestRegistry(unittest.TestCase):
    def test_env_selects_tracknetv5_and_default_is_untouched(self):
        from worker.shuttle import TRACKERS, get_tracker

        self.assertIn("tracknetv5", TRACKERS)
        sentinel = object()
        with mock.patch.dict(os.environ, {"SHUTTLE_TRACKER": "tracknetv5"}), \
                mock.patch.object(t5, "TrackNetV5Tracker", lambda **kw: sentinel):
            self.assertIs(get_tracker(), sentinel)

    def test_eval_shuttle_lists_the_backend(self):
        from worker.eval_shuttle import TRACKERS

        self.assertIn("tracknetv5", TRACKERS)
        self.assertEqual(TRACKERS[-1], "fake")

    def test_tracker_name_carries_variant_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tracknetv5_lite.pt"
            p.write_bytes(b"x")
            name = t5.tracker_name("lite", p)
            self.assertTrue(name.startswith("tracknetv5-lite@"))
            self.assertEqual(name.split("@")[1], checkpoint_tag(p))


if __name__ == "__main__":
    unittest.main()
