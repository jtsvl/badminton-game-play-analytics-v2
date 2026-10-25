"""Opt-in smoke test for the VideoJev model wrapper (stroke/vlm_model.py).

Self-skips unless torch + transformers + av are installed, the weights are present and
STROKE_VLM_SMOKE=1 is set — loading the 4B model takes ~16 GB in fp32, so this never runs by
accident in the suite (precedent: tests/test_stroke_model.py's vendor guard). What it pins is
OUR plumbing against the remote code: a synthetic 1.4 s clip goes through prepare_video →
encode_video → predict_features and comes back as 7 finite probabilities summing to 1, with the
frame count our expected_frames() predicted. Run on the Mac (runbook worker/training/videojev_pilot.md):

    STROKE_VLM_SMOKE=1 uv run pytest tests/test_stroke_vlm_smoke.py
"""
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from worker.stroke.vlm_eval import (DEFAULT_CONTEXT, PRESETS, STROKE_VLM_MODEL, build_question,
                                    expected_frames)

_READY = (all(find_spec(m) for m in ("torch", "transformers", "av"))
          and Path(STROKE_VLM_MODEL).exists() and os.environ.get("STROKE_VLM_SMOKE") == "1")
_SKIP = ("needs the vlm extra (uv sync --extra ml --extra vlm), the VideoJev-4B snapshot at "
         "$STROKE_VLM_MODEL and STROKE_VLM_SMOKE=1")


def write_synthetic_clip(path: Path, *, duration_s: float = 1.4, fps: int = 30,
                         size: tuple[int, int] = (320, 180)) -> None:
    """A moving bright blob on a dark court-ish background — enough for the decoder/sampler path."""
    import av
    import numpy as np

    w, h = size
    n = int(round(duration_s * fps))
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[..., 1] = 40                                  # dark green floor
            cx = int(w * (0.2 + 0.6 * i / max(1, n - 1)))
            frame[h // 2 - 8:h // 2 + 8, max(0, cx - 8):min(w, cx + 8)] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@unittest.skipUnless(_READY, _SKIP)
class TestVideoJevSmoke(unittest.TestCase):
    def test_seven_finite_probabilities_on_cpu_fp32(self):
        import av

        from worker.stroke import vlm_model as vm

        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "synthetic_r01_h01_0000.70s_deadbeef.mp4"
            write_synthetic_clip(clip)
            with av.open(str(clip)) as c:
                s = c.streams.video[0]
                duration = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
            model, processor, prov = vm.load_videojev(STROKE_VLM_MODEL, device="cpu", dtype="fp32")
            self.assertEqual(prov["dtype"], "fp32")
            preset = PRESETS["plain_fps2"]
            features, record, wall = vm.encode_clip(model, processor, clip, fps=preset.fps,
                                                    max_frames=preset.max_frames,
                                                    context=DEFAULT_CONTEXT)
            self.assertEqual(int(record["frames"]), expected_frames(duration, preset.fps))
            probs = vm.predict_from_features(model, processor, features, record,
                                             build_question(preset), context=DEFAULT_CONTEXT)
        self.assertEqual(len(probs), 7)
        self.assertAlmostEqual(sum(probs), 1.0, places=3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertGreater(wall, 0.0)


if __name__ == "__main__":
    unittest.main()
