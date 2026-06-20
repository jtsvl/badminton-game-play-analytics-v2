"""Tests for the pure shuttle-label interpolation (worker/shuttle/label.py).

`interpolate_anchors` turns the sparse hand-clicked anchors + segment boundaries from the labeler into the
DENSE per-frame answer key that feeds BOTH the TrackNetV3 training target and the precision/recall metric —
so its edge cases (boundaries, occlusion marks, never-extrapolate) are exactly where sloppy ground truth
would slip in. Mirrors the house rule in geometry.fill_track_gaps: interpolate between two anchors only,
linearly, never extrapolate (specs/features/shuttle-finetune-plan.md §2).
"""
import unittest

from worker.shuttle.label import interpolate_anchors


def _anchor(frame, px=None, py=None, visibility=1):
    return {"frame_idx": frame, "px": px, "py": py, "visibility": visibility}


class TestInterpolateAnchors(unittest.TestCase):
    def assertDense(self, out, expected):
        """Compare dense rows: frame/visibility/interpolated exactly, px/py approximately (the function
        keeps raw interpolation floats — rounding is the labeler's job at save time, like round4)."""
        self.assertEqual(len(out), len(expected))
        for got, exp in zip(out, expected):
            self.assertEqual((got[0], got[3], got[4]), (exp[0], exp[3], exp[4]))
            for g, e in ((got[1], exp[1]), (got[2], exp[2])):
                if e is None:
                    self.assertIsNone(g)
                else:
                    self.assertAlmostEqual(g, e, places=9)

    def test_empty_and_single(self):
        self.assertEqual(interpolate_anchors([], []), [])
        self.assertEqual(
            interpolate_anchors([_anchor(5, 0.2, 0.3)], []),
            [(5, 0.2, 0.3, 1, False)],
        )

    def test_linear_fill_within_segment(self):
        # two visible anchors, no boundary between → every interior frame filled on the straight line.
        out = interpolate_anchors([_anchor(0, 0.0, 0.0), _anchor(4, 0.4, 0.8)], [])
        self.assertDense(
            out,
            [
                (0, 0.0, 0.0, 1, False),   # anchor
                (1, 0.1, 0.2, 1, True),    # filled
                (2, 0.2, 0.4, 1, True),
                (3, 0.3, 0.6, 1, True),
                (4, 0.4, 0.8, 1, False),   # anchor
            ],
        )

    def test_boundary_between_anchors_blocks_fill(self):
        # a contact/segment boundary strictly between two anchors → never bridge it; the gap is vis=0.
        out = interpolate_anchors([_anchor(0, 0.0, 0.0), _anchor(4, 0.4, 0.8)], [2])
        self.assertEqual(
            out,
            [
                (0, 0.0, 0.0, 1, False),
                (1, None, None, 0, True),   # gap — not interpolated across the boundary
                (2, None, None, 0, True),
                (3, None, None, 0, True),
                (4, 0.4, 0.8, 1, False),
            ],
        )

    def test_boundary_on_an_anchor_does_not_block(self):
        # the contact frame itself carries an anchor → the two sub-segments meet at it but never bridge
        # across; a boundary sitting on an anchor must not turn its neighbours into gaps.
        out = interpolate_anchors(
            [_anchor(0, 0.0, 0.0), _anchor(2, 0.2, 0.2), _anchor(4, 0.0, 0.4)], [2]
        )
        self.assertDense(
            out,
            [
                (0, 0.0, 0.0, 1, False),
                (1, 0.1, 0.1, 1, True),
                (2, 0.2, 0.2, 1, False),    # anchor on the boundary
                (3, 0.1, 0.3, 1, True),
                (4, 0.0, 0.4, 1, False),
            ],
        )

    def test_invisible_anchor_is_hard_override_and_blocks_neighbours(self):
        # a vis=0 anchor (occluded / out of frame) emits X=Y=0 and blocks position fill on both sides:
        # you can't interpolate a position to or from an unknown point.
        out = interpolate_anchors(
            [_anchor(0, 0.0, 0.0), _anchor(2, visibility=0), _anchor(4, 0.4, 0.8)], []
        )
        self.assertEqual(
            out,
            [
                (0, 0.0, 0.0, 1, False),
                (1, None, None, 0, True),   # neighbour not bridged
                (2, None, None, 0, False),  # the invisible anchor itself (hand-marked → not interpolated)
                (3, None, None, 0, True),
                (4, 0.4, 0.8, 1, False),
            ],
        )

    def test_never_extrapolate_outside_anchor_span(self):
        # nothing is produced before the first or after the last anchor (one-sided, no second anchor).
        out = interpolate_anchors([_anchor(3, 0.3, 0.3), _anchor(5, 0.5, 0.5)], [])
        frames = [r[0] for r in out]
        self.assertEqual(frames, [3, 4, 5])

    def test_unsorted_anchors_are_sorted(self):
        out = interpolate_anchors([_anchor(4, 0.4, 0.8), _anchor(0, 0.0, 0.0)], [])
        self.assertEqual([r[0] for r in out], [0, 1, 2, 3, 4])
        self.assertEqual(out[0], (0, 0.0, 0.0, 1, False))

    def test_multi_segment_arc(self):
        # a realistic rally: serve arc 0..3, contact at 3, return arc 3..6.
        out = interpolate_anchors(
            [_anchor(0, 0.1, 0.9), _anchor(3, 0.5, 0.1), _anchor(6, 0.9, 0.9)], [3]
        )
        # all frames visible & filled; the kink lives at the shared anchor 3, not bridged.
        self.assertTrue(all(r[3] == 1 for r in out))
        self.assertEqual([r[0] for r in out], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(out[3], (3, 0.5, 0.1, 1, False))


if __name__ == "__main__":
    unittest.main()
