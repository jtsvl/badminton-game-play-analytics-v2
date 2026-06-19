"""Tests for gate_track's pure track-building (worker/pipeline/gate_track.build_role_tracks).

The orchestration that turns in-court (t, frame, track, u, v) samples into role-labelled tracks —
including the Phase-4 re-ID stitch (recover a dropped fragment into the role it continues) and the
bounded short-hole interp — is a pure function so it can be unit-tested without store/parquet/I-O.
Scenario mirrors f35d0984: the far-left player (B1, track 6) is lost after a few frames and
re-detected under a fresh id (64) that enforce_n_tracks would otherwise discard.
"""
import unittest

from worker.pipeline.gate_track import build_role_tracks

FRAME_T = {f: f * 0.1 for f in range(10)}


def _samples():
    s = []
    def add(tid, frames, u, v):
        for f in frames:
            s.append((FRAME_T[f], f, tid, u, v))
    add(1, range(10), 0.30, 0.80)   # near_left
    add(2, range(10), 0.70, 0.80)   # near_right
    add(3, range(10), 0.55, 0.20)   # far_right (B2), fully detected
    add(6, range(0, 5), 0.45, 0.05)  # far_left (B1), lost after frame 4
    add(64, range(7, 10), 0.50, 0.05)  # B1 re-detected under a new id (fragment), hole at 5,6
    return s


def _counts(samples):
    c = {}
    for _, _, tid, _, _ in samples:
        c[tid] = c.get(tid, 0) + 1
    return c


class TestBuildRoleTracks(unittest.TestCase):
    def setUp(self):
        self.samples = _samples()
        self.rows, self.roles = build_role_tracks(
            self.samples, _counts(self.samples), FRAME_T, n=4, discipline="doubles", max_gap=8)

    def _rows_for(self, role):
        return sorted([r for r in self.rows if r["role"] == role], key=lambda r: r["t"])

    def test_dropped_fragment_is_reid_stitched_into_the_far_left_role(self):
        self.assertEqual(self.roles.get(64), "far_left")

    def test_far_left_now_spans_the_whole_clip(self):
        fl = self._rows_for("far_left")
        self.assertEqual(len(fl), 10)  # frames 0..9 all present after re-ID + interp

    def test_sources_tag_detected_reid_and_interp(self):
        by_t = {round(r["t"], 6): r["source"] for r in self._rows_for("far_left")}
        self.assertEqual(by_t[round(FRAME_T[2], 6)], "detected")  # original track-6 detection
        self.assertEqual(by_t[round(FRAME_T[5], 6)], "interp")    # short hole filled
        self.assertEqual(by_t[round(FRAME_T[6], 6)], "interp")
        self.assertEqual(by_t[round(FRAME_T[8], 6)], "reid")      # recovered fragment

    def test_interp_positions_stay_between_the_anchors(self):
        for r in self._rows_for("far_left"):
            self.assertTrue(0.44 <= r["u"] <= 0.51)   # bounded by the two anchor u's
            self.assertAlmostEqual(r["v"], 0.05, places=6)

    def test_primary_roles_are_unchanged_and_not_filled(self):
        nl = self._rows_for("near_left")
        self.assertEqual(len(nl), 10)
        for r in nl:
            self.assertEqual(r["source"], "detected")
            self.assertAlmostEqual(r["u"], 0.30, places=6)


class TestBuildRoleTracksNoStitch(unittest.TestCase):
    """With exactly N fully-detected tracks and no holes, behaviour is the plain per-role output
    (every row 'detected', no fabricated frames) — no regression vs the pre-Phase-4 pipeline."""

    def test_singles_two_full_tracks_all_detected(self):
        s = [(f * 0.1, f, 1, 0.5, 0.8) for f in range(6)] + \
            [(f * 0.1, f, 2, 0.5, 0.2) for f in range(6)]
        counts = {1: 6, 2: 6}
        rows, roles = build_role_tracks(s, counts, {f: f * 0.1 for f in range(6)},
                                        n=2, discipline="singles", max_gap=8)
        self.assertEqual(roles, {1: "near", 2: "far"})
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["source"] == "detected" for r in rows))


if __name__ == "__main__":
    unittest.main()
