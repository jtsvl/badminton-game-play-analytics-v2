"""swing — pure swing-activity events from RTMPose keypoints (specs/features/pose-swing-fusion.md).

Synthetic pose rows {"t","track","kp","kp_score"} (Halpe-26). At the 12 fps analysis rate a swing
is 1–2 samples, so the module emits threshold-runs (no local-max logic) in torso-lengths/second:
scale-free across near/far court, lunge-proof via the rolling-median torso.
"""
import unittest

from worker.geometry import ELBOW_KP, HEAD_KP, HIP_KP, WRIST_KP
from worker.rally.swing import swing_events

DT = 1.0 / 12.0  # analysis cache frame interval


def pose_row(t, track=7, wrist=None, elbow=None, torso_len=1.0,
             wrist_conf=0.9, elbow_conf=0.9, torso_conf=0.9):
    """One Halpe-26 pose row: right wrist/elbow set explicitly, torso = head(17)→hip(19) vertical."""
    kp = [[0.0, 0.0] for _ in range(26)]
    score = [0.0] * 26
    kp[HEAD_KP] = [0.0, 0.0]
    kp[HIP_KP] = [0.0, torso_len]
    score[HEAD_KP] = score[HIP_KP] = torso_conf
    if wrist is not None:
        kp[WRIST_KP[1]] = list(wrist)
        score[WRIST_KP[1]] = wrist_conf
    if elbow is not None:
        kp[ELBOW_KP[1]] = list(elbow)
        score[ELBOW_KP[1]] = elbow_conf
    return {"t": t, "track": track, "kp": kp, "kp_score": score}


def wrist_track(steps, track=7, start=0.0, torso_len=1.0, **kw):
    """Rows whose right wrist moves `steps[i]` (x-units) between consecutive frames."""
    rows, x = [], 0.0
    for i, step in enumerate([0.0] + list(steps)):
        x += step
        rows.append(pose_row(start + i * DT, track=track, wrist=(x, 0.0), torso_len=torso_len, **kw))
    return rows


class TestSwingEvents(unittest.TestCase):
    def test_fast_wrist_motion_emits_event(self):
        rows = wrist_track([0.5, 0.5])  # 0.5 torso-lengths per frame = 6 tl/s
        evs = swing_events(rows)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["track"], 7)
        self.assertAlmostEqual(evs[0]["strength"], 6.0, places=3)
        self.assertTrue(0.0 <= evs[0]["t"] <= 2 * DT)

    def test_walkback_speed_below_floor_ignored(self):
        rows = wrist_track([0.1] * 6)  # 1.2 tl/s — arm swings while walking back
        self.assertEqual(swing_events(rows), [])

    def test_low_conf_wrist_falls_back_to_elbow_with_gain(self):
        # Wrist never confident; elbow moves 0.3/frame = 3.6 tl/s, x1.8 gain = 6.48 tl/s.
        rows = []
        for i, x in enumerate([0.0, 0.3, 0.6]):
            rows.append(pose_row(i * DT, wrist=(0.0, 0.0), wrist_conf=0.05, elbow=(x, 0.0)))
        evs = swing_events(rows)
        self.assertEqual(len(evs), 1)
        self.assertAlmostEqual(evs[0]["strength"], 3.6 * 1.8, places=3)

    def test_blurred_peak_frame_bridged_across_gap(self):
        # Middle frame all-unconfident (motion blur at the swing peak): displacement over the
        # true 2*dt still shows the swing as one big step.
        rows = [
            pose_row(0 * DT, wrist=(0.0, 0.0)),
            pose_row(1 * DT, wrist=(0.4, 0.0), wrist_conf=0.0, torso_conf=0.0),
            pose_row(2 * DT, wrist=(0.8, 0.0)),
        ]
        evs = swing_events(rows)
        self.assertEqual(len(evs), 1)
        self.assertAlmostEqual(evs[0]["strength"], 0.8 / (2 * DT), places=3)  # 4.8 tl/s

    def test_tracker_dropout_does_not_fabricate_speed(self):
        # 0.5 s hole; the player re-appears 2 torso-lengths away. dt > SWING_MAX_DT_S must be
        # skipped — 2.0/0.5 = 4 tl/s would clear the floor and fake a swing.
        rows = wrist_track([0.02], start=0.0)
        rows += wrist_track([0.02], track=7, start=0.6)
        for r in rows[2:]:  # shift the second cluster far away
            r["kp"][WRIST_KP[1]][0] += 2.0
        self.assertEqual(swing_events(rows), [])

    def test_far_player_half_scale_same_strength(self):
        near = wrist_track([0.5, 0.5], track=1, torso_len=1.0)
        far = wrist_track([0.25, 0.25], track=2, torso_len=0.5)
        evs = swing_events(near + far)
        self.assertEqual(len(evs), 2)
        by_track = {e["track"]: e["strength"] for e in evs}
        self.assertAlmostEqual(by_track[1], by_track[2], places=6)

    def test_crouch_does_not_spike_strength(self):
        # Modest arm motion (2.4 tl/s, below floor). A 3-frame lunge crushes the instantaneous
        # torso to 0.35 — naive normalization would read 6.9 tl/s and fire. The rolling MEDIAN
        # torso must hold ~1.0 through the lunge.
        rows = []
        x = 0.0
        for i in range(24):  # 2 s
            torso = 0.35 if i in (10, 11, 12) else 1.0
            rows.append(pose_row(i * DT, wrist=(x, 0.0), torso_len=torso))
            x += 0.2
        self.assertEqual(swing_events(rows), [])

    def test_adjacent_threshold_runs_coalesce(self):
        # above, above, below, above, above with a 0.167 s hole < SWING_MERGE_S -> ONE event...
        rows = wrist_track([0.5, 0.5, 0.01, 0.5, 0.5])
        self.assertEqual(len(swing_events(rows)), 1)
        # ...but a 0.417 s below-floor stretch > SWING_MERGE_S stays two distinct events.
        rows = wrist_track([0.5, 0.5] + [0.01] * 5 + [0.5, 0.5])
        self.assertEqual(len(swing_events(rows)), 2)

    def test_empty_or_all_low_conf_rows_yield_no_events(self):
        self.assertEqual(swing_events([]), [])
        rows = wrist_track([0.5, 0.5], wrist_conf=0.0, elbow_conf=0.0, torso_conf=0.0)
        self.assertEqual(swing_events(rows), [])


if __name__ == "__main__":
    unittest.main()
