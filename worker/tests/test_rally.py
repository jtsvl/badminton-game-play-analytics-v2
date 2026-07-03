"""Stage 6 rally segmentation (worker/worker/rally/segment.py) — pure, image-space.

All trajectory/serve logic runs in NORMALIZED IMAGE space (x=px/W, y=py/H ∈ [0,1]), never court (u,v):
the ground homography can't place an airborne shuttle (its v goes negative in flight). Court (u,v) is
carried through only so find_rally_end can expose the floor-contact candidate for the future in/out
feature. Synthetic fixtures; no torch/cv2 (runs under the dev extra).
"""
import unittest

from worker.rally import segment as seg


def det(t, x, y, conf=0.7, frame=None, u=0.5, v=0.5):
    """One shuttle detection in the shape build_rallies consumes (image-space x,y + court u,v)."""
    return {"t": t, "x": x, "y": y, "conf": conf,
            "frame": int(round(t * 30)) if frame is None else frame, "u": u, "v": v}


def line(t0, x0, y0, dx, dy, n, dt=1 / 30, conf=0.7):
    """n collinear detections from (x0,y0) stepping (dx,dy) each dt — a straight flight segment."""
    return [det(t0 + i * dt, x0 + i * dx, y0 + i * dy, conf=conf) for i in range(n)]


def rally_with_landing(t0=0.0, u=0.5, v=0.9, n_flat=4):
    """A full rally whose tail is a visible landing: level flight → descent → at rest on the floor.

    The resting detections carry the landing's court (u, v); in-flight ones carry a bogus (0.3, 0.3)
    to prove the landing position comes from the contact detection, not the flight."""
    dets = line(t0, 0.10, 0.30, 0.01, 0.0, 12)
    t, x, y = t0 + 12 / 30, 0.10 + 12 * 0.01, 0.30
    for _ in range(5):                       # descent (vy ≈ 0.6 norm/s)
        dets.append(det(t, x, y, u=0.3, v=0.3))
        t, x, y = t + 1 / 30, x + 0.01, y + 0.02
    for _ in range(n_flat):                  # at rest — floor contact
        dets.append(det(t, x, y, u=u, v=v))
        t, y = t + 1 / 30, y + 0.001
    return dets


class TestGroupTrajectories(unittest.TestCase):
    def test_temporal_gap_splits_runs(self):
        dets = line(0.0, 0.1, 0.5, 0.01, 0.0, 3) + line(2.0, 0.2, 0.5, 0.01, 0.0, 3)
        runs = seg.group_trajectories(dets)
        self.assertEqual([len(r) for r in runs], [3, 3])

    def test_small_steps_stay_one_run(self):
        dets = line(0.0, 0.1, 0.5, 0.02, 0.01, 6)   # steps ≈0.022 < max_jump
        runs = seg.group_trajectories(dets)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]), 6)

    def test_collinear_large_steps_kept_as_smash(self):
        # large steps (0.15 > max_jump) but perfectly collinear → one fast run, not split.
        dets = line(0.0, 0.1, 0.5, 0.15, 0.0, 4)
        runs = seg.group_trajectories(dets)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]), 4)

    def test_non_collinear_jump_breaks_run(self):
        # establish a rightward direction, then a big sideways-down teleport (a foreign shuttle).
        dets = line(0.0, 0.10, 0.5, 0.08, 0.0, 3) + [det(0.10, 0.16, 0.9)]
        runs = seg.group_trajectories(dets)
        self.assertEqual([len(r) for r in runs], [3, 1])


class TestRejectForeignRuns(unittest.TestCase):
    def test_isolated_short_run_dropped(self):
        rally = line(0.0, 0.1, 0.5, 0.02, 0.0, 8)            # long, sustained
        foreign = line(5.0, 0.85, 0.2, 0.01, 0.0, 2)         # short, far in time + space
        kept = seg.reject_foreign_runs([rally, foreign])
        self.assertEqual(kept, [rally])

    def test_smash_fragment_near_long_run_kept(self):
        rally = line(0.0, 0.1, 0.5, 0.02, 0.0, 8)            # ends ≈(0.24,0.5) at t≈0.233
        frag = line(0.30, 0.26, 0.5, 0.02, 0.0, 2)           # right after, continuing nearby
        kept = seg.reject_foreign_runs([rally, frag])
        self.assertEqual(kept, [rally, frag])

    def test_all_short_runs_means_no_rally(self):
        a = line(0.0, 0.1, 0.5, 0.01, 0.0, 2)
        b = line(5.0, 0.8, 0.2, 0.01, 0.0, 2)
        self.assertEqual(seg.reject_foreign_runs([a, b]), [])


class TestMergeRunsIntoRallies(unittest.TestCase):
    # Real rallies clear the noise floor (>= MIN_RALLY_S span and >= MIN_DETS detections); 18 dets at
    # 30fps ≈ 0.57s span, comfortably above the 0.5s / 5-det defaults.
    def test_short_dropout_bridged_into_one_rally(self):
        a = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)            # ends t≈0.566
        b = line(1.0, 0.20, 0.5, 0.004, 0.0, 18)            # 0.43s gap (< bridge) → same rally
        rallies = seg.merge_runs_into_rallies([a, b])
        self.assertEqual(len(rallies), 1)
        self.assertEqual(len(rallies[0]), 36)

    def test_long_gap_splits_into_two_rallies(self):
        a = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)
        b = line(8.0, 0.20, 0.5, 0.004, 0.0, 18)            # 7.4s gap → separate rally
        rallies = seg.merge_runs_into_rallies([a, b])
        self.assertEqual(len(rallies), 2)

    def test_subsecond_span_discarded(self):
        short = line(0.0, 0.1, 0.5, 0.02, 0.0, 3)           # 0.067s span, 3 dets → not a rally
        self.assertEqual(seg.merge_runs_into_rallies([short]), [])


def swing(t, strength, track=1):
    """One swing-activity event in the shape worker.rally.swing.swing_events emits."""
    return {"t": t, "track": track, "strength": strength}


class TestSwingBridge(unittest.TestCase):
    # a ends t≈0.567; b starts t=2.6 → gap≈2.033s, inside (BRIDGE_S=1.5, RALLY_SWING_BRIDGE_S=2.5].
    # Interior search window for a bridging swing: [0.567, 2.6 - RALLY_SERVE_EXCLUSION_S(0.75)=1.85].
    A = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)
    B = line(2.6, 0.20, 0.5, 0.004, 0.0, 18)

    def test_strong_interior_swing_bridges_by_default(self):
        # RALLY_SWING_BRIDGE promoted on 2026-07-03 (T8 verdict: thr=4.0 tl/s, ~80% split recovery,
        # false_merges untestable on the 5-game corpus but zero measured — see spec).
        rallies = seg.merge_runs_into_rallies([self.A, self.B], swings=[swing(1.2, 8.0)])
        self.assertEqual(len(rallies), 1)

    def test_can_still_be_disabled_explicitly(self):
        rallies = seg.merge_runs_into_rallies([self.A, self.B], swings=[swing(1.2, 8.0)],
                                              swing_bridge=False)
        self.assertEqual(len(rallies), 2)

    def test_no_swings_provided_stays_split_even_by_default(self):
        rallies = seg.merge_runs_into_rallies([self.A, self.B], swings=None)
        self.assertEqual(len(rallies), 2)

    def test_weak_swing_below_threshold_stays_split(self):
        rallies = seg.merge_runs_into_rallies([self.A, self.B], swings=[swing(1.2, 2.0)],
                                              swing_bridge=True)
        self.assertEqual(len(rallies), 2)

    def test_swing_inside_serve_exclusion_zone_does_not_bridge(self):
        # t=2.2 is within the last 0.75s before b starts — a quick re-serve, not a dropout bridge.
        rallies = seg.merge_runs_into_rallies([self.A, self.B], swings=[swing(2.2, 8.0)],
                                              swing_bridge=True)
        self.assertEqual(len(rallies), 2)

    def test_gap_beyond_swing_bridge_cap_never_bridges(self):
        far_b = line(3.6, 0.20, 0.5, 0.004, 0.0, 18)          # gap≈3.03s > RALLY_SWING_BRIDGE_S=2.5
        rallies = seg.merge_runs_into_rallies([self.A, far_b], swings=[swing(2.0, 8.0)],
                                              swing_bridge=True)
        self.assertEqual(len(rallies), 2)

    def test_short_gap_still_bridges_without_any_swing(self):
        close_b = line(1.0, 0.20, 0.5, 0.004, 0.0, 18)        # 0.43s gap — the original BRIDGE_S path
        rallies = seg.merge_runs_into_rallies([self.A, close_b], swing_bridge=True)
        self.assertEqual(len(rallies), 1)


class TestFindServeFrame(unittest.TestCase):
    SETTLED = [(9.5, 1.0), (9.7, 1.2), (9.9, 0.8)]          # players calm before the serve
    MOVING = [(9.5, 12.0), (9.7, 10.0), (9.9, 9.0)]         # already in motion (no settled window)

    def test_serve_snaps_to_first_detection_when_players_settled(self):
        rally = line(10.0, 0.5, 0.6, 0.01, -0.02, 6)
        s = seg.find_serve_frame(rally, self.SETTLED)
        self.assertAlmostEqual(s["t"], 10.0)
        self.assertTrue(s["settled"])
        self.assertEqual(s["frame"], rally[0]["frame"])

    def test_serve_falls_back_with_lower_confidence_when_not_settled(self):
        rally = line(10.0, 0.5, 0.6, 0.01, -0.02, 6)
        moving = seg.find_serve_frame(rally, self.MOVING)
        settled = seg.find_serve_frame(rally, self.SETTLED)
        self.assertFalse(moving["settled"])
        self.assertAlmostEqual(moving["t"], 10.0)            # still the first detection (fallback)
        self.assertLess(moving["confidence"], settled["confidence"])

    def test_serve_without_player_energy_is_fallback(self):
        rally = line(10.0, 0.5, 0.6, 0.01, -0.02, 6)
        self.assertFalse(seg.find_serve_frame(rally, None)["settled"])


class TestFindRallyEnd(unittest.TestCase):
    def test_returns_last_reliable_point_with_image_and_court(self):
        rally = [det(10.0, 0.30, 0.40, conf=0.8, u=0.50, v=0.60),
                 det(10.1, 0.32, 0.42, conf=0.9, u=0.52, v=0.95),
                 det(10.2, 0.90, 0.90, conf=0.3, u=0.99, v=0.99)]   # last is a low-conf blip → skipped
        end = seg.find_rally_end(rally)
        self.assertAlmostEqual(end["t"], 10.1)
        self.assertAlmostEqual(end["x"], 0.32)
        self.assertAlmostEqual(end["y"], 0.42)
        self.assertAlmostEqual(end["u"], 0.52)               # court coords exposed for future in/out
        self.assertAlmostEqual(end["v"], 0.95)


class TestCountHits(unittest.TestCase):
    def test_counts_vertical_direction_reversals(self):
        ys = [0.2, 0.4, 0.6, 0.5, 0.3, 0.5, 0.7]             # down, down, up, up, down, down → 2 reversals
        rally = [det(10.0 + i / 30, 0.5, y) for i, y in enumerate(ys)]
        self.assertEqual(seg.count_hits(rally), 2)

    def test_none_when_too_few_points(self):
        self.assertIsNone(seg.count_hits([det(10.0, 0.5, 0.5), det(10.1, 0.5, 0.6)]))


class TestBuildRallies(unittest.TestCase):
    def test_empty_detections_yield_no_rallies(self):
        out = seg.build_rallies([])
        self.assertEqual(out["rallies"], [])
        self.assertEqual(out["events"], [])

    def test_emits_contract_shaped_rows(self):
        dets = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)
        out = seg.build_rallies(dets)
        self.assertEqual(len(out["rallies"]), 1)
        r = out["rallies"][0]
        self.assertEqual(set(r), {"idx", "t_start_s", "t_end_s", "shot_count",
                                  "winner_side", "end_shot_type", "confidence"})
        self.assertEqual(r["idx"], 0)
        self.assertIsNone(r["winner_side"])         # not inferred here — decoupled from the match score
        self.assertIsNone(r["end_shot_type"])       # refined by stage 7
        self.assertLessEqual(r["t_start_s"], r["t_end_s"])
        types = {e["type"] for e in out["events"]}
        self.assertIn("serve", types)
        self.assertIn("rally_end", types)
        for e in out["events"]:
            self.assertEqual(set(e) >= {"rally_idx", "t_s", "type", "payload"}, True)

    def test_rally_end_event_carries_court_coords_for_inout(self):
        dets = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)
        out = seg.build_rallies(dets)
        end_ev = next(e for e in out["events"] if e["type"] == "rally_end")
        self.assertIn("u", end_ev["payload"])
        self.assertIn("v", end_ev["payload"])

    def test_every_rally_gets_exactly_one_landing_event(self):
        out = seg.build_rallies(rally_with_landing())
        landings = [e for e in out["events"] if e["type"] == "landing"]
        self.assertEqual(len(landings), 1)
        p = landings[0]["payload"]
        self.assertTrue(p["located"])
        for key in ("call", "line", "u", "v", "px", "py", "margin_cm", "band_cm", "confidence"):
            self.assertIn(key, p)
        self.assertEqual(p["call"], "in")            # (0.5, 0.9) is comfortably inside
        self.assertAlmostEqual(p["u"], 0.5)          # position from the contact det, not the flight
        self.assertAlmostEqual(p["v"], 0.9)
        # the event time is the contact time, inside the rally span
        r = out["rallies"][0]
        self.assertGreaterEqual(landings[0]["t_s"], r["t_start_s"])
        self.assertLessEqual(landings[0]["t_s"], r["t_end_s"])

    def test_degraded_landing_is_honest_and_positionless(self):
        # level flight only — nothing ever descends, so there is no landing to locate
        out = seg.build_rallies(line(0.0, 0.10, 0.5, 0.004, 0.0, 18))
        p = next(e for e in out["events"] if e["type"] == "landing")["payload"]
        self.assertFalse(p["located"])
        self.assertEqual(p["call"], "uncertain")
        self.assertNotIn("u", p)                     # never a fabricated position

    def test_discipline_switches_the_boundary(self):
        # 50 cm outside the singles side line = 4 cm inside the doubles line:
        # out for singles (past the ±45 cm band), uncertain for doubles (well inside the band)
        u = 0.46 / 6.10 - 0.50 / 6.10
        singles = seg.build_rallies(rally_with_landing(u=u, v=0.9))
        doubles = seg.build_rallies(rally_with_landing(u=u, v=0.9), discipline="doubles")
        p_s = next(e for e in singles["events"] if e["type"] == "landing")["payload"]
        p_d = next(e for e in doubles["events"] if e["type"] == "landing")["payload"]
        self.assertEqual(p_s["call"], "out")
        self.assertEqual(p_d["call"], "uncertain")

    def test_winner_side_stays_none_in_pure_logic(self):
        # even for a clear out landing — mapping halves to sides is the glue's job (needs match_players)
        out = seg.build_rallies(rally_with_landing(u=-0.2, v=0.9))
        self.assertIsNone(out["rallies"][0]["winner_side"])

    def test_dead_time_between_rallies_excluded(self):
        rally1 = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)        # t 0.0..0.566
        dead = [det(3.0, 0.8, 0.2), det(3.7, 0.2, 0.8), det(4.4, 0.6, 0.6)]  # isolated foreign blips
        rally2 = line(6.0, 0.20, 0.5, 0.004, 0.0, 18)        # t 6.0..6.566
        out = seg.build_rallies(rally1 + dead + rally2)
        self.assertEqual(len(out["rallies"]), 2)
        spans = [(r["t_start_s"], r["t_end_s"]) for r in out["rallies"]]
        # no rally covers the dead-time blips at t≈3.0–4.4
        for t0, t1 in spans:
            self.assertFalse(t0 <= 3.7 <= t1)

    def test_swings_threaded_through_to_the_bridge_rule(self):
        rally1 = line(0.0, 0.10, 0.5, 0.004, 0.0, 18)         # ends t≈0.567
        rally2 = line(2.6, 0.20, 0.5, 0.004, 0.0, 18)         # gap≈2.033s — needs the swing bridge
        dets = rally1 + rally2
        on = seg.build_rallies(dets, swings=[swing(1.2, 8.0)])
        self.assertEqual(len(on["rallies"]), 1)               # bridged by default (T8 promoted)
        off = seg.build_rallies(dets, swings=[swing(1.2, 8.0)], swing_bridge=False)
        self.assertEqual(len(off["rallies"]), 2)


if __name__ == "__main__":
    unittest.main()
