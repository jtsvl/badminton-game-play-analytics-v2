"""Tests for the stroke labeling clip exporter (worker/stroke_clips.py).

The exporter turns stage-6 provisional hit events into short real-video clips (from the 720p
proxy, audio kept) plus a Label Studio task list. The pure mapping is what these tests pin:
window clamping (a hit right at the video edge must not produce a negative start or overshoot),
hit filtering against current rally spans (rally-review edits may have removed rallies — stray
hits would become junk clips), stable sortable filenames, the task JSON the later
import-stroke-labels step will join back to DB rows, and the exact ffmpeg argv (re-encode for
frame-accurate cuts — stream-copy would snap to the proxy's 2 s keyframes).
(specs/features/stroke-labeling.md)
"""
import unittest

from worker import stroke_clips as sc


class TestClipWindow(unittest.TestCase):
    def test_normal_window(self):
        start, dur = sc.clip_window(10.0, pre=0.7, post=0.7, duration_s=554.7)
        self.assertAlmostEqual(start, 9.3)
        self.assertAlmostEqual(dur, 1.4)

    def test_clamps_start_at_zero(self):
        # a serve right at video start: window must not begin before 0, end stays t+post.
        start, dur = sc.clip_window(0.3, pre=0.7, post=0.7, duration_s=554.7)
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(dur, 1.0)

    def test_clamps_end_at_duration(self):
        start, dur = sc.clip_window(553.0, pre=0.7, post=0.7, duration_s=553.26)
        self.assertAlmostEqual(start, 552.3)
        self.assertAlmostEqual(dur, 0.96)

    def test_unknown_duration_skips_end_clamp(self):
        # match_videos.duration_s is nullable — without it we can only clamp the start.
        start, dur = sc.clip_window(553.0, pre=0.7, post=0.7, duration_s=None)
        self.assertAlmostEqual(start, 552.3)
        self.assertAlmostEqual(dur, 1.4)


class TestHitsInRallies(unittest.TestCase):
    RALLIES = [
        {"id": "ra", "idx": 0, "t_start_s": 5.0, "t_end_s": 12.0},
        {"id": "rb", "idx": 1, "t_start_s": 20.0, "t_end_s": 30.0},
    ]

    def test_filters_to_hits_inside_rally_spans(self):
        events = [
            {"id": "e1", "type": "hit", "t_s": 6.0},
            {"id": "e2", "type": "hit", "t_s": 15.0},   # between rallies — dropped
            {"id": "e3", "type": "rally_start", "t_s": 5.0},  # not a hit — dropped
            {"id": "e4", "type": "hit", "t_s": 21.0},
        ]
        hits = sc.hits_in_rallies(events, self.RALLIES)
        self.assertEqual([h["event"]["id"] for h in hits], ["e1", "e4"])

    def test_rally_boundaries_are_inclusive(self):
        events = [
            {"id": "e1", "type": "hit", "t_s": 5.0},
            {"id": "e2", "type": "hit", "t_s": 12.0},
        ]
        hits = sc.hits_in_rallies(events, self.RALLIES)
        self.assertEqual(len(hits), 2)

    def test_annotates_rally_idx_hit_idx_and_first_of_rally(self):
        events = [
            {"id": "e2", "type": "hit", "t_s": 8.0},
            {"id": "e1", "type": "hit", "t_s": 6.0},   # unsorted input on purpose
            {"id": "e3", "type": "hit", "t_s": 21.0},
        ]
        hits = sc.hits_in_rallies(events, self.RALLIES)
        # sorted by time; hit_idx is 1-based within its rally; only the earliest hit of a
        # rally carries the serve hint.
        self.assertEqual(
            [(h["event"]["id"], h["rally_idx"], h["hit_idx"], h["first_of_rally"]) for h in hits],
            [("e1", 0, 1, True), ("e2", 0, 2, False), ("e3", 1, 1, True)],
        )

    def test_no_rallies_yields_no_hits(self):
        events = [{"id": "e1", "type": "hit", "t_s": 6.0}]
        self.assertEqual(sc.hits_in_rallies(events, []), [])


class TestClipFilename(unittest.TestCase):
    def test_builds_sortable_self_describing_name(self):
        name = sc.clip_filename("Session2_Game_1.mp4", rally_idx=3, hit_idx=5,
                                t_s=12.4, event_id="abcdef12-3456-7890-abcd-ef1234567890")
        self.assertEqual(name, "Session2_Game_1_r03_h05_0012.40s_abcdef12.mp4")

    def test_time_padding_keeps_lexicographic_order(self):
        early = sc.clip_filename("g.mp4", rally_idx=0, hit_idx=1, t_s=9.5, event_id="a" * 8)
        late = sc.clip_filename("g.mp4", rally_idx=10, hit_idx=1, t_s=100.0, event_id="b" * 8)
        self.assertLess(early, late)

    def test_sanitizes_awkward_label_characters(self):
        name = sc.clip_filename("Halle Doppel (Test).mov", rally_idx=0, hit_idx=1,
                                t_s=1.0, event_id="x" * 8)
        self.assertTrue(name.startswith("Halle_Doppel_Test_"))
        self.assertNotIn(" ", name)
        self.assertNotIn("(", name)


class TestTaskDict(unittest.TestCase):
    def test_builds_label_studio_task(self):
        hit = {"event": {"id": "abcdef12-3456", "t_s": 12.4},
               "rally_idx": 3, "hit_idx": 5, "first_of_rally": False}
        task = sc.task_dict(hit, game="Session2_Game_1",
                            clip_relpath="stroke/Session2_Game_1/clip.mp4",
                            video_id="vid-1", match_id="m-1")
        self.assertEqual(task["data"]["video"], "/data/local-files/?d=stroke/Session2_Game_1/clip.mp4")
        self.assertEqual(task["data"]["game"], "Session2_Game_1")
        self.assertEqual(task["data"]["rally_idx"], 3)
        self.assertEqual(task["data"]["hit_idx"], 5)
        self.assertEqual(task["data"]["hit_t_s"], 12.4)
        # shown to the labeler as a hint, never auto-filled (bias risk)
        self.assertEqual(task["data"]["serve_hint"], "")
        # IDs ride along so import-stroke-labels can join back to DB rows without guesswork
        self.assertEqual(task["data"]["event_id"], "abcdef12-3456")
        self.assertEqual(task["data"]["video_id"], "vid-1")
        self.assertEqual(task["data"]["match_id"], "m-1")

    def test_first_of_rally_sets_serve_hint(self):
        hit = {"event": {"id": "e", "t_s": 5.0},
               "rally_idx": 0, "hit_idx": 1, "first_of_rally": True}
        task = sc.task_dict(hit, game="g", clip_relpath="stroke/g/c.mp4",
                            video_id="v", match_id="m")
        self.assertEqual(task["data"]["serve_hint"],
                         "Erster Schlag im Ballwechsel (vermutlich Aufschlag)")


class TestFfmpegCmd(unittest.TestCase):
    def test_frame_accurate_cut_with_audio(self):
        cmd = sc.ffmpeg_cmd("/cache/proxy_720p.mp4", "/out/clip.mp4", start=9.3, dur=1.4)
        # -ss BEFORE -i: fast keyframe seek that is still frame-accurate because we re-encode.
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-ss") + 1], "9.300")
        self.assertEqual(cmd[cmd.index("-t") + 1], "1.400")
        # re-encode (NOT -c copy): stream-copy would snap the cut to the proxy's 2 s keyframes.
        self.assertNotIn("copy", cmd)
        self.assertIn("h264_videotoolbox", cmd)
        # audio kept — the hit sound is a first-class labeling signal.
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "aac")
        self.assertEqual(cmd[-1], "/out/clip.mp4")
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-y", cmd)


class TestSampleEvenly(unittest.TestCase):
    def test_returns_all_when_n_exceeds_length(self):
        self.assertEqual(sc.sample_evenly([1, 2, 3], 5), [1, 2, 3])

    def test_deterministic_even_spread_keeps_ends(self):
        # deterministic (reproducible pilot batches) — first and last always included.
        picked = sc.sample_evenly(list(range(10)), 3)
        self.assertEqual(len(picked), 3)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 9)
        self.assertEqual(picked, sorted(picked))

    def test_zero_or_negative_n_returns_empty(self):
        self.assertEqual(sc.sample_evenly([1, 2, 3], 0), [])
