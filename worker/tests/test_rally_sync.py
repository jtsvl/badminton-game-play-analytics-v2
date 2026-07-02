"""rally_sync — DB→cache resync of rallies/events before aggregate (FR-35 rally review).

The trainer's drop/merge edits live in Postgres, but `aggregate._rally_spans` reads
`rallies.parquet` from the cache — so a review job must rewrite the parquet from the
(authoritative) DB rows or the edit would be silently invisible. We test against a fake
store (canned DB rows + upload recorder) and real polars in a tmp cache dir, and lock the
aggregate contract by running `_rally_spans` over the file the stage wrote.
"""
import json
import tempfile
import unittest
from pathlib import Path

from worker.pipeline import aggregate, rally_sync


class _Cfg:
    def __init__(self, cache_dir: str, artifact_bucket: str = "analysis-artifacts"):
        self.frame_cache_dir = cache_dir
        self.artifact_bucket = artifact_bucket


class _Store:
    """Canned rally/event DB rows + a recorder for artifact uploads."""

    def __init__(self, rallies=None, events=None):
        self._rallies = rallies or []
        self._events = events or []
        self.uploads: list[tuple[str, str]] = []

    def get_rallies(self, match_id):
        return self._rallies

    def get_events(self, match_id):
        return self._events

    def upload(self, bucket, path, data, content_type):
        self.uploads.append((bucket, path))


def _db_rally(rid, idx, t0, t1, **extra):
    row = {"id": rid, "match_id": "M1", "idx": idx, "t_start_s": t0, "t_end_s": t1,
           "shot_count": 3, "winner_side": None, "end_shot_type": None,
           "confidence": 0.9, "edited_at": None}
    row.update(extra)
    return row


def _db_event(rally_id, t_s, etype="hit", payload=None):
    return {"id": f"ev-{t_s}", "match_id": "M1", "rally_id": rally_id, "t_s": t_s,
            "type": etype, "payload": payload, "confidence": 0.3}


def _ctx(tmp, store, video_id="vid-1"):
    return {"store": store, "cfg": _Cfg(tmp),
            "job": {"video_id": video_id, "match_id": "M1"}}


class TestRallySync(unittest.TestCase):
    def test_writes_rally_parquet_in_rally_hit_schema(self):
        import polars as pl
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(rallies=[_db_rally("r0", 0, 1.0, 4.0, edited_at="2026-07-02T10:00:00Z"),
                                    _db_rally("r2", 2, 8.0, 12.0)])
            ctx = _ctx(tmp, store)
            rally_sync.run(ctx)
            df = pl.read_parquet(ctx["rallies_path"])
            # exact stage-6 schema: DB-only columns (id, match_id, edited_at) are stripped
            self.assertEqual(df.columns, ["idx", "t_start_s", "t_end_s", "shot_count",
                                          "winner_side", "end_shot_type", "confidence"])
            self.assertEqual(df["idx"].to_list(), [0, 2])   # idx gaps survive as-is
            self.assertEqual(df["t_end_s"].to_list(), [4.0, 12.0])

    def test_events_remap_rally_id_to_idx_and_payload_to_json(self):
        import polars as pl
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(
                rallies=[_db_rally("r0", 0, 1.0, 4.0), _db_rally("r1", 1, 8.0, 12.0)],
                events=[_db_event("r0", 1.0, "serve", {"settled": True}),
                        _db_event("r1", 12.0, "rally_end", {"u": 0.5, "v": 0.95})])
            ctx = _ctx(tmp, store)
            rally_sync.run(ctx)
            df = pl.read_parquet(ctx["events_path"])
            self.assertEqual(df.columns, ["rally_idx", "t_s", "type", "payload", "confidence"])
            self.assertEqual(df["rally_idx"].to_list(), [0, 1])
            self.assertEqual(json.loads(df["payload"][0]), {"settled": True})

    def test_orphan_and_unknown_rally_events_are_skipped_with_a_log(self):
        import polars as pl
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(
                rallies=[_db_rally("r0", 0, 1.0, 4.0)],
                events=[_db_event("r0", 1.0),
                        _db_event(None, 5.0),          # orphaned by an earlier delete
                        _db_event("r-gone", 6.0)])     # unknown rally_id
            ctx = _ctx(tmp, store)
            with self.assertLogs("worker.pipeline.rally_sync", level="WARNING") as logs:
                rally_sync.run(ctx)
            df = pl.read_parquet(ctx["events_path"])
            self.assertEqual(len(df), 1)
            self.assertTrue(any("2" in m for m in logs.output))   # skipped count surfaced

    def test_zero_rallies_writes_empty_parquet_and_spans_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(tmp, _Store())
            rally_sync.run(ctx)
            cache = Path(tmp) / "vid-1"
            self.assertTrue((cache / "rallies.parquet").exists())
            # aggregate contract: empty parquet ⇒ None ⇒ whole-clip gating fallback
            self.assertIsNone(aggregate._rally_spans(ctx, cache))

    def test_aggregate_reads_the_edited_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(rallies=[_db_rally("r0", 0, 1.0, 4.0),
                                    _db_rally("r1", 1, 8.0, 12.0)])
            ctx = _ctx(tmp, store)
            rally_sync.run(ctx)
            spans = aggregate._rally_spans(ctx, Path(tmp) / "vid-1")
            self.assertEqual(spans, [(1.0, 4.0), (8.0, 12.0)])

    def test_uploads_both_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(rallies=[_db_rally("r0", 0, 1.0, 4.0)])
            rally_sync.run(_ctx(tmp, store, video_id="vid-9"))
            self.assertEqual(store.uploads,
                             [("analysis-artifacts", "vid-9/rallies.parquet"),
                              ("analysis-artifacts", "vid-9/events.parquet")])

    def test_creates_cache_dir_on_fresh_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(tmp, _Store())   # cache dir for vid-1 does not exist yet
            rally_sync.run(ctx)
            self.assertTrue(Path(ctx["rallies_path"]).exists())


if __name__ == "__main__":
    unittest.main()
