"""proxy — 720p playback proxy stage (specs/features/video-proxy.md).

The stage transcodes the source to a small streamable MP4 (ffmpeg CLI, VideoToolbox) and
uploads it via TUS resumable chunks — the Kong gateway kills any single request at ~60 s
(spiked 2026-07-02), which a plain POST of the larger games would hit. The stage must NEVER
raise: process_job's failure path sets matches.status='failed', and a supplementary playback
artifact must not knock a match out of review. We test against a fake store and mocked
ffmpeg (shutil.which + subprocess.run patched in the proxy module).
"""
import base64
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from worker.pipeline import proxy
from worker.pipeline.ingest import resolve_source
from worker.store import Store


class _Cfg:
    def __init__(self, cache_dir: str, video_dir: str):
        self.frame_cache_dir = cache_dir
        self.local_video_dir = video_dir
        self.artifact_bucket = "analysis-artifacts"
        self.video_bucket = "match-videos"


class _Store:
    """Canned match_videos row + recorders for resumable uploads / proxy_path writes."""

    def __init__(self, video=None, upload_error=None):
        self._video = video
        self._upload_error = upload_error
        self.resumable_uploads: list[tuple[str, str, str]] = []
        self.proxy_set: list[tuple[str, str]] = []
        self.downloaded: tuple[str, str] | None = None

    def get_video(self, video_id):
        return self._video

    def upload_resumable(self, bucket, path, file_path, content_type):
        if self._upload_error:
            raise self._upload_error
        self.resumable_uploads.append((bucket, path, content_type))

    def set_video_proxy(self, video_id, proxy_path):
        self.proxy_set.append((video_id, proxy_path))

    def download(self, bucket, path):
        self.downloaded = (bucket, path)
        return b"video-bytes"


def _video(**extra):
    return {"id": "vid-1", "storage_path": "Game.mp4", "proxy_path": None, **extra}


def _ctx(tmp, store, params=None):
    videos = Path(tmp) / "videos"
    videos.mkdir(exist_ok=True)
    return {
        "store": store,
        "cfg": _Cfg(str(Path(tmp) / "cache"), str(videos)),
        "job": {"video_id": "vid-1", "match_id": "M1", "params": params or {}},
    }


def _fake_ffmpeg_ok(cmd, **kw):
    """Stand-in transcode: write the output file the real ffmpeg would produce."""
    Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
    Path(cmd[-1]).write_bytes(b"proxy-bytes")
    return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")


def _with_source(ctx):
    """Drop a nonempty source file into the local video dir (decode-in-place path)."""
    src = Path(ctx["cfg"].local_video_dir) / "Game.mp4"
    src.write_bytes(b"4k-source")
    return src


class TestFfmpegCmd(unittest.TestCase):
    def test_cmd_is_the_spike_proven_recipe(self):
        cmd = proxy._ffmpeg_cmd(Path("/in.mp4"), Path("/out.mp4"))
        joined = " ".join(cmd)
        self.assertIn("scale=-2:720", joined)          # 720p, aspect preserved
        self.assertIn("-r 30", joined)                 # halved fps
        self.assertIn("h264_videotoolbox", joined)     # hw encode
        self.assertIn("0:a:0?", joined)                # audio optional (hit sounds)
        self.assertIn("+faststart", joined)            # progressive streaming
        self.assertEqual(cmd[-1], "/out.mp4")


class TestProxyRun(unittest.TestCase):
    def test_happy_path_uploads_and_records_proxy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=_video())
            ctx = _ctx(tmp, store)
            _with_source(ctx)
            with patch.object(proxy.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(proxy.subprocess, "run", side_effect=_fake_ffmpeg_ok):
                proxy.run(ctx)
            self.assertEqual(store.resumable_uploads,
                             [("analysis-artifacts", "vid-1/proxy_720p.mp4", "video/mp4")])
            self.assertEqual(store.proxy_set, [("vid-1", "vid-1/proxy_720p.mp4")])

    def test_skips_when_proxy_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=_video(proxy_path="vid-1/proxy_720p.mp4"))
            with patch.object(proxy.subprocess, "run") as run_mock:
                proxy.run(_ctx(tmp, store))
            run_mock.assert_not_called()
            self.assertEqual(store.resumable_uploads, [])

    def test_force_param_regenerates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=_video(proxy_path="vid-1/proxy_720p.mp4"))
            ctx = _ctx(tmp, store, params={"force": True})
            _with_source(ctx)
            with patch.object(proxy.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(proxy.subprocess, "run", side_effect=_fake_ffmpeg_ok):
                proxy.run(ctx)
            self.assertEqual(len(store.resumable_uploads), 1)

    def test_missing_ffmpeg_warns_and_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=_video())
            with patch.object(proxy.shutil, "which", return_value=None), \
                 self.assertLogs("worker.pipeline.proxy", level="WARNING"):
                proxy.run(_ctx(tmp, store))   # must not raise
            self.assertEqual(store.resumable_uploads, [])

    def test_ffmpeg_failure_warns_and_returns(self):
        def fail(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom")
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=_video())
            ctx = _ctx(tmp, store)
            _with_source(ctx)
            with patch.object(proxy.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(proxy.subprocess, "run", side_effect=fail), \
                 self.assertLogs("worker.pipeline.proxy", level="WARNING"):
                proxy.run(ctx)
            self.assertEqual(store.resumable_uploads, [])
            self.assertEqual(store.proxy_set, [])

    def test_upload_failure_never_raises_and_skips_proxy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=_video(), upload_error=RuntimeError("gateway timeout"))
            ctx = _ctx(tmp, store)
            _with_source(ctx)
            with patch.object(proxy.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(proxy.subprocess, "run", side_effect=_fake_ffmpeg_ok), \
                 self.assertLogs("worker.pipeline.proxy", level="WARNING"):
                proxy.run(ctx)   # must not raise
            self.assertEqual(store.proxy_set, [])

    def test_missing_video_row_warns_and_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store(video=None)
            with self.assertLogs("worker.pipeline.proxy", level="WARNING"):
                proxy.run(_ctx(tmp, store))
            self.assertEqual(store.resumable_uploads, [])


class _FakeTusResponse:
    def __init__(self, status_code, headers=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class _FakeTusStorage:
    """Mimics the live self-hosted Storage TUS semantics (verified 2026-07-02): the creation
    POST succeeds even with x-upsert when the object already exists — the first PATCH then
    409s; a plain retry 409s again; delete-then-reupload succeeds. Doubles as the httpx.Client
    stand-in (upload_resumable constructs one internally)."""

    def __init__(self, existing=()):
        self.objects = set(existing)
        self._pending = {}  # location -> (object path, declared length)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None):
        meta = dict(p.strip().split(" ", 1) for p in headers["Upload-Metadata"].split(","))
        path = base64.b64decode(meta["objectName"]).decode()
        loc = f"/upload/resumable/session-{len(self._pending)}"
        self._pending[loc] = (path, int(headers["Upload-Length"]))
        return _FakeTusResponse(201, {"location": loc})

    def patch(self, url, headers=None, content=b""):
        path, length = self._pending[url[url.index("/upload/resumable"):]]
        if path in self.objects:
            return _FakeTusResponse(409, text="The resource already exists")
        offset = int(headers["Upload-Offset"]) + len(content)
        if offset >= length:
            self.objects.add(path)
        return _FakeTusResponse(204, {"upload-offset": str(offset)})


class _FakeStorageAPI:
    """storage.from_(bucket) surface: remove() is a no-op success for missing objects
    (verified live), which is what lets upload_resumable delete unconditionally."""

    def __init__(self, tus):
        self._tus = tus
        self.removed = []

    def from_(self, bucket):
        return self

    def remove(self, paths):
        self.removed.extend(paths)
        for p in paths:
            self._tus.objects.discard(p)
        return []


class TestUploadResumableOverwrite(unittest.TestCase):
    """upload_resumable must overwrite existing objects (specs/fix-later.md TUS 409): the
    Storage version ignores x-upsert on the TUS path, so the fix deletes the target first."""

    def _store(self, tus):
        store = Store.__new__(Store)  # skip __init__ (would build a live supabase client)
        store._url = "http://fake-kong"
        store._key = "service-role-key"
        store._sb = SimpleNamespace(storage=_FakeStorageAPI(tus))
        return store

    def _upload(self, store, tus):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "proxy_720p.mp4"
            f.write_bytes(b"new-proxy-bytes")
            with patch("httpx.Client", return_value=tus):
                store.upload_resumable("analysis-artifacts", "vid-1/proxy_720p.mp4", f, "video/mp4")

    def test_overwrites_existing_object(self):
        tus = _FakeTusStorage(existing={"vid-1/proxy_720p.mp4"})
        store = self._store(tus)
        self._upload(store, tus)  # must not raise despite the pre-existing object
        self.assertIn("vid-1/proxy_720p.mp4", tus.objects)
        self.assertEqual(store._sb.storage.removed, ["vid-1/proxy_720p.mp4"])

    def test_fresh_path_uploads_fine(self):
        tus = _FakeTusStorage()
        store = self._store(tus)
        self._upload(store, tus)
        self.assertIn("vid-1/proxy_720p.mp4", tus.objects)


class TestResolveSource(unittest.TestCase):
    """resolve_source (extracted from ingest): local videos/ → cached download → bucket fetch."""

    def test_prefers_local_file_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store()
            ctx = _ctx(tmp, store)
            src = _with_source(ctx)
            cache = Path(tmp) / "cache" / "vid-1"
            got = resolve_source(ctx["cfg"], "Game.mp4", cache, store)
            self.assertEqual(got, src)
            self.assertIsNone(store.downloaded)

    def test_uses_cached_download_without_refetching(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store()
            ctx = _ctx(tmp, store)
            cache = Path(tmp) / "cache" / "vid-1"
            cache.mkdir(parents=True)
            cached = cache / "source.mp4"
            cached.write_bytes(b"already-here")
            got = resolve_source(ctx["cfg"], "Game.mp4", cache, store)
            self.assertEqual(got, cached)
            self.assertIsNone(store.downloaded)

    def test_downloads_from_video_bucket_as_last_resort(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _Store()
            ctx = _ctx(tmp, store)
            cache = Path(tmp) / "cache" / "vid-1"
            got = resolve_source(ctx["cfg"], "Game.mp4", cache, store)
            self.assertEqual(store.downloaded, ("match-videos", "Game.mp4"))
            self.assertEqual(got.read_bytes(), b"video-bytes")


if __name__ == "__main__":
    unittest.main()
