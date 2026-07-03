"""Supabase API store — all worker DB + storage I/O over the service-role REST client.

The self-hosted Supabase (Coolify) exposes only the Kong API gateway, not direct Postgres, so
`SUPABASE_DB_URL` is blank and we go through supabase-py (PostgREST) instead of psycopg. The service
role bypasses RLS (specs/40-worker-contract.md: "all worker writes use service-role").

Job claiming is a conditional update (`...where id=$ and status='queued'`) which is atomic enough for a
single local worker — the guard makes a double-claim a no-op rather than a race. This mirrors the
JobStore interface (claim_job/set_stage/finish) so __main__ can use either path.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("worker.store")

DEFAULT_STALE_JOB_TIMEOUT_S = 3600.0


def _now() -> str:
    """ISO-8601 UTC timestamp. PostgREST sends values verbatim, so we cannot pass SQL now()."""
    return datetime.now(timezone.utc).isoformat()


def _rally_insert_rows(match_id: str, rallies: List[dict]) -> List[dict]:
    """Map build_rallies() rally dicts → `rallies` table rows (scoped to the match)."""
    return [{"match_id": match_id, "idx": r["idx"], "t_start_s": r["t_start_s"],
             "t_end_s": r["t_end_s"], "shot_count": r["shot_count"],
             "winner_side": r["winner_side"], "end_shot_type": r["end_shot_type"],
             "confidence": r["confidence"]} for r in rallies]


def _event_insert_rows(match_id: str, events: List[dict], id_by_idx: Dict[int, str]) -> List[dict]:
    """Map build_rallies() event dicts → `events` table rows, resolving rally_idx → the inserted rally_id."""
    return [{"match_id": match_id, "rally_id": id_by_idx.get(e["rally_idx"]),
             "t_s": e["t_s"], "type": e["type"], "payload": e["payload"],
             "confidence": e["confidence"]} for e in events]


class Store:
    def __init__(
        self,
        supabase_url: str,
        service_role_key: str,
        stale_job_timeout_s: float = DEFAULT_STALE_JOB_TIMEOUT_S,
    ) -> None:
        from supabase import create_client  # lazy: only when running against Supabase

        self._sb = create_client(supabase_url, service_role_key)
        self._url = supabase_url.rstrip("/")
        self._key = service_role_key
        self._worker_id = "worker-1"
        self._stale_job_timeout_s = stale_job_timeout_s

    def with_worker(self, worker_id: str) -> "Store":
        self._worker_id = worker_id
        return self

    # ── job queue ──────────────────────────────────────────────────────────
    def claim_job(self) -> Optional[dict]:
        """Claim the oldest queued job; with the queue empty, reclaim a stale running one (a job whose
        worker died leaves status='running' forever — specs/40). `claimed_at` is the liveness signal:
        set_stage refreshes it, so "stale" means no stage finished within the timeout."""
        job = self._claim_where(status="queued")
        if job is not None:
            return job
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=self._stale_job_timeout_s)
        ).isoformat()
        job = self._claim_where(status="running", claimed_before=cutoff)
        if job is not None:
            log.warning("reclaimed stale job %s (no heartbeat since %s)", job["id"], cutoff)
        return job

    def _claim_where(self, status: str, claimed_before: Optional[str] = None) -> Optional[dict]:
        """Select-then-conditional-update claim. The update re-applies the selection filters, so a
        concurrent claim makes it a no-op (empty data) instead of a double-claim."""
        q = (
            self._sb.table("processing_jobs")
            .select("id, match_id, video_id, stage, params")
            .eq("status", status)
        )
        if claimed_before is not None:
            q = q.lt("claimed_at", claimed_before)
        rows = (q.order("created_at").limit(1).execute()).data or []
        if not rows:
            return None
        job = rows[0]
        upd = (
            self._sb.table("processing_jobs")
            .update(
                {
                    "status": "running",
                    "claimed_by": self._worker_id,
                    "claimed_at": _now(),
                    "started_at": _now(),
                }
            )
            .eq("id", job["id"])
            .eq("status", status)  # optimistic guard
        )
        if claimed_before is not None:
            upd = upd.lt("claimed_at", claimed_before)
        if not (upd.execute().data or []):
            return None  # lost the race; caller retries on next poll
        return job

    def set_stage(self, job_id: str, stage: str, progress: float) -> None:
        # claimed_at doubles as the heartbeat (see claim_job) — refresh it on every stage.
        self._sb.table("processing_jobs").update(
            {"stage": stage, "progress": progress, "claimed_at": _now()}
        ).eq("id", job_id).execute()

    def finish(self, job_id: str, ok: bool, error: Optional[str] = None) -> None:
        patch: Dict[str, Any] = {
            "status": "succeeded" if ok else "failed",
            "error": error,
            "finished_at": _now(),
        }
        if ok:
            patch["progress"] = 1
        self._sb.table("processing_jobs").update(patch).eq("id", job_id).execute()

    def enqueue(
        self, match_id: str, video_id: Optional[str], stage: str, params: Optional[dict] = None
    ) -> dict:
        row = {
            "match_id": match_id,
            "video_id": video_id,
            "stage": stage,
            "status": "queued",
            "params": params or {},
        }
        res = self._sb.table("processing_jobs").insert(row).execute()
        return (res.data or [row])[0]

    # ── matches / videos ───────────────────────────────────────────────────
    def get_match(self, match_id: str) -> Optional[dict]:
        res = self._sb.table("matches").select("*").eq("id", match_id).limit(1).execute()
        return (res.data or [None])[0]

    def get_match_by_slug(self, slug: str) -> Optional[dict]:
        res = self._sb.table("matches").select("*").eq("slug", slug).limit(1).execute()
        return (res.data or [None])[0]

    def update_match_status(self, match_id: str, status: str, **extra: Any) -> None:
        self._sb.table("matches").update({"status": status, **extra}).eq("id", match_id).execute()

    def get_video(self, video_id: str) -> Optional[dict]:
        res = self._sb.table("match_videos").select("*").eq("id", video_id).limit(1).execute()
        return (res.data or [None])[0]

    def get_video_by_match(self, match_id: str) -> Optional[dict]:
        res = (
            self._sb.table("match_videos")
            .select("*")
            .eq("match_id", match_id)
            .limit(1)
            .execute()
        )
        return (res.data or [None])[0]

    def update_video_meta(
        self, video_id: str, fps: float, duration_s: float, width: int, height: int
    ) -> None:
        self._sb.table("match_videos").update(
            {"fps": fps, "duration_s": duration_s, "width": width, "height": height}
        ).eq("id", video_id).execute()

    def set_video_proxy(self, video_id: str, proxy_path: str) -> None:
        self._sb.table("match_videos").update({"proxy_path": proxy_path}).eq(
            "id", video_id
        ).execute()

    # ── calibration ────────────────────────────────────────────────────────
    def get_calibration(self, video_id: str) -> Optional[dict]:
        res = (
            self._sb.table("court_calibrations")
            .select("*")
            .eq("video_id", video_id)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        return (res.data or [None])[0]

    def insert_calibration(self, fields: dict) -> dict:
        res = self._sb.table("court_calibrations").insert(fields).execute()
        return (res.data or [fields])[0]

    def update_calibration(self, calibration_id: str, fields: dict) -> None:
        self._sb.table("court_calibrations").update({**fields, "updated_at": _now()}).eq(
            "id", calibration_id
        ).execute()

    # ── players / stats / tracks (Slice 2 mostly) ──────────────────────────
    def get_match_players(self, match_id: str) -> List[dict]:
        res = self._sb.table("match_players").select("*").eq("match_id", match_id).execute()
        return res.data or []

    def set_track_label(self, match_player_id: str, track_label: int) -> None:
        self._sb.table("match_players").update({"track_label": track_label}).eq(
            "id", match_player_id
        ).execute()

    def upsert_player_match_stats(self, row: dict) -> None:
        self._sb.table("player_match_stats").upsert(
            row, on_conflict="match_id,player_id"
        ).execute()

    def upsert_player_shot_errors(
        self, match_id: str, player_id: str, shot_type: str, errors: int
    ) -> None:
        """Errors from rally-ending out landings, per shot_type (``'unknown'`` until stage 7 exists).

        Only the ``errors`` column is sent — PostgREST merge-duplicates leaves a future stroke-stage
        ``count`` on the same row untouched."""
        self._sb.table("player_shot_stats").upsert(
            {"match_id": match_id, "player_id": player_id, "shot_type": shot_type,
             "errors": errors},
            on_conflict="match_id,player_id,shot_type",
        ).execute()

    def upsert_track_keyframes(self, video_id: str, subject: str, data: list) -> None:
        self._sb.table("track_keyframes").upsert(
            {"video_id": video_id, "subject": subject, "data": data},
            on_conflict="video_id,subject",
        ).execute()

    def upsert_detection_corrections(self, rows: List[dict]) -> None:
        """Persist verify-tool corrections (the scorecard answer key). Idempotent on (video_id,source_id)
        so re-importing the same corrections.json replaces rather than duplicates (0004 migration)."""
        if not rows:
            return
        self._sb.table("detection_corrections").upsert(
            rows, on_conflict="video_id,source_id"
        ).execute()

    def upsert_shuttle_labels(self, rows: List[dict]) -> None:
        """Persist verify-tool shuttle labels (the fine-tune answer key + scoreboard GT). Idempotent on
        (video_id,source_id) where source_id = frame_idx, so re-importing the same shuttle_labels.json
        replaces rather than duplicates (0005 migration)."""
        if not rows:
            return
        self._sb.table("shuttle_labels").upsert(
            rows, on_conflict="video_id,source_id"
        ).execute()

    def insert_rally_corrections(self, rows: List[dict]) -> None:
        """Append-only persist of rally-review corrections (the rally_hit eval/threshold-tuning
        corpus, 0011 migration). Unlike upsert_detection_corrections/upsert_shuttle_labels this is
        a PLAIN INSERT: editing the same rally across separate review saves is itself useful
        signal, so repeats are kept rather than replaced."""
        if not rows:
            return
        self._sb.table("rally_corrections").insert(rows).execute()

    def get_rallies(self, match_id: str) -> List[dict]:
        """A match's rally rows, ordered by idx (rally_sync maps events' rally_id → idx off these)."""
        res = self._sb.table("rallies").select("*").eq("match_id", match_id).order("idx").execute()
        return res.data or []

    def get_events(self, match_id: str) -> List[dict]:
        res = self._sb.table("events").select("*").eq("match_id", match_id).execute()
        return res.data or []

    def replace_rallies(self, match_id: str, rallies: List[dict], events: List[dict]) -> int:
        """Idempotently replace a match's rallies + events (stage 6 output). Returns the rally count.

        The ONLY delete-then-insert in this module: `events` has no natural upsert key (unlike `rallies`'
        unique(match_id,idx)), so a re-run must clear the match's rows first or events would pile up.
        PostgREST refuses an unfiltered delete, so the `.eq("match_id", …)` filter is load-bearing — never
        drop it. Events are deleted before rallies (events.rally_id FKs rallies)."""
        row = (self._sb.table("matches").select("rally_edited_at").eq("id", match_id)
               .execute().data or [{}])[0]
        if row.get("rally_edited_at"):
            # FR-35 rally review: the trainer dropped/merged rallies; a full re-run regenerates them
            # from scratch. Surface the clobber loudly (specs/features/rally-shot-review.md).
            log.warning("replace_rallies: overwriting trainer rally edits for match %s "
                        "(rally_edited_at=%s) — review the rallies again after this run",
                        match_id, row["rally_edited_at"])
        self._sb.table("events").delete().eq("match_id", match_id).execute()
        self._sb.table("rallies").delete().eq("match_id", match_id).execute()
        if not rallies:
            return 0
        inserted = (
            self._sb.table("rallies").insert(_rally_insert_rows(match_id, rallies)).execute().data or []
        )
        id_by_idx = {row["idx"]: row["id"] for row in inserted}
        event_rows = _event_insert_rows(match_id, events, id_by_idx)
        if event_rows:
            self._sb.table("events").insert(event_rows).execute()
        return len(rallies)

    def log_job_run(
        self, job: str, match_id: Optional[str], is_ok: bool, detail: str = ""
    ) -> None:
        try:
            self._sb.table("job_runs").insert(
                {"job": job, "match_id": match_id, "is_ok": is_ok, "detail": detail,
                 "finished_at": _now()}
            ).execute()
        except Exception as e:  # noqa: BLE001 — audit log must never break a job
            log.warning("job_runs insert failed: %s", e)

    # ── creation helpers (enqueue CLI) ─────────────────────────────────────
    def create_player(self, name: str, first_name: str) -> dict:
        res = self._sb.table("players").insert({"name": name, "first_name": first_name}).execute()
        return (res.data or [{}])[0]

    def create_match(self, slug: str, discipline: str, **extra: Any) -> dict:
        row = {"slug": slug, "discipline": discipline, "status": "uploaded",
               "is_test": True, **extra}
        res = self._sb.table("matches").insert(row).execute()
        return (res.data or [row])[0]

    def create_video(self, match_id: str, storage_path: str, **extra: Any) -> dict:
        row = {"match_id": match_id, "storage_path": storage_path, **extra}
        res = self._sb.table("match_videos").insert(row).execute()
        return (res.data or [row])[0]

    def create_match_player(
        self, match_id: str, player_id: str, side: str, slot: int
    ) -> dict:
        row = {"match_id": match_id, "player_id": player_id, "side": side, "slot": slot}
        res = self._sb.table("match_players").insert(row).execute()
        return (res.data or [row])[0]

    # ── storage ────────────────────────────────────────────────────────────
    def download(self, bucket: str, path: str) -> bytes:
        return self._sb.storage.from_(bucket).download(path)

    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        self._sb.storage.from_(bucket).upload(
            path, data, {"content-type": content_type, "upsert": "true"}
        )

    def upload_resumable(
        self, bucket: str, path: str, file_path: str | Path, content_type: str
    ) -> None:
        """TUS chunked upload for large files. The Kong gateway kills any single request at
        ~60 s (measured 2026-07-02: HTTP 499 on a 60 MB plain POST at ~9 Mbps), so anything
        bigger than ~40 MB must go up in chunks. Supabase's resumable endpoint expects
        exactly 6 MiB chunks (except the last)."""
        import base64

        import httpx

        chunk_size = 6 * 1024 * 1024
        file_path = Path(file_path)
        size = file_path.stat().st_size

        # The self-hosted Storage ignores x-upsert on the TUS path: with an existing object
        # at the target, the creation POST still succeeds but the first PATCH 409s, and a
        # plain retry 409s again (verified live 2026-07-02). remove() is a no-op success for
        # missing objects, so delete unconditionally before the creation POST.
        self._sb.storage.from_(bucket).remove([path])

        def b64(s: str) -> str:
            return base64.b64encode(s.encode()).decode()

        headers = {"Authorization": f"Bearer {self._key}", "Tus-Resumable": "1.0.0"}
        with httpx.Client(timeout=120) as client:
            res = client.post(
                f"{self._url}/storage/v1/upload/resumable",
                headers={
                    **headers,
                    "Upload-Length": str(size),
                    "x-upsert": "true",
                    "Upload-Metadata": (
                        f"bucketName {b64(bucket)},objectName {b64(path)},"
                        f"contentType {b64(content_type)}"
                    ),
                },
            )
            res.raise_for_status()
            loc = res.headers["location"]
            if loc.startswith("/"):
                loc = self._url + loc
            offset = 0
            with file_path.open("rb") as f:
                while offset < size:
                    chunk = f.read(chunk_size)
                    res = client.patch(
                        loc,
                        headers={
                            **headers,
                            "Upload-Offset": str(offset),
                            "Content-Type": "application/offset+octet-stream",
                        },
                        content=chunk,
                    )
                    res.raise_for_status()
                    offset = int(res.headers["upload-offset"])
