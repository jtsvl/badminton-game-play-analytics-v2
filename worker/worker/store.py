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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("worker.store")


def _now() -> str:
    """ISO-8601 UTC timestamp. PostgREST sends values verbatim, so we cannot pass SQL now()."""
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, supabase_url: str, service_role_key: str) -> None:
        from supabase import create_client  # lazy: only when running against Supabase

        self._sb = create_client(supabase_url, service_role_key)
        self._worker_id = "worker-1"

    def with_worker(self, worker_id: str) -> "Store":
        self._worker_id = worker_id
        return self

    # ── job queue ──────────────────────────────────────────────────────────
    def claim_job(self) -> Optional[dict]:
        """Claim the oldest queued job. Conditional update => safe against a concurrent claim."""
        q = (
            self._sb.table("processing_jobs")
            .select("id, match_id, video_id, stage, params")
            .eq("status", "queued")
            .order("created_at")
            .limit(1)
            .execute()
        )
        rows = q.data or []
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
            .eq("status", "queued")  # optimistic guard
            .execute()
        )
        if not (upd.data or []):
            return None  # lost the race; caller retries on next poll
        return job

    def set_stage(self, job_id: str, stage: str, progress: float) -> None:
        self._sb.table("processing_jobs").update(
            {"stage": stage, "progress": progress}
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
