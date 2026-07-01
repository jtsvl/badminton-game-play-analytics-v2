"""Environment-driven worker configuration (see .env.example / specs/40-worker-contract.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass(frozen=True)
class Config:
    supabase_url: str
    service_role_key: str
    db_url: str
    video_bucket: str
    artifact_bucket: str
    worker_id: str
    poll_interval_s: float
    frame_cache_dir: str
    local_video_dir: str
    calib_frame_count: int

    @property
    def configured(self) -> bool:
        """True when we can reach Supabase as the worker (service_role over the API URL).
        A direct `db_url` is optional — job claiming uses a service_role RPC over the API."""
        return bool(self.supabase_url and self.service_role_key)


def _load_dotenv_if_available() -> None:
    """Load the repo-root .env when python-dotenv is installed. No-op otherwise (keeps the
    dependency-free smoke run working); os.environ still wins for anything already set."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)


def load_config() -> Config:
    _load_dotenv_if_available()
    return Config(
        supabase_url=os.environ.get("SUPABASE_URL", ""),
        service_role_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        db_url=os.environ.get("SUPABASE_DB_URL", ""),
        video_bucket=os.environ.get("SUPABASE_VIDEO_BUCKET", "match-videos"),
        artifact_bucket=os.environ.get("SUPABASE_ARTIFACT_BUCKET", "analysis-artifacts"),
        worker_id=os.environ.get("WORKER_ID", "worker-1"),
        poll_interval_s=float(os.environ.get("WORKER_POLL_INTERVAL_SECONDS", "5")),
        frame_cache_dir=os.environ.get("WORKER_FRAME_CACHE_DIR", "./worker/.cache"),
        local_video_dir=os.environ.get("WORKER_LOCAL_VIDEO_DIR", str(_PROJECT_ROOT / "videos")),
        calib_frame_count=int(os.environ.get("CALIB_FRAME_COUNT", "12")),
    )
