"""Diagnostic: confirm the worker's .env loads and the service_role key works. Prints NO secrets.

Run:  worker/.venv/bin/python worker/scripts/check_env.py   (from the repo root)
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

# .../worker — the directory that CONTAINS the `worker` package (worker/worker/)
WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKER_DIR)

from worker.config import load_config  # noqa: E402


def jwt_role(token: str):
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        data = json.loads(base64.urlsafe_b64decode(part))
        return data.get("role"), data.get("exp")
    except Exception:
        return None, None


def main() -> int:
    cfg = load_config()
    print(f"configured           = {cfg.configured}")
    print(f"SUPABASE_URL set      = {bool(cfg.supabase_url)}")
    print(f"SERVICE_ROLE set      = {bool(cfg.service_role_key)} (len={len(cfg.service_role_key)})")
    print(f"DB_URL set            = {bool(cfg.db_url)}")
    role, exp = jwt_role(cfg.service_role_key)
    print(f"service_role JWT role = {role}  exp={exp}")

    if not cfg.configured:
        print("RESULT: not configured — fill SUPABASE_URL + SERVICE_ROLE in .env")
        return 1

    # Live API: insert+delete a job_runs row as service_role (proves the key is valid + RLS bypass).
    base = cfg.supabase_url.rstrip("/") + "/rest/v1/job_runs"
    hdr = {
        "apikey": cfg.service_role_key,
        "Authorization": f"Bearer {cfg.service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    body = json.dumps({"job": "env_check", "is_ok": True, "detail": "worker env load test"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(base, data=body, headers=hdr, method="POST"), timeout=20) as r:
            print(f"service_role INSERT   = HTTP {r.status} (write OK)")
        with urllib.request.urlopen(urllib.request.Request(base + "?job=eq.env_check", headers=hdr, method="DELETE"), timeout=20) as r:
            print(f"service_role DELETE   = HTTP {r.status} (cleanup OK)")
    except urllib.error.HTTPError as e:
        print(f"API check FAILED: HTTP {e.code} — {e.read().decode()[:200]}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"API check error: {e}")
        return 1
    print("RESULT: OK — env loads and service_role works against the live API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
