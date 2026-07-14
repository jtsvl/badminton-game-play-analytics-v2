"""Shared label→role→timeline binding (the fix for the one-track_label-per-player join).

`match_players.track_label` identifies WHO a player is (one raw tracker id, trainer-overridable
per FR-35) — but a player's actual timeline in tracks.parquet spans many ids: re-ID fragments
keep their own `track_id` (`source="reid"`), only interp gap-fills adopt the primary's. Binding
a consumer to the single labeled id silently drops every fragment row (the pose/overlay/stats
coverage bug, specs/features/pose-skeleton-overlay.md §Coverage). The correct binding is:
label → that tid's ROLE → every row of that role. Pure functions over plain dict rows.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

log = logging.getLogger("worker.pipeline.identity")


def track_roles(track_rows: Sequence[dict]) -> Dict[int, str]:
    """Map every track_id in tracks.parquet rows to its role; first seen role wins.

    build_role_tracks emits one role per tid — a second role for the same tid would be a bug
    upstream, so it is ignored with a warning rather than silently rebinding."""
    out: Dict[int, str] = {}
    for r in track_rows:
        tid, role = r.get("track_id"), r.get("role")
        if tid is None or not role:
            continue
        tid = int(tid)
        if tid in out:
            if out[tid] != role:
                log.warning("identity: track %d appears under roles %r and %r — keeping %r",
                            tid, out[tid], role, out[tid])
            continue
        out[tid] = role
    return out


def role_players(match_players: Sequence[dict], track_rows: Sequence[dict]) -> Dict[str, dict]:
    """role → match_player for every labeled player, via the labeled tid's role.

    The trainer may point track_label at any tid (primary or re-ID fragment) — the binding
    follows that tid's role. A label absent from tracks.parquet binds nothing (warned). If two
    players' labels resolve to the same role (mislabel), the label with the most non-interp rows
    wins and the loser binds nothing — loud, deterministic, and no double-drawn skeletons."""
    roles = track_roles(track_rows)
    presence = Counter(
        int(r["track_id"]) for r in track_rows
        if r.get("track_id") is not None and r.get("source", "detected") != "interp"
    )
    out: Dict[str, dict] = {}
    winner_label: Dict[str, int] = {}
    for mp in sorted(match_players, key=lambda m: (str(m.get("side")), int(m.get("slot") or 0))):
        label = mp.get("track_label")
        if label is None:
            continue
        label = int(label)
        role = roles.get(label)
        if role is None:
            log.warning("identity: track_label %d not in tracks.parquet — player %s binds nothing",
                        label, mp.get("player_id"))
            continue
        if role in out:
            log.warning("identity: labels %d and %d both resolve to role %r — keeping the most-present",
                        winner_label[role], label, role)
            if presence[label] <= presence[winner_label[role]]:
                continue
        out[role] = mp
        winner_label[role] = label
    return out


def role_frame_index(
    track_rows: Sequence[dict], t_to_frame: Mapping[float, int]
) -> Dict[Tuple[int, int], str]:
    """(frame, track_id) → role for every NON-interp row (via the manifest's t→frame inversion).

    interp rows are synthetic positions with no pose row behind them — excluding them here is
    what keeps the pose join from fabricating skeletons in gap-filled frames."""
    out: Dict[Tuple[int, int], str] = {}
    for r in track_rows:
        tid, role = r.get("track_id"), r.get("role")
        if tid is None or not role or r.get("source", "detected") == "interp":
            continue
        frame = t_to_frame.get(round(float(r["t"]), 6))
        if frame is None:
            continue
        out[(int(frame), int(tid))] = role
    return out
