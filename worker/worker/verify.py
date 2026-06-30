"""Interactive frame-verification tool — scrub frames + inspect detections/tracking in the browser.

Run from the worker/ dir:  uv run python -m worker verify [video_id] [--port 8000]
Then open the printed URL. Slider / arrow keys step through frames; toggles show every detected
person (faint), the 4 kept in-court players (bold + label + foot dot), the calibrated court outline,
and a top-down map of where each player was placed.

Correction workflow (direct manipulation on the frame):
  • DRAG a player's coloured dot to where the feet really are  → records a position fix.
  • CLICK empty ground, then pick the player                  → records a missing / misplaced player.
  • CLICK a player's dot                                       → mark "kein Spieler" or fix the role.
All corrections are written to corrections.json in the video's cache folder and can be exported with
"JSON herunterladen". No new deps (stdlib http.server + PIL + polars).
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageFont

from .config import load_config
from .geometry import UNIT_SQUARE, ankle_point, apply_homography, solve_homography
from .store import Store

log = logging.getLogger("worker.verify")

ROLE_LABEL = {"near_left": "A1", "near_right": "A2", "far_left": "B1", "far_right": "B2", "near": "A", "far": "B"}
ROLE_COLOR = {
    "near_left": (250, 136, 50), "near_right": (255, 184, 102),
    "far_left": (23, 88, 166), "far_right": (96, 156, 222),
    "near": (250, 136, 50), "far": (23, 88, 166),
}
# colour keyed by the on-screen label ("A1"…) — for corrections, which carry labels not internal roles
LABEL_COLOR = {ROLE_LABEL[k]: v for k, v in ROLE_COLOR.items()}
OUT_W = 1280


def resolve_cors(cors) -> dict:
    """Collapse a frame's corrections to one effective override per player label.
    Returns {label: {"box":[x,y,w,h]|None, "foot":[nx,ny]|None, "uv":[u,v]|None,
                     "label":str|None, "spurious":bool}} — all normalised 0..1.
    Geometry precedence: an explicit box (bbox/added) wins; otherwise the latest foot_norm."""
    out: dict[str, dict] = {}
    for c in cors:
        role = c.get("role")
        if not role:
            continue
        o = out.setdefault(role, {"box": None, "foot": None, "uv": None, "label": None, "spurious": False})
        if c.get("type") == "spurious":
            o["spurious"] = True
        if c.get("correct_role"):           # wrong_role → relabel
            o["label"] = c["correct_role"]
        if c.get("bbox"):                   # bbox/added carry an explicit box
            o["box"] = c["bbox"]
        if c.get("foot_norm"):
            o["foot"] = c["foot_norm"]
        if c.get("court_uv"):
            o["uv"] = c["court_uv"]
    return out


@dataclass
class DebugData:
    cache: Path
    frames: list[dict]
    det: Any  # polars DataFrame
    trk: Any
    corners: list[tuple[float, float]]
    h_uv2px: list[float]
    role_by_tid: dict[int, str]
    width: int
    height: int


def load_data(video_id: str, cfg) -> DebugData:
    import polars as pl

    store = Store(cfg.supabase_url, cfg.service_role_key)
    cal = store.get_calibration(video_id)
    if not cal or not cal.get("confirmed_points"):
        raise SystemExit(f"no confirmed calibration for video {video_id}")
    corners = [(float(x), float(y)) for x, y in cal["confirmed_points"]]
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id
    manifest = json.loads((cache / "frames.json").read_text())
    det = pl.read_parquet(cache / "detections.parquet")
    trk = pl.read_parquet(cache / "tracks.parquet")
    role_by_tid = {int(r["track_id"]): r["role"] for r in trk.unique(subset=["track_id"]).iter_rows(named=True)}
    return DebugData(
        cache=cache, frames=manifest["frames"], det=det, trk=trk, corners=corners,
        h_uv2px=solve_homography(UNIT_SQUARE, corners), role_by_tid=role_by_tid,
        width=manifest["width"], height=manifest["height"],
    )


def map_rect(data: DebugData) -> dict:
    """The mini-map's rectangle in normalised image coords — must match the block in annotate()."""
    W, H = data.width, data.height
    mw = int(max(W, H) * 0.11)
    mh = int(mw * 13.40 / 6.10)
    pad = int(mw * 0.18)
    mx, my = W - mw - pad, pad
    return {"x": round(mx / W, 4), "y": round(my / H, 4), "w": round(mw / W, 4), "h": round(mh / H, 4)}


def kept_detections(data: DebugData, idx: int) -> list[dict]:
    """The 4 in-court players for this frame: pixel ankle/bbox (for the frame) + court uv (for the map)."""
    import polars as pl

    fmeta = data.frames[idx]
    t = fmeta["t"]
    kept_ids = list(data.role_by_tid)
    W, H = data.width, data.height
    # court position (u,v) per track, taken from the nearest-in-time track row — same as annotate's map dots
    uv_by_tid: dict[int, list[float]] = {}
    for tid in kept_ids:
        sub = sorted(data.trk.filter(pl.col("track_id") == tid).iter_rows(named=True), key=lambda r: abs(r["t"] - t))
        if sub:
            uv_by_tid[tid] = [round(sub[0]["u"], 4), round(sub[0]["v"], 4)]
    out = []
    for r in data.det.filter((pl.col("frame") == fmeta["frame"]) & (pl.col("track").is_in(kept_ids))).iter_rows(named=True):
        tid = int(r["track"])
        role = data.role_by_tid.get(tid, "")
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        ax, ay = ankle_point((x, y, w, h))
        out.append({
            "role": role,
            "label": ROLE_LABEL.get(role, role),
            "ankle": [round(ax / W, 4), round(ay / H, 4)],
            "bbox": [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)],
            "uv": uv_by_tid.get(tid),
        })
    return out


def annotate(data: DebugData, idx: int, *, show_all=True, show_court=True, show_map=True, hide=(), cors=()) -> Image.Image:
    import polars as pl

    fmeta = data.frames[idx]
    t = fmeta["t"]
    W, H = data.width, data.height
    sw = max(W, H) / 640
    font = ImageFont.load_default(size=int(max(W, H) * 0.02))
    kept_ids = list(data.role_by_tid)
    cor = resolve_cors(cors)   # {label: effective override} — moves the bold box/dot/label/map-dot

    img = Image.open(data.cache / "frames" / f"{fmeta['frame']:06d}.jpg").convert("RGB")
    d = ImageDraw.Draw(img)

    if show_all:
        for r in data.det.filter(pl.col("frame") == fmeta["frame"]).iter_rows(named=True):
            d.rectangle([r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]], outline=(170, 170, 170), width=int(sw))

    if show_court:
        d.line([*[tuple(p) for p in data.corners], tuple(data.corners[0])], fill=(255, 255, 255), width=int(sw * 2.2))
        d.line([apply_homography(data.h_uv2px, 0, 0.5), apply_homography(data.h_uv2px, 1, 0.5)], fill=(255, 255, 255), width=int(sw * 1.6))

    def draw_player(x, y, w, h, col, text):   # bold box + foot dot + label, at whatever (x,y,w,h) it's given
        d.rectangle([x, y, x + w, y + h], outline=col, width=int(sw * 3))
        ax, ay = ankle_point((x, y, w, h))
        d.ellipse([ax - sw * 5, ay - sw * 5, ax + sw * 5, ay + sw * 5], fill=col, outline=(255, 255, 255), width=int(sw))
        d.text((x, y - max(W, H) * 0.026), text, fill=col, font=font, stroke_width=int(sw), stroke_fill=(0, 0, 0))

    drawn = set()   # labels already baked from the parquet → don't double-draw as added/missing below
    for r in data.det.filter((pl.col("frame") == fmeta["frame"]) & (pl.col("track").is_in(kept_ids))).iter_rows(named=True):
        role = data.role_by_tid.get(int(r["track"]), "")
        label = ROLE_LABEL.get(role, role)
        if label in hide:   # hidden player → don't bake the bold box/dot/label
            continue
        drawn.add(label)
        o = cor.get(label)
        col = ROLE_COLOR.get(role, (250, 136, 50))
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        if o and o["box"]:                                   # explicit corrected box
            x, y, w, h = o["box"][0] * W, o["box"][1] * H, o["box"][2] * W, o["box"][3] * H
        elif o and o["foot"]:                                # position fix → slide the box to the new ankle
            x, y = o["foot"][0] * W - w / 2, o["foot"][1] * H - h
        if o and o["spurious"]:                              # "kein Spieler" → thin struck box, no bold marker
            d.rectangle([x, y, x + w, y + h], outline=(120, 120, 120), width=int(sw))
            d.line([x, y, x + w, y + h], fill=(200, 80, 80), width=int(sw))
            d.line([x + w, y, x, y + h], fill=(200, 80, 80), width=int(sw))
            continue
        draw_player(x, y, w, h, col, (o["label"] if o else None) or label)

    for label, o in cor.items():                             # added / missing players (no parquet detection)
        if label in drawn or label in hide or o["spurious"]:
            continue
        col = LABEL_COLOR.get(label, (250, 136, 50))
        if o["box"]:
            draw_player(o["box"][0] * W, o["box"][1] * H, o["box"][2] * W, o["box"][3] * H, col, o["label"] or label)
        elif o["foot"]:
            ax, ay = o["foot"][0] * W, o["foot"][1] * H
            d.ellipse([ax - sw * 5, ay - sw * 5, ax + sw * 5, ay + sw * 5], fill=col, outline=(255, 255, 255), width=int(sw))
            d.text((ax, ay - max(W, H) * 0.046), o["label"] or label, fill=col, font=font, stroke_width=int(sw), stroke_fill=(0, 0, 0))

    if show_map:
        mw = int(max(W, H) * 0.11)
        mh = int(mw * 13.40 / 6.10)
        pad = int(mw * 0.18)
        mx, my = W - mw - pad, pad
        d.rectangle([mx - pad, my - pad, mx + mw + pad, my + mh + pad], fill=(10, 10, 12))
        # court lines — all proportions from the BWF standard (13.40 m long × 6.10 m wide doubles)
        LN = (130, 130, 130)
        fx = lambda m: mx + mw * m / 6.10   # metres across the width → px
        fy = lambda m: my + mh * m / 13.40  # metres along the length → px
        # outer = doubles boundary
        d.rectangle([mx, my, mx + mw, my + mh], outline=(255, 255, 255), width=int(sw))
        # net (centre, 6.70 m from each back line) — brighter than the interior lines
        d.line([(mx, fy(6.70)), (mx + mw, fy(6.70))], fill=(200, 200, 200), width=int(sw))
        # short service lines, 1.98 m either side of the net
        for yy in (fy(6.70 - 1.98), fy(6.70 + 1.98)):
            d.line([(mx, yy), (mx + mw, yy)], fill=LN, width=int(sw))
        # doubles long service lines, 0.76 m from each back line
        for yy in (fy(0.76), fy(13.40 - 0.76)):
            d.line([(mx, yy), (mx + mw, yy)], fill=LN, width=int(sw))
        # singles side lines, 0.46 m inside each doubles side line (full length)
        for xx in (fx(0.46), fx(6.10 - 0.46)):
            d.line([(xx, my), (xx, my + mh)], fill=LN, width=int(sw))
        # centre line — splits the service courts: back line → short service line, on EACH half (never across the net)
        cx = mx + mw / 2
        d.line([(cx, my), (cx, fy(6.70 - 1.98))], fill=LN, width=int(sw))
        d.line([(cx, fy(6.70 + 1.98)), (cx, my + mh)], fill=LN, width=int(sw))
        def map_dot(u, v, col):
            px, py = mx + u * mw, my + v * mh
            d.ellipse([px - sw * 6, py - sw * 6, px + sw * 6, py + sw * 6], fill=col, outline=(255, 255, 255), width=int(sw))

        covered = set()
        for tid, role in data.role_by_tid.items():
            label = ROLE_LABEL.get(role, role)
            covered.add(label)
            if label in hide:   # hidden player → no map dot either
                continue
            o = cor.get(label)
            if o and o["spurious"]:
                continue
            if o and o["uv"]:                            # corrected court position
                map_dot(o["uv"][0], o["uv"][1], ROLE_COLOR.get(role, (250, 136, 50)))
                continue
            sub = sorted(data.trk.filter(pl.col("track_id") == tid).iter_rows(named=True), key=lambda r: abs(r["t"] - t))
            if not sub:
                continue
            map_dot(sub[0]["u"], sub[0]["v"], ROLE_COLOR.get(role, (250, 136, 50)))

        for label, o in cor.items():                     # added / missing players' court dots
            if label in covered or label in hide or o["spurious"] or not o["uv"]:
                continue
            map_dot(o["uv"][0], o["uv"][1], LABEL_COLOR.get(label, (250, 136, 50)))

    return img.resize((OUT_W, int(OUT_W * H / W)))


# __NMAX__ / __FRAMES__ / __VIDEO_ID__ are substituted by str.replace so the JS can use {} freely.
_PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Detektion prüfen</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0a0a0b;color:#eee;font-family:Inter,system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #222}
h1{font-size:15px;margin:0;color:#FA8832;font-weight:700}
.meta{font-size:13px;color:#aaa}
.legend{font-size:12px;color:#bbb;display:flex;gap:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
#wrap{padding:12px 16px;display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
#imgcol{flex:1;min-width:0}
#stage{position:relative;display:block;width:100%;aspect-ratio:var(--ar,16 / 9);margin:0 auto}
img#frame{width:100%;height:100%;display:block;border-radius:8px;background:#000}
#overlay{position:absolute;inset:0;cursor:crosshair}
.ctl{display:flex;gap:12px;align-items:center;margin:10px 0;flex-wrap:wrap}
input[type=range]{flex:1;min-width:240px;accent-color:#FA8832}
button{background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:14px}
button:hover{border-color:#FA8832}
button.primary{background:#FA8832;color:#1A0500;border-color:#FA8832;font-weight:700}
label{font-size:13px;color:#ccc;cursor:pointer}
input[type=number]{width:70px;background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:6px;padding:4px}
textarea{background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:6px;padding:5px 8px;font-size:13px;width:100%;resize:vertical;min-height:48px}

/* draggable player markers on the frame */
.marker{position:absolute;width:28px;height:28px;margin:-14px 0 0 -14px;border-radius:50%;
  border:2px solid #fff;box-shadow:0 0 0 2px rgba(0,0,0,.55);cursor:grab;touch-action:none;
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#1A0500;user-select:none}
.marker:active{cursor:grabbing}
/* corrected marker: subtle white ring glow */
.marker.corrected{box-shadow:0 0 0 3px rgba(255,255,255,0.9),0 0 10px 2px rgba(255,255,255,0.25)}
/* "kein Spieler" marker: dimmed + struck */
.marker.spurious{opacity:.45;box-shadow:none;filter:grayscale(.6);text-decoration:line-through}
.mmarker.spurious{opacity:.4;filter:grayscale(.6)}
/* ghost markers showing the original (pre-correction) position */
.orig-dot{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
  border:2px dashed rgba(255,255,255,0.45);background:transparent;pointer-events:none}
.orig-box{position:absolute;border:2px dashed rgba(255,255,255,0.42);pointer-events:none;border-radius:2px}
/* draggable rings over the court-map dots (Vogelperspektive) */
.mmarker{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;
  border:3px solid #fff;background:transparent;cursor:grab;touch-action:none;box-shadow:0 0 0 1px rgba(0,0,0,.6)}
.mmarker:active{cursor:grabbing}
/* editable person boxes (Rahmen bearbeiten) */
.bbox{position:absolute;border:2px solid #FA8832;background:rgba(250,136,50,.10);cursor:move;touch-action:none}
.bbox.drawing{border-style:dashed;border-color:#fff;background:rgba(255,255,255,.06);cursor:crosshair}
.bbox .blabel{position:absolute;top:-17px;left:-2px;font-size:11px;font-weight:700;color:#1A0500;
  background:#FA8832;padding:0 4px;border-radius:3px;white-space:nowrap}
.bhandle{position:absolute;width:13px;height:13px;background:#fff;border:1px solid #000;border-radius:2px;touch-action:none}
.bhandle.nw{left:-7px;top:-7px;cursor:nwse-resize}
.bhandle.ne{right:-7px;top:-7px;cursor:nesw-resize}
.bhandle.sw{left:-7px;bottom:-7px;cursor:nesw-resize}
.bhandle.se{right:-7px;bottom:-7px;cursor:nwse-resize}
/* per-player show/hide chips */
.rchip{font-size:12px;font-weight:700;color:#1A0500;background:var(--rc,#FA8832);
  border:2px solid var(--rc,#FA8832);border-radius:12px;padding:2px 11px;cursor:pointer}
.rchip.off{background:transparent;color:#777;border-color:#444;text-decoration:line-through}
.hint{position:absolute;left:8px;top:8px;background:rgba(10,10,12,.78);color:#ccc;font-size:12px;
  padding:4px 8px;border-radius:6px;pointer-events:none;max-width:60%}

/* floating picker popup */
.popup{position:absolute;z-index:20;background:#15151a;border:1px solid #FA8832;border-radius:8px;
  padding:10px;width:210px;box-shadow:0 6px 24px rgba(0,0,0,.6);font-size:13px}
.popup h3{margin:0 0 6px;font-size:12px;color:#FA8832;font-weight:700}
.popup .roles{display:flex;gap:4px;margin-bottom:6px}
.popup .roles button{flex:1;padding:6px 0}
.popup .bottom{display:flex;gap:6px;margin-top:4px}
.popup .bottom button{flex:1}

/* toast */
#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#FA8832;color:#1A0500;
  font-weight:700;font-size:13px;padding:8px 16px;border-radius:8px;opacity:0;transition:opacity .2s;pointer-events:none}
#toast.show{opacity:1}

/* correction sidebar */
#cpanel{width:280px;min-width:240px;background:#111;border:1px solid #222;border-radius:10px;padding:12px;flex-shrink:0}
#cpanel h2{margin:0 0 4px;font-size:14px;color:#FA8832;font-weight:700}
#help{font-size:12px;color:#888;margin:0 0 10px;line-height:1.5}
.c-item{background:#1a1a1c;border:1px solid #2a2a2c;border-radius:6px;padding:8px 10px;margin:4px 0;font-size:13px;display:flex;gap:8px;align-items:flex-start}
.c-item .c-role{font-weight:700;min-width:26px}
.c-item .c-type{color:#aaa}
.c-item .c-note{color:#ccc;margin-top:3px}
.c-item .c-pos{color:#555;font-size:11px;margin-top:2px}
.c-del{background:none;border:none;color:#555;cursor:pointer;font-size:18px;padding:0;line-height:1;flex-shrink:0}
.c-del:hover{color:#e74c3c}
.c-empty{color:#555;font-size:13px;margin:6px 0}
#totalinfo{font-size:12px;color:#555;margin:8px 0 4px}
.flagbar{display:flex;flex-wrap:wrap;gap:4px}
.flagbar a{font-size:11px;color:#FA8832;text-decoration:none;background:#1a1a1c;border:1px solid #333;border-radius:4px;padding:1px 5px}
.flagbar a:hover{border-color:#FA8832}
</style></head><body>
<header>
  <h1>Detektion prüfen</h1>
  <span class="meta" id="meta"></span>
  <span class="legend">
    <span><span class="dot" style="background:#FA8832"></span>Nah (A1/A2)</span>
    <span><span class="dot" style="background:#1758A6"></span>Fern (B1/B2)</span>
    <span><span class="dot" style="background:#aaa"></span>erkannt, verworfen</span>
  </span>
</header>
<div id="wrap">
  <div id="imgcol">
    <div class="ctl">
      <button onclick="step(-1)">&#9664; Zurück</button>
      <input id="sl" type="range" min="0" max="__NMAX__" value="0" oninput="go(+this.value)">
      <button onclick="step(1)">Weiter &#9654;</button>
      <label>Frame <input id="num" type="number" min="0" max="__NMAX__" value="0" onchange="go(+this.value)"></label>
    </div>
    <div class="ctl">
      <label><input type="checkbox" id="all" checked onchange="render()"> Alle Personen</label>
      <label><input type="checkbox" id="court" checked onchange="render()"> Spielfeld</label>
      <label><input type="checkbox" id="map" checked onchange="render()"> Vogelperspektive</label>
      <label><input type="checkbox" id="boxedit" onchange="render()"> Rahmen bearbeiten</label>
    </div>
    <div class="ctl" id="rolebar">
      <span style="font-size:13px;color:#888">Spieler anzeigen:</span>
    </div>
    <div id="stage">
      <img id="frame" src="">
      <div id="overlay"></div>
      <div class="hint">Punkt/Ring ziehen = Position (Bild &amp; Karte synchron) · freie Fläche klicken = fehlenden Spieler setzen · „Rahmen bearbeiten" = Box verschieben/größer ziehen oder neue Box aufziehen</div>
    </div>
  </div>

  <div id="cpanel">
    <h2>Korrekturen</h2>
    <p id="help">Stimmt eine Position nicht, <b>zieh</b> den farbigen Punkt (Bild) oder Ring (Karte) — beide bewegen sich
       synchron mit. Fehlt ein Spieler, <b>klick</b> auf den Boden und wähle ihn aus. Für die Box: oben
       „Rahmen bearbeiten" anhaken, dann verschieben/an den Ecken ziehen oder eine neue Box aufziehen.</p>
    <div id="clist"></div>
    <div id="totalinfo"></div>
    <div id="flagbar" class="flagbar"></div>
    <button onclick="downloadJSON()" style="width:100%;margin-top:12px">JSON herunterladen</button>
  </div>
</div>
<div id="toast"></div>

<script>
const FRAMES = __FRAMES__;
const VIDEO_ID = "__VIDEO_ID__";
const NMAX = __NMAX__;
// court homography (px<->uv) + image size, so a drag in one view projects into the other
const H_UV2PX = __H_UV2PX__, H_PX2UV = __H_PX2UV__, IMG_W = __IMG_W__, IMG_H = __IMG_H__;
const ROLE_COLORS = {A1:'#FA8832', A2:'#FFB866', B1:'#1758A6', B2:'#609CDE'};
const TYPE_LABELS = {position:'Position', missing:'Fehlend', bbox:'Rahmen geändert', added:'Rahmen hinzugefügt',
  wrong_role:'Falsche Rolle', spurious:'Kein Spieler', other:'Sonstiges',
  wrong_bbox:'Position falsch', court_pos:'Court-Position'};   // last two: legacy labels
const ALL_ROLES = ['A1','A2','B1','B2'];

let i = 0;
let dets = [];            // kept detections for the current frame
let mapRect = null;       // mini-map rectangle in normalised image coords
let corrections = {};     // {frameIdx: [correction, ...]}
let els = {dot:{}, ring:{}, box:{}};  // live overlay elements per role, for live sync
let suppressClick = false;            // swallow the click that ends a box-draw drag
let hidden = new Set();               // roles whose overlay + baked box are hidden (declutter overlaps)
let corrVersion = 0;                  // bumped on every save/delete → busts the <img> cache so the re-baked frame loads
const $ = id => document.getElementById(id);
const round4 = x => Math.round(x * 10000) / 10000;
const clampN = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

// project a point through a 3x3 row-major homography (h8 = 1)
function applyH(H, x, y) {
  const d = H[6]*x + H[7]*y + H[8];
  return [(H[0]*x + H[1]*y + H[2]) / d, (H[3]*x + H[4]*y + H[5]) / d];
}
const normToUV = (nx, ny) => applyH(H_PX2UV, nx * IMG_W, ny * IMG_H);       // image (0..1) -> court (u,v)
const uvToNorm = (u, v) => { const [px, py] = applyH(H_UV2PX, u, v); return [px / IMG_W, py / IMG_H]; };
function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; }

// ── navigation ───────────────────────────────────────────────────────────────

function frameSrc() {
  const hide = [...hidden].sort().join(',');
  return `/frame?idx=${i}&all=${$('all').checked?1:0}&court=${$('court').checked?1:0}&map=${$('map').checked?1:0}&hide=${hide}&v=${corrVersion}`;
}

async function render() {
  closePopup();
  $('frame').src = frameSrc();
  $('sl').value = i; $('num').value = i;
  updateMeta();
  await loadDets();
  buildOverlay();
  renderList();
}

// per-player show/hide chips — hiding a player drops its box/dot/ring AND its baked box
function buildRoleBar() {
  const bar = $('rolebar');
  bar.querySelectorAll('.rchip').forEach(c => c.remove());
  for (const role of ALL_ROLES) {
    const c = document.createElement('button');
    c.className = 'rchip' + (hidden.has(role) ? ' off' : '');
    c.textContent = role;
    c.style.setProperty('--rc', ROLE_COLORS[role]);
    c.title = (hidden.has(role) ? 'einblenden' : 'ausblenden') + ` (${role})`;
    c.onclick = () => { hidden.has(role) ? hidden.delete(role) : hidden.add(role); buildRoleBar(); render(); };
    bar.appendChild(c);
  }
}

function updateMeta() {
  const f = FRAMES[i];
  const flag = corrections[i] && corrections[i].length ? ` · ⚑ ${corrections[i].length} Korrektur(en)` : '';
  $('meta').textContent = `Frame ${i}/${NMAX} · ${f.t.toFixed(1)} s · ${f.n} erkannt${flag}`;
}

function go(n) { i = Math.max(0, Math.min(NMAX, n)); render(); }
function step(d) { go(i + d); }

document.addEventListener('keydown', e => {
  if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
    e.preventDefault();
    const list = corrections[i] || [];
    if (list.length) deleteCorrection(i, list.length - 1);
    return;
  }
  if (e.key === 'ArrowRight') step(1);
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'Escape') closePopup();
});

// ── overlay: draggable markers + click-to-add ──────────────────────────────────

async function loadDets() {
  const d = await (await fetch('/dets?idx=' + i)).json();
  dets = d.players;
  mapRect = d.map;
}

function ovPos(e) {
  const r = $('overlay').getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

// collapse this frame's corrections to one effective override per label (mirrors the server)
function resolveCors() {
  const out = {};
  for (const c of (corrections[i] || [])) {
    const o = out[c.role] || (out[c.role] = {box:null, foot:null, uv:null, label:null, spurious:false, origFoot:null, origBox:null});
    if (c.type === 'spurious') o.spurious = true;
    if (c.correct_role) o.label = c.correct_role;
    if (c.bbox) o.box = c.bbox;
    if (c.foot_norm) o.foot = c.foot_norm;
    if (c.court_uv) o.uv = c.court_uv;
    if (c.orig_foot) o.origFoot = c.orig_foot;
    if (c.orig_bbox) o.origBox = c.orig_bbox;
  }
  return out;
}

// subtle dashed rectangle marking a box's former position
function addGhostBox(box, color) {
  const g = document.createElement('div');
  g.className = 'orig-box';
  g.style.left = (box[0]*100)+'%'; g.style.top = (box[1]*100)+'%';
  g.style.width = (box[2]*100)+'%'; g.style.height = (box[3]*100)+'%';
  g.style.borderColor = color ? hexA(color, .42) : 'rgba(255,255,255,.42)';
  $('overlay').appendChild(g);
}

function buildOverlay() {
  const ov = $('overlay');
  ov.innerHTML = '';
  els = {dot: {}, ring: {}, box: {}};
  const boxMode = $('boxedit').checked;
  const mapOn = mapRect && $('map').checked;
  const cors = resolveCors();

  // one row per detection, plus any corrected label with no detection (added / missing players)
  const detByLabel = {};
  for (const d of dets) detByLabel[d.label] = d;
  const rows = dets.map(d => ({label: d.label, det: d, cor: cors[d.label]}));
  for (const label in cors) if (!(label in detByLabel)) rows.push({label, det: null, cor: cors[label]});

  // effective (corrected if present) + original positions per row
  const resolve = ({label, det, cor}) => ({
    label, det, cor,
    color: ROLE_COLORS[label] || '#FA8832',
    foot: (cor && cor.foot) || (det && det.ankle) || null,
    uv:   (cor && cor.uv)   || (det && det.uv)   || null,
    box:  (cor && cor.box)  || (det && det.bbox) || null,
    origFoot: det ? det.ankle : (cor && cor.origFoot) || null,
    origBox:  (cor && cor.origBox) || (det && det.bbox) || null,
    moved: !!(cor && (cor.foot || cor.box)),
  });

  // editable person boxes first, so the dots/handles sit on top
  if (boxMode) {
    for (const row of rows) {
      if (hidden.has(row.label)) continue;
      const r = resolve(row);
      if (!r.box) continue;
      if (r.cor && r.cor.box && r.origBox) addGhostBox(r.origBox, r.color);   // former box
      addEditableBox({x:r.box[0], y:r.box[1], w:r.box[2], h:r.box[3]}, r.label, false);
    }
  }

  for (const row of rows) {
    if (hidden.has(row.label)) continue;
    const r = resolve(row);
    const drag = r.det || {label: r.label, ankle: r.foot, uv: r.uv, bbox: r.box};   // pseudo-det for added/missing

    // ghost dot at the former position when this player was moved
    if (r.cor && r.moved && r.origFoot) {
      const g = document.createElement('div');
      g.className = 'orig-dot'; g.dataset.role = r.label;
      g.style.left = (r.origFoot[0] * 100) + '%';
      g.style.top  = (r.origFoot[1] * 100) + '%';
      g.style.borderColor = r.color;
      ov.appendChild(g);
    }

    // video foot dot — drag corrects the position; the map ring follows live
    if (r.foot) {
      const m = document.createElement('div');
      m.className = 'marker' + (r.cor ? ' corrected' : '') + (r.cor && r.cor.spurious ? ' spurious' : '');
      m.style.left = (r.foot[0] * 100) + '%';
      m.style.top  = (r.foot[1] * 100) + '%';
      m.style.background = r.color;
      m.textContent = (r.cor && r.cor.label) || r.label;
      m.title = r.label + ' — ziehen: Position (Bild + Karte synchron)';
      attachPosDrag(m, drag);
      ov.appendChild(m);
      els.dot[r.label] = m;
    }

    // matching ring on the court map — drag corrects the court position; the video dot follows live
    if (mapOn && r.uv) {
      const m = document.createElement('div');
      m.className = 'mmarker' + (r.cor && r.cor.spurious ? ' spurious' : '');
      m.style.left = ((mapRect.x + r.uv[0] * mapRect.w) * 100) + '%';
      m.style.top  = ((mapRect.y + r.uv[1] * mapRect.h) * 100) + '%';
      m.style.borderColor = r.color;
      m.title = r.label + ' — ziehen: Court-Position (Karte + Bild synchron)';
      attachRingDrag(m, drag);
      ov.appendChild(m);
      els.ring[r.label] = m;
    }
  }
}

// move a role's two markers together (live, while dragging)
function syncRing(role, u, v) {
  const ring = els.ring[role];
  if (ring && mapRect) {
    ring.style.left = ((mapRect.x + clampN(u) * mapRect.w) * 100) + '%';
    ring.style.top  = ((mapRect.y + clampN(v) * mapRect.h) * 100) + '%';
  }
}
function syncDot(role, nx, ny) {
  const dot = els.dot[role];
  if (dot) { dot.style.left = (nx * 100) + '%'; dot.style.top = (ny * 100) + '%'; }
}

// video foot dot: drag in image space, project to court, move both markers, save one 'position' correction
function attachPosDrag(el, det) {
  let sx, sy, moved, dragging = false;
  el.addEventListener('pointerdown', e => {
    e.preventDefault(); e.stopPropagation();
    try { el.setPointerCapture(e.pointerId); } catch (_) {}
    sx = e.clientX; sy = e.clientY; moved = false; dragging = true;
    closePopup();
  });
  el.addEventListener('pointermove', e => {
    if (!dragging) return;
    if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 4) moved = true;
    if (!moved) return;
    const [nx, ny] = ovPos(e);
    el.style.left = (nx * 100) + '%'; el.style.top = (ny * 100) + '%';
    const [u, v] = normToUV(nx, ny);
    syncRing(det.label, u, v);
  });
  el.addEventListener('pointerup', e => {
    if (!dragging) return;
    dragging = false;
    try { el.releasePointerCapture(e.pointerId); } catch (_) {}
    const [nx, ny] = ovPos(e);
    if (moved) {
      const [u, v] = normToUV(nx, ny);
      save({role: det.label, type: 'position', edited_in: 'video',
        foot_norm: [round4(nx), round4(ny)], court_uv: [round4(clampN(u)), round4(clampN(v))],
        orig_foot: det.ankle, orig_uv: det.uv});
      toast(`${det.label}: Position korrigiert (Bild + Karte synchron)`);
    } else {
      markerMenu(det, nx, ny);
    }
  });
}

// court-map ring: drag in court space, project to image, move both markers, save one 'position' correction
function attachRingDrag(el, det) {
  let moved, dragging = false;
  el.addEventListener('pointerdown', e => {
    e.preventDefault(); e.stopPropagation();
    try { el.setPointerCapture(e.pointerId); } catch (_) {}
    moved = false; dragging = true;
    closePopup();
  });
  el.addEventListener('pointermove', e => {
    if (!dragging) return;
    moved = true;
    const [nx, ny] = ovPos(e);
    const u = clampN((nx - mapRect.x) / mapRect.w), v = clampN((ny - mapRect.y) / mapRect.h);
    el.style.left = ((mapRect.x + u * mapRect.w) * 100) + '%';
    el.style.top  = ((mapRect.y + v * mapRect.h) * 100) + '%';
    const [fnx, fny] = uvToNorm(u, v);
    syncDot(det.label, fnx, fny);
  });
  el.addEventListener('pointerup', e => {
    if (!dragging) return;
    dragging = false;
    try { el.releasePointerCapture(e.pointerId); } catch (_) {}
    if (!moved) return;
    const [nx, ny] = ovPos(e);
    const u = clampN((nx - mapRect.x) / mapRect.w), v = clampN((ny - mapRect.y) / mapRect.h);
    const [fnx, fny] = uvToNorm(u, v);
    save({role: det.label, type: 'position', edited_in: 'map',
      court_uv: [round4(u), round4(v)], foot_norm: [round4(fnx), round4(fny)],
      orig_uv: det.uv, orig_foot: det.ankle});
    toast(`${det.label}: Position korrigiert (Karte + Bild synchron)`);
  });
}

// an editable rectangle: drag the body to move, drag a corner to resize; commits on release
function addEditableBox(box, label, isNew) {
  const ov = $('overlay');
  const color = ROLE_COLORS[label] || '#FA8832';
  const el = document.createElement('div');
  el.className = 'bbox';
  el.style.borderColor = color;
  el.style.background = hexA(color, 0.10);
  const place = () => {
    el.style.left = (box.x * 100) + '%'; el.style.top = (box.y * 100) + '%';
    el.style.width = (box.w * 100) + '%'; el.style.height = (box.h * 100) + '%';
  };
  place();
  const lab = document.createElement('div');
  lab.className = 'blabel'; lab.textContent = label; lab.style.background = color;
  el.appendChild(lab);

  el.addEventListener('pointerdown', e => {                 // move
    if (e.target.classList.contains('bhandle')) return;
    e.preventDefault(); e.stopPropagation();
    try { el.setPointerCapture(e.pointerId); } catch (_) {}
    const r = ov.getBoundingClientRect();
    const sx = e.clientX, sy = e.clientY, ox = box.x, oy = box.y; let moved = false;
    const mv = ev => {
      if (Math.abs(ev.clientX - sx) + Math.abs(ev.clientY - sy) > 3) moved = true;
      box.x = clampN(ox + (ev.clientX - sx) / r.width, 0, 1 - box.w);
      box.y = clampN(oy + (ev.clientY - sy) / r.height, 0, 1 - box.h);
      place();
    };
    const up = ev => {
      el.removeEventListener('pointermove', mv); el.removeEventListener('pointerup', up);
      try { el.releasePointerCapture(ev.pointerId); } catch (_) {}
      if (moved) commitBox(box, label, isNew);
    };
    el.addEventListener('pointermove', mv); el.addEventListener('pointerup', up);
  });

  for (const corner of ['nw','ne','sw','se']) {              // resize
    const h = document.createElement('div');
    h.className = 'bhandle ' + corner;
    h.addEventListener('pointerdown', e => {
      e.preventDefault(); e.stopPropagation();
      try { h.setPointerCapture(e.pointerId); } catch (_) {}
      const r = ov.getBoundingClientRect();
      const sx = e.clientX, sy = e.clientY, o = {x:box.x, y:box.y, w:box.w, h:box.h};
      const mv = ev => {
        const dx = (ev.clientX - sx) / r.width, dy = (ev.clientY - sy) / r.height;
        let x = o.x, y = o.y, w = o.w, hh = o.h;
        if (corner.includes('w')) { x = o.x + dx; w = o.w - dx; }
        if (corner.includes('e')) { w = o.w + dx; }
        if (corner.includes('n')) { y = o.y + dy; hh = o.h - dy; }
        if (corner.includes('s')) { hh = o.h + dy; }
        if (w > 0.01) { box.x = x; box.w = w; }
        if (hh > 0.01) { box.y = y; box.h = hh; }
        place();
      };
      const up = ev => {
        h.removeEventListener('pointermove', mv); h.removeEventListener('pointerup', up);
        try { h.releasePointerCapture(ev.pointerId); } catch (_) {}
        commitBox(box, label, isNew);
      };
      h.addEventListener('pointermove', mv); h.addEventListener('pointerup', up);
    });
    el.appendChild(h);
  }
  ov.appendChild(el);
  els.box[label] = el;
  return el;
}

function commitBox(box, label, isNew) {
  const b = [round4(box.x), round4(box.y), round4(box.w), round4(box.h)];
  const foot = [round4(box.x + box.w / 2), round4(box.y + box.h)];   // foot = box bottom-centre
  const [u, v] = normToUV(foot[0], foot[1]);
  const court = [round4(clampN(u)), round4(clampN(v))];
  if (isNew) {
    save({role: label, type: 'added', bbox: b, foot_norm: foot, court_uv: court});
  } else {
    const orig = dets.find(d => d.label === label);
    save({role: label, type: 'bbox', bbox: b, orig_bbox: orig ? orig.bbox : null, foot_norm: foot, court_uv: court});
  }
  toast(`${label}: Rahmen ${isNew ? 'hinzugefügt' : 'geändert'}`);
}

// draw a brand-new box by dragging on empty ground (only while "Rahmen bearbeiten" is on)
$('overlay').addEventListener('pointerdown', e => {
  if (!$('boxedit').checked || e.target.id !== 'overlay') return;
  e.preventDefault();
  const ov = $('overlay'), r = ov.getBoundingClientRect();
  const sxn = (e.clientX - r.left) / r.width, syn = (e.clientY - r.top) / r.height;
  let box = null;
  const tmp = document.createElement('div'); tmp.className = 'bbox drawing'; ov.appendChild(tmp);
  try { ov.setPointerCapture(e.pointerId); } catch (_) {}
  const mv = ev => {
    const x2 = (ev.clientX - r.left) / r.width, y2 = (ev.clientY - r.top) / r.height;
    box = {x: Math.min(sxn, x2), y: Math.min(syn, y2), w: Math.abs(x2 - sxn), h: Math.abs(y2 - syn)};
    tmp.style.left = (box.x*100)+'%'; tmp.style.top = (box.y*100)+'%';
    tmp.style.width = (box.w*100)+'%'; tmp.style.height = (box.h*100)+'%';
  };
  const up = ev => {
    ov.removeEventListener('pointermove', mv); ov.removeEventListener('pointerup', up);
    try { ov.releasePointerCapture(ev.pointerId); } catch (_) {}
    tmp.remove();
    if (box && box.w > 0.01 && box.h > 0.01) { suppressClick = true; newBoxPicker(box); }
  };
  ov.addEventListener('pointermove', mv); ov.addEventListener('pointerup', up);
});

$('overlay').addEventListener('click', e => {
  if (suppressClick) { suppressClick = false; return; }   // this click ended a box-draw
  if (e.target.id !== 'overlay') return;                  // markers/boxes handle their own clicks
  const [nx, ny] = ovPos(e);
  rolePicker(nx, ny);
});

// ── popups ─────────────────────────────────────────────────────────────────────

function closePopup() { document.querySelectorAll('.popup').forEach(p => p.remove()); }

function popupAt(nx, ny) {
  closePopup();
  const p = document.createElement('div');
  p.className = 'popup';
  p.style.left = Math.min(nx * 100, 70) + '%';
  p.style.top  = Math.min(ny * 100, 80) + '%';
  p.addEventListener('click', e => e.stopPropagation());
  $('overlay').appendChild(p);
  return p;
}

// click on empty ground → which player belongs here?
function rolePicker(nx, ny) {
  const foot = [round4(nx), round4(ny)];
  const [u, v] = normToUV(nx, ny);
  const court = [round4(clampN(u)), round4(clampN(v))];
  const p = popupAt(nx, ny);
  p.innerHTML = `<h3>Welcher Spieler steht hier?</h3><div class="roles"></div>
    <textarea id="pnote" placeholder="Notiz (optional)"></textarea>
    <div class="bottom"><button onclick="closePopup()">Abbrechen</button></div>`;
  const row = p.querySelector('.roles');
  for (const role of ALL_ROLES) {
    const b = document.createElement('button');
    b.textContent = role;
    b.style.color = ROLE_COLORS[role];
    b.onclick = () => {
      // detection for this role exists elsewhere → it's mislocated; otherwise the player was missed entirely
      const exists = dets.find(d => d.label === role);
      save({
        role, type: exists ? 'position' : 'missing', edited_in: 'video',
        foot_norm: foot, court_uv: court,
        orig_foot: exists ? exists.ankle : null, orig_uv: exists ? exists.uv : null,
        note: $('pnote').value.trim()
      });
      toast(`${role}: ${exists ? 'Position korrigiert' : 'als fehlend gesetzt'}`);
    };
    row.appendChild(b);
  }
}

// click on an existing dot → wrong role or not-a-player
function markerMenu(det, nx, ny) {
  const p = popupAt(nx, ny);
  p.innerHTML = `<h3>${det.label} stimmt nicht?</h3>
    <button style="width:100%;margin-bottom:6px" onclick="closePopup()" id="spurBtn">Kein Spieler (Zuschauer/Fehler)</button>
    <div style="font-size:11px;color:#888;margin-bottom:4px">…oder das ist eigentlich:</div>
    <div class="roles"></div>
    <div class="bottom"><button onclick="closePopup()">Abbrechen</button></div>`;
  p.querySelector('#spurBtn').onclick = () => {
    save({role: det.label, type: 'spurious', foot_norm: det.ankle, court_uv: det.uv});
    toast(`${det.label}: als „kein Spieler" markiert`);
  };
  const row = p.querySelector('.roles');
  for (const role of ALL_ROLES.filter(r => r !== det.label)) {
    const b = document.createElement('button');
    b.textContent = role;
    b.style.color = ROLE_COLORS[role];
    b.onclick = () => {
      save({role: det.label, type: 'wrong_role', correct_role: role, foot_norm: det.ankle, court_uv: det.uv});
      toast(`${det.label} → ${role}`);
    };
    row.appendChild(b);
  }
}

// after drawing a new box → which player is it?
function newBoxPicker(box) {
  const p = popupAt(clampN(box.x + box.w, 0, 0.7), clampN(box.y, 0, 0.8));
  p.innerHTML = `<h3>Neuer Rahmen — welcher Spieler?</h3><div class="roles"></div>
    <div class="bottom"><button onclick="closePopup()">Abbrechen</button></div>`;
  const row = p.querySelector('.roles');
  for (const role of ALL_ROLES) {
    const b = document.createElement('button');
    b.textContent = role;
    b.style.color = ROLE_COLORS[role];
    b.onclick = () => { closePopup(); addEditableBox(box, role, true); commitBox(box, role, true); };
    row.appendChild(b);
  }
}

// ── persistence ─────────────────────────────────────────────────────────────────

async function save(c) {
  c.frame_idx = i;
  c.t_s = FRAMES[i].t;
  const resp = await fetch('/correct', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(c)
  });
  c._id = (await resp.json()).id;
  const arr = corrections[i] = corrections[i] || [];
  // one correction per (role, type) per frame — re-adjusting replaces, never piles up (server does the same)
  for (let k = arr.length - 1; k >= 0; k--) if (arr[k].role === c.role && arr[k].type === c.type) arr.splice(k, 1);
  arr.push(c);
  closePopup();
  corrVersion++;                 // re-bake: the bold box/dot/label + map-dot move to the corrected spot
  $('frame').src = frameSrc();
  updateMeta();
  renderList();
  renderFlagBar();
  buildOverlay();                // corrections-aware → marker stays at the corrected position, glow + ghost applied
}

async function deleteCorrection(fidx, cidx) {
  await fetch(`/correct?frame=${fidx}&idx=${cidx}`, {method: 'DELETE'});
  corrections[fidx].splice(cidx, 1);
  if (!corrections[fidx].length) delete corrections[fidx];
  corrVersion++;
  updateMeta();
  renderList();
  renderFlagBar();
  if (fidx === i) { $('frame').src = frameSrc(); buildOverlay(); }   // restore the original bake + overlay
}

// ── sidebar rendering ────────────────────────────────────────────────────────────

function renderList() {
  const list = corrections[i] || [];
  $('clist').innerHTML = list.length === 0
    ? '<p class="c-empty">Keine Korrekturen für diesen Frame.</p>'
    : list.map((c, idx) => `
      <div class="c-item">
        <div style="flex:1">
          <span class="c-role" style="color:${ROLE_COLORS[c.role] || '#FA8832'}">${c.role}</span>
          <span class="c-type">${TYPE_LABELS[c.type] || c.type}${c.correct_role ? ' → ' + c.correct_role : ''}</span>
          ${c.note ? `<div class="c-note">${c.note}</div>` : ''}
          ${detailLine(c)}
        </div>
        <button class="c-del" onclick="deleteCorrection(${i},${idx})" title="Löschen">×</button>
      </div>`).join('');
}

function detailLine(c) {
  const parts = [];
  if (c.bbox) parts.push(`Rahmen ${(c.bbox[2]*100).toFixed(0)}×${(c.bbox[3]*100).toFixed(0)}%`);
  if (c.court_uv) parts.push(`Court ${c.court_uv[0].toFixed(2)}/${c.court_uv[1].toFixed(2)}`);
  else if (c.pos_norm) parts.push(`Bild ${Math.round(c.pos_norm[0]*100)},${Math.round(c.pos_norm[1]*100)}%`);
  return parts.length ? `<div class="c-pos">${parts.join(' · ')}</div>` : '';
}

function renderFlagBar() {
  const keys = Object.keys(corrections).map(Number).sort((a, b) => a - b);
  const total = keys.reduce((s, k) => s + corrections[k].length, 0);
  $('totalinfo').textContent = total ? `${total} Korrekturen in ${keys.length} Frames` : '';
  $('flagbar').innerHTML = keys.map(k =>
    `<a href="#" onclick="go(${k});return false;" title="${corrections[k].length}x">${k}</a>`).join('');
}

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 1600);
}

async function downloadJSON() {
  const text = await (await fetch('/corrections')).text();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
  a.download = `corrections_${VIDEO_ID.slice(0, 8)}.json`;
  a.click();
}

// ── init ───────────────────────────────────────────────────────────────────────

// keep the frame as large as the window allows, locked to the video's real aspect ratio:
// fill the column width (grows past 1280 on big monitors), but never taller than the viewport.
function fitStage() {
  const stage = $('stage');
  stage.style.maxWidth = '';                            // measure the uncapped top edge first
  const avail = window.innerHeight - stage.getBoundingClientRect().top - 16;   // room below the frame
  stage.style.maxWidth = Math.max(360, avail * (IMG_W / IMG_H)) + 'px';
}
window.addEventListener('resize', fitStage);

$('frame').addEventListener('load', buildOverlay);   // re-anchor markers once the frame's size is known
document.documentElement.style.setProperty('--ar', IMG_W + ' / ' + IMG_H);   // lock stage to video aspect
buildRoleBar();
fitStage();
(async () => {
  const data = await (await fetch('/corrections')).json();
  for (const c of data.corrections || []) (corrections[c.frame_idx] = corrections[c.frame_idx] || []).push(c);
  renderFlagBar();
  render();
})();
</script>
</body></html>"""


def _serve(data: DebugData, video_id: str, port: int) -> None:
    counts = data.det.group_by("frame").len().to_dict(as_series=False)
    by_frame = dict(zip(counts["frame"], counts["len"]))
    frames_meta = json.dumps([{"t": round(f["t"], 2), "n": int(by_frame.get(f["frame"], 0))} for f in data.frames])
    nmax = len(data.frames) - 1

    page = (
        _PAGE
        .replace("__NMAX__", str(nmax))
        .replace("__FRAMES__", frames_meta)
        .replace("__VIDEO_ID__", video_id)
        .replace("__H_UV2PX__", json.dumps(data.h_uv2px))
        .replace("__H_PX2UV__", json.dumps(solve_homography(data.corners, UNIT_SQUARE)))
        .replace("__IMG_W__", str(data.width))
        .replace("__IMG_H__", str(data.height))
    )

    corrections_path = data.cache / "corrections.json"
    corrections: dict[int, list[dict]] = {}   # corrections[frame_idx] = [correction, ...]
    _id_counter = [0]

    def _load():
        if corrections_path.exists():
            for c in json.loads(corrections_path.read_text()).get("corrections", []):
                c.setdefault("_id", _id_counter[0])
                _id_counter[0] += 1
                corrections.setdefault(int(c["frame_idx"]), []).append(c)

    def _flat() -> list[dict]:
        return sorted((c for lst in corrections.values() for c in lst), key=lambda c: c["frame_idx"])

    def _save():
        corrections_path.write_text(
            json.dumps({"video_id": video_id, "corrections": _flat()}, indent=2, ensure_ascii=False)
        )

    _load()

    def _cor_sig(idx: int) -> str:
        """Cache key for the frame's correction state. Every save/delete assigns a fresh _id,
        so the set of _ids on a frame changes exactly when its corrections do."""
        return ",".join(str(c["_id"]) for c in corrections.get(idx, []))

    @lru_cache(maxsize=512)
    def render_jpeg(idx: int, show_all: bool, show_court: bool, show_map: bool, hide_key: str = "", cor_sig: str = "") -> bytes:
        hide = tuple(h for h in hide_key.split(",") if h)
        buf = io.BytesIO()
        annotate(data, idx, show_all=show_all, show_court=show_court, show_map=show_map,
                 hide=hide, cors=corrections.get(idx, [])).save(buf, "JPEG", quality=85)
        return buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send_json(self, body, status: int = 200) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_bytes(self, raw: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send_bytes(page.encode(), "text/html; charset=utf-8")
                return
            if u.path == "/frame":
                q = parse_qs(u.query)
                idx = max(0, min(nmax, int(q.get("idx", ["0"])[0])))
                hide_key = ",".join(sorted(h for h in q.get("hide", [""])[0].split(",") if h))
                png = render_jpeg(idx, q.get("all", ["1"])[0] == "1", q.get("court", ["1"])[0] == "1",
                                  q.get("map", ["1"])[0] == "1", hide_key, _cor_sig(idx))
                self._send_bytes(png, "image/jpeg")
                return
            if u.path == "/dets":
                idx = max(0, min(nmax, int(parse_qs(u.query).get("idx", ["0"])[0])))
                self._send_json({"map": map_rect(data), "players": kept_detections(data, idx)})
                return
            if u.path == "/corrections":
                self._send_json({"video_id": video_id, "corrections": _flat()})
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path == "/correct":
                c = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                fidx = int(c["frame_idx"])
                lst = corrections.setdefault(fidx, [])
                # one correction per (role, type) per frame — re-adjusting replaces the previous one
                lst[:] = [x for x in lst if not (x.get("role") == c.get("role") and x.get("type") == c.get("type"))]
                c["_id"] = _id_counter[0]
                _id_counter[0] += 1
                lst.append(c)
                _save()
                self._send_json({"ok": True, "id": c["_id"]})
                return
            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):
            u = urlparse(self.path)
            if u.path == "/correct":
                q = parse_qs(u.query)
                fidx, cidx = int(q["frame"][0]), int(q["idx"][0])
                if fidx in corrections and 0 <= cidx < len(corrections[fidx]):
                    corrections[fidx].pop(cidx)
                    if not corrections[fidx]:
                        del corrections[fidx]
                    _save()
                self._send_json({"ok": True})
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    log.info("verify tool for %s — open %s  (Ctrl-C to stop)", video_id, url)
    n_existing = sum(len(v) for v in corrections.values())
    existing_note = f" · {n_existing} bestehende Korrekturen geladen" if n_existing else ""
    print(f"\n  ▶ Detektion prüfen: {url}\n    {len(data.frames)} Frames · Pfeiltasten/Slider zum Blättern{existing_note}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Shuttle labeler (`verify --shuttle`) — a focused sibling of the player tool above.
#
# Same building blocks (ThreadingHTTPServer + HTML/JS canvas + frames cache + Cmd+Z undo + ←/→ nav +
# JSON persistence), but the shuttle is a handful of low-contrast pixels, so it serves the NATIVE 4K
# frame (no 1280 downscale) and adds a hover-loupe for true-pixel clicks. The trainer clicks sparse
# anchors on each arc; the pure shuttle.label.interpolate_anchors fills every frame, and the server
# returns that dense path live so the user adds anchors where the line drifts off the shuttle.
# ════════════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class ShuttleData:
    cache: Path
    frames: list[dict]           # manifest frames [{frame, t}, ...]
    width: int
    height: int
    guide: list | None           # airspace polygon, normalised image coords (faint mask outline) or None
    crop: list | None            # crop box [x0,y0,x1,y1] normalised, or None — both just visual guides


def load_shuttle_data(video_id: str, cfg) -> ShuttleData:
    """Frames manifest + cache for the shuttle labeler. Calibration is OPTIONAL: when present it adds a
    faint airspace/crop guide (the region the tracker actually sees) so the user labels where it matters,
    but labeling works without it."""
    cache = Path(cfg.frame_cache_dir).expanduser() / video_id
    manifest = json.loads((cache / "frames.json").read_text())
    w, h = int(manifest["width"]), int(manifest["height"])
    guide = crop = None
    try:
        cal = Store(cfg.supabase_url, cfg.service_role_key).get_calibration(video_id)
        poly = (cal or {}).get("mask_polygon_px") or (cal or {}).get("confirmed_points")
        if poly:
            from .geometry import court_airspace_polygon, court_crop_box
            from .pipeline.shuttle import AIR_DOWN, AIR_OUT, AIR_SIDE, AIR_UP, BBOX_MARGIN

            pts = [(float(x), float(y)) for x, y in poly]
            air = court_airspace_polygon(pts, w, h, up=AIR_UP, side=AIR_SIDE, down=AIR_DOWN, out=AIR_OUT)
            cb = court_crop_box(air, w, h, up=0.0, side=BBOX_MARGIN, down=0.0)
            guide = [[round(px / w, 5), round(py / h, 5)] for px, py in air]
            crop = [round(cb[0] / w, 5), round(cb[1] / h, 5), round(cb[2] / w, 5), round(cb[3] / h, 5)]
    except Exception as e:  # noqa: BLE001 — the guide is a nicety; never block labeling on calibration
        log.warning("shuttle labeler: no calibration guide (%s)", e)
    return ShuttleData(cache=cache, frames=manifest["frames"], width=w, height=h, guide=guide, crop=crop)


_SHUTTLE_PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Shuttle markieren</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0a0a0b;color:#eee;font-family:Inter,system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #222}
h1{font-size:15px;margin:0;color:#FA8832;font-weight:700}
.meta{font-size:13px;color:#aaa}
.legend{font-size:12px;color:#bbb;display:flex;gap:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
#wrap{padding:12px 16px;display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
#imgcol{flex:1;min-width:0}
#stage{position:relative;display:block;width:100%;aspect-ratio:var(--ar,16 / 9);margin:0 auto}
img#frame{width:100%;height:100%;display:block;border-radius:8px;background:#000}
#ov{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
.ctl{display:flex;gap:12px;align-items:center;margin:10px 0;flex-wrap:wrap}
input[type=range]{flex:1;min-width:240px;accent-color:#FA8832}
button{background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:14px}
button:hover{border-color:#FA8832}
label{font-size:13px;color:#ccc}
input[type=number]{width:80px;background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:6px;padding:4px}
/* loupe: a magnified inset of the NATIVE 4k frame, following the cursor */
#loupe{position:absolute;width:200px;height:200px;border-radius:50%;border:2px solid #FA8832;
  box-shadow:0 4px 18px rgba(0,0,0,.7);pointer-events:none;display:none;z-index:30;
  background-repeat:no-repeat;background-color:#000}
#loupe::after{content:"";position:absolute;left:50%;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;
  border:1px solid rgba(250,136,50,.9);border-radius:50%}
#loupe::before{content:"";position:absolute;left:50%;top:50%;width:1px;height:1px;
  box-shadow:0 0 0 1px #FA8832, 6px 0 0 0 #FA8832,-6px 0 0 0 #FA8832,0 6px 0 0 #FA8832,0 -6px 0 0 #FA8832}
#side{width:280px;min-width:240px;background:#111;border:1px solid #222;border-radius:10px;padding:12px;flex-shrink:0}
#side h2{margin:0 0 4px;font-size:14px;color:#FA8832;font-weight:700}
#help{font-size:12px;color:#888;margin:0 0 10px;line-height:1.55}
#help b{color:#bbb}
.flagbar{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.flagbar a{font-size:11px;color:#FA8832;text-decoration:none;background:#1a1a1c;border:1px solid #333;border-radius:4px;padding:1px 5px}
.flagbar a.bnd{color:#e74c3c;border-color:#5a2a2a}
.flagbar a.occ{color:#888}
#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#FA8832;color:#1A0500;
  font-weight:700;font-size:13px;padding:8px 16px;border-radius:8px;opacity:0;transition:opacity .2s;pointer-events:none}
#toast.show{opacity:1}
svg .air{fill:rgba(250,136,50,.05);stroke:rgba(250,136,50,.45);stroke-width:2;vector-effect:non-scaling-stroke}
svg .crop{fill:none;stroke:rgba(255,255,255,.28);stroke-width:1;stroke-dasharray:6 6;vector-effect:non-scaling-stroke}
svg .fill{fill:#39d98a;opacity:.85}
svg .gap{fill:#e74c3c;opacity:.85}
svg .anc{fill:#FA8832;stroke:#1A0500;stroke-width:2;vector-effect:non-scaling-stroke}
svg .occ{fill:none;stroke:#888;stroke-width:3;vector-effect:non-scaling-stroke}
svg .cur{stroke:#fff;stroke-width:3;vector-effect:non-scaling-stroke}
svg polyline.path{fill:none;stroke:rgba(57,217,138,.35);stroke-width:1;vector-effect:non-scaling-stroke}
#jsonpanel{margin-top:12px}
.jsonbar{display:flex;align-items:center;justify-content:space-between;background:#1a1a1c;border:1px solid #333;border-bottom:none;border-radius:8px 8px 0 0;padding:5px 8px}
.jsonbar .jsonlabel{font-size:11px;color:#888;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.copy{display:inline-flex;align-items:center;justify-content:center;padding:4px 6px;line-height:0;background:transparent;border:1px solid #333;color:#ccc}
.copy:hover{border-color:#FA8832;color:#FA8832}
.jsonview{margin:0;max-height:320px;overflow:auto;background:#0d0d0e;border:1px solid #333;border-radius:0 0 8px 8px;padding:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;line-height:1.45;color:#cfcfcf;white-space:pre;tab-size:2}
</style></head><body>
<header>
  <h1>Shuttle markieren</h1>
  <span class="meta" id="meta"></span>
  <span class="legend">
    <span><span class="dot" style="background:#FA8832"></span>Anker (Klick)</span>
    <span><span class="dot" style="background:#39d98a"></span>interpoliert</span>
    <span><span class="dot" style="background:#e74c3c"></span>Lücke / Grenze</span>
  </span>
</header>
<div id="wrap">
  <div id="imgcol">
    <div class="ctl">
      <button onclick="step(-1)">&#9664; Zurück</button>
      <input id="sl" type="range" min="0" max="__NMAX__" value="0" oninput="go(+this.value)">
      <button onclick="step(1)">Weiter &#9654;</button>
      <label>Frame <input id="num" type="number" min="0" max="__NMAX__" value="0" onchange="go(+this.value)"></label>
      <label>Ansicht
        <select id="view" onchange="setView(this.value)" title="Kontrast/Sichtbarkeit (Taste C blättert)">
          <option value="normal">Normal</option>
          <option value="kontrast">Kontrast</option>
          <option value="abgedunkelt">Abgedunkelt</option>
          <option value="invertiert">Invertiert</option>
          <option value="graustufen">Graustufen</option>
          <option value="lokalkontrast">Lokalkontrast (CLAHE)</option>
          <option value="bewegung">Bewegung (Hintergrund)</option>
        </select>
      </label>
      <label title="Nach dem Setzen eines Ankers automatisch zum nächsten Frame springen">
        <input id="autoadv" type="checkbox" onchange="setAutoAdvance(this.checked)"> Auto-Weiter
      </label>
    </div>
    <div id="stage">
      <img id="frame" src="">
      <svg id="ov" preserveAspectRatio="none" viewBox="0 0 __IMG_W__ __IMG_H__"></svg>
      <div id="loupe"></div>
    </div>
  </div>
  <div id="side">
    <h2>Shuttle-Spur</h2>
    <p id="help"><b>Klick</b> = Shuttle-Punkt (Anker) im aktuellen Frame. <b>Shift-/Rechtsklick</b> = Frame
       als <b>verdeckt</b> markieren (visibility 0). Taste <b>A</b> = <b>Schlag-/Segmentgrenze</b> (es wird
       nie darüber interpoliert). <b>X</b> oder <b>Cmd/Ctrl+Z</b> = Anker im Frame löschen. Die <b>Lupe</b> folgt dem Cursor
       für pixelgenaue Klicks. Setz Anker dort, wo die grüne Linie vom Shuttle abweicht (am Scheitelpunkt dicht).</p>
    <div id="info" style="font-size:12px;color:#aaa"></div>
    <div id="flagbar" class="flagbar"></div>
    <div id="jsonpanel">
      <div class="jsonbar">
        <span class="jsonlabel">shuttle_labels.json</span>
        <button class="copy" onclick="copyJSON()" title="JSON kopieren" aria-label="JSON kopieren">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        </button>
      </div>
      <pre id="json" class="jsonview"></pre>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
const FRAMES = __FRAMES__;
const VIDEO_ID = "__VIDEO_ID__";
const NMAX = __NMAX__;
const IMG_W = __IMG_W__, IMG_H = __IMG_H__;
const GUIDE = __GUIDE__;   // airspace polygon (normalised) or null
const CROP = __CROP__;     // crop box [x0,y0,x1,y1] normalised or null
const ZOOM = 4;            // loupe magnification (screen px per source px)
const $ = id => document.getElementById(id);
const round4 = x => Math.round(x * 10000) / 10000;
const clampN = v => Math.max(0, Math.min(1, v));

let i = 0;
let path = [];                 // dense [[frame,px,py,vis,interp], ...] from the server
let anchors = {};              // frameIdx -> anchor (for the nav strip)
let boundaries = new Set();

// Contrast/visibility views — make the low-contrast white shuttle easier to spot on the white wall.
// CSS-filter presets are applied to the frame AND the loupe (never the SVG markers, so anchors stay
// crisp + on-brand); "bewegung" instead asks the server for the background-difference frame. 'c' cycles.
const VIEWS = ['normal', 'kontrast', 'abgedunkelt', 'invertiert', 'graustufen', 'lokalkontrast', 'bewegung'];
const FILTER = {
  normal: 'none',
  kontrast: 'contrast(1.9) brightness(0.95)',
  abgedunkelt: 'brightness(0.5) contrast(2.4)',
  invertiert: 'invert(1) contrast(1.5)',
  graustufen: 'grayscale(1) contrast(2.2) brightness(0.95)',
  lokalkontrast: 'none',                     // server applies CLAHE; no extra CSS
  bewegung: 'none',                          // server subtracts the background; no extra CSS
};
const SERVER_ENH = { lokalkontrast: 'clahe', bewegung: 'diff' };   // these views are rendered server-side
let view = localStorage.getItem('shuttleView') || 'normal';
if (!VIEWS.includes(view)) view = 'normal';
const serverEnh = () => SERVER_ENH[view] || 'none';

// Auto-advance: after placing an anchor, jump to the next frame (fast labeling). Persisted, default on.
let autoAdvance = localStorage.getItem('shuttleAutoAdvance') !== '0';
function setAutoAdvance(on) { autoAdvance = !!on; localStorage.setItem('shuttleAutoAdvance', on ? '1' : '0'); }

function frameSrc() { return `/frame?idx=${i}&enh=${serverEnh()}`; }   // browser caches per (idx, view)

function applyView() {
  const f = FILTER[view] || 'none';
  $('frame').style.filter = f;
  $('loupe').style.filter = f;             // loupe magnifies the same source → keep it enhanced too
  const sel = $('view'); if (sel) sel.value = view;
}
function setView(v) {
  view = VIEWS.includes(v) ? v : 'normal';
  localStorage.setItem('shuttleView', view);
  applyView();
  $('frame').src = frameSrc();             // re-fetch (server enh may have changed); loupe follows on hover
}
function cycleView() { setView(VIEWS[(VIEWS.indexOf(view) + 1) % VIEWS.length]); }

function ovXY(e) {
  const r = $('ov').getBoundingClientRect();
  return [clampN((e.clientX - r.left) / r.width), clampN((e.clientY - r.top) / r.height)];
}

function drawOverlay() {
  let s = '';
  if (CROP) s += `<rect class="crop" x="${CROP[0]*IMG_W}" y="${CROP[1]*IMG_H}" width="${(CROP[2]-CROP[0])*IMG_W}" height="${(CROP[3]-CROP[1])*IMG_H}"/>`;
  if (GUIDE) s += `<polygon class="air" points="${GUIDE.map(p => `${p[0]*IMG_W},${p[1]*IMG_H}`).join(' ')}"/>`;
  const line = path.filter(r => r[3] === 1 && r[1] != null).map(r => `${r[1]*IMG_W},${r[2]*IMG_H}`).join(' ');
  if (line) s += `<polyline class="path" points="${line}"/>`;
  const R = IMG_W * 0.0035;
  // Per-frame view: the faint polyline above carries the whole arc for context, but only the CURRENT
  // frame's marker is drawn — dots from other frames are hidden so each frame reads cleanly.
  for (const [f, px, py, vis, interp] of path) {
    if (f !== i || vis !== 1 || px == null) continue;
    const cls = interp ? 'fill' : 'anc';
    s += `<circle class="${cls} cur" cx="${px*IMG_W}" cy="${py*IMG_H}" r="${interp ? R : R*1.8}"/>`;
  }
  // occlusion + current-frame crosshair markers that have no image position: draw on the frame edge isn't
  // useful, so they live in the nav strip instead. Show the current frame's anchor highlight only.
  $('ov').innerHTML = s;
}

function renderStrip() {
  const frames = Object.keys(anchors).map(Number).sort((a, b) => a - b);
  const html = frames.map(f => {
    const occ = anchors[f].visibility === 0 ? ' occ' : '';
    return `<a href="#" class="anc${occ}" onclick="go(${f});return false;" title="${occ ? 'verdeckt' : 'Anker'}">${f}</a>`;
  }).concat([...boundaries].sort((a, b) => a - b).map(b =>
    `<a href="#" class="bnd" onclick="go(${b});return false;" title="Grenze">|${b}</a>`)).join('');
  $('flagbar').innerHTML = html;
  const nVis = path.filter(r => r[3] === 1).length;
  $('info').textContent = `${frames.length} Anker · ${boundaries.size} Grenzen · ${path.length} Frames markiert (${nVis} sichtbar)`;
}

function updateMeta() {
  const f = FRAMES[i];
  const a = anchors[i] ? (anchors[i].visibility === 0 ? ' · ⛔ verdeckt' : ' · ● Anker') : '';
  const b = boundaries.has(i) ? ' · | Grenze' : '';
  $('meta').textContent = `Frame ${i}/${NMAX} · ${f.t.toFixed(2)} s${a}${b}`;
}

function render() {
  const src = frameSrc();
  $('frame').src = src;
  // Keep the loupe in sync on frame change: it's otherwise only refreshed on mousemove, so advancing
  // frames with the cursor parked over the stage would leave it magnifying the previous frame. Position
  // stays correct (the cursor hasn't moved) — only the source frame needs updating.
  if ($('loupe').style.display === 'block') $('loupe').style.backgroundImage = `url(${src})`;
  $('sl').value = i; $('num').value = i;
  updateMeta(); drawOverlay(); renderStrip();
}
function go(n) { i = Math.max(0, Math.min(NMAX, n)); render(); }
function step(d) { go(i + d); }

function applyState(d) {   // server response carries the fresh anchors/boundaries/dense path
  path = d.path || [];
  anchors = {};
  for (const a of d.anchors || []) anchors[a.frame_idx] = a;
  boundaries = new Set(d.boundaries || []);
  renderJSON();
}

async function placeAnchor(nx, ny, visible) {
  const body = {frame_idx: i, t_s: FRAMES[i].t, visibility: visible ? 1 : 0};
  if (visible) { body.px = round4(nx); body.py = round4(ny); }
  applyState(await (await fetch('/anchor', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json());
  toast(visible ? 'Anker gesetzt' : 'als verdeckt markiert');
  // auto-advance to the next frame for rapid labeling (toggle in the toolbar); step(1)→render() redraws
  // overlay/strip/meta + loupe. When off, redraw in place so the just-placed anchor still shows.
  if (autoAdvance) step(1);
  else { drawOverlay(); renderStrip(); updateMeta(); }
}
async function toggleBoundary() {
  applyState(await (await fetch('/boundary', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({frame_idx: i})})).json());
  toast(boundaries.has(i) ? 'Grenze gesetzt' : 'Grenze entfernt');
  drawOverlay(); renderStrip(); updateMeta();
}
async function undo() {
  if (!anchors[i]) { toast('kein Anker in diesem Frame'); return; }
  applyState(await (await fetch(`/anchor?frame=${i}`, {method: 'DELETE'})).json());
  toast('Anker gelöscht');
  drawOverlay(); renderStrip(); updateMeta();
}

$('ov').addEventListener('click', e => { const [nx, ny] = ovXY(e); placeAnchor(nx, ny, !e.shiftKey); });
$('ov').addEventListener('contextmenu', e => { e.preventDefault(); const [nx, ny] = ovXY(e); placeAnchor(nx, ny, false); });

// ── loupe: magnify the native 4k frame around the cursor ──────────────────────────
const loupe = $('loupe');
$('stage').addEventListener('mousemove', e => {
  const st = $('stage').getBoundingClientRect();
  const lx = e.clientX - st.left, ly = e.clientY - st.top;
  if (lx < 0 || ly < 0 || lx > st.width || ly > st.height) { loupe.style.display = 'none'; return; }
  const nx = lx / st.width, ny = ly / st.height;
  loupe.style.backgroundImage = `url(${frameSrc()})`;
  loupe.style.backgroundSize = `${IMG_W * ZOOM}px ${IMG_H * ZOOM}px`;
  loupe.style.backgroundPosition = `${100 - nx * IMG_W * ZOOM}px ${100 - ny * IMG_H * ZOOM}px`;
  loupe.style.left = (lx + 24) + 'px';
  loupe.style.top = (ly - 220) + 'px';
  loupe.style.display = 'block';
});
$('stage').addEventListener('mouseleave', () => { loupe.style.display = 'none'; });

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') { e.preventDefault(); undo(); return; }
  if (e.key === 'x' || e.key === 'X') { e.preventDefault(); undo(); return; }   // delete this frame's anchor
  if (e.key === 'a' || e.key === 'A') { e.preventDefault(); toggleBoundary(); return; }
  if (e.key === 'c' || e.key === 'C') { e.preventDefault(); cycleView(); return; }
  if (e.key === 'ArrowRight') step(1);
  if (e.key === 'ArrowLeft') step(-1);
});

function toast(msg) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(t._timer); t._timer = setTimeout(() => t.classList.remove('show'), 1500);
}
function labelsJSON() {
  // mirror the saved shuttle_labels.json: sparse anchors + segment boundaries (the dense `path` is derived)
  const arr = Object.values(anchors).sort((a, b) => a.frame_idx - b.frame_idx);
  const bnd = [...boundaries].sort((a, b) => a - b);
  return JSON.stringify({video_id: VIDEO_ID, anchors: arr, boundaries: bnd}, null, 2);
}
function renderJSON() { const el = $('json'); if (el) el.textContent = labelsJSON(); }
async function copyJSON() {
  const text = labelsJSON();
  try {
    await navigator.clipboard.writeText(text);   // 127.0.0.1 is a secure context, so this is available
  } catch (e) {
    const ta = document.createElement('textarea');   // fallback for older/locked-down browsers
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } finally { ta.remove(); }
  }
  toast('JSON kopiert');
}

function fitStage() {
  const stage = $('stage');
  stage.style.maxWidth = '';
  const avail = window.innerHeight - stage.getBoundingClientRect().top - 16;
  stage.style.maxWidth = Math.max(360, avail * (IMG_W / IMG_H)) + 'px';
}
window.addEventListener('resize', fitStage);
document.documentElement.style.setProperty('--ar', IMG_W + ' / ' + IMG_H);
fitStage();
applyView();   // restore the saved contrast view (filter + selector) before the first frame loads
$('autoadv').checked = autoAdvance;   // reflect the saved auto-advance preference in the toolbar
(async () => { applyState(await (await fetch('/labels')).json()); render(); })();
</script>
</body></html>"""


def _serve_shuttle(data: ShuttleData, video_id: str, port: int) -> None:
    from .shuttle.label import interpolate_anchors

    frames_meta = json.dumps([{"t": round(f["t"], 3)} for f in data.frames])
    nmax = len(data.frames) - 1
    page = (
        _SHUTTLE_PAGE
        .replace("__NMAX__", str(nmax))
        .replace("__FRAMES__", frames_meta)
        .replace("__VIDEO_ID__", video_id)
        .replace("__IMG_W__", str(data.width))
        .replace("__IMG_H__", str(data.height))
        .replace("__GUIDE__", json.dumps(data.guide))
        .replace("__CROP__", json.dumps(data.crop))
    )

    labels_path = data.cache / "shuttle_labels.json"
    anchors: dict[int, dict] = {}     # frame_idx -> anchor (one per frame; re-clicking replaces)
    boundaries: set[int] = set()
    _id_counter = [0]

    def _load():
        if labels_path.exists():
            d = json.loads(labels_path.read_text())
            for a in d.get("anchors", []):
                a.setdefault("_id", _id_counter[0])
                _id_counter[0] += 1
                anchors[int(a["frame_idx"])] = a
            boundaries.update(int(b) for b in d.get("boundaries", []))

    def _anchor_list() -> list[dict]:
        return sorted(anchors.values(), key=lambda a: a["frame_idx"])

    def _save():
        labels_path.write_text(json.dumps(
            {"video_id": video_id, "anchors": _anchor_list(), "boundaries": sorted(boundaries)},
            indent=2, ensure_ascii=False))

    def _state() -> dict:
        # the dense interpolated path is the single tested source of truth (shuttle.label) — the same
        # function the importer + exporter use, so the live preview can't drift from the answer key.
        return {"video_id": video_id, "anchors": _anchor_list(), "boundaries": sorted(boundaries),
                "path": interpolate_anchors(_anchor_list(), boundaries)}

    _load()

    # Background-difference view ("Bewegung"): subtract the static per-clip median so the moving shuttle
    # pops on the white wall — the labeler's strongest spotting aid. Median is built once from a sample of
    # frames; per-frame diff JPEGs are lru-cached. cv2/numpy imported lazily (ml extra) as in pipeline/shuttle.
    _median = [None]

    def _background_median():
        if _median[0] is None:
            import cv2
            import numpy as np

            step = max(1, len(data.frames) // 25)   # ~25 evenly-spaced frames is plenty for a static wall
            stack = []
            for k in range(0, len(data.frames), step):
                img = cv2.imread(str(data.cache / "frames" / f"{int(data.frames[k]['frame']):06d}.jpg"))
                if img is not None:
                    stack.append(img)
            if not stack:
                raise RuntimeError("no frames to build background median")
            _median[0] = np.median(np.stack(stack), axis=0).astype("uint8")
            log.info("shuttle labeler: background median from %d sampled frames", len(stack))
        return _median[0]

    def _read_frame(idx: int):
        import cv2

        fnum = int(data.frames[idx]["frame"])
        img = cv2.imread(str(data.cache / "frames" / f"{fnum:06d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"{fnum:06d}.jpg")
        return img

    def _encode(img) -> bytes:
        import cv2

        _ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    @lru_cache(maxsize=512)
    def _diff_jpeg(idx: int) -> bytes:
        from .shuttle.enhance import diff_view

        return _encode(diff_view(_read_frame(idx), _background_median()))

    @lru_cache(maxsize=512)
    def _clahe_jpeg(idx: int) -> bytes:
        # Local-contrast (CLAHE on L) — v1's named white-shuttle cue; complements the diff view at the apex.
        from .shuttle.enhance import clahe_view

        return _encode(clahe_view(_read_frame(idx)))

    _ENH = {"diff": _diff_jpeg, "clahe": _clahe_jpeg}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send_json(self, body, status: int = 200) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_bytes(self, raw: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send_bytes(page.encode(), "text/html; charset=utf-8")
                return
            if u.path == "/frame":
                q = parse_qs(u.query)
                idx = max(0, min(nmax, int(q.get("idx", ["0"])[0])))
                fnum = int(data.frames[idx]["frame"])
                jpg = data.cache / "frames" / f"{fnum:06d}.jpg"   # native 4k — streamed straight, no re-encode
                enh = _ENH.get(q.get("enh", ["none"])[0])
                if enh is not None:
                    try:
                        self._send_bytes(enh(idx), "image/jpeg")
                        return
                    except Exception as e:  # noqa: BLE001 — fall back to the raw frame if cv2/median fails
                        log.warning("shuttle labeler: enhanced view failed (%s) — serving raw frame", e)
                if not jpg.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                self._send_bytes(jpg.read_bytes(), "image/jpeg")
                return
            if u.path == "/labels":
                self._send_json(_state())
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/anchor":
                fidx = int(body["frame_idx"])
                a = {"frame_idx": fidx, "t_s": body.get("t_s"), "visibility": int(body.get("visibility", 1)),
                     "_id": _id_counter[0]}
                _id_counter[0] += 1
                if a["visibility"] == 1:                 # a visible anchor carries the clicked pixel
                    a["px"], a["py"] = body.get("px"), body.get("py")
                anchors[fidx] = a                        # one anchor per frame — re-clicking replaces
                _save()
                self._send_json(_state())
                return
            if u.path == "/boundary":
                fidx = int(body["frame_idx"])
                boundaries.discard(fidx) if fidx in boundaries else boundaries.add(fidx)
                _save()
                self._send_json(_state())
                return
            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):
            u = urlparse(self.path)
            if u.path == "/anchor":
                fidx = int(parse_qs(u.query)["frame"][0])
                anchors.pop(fidx, None)
                _save()
                self._send_json(_state())
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    log.info("shuttle labeler for %s — open %s  (Ctrl-C to stop)", video_id, url)
    n_anchor = len(anchors)
    note = f" · {n_anchor} Anker geladen" if n_anchor else ""
    print(f"\n  ▶ Shuttle markieren: {url}\n    {len(data.frames)} Frames · Klick=Anker, Shift/Rechtsklick=verdeckt, "
          f"A=Grenze, X/Cmd+Z=löschen{note}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")


def verify_cli(argv: list[str]) -> None:
    video_id = "f35d0984-207c-4ee1-8283-ef8ba58eca82"
    port = 8000
    shuttle = False
    rest = list(argv)
    if "--shuttle" in rest:
        shuttle = True
        rest.remove("--shuttle")
    if "--port" in rest:
        p = rest.index("--port")
        port = int(rest[p + 1])
        del rest[p : p + 2]
    if rest:
        video_id = rest[0]
    cfg = load_config()
    if not cfg.configured:
        raise SystemExit("not configured — fill .env")
    if shuttle:
        _serve_shuttle(load_shuttle_data(video_id, cfg), video_id, port)
    else:
        _serve(load_data(video_id, cfg), video_id, port)
