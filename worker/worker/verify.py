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
OUT_W = 1280


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


def annotate(data: DebugData, idx: int, *, show_all=True, show_court=True, show_map=True, hide=()) -> Image.Image:
    import polars as pl

    fmeta = data.frames[idx]
    t = fmeta["t"]
    W, H = data.width, data.height
    sw = max(W, H) / 640
    font = ImageFont.load_default(size=int(max(W, H) * 0.02))
    kept_ids = list(data.role_by_tid)

    img = Image.open(data.cache / "frames" / f"{fmeta['frame']:06d}.jpg").convert("RGB")
    d = ImageDraw.Draw(img)

    if show_all:
        for r in data.det.filter(pl.col("frame") == fmeta["frame"]).iter_rows(named=True):
            d.rectangle([r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]], outline=(170, 170, 170), width=int(sw))

    if show_court:
        d.line([*[tuple(p) for p in data.corners], tuple(data.corners[0])], fill=(255, 255, 255), width=int(sw * 2.2))
        d.line([apply_homography(data.h_uv2px, 0, 0.5), apply_homography(data.h_uv2px, 1, 0.5)], fill=(255, 255, 255), width=int(sw * 1.6))

    for r in data.det.filter((pl.col("frame") == fmeta["frame"]) & (pl.col("track").is_in(kept_ids))).iter_rows(named=True):
        role = data.role_by_tid.get(int(r["track"]), "")
        if ROLE_LABEL.get(role, role) in hide:   # hidden player → don't bake the bold box/dot/label
            continue
        col = ROLE_COLOR.get(role, (250, 136, 50))
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        d.rectangle([x, y, x + w, y + h], outline=col, width=int(sw * 3))
        ax, ay = ankle_point((x, y, w, h))
        d.ellipse([ax - sw * 5, ay - sw * 5, ax + sw * 5, ay + sw * 5], fill=col, outline=(255, 255, 255), width=int(sw))
        d.text((x, y - max(W, H) * 0.026), ROLE_LABEL.get(role, role), fill=col, font=font, stroke_width=int(sw), stroke_fill=(0, 0, 0))

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
        for tid, role in data.role_by_tid.items():
            if ROLE_LABEL.get(role, role) in hide:   # hidden player → no map dot either
                continue
            sub = sorted(data.trk.filter(pl.col("track_id") == tid).iter_rows(named=True), key=lambda r: abs(r["t"] - t))
            if not sub:
                continue
            col = ROLE_COLOR.get(role, (250, 136, 50))
            px, py = mx + sub[0]["u"] * mw, my + sub[0]["v"] * mh
            d.ellipse([px - sw * 6, py - sw * 6, px + sw * 6, py + sw * 6], fill=col, outline=(255, 255, 255), width=int(sw))

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
/* ghost dot showing the original (pre-correction) position */
.orig-dot{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
  border:2px dashed rgba(255,255,255,0.45);background:transparent;pointer-events:none}
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

async function render() {
  closePopup();
  const hide = [...hidden].sort().join(',');
  $('frame').src = `/frame?idx=${i}&all=${$('all').checked?1:0}&court=${$('court').checked?1:0}&map=${$('map').checked?1:0}&hide=${hide}`;
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

function buildOverlay() {
  const ov = $('overlay');
  ov.innerHTML = '';
  els = {dot: {}, ring: {}, box: {}};
  const boxMode = $('boxedit').checked;

  // editable person boxes (drawn first so the dots/handles sit on top)
  if (boxMode) {
    for (const d of dets) if (!hidden.has(d.label)) addEditableBox({x:d.bbox[0], y:d.bbox[1], w:d.bbox[2], h:d.bbox[3]}, d.label, false);
  }

  // index latest correction per role for the current frame
  const frameCors = {};
  for (const c of (corrections[i] || [])) frameCors[c.role] = c;

  // video foot dots — drag corrects the position; the map ring follows live
  for (const d of dets) {
    if (hidden.has(d.label)) continue;
    const cor = frameCors[d.label];
    // ghost dot at the original (pre-correction) ankle position
    if (cor && cor.orig_foot) {
      const g = document.createElement('div');
      g.className = 'orig-dot';
      g.style.left = (cor.orig_foot[0] * 100) + '%';
      g.style.top  = (cor.orig_foot[1] * 100) + '%';
      g.style.borderColor = ROLE_COLORS[d.label] || '#FA8832';
      ov.appendChild(g);
    }
    const m = document.createElement('div');
    m.className = 'marker' + (cor ? ' corrected' : '');
    m.style.left = (d.ankle[0] * 100) + '%';
    m.style.top  = (d.ankle[1] * 100) + '%';
    m.style.background = ROLE_COLORS[d.label] || '#FA8832';
    m.textContent = d.label;
    m.title = d.label + ' — ziehen: Position (Bild + Karte synchron)';
    attachPosDrag(m, d);
    ov.appendChild(m);
    els.dot[d.label] = m;
  }

  // matching rings on the court map — drag corrects the court position; the video dot follows live
  if (mapRect && $('map').checked) {
    for (const d of dets) {
      if (!d.uv || hidden.has(d.label)) continue;
      const m = document.createElement('div');
      m.className = 'mmarker';
      m.style.left = ((mapRect.x + d.uv[0] * mapRect.w) * 100) + '%';
      m.style.top  = ((mapRect.y + d.uv[1] * mapRect.h) * 100) + '%';
      m.style.borderColor = ROLE_COLORS[d.label] || '#FA8832';
      m.title = d.label + ' — ziehen: Court-Position (Karte + Bild synchron)';
      attachRingDrag(m, d);
      ov.appendChild(m);
      els.ring[d.label] = m;
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
  updateMeta();      // refresh the ⚑ count without reloading the frame (keeps a dragged marker in place)
  renderList();
  renderFlagBar();
  buildOverlay();    // apply corrected highlight + ghost dot immediately
}

async function deleteCorrection(fidx, cidx) {
  await fetch(`/correct?frame=${fidx}&idx=${cidx}`, {method: 'DELETE'});
  corrections[fidx].splice(cidx, 1);
  if (!corrections[fidx].length) delete corrections[fidx];
  updateMeta();
  renderList();
  renderFlagBar();
  if (fidx === i) buildOverlay();   // refresh highlight state for the current frame
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

    @lru_cache(maxsize=512)
    def render_jpeg(idx: int, show_all: bool, show_court: bool, show_map: bool, hide_key: str = "") -> bytes:
        hide = tuple(h for h in hide_key.split(",") if h)
        buf = io.BytesIO()
        annotate(data, idx, show_all=show_all, show_court=show_court, show_map=show_map, hide=hide).save(buf, "JPEG", quality=85)
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
                png = render_jpeg(idx, q.get("all", ["1"])[0] == "1", q.get("court", ["1"])[0] == "1", q.get("map", ["1"])[0] == "1", hide_key)
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


def verify_cli(argv: list[str]) -> None:
    video_id = "f35d0984-207c-4ee1-8283-ef8ba58eca82"
    port = 8000
    rest = list(argv)
    if "--port" in rest:
        p = rest.index("--port")
        port = int(rest[p + 1])
        del rest[p : p + 2]
    if rest:
        video_id = rest[0]
    cfg = load_config()
    if not cfg.configured:
        raise SystemExit("not configured — fill .env")
    _serve(load_data(video_id, cfg), video_id, port)
