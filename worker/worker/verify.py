"""Interactive frame-verification tool — scrub frames + inspect detections/tracking in the browser.

Run from the worker/ dir:  uv run python -m worker verify [video_id] [--port 8000]
Then open the printed URL. Slider / arrow keys step through frames; toggles show every detected
person (faint), the 4 kept in-court players (bold + label + foot dot), the calibrated court outline,
and a top-down map of where each player was placed. No new deps (stdlib http.server + PIL + polars).
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


def annotate(data: DebugData, idx: int, *, show_all=True, show_court=True, show_map=True) -> Image.Image:
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
        d.rectangle([mx, my, mx + mw, my + mh], outline=(255, 255, 255), width=int(sw))
        d.line([(mx, my + mh / 2), (mx + mw, my + mh / 2)], fill=(180, 180, 180), width=int(sw))
        for tid, role in data.role_by_tid.items():
            sub = sorted(data.trk.filter(pl.col("track_id") == tid).iter_rows(named=True), key=lambda r: abs(r["t"] - t))
            if not sub:
                continue
            col = ROLE_COLOR.get(role, (250, 136, 50))
            px, py = mx + sub[0]["u"] * mw, my + sub[0]["v"] * mh
            d.ellipse([px - sw * 6, py - sw * 6, px + sw * 6, py + sw * 6], fill=col, outline=(255, 255, 255), width=int(sw))

    return img.resize((OUT_W, int(OUT_W * H / W)))


_PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Detektion prüfen</title><style>
 body{{margin:0;background:#0a0a0b;color:#eee;font-family:Inter,system-ui,sans-serif}}
 header{{padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #222}}
 h1{{font-size:15px;margin:0;color:#FA8832;font-weight:700}}
 .meta{{font-size:13px;color:#aaa}} .legend{{font-size:12px;color:#bbb;display:flex;gap:12px}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}
 #wrap{{padding:12px 16px}} img{{width:100%;max-width:1280px;display:block;border-radius:8px;background:#000}}
 .ctl{{display:flex;gap:12px;align-items:center;margin:10px 0;flex-wrap:wrap}}
 input[type=range]{{flex:1;min-width:240px;accent-color:#FA8832}}
 button{{background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:14px}}
 label{{font-size:13px;color:#ccc;cursor:pointer}} input[type=number]{{width:70px;background:#1a1a1c;color:#eee;border:1px solid #333;border-radius:6px;padding:4px}}
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
 <div class="ctl">
  <button onclick="step(-1)">◀ Zurück</button>
  <input id="sl" type="range" min="0" max="{NMAX}" value="0" oninput="go(+this.value)">
  <button onclick="step(1)">Weiter ▶</button>
  <label>Frame <input id="num" type="number" min="0" max="{NMAX}" value="0" onchange="go(+this.value)"></label>
 </div>
 <div class="ctl">
  <label><input type="checkbox" id="all" checked onchange="render()"> Alle Personen</label>
  <label><input type="checkbox" id="court" checked onchange="render()"> Spielfeld</label>
  <label><input type="checkbox" id="map" checked onchange="render()"> Vogelperspektive</label>
 </div>
 <img id="frame" src="">
</div>
<script>
 const FRAMES = {FRAMES};
 let i = 0;
 const $ = id => document.getElementById(id);
 function render(){{
   const q = `idx=${{i}}&all=${{$('all').checked?1:0}}&court=${{$('court').checked?1:0}}&map=${{$('map').checked?1:0}}`;
   $('frame').src = '/frame?' + q;
   const f = FRAMES[i];
   $('meta').textContent = `Frame ${{i}}/{NMAX} · ${{f.t.toFixed(1)}} s · ${{f.n}} Personen erkannt · 4 behalten`;
 }}
 function go(n){{ i = Math.max(0, Math.min({NMAX}, n)); $('sl').value=i; $('num').value=i; render(); }}
 function step(d){{ go(i+d); }}
 document.addEventListener('keydown', e => {{ if(e.key==='ArrowRight')step(1); if(e.key==='ArrowLeft')step(-1); }});
 render();
</script></body></html>"""


def _serve(data: DebugData, video_id: str, port: int) -> None:
    counts = data.det.group_by("frame").len().to_dict(as_series=False)
    by_frame = dict(zip(counts["frame"], counts["len"]))
    frames_meta = json.dumps([{"t": round(f["t"], 2), "n": int(by_frame.get(f["frame"], 0))} for f in data.frames])
    nmax = len(data.frames) - 1
    page = _PAGE.format(NMAX=nmax, FRAMES=frames_meta)

    @lru_cache(maxsize=256)
    def render_jpeg(idx: int, show_all: bool, show_court: bool, show_map: bool) -> bytes:
        buf = io.BytesIO()
        annotate(data, idx, show_all=show_all, show_court=show_court, show_map=show_map).save(buf, "JPEG", quality=85)
        return buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if u.path == "/frame":
                q = parse_qs(u.query)
                idx = max(0, min(nmax, int(q.get("idx", ["0"])[0])))
                png = render_jpeg(idx, q.get("all", ["1"])[0] == "1", q.get("court", ["1"])[0] == "1", q.get("map", ["1"])[0] == "1")
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    log.info("verify tool for %s — open %s  (Ctrl-C to stop)", video_id, url)
    print(f"\n  ▶ Detektion prüfen: {url}\n    {len(data.frames)} Frames · Pfeiltasten/Slider zum Blättern\n")
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
