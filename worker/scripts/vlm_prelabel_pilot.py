"""VLM pre-labeling pilot — measure a LOCAL vision-language model against Session-2 labels.

Adoption gate (decided before running, specs/features/stroke-labeling.md): the VLM becomes a
pre-labeler ONLY if its top-1 agreement beats the GBDT's 0.534 overall AND materially beats
it on net (0.30) / smash (0.21). Otherwise this rung is dropped and the numbers documented.
Local-only by user decision 2026-07-12 (club members' footage never leaves the Mac); the
model runs via mlx-vlm on Apple Silicon. Honest expectations: VLMs are weak at fine-grained
temporal action recognition, the 9 sampled frames lose motion continuity, and ALL audio —
one of the three signals the human labeling flow was designed around — is gone. The pilot is
run because it is cheap and decisive, not because it is likely to win.

Run (mlx-vlm is NOT a worker dependency — pull it ad hoc):

    cd worker && uv run --with mlx-vlm python scripts/vlm_prelabel_pilot.py \
        --labeling-dir worker/.labeling [--per-class 15] [--seed 42] [--limit N]

Output: <labeling>/vlm_pilot/vlm_predictions.parquet in the OOF column shape, so
`python -m worker stroke-assist-report --oof <that file>` prints the identical
agreement/threshold report for the head-to-head vs the GBDT. (Single-label confidence is
spread uniformly over the other classes — the top-2 line is not meaningful for the VLM.)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

WORKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_DIR))

from worker.import_stroke_labels import GERMAN_BY_CANONICAL, SHOT_TYPE  # noqa: E402
from worker.stroke.train import CLASSES  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
N_FRAMES = 9
FRAME_SIZE = "640:360"

PROMPT = """Du siehst {n} Standbilder (zeitlich geordnet, gleichmäßig über ~1,4 s) aus einem \
Badminton-Doppel, aufgenommen erhöht hinter dem Feld. Das Netz verläuft horizontal in der \
Bildmitte. In der MITTE des Clips schlägt ein Spieler den Federball. Bestimme die Schlagart \
des Schlags in der Clip-Mitte.

Antwortmöglichkeiten (genau eine):
- Clear: hoch und weit bis zur gegnerischen Grundlinie, aus dem Hinterfeld
- Smash: steil und schnell nach unten geschlagen, meist aus dem Hinterfeld, Arm über Kopf
- Drop: aus dem Hinterfeld sanft kurz hinter das Netz gespielt
- Netzschlag: weicher Schlag aus dem Vorderfeld knapp über die Netzkante
- Drive: flacher, schneller Schlag etwa auf Netzhöhe parallel zum Boden
- Lift: aus dem Vorderfeld hoch ins gegnerische Hinterfeld gehoben
- Aufschlag: Ballwechsel-Eröffnung, im Doppel meist kurz und flach von unten
- Unsicher: nicht erkennbar
- Kein Schlag: kein Schlag in der Clip-Mitte (Fehldetektion)
{serve_hint}
Antworte NUR mit JSON: {{"label": "<eine der Antwortmöglichkeiten>", "confidence": <0..1>}}"""

SERVE_HINT = "\nHinweis: Dies ist der erste Schlag des Ballwechsels (vermutlich Aufschlag).\n"


# --- pure helpers (TDD in tests/test_vlm_pilot.py) -------------------------------------------

def stratified_sample(rows: Sequence[dict], per_class: int, seed: int) -> List[dict]:
    """min(per_class, n) rows per shot_type, deterministic under `seed`, class-sorted."""
    rng = random.Random(seed)
    by_cls: dict = {}
    for r in rows:
        by_cls.setdefault(r["shot_type"], []).append(r)
    out: List[dict] = []
    for cls in sorted(by_cls):
        rows_cls = by_cls[cls]
        take = rows_cls if len(rows_cls) <= per_class else rng.sample(rows_cls, per_class)
        out.extend(sorted(take, key=lambda r: str(r["event_id"])))
    return out


def parse_vlm_answer(text: str) -> Optional[Tuple[str, float]]:
    """(canonical label, confidence in [0,1]) from the model's reply, or None on garbage."""
    m = re.search(r'\{[^{}]*"label"[^{}]*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    label = obj.get("label")
    if label not in SHOT_TYPE:
        return None
    conf = obj.get("confidence", 0.5)
    try:
        conf = min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.5
    return SHOT_TYPE[label], conf


# --- measurement glue -------------------------------------------------------------------------

def clip_paths_from_tasks(tasks_glob: str, labeling_root: Path) -> dict:
    """{event_id: clip path} from the original exporter task JSONs."""
    out: dict = {}
    paths = sorted(Path().glob(tasks_glob)) if tasks_glob else []
    for p in paths or sorted(labeling_root.glob("tasks_*.json")):
        for task in json.loads(Path(p).read_text(encoding="utf-8")):
            rel = task["data"]["video"].split("?d=", 1)[-1]
            out[task["data"]["event_id"]] = {
                "path": labeling_root / rel,
                "first_of_rally": bool(task["data"].get("first_of_rally")),
            }
    return out


def extract_frames(clip: Path, out_dir: Path, n: int = N_FRAMES) -> List[Path]:
    """n frames spread evenly over the clip via ffprobe duration + per-frame seeks."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(clip)], capture_output=True, text=True, timeout=30)
    dur = float(probe.stdout.strip() or 1.4)
    frames = []
    for i in range(n):
        t = dur * i / (n - 1) if n > 1 else 0.0
        t = min(t, max(0.0, dur - 0.05))  # last frame: back off EOF
        f = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-ss", f"{t:.3f}", "-i", str(clip),
                        "-frames:v", "1", "-vf", f"scale={FRAME_SIZE}", str(f)],
                       capture_output=True, timeout=30)
        if f.exists():
            frames.append(f)
    return frames


def uniform_probs(pred: str, conf: float) -> dict:
    """prob_* columns: confidence on the predicted class, remainder spread uniformly.
    Keeps the OOF parquet shape; top-2 stats are NOT meaningful for a single-label model."""
    rest = (1.0 - conf) / (len(CLASSES) - 1) if pred in CLASSES else 1.0 / len(CLASSES)
    return {f"prob_{c}": (conf if c == pred else rest) if pred in CLASSES else rest
            for c in CLASSES}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labeling-dir", required=True,
                    help="dir holding stroke_labels.parquet, tasks_*.json and the clips")
    ap.add_argument("--tasks", default="", help="override glob for the task JSONs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="stop after N clips (smoke run)")
    ap.add_argument("--out", help="output parquet "
                                  "(default: <labeling>/vlm_pilot/vlm_predictions.parquet)")
    args = ap.parse_args()

    import polars as pl
    from mlx_vlm import generate, load  # applies the chat template internally (0.6.x)

    labeling = Path(args.labeling_dir).expanduser()
    out_path = (Path(args.out).expanduser() if args.out
                else labeling / "vlm_pilot" / "vlm_predictions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [r for r in pl.read_parquet(labeling / "stroke_labels.parquet").iter_rows(named=True)
              if r["shot_type"] in CLASSES]  # the 7 trained classes; unknown has no truth
    sample = stratified_sample(labels, per_class=args.per_class, seed=args.seed)
    if args.limit:
        sample = sample[:args.limit]
    clips = clip_paths_from_tasks(args.tasks, labeling)

    print(f"loading {args.model} …")
    model, processor = load(args.model)

    rows, skipped, t_start = [], 0, time.monotonic()
    for i, lab in enumerate(sample):
        info = clips.get(lab["event_id"])
        if info is None or not info["path"].exists():
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as d:
            frames = extract_frames(info["path"], Path(d))
            if not frames:
                skipped += 1
                continue
            prompt = PROMPT.format(n=len(frames),
                                   serve_hint=SERVE_HINT if info["first_of_rally"] else "")
            reply = generate(model, processor, prompt,
                             [str(f) for f in frames], max_tokens=80, verbose=False)
        parsed = parse_vlm_answer(reply.text if hasattr(reply, "text") else str(reply))
        # honesty: an unparsable or non-class answer ("Unsicher"/"Kein Schlag") stays what
        # it is — pred never equals the 7-class truth, so it counts as a miss everywhere
        pred, conf = parsed if parsed else ("unparsed", 0.0)
        rows.append({
            "event_id": lab["event_id"], "video_id": lab["video_id"], "t_s": lab["t_s"],
            "shot_type": lab["shot_type"], "is_calibration": False, "fold": -1,
            **uniform_probs(pred, conf),
            "pred": pred,
            "p_max": conf if pred in CLASSES else 0.0,
            "vlm_raw_label": pred,
        })
        done = i + 1
        rate = (time.monotonic() - t_start) / done
        print(f"[{done}/{len(sample)}] {lab['shot_type']:<6} -> {pred:<8} "
              f"({conf:.2f})  ~{rate:.1f}s/clip")

    if not rows:
        print("no clips scored — check --labeling-dir / task JSONs")
        return 1
    pl.DataFrame(rows).write_parquet(out_path)
    n_hit = sum(1 for r in rows if r["pred"] == r["shot_type"])
    n_class = sum(1 for r in rows if r["pred"] in CLASSES)
    print(f"\n{len(rows)} clips scored ({skipped} skipped): "
          f"{n_class} class answers, {len(rows) - n_class} abstained/unparsed; "
          f"top-1 agreement {n_hit / len(rows):.3f} (misses include abstentions) "
          f"-> {out_path}")
    print("full report: python -m worker stroke-assist-report --oof", out_path)
    print("adoption gate: beats GBDT 0.534 overall AND materially on net(0.30)/smash(0.21)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
