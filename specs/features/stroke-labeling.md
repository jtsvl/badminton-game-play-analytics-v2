# Feature — Stroke labeling (training data for stage 7)

**Status: BUILT (2026-07-03; import side 2026-07-05).** The labeling on-ramp for stroke
classification (`features/stroke-classification.md`, FR-21/22/23): labeling is the only gate to
training the stage-7 model, so this tooling exists before any model code. Export CLI +
self-hosted Label Studio + import CLI; all 738 Session-2 clips are labeled and imported.

## What the labeler sees and why

**Real video with audio** cut from the existing 720p proxy (`features/video-proxy.md` keeps audio
for exactly this kind of use) — NOT a skeleton/shuttle overlay render. Swing shape, shuttle arc AND
sound (a smash's crack vs. a net shot's soft tap vs. a drop's near-silence) are the distinguishing
signals a human actually uses, especially for the hard pairs (drop/net, clear/lift). The
pose/shuttle/court parquet is the *model's* input at training time, not the labeler's.

**Clip window: hit ± 0.7 s** (default; `--pre/--post` to tune). Wider than stage 6's ±0.5 s search
margin on purpose: the labeler needs the full backswing-through-follow-through. Frame-strips from
real Session-2 clips (pilot 2026-07-03) show the provisional hit timing drifts up to ~0.2 s, which
±0.7 s absorbs; a fast exchange can put the previous/next stroke at the clip edges — the labeler
instruction is therefore **"label the stroke at the clip's center"**.

## Label schema (settled with the trainer 2026-07-03)

Two questions per clip:

1. **Schlagart** — the coarse 7 (FR-23) plus two honesty options:
   `Clear, Smash, Drop, Netzschlag, Drive, Lift, Aufschlag` + **`Unsicher`** (human can't tell —
   maps to `unknown`, FR-22 honesty applies to labels too) + **`Kein Schlag`** (stage-6 false
   positive — nobody hit; doubles as free validation data for hit detection precision).
2. **Wer hat geschlagen** — `vorne-links` / `vorne-rechts` / `hinten-links` / `hinten-rechts`
   (near/far side **and** left/right as seen on screen). **All Session-2 matches are doubles**
   (`discipline='doubles'` in `matches`, confirmed 2026-07-04 — the earlier "singles-first" framing
   of this question was wrong for this batch: with 2 players per side, `vorne`/`hinten` alone can't
   tell the teammates apart). The 4 choices match `geometry.role_assign`'s doubles roles
   (`near_left`/`near_right`/`far_left`/`far_right`) 1:1. Ground truth to validate geometric
   attribution, whose known weak spot is net play (`stroke-classification.md` honest limits).
   Skipped for `Kein Schlag`. **Not yet consumed**: `stroke-classification.md` still scopes doubles
   *attribution* (who of the pair hit, for training) to P3+ — collecting the 4-way label now is
   cheap and avoids relabeling later, but the stage-7 model won't use the left/right split until
   then; only the shot-type label is used near-term.

**Not labeled: outcome (winner/error).** Only rally-ending shots have one (~150 of ~738 clips),
and `20-data-model.md:55-64` already derives winners/errors from rally endings + landing in/out
(`features/shuttle-landing-inout.md`). Rally outcomes are corrected in the rally-review UI, not here.

**Serve hint, never auto-fill:** the first hit of each rally shows „Erster Schlag im Ballwechsel
(vermutlich Aufschlag)" as display-only text. Pre-selecting `Aufschlag` would bias every
rally-start false positive toward serve.

Canonical mapping for the (future) import step:

| Label Studio (German UI) | canonical `shot_type` |
|---|---|
| Clear | `clear` |
| Smash | `smash` |
| Drop | `drop` |
| Netzschlag | `net` |
| Drive | `drive` |
| Lift | `lift` |
| Aufschlag | `serve` |
| Unsicher | `unknown` |
| Kein Schlag | `no_hit` (excluded from stroke training; feeds hit-detection eval) |
| vorne-links | `near_left` |
| vorne-rechts | `near_right` |
| hinten-links | `far_left` |
| hinten-rechts | `far_right` |

## Export CLI (`worker/worker/stroke_clips.py`)

```
python -m worker export-stroke-clips --video-id <uuid> [--video-id <uuid> ...]
       [--out DIR] [--pre 0.7] [--post 0.7] [--sample N] [--limit N]
```

- Reads `events` `type='hit'` + `rallies` for the video's match; **drops hits outside current
  rally spans** (rally-review edits may have removed rallies since stage 6 wrote the events).
- Cuts `[t_s−pre, t_s+post]` (clamped to `[0, duration_s]`) from the local proxy
  (`<frame_cache>/<video_id>/proxy_720p.mp4`) with a **re-encode** — the proxy has keyframes only
  every 2 s (`-g 60`), so `-c copy` would snap cuts up to 2 s off; `-ss` before `-i` + re-encode is
  fast AND frame-accurate. House VideoToolbox recipe, audio kept (aac 96k). ~740 clips ≈ minutes.
- Writes `{out}/stroke/<game>/<game>_r<NN>_h<NN>_<ttttt.tt>s_<event8>.mp4` (sortable names) and
  `{out}/tasks_<game>.json` in Label Studio import format. Task `data` carries `event_id`,
  `video_id`, `match_id`, `rally_idx`, `hit_idx`, `hit_t_s`, `first_of_rally`, `serve_hint` — the
  ids make the later import join back to DB rows without guesswork.
- Default `--out`: `.labeling/` next to the frame cache (git-ignored via `**/.labeling/`).
- `--sample N` = deterministic even spread (reproducible pilot batches); `--limit N` = first N.
- Pure mapping TDD'd in `worker/tests/test_stroke_clips.py`; I/O shell smoke-tested on real proxies.

## Label Studio (self-hosted on the Mac Studio)

- **Install:** `uv tool install label-studio` (1.23.0, native — no Docker; the clips are local
  files and Docker on macOS only adds a VM + mount hop). Data in `~/.local/share/label-studio`.
- **Start** (port 8081 — 8080 is taken by Signal on this Mac):

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/jtsvl/Projects/badminton-game-play-analytics-v2/worker/worker/.labeling
export LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true  # API-token access; only honored at FIRST boot (org creation)
label-studio start --no-browser --port 8081
```

- Login: `jt.svl@icloud.com` (password set at first start — change under *Account & Settings*).
- One project **„Schlagarten Session 2"**; tasks reference clips as
  `/data/local-files/?d=stroke/<game>/<file>.mp4` (relative to the document root).
- **Required in 1.23:** the project needs a *Local Files* source storage whose path covers the
  clips (Settings → Cloud Storage → Add Source Storage → Local Files, path
  `<document root>/stroke`) — since 1.23 the `/data/local-files/` endpoint only serves files
  under a registered storage of an accessible project. Do **not** press Sync (tasks come from the
  JSON import; the storage row exists purely to authorize serving).
- Import: *Project → Import* → the `tasks_*.json` files (or via API).
- Labeling config (German, du-form; number hotkeys → ~2 keystrokes per clip):

```xml
<View>
  <Header value="Ballwechsel $rally_idx · Schlag $hit_idx · $game"/>
  <Text name="hint" value="$serve_hint"/>
  <Video name="video" value="$video" height="450" frameRate="30"/>
  <Choices name="shot_type" toName="video" choice="single" showInLine="true" required="true"
           requiredMessage="Bitte Schlagart wählen (oder Unsicher/Kein Schlag)">
    <Choice value="Clear" hotkey="1"/>
    <Choice value="Smash" hotkey="2"/>
    <Choice value="Drop" hotkey="3"/>
    <Choice value="Netzschlag" hotkey="4"/>
    <Choice value="Drive" hotkey="5"/>
    <Choice value="Lift" hotkey="6"/>
    <Choice value="Aufschlag" hotkey="7"/>
    <Choice value="Unsicher" hotkey="8"/>
    <Choice value="Kein Schlag" hotkey="9"/>
  </Choices>
  <View visibleWhen="choice-selected" whenTagName="shot_type"
        whenChoiceValue="Clear,Smash,Drop,Netzschlag,Drive,Lift,Aufschlag,Unsicher">
    <Header value="Wer hat geschlagen?" size="4"/>
    <Choices name="hitter" toName="video" choice="single" showInLine="true">
      <Choice value="vorne-links" hotkey="q"/>
      <Choice value="vorne-rechts" hotkey="w"/>
      <Choice value="hinten-links" hotkey="e"/>
      <Choice value="hinten-rechts" hotkey="r"/>
    </Choices>
  </View>
</View>
```

## Import CLI (`worker/worker/import_stroke_labels.py`) — BUILT 2026-07-05

```
python -m worker import-stroke-labels <ls-export.json> [--out DIR]
```

Reads the Label Studio JSON export, maps German choice values → canonical tokens (table above),
and writes two parquets with identical schema
(`event_id, match_id, video_id, t_s, shot_type, hitter, rally_idx, hit_idx`) next to the export
(or `--out`): **`stroke_labels.parquet`** (training — `unknown` included, it is the taxonomy's
honesty option; downstream training filters as it wishes) and **`hit_detection_eval.parquet`**
(the `no_hit` rows — stage-6 false positives, feeds hit-detection eval, never stroke training).

- **DB-free by design**: the task `data` carries `event_id`/`match_id`/`video_id`/`hit_t_s`
  verbatim from the `events` rows the exporter read, so `t_s` comes straight from `hit_t_s` and a
  DB round-trip would validate nothing; the join back to `events` happens when stage 7 consumes
  the parquet. Corruption is guarded where it can enter: unmapped German values, a missing
  `shot_type`, or a duplicate `event_id` are hard `ValueError`s — never silently coerced.
- Unlabeled / all-cancelled tasks are skipped (counted in the summary); with multiple
  annotations the latest non-cancelled wins (a relabel supersedes). Missing `hitter` → null.
- Pure `rows_from_export`/`split_rows` TDD'd in `worker/tests/test_import_stroke_labels.py`.

**Session-2 result (2026-07-05, all 738 tasks labeled):** 695 training rows + 43 `no_hit`.
shot_type: drop 165, lift 153, unknown 141, clear 78, smash 61, serve 55, drive 22, net 20.
184 rows without hitter (43 = all `no_hit` + 141 mostly-`Unsicher` — the hitter question was
shown for `Unsicher` but not required, and was mostly skipped there; known label-data gap).

## Honest limits

- Clips inherit stage 6's provisional hit timing (±0.2 s drift observed) — the window absorbs it,
  but a badly mistimed detection becomes an `Unsicher`/`Kein Schlag`, which is the correct signal.
- 720p/30fps is enough for a *human* with audio; the model trains on the 4K-derived parquet
  features, so labeling quality, not clip resolution, is the ceiling.
- Label Studio runs only on this Mac (localhost) — labeling is a trainer-at-the-Mac activity;
  nothing is uploaded anywhere.
- **Doubles net-play attribution is the hardest case for the left/right split**, same root cause as
  the shot-type net-play limit above (both players converge on the net, occlusion stacks them from
  the overhead angle) — `Unsicher` is correct there too, don't force a left/right guess.
