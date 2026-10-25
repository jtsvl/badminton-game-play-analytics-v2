# FineBadminton — Evaluation (Phase-0-Nachtrag)

**Datum:** 2026-07-13 · Anlass: User fand Download-Pfade (HuggingFace), die die Recherche
verpasst hatte, und will die LLM-Analyse-Funktionalität bewertet haben. · Prüftiefe: alle
6 Quellen + Cross-Referenzen direkt geladen (Firecrawl + HF-API); **alle 70
Annotations-JSONs heruntergeladen und selbst ausgezählt** — die Zahlen unten sind eigene
Messungen am Datenbestand, nicht Paper-Zitate, sofern nicht anders markiert.

## 1. Was FineBadminton wirklich liefert — Memo-Korrektur

**Der Memo-Befund „kein Download-Pfad" ist FALSCH und wird korrigiert.** Beide
HF-Datasets sind frei (nicht gated) und **Apache-2.0-lizenziert** — ein echter
Lizenz-Grant, besser als ShuttleSet-Situation (dort nur Repo-MIT) und ganz ohne
Override-Bedarf:

| Repo | Inhalt (selbst geprüft, HF-API-Dateibaum) | Größe |
|---|---|---|
| `iLearn-Lab/Finebadminton-20K` | **70 komplette Match-Videos (mp4)** + 70 Annotations-JSONs | 43,4 GB (Videos) + 27 MB (JSONs) |
| `iLearn-Lab/FineBadmintonBenchmark` | 984 QA-Video-Clips + 12 Benchmark-Task-JSONs (en+zh, MCQ + offene Fragen) | 0,8 GB |
| `iLearn-Lab/MM25-FineBadminton` (GitHub) | Demo-Website + Apache-2.0-LICENSE; verweist auf `Moujuruo/…`-Spiegel derselben HF-Datasets | — |

Selbst gezählter Bestand der 70 JSONs: **2 066 Rounds (Rallies), 20 757 Hits („20K"),
74 Spieler, ausnahmslos EINZEL** (2 Spieler je Round). Quellmaterial: BWF-YouTube-Broadcast,
720p+, 25 fps, Kamera hinter dem Feld [Paper §3.1].

**Diskrepanz Paper ↔ Release (ehrlich):** Das Paper beschreibt 120 Matches / 3 215
Rallies / 33 325 Strokes — auf HF liegt ein ~62-%-Subset (70/2 066/20 757).
Annotations-Rauschen existiert: 27 leere hit types, eine Handvoll chinesischer
Fragment-Subtypen (劈吊, 拨腰 …), 1× chinesische Ball-Area — unkritisch, aber beim
Einlesen filtern.

**Annotations-Schema je Hit** (aus den JSONs verifiziert; identisch zur Demo-Seite):
`start_frame`/`end_frame` (!), Spielername, `get_point` (score/lose the point),
3 Ebenen — Foundational (hit type, subtype[], hitter top/bottom, ball area),
Tactical (player actions[], shot characteristics[]), Decision (Freitext-Kommentar,
Quality **1–7**). Je Round: `playerList` (Server/Receiver), `evaluations_new` mit
`score_reason`/`lose_reason`/`total_eva` (Ø 41/44/63 Wörter [Paper §3.3]).

## 1a. Nachtrag: die Demo zeigt ein REICHHALTIGERES Schema als der Bulk-Download

**User-Fund (Screenshot der Demo-„Hitting Information"-Karte):** die Live-Demo zeigt vier
Tactical/Decision-Felder (Player Actions, Shot Characteristics, Strategies, Hit Outcomes),
der öffentliche 20K-Datensatz hat nur zwei (`player actions`, `shot characteristics`).
Nachrecherchiert: Die Demo lädt NICHT den HF-Datensatz, sondern eine eigene Datei aus dem
GitHub-Repo (`dataset/transformed_combined_rounds_output_en_evals_translated.json`,
per `script.js` bestätigt — keine Client-seitige Ableitung, echte Zusatzfelder). Diese
Datei ist aber nur ein **40-Runden-Showcase** (13 Matches, 414 Hits), NICHT der volle
Datensatz.

Ausgezählt (414 Hits): **`hit_frame`** — ein einzelner präziser Kontakt-Frame, getrennt
vom Bewegungs-Fenster `start_frame`/`end_frame` — bei **100 %** aller Hits vorhanden;
**`strategies`** (taktische Absicht: passive, defensive, intercept, deception,
hesitation …), sauber getrennt von `player_actions` (reine Technik: forehand/backhand),
bei **86 %**; **`hit_outcomes`** (Werte nur „out"/„Net fault" gesehen) bei nur **2,9 %**
— ausschließlich auf Rally-Endschlägen.

**Konsequenzen:**
- **`hit_frame` ist externe Bestätigung, kein neuer Befund an sich**: selbst
  FineBadmintons eigene Pipeline braucht einen EINZELNEN präzisen Hit-Frame, nicht nur
  ein Fenster — stützt unser G0-Design (±50 ms) und den eingebrannten Marker (Memo §9,
  Stufe 0) zusätzlich.
- **`hit_outcomes` validiert unser Stufe-4/G4-Design**, ändert nichts daran: kleines
  Vokabular, nur auf dem letzten Schlag einer Rally — exakt unser Landing-in/aus-Ansatz.
- **`strategies` vs. `player_actions` ist ein echter Schema-Vorschlag für den v2-Spec**:
  Technik (Vor-/Rückhand) und taktische Absicht (passiv/defensiv/Antizipation) als ZWEI
  getrennte Felder statt eines gemischten Arrays — ermöglicht saubere Stats wie „wie oft
  passiv gespielt" pro Spieler. Aktualisiert F3 unten.
- **Ehrlicher Vorbehalt**: diese drei Felder sind NICHT im 20 757-Hit-Bulk-Datensatz
  enthalten — nur als Schema-Vorbild nutzbar, nicht als Pretraining-Masse für genau diese
  drei Felder (wohl aber weiterhin für Typ/Subtyp/Ball-Area/Player-Actions/Shot-Characteristics,
  die IM Bulk-Datensatz stecken).

## 2. Daten-Wert für uns

### 2.1 Der unterschätzte Hauptwert: die Videos SHIPPEN

ShuttleSet liefert nur CSVs + YouTube-Links (Rot-Risiko); FineBadminton liefert die
70 Matches als Dateien. Das eröffnet einen Weg, den Checkpoint-Transfer nicht bietet:
**wir können unsere EIGENE Pipeline (feingetunter TrackNetV3 + RTMPose + Homographie)
über 70 Profi-Matches laufen lassen und 20 757 gelabelte Strokes in UNSEREM
Feature-Raum erzeugen** — Pretraining ohne den Normierungs-/fps-Mismatch, der beim
Laden fremder Checkpoints droht (v1s Transfer-Bremse, Memo-Risiko 4). Zusätzlich:

- **`start_frame`/`end_frame` je Hit = 20 757 Hit-Fenster** → Trainingsdaten für unseren
  G0-Hit-Detektor (HitNet-Muster) — die Ressource, die MonoTrack nicht mitliefert
  (Code ohne Gewichte). Vorsicht: 25-fps-Broadcast-Einzel, Domain-Gap bleibt.
- **Ball-Area (9 Zonen, 3×3-Raster, selbst ausgezählt)** → Platzierungs-Vokabular +
  schwaches Landing-Signal.
- Die Round-Frame-Ranges ersparen eigene Rally-Segmentierung der Fremd-Videos.

### 2.2 Taxonomie-Crosswalk 11 Typen ↔ ShuttleSet-18 ↔ unser grob-8

Die 11 Typen bauen laut Paper auf dem **ShuttleSet22-Framework** auf (+ Experten-Input)
— die Zuordnung zu unserem grob-8 ist fast verlustfrei (eigene Zählung, n=20 757):

| FineBadminton (n) | → unser grob-8 |
|---|---|
| serve 2 065 (short/high/flick serve) | Aufschlag |
| clear 1 368 (attacking clear) | Clear |
| kill 3 015 (jump/full/slice/common/stick smash!) | Smash |
| drop shot 1 372 (slice/reverse-slice/stop/blocked) | Drop |
| drive 1 816 (flat/high drive) | Drive |
| net shot 2 274 + cross-court net 678 + net kill 460 | Netzspiel (Netzdrop/Cross/Töten) |
| net lift 705 + push shot 4 734 (Subtyp „flat lift" 3 225) | Lift bzw. Netzspiel/Push — **einzige unscharfe Kante**: deren „push shot" mischt unser Push und flache Lifts; beim Mapping Subtyp beachten |
| block 2 243 (high block) | **Abwehr (Block)** — stützt die Abwehr-Umgruppierung aus Memo-Entscheidung 1 |

„stick smash" als eigener Subtyp (219×) bestätigt nebenbei die Stick-Smash-Korrektur.
**Konsequenz: FineBadminton kann unseren grob-8-Kopf DIREKT pretrainen** (nach
Subtyp-bewusstem Mapping), als zweites Standbein neben den BST-Checkpoints.

### 2.3 Grenzen

Nur Einzel (Klassen-Prioren im Doppel anders), Broadcast-Kamera (bewegt, hinter dem
Feld — unsere ist fix erhöht), 25 fps, keine Doppel-Zuordnungshilfe, kein Amateur-Material,
Audio = Broadcast-Kommentar (für unser Audio-Thema wertlos). Der Domain-Gap aus dem Memo
(§3) bleibt bestehen — FineBadminton verbessert die DATEN-Seite, nicht die Domain-Seite.

## 3. Funktionalitäts-Mapping (was dir gefällt → unser Produkt)

Kern-Einsicht, jetzt am Paper VERIFIZIERT (§3.2): **das LLM klassifiziert kein Video.**
Deren Pipeline ist dreistufig — (1) Structural Parsing mit klassischen CV-Modellen
(TrackNetV3 — derselbe Tracker wie unserer! —, FastRCNN, VideoMAE-Hit-Detektor), (2) MLLM
beschreibt jeden Stroke aus strukturierten Daten + nur 3 Frames, (3) ein Refining-LLM
glättet im Rally-Kontext, ein Summarizing-LLM erzeugt Scoring/Losing/Overall — und alles
wird von 2 menschlichen Annotatoren geprüft. Genau das Muster, das zu unserem
Honest-ML-Ansatz passt: **CV liefert Fakten, LLM formuliert, Mensch/Konfidenz gate-t.**

| FineBadminton-Feature | Bei uns | Tier |
|---|---|---|
| Round-Splitting, Hit-Zählung, Round-Picker-UI | **haben wir** (Rally-Segmentierung, shot_count, Rally-Review); deren Demo validiert unsere geplante Timeline-Strip + Klick-zu-Rally-UX | vorhanden |
| Hit-Karte Ebene 1 (Typ/Subtyp/Hitter/Ball-Area) | = unsere Stufen 1–3 + Landing; **Ball-Area-9-Raster übernehmen** (billig: Landing/Shuttle-Position → 3×3-Zone) | Spec-Kandidat, günstig |
| Hit-Karte Ebene 2 (straight/cross-court, over head) | **rein geometrisch aus unserem Shuttle-Track ableitbar** (Trajektorien-Vektor); Vor-/Rückhand aus Pose = Schätzung. **Aktualisiert (§1a): als zwei getrennte Felder führen** — `player_actions` (Technik) und `strategies` (Absicht: passiv/defensiv/Antizipation) statt eines Mischarrays; Intent-Tags bleiben Schätzungs-Ebene, aber als eigenes Feld statt LLM-Freitext | Spec-Kandidat: Geometrie + technik/absicht-Split ja, Intent-Werte = Schätzungs-Ebene |
| Hit-Karte Ebene 3 (Quality 1–7 + Kommentar) | subjektiv; bei denen LLM-Vorschlag + Mensch. Bei uns: optionale „Schätzung"-Spalte, im Review korrigierbar — NICHT gate-fähig | optional, Schätzung |
| **Rally-Text (Scoring/Losing/Overall Reason)** | **empfohlen als neues Produkt-Feature**: deutscher Rally-Kommentar, generiert aus unseren Stroke-Sequenzen + Outcomes (Struktur→Text, kein Video-Verständnis), klar als KI-Text markiert, sinnvoll erst wenn G2 (Klassen) + G4 (Outcomes) die Fakten liefern — sonst eloquenter Unsinn über falsche Daten. LLM-Wahl offen: lokal (Qwen3 auf dem Studio) vs. Claude-API — **deine Entscheidung, dokumentiert als offene Frage** | neues Feature, nach G2/G4 |
| VLM-Video-Verständnis als Analysepfad | bleibt verworfen — deren eigener Benchmark: Gemini 2.5 Pro 38,6 %, GPT-4.1 32,4 %; selbst ein AUF FineBadminton feingetuntes Qwen2.5-VL-7B erreicht nur 42,1 % | verworfen (bestätigt) |

## 4. Architektur-Deltas (bringt deren Aufbau uns Verbesserungen?)

1. **Validierung, keine Revolution**: deren Structural-Parsing-Stufe entspricht unserer
   Pipeline (sogar derselbe Shuttle-Tracker); deren Hit-Detektor (VideoMAE-Features +
   Attention-Pooling + Temporal-Transformer → Frame-weise Hit-Wahrscheinlichkeit) ist die
   schwere Video-Variante unseres geplanten leichten Pose+Shuttle-GRU. Wir bleiben beim
   GRU (nutzt unsere verifizierten Inputs, ~16k Parameter, MPS-freundlich) — aber ihre
   Wahl bestätigt: gelernte Hit-Erkennung, nicht Heuristik.
2. **Neuer Hebel fürs Pretraining (Empfehlung)**: FineBadminton durch UNSERE Pipeline
   verarbeiten → grob-8-Pretraining + Hit-Detektor-Pretraining im eigenen Feature-Raum
   (§2.1). Aufwand: 43-GB-Download + ~70 Matches × Pipeline-Lauf (Batch, unbeaufsichtigt).
3. **VLM-Pilot-Verbesserung (Empfehlung, direkt übernehmbar)**: deren Hit-Centric
   Keyframe Selection verbesserte JEDES Modell gegenüber uniformem Sampling — wir kennen
   die Hit-Zeit exakt, also Frames um den Hit herum dicht sampeln statt gleichmäßig.
   Passt zum bestehenden Pilot-Design (16–32 Frames, mlx-vlm).
4. **Labeling-Pipeline**: „MLLM-Vorschlag + Mensch verfeinert" = exakt unser
   GBDT-Prelabel + Label-Studio-Muster aus v1; keine Änderung nötig. Ihre
   Qualitätssicherung (2 unabhängige Annotatoren + Senior-Konsolidierung) ist für einen
   Verein unrealistisch — unser Ersatz bleibt: Konfidenz-Gates + Review-Korrekturen.
5. **Nicht übernehmen**: FBBench-artige MCQ-Evaluation (für uns ohne Nutzen, wir messen
   direkt gegen Labels); Quality-Sterne als Trainingsziel (subjektiv, nicht gate-fähig).

## 5. Empfehlungs-Checkliste (zum Abhaken für den v2-Spec)

| # | Vorschlag | Kosten | Empfehlung |
|---|---|---|---|
| F1 | FineBadminton-20K als **zweite Pretrain-Quelle**, durch unsere Pipeline verarbeitet (grob-8 + Hit-Detektor) | 43 GB + Batch-Compute | **ja** (Phase 2/3) |
| F2 | **Ball-Area-3×3-Raster** als abgeleitetes Feld je Stroke (aus Shuttle-Track) | klein | **ja** |
| F3 | **Geometrische Shot-Characteristics** (straight/cross-court) je Stroke, PLUS technik/absicht als getrennte Felder (`player_actions`/`strategies` statt Mischarray — bestätigt echtes Schema, §1a) | klein | **ja** |
| F8 | **`hit_frame` als eigenes Konzept ins v2-Spec übernehmen**: unser Hit-Detektor liefert bereits einen einzelnen Frame (kein Fenster) — FineBadmintons 100-%-Coverage bestätigt, dass das der richtige Detailgrad ist, keine neue Arbeit | ~0 | **ja, als Design-Bestätigung dokumentieren** |
| F4 | **LLM-Rally-Kommentar deutsch** (Scoring/Losing/Overall) aus Stroke-Sequenz + Outcomes, „Schätzung"-Kennzeichnung, Review-korrigierbar | mittel; erst nach G2/G4 | **ja, als Post-Gate-Feature**; LLM-Wahl (lokal vs. Claude-API) = deine Entscheidung |
| F5 | Per-Hit-**Quality-Sterne** (Schätzung, 5er- oder 7er-Skala) | klein–mittel | optional — nur wenn du es als Trainer nützlich findest |
| F6 | VLM-Pilot: **Hit-zentriertes Frame-Sampling** übernehmen | ~0 | **ja** |
| F7 | Vor-/Rückhand-Schätzung aus Pose (Ebene 2) | mittel, ungetestet | später prüfen (Phase-1-Messung möglich) |

## 6. Quellen dieses Nachtrags

Alle direkt geladen 2026-07-13 (Einzel-Prüfung im Haupt-Loop, keine 3-Stimmen-Welle —
Zahlen aus den JSONs sind eigene Auszählungen und insofern härter als Zitate):
arxiv.org/abs/2508.07554 + /html/2508.07554v1 (Paper, ACM MM 2025, S. 12776–12783) ·
github.com/iLearn-Lab/MM25-FineBadminton (Apache-2.0-LICENSE, README) ·
ilearn-lab.github.io/MM25-FineBadminton (Demo) ·
huggingface.co/datasets/iLearn-Lab/{Finebadminton-20K, FineBadmintonBenchmark}
(HF-API: gated=false, license apache-2.0, Dateibäume; Spiegel unter Moujuruo/…) ·
70 Annotations-JSONs + 2 Benchmark-JSONs (Stichproben) lokal geparst.
