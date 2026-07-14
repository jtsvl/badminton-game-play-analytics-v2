# Schlagklassifikation v2 — Phase-0 Research-Memo

**Datum:** 2026-07-13 · **Status: ZUR FREIGABE** · Grundlage: `docs/prompts/2026-07-13-stroke-v2-kickoff.md`

> §1 fasst alles zusammen und nennt die drei Entscheidungen, die du treffen musst.
> Der Rest ist Beleg-Material. Quellen mit Prüfstatus: `sources.md` · Taxonomie-Tabellen:
> `taxonomy-crosswalk.md`.

---

## 1. Zusammenfassung & erbetene Entscheidungen

**Was die Recherche ergab, in vier Sätzen:** Die Modellfrage ist gelöst und günstig für uns —
das beste veröffentlichte Badminton-Schlagmodell (BST, CVPRW 2026) nutzt exakt die drei
Eingaben, die unsere Pipeline schon produziert (RTMPose-Pose, TrackNetV3-Shuttle,
Court-Homographie), ist MIT-lizenziert, und seine ShuttleSet-Checkpoints sind heute real
herunterladbar. Die ehrliche Nachricht: selbst SOTA erreicht auf sauberem Profi-Material mit
~33 000 Labels nur 0.83–0.85 (25–31 Klassen) und auf der feinen 18er-Taxonomie nur 0.65 —
**mit** Datenleck; unser 0.70-Ziel ist auf fein-18 unerreichbar und wird nur über grob-8 +
Transfer + mehr eigene Labels realistisch. Hit-Erkennung und Doppel-Zuordnung sind die
eigentlichen Baustellen: die Literatur ist dort dünn (Doppel-Zuordnung: EIN publiziertes
Ergebnis, 0.67), aber sie zeigt eine klare Architektur — gelernter Video-Hit-Detektor
(HitNet-Muster) plus Audio als Timing-Verfeinerung *innerhalb* von Video-Kandidatenfenstern,
nie als eigenständiger Detektor. Für VLM-Vorlabeling ist Qwen3-VL lokal technisch machbar
(echter Video-Modus in mlx-vlm und llama.cpp, 30B-A3B passt in 64 GB), aber alle gemessenen
VLM-Ergebnisse auf feinkörnigem Sport liegen bei 0.32–0.42 — der Pilot bleibt sinnvoll,
die Erwartung ist „Labeling-Hilfe", nicht „Ersatz".

**Entscheidung 1 — Taxonomie-Korrekturen (Details §7 + `taxonomy-crosswalk.md`):**
Die 18er-Feintaxonomie ist bestätigt ShuttleSets natives Vokabular (der Widerspruch im
v1-Spec ist widerlegt). Vier Korrekturen an den deutschen Labels werden vorgeschlagen:
1. „Handgelenk-Smash" → **„Stick-Smash"** (DBV-Begriff; „Handgelenk-Smash" existiert in
   keiner deutschen Quelle).
2. „Rush/Netzkill" → **„Töten"** (DBV-Begriff).
3. „langer Aufschlag" → **„hoher Aufschlag"** (kanonisch); Flick-Aufschlag heißt „Swip".
4. Die drei Abwehrschläge unter EINE grobe Klasse **„Abwehr"** gruppieren, mit DBVs eigenem
   Schirm „Doppelabwehr": **Konter** (defensive return drive), **hohe Abwehr** (defensive
   return lob), **Block** (return net). Das behebt auch die Struktur-Schwäche, dass „Abwehr"
   bisher nur ein einziges Mitglied hatte.
   Außerdem: „flacher Treibball" kollidiert mit dem deutschen „Treibball" (= Drive);
   Empfehlung: driven flight (nur 68 von 36 484 ShuttleSet-Schlägen!) in Drive aufgehen
   lassen statt eigenes Fein-Label.

**Entscheidung 2 — Architektur-Richtung (Details §9):** vierstufig, Datenstrategie zuerst:
(a) 30-fps-Re-Ingest + Neu-Labeling unter korrigierter Taxonomie; (b) gelernter
Hit-Detektor nach dem HitNet-Muster (Pose+Shuttle+Court → GRU) mit Audio-Onset-Verfeinerung
im Kandidatenfenster; (c) Zuordnung als 5-Wege-Kopf desselben Detektors (4 Spieler + kein
Hit) mit „Unsicher" als erstklassigem Ausgang; (d) Schlagklassifikation grob-8 mit
BST-4-Slot ab ShuttleSet-Checkpoint, GBDT als Baseline und Labeling-Assist. Die
BadmintonDB-trainierten Checkpoints sind per deinem Override (2026-07-13, Ausnahme von
Locked Decision 6) ebenfalls nutzbar — grob-8 statt fein-18 bleibt trotzdem die Empfehlung,
aus Genauigkeitsgründen (fein-18 = 0.65 selbst mit Leakage), nicht aus Lizenzgründen.
Gates je Stufe (§12).

**Entscheidung 3 — VLM-Pilot-Gate (Details §8):** Pilot wie geplant (~100 Clips,
prä-registriert), aber mit korrigierter Erwartung: Adoptions-Gate Makro-Übereinstimmung
≥0.60 UND keine grobe Klasse <0.30 UND messbare Zeitersparnis; Abbruch bei <0.5 auf den
ersten 30 Clips. Erwartung laut Literatur: deutlich unter Gate — der Pilot ist billig und
klärt es endgültig.

**Ehrliche Genauigkeits-Erwartung (§10, Schätzung):** Session 1 (~700 Labels): grob-8
LOGO-CV ~0.55–0.65 — unter Gate, nur Labeling-Assist. Session 2 (~1 400, echtes
Cross-Session-Eval): ~0.60–0.70. Session 3 (~2 100): ~0.65–0.75. Das 0.70-Gate ist
frühestens ab Session 2 realistisch, mit allen Hebeln (präzise Hit-Zeiten, Pretrain,
Imbalance-Behandlung, ggf. SSL auf ungelabelten Schlägen).

**Verifikations-Hinweis (Transparenz):** Die Verifikation ist vollständig — die beim
ersten Lauf am Budget-Limit ausgefallene Welle wurde nachgeholt (107/108 Prüf-Agenten
erfolgreich; bei einer DBV-Behauptung entschieden 2 statt 3 Stimmen). Ergebnis über alle
nachgeprüften tragenden Behauptungen: 19 von 24 bestätigt 3:0; die 5 „widerlegten" waren
Detail-Korrekturen, kein Umsturz — die drei relevanten sind eingearbeitet: (a) HitNets
F1 0.946 gehört zur Nachverarbeitungsstufe, deren „abwechselnde-Schläger"-Regel
Einzel-Logik ist (roh: F1 0.866, §5/§6); (b) mlx-vlm verwirft die fps-Metadaten vor dem
Qwen3-VL-Prozessor (§8); (c) llama.cpp dekodiert Video standardmäßig mit FIX 4 fps (§8).

---

## 2. Methode & Verlässlichkeit der Quellen

Zwei mehrstufige Recherche-Durchläufe (2026-07-13):

- **Durchlauf 1** (generischer Deep-Research-Harness, 108 Agenten): 6 Suchwinkel, 25
  Primärquellen geladen, 125 falsifizierbare Behauptungen extrahiert, 25 adversarial
  verifiziert (je 3 unabhängige Prüf-Agenten, 2/3-Mehrheit). 24 bestätigt, 1 widerlegt.
- **Durchlauf 2** (gezielter Lücken-Harness): je ein Tiefen-Rechercheur für sechs offene
  Stränge, danach 3-Stimmen-Verifikation je tragender Behauptung. Die Welle fiel zunächst
  dem Monats-Budget-Limit zum Opfer und wurde per Workflow-Resume vollständig nachgeholt
  (107/108 Prüfer erfolgreich; eine DBV-Behauptung mit 2:0 statt 3:0 entschieden). Alle
  „Widerlegungen" beider Läufe waren Detail-Korrekturen — jeweils gegen die Prüfer-
  Korrekturtexte abgeglichen und eingearbeitet.

Kennzeichnung: **[verifiziert]** = 3-Stimmen-geprüft (bzw. 2:0) · **[Schätzung]** =
Synthese-Schluss, keine Messung. Negativ-Befunde („die Literatur schweigt zu X") sind
eigene Ergebnisse und als solche aufgeführt — sie bedeuten: dort sind WIR die Ersten, mit
allen Konsequenzen.

Beim Lesen der Zahlen: Profi-Broadcast-Ergebnisse übertragen sich nicht direkt auf
Amateur-Doppel mit fester Hallenkamera; wo ein Paper Trainings- und Testdaten aus demselben
Match mischt („same-match leakage"), ist die Zahl geschönt — jeweils vermerkt.

## 3. Modell-Landschaft (Thema A) — vollständig verifiziert

| Modell | Bestes verifiziertes Ergebnis | Protokoll | Gewichte? | Lizenz |
|---|---|---|---|---|
| **BST** (BST-CG-AP, CVPRW 2026) | 0.832 top-1 / 0.810 macro-F1, 25 Kl. | ShuttleSet ~33.5k Schläge, match-getrennt (leak-frei) | **ja, live geprüft** | Repo MIT |
| **TemPose-TF** (CVPRW 2023, peer-reviewed) | 0.907 auf 13 Kl. (Badminton Olympics) | leak-frei, Testset = 1 Video | nicht verifiziert | — |
| TemPose-TF-ASF (Preprint 2026) | 0.854 / 0.761, 31 Kl. | ShuttleSet, match-getrennt | nicht verifiziert | — |
| Reine Skelett-GCNs (BlockGCN, SkateFormer) | 0.79–0.81 | dito | teils | — |

Tragende Fakten (alle [verifiziert]):

- **BST-Gewichte real ladbar** (Stand 2026-07-13): Drive-Ordner „weight" = ALLE
  ShuttleSet-Checkpoints (35-Klassen plain, 25-Klassen mit Suffix `_merged`); Ordner
  „weight_on_badDB" = BadmintonDB-trainiert (18 Klassen, 10 Seeds) — per User-Override
  nutzbar (§4), als zusätzliche Transfer-Quelle geführt. Ehrlicher Vorbehalt: unter
  8:1:1-Leakage-Split auf 8 Matches zweier Spieler trainiert; ihr Transfer-Wert auf fremdes
  Material ist ungetestet und vermutlich unter dem der ShuttleSet-Checkpoints (match-getrennt,
  4–5× mehr Daten). Repo unverändertes MIT; ShuttleSet (MIT) legt abgeleiteten Modellen
  nichts auf.
- **Eingaben = unsere Pipeline**: 2D-Pose (RTMPose!), TrackNetV3-Shuttle, Spielerposition
  via Homographie; Keypoint-Normierung über BBox-Diagonale; Sequenzlänge 30 Frames
  (ShuttleSet-Variante). COCO-17-vs-Halpe-26-Layout steht nicht im Paper → lokaler
  Smoke-Test nötig. Hit-Frames kommen aus Annotationen — **BST löst weder Hit-Erkennung
  noch Zuordnung** (verweist selbst auf HitNet, §5).
- **BST ist strikt Einzel** (Interaktionsmodul = genau 2 Posen; Doppel = Future Work).
  Unsere v1-4-Slot-Verallgemeinerung bleibt eigene Vorleistung (89 % der Parameter mappen,
  gemessen +7–12 Punkte Transfer).
- **Fusion ist tragend**: Pose+Shuttle+Position schlägt Pose-only um ~5 Punkte (TemPose
  0.856→0.907; GCNs 0.79–0.81 vs. Fusion 0.83–0.85). BSTs Shuttle-Strom-Ablation: +4.3
  Punkte. Unser feingetunter TrackNetV3-Track ist also ein echter Vorteil.
- **Modellwahl ist NICHT der große Hebel**: BST vs. TemPose-TF < 1 Punkt, teils
  Clip-Schneide-Strategie. Datenstrategie dominiert — deckt sich mit v1.
- **Fein-18 selbst mit Leakage nur 0.65** (BST auf BadmintonDB-Protokoll) → trainiert wird
  grob-8, fein-18 ist Label-/Reporting-Ebene.
- **Label-Effizienz-Datenpunkt** (einziger im Rennsport): BST bei 25 % der
  ShuttleSet-Labels (~6–8k!): 0.825→0.692. Bei unseren 700–2 100 Labels liegt der
  Scratch-Fall weit darunter — Pretrain + eigene Hebel sind Pflicht (§10).

Lokale v1-Prioren zur Einordnung: BST4 scratch 0.400 / +Pretrain 0.490 / GBDT 0.536
(554 Labels, 12 fps, ±0.2 s Hit-Drift). Vorsicht: „Bäume schlagen Transformer" wurde unter
Drift gemessen; präzise Hit-Zeiten können die Rangfolge kippen (M4 misst das). v1s „keine
Konfidenz-Separation" stammt von einem memorierenden Modell ohne Kalibrierung — Abstention
ist ungetestet, nicht unmöglich.

## 4. Datensätze & Lizenzen (Thema B) — vollständig verifiziert

| Datensatz | Umfang | Lizenz | Verfügbar heute | Für uns |
|---|---|---|---|---|
| **ShuttleSet** | 44 Profi-EINZEL-Matches, **36 484 Schlagzeilen** (CSV), 18 Typen + unknown | **MIT** (Repo-weit) + Zitier-Bitte | ja (CSVs im Repo; Videos = YouTube-Links) | Pretrain-Anker ✔ |
| **ShuttleSet22** | 58 Matches (8×2021 + 50×2022), 52 356 Zeilen; offizieller Split 33 612 | MIT | ja; **keine Video-URLs** | Zusatz-Pretrain (Annotations-only) |
| **BadmintonDB** | 9 Momota–Ginting-Matches, ~10 native Typen | KEIN Lizenz-Grant (API: license=null, keine LICENSE-Datei) | ja | **nutzbar — User-Override 2026-07-13** (Ausnahme von Locked Decision 6; Restrisiko nur bei späterer Distribution). Öffnet auch die darauf trainierten BST-18-Checkpoints |
| VideoBadminton | 7 822 Clips, 18 Klassen, Uni-Team-Training (kein Broadcast!) | keine Lizenz-Datei | ja (Drive-Zip live) | **nutzbar — User-Override 2026-07-13** (dito). Realer Kandidat als Zusatz-Pretrain/Eval: näher an Amateur-Bedingungen als Broadcast; offene Frage: enthält das Zip Audio-Spuren? (wäre die einzige öffentliche Audio-Quelle) |
| **FineBadminton** (ACM MM'25) | 70 Matches (Videos SHIPPEN!, 43 GB) + 20 757 annotierte Hits, 2 066 Rounds, nur Einzel | **Apache-2.0** (HF-Card + Repo-LICENSE) | **ja — HF, frei** (Korrektur 2026-07-13: User fand `iLearn-Lab/Finebadminton-20K` + `FineBadmintonBenchmark`; Erstbefund „kein Download-Pfad" war falsch) | **zweite Pretrain-Quelle** — Details + Empfehlungs-Checkliste: `finebadminton-eval.md` |
| BFMD / Shot2Tactic (2025/26, enthalten Doppel) | — | — | kein Download-Pfad | beobachten |

Erstmals berechnet (kein publiziertes Verteilungs-Table existiert — direkt aus den geklonten
CSVs gezählt): ShuttleSet-Klassenverteilung ist extrem schief — net shot 6 290 · lob 5 331 ·
return net 3 620 · push 2 925 · clear 2 922 · smash 2 586 · drop 2 144 · short service
2 051 · wrist smash 1 648 · unknown 1 407 · cross-court net 1 371 · passive drop 1 356 ·
drive 700 · rush 512 · back-court drive 473 · def. return drive 406 · long service 373 ·
def. return lob 301 · **driven flight 68**. Sechs der 18 Klassen haben <520 Beispiele —
auch das Pretraining hat also dünne Ränder, und driven flight ist faktisch leer (stützt
Entscheidung 1).

**Negativ-Befunde [verifiziert]:** Es existiert NIRGENDS ein Datensatz mit (a)
Amateur-Material, (b) fester Einzelkamera, (c) Doppel oder (d) Audio-Annotationen. Unsere
eigenen Daten sind für alle vier Aspekte die einzige Quelle — was auch heißt: unser
Audio-Hit-Eval-Set und unsere Doppel-Zuordnungs-Messung wären je für sich neuartige Evidenz.

## 5. Hit-Erkennung: Video-Detektor + Audio-Verfeinerung (Thema C) — [verifiziert]

Kernbefund: **Die Literatur stützt Audio als Timing-Verfeinerung in video-basierten
Kandidatenfenstern — als eigenständiger Detektor bei Mikrofon-Distanz trägt Audio nicht.**

- Der einzige Fern-Mikrofon-Präzedenzfall (Broadcast-Tennis, AVSP 2011, DOPPEL), Zahlen
  3:0 gegen das Paper geprüft: Audio-only roh F nur 0.41–0.50; mit Konfidenz-Maß 0.68–0.77
  (≈ visuell allein: 0.68–0.71); erst die **A/V-Fusion gewinnt klar: 0.77–0.84**. Deren
  Fusions-Muster ist exakt unser Plan: Audio-Hit als Verteilung um den Video-Kandidaten,
  ±8 Frames Toleranz — die explizit auch die Schalllaufzeit zum Mikro absorbiert.
- Nahfeld-Referenz (Tischtennis, 2025): Sub-ms-Onset-Präzision, Recall ~0.95–1.0 — aber
  Richtmikro in 0.5–2 m, kontrollierte Aufnahmen. Nicht unsere Bedingungen; zeigt nur:
  WENN das Signal hörbar ist, ist Timing kein Problem.
- Generische Onset-Detektoren (SuperFlux, madmom-CNN) werden bei ±25 ms Toleranz mit
  F 0.82–0.90 evaluiert (auf MUSIK) — ±25 ms wäre ~8× besser als die heutige ±0.2 s-Drift.
- **HitNet (MonoTrack, CVPRW 2022) ist die Video-Detektor-Vorlage**: 2-Layer-GRU (~16k
  Parameter!) über 12-Frame-Fenster aus Court+Pose+Shuttle-Features, 3 Klassen (kein Hit /
  nah / fern), Broadcast-Einzel. Verifikations-Korrektur: **roh F1 0.866; die 0.946
  entstehen erst durch eine Optimierungs-Nachstufe** (erwartete Hit-Zahl, min. 0.5 s
  Abstand, **abwechselnde Schläger — Einzel-Logik!** Im Doppel gilt nur „abwechselnde
  Seiten"). Trainings-Code öffentlich, KEINE Gewichte, Lizenz ungeklärt → als Muster
  nachbauen, nicht importieren. Wichtig: nutzt Pose — fängt also auch flache Schläge, die
  unsere Vertikal-Reversal-Heuristik strukturell verpasst (deshalb ist v1s Klassenzählung
  mutmaßlich verzerrt).
- Werkzeug-Lizenzen alle kompatibel: librosa ISC · madmom BSD (Modelle CC BY-NC-SA — für
  uns ok) · PANNs MIT (Zenodo-Gewichte) · BEATs MIT · YAMNet Apache-2.0.

**Harte Randbedingung (User-Review 2026-07-13): Mehrfeld-Betrieb ist der NORMALFALL, nicht
das Risiko.** In der Halle laufen immer parallel Spiele auf Nachbarfeldern — deren Impacts
sind spektral identisch mit unseren. Konsequenzen fürs Design: (a) Audio detektiert NIE
eigenständig, nur innerhalb video-abgeleiteter Kandidatenfenster (deckt sich mit der
Literatur); (b) auch INNERHALB eines ±150-ms-Fensters kann zufällig ein Nachbarfeld-Onset
liegen — das G0-Eval zählt „falscher Onset gewählt" als eigenen Fehlermodus; (c)
Unterscheidungs-Merkmale nach Stärke: Koinzidenz mit unserem Shuttle-/Pose-Signal >
Lautstärke (Nachbarfelder sind weiter vom Mikro) > Stereo-Links/Rechts-Ungleichgewicht.

**Wo die Literatur schweigt (= wir messen selbst, M1/M2/M9):** Kamera-Mikro in halliger
Mehrfeld-Halle; Nachbarfeld-Impacts von eigenen unterscheiden (nächstes Analogon verlangt
Mikrofon-Arrays); Schalllaufzeit-Kompensation pro Hit (~40 ms übers Feld > 1 Frame @30fps);
DJI-A/V-Sync-Offset. Keine dieser Fragen hat eine publizierte Antwort — M1 (Audio-Onsets ×
738 bekannte Hits) beantwortet sie auf unserem Material in ~3 h, inklusive
Nachbarfeld-Onset-Rate (Onsets außerhalb eigener Rallies = Fremd-Rauschboden, denn unser
Feld ruht, die Nachbarn nicht) und Koinzidenz-Wahrscheinlichkeit im Kandidatenfenster.

## 6. Schlägerzuordnung im Doppel (Thema D) — [verifiziert]

- **Es gibt genau EIN publiziertes monokulares Doppel-Ergebnis** (arXiv 2508.13507, 2025):
  Einzel-trainiert, zero-shot auf 2 Profi-Doppel-Matches, Hitter = argmax über 4
  Spieler-Klassifikatoren → **0.67 Accuracy / 0.71 F1**. Das ist die gesamte Messlatte —
  niedrig. (Und die Metrik vermischt Hit-Erkennung mit Hitter-Wahl.)
- Bester Einzel-Präzedenzfall: HitNets 3-Wege-Kopf IST Zuordnung (nah/fern). Die
  naheliegende, nirgends publizierte Erweiterung: **5-Wege-Kopf (4 Spieler + kein Hit)**
  auf unseren Pose+Shuttle+Court-Features — Hit-Erkennung und Zuordnung in einem Modell.
- Heuristik-Vorlage (EventAnchor, CHI 2021): Richtungswechsel des Balls + minimale Distanz
  zu Hand-Keypoints (Fallback Hals bei Okklusion) — exakt unsere M6-Baseline gegen die
  ~554 vorhandenen hitter-Labels.
- Schläger-Detektion als Tiebreaker: RacketDB (VISAPP 2025, CC-BY-4.0!) ist verblüffend
  nah an unserem Setting (4K-Rückansicht, Amateur-DOPPEL) — aber Schläger im Mittel
  16×29 px, mAP50 nur 0.77, oft unsichtbar, keine Gewichte. Höchstens später als Zusatz-Cue.
- CoachAI-2023-Sieger verwarf Trajektorien-Richtungswechsel-Zählung ausdrücklich als zu
  verrauscht (nutzte X3D-Video-Klassifikation) — Bestätigung, die Reversal-Heuristik
  abzulösen.
- **Nirgends gemessen:** der fundamentale Ambiguitäts-Anteil aus einer Kamera
  (Netz-Situationen, Überlappung). Unsere eigene Messung auf den 738 Clips wäre neuartige
  Evidenz und dimensioniert den ehrlichen Abstention-Floor für G1.

## 7. Taxonomie (Thema F) — [verifiziert] (Kernfrage UND DBV-Mapping)

**Aufgelöst zugunsten des Kickoffs:** Die 18er-Feintaxonomie ist ShuttleSets natives
Vokabular (KDD-2023-Paper + Repo, 5 unabhängige Primärquellen; die Gegenaussage in
`specs/features/stroke-classification.md:46` wurde 0:3 widerlegt — im v2-Spec korrigieren).
Unsere Labels sind damit checkpoint-kompatibel; der Transfer-Pfad ist frei.

**DBV-Prüfung** (beide offiziellen PDFs vollständig extrahiert: Rahmentrainingskonzeption
2017, 50 S.; Bausteinsystem, 7 S.): Es gibt KEINE offizielle flache DBV-Schlagtaxonomie —
Begriffe stehen verstreut in Alters-Bausteinen und sind stark anglisiert (Stick-Smash,
Halfsmash, Push, Lift, Swip, Töten). Das validiert unser Vorgehen „ShuttleSet-Schema +
deutsche Glossen" und ergibt die vier Korrekturen aus Entscheidung 1. Vollständige
Kreuztabelle (18 fein ↔ DBV ↔ grob-8 ↔ ShuttleSet-Original inkl. Klassengrößen):
`taxonomy-crosswalk.md`. Kern:

| Vorschlag alt | Korrektur | Beleg |
|---|---|---|
| Handgelenk-Smash | **Stick-Smash** | DBV Bausteinsystem („Stick-Variationen"), mobilesport.ch, Ruhnow; „Handgelenk-Smash" in null Quellen |
| Rush/Netzkill | **Töten** | DBV Bausteinsystem wörtlich; „Netzkill" nirgends |
| langer Aufschlag | **hoher Aufschlag** | kanonisch (Wikibooks, sportunterricht.de); DBV: „kurzer RH-Aufschlag", Swip |
| Netz-Return / Abwehr-Lob / Abwehr-Drive | **Block / hohe Abwehr / Konter**, zusammen als grobe Klasse „Abwehr" | DBV: „Doppelabwehr (Konter/hohe Abwehr/aktiver Block)" |
| flacher Treibball (driven flight) | **in Drive aufgehen lassen** | „Treibball" = Drive im Deutschen (Brahms, Duden); ShuttleSet: 68/36 484 Schläge, 0 bei Damen |

Doppel-Hinweis: DBV behandelt Doppel über Taktik-Systeme (Angriffssysteme, „Schläge 1–4",
Annahmemuster mit Push), nicht über eigenes Schlagvokabular — grob-8 mit der
Abwehr-Umgruppierung deckt das ab. Offen (Phase 1): exakte ShuttleSet-Definitionen von
passive drop / return net aus Wang et al. 2022 (ACM TIST) ziehen, bevor die deutschen
Glossen einfrieren.

## 8. VLM-Vorlabeling: Qwen3-VL-Pilot (Thema E) — [verifiziert]

**Technisch machbar, Erwartung gedämpft — und die Nach-Verifikation fand zwei Fallen, die
den 0.267-Fehlschlag exakt reproduziert hätten:**

- **mlx-vlm** (seit v0.3.4) hat Qwen3-VL dense + MoE mit echtem Video-Pfad
  (`pixel_values_videos`, temporal_patch_size 2). **Verifikations-Korrektur:** die
  fps-Metadaten werden vor dem Qwen3-VL-Prozessor STILL VERWORFEN (kwargs-Filter gegen die
  Prozessor-Signatur, die keinen fps-Parameter hat) — das Timestamp-Alignment, mit dem
  Qwen3-VL wirbt, kommt dort also womöglich nie an. Außerdem: der DeepStack-Fix kam erst
  in v0.6.3 (nicht v0.6.2) → Pilot pinnt mlx-vlm ≥ 0.6.3 und prüft das Frame/Timestamp-
  Verhalten explizit als erste Pilot-Zelle.
- **llama.cpp**: Qwen3-VL seit 2025-10, Video-Input seit 2026-06 (PR #24269,
  ffmpeg-Subprozess; Audio-Spur wird verworfen; MTMD_VIDEO evtl. Compile-Flag).
  **Verifikations-Korrektur:** Video wird standardmäßig mit **FIX 4 fps** dekodiert
  (`fps_target = 4.0f` hartkodiert; kein CLI-Flag zum Ändern) — ein 1-s-Clip ergäbe ~4
  Frames: exakt der Fehlermodus des 0.267-Piloten. Ohne Quell-Build-Anpassung ist die
  llama.cpp-Route für unsere Kurz-Clips derzeit ungeeignet → **Pilot läuft über mlx-vlm**.
  Offizielle GGUF 30B-A3B: Q4_K_M 18.6 GB / Q8_0 32.5 GB — passt in 64 GB [verifiziert, 3:0].
- **Dieselbe Falle steckt auch in Qwen3-VLs eigenen Defaults**: fps=2, min. 4 Frames —
  der Pilot MUSS die Frame-Zahl explizit steuern (16–32 Frames pro Clip) und verifizieren,
  dass sie wirklich ankommt.
- **Übernahme aus FineBadminton (Nachtrag 2026-07-13):** Frames hit-zentriert sampeln
  statt gleichmäßig — deren „Hit-Centric Keyframe Selection" verbesserte jedes getestete
  Modell, und wir kennen die Hit-Zeit exakt (siehe `finebadminton-eval.md` §4.3).
- **Audio-Option existiert**: Qwen3-Omni-30B-A3B GGUFs (Q8_0+mmproj ≈ 34 GB) bzw.
  mlx-vlm-Modul — aber kein End-to-End-Beispiel „Video-Frames + Clip-Audio in einem
  Prompt"; wäre Neuland im Pilot (eine Pilot-Zelle wert, keine Voraussetzung).
- **Gemessene Realität**: FineBadminton (Profi-Einzel!): beste Modelle 32–42 % auf
  Badminton-MCQs, Aktions-Unterscheidung am schwersten; MotionBench: SOTA-VLMs <60 %.
  Kein Paper zeigt irgendwo VLM ≥0.70 auf einer Schlag-Taxonomie.

**Prä-registrierter Pilot (läuft erst nach Memo- UND Spec-Freigabe):** ~100 stratifizierte
Clips über die 8 groben Klassen (net/drive überabgetastet), mit dem eingebrannten
Hit-Marker aus Stufe 0 (Prompt: „klassifiziere den markierten Schlag" — löst das
Mehrfach-Schläge-im-Clip-Problem auch fürs VLM), Referenz = menschliche Fein-Labels,
Metriken = Makro-/Klassen-Übereinstimmung, Konfusionsstruktur, Abstention-Verhalten,
Wandzeit/Clip. **Gate: Makro ≥0.60 UND keine grobe Klasse <0.30 UND
gemessene Zeitersparnis >0** (20-Clip-Timing-Test mit/ohne Assist). **Abbruch: <0.5 auf den
ersten 30 Clips** — kein Prompt-Tuning-Marathon. Immer neben dem 0.267-Prior berichtet.
Realistische Rolle bei Gate-Pass: Vorsortierung/Triage im Labeling, nie Pipeline.

## 9. Empfohlene Architektur

**Prinzip (aus v1-Post-Mortem + Recherche): Datenstrategie vor Modellwahl; jede Stufe hat
eigenes Gate; Abstention ist überall erstklassig.**

**Stufe 0 — Datenfundament:** 30-fps-Re-Ingest (M8 verifiziert vorher fps-Empfindlichkeit
und Checkpoint-fps; Quelle ist 60 fps — 30 ist der saubere 2:1-Griff, 60 bleibt für
Hit-Frame-Verfeinerung möglich). Neu-Labeling aller 738 Events unter der korrigierten
Fein-Taxonomie inkl. 4-Wege-Hitter + „Unsicher"/„Kein Schlag" (Tooling aus v1 wiederverwenden:
`stroke_clips.py` mit Audio, Label-Studio-Config, `import_stroke_labels.py`).
**Anforderung aus dem User-Review (2026-07-13): der Ziel-Schlag muss im Clip eindeutig
markiert sein** — v1-Clips (±0.7 s) enthielten in schnellen Doppel-Wechseln oft 2–3
Schläge, und es war unklar, welcher gemeint ist. Lösung im Export (Re-Encode läuft ohnehin):
eingebrannter Marker am exakten Hit-Zeitpunkt — roter Rahmen-Blitz + Kreis an der
Shuttle-Pixelposition (px,py liegen im Shuttle-Track vor) — plus „Schlag N von M im
Ballwechsel" im Label-Studio-Task. M7 validiert das Marker-Design an 30–50 Clips, BEVOR
die 738 exportiert werden; derselbe Marker dient dem VLM-Pilot („klassifiziere den
markierten Schlag"). Session-1-Footage (`Game_1..6.mp4`, liegt ungelabelt auf der Platte)
früh anlabeln (~200 Schläge) → **echtes Cross-Session-Eval ab Tag 1 der Modellarbeit**.

**Stufe 1 — Hit-Erkennung v2:** HitNet-Muster nachgebaut auf unseren Artefakten: GRU/TCN
über (Pose aller Spieler + Shuttle + Court) @30 fps → Hit-Kandidaten; dann Audio-Onset
(SuperFlux-Klasse) im ±150-ms-Fenster als Timing-Verfeinerung, mit A/V-Offset- und
Schalllaufzeit-Kalibrierung (M9). Eigenes Eval-Set: ≥150 handannotierte Hits (M2-Erweiterung).
Erwartung [Schätzung]: Recall deutlich über der Reversal-Heuristik (Pose sieht flache
Schläge), Timing ±1 Frame @30 fps für den hörbaren Anteil. Gate G0.

**Stufe 2 — Zuordnung:** 5-Wege-Kopf (4 Quadranten-Slots + kein Hit) im selben Modell —
Hit und Hitter gemeinsam; M6-Baseline (nächstes Handgelenk zur Shuttle-Position) als
Untergrenze; „Unsicher" ab definierter Margin. Netz-Situationen separat ausgewiesen.
Referenz: die einzige publizierte Doppel-Zahl ist 0.67 — unsere Baseline könnte das bereits
schlagen. Gate G1.

**Stufe 3 — Schlagklassifikation:** BST-4-Slot (v1-Vorleistung) initialisiert vom
ShuttleSet-25-Checkpoint (MIT-sauber; `_merged`-Dateien), trainiert auf **grob-8**
(korrigierte Taxonomie) mit: logit-adjusted Loss für die dünnen Klassen [verifiziert
wirksam auf CIFAR-LT, bei uns zu messen], konservative Augmentierung (Seiten-Spiegelung +
leichte zeitliche Jitter; „extreme" Bundles schaden laut Literatur), Temperatur-Skalierung
auf OOF + max-softmax-Abstention, marginales Split-Conformal für Coverage-Aussagen
(klassen-konditional ist bei n=20–30 mathematisch nicht tragfähig [verifiziert]).
Optionaler Hebel mit Literatur-Rückenwind: SSL-Pretraining (SkeletonMAE-artig) auf unseren
Tausenden ungelabelten Schlägen, gestapelt auf den ShuttleSet-Init — bei NTU recovert 10 %
Labels damit 84–96 % der Vollleistung [verifiziert, aber Same-Domain-Bedingungen]. Seit dem
Lizenz-Override zusätzlich verfügbar: VideoBadminton (7 822 Uni-Trainings-Clips) als
Zusatz-Pretrain/Zwischendomäne und die BadmintonDB-BST-Checkpoints als Zweit-Init zum
Vergleich. **Stärkster neuer Hebel (Nachtrag 2026-07-13): FineBadminton-20K** — 70
Profi-Matches mit gelieferten Videos + 20 757 Hit-Annotationen inkl. Hit-Frame-Fenstern
(Apache-2.0): durch UNSERE Pipeline verarbeitet ergibt das grob-8- und
Hit-Detektor-Pretraining im eigenen Feature-Raum, ohne Checkpoint-Normierungs-Mismatch
(`finebadminton-eval.md` §2). GBDT bleibt Baseline in jedem Report + Labeling-Assist
(`labeling_assist_only`). Gates G2+G3.

**Stufe 4 — Outcomes:** Rally-End-Schlag × Landing (in/aus/unsicher) × winner_side ×
Score-Progressions-Konsistenz (M5-Baseline). Netzroller: descoped auf Review-Tag, außer M10
findet ein reales Signal. forced/unforced dauerhaft „Schätzung" mit Review-Override. Gate G4.

**UI** (nach Gates, Muster liegen bereit): Event-Marker-Overlay als vierter
SVG-Layer neben `SkeletonOverlay` (Events-Tabelle passt schon), Rally-Timeline unter dem
Video am `vp.time/seek`-Kontrakt, Stroke-Review als Klon des RallyReview-Quartetts mit
append-only `stroke_corrections` (0011-Muster), Stats über `ShotMixCard`/`SHOT_DE`.

**Verworfene Alternativen:** Video-native Klassifikatoren (X3D/VideoMAE) — keine
Badminton-Gewichte, MPS-Risiko, und Fusion-Transformer schlagen sie auf ShuttleSet;
Audio-standalone-Hit-Detektor — von der Fern-Mikro-Literatur direkt widerlegt;
VLM als primärer Labeler — 0.267 gemessen, Literatur-Decke 0.42. (BadmintonDB-Checkpoints
sind seit dem User-Override nicht mehr ausgeschlossen — nur nachrangig: Leakage-Training,
kleinere Datenbasis; als Zweit-Init in Stufe 3 einen Versuch wert.)

## 10. Ehrliche Genauigkeits-Erwartung (1–3 Sessions) — [Schätzung mit Ankern]

Verifizierte Anker: SOTA-Decke 0.83–0.85 bei ~33k Profi-Einzel-Labels · BST bei 25 % der
Labels (~6–8k): 0.69 · fein-18 mit Leakage: 0.65 · v1 lokal: 0.49 (Pretrain, 554 Labels,
unter Hit-Drift) · NTU-Semi-Supervised: 1 % Labels + gutes Pretraining recovert 58–90 %.

| Stand | grob-8, realistisch (LOGO-CV bzw. Cross-Session) | Einordnung |
|---|---|---|
| Session 1 (~700 Labels) | **~0.55–0.65** | unter Gate; Labeling-Assist-Carve-out |
| Session 2 (~1 400, echtes Cross-Session) | **~0.60–0.70** | Gate-Nähe NUR mit allen Hebeln (G0-Timing, Pretrain, Imbalance, ggf. SSL) |
| Session 3 (~2 100) | **~0.65–0.75** | Gate-Pass plausibel, nicht garantiert |

Dünne Klassen bleiben der Engpass: net n=20 / drive n=22 in v1 sind mutmaßlich
Detektor-Artefakte (M2 klärt das) — ein besserer Hit-Detektor VERGRÖSSERT vermutlich genau
diese Klassen. M3 (Lernkurve) ersetzt diese Schätzspanne durch eine gemessene Steigung,
BEVOR das Spec Zahlen einfriert. Same-Session-LOGO-CV überzeichnet grundsätzlich —
Cross-Session ist ab Session 2 Pflichtmaßstab.

## 11. Phase-1-Messplan (vor jedem Modell-Training)

Beschlossen beim Planungs-Checkpoint: Phase 0 blieb strikt Recherche; diese Wegwerf-
Messungen (Scratchpad, kein Produktions-Code) sind die ersten Schritte nach Freigabe:

| # | Messung | Beantwortet | Aufwand |
|---|---|---|---|
| M1 | Audio-Onsets × 738 bekannte Hits (1 Spiel) | Timing-Präzision, **Nachbarfeld-Onset-Rate + Koinzidenz-Wahrscheinlichkeit im Kandidatenfenster** (Mehrfeld-Betrieb ist Normalfall), erste Recall-Schätzung | ~3 h |
| M2 | Manuelle Hit-Zählung, 10–15 Rallies @0.25× (Trainer) | echter Heuristik-Recall je Schlagtyp; Basis des G0-Eval-Sets | 1–2 h |
| M3 | GBDT-Lernkurve (140/280/554, vorhandener Harness) | Punkte/Label-Verdopplung → G2-Zielzahl | 2–3 h |
| M4 | Hit-Zeit-Jitter-Sensitivität | Wert präziser Hit-Zeiten; GBDT-vs-BST-Rangfolge | 1–2 h |
| M5 | winner_side vs. Spielstand (1 Spiel) | Outcome-Baseline | ~1 h |
| M6 | Hitter-Baseline (nächstes Handgelenk) vs. ~554 hitter-Labels | braucht Zuordnung ein Modell? | 1–2 h |
| M7 | Fein-Labeling-Probelauf 30–50 Clips, getimt (Trainer) | Clips/h, Unsicher-Quote, Taxonomie-Praxistauglichkeit, **Validierung des eingebrannten Hit-Markers** (User-Review: Ziel-Schlag muss eindeutig sein) | ~2 h |
| M8 | Smash @12 vs. 60 fps + Checkpoint-fps | 30-vs-60-Entscheidung; fps-Mismatch als v1-Transfer-Bremse? | ~2 h |
| M9 | A/V-Sync-Kalibrierung (mit M1) | Mux-Offset + Schalllaufzeit | ~1 h |
| M10 | Netzroller-Evidenz (5–10 bekannte Fälle) | Scope oder Descope | 1–2 h |

## 12. Gate-Struktur (ersetzt v1s einzelnes 0.70-Gate)

v1s Gate war strukturell kaputt: drop+lift+clear = 71 % der Labels — Kopfklassen-Treffer
bestehen 0.70 bei net/drive-Recall 0.00. „Bestanden" heißt: ein Trainer wird von der
Tabelle nicht systematisch in die Irre geführt.

- **G0 Hit-Timing** (gate-t alles): Recall ≥0.90 UND Präzision ≥0.90 bei ±50 ms auf
  handannotiertem Eval-Set (≥150 Hits, ≥2 Spiele); Median |Δt| ≤25 ms; „falscher Onset
  gewählt" (Nachbarfeld-Koinzidenz im Kandidatenfenster) wird separat ausgewiesen.
- **G1 Zuordnung**: Coverage ≥0.80 bei Genauigkeit ≥0.85; Netz-Situationen separat.
- **G2 Schlagklasse**: Makro-Recall ≥0.55 UND jede Klasse mit n≥30: Recall ≥0.40 UND
  Train-Eval-Lücke ≤15 Punkte (Memorisierungs-Alarm). Cross-Session Pflicht ab Session 2.
  Gesamtziel-Zahl nach M3 fixiert.
- **G3 Abstention**: Coverage ≥0.60, Covered-Accuracy ≥0.80, Abstention informativ
  (Genauigkeit auf abstinenten Schlägen ≥15 Punkte unter Covered-Accuracy), ECE ≤0.10 nach
  Temperatur-Skalierung. Coverage-Aussagen via marginalem Conformal (n≈550–700: ±3–5 Punkte
  realisierte Schwankung — ausgewiesen).
- **G4 Outcomes**: Winner/Error ≥0.80 auf Rally-Endschlägen (mit Abstention); Netz/Aus nur
  über Landing-Konfidenz-Floor; forced/unforced nie gate-pflichtig (dauerhaft Schätzung);
  Score-Konsistenz je Spiel als Dauer-Regressionssignal.
- **G5 Trainer-Nutzen**: gemessene Korrekturrate im Review auf frischem Spiel vor jeder
  Auto-Accept-Stufe.
- **Teil-Auslieferung**: Hit-Marker + Timeline ab G0 · Labels unterhalb G2 nur
  Labeling-Assist · Outcomes unterhalb G4 nur winner_side.

## 13. Risiken & offene Fragen

1. **Audio-SNR in der Halle ungemessen** — M1 zuerst; wenn Impacts unhörbar sind, bleibt
   der Video-Detektor allein (immer noch besser als die Reversal-Heuristik) und G0s
   ±50 ms wackelt. Verschärfung aus dem User-Review: Nachbarfelder spielen IMMER — auch
   im Kandidatenfenster kann ein fremder Onset liegen; „falscher Onset gewählt" ist ein
   eigener G0-Fehlermodus, M1 beziffert seine Häufigkeit.
2. **Zuordnungs-Ambiguitätsrate unbekannt** — Literatur schweigt; eigene Messung nötig.
   Wenn >20 % der Doppel-Hits fundamental ambig sind, muss G1s Coverage-Ziel runter
   (ehrlich statt geraten).
3. **Klassen-Prior-Verschiebung**: v1-Klassenzählung ist detektor-verzerrt; besserer
   Recall verschiebt die Verteilung — Stratifizierung und Gates erst NACH M2 einfrieren.
4. **fps-Mismatch als stille Transfer-Bremse** (M8) — falls die Checkpoints 30 fps
   erwarten, war v1s +7–12-Punkte-Transfer eine Untergrenze; gut für uns, aber unbewiesen.
5. ~~Verifikations-Lücke~~ **ERLEDIGT 2026-07-13**: die ausgefallene Welle wurde
   nachgeholt (107/108 Prüfer; 19/24 Behauptungen 3:0 bestätigt, 5 Detail-Korrekturen
   eingearbeitet — HitNet roh vs. optimiert, mlx-vlm-fps-Verwurf, llama.cpp-4-fps-Default,
   ShuttleSet22-Details, Focal-Loss-Satz gestrichen). Einzige Restlücke: eine
   DBV-Behauptung mit 2:0 statt 3:0 entschieden (ein Prüfer-Serverfehler).
6. **Lizenz-Override-Restrisiko** (BadmintonDB, VideoBadminton — beide ohne Lizenz-Grant,
   per User-Entscheid 2026-07-13 freigegeben): unkritisch solange die App
   nicht-distribuiert bleibt; falls das Projekt je distribuiert wird, müssen beide
   Abhängigkeiten (und darauf trainierte Modelle) ersetzt werden.
7. **Wang et al. 2022 (ACM TIST)** noch nicht gezogen — exakte Fein-Klassen-Definitionen
   vor Labeling-Start pinnen (Phase-1-Aufgabe, kostet <1 h).

## 14. Quellen

`sources.md` — vollständig, mit Prüfstatus je Behauptung.
