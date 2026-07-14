# Quellen — Phase-0 Research (Schlagklassifikation v2)

Stand 2026-07-13 (Nach-Verifikation abgeschlossen: die am Budget-Limit ausgefallene Welle
wurde per Workflow-Resume nachgeholt, 107/108 Prüfer erfolgreich). Prüfstatus:
**[P]** = Primärquelle, adversarial verifiziert (3 unabhängige Prüf-Agenten, ≥2/3-Mehrheit;
in einem Fall 2:0 nach Prüfer-Serverfehler) ·
**[p]** = Primärquelle geladen/zitiert, nicht tragend → nicht einzeln nachgeprüft ·
**[S]** = Sekundärquelle.

## Modelle & Taxonomie-Kernfrage (Durchlauf 1 — vollständig verifiziert)

| Quelle | Status | Trägt |
|---|---|---|
| github.com/Va6lue/BST-Badminton-Stroke-type-Transformer | [P] | Gewichte real ladbar (Drive live geprüft; „weight" = alle ShuttleSet-Ckpts, 25-Kl. mit `_merged`-Suffix; „weight_on_badDB" = BadmintonDB-trainiert, nutzbar per User-Override 2026-07-13), MIT, Input-Pipeline, 2-Spieler-Architektur |
| arxiv.org/abs/2502.21085 (+ v1/v3-HTML) — BST, CVPRW 2026 | [P] | Eingabeströme (RTMPose/TrackNetV3/Homographie), Einzel-only, HitNet-Verweis, 35=17×2+1-Arithmetik, 25-%-Daten-Ablation 0.825→0.692 |
| openaccess.thecvf.com …CVPR2026W…Chang_BST…paper.pdf | [P] | 0.8322/0.8097 (25 Kl., match-getrennt); BadmintonDB-18: 0.6517 unter Leakage-Split (CVF-PDF 403t für Bots — identische arXiv-v4 geprüft) |
| arxiv.org/pdf/2605.02558 — TemPose-TF-ASF (Preprint) | [P] | ShuttleSet-SOTA 0.854/0.761 (31 Kl.); GCNs 0.79–0.81. Einzelquelle, Eigen-SOTA → Konfidenz „mittel" |
| openaccess.thecvf.com …CVPR2023W…Ibh_TemPose…paper.pdf | [P] | 0.907 auf 13 Kl. (leak-frei, Testset=1 Video); Fusion-Ablation 0.856→0.907 |
| arxiv.org/abs/2306.04948 — ShuttleSet (KDD 2023) | [P] | „18 distinct classes" nativ; 36 492 Schläge; nur Profi-Einzel; driven-flight-Zahlen |
| github.com/wywyWang/CoachAI-Projects | [P] | 18-Klassen-Enumeration; MIT; match.csv-URLs; Klassenzählung aus CSVs |
| dl.acm.org/doi/10.1145/3552437.3555696 — BadmintonDB-Paper | [P] | nativ ~10 grobe Typen |
| raw.githubusercontent.com/Va6lue/…/LICENSE | [P] | unverändertes MIT |
| researchgate.net/publication/373134028 | [S] | stützt TemPose-Zahlen |

**Widerlegt (0:3):** „ShuttleSet = 35/25 Klassen, BadmintonDB = die 18er-Taxonomie"
(`specs/features/stroke-classification.md:46`) — die 18er ist ShuttleSets native Taxonomie.

## Datensatz-Lizenzen & Verfügbarkeit (Durchlauf 2 — vollständig verifiziert)

| Quelle | Status | Trägt |
|---|---|---|
| github.com/wywyWang/CoachAI-Projects (Sparse-Clone + API) | [P] | ShuttleSet: MIT, 36 484 Zeilen, alle 19 Klassenzählungen exakt; Videos nur als YouTube-Links |
| …/CoachAI-Challenge-IJCAI2023/ShuttleSet22 | [P] | 58 Matches (8×2021+50×2022), 140 Set-CSVs, 52 356 Zeilen, KEINE Video-URLs; offizieller Split 33 612 |
| api.github.com/repos/kwban/badminton-db | [P] | license=null, keine LICENSE-Datei → kein Lizenz-Grant. **Per User-Override 2026-07-13 trotzdem nutzbar** (Ausnahme von Locked Decision 6, solange nicht-distribuiert) |
| github.com/qilimk/VideoBadminton + Drive-Zip | [P] | 18 Klassen (nicht 14), 7 822 Clips, Uni-Training-Footage, Zip live, KEINE Lizenz. **Per User-Override 2026-07-13 nutzbar** (dito) |
| ilearn-lab.github.io/MM25-FineBadminton · arxiv 2603.25533 (BFMD) · arxiv 2510.14617 · github.com/Ning-D/Drone_BD_ControlArea | [P] | Doppel-Datensätze 2025/26 existieren, aber kein nutzbarer Download/keine Lizenz. **FineBadminton-Teil KORRIGIERT 2026-07-13** — User fand HF-Download-Pfade, s. u. |

## FineBadminton-Nachtrag (2026-07-13 — User-Fund; Einzel-Prüfung Haupt-Loop + eigene Auszählung)

| Quelle | Status | Trägt |
|---|---|---|
| huggingface.co/datasets/iLearn-Lab/Finebadminton-20K (+ API-Metadaten/Dateibaum) | [P] | gated=false, **Apache-2.0**, 70 Match-Videos (43,4 GB) + 70 Annotations-JSONs; Spiegel: Moujuruo/… |
| huggingface.co/datasets/iLearn-Lab/FineBadmintonBenchmark (+ API) | [P] | Apache-2.0, 984 QA-Clips + 12 Task-JSONs (en+zh) |
| alle 70 Annotations-JSONs (lokal geparst) | [P — eigene Messung] | 2 066 Rounds / 20 757 Hits / 74 Spieler / nur Einzel; 11 Hit-Typen + 20 Subtypen; Quality 1–7; Ball-Area 3×3; score/lose_reason/total_eva je Round; Hit-Frame-Fenster je Schlag; Annotations-Rauschen dokumentiert |
| github.com/iLearn-Lab/MM25-FineBadminton | [p] | Apache-2.0-LICENSE (Commit 2026-03-29), Demo-Site, Autoren (HIT Shenzhen u. a.) |
| arxiv.org/html/2508.07554v1 (Volltext §3) | [p] | Pipeline: TrackNetV3 + FastRCNN + VideoMAE-Hit-Detektor → MLLM-Beschreibung (3 Frames + Strukturdaten) → Refining/Summarizing-LLM → 2-Annotatoren-Review; Paper: 120 Matches/33 325 Strokes (HF = ~62-%-Subset); Taxonomie auf ShuttleSet22 aufgebaut; Hit-Centric Keyframe Selection verbessert alle Modelle; feingetuntes Qwen2.5-VL-7B max. 42,1 % |
| sites.google.com/view/coachai-challenge-2023 | [P] | nur 2023er-Edition; Track-2-Daten = ShuttleSet22 (MIT); Track-1-Daten ohne öffentlichen Pfad |

## Label-Effizienz & Kalibrierung (Durchlauf 2 — vollständig verifiziert)

| Quelle | Status | Trägt |
|---|---|---|
| arxiv 2502.21085v3 (BST 25-%-Ablation) | [P] | einziger Rennsport-Label-Effizienz-Punkt: 0.8254→0.6924 |
| ar5iv 2112.03590 (CrosSCLR/Moliner CVPRW22) · 2202.02232 (AimCLR) · 2211.13466 (HiCLR) | [P] | NTU-60 semi-supervised: 1 % Labels 51–79 %, 10 % 74–85 %; shear+crop tragend; „extreme" Bundles schaden |
| ar5iv 2209.02399 (SkeletonMAE) | [P] | 10 % Labels = 84 % Recovery; SSL-auf-eigenen-Daten-Protokoll |
| ar5iv 2107.07511 (Conformal Gentle Intro) + arxiv 2509.15349 | [P] | Beta-Gesetz der Coverage; n≈100/Klasse für ±5 % → klassen-konditional bei n=20–30 nicht tragfähig |
| ar5iv 2007.07314 (Logit Adjustment, ICLR 2021) | [P] | Tail-Loss-Zahlen exakt bestätigt; Korrektur: Paper enthält KEINEN Focal-Loss-Vergleich; „8 % relativ" nur CIFAR-10 |
| arxiv 2102.12182 · ar5iv 1810.11586 · arxiv 2506.23492 | [p] | Temperatur-Skalierung stabil bei kleinen Validierungsmengen |

## Audio-Hit-Detection (Durchlauf 2 — nachverifiziert, alle 6 tragenden Behauptungen 3:0)

| Quelle | Status | Trägt |
|---|---|---|
| isca-archive.org/avsp_2011/huang11_avsp.pdf | [P] | Fern-Mikro-Präzedenz (Broadcast-Tennis-DOPPEL): audio-only roh F 0.41–0.50, mit Konfidenz-Maß 0.68–0.77 ≈ visual 0.68–0.71; Fusion 0.77–0.84; ±8-Frame-Fenster-Muster inkl. Schalllaufzeit. Alle Tabellenwerte gegen das (AES-verschlüsselte) PDF geprüft |
| arxiv.org/abs/2409.11760 (+HTML v1/v2) — Tischtennis-Onsets | [P] | Nahfeld: P 0.98/R 1.0, 0.09 ms Onset; Bedingungen nicht übertragbar (Richtmikro 0.5–2 m) |
| ar5iv 2204.01899 + github.com/jhwang7628/monotrack — MonoTrack/HitNet | [P] | Video-Hit-Detektor-Vorlage: GRU ~16k Params; **Korrektur: roh F1 0.866; 0.946 erst nach Optimierungs-Nachstufe (inkl. „abwechselnde Schläger" = Einzel-Logik)**; Code ja, Gewichte nein, Lizenz ungeklärt |
| dafx.de …superflux (DAFx-13) · ofai.at …2014_icassp (Böck-CNN) · madmom-Doku | [P] | Onset-Eval-Standard ±25 ms, F 0.82–0.90 (Musik); SuperFlux 5–10 ms Raster |
| LICENSE-Dateien: librosa (ISC) · aubio (GPL-3) · CPJKU/madmom (BSD+CC-BY-NC-SA-Modelle) · qiuqiangkong/audioset_tagging_cnn (MIT, Zenodo 3987831) · microsoft/unilm/BEATs (MIT) · tensorflow/models/YAMNet (Apache-2.0) | [P] | alle für nicht-kommerzielles Projekt nutzbar |
| pmc.ncbi.nlm.nih.gov/articles/PMC11244353 (Sensors 2024) | [P] | Video-Video-Fusion (Swing+Trajektorie) F1 0.905, Einzel, kein Split-Protokoll angegeben |

## Doppel-Zuordnung (Durchlauf 2 — nachverifiziert)

| Quelle | Status | Trägt |
|---|---|---|
| arxiv.org/html/2508.13507v1 | [P] | EINZIGES monokulares Doppel-Ergebnis: 0.6686 acc / 0.7104 F1, zero-shot, 2 Matches; Metrik vermischt Detektion+Zuordnung |
| ar5iv 2204.01899 (HitNet 3-Wege-Kopf) | [P] | nah/fern-Zuordnung = Klassenausgang (Korrektur s. Audio-Sektion); 5-Wege-Erweiterung nirgends publiziert |
| scitepress.org/Papers/2025/131597 (RacketDB, VISAPP 2025) | [P] | CC-BY-4.0-Datensatz, Amateur-DOPPEL 4K; Schläger 16×29 px, mAP50 0.77, keine Gewichte |
| arxiv 2308.12645 (CoachAI-2023-Sieger) | [p] | verwarf Richtungswechsel-Zählung als zu verrauscht; X3D-Hitter-Klassifikation (nicht tragend → nicht einzeln geprüft) |
| arxiv 2101.04954 (EventAnchor, CHI 2021) | [p] | Heuristik-Vorlage: Richtungswechsel + Hand-Keypoint-Distanz (M6-Baseline) |
| arxiv 2305.04247 (Drohnen-Doppel-Datensatz) · arxiv 2411.08216 (SportsMOT-IDSw) | [p] | Okklusion als Mehr-Kamera-Motiv; ID-Switch-Proxys |

## Qwen3-VL lokal (Durchlauf 2 — nachverifiziert, zwei tragende Korrekturen)

| Quelle | Status | Trägt |
|---|---|---|
| huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF (API-Tree) | [P] 3:0 | Q4_K_M 18.56 GB / Q8_0 32.48 GB + mmproj → passt in 64 GB |
| github.com/Blaizzy/mlx-vlm (Code: processing_qwen3_vl.py, utils.py, Releases) | [P] | Qwen3-VL-Video-Pfad seit v0.3.4 (pixel_values_videos echt); **Korrektur: fps-Metadaten werden vor dem Prozessor verworfen (kwargs-Filter) → Timestamp-Alignment fraglich; DeepStack-Fix erst v0.6.3** |
| github.com/ggml-org/llama.cpp PR #24269, Issues 18389/16960/17660, mtmd-Quellen | [P] | Video-Input seit 2026-06; Audio-Spur verworfen; MTMD_VIDEO Compile-Flag; **Korrektur: Default ist FIX 4 fps (`fps_target=4.0f` hartkodiert, kein CLI-Flag) — nicht Original-fps → für 1-s-Clips ungeeignet ohne Quell-Build** |
| github.com/QwenLM/Qwen3-VL (vision_process.py + README) | [P] | Default fps=2 / min 4 Frames → 1-s-Clip = 4 Frames ohne Override; Interleaved-MRoPE/DeepStack/Timestamp-Alignment = Herstellerangaben |
| ggml-org/Qwen3-Omni-30B-A3B-GGUF (API-Tree) · mlx-vlm qwen3_omni_moe | [P] | Audio-fähige lokale Variante existiert; kein End-to-End-Beispiel Video+Audio |
| arxiv 2508.07554 (FineBadminton) · 2501.02955 (MotionBench) · 2410.08474 (SPORTU) | [P] 2:1 | gemessene VLM-Decke auf feinkörnigem Sport: 0.32–0.42 bzw. <0.60 |
| github.com/waybarrios/vllm-mlx · Community-Benchmarks | [S] | 30B-A3B-4bit ~68–128 tok/s auf M4 Max (nur Text-Decode-Proxys) |

## DBV-Terminologie (Durchlauf 2 — nachverifiziert; eine Behauptung 2:0 nach Prüfer-Serverfehler)

| Quelle | Status | Trägt |
|---|---|---|
| badminton.de/…/dbv-rahmentrainingskonzeption_2017.pdf (50 S., vollextrahiert) | [P] | DBV-Vokabular; grep-verifizierte Abwesenheiten (Handgelenk-Smash, Netzkill, Flick, hoher Aufschlag …) |
| badminton.de/…/dbv-bausteinsystem.pdf (7 S., vollextrahiert) | [P] | „Stick-Variationen", „Töten", „Doppelabwehr (Konter/hohe Abwehr/aktiver Block)", „Push". Doppelabwehr-Mappings stehen auf dieser Quelle allein; ein unzutreffendes Sekundärzitat (tennistraining.de) wurde entfernt |
| mobilesport.ch (Stick-Smash) · de.wikibooks.org Badminton: Schläge · sportunterricht.de/badminton · badlab.de-Lexikon · KNSU-PDF · sportfachbuch.de/pdfs/0641.pdf (Kelzenberg) | [S] | „übliche" Termini (hoher Aufschlag, Angriffs-/Verteidigungsclear, Rushgriff) |
| dersportverlag.de …9783840378751_p.pdf (Brahms Handbuch, Inhaltsverzeichnis) · duden.de/Treibball | [S] | „Der Drive (Treibschlag)" → Treibball-Kollision |

## Offene Nachprüfungen

1. ~~AVSP-2011 / HitNet / 2508.13507~~ **ERLEDIGT** — Nach-Verifikation 2026-07-13
   (Korrekturen oben eingearbeitet).
2. Wang et al. 2022 (ACM TIST) — exakte Fein-Klassen-Definitionen (Phase-1-Pflicht vor
   Labeling).
3. monotrack-Repo-LICENSE-Typ lesen, falls HitNet-Code doch importiert statt nachgebaut
   werden soll.
