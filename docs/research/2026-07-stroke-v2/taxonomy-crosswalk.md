# Taxonomie-Kreuztabelle — ShuttleSet-18 ↔ Deutsch (DBV-geprüft) ↔ grob-8

Stand 2026-07-13. ShuttleSet-Klassengrößen direkt aus den CSVs gezählt (36 484 Zeilen,
[verifiziert 3:0]); DBV-Termini aus den vollextrahierten offiziellen PDFs
(Rahmentrainingskonzeption 2017, Bausteinsystem) — [verifiziert] (Nach-Verifikation
2026-07-13: Stick-Smash, Töten, Treibball-Kollision und die Doppelabwehr-Mappings je 3:0
bzw. 2:0 bestätigt; einzige Korrektur: ein unzutreffendes Sekundärzitat entfernt).

**Wichtig:** Es existiert KEINE offizielle flache DBV-Schlagtaxonomie. Die DBV-Begriffe
stehen verstreut in Alters-Bausteinen (U11–O19) und sind stark anglisiert. „DBV-Term" heißt
unten: wörtlich in einem der beiden PDFs; „üblich" heißt: etabliert in Schul-/Club-/
Lexikon-Material (Wikibooks, sportunterricht.de, badlab, KNSU), aber nicht in den DBV-PDFs.

## Feine Ebene (Label-Studio-Labels)

| # | ShuttleSet (EN / 中文) | n (SSet) | Kickoff-Label | **Empfohlenes Label** | Beleg / Anmerkung |
|---|---|---|---|---|---|
| 1 | short service 發短球 | 2 051 | kurzer Aufschlag | kurzer Aufschlag | DBV: „kurzer RH-Aufschlag"; VH/RH-Unterscheidung labeln wir nicht |
| 2 | long service 發長球 | 373 | langer Aufschlag | **hoher Aufschlag** | kanonisch (Wikibooks, sportunterricht.de); Flick = „Swip" — bleibt außerhalb (im Zweifel „hoher Aufschlag") |
| 3 | clear 長球 | 2 922 | Clear | Clear | Angriffs-/Verteidigungsclear nur „üblich", DBV differenziert „Clear-Höhe" — keine Fein-Trennung |
| 4 | smash 殺球 | 2 586 | Smash | Smash | DBV kennt zusätzlich Halfsmash/80%-Smash — nicht labeln |
| 5 | wrist smash 點扣 | 1 648 | Handgelenk-Smash | **Stick-Smash** | DBV „Stick-Variationen (Stick-Drop/-Smash/-Clear)"; mobilesport.ch; „Handgelenk-Smash" existiert nicht |
| 6 | drop 切球 | 2 144 | Drop | Drop | DBV kennt Schnitt-Drops (Innen-/Außenschnitt) — nicht labeln |
| 7 | passive drop 過度切球 | 1 356 | passiver Drop | passiver Drop | kein etablierter deutscher Begriff (DBV-nächstes: „Neutralisation"); Definition aus Wang et al. 2022 pinnen (Phase 1) |
| 8 | drive 平球 | 700 | Drive | Drive | DBV: Drive, „schieben"; deutsch auch „Treibschlag" |
| 9 | driven flight 小平球 | **68** | flacher Treibball | **→ in Drive aufgehen lassen** | Namens-Kollision („Treibball" = Drive: Brahms, Duden); 0.18 % / 0 bei Damen — als Fein-Label nicht lebensfähig |
| 10 | back-court drive 後場抽平球 | 473 | Hinterfeld-Drive | Hinterfeld-Drive | kein DBV-Term; verständliche Bildung, behalten |
| 11 | net shot 放小球 | 6 290 | Netzdrop | Netzdrop | DBV: „Netztechnik Vh/Rh Stop"; „Netzdrop" üblich |
| 12 | return net 擋小球 | 3 620 | Netz-Return | **Block** | DBV: „(aktiver) Block", „kurze Abwehr"; „Netz-Return" ist kein deutsches Coaching-Vokabular. Semantik (nur Smash-Block oder jeder weiche Netz-Return?) aus Wang et al. 2022 pinnen |
| 13 | cross-court net shot 勾球 | 1 371 | Cross-Netzdrop | Cross-Netzdrop | DBV: „cross Lösungen am Netz" |
| 14 | rush 撲球 | 512 | Rush/Netzkill | **Töten** | DBV wörtlich: „Töten (Ball auf dem Boden ‚versenken')"; „Rush" nur als Griff (Rushgriff), „Netzkill" nirgends |
| 15 | push 推球 | 2 925 | Push | Push | DBV verwendet „Push" selbst (Doppel-Annahmemuster) |
| 16 | lob 挑球 | 5 331 | Lob/Lift | Lob (Lift) | DBV: „Lob (Lift)", „Swip/Lift" |
| 17 | defensive return lob 防守回挑 | 301 | Abwehr-Lob | **hohe Abwehr** | DBV: „Doppelabwehr (Konter/hohe Abwehr)" |
| 18 | defensive return drive 防守回抽 | 406 | Abwehr-Drive | **Konter** | DBV wörtlich („Doppelabwehr (Konter/…)") |
| — | unknown 未知球種 | 1 407 | Unsicher | Unsicher | nie trainiert |
| — | (kein Schlag) | — | Kein Schlag | Kein Schlag | Hit-Detektor-Eval, nie trainiert |

Ergebnis: **17 Fein-Labels** (driven flight in Drive aufgegangen) + Unsicher + Kein Schlag.

## Grobe Ebene (Trainingsklassen)

Änderung gegen Kickoff: die drei Abwehrschläge bilden zusammen die grobe Klasse **Abwehr**
(DBV-Schirm „Doppelabwehr"); dafür wandern Block aus „Netzspiel" und hohe Abwehr aus „Lift"
dorthin. Bleibt bei **8 groben Klassen**:

| Grob (trainiert) | Feine Mitglieder (empfohlen) | erwartete Stützung* |
|---|---|---|
| Aufschlag | kurzer Aufschlag · hoher Aufschlag | gut (Serve-Erkennung existiert) |
| Clear | Clear | gut |
| Smash | Smash · Stick-Smash | mittel |
| Drop | Drop · passiver Drop | gut |
| Drive | Drive (inkl. driven flight) · Hinterfeld-Drive | im Doppel vermutlich unterschätzt (Detektor-Bias, M2) |
| Netzspiel | Netzdrop · Cross-Netzdrop · Töten · Push | gut |
| Lift | Lob (Lift) | gut |
| **Abwehr** | **Konter · hohe Abwehr · Block** | im Doppel real häufig; v1-Zählung nicht belastbar |

\* v1-Klassenzählungen sind mutmaßlich detektor-verzerrt (flache Schläge unterzählt) —
Stützung erst nach M2 (echter Hit-Recall) und dem Neu-Labeling belastbar.

## Abgleich mit den bekannten harten Paaren

Publizierte Konfusionen (BST/ShuttleSet-Benchmarks; v1-GBDT lokal): drop↔net und
clear↔lift. Die Umgruppierung entschärft NICHTS davon (beide Paare bleiben getrennte grobe
Klassen), verschärft aber auch nichts: Block (ehem. Netz-Return) ist kinematisch von
Netzdrop unterscheidbar über den vorausgehenden Gegner-Schlag (Kontext-Feature Rally-Folge)
— dieselbe Quelle, aus der BSTs CleanGate/AimPlayer-Module ihren Gewinn ziehen.

## Doppel-Tauglichkeit

DBV behandelt Doppel über Taktik (Angriffssysteme Level 0–3, Spieleröffnung „Schläge 1–4",
Annahmemuster mit Push), nicht über eigenes Schlagvokabular. ShuttleSet ist reines
Profi-EINZEL — die Klassen-Prioren verschieben sich im Doppel deutlich Richtung
Drive/Abwehr/Netz (flaches Spiel). Das ist ein Verteilungs-, kein Vokabular-Problem:
grob-8 mit der Abwehr-Gruppe deckt Doppel ab.
