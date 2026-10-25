# Badminton stroke/shot-type recognition, hit (shuttle-contact) detection and rally segmentation from video — state of the art 2025 to Oct 2026

> **How these notes were made (please read first).** In this session the network blocked arxiv.org, openaccess.thecvf.com, alphaxiv, emergentmind, Semantic Scholar, ACM, Springer, MDPI and ResearchGate, both for page fetches and for direct downloads. Only GitHub (git clone, raw files) and a web-search engine that returns short summaries worked. So:
> - **Numbers from GitHub repos** (mainly BST's `result_table.xlsx`, licence files, READMEs) were read directly from the primary artifact. They are still **self-reported by authors** and have not been independently checked.
> - **Numbers for papers without a repo I could read** (TemPose-TF-ASF, DSTA/Fine-Badminton, UMEG-Net, rally detection, TactiPlay, etc.) come from search-engine summaries of abstracts, HTML pages and third-party leaderboards. They are **unverified secondary readings** and should be checked against the PDFs before anyone relies on them.
> - "Earlier" marks pre-2025 baselines.

## Q1. What new stroke-classification models appeared in 2025–2026, and how do they compare with BST on ShuttleSet / ShuttleSet22 / VideoBadminton / other benchmarks?

### Takeaway
BST (arXiv Feb 2025 → v4 Mar 2026, CVPRW 2026 CVsports) is still the reference pose + shuttle + court-position fusion model with public code (MIT) and weights. Its own results table shows only a small edge over the strongest TemPose variant (TemPose-TF) when all training data is used (≈ +0.1 pp accuracy on the 35-class split). The edge grows to several points in the low-data regime (25 % of training data). The one 2026 successor in the same family, TemPose-TF-ASF (May 2026), reports higher absolute numbers (0.854 acc / 0.761 macro-F1). It does this by adding context from the neighbouring strokes, which requires two passes. It is not directly comparable to BST's table, and I found no code for it. Video-transformer work in 2025–2026 moved to *localization* (DSTA on Fine-Badminton) and captioning (BFMD, FineBadminton) rather than clip-level stroke classification.

### Cited Findings
**BST — Badminton Stroke-type Transformer (Chang, NTHU)**
- arXiv 2502.21085, latest v4 revised March 2026; accepted to the CVPR 2026 Workshop (12th CVsports). It classifies singles stroke types from per-player swing clips cut out of broadcast video, using pose estimation, shuttle tracking and court-line detection as inputs — [arXiv abs](https://arxiv.org/abs/2502.21085); [CVF Open Access CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Chang_BST_Badminton_Stroke-type_Transformer_for_Skeleton-based_Action_Recognition_in_Racket_CVPRW_2026_paper.html); [repo README badges](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)
- Architecture: player pose and position act as Query, and the shuttle trajectory supplies Key/Value in a Transformer encoder. "Clean Gate" (CG) filters noisy trajectory segments using player–shuttle interaction strength. "Aim Player" (AP) matches the trajectory to each player's motion — [search summary of arXiv/CVPRW paper](https://arxiv.org/abs/2502.21085)
- Variants in code: `BST_0`, `BST_CG`, `BST_AP`, `BST_CG_AP`. "BST-CG is identical to BST-1 from the arXiv preprint v2" — [README](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)
- Label sets: ShuttleSet 35 classes (34 + None; the rare "driven flight/小平球" class is moved into None), ShuttleSet_merged 25 classes (None + 24), BadmintonDB 18 classes, TenniSet 6 classes — [README](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)
- **Author-reported results** (means over 5 seeds; "F1-score (Avg)" = macro-F1). I read these directly from the repo's results workbook — [result_table.xlsx](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/results_and_images/result_table.xlsx):
  - *ShuttleSet 35-class, seq_len=30, 2D pose, full training set:*

    | Model | Acc | Macro-F1 |
    |---|---|---|
    | BST-CG-AP | 0.7695 | 0.7043 (top-2 acc 0.9267; worst-class F1 0.2446) |
    | BST-AP | 0.7689 | 0.7023 |
    | BST-CG | 0.7681 | 0.7004 |
    | BST | 0.7676 | 0.6966 |
    | BST-0 | 0.7626 | 0.6908 |
    | TemPose-TF | 0.7683 | 0.7027 |
    | TemPose-SF | 0.7602 | 0.6922 |
    | TemPose-PF | 0.7384 | 0.6670 |
    | TemPose-V (J+B) | 0.7216 | 0.6378 |
    | ST-GCN | 0.7198 | 0.6396 |
    | SkateFormer | 0.7156 | 0.6304 |
    | ProtoGCN | 0.7152 | 0.6282 |
    | BlockGCN | 0.7108 | 0.6310 |

    3D pose was worse than 2D (BST-CG-AP 3D: 0.7525 / 0.6760).
  - *ShuttleSet 35-class, seq_len=100* ("between 2 hits with max limits" window): BST-CG-AP 0.7710 / 0.7042; BST-CG 0.7704 / 0.7026; TemPose-TF 0.7580 / 0.6902. (This BST-CG-AP row matches the "Acc 0.7710, Macro-F1 0.7042" figure quoted on a secondary site — [emergentmind](https://www.emergentmind.com/topics/shuttleset).)
  - *ShuttleSet 35-class, 25 % of training data:* BST-CG 0.6436 / 0.5266; BST-AP 0.6374 / 0.5276; TemPose-TF 0.5822 / 0.4372; TemPose-SF 0.5888 / 0.4552.
  - *ShuttleSet merged 25-class, seq_len=30:* BST-CG-AP 0.8254 / 0.7983; BST-AP 0.8229 / 0.7992; TemPose-TF 0.8189 / 0.7943; ST-GCN 0.7758 / 0.7352; ProtoGCN 0.7746 / 0.7306. *seq_len=100:* BST-CG-AP 0.8322 / 0.8097. *25 % data:* BST-AP 0.7038 / 0.6302 vs TemPose-TF 0.6636 / 0.5904.
  - *BadmintonDB 18-class:* BST-AP 0.6517 / 0.5799; BST-0 0.6460 / 0.5671; TemPose-TF 0.6199 / 0.5448; ST-GCN 0.5602 / 0.4870.
  - *TenniSet 6-class:* BST-AP 0.9923 / 0.9922 vs ST-GCN 0.9890 / 0.9882 (this benchmark is saturated).
- In the paper's framing, BST variants trained on only 25 % of the data "significantly outperformed" TemPose — [search summary of BST paper](https://arxiv.org/pdf/2502.21085)
- **Clip cutting depends on known hit frames.** `gen_my_dataset.py` cuts each training/test clip around ShuttleSet's annotated `frame_num` (hit time). Two strategies are used: a fixed ±t window, or "between_2_hits(_with_max_limits)", which runs from the previous hit to the next hit. So benchmark accuracy assumes ground-truth hit timing — [gen_my_dataset.py](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/ShuttleSet/gen_my_dataset.py)
- Inputs and normalization: 2D pose from MMPose (joints relative to the player bounding box, normalized by its diagonal); shuttle from TrackNetV3 (the paper used the attention variant by alenzenx), "normalized by the video resolution"; player position = midpoint of the feet projected into court coordinates and normalized by the court rectangle — [README](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)
- Independent (non-author) note from a third-party BST-based pipeline: "BST is a model that classifies a given segment". The author's variable window runs from the previous hit to slightly past the next hit, so it depends on neighbouring hits and rally boundaries. That pipeline's own adapter cuts at midpoints between hits instead, which is "not the same" as the author window. Its 35-class "unknown" output was not trained to reject non-hits — [dengty2002-afk/badminton-data-analysis ASSESSMENT.md](https://github.com/dengty2002-afk/badminton-data-analysis/blob/main/experiments/swingnet/downstream-impact-20260913/ASSESSMENT.md)

**TemPose-TF-ASF (Liu & Lee, arXiv 2605.02558, May 2026) — preprint**
- A context-aware extension of TemPose. It feeds in stroke-type information from both the preceding and the following stroke. It is two-stage: the baseline's first-pass predictions are reused as estimated context to jointly train the ASF module and the classifier. Adding ASF to other strong methods is also reported to help — [arXiv abs via search](https://arxiv.org/pdf/2605.02558); [emergentmind listing](https://www.emergentmind.com/papers/2605.02558)
- Reported (Table 2, per search summary): TemPose-TF-ASF acc 0.854, macro-F1 0.761, Acc-2 0.945, ≈1.95 M params. Its own TemPose-TF baseline: 0.835 / 0.742 / 0.927, ≈1.71 M params. Dataset: ShuttleSet — [arXiv html v1](https://arxiv.org/html/2605.02558v1). **Unverified.** The class count and split were not visible, and its TemPose-TF baseline is much higher than BST's TemPose-TF (0.7683 on 35-class; 0.8189 on 25-class), so the protocols clearly differ — [BST result_table.xlsx](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/results_and_images/result_table.xlsx)

**TemPose (earlier, 2023)** — a skeleton transformer for fine-grained badminton motion that adds extra input streams (shuttle position) to its temporal transformer layers. It is the main baseline that BST and ASF build on — [ITU/AAU publication page](https://pure.itu.dk/en/publications/tempose-a-new-skeleton-based-transformer-model-designed-for-fine-/); [CVPRW submission PDF](https://vbn.aau.dk/files/700594667/TemPose_CVPRSubmitted.pdf)

**VideoBadminton (earlier, arXiv 2403.12385, v2 Nov 2024) — RGB video benchmark**
- Self-recorded (not broadcast): 18 classes, 7,822 clips, 145 min, 19 athletes from the National Central University team; Auburn University + NCU — [arXiv v1 html](https://arxiv.org/html/2403.12385v1); [hyper.ai dataset page](https://hyper.ai/en/datasets/30582)
- Repo model zoo (per search summary): SlowFast top-1 82.80 %, top-5 97.54 %, mean class acc 73.80 %; TimeSformer 73.18 % / 94.78 % / 57.70 %. The paper's Table 2 reports balanced subsets (VideoBadminton-10/-50), so check which split these numbers belong to — [qilimk/VideoBadminton](https://github.com/qilimk/VideoBadminton); [Moonlight review](https://www.themoonlight.io/en/review/benchmarking-badminton-action-recognition-with-a-new-fine-grained-dataset)
- I found no VideoMAE/InternVideo fine-tune numbers on VideoBadminton.

**Video-transformer / multimodal work 2025–2026 (mostly beyond clip classification)**
- DSTA (arXiv 2605.23355, May 2026; IEEE TMM header): a parameter-efficient adapter with three parallel branches (temporal change, vertical spatial change, horizontal spatial change) for fine-grained badminton temporal action localization. Results are under Q2 — [arXiv html](https://arxiv.org/html/2605.23355); [search summary](https://www.alphaxiv.org/abs/2605.23355)
- FineBadminton (He et al., arXiv 2508.07554, ACM MM 2025): BWF broadcast matches; a three-level hierarchy (Foundational Actions / Tactical Semantics / Decision Evaluation); 11 primary stroke categories built on ShuttleSet22 plus expert input; annotation by MLLM proposals + human refinement; FBBench is for MLLMs — [arXiv](https://arxiv.org/html/2508.07554v1); [project page](https://finebadminton.github.io/FineBadminton/)
- BFMD (Ding, Fujii, Tamaki, arXiv 2603.25533, 2026): a VideoMAE-based dense shot-captioning model with "Semantic Feedback". Multimodal inputs plus semantic feedback beat an RGB-only baseline (no numbers retrieved; SOTAVerified lists it as unverified) — [arXiv](https://arxiv.org/html/2603.25533v1); [SOTAVerified](https://sotaverified.org/papers/260325533)
- Pose-only, everyday-video student paper (Excel@FIT, VUT Brno 2026): 89.5 % accuracy on 10 classes. DROP/LIFT and SMASH/CLEAR remain hard to tell apart — [excel.fit.vutbr.cz PDF](https://excel.fit.vutbr.cz/submissions/2026/018/18.pdf)
- A 2026 Springer paper, "spatio-temporal graph convolutional meta-learning framework", claims 93.7 % recognition accuracy and few-shot generalization. Its cross-domain section is hedged and it does not clearly test amateur video — [Springer Discover AI](https://link.springer.com/article/10.1007/s44163-026-01183-6) (treat with caution)
- Community repos from 2026 (no papers): MMAction2 STGCN++/PoseC3D for badminton — [Exiaee/mmaction2-for-badminton](https://github.com/Exiaee/mmaction2-for-badminton). A TrackNetV3 + RTMPose + 8-layer BST pipeline with an LSTM pose-only baseline — [RichardPinter/badminton_shot_type](https://github.com/RichardPinter/badminton_shot_type). A 12-class learned stroke classifier merged into 7 groups — [Good-Badminton README](https://github.com/seafood-rice/Good-Badminton)

**Forehand/backhand:** ShuttleSet annotates `backhand` and `aroundhead` flags per stroke, alongside 18 shot types, hitter/opponent locations and landing coordinates. A model could be trained on these attributes, but none of the 2025–2026 papers I found report a dedicated forehand/backhand result — [ShuttleSet README (CoachAI-Projects)](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)

**VLM as a stroke classifier with numbers:** FineBadminton/FBBench reports that current MLLMs "still face significant challenges". I could not retrieve per-model stroke-classification accuracies — [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554)

### Inferences
- On the full ShuttleSet data, BST, TemPose-TF and TemPose-SF are within about 1 pp of each other. What matters most is the input modality: adding shuttle trajectory gives about +4 pp over pose-only, and adding court position gives a bit more. The exact architecture matters much less. For a small amateur club with little labelled data, BST's larger low-data margin (+5–6 pp at 25 % data) is the more relevant property.
- Worst-class F1 stays at about 0.2–0.3 on the 35-class split even for the best BST. Fine-grained rare classes are unreliable. Collapsing to a club-level taxonomy (clear/smash/drop/net/drive/lift/serve) brings results closer to the 25-class numbers, around 0.80 macro-F1, and probably higher with fewer classes (inference).
- ASF's gain comes from rally context (neighbouring stroke types). For an offline, post-match pipeline like the user's, this is cheap to copy, e.g. a second-pass sequence model over BST logits. However, I found no ASF code (see Q5).
- Every published BST number assumes clips centred on ground-truth hits. Real end-to-end accuracy ≈ hit-detector precision/recall × conditional stroke accuracy (the third-party repo states exactly this formula; see Q2).

### Gaps
- I found no ShuttleSet22 stroke-*classification* benchmark results. ShuttleSet22 is used for stroke *forecasting* (CoachAI 2023 Track 2) — [ShuttleSet22 paper](https://ar5iv.labs.arxiv.org/html/2306.15664).
- I found no 2025–2026 VideoMAE/InternVideo clip-classification fine-tunes on ShuttleSet or VideoBadminton with numbers.
- TemPose-TF-ASF: class count, split, code and licence are unknown.
- BST's paper-PDF tables could not be opened here. The repo workbook was used instead (same author, but the paper may report different "best seed" figures).
- I found no MDPI Sensors / Applied Sciences 2025–2026 video-based stroke classifier with a verifiable benchmark.

## Q2. What is the state of hit detection and rally segmentation (event spotting, temporal action localization, audio-based hit detection, shuttle-trajectory-turn detection), and are there badminton-specific results?

### Takeaway
Hit detection has moved toward precise event spotting (PES). The strongest 2025–2026 badminton-relevant method is UMEG-Net (AAAI 2026): a few-shot graph over skeletons + shuttle + court keypoints, evaluated on ShuttleSet contact frames. Its code has **no licence file**, so it is off-limits under the project policy. Practical hit detectors that have been measured on real footage (rules / SwingNet / fusion) reach only about 68–71 % F1 at ±5 frames in one small third-party test. Rally segmentation is a well-handled problem in broadcast footage (≈95 % in one 2025 paper; frame-level rally/interruption labels in BFMD). For amateur footage the only number I found is TactiPlay's 75 %. I found no badminton-specific audio hit-detection paper.

### Cited Findings
**Hit / contact-frame detection and event spotting**
- UMEG-Net (Liu et al., arXiv 2511.14186, AAAI 2026): a unified multi-entity graph of player skeletons, sport-specific object keypoints (shuttle) and court corners. It uses advanced GCN + multi-scale temporal shift and has 2.2 M params. Multimodal distillation produces an RGB student (UMEG-Net-distill) — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37681); [repo README](https://github.com/LZYAndy/UMEG-Net)
- Their ShuttleSet PES set has 3,685 rally clips, average 10.9 s and 10.5 shots per rally. Each shot is annotated with the exact racket–shuttle contact frame and its type (36 categories) — [arXiv html](https://arxiv.org/html/2511.14186v1)
- Results (search summary, unverified): on ShuttleSet, UMEG-Net beats the best skeleton baseline (BlockGCN) by +2.1 % F1_evt and +4.6 % Edit. Across five datasets, F1_evt is +1.3 to 5.5 % and Edit +1.3 to 16.4 % over the best PES baselines. Distillation adds +5.8 % F1_evt / +6.7 % Edit on average. This is the 100-clip few-shot setting, averaged over 5 random splits. **Absolute ShuttleSet F1 not retrieved** — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37681/41643); [emergentmind topic](https://www.emergentmind.com/topics/unified-multi-entity-graph-network-umeg-net)
- Follow-up: "From Skeletons to Pixels" (NUS B.Comp. dissertation, arXiv 2604.22839, Apr 2026) adds prediction-level (Adaptive Weight) and representation-level (AMD-FED) skeleton→RGB distillation. It is evaluated on F³Set-Tennis and figure skating, **not badminton** — [alphaxiv](https://www.alphaxiv.org/abs/2604.22839)
- F³Set (arXiv 2504.08222, ICLR 2025): a benchmark for fast, frequent, fine-grained events with precise timestamps. It includes a badminton subset (28 elements / 1,008 event types per search summary). Its method F³ED combines an event localizer, a multi-label classifier and an optional contextual module. Metrics: event-wise mean F1, element-wise mean F1, edit score — [arXiv](https://arxiv.org/html/2504.08222v1); [F3Set README](https://github.com/F3Set/F3Set)
- Survey of action spotting / precise event detection in sports (2025) — [arXiv 2505.03991](https://arxiv.org/html/2505.03991v1)
- **Third-party measured comparison** (Sep 2026, not peer-reviewed): 6 regression videos, 174 manually labelled hits, strict match = ±5 frames plus per-label uncertainty:

  | Method | Precision | Recall | F1 |
  |---|---|---|---|
  | Hand-written rules | 62.32 % | 74.14 % | 67.72 % |
  | SwingNet (official weights, adapted) | 63.86 % | 74.14 % | 68.62 % |
  | Fixed-parameter fusion | 61.80 % | 82.76 % | 70.76 % |

  Loose (+6 frames) F1 ≈ 77–78 %. Fusion produced 33.9 % more events than ground truth. The authors say this "is not BST accuracy and not a blind test" — [dengty2002-afk README](https://github.com/dengty2002-afk/badminton-data-analysis); [ASSESSMENT.md](https://github.com/dengty2002-afk/badminton-data-analysis/blob/main/experiments/swingnet/downstream-impact-20260913/ASSESSMENT.md)
- The same assessment gives the error-propagation rule: end-to-end typed precision = 0.618 × q and typed recall = 0.828 × q, where q is BST's accuracy on correctly detected hits. A missed or extra hit also shifts the neighbouring BST windows — [ASSESSMENT.md](https://github.com/dengty2002-afk/badminton-data-analysis/blob/main/experiments/swingnet/downstream-impact-20260913/ASSESSMENT.md)
- FineBadminton's MLLM baseline uses a "Hit-Centric Keyframe Selection" module, i.e. a hit-event detector picks the keyframes — [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554)
- *Earlier baselines:* Automated hit-frame detection (Chien & Yu, arXiv 2307.16000, 2023) infers hits from changes in shuttle direction. A transformer predicts shuttle-direction sequences from player keypoints. It reports 81 % (rally trimming) and 96 % (direction) accuracy — [arXiv](https://arxiv.org/html/2307.16000v2). "A New Perspective for Shuttlecock Hitting Event Detection" (2023) uses SwingNet on image sequences — [arXiv 2306.10293](https://arxiv.org/html/2306.10293). The CoachAI 2023 "All Deep System" entry found TrackNet-based shuttle detection insufficient for hit timing — [arXiv 2308.12645](https://arxiv.org/pdf/2308.12645). A 2023 icSPORTS paper used footwork trajectory + skeleton for hit timing/location — [SciTePress](https://www.scitepress.org/Papers/2023/121627/121627.pdf)
- Audio: no badminton audio hit-detection paper was found. The closest analog is multimodal (audio + video) hit detection in padel (CVPRW 2024). It notes the field moved from MFCCs to CNNs on spectrograms — [Decorte et al. CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.pdf)

**Temporal action localization (stroke intervals)**
- Fine-Badminton (introduced with DSTA, arXiv 2605.23355, May 2026): 31 professional matches, 29 stroke categories, 2,104 rallies, 27,597 interval-annotated actions — [arXiv html](https://arxiv.org/html/2605.23355)
- DSTA results (self-reported; average mAP over tIoU 0.3–0.7, THUMOS14 protocol): **Fine-Badminton 66.23 %** vs TIA 64.08 %. **ShuttleSet 74.67** vs AIM 73.17 and TIA 71.35. More than +4 pp at tIoU 0.7 vs TIA. ShuttleSet had to be adapted, because it lacks native interval annotations — [search summary of arXiv](https://arxiv.org/pdf/2605.23355); [sota2 leaderboard](https://www.sota2.com/research/sota/temporal-action-localization-on-fine-badminton). **Conflict:** another summary describes the Fine-Badminton margin as "1.6 points above TIA", not 2.15 — [benchmarklist](https://benchmarklist.com/benchmarks/decoupling_spatio_temporal_adapter_for_fine_grained_badminton_action_localization/)

**Rally start/end segmentation**
- See, Paramesran, Hassan & Krishnasamy (2025, Springer Nature venue): BiLSTM/CNN detect the service start, and gradient computation detects the rally end (frames with no shuttle motion). About 95 % average accuracy over 187 rallies of a second dataset (abstract-level) — [Monash research portal](https://research.monash.edu/en/publications/a-comparative-analysis-of-deep-learning-models-and-gradient-compu/)
- BFMD (2026): 19 full broadcast matches (singles + doubles), >20 h, 1,687 rallies, 16,751 hit events. Frame-level manual labels split rallies from interruptions (replays, Hawk-Eye). Rally boundaries are defined from shuttle contact and point transitions — [arXiv](https://arxiv.org/html/2603.25533v1); [arXiv PDF](https://arxiv.org/pdf/2603.25533)
- TactiPlay (arXiv 2607.27125, Jul 2026; **amateur players**): 75 % segmentation accuracy on 109 test rallies. Winner and win-reason accuracy is 68 % on the successfully segmented rallies — [arXiv PDF](https://arxiv.org/pdf/2607.27125)
- Shot2Tactic-Caption (arXiv 2510.14617, Oct 2025): 5,494 shot and 544 tactic captions; a ResNet50 spatio-temporal encoder worked best. A third-party summary gives 89.28 % for "tactic unit detection" (unverified) — [arXiv html](https://arxiv.org/html/2510.14617v1)
- ShuttleSet annotates rally start/end and per-shot hit time (`frame_num`), and was split into per-rally clips before shot labelling — [ShuttleSet README](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet); [ShuttleSet paper](https://arxiv.org/pdf/2306.04948)
- *Earlier:* MonoTrack inferred rallies from gaps between detected hits (HitNet) — [ResearchGate](https://www.researchgate.net/publication/362900461_MonoTrack_Shuttle_trajectory_reconstruction_from_monocular_badminton_video)

### Inferences
- The reported numbers suggest hit detection is now the weak link, not stroke classification. A detector at about 70 % strict F1 caps any downstream stroke metric well below BST's 0.77–0.83 clip-level accuracy.
- UMEG-Net's input recipe (skeletons + shuttle + court-corner keypoints in one graph, few-shot) fits the user's existing pose / TrackNet / homography stack closely. Its idea, i.e. a small GCN over keypoints trained on a few hundred labelled amateur rallies, can be re-implemented from the paper without using its unlicensed code.
- For amateur footage without broadcast cuts, the "no shuttle motion for N frames" rule and a serve-start detector, as in See et al., look like the simplest robust baseline. TactiPlay's 75 % on amateur rallies is a sober reference point.

### Gaps
- Absolute UMEG-Net / F³ED F1 numbers on badminton were not retrievable.
- I found no badminton audio-based or audio-visual hit-detection paper for 2025–2026.
- I found no badminton result for TAL methods like ActionFormer/TriDet beyond what the DSTA paper compares against (AIM, TIA). Their exact per-threshold tables were not retrieved.
- I found no CoachAI Badminton Challenge 2025/2026 track on hit detection or stroke classification. The 2023 challenge tracks were stroke forecasting and automatic annotation — [ShuttleSet22](https://ar5iv.labs.arxiv.org/html/2306.15664); [CoachAI-Challenge-IJCAI2023 folder](https://github.com/wywyWang/CoachAI-Projects).

## Q3. How well do these models transfer from broadcast/pro footage to amateur footage with different camera angles? Any domain-adaptation or few-shot results?

### Takeaway
I found no published study that measures BST, TemPose or DSTA when moved from broadcast to amateur or club footage. The transfer-relevant evidence is indirect, and every number below is self-reported:
- pose-based models transferred singles→doubles "2× better" than vision-only (Baek & Yun 2025);
- few-shot PES (UMEG-Net, 100 clips);
- BST's low-data gains (within-domain);
- one amateur system, TactiPlay, with 75 % rally segmentation.

### Cited Findings
- Baek & Yun, "Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models" (arXiv 2508.13507, Aug 2025; journal version on ScienceDirect). ViTPose keypoints from ShuttleSet singles, a contrastive (NT-Xent) ST-GCN embedding, and separate front-court / back-court backbones for orientation and handedness. A Transformer classifier decides shot occurrence. Embeddings are transferred to doubles without retraining. The claim is "pose-based models achieve 2× better transferability than vision-only methods" (abstract; tables not retrieved) — [arXiv abs](https://www.arxiv.org/abs/2508.13507); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0923596526001888); [Moonlight review](https://www.themoonlight.io/en/review/bridging-the-gap-doubles-badminton-analysis-with-singles-trained-models)
- BST's 25 %-data results (Q1 table) are within-domain only. The authors present them as better few-example generalization, not cross-domain transfer — [BST paper via search](https://arxiv.org/pdf/2502.21085); [result_table.xlsx](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/results_and_images/result_table.xlsx)
- BST is camera-sensitive by construction. Shuttle coordinates are "normalized by the video resolution" (image space). Joints are bounding-box-relative. Only player position is in court coordinates (via homography). Inference must also copy the training normalization exactly — [README](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)
- UMEG-Net targets few-shot conditions explicitly (25 / 100-clip settings). It claims to beat baselines "significantly" in few-shot, and its RGB-distilled student is for cases where keypoints are unreliable — [repo README](https://github.com/LZYAndy/UMEG-Net); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37681)
- TactiPlay (2026) is the only 2025–2026 system I found aimed at amateurs: 75 % rally segmentation on 109 test rallies — [arXiv 2607.27125](https://arxiv.org/pdf/2607.27125)
- The third-party BST pipeline flags court-half differences (near vs far player), camera position and sequence integrity as error sources. It warns that its small 6-video numbers cannot be extrapolated — [ASSESSMENT.md](https://github.com/dengty2002-afk/badminton-data-analysis/blob/main/experiments/swingnet/downstream-impact-20260913/ASSESSMENT.md)
- Datasets with a non-broadcast domain: VideoBadminton (self-recorded, NCU team, 18 classes) — [arXiv](https://arxiv.org/html/2403.12385v1). I found no cross-dataset (ShuttleSet→VideoBadminton) experiments.
- A 2022 review of OpenPose-based stroke classification names cross-domain adaptability as open future work. The Springer 2026 meta-learning paper's few-shot / cross-domain claims are hedged — [search summary](https://www.cjig.cn/en/article/doi/10.11834/jig.210407/); [Springer 2026](https://link.springer.com/article/10.1007/s44163-026-01183-6)

### Inferences
- Expect a real accuracy drop on club footage, for three reasons: different camera height and angle (the user's DJI ultrawide, after undistortion), lower frame rate and resolution, and amateur technique that is less stereotyped than pro technique. No published number tells how big that drop is. The user's own labelled clips are the only reliable evaluation.
- Two changes look most likely to help, though neither is validated: (a) re-express BST's shuttle input in court coordinates or camera-normalized coordinates instead of image resolution, and (b) fine-tune on a small amateur set. That second step is consistent with BST's and UMEG-Net's low-data results.
- Pose-based models are probably more transferable than RGB models (Baek & Yun's 2×, self-reported). This supports keeping the pose + trajectory route rather than switching to VideoMAE-style RGB classifiers for amateur footage.

### Gaps
- I found no published broadcast→amateur stroke-classification accuracy, and no domain-adaptation results (e.g. adversarial, test-time adaptation) specific to badminton.
- Baek & Yun's actual accuracy tables and baseline definitions were not retrieved, so the 2× claim cannot be checked.

## Q4. Are there doubles-specific stroke / hitter attribution methods?

### Takeaway
Doubles is the most active new-data area of 2025–2026: Baek & Yun's singles→doubles transfer, BFMD (singles + doubles), and a reported DShuttleNet doubles dataset (Oct 2026). Still, I found **no published, benchmarked method for within-team hitter attribution**, i.e. which of the two partners hit the shuttle. The only implementation I found is an open-source project that has not been validated on real footage.

### Cited Findings
- Baek & Yun (2025): first "systematic cross-format transfer" in racket sports. Doubles tracking uses YOLOv11x + BoT-SORT plus velocity-based extrapolation during brief disappearances, which reassigns identities and fixes ID switches. Shot *occurrence* is detected with singles-trained pose embeddings. The authors note doubles-specific datasets are still needed — [arXiv html](https://arxiv.org/html/2508.13507v1); [emergentmind](https://www.emergentmind.com/papers/2508.13507)
- DShuttleNet (arXiv 2610.14016): described as the first public stroke-level doubles dataset. 58 matches, 4,483 rallies, 54,141 shots from international tournaments 2022–2025. Frame-level (x, y) for all four players and the shuttle, expert checking of sampled rallies, and point-loss reasons — [arXiv html](https://arxiv.org/html/2610.14016). **Flag:** this appeared in only one search result, and a second search could not find it. Verify before citing.
- BFMD includes doubles matches, with shot types, shuttle trajectories and pose keypoints for 16,751 hit events — [SOTAVerified](https://sotaverified.org/papers/260325533); [arXiv](https://arxiv.org/pdf/2603.25533)
- *Earlier:* Ding et al. (2023, Multimedia Tools and Applications) built a drone (top + back view) doubles dataset for control-area / teamwork estimation, with code and data on GitHub — [arXiv 2305.04247](https://arxiv.org/abs/2305.04247v3); [Springer](https://link.springer.com/article/10.1007/s11042-023-16362-1/fulltext.html)
- Open-source attempt: Good-Badminton PRs add 4-player identity, within-side "who-hit" attribution with a confidence score (a short frame window around each hit), and per-player stats. A PR states the doubles identity logic has not been run on real footage. A later PR withholds stroke labels in doubles — [PR #14](https://github.com/seafood-rice/Good-Badminton/pull/14); [PR #15](https://github.com/seafood-rice/Good-Badminton/pull/15)
- DoublesEval (arXiv 2608.24439) evaluates VLM multi-agent tactical reasoning on pro doubles. It is out of scope here (VLM), noted only for completeness — [arXiv](https://arxiv.org/pdf/2608.24439)
- F³Set includes a tennis-doubles subset, not badminton doubles, per search summary — [arXiv 2504.08222](https://arxiv.org/html/2504.08222v1)

### Inferences
- For P3+ doubles, the practical recipe from the literature is to:
  1. keep pose-based stroke models trained on singles, since they transfer;
  2. attribute the hitter by checking, at the detected contact frame, which partner's wrist or racket keypoint is closest to the shuttle;
  3. invest in identity tracking with ID-switch repair.

  The user's pose/swing-fusion hit detector already provides most of this. This is an inference; no benchmark backs it.

### Gaps
- I found no published within-team hitter-attribution accuracy numbers.
- I found no doubles stroke-classification benchmark with numbers. DShuttleNet, if real, may enable one, but its licence and access are unknown.

## Q5. Which have code + weights available, under which licences, and are there notes on PyTorch MPS / CoreML / ONNX portability?

### Takeaway
Among the 2025–2026 stroke models, only BST has clearly reusable code (MIT) **and** public weights. Several key repos carry **no licence file**: UMEG-Net, F³Set, VideoBadminton, the 2023 SwingNet hitting-event repo, BadmintonDB, and RichardPinter's pipeline. One more, dengty2002, explicitly grants no licence. All of these fall under the project's "no licence grant → avoid" rule. I found no published MPS/CoreML/ONNX results for any of these stroke or hit models. BST uses only standard PyTorch layers, but its inference script hard-codes CUDA-or-CPU.

### Cited Findings
**Licences and availability** (licence files checked directly via `git ls-tree` on the default branch, Oct 2026):

| Project | Licence | Weights / notes |
|---|---|---|
| **BST** — [repo](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer) | **MIT**, © 2025 Jing-Yuan Chang | Weights for ShuttleSet / BadmintonDB / TenniSet on Google Drive; pre-extracted .npy features; last commit 2026-10-15; env Python 3.11, PyTorch 2.4.0+cu121 |
| **TrackNetV3** — attention variant used by BST ([alenzenx](https://github.com/alenzenx/TrackNetV3)) and rectification variant ([qaz812345](https://github.com/qaz812345/TrackNetV3), last commit 2026-09-03) | **MIT** (both) | |
| **CoachAI-Projects** (ShuttleSet / ShuttleSet22 code, last commit 2026-07-17) — [repo](https://github.com/wywyWang/CoachAI-Projects) | **MIT** | No separate data licence found in the ShuttleSet README; videos are YouTube links |
| **UMEG-Net** — [repo](https://github.com/LZYAndy/UMEG-Net) | **No LICENSE file** (last commit 2025-11-10) | README says keypoint features "will be uploaded soon" |
| **F³Set / F³ED** — [repo](https://github.com/F3Set/F3Set) | **No LICENSE file** (last commit 2024-10-02) | README says trained models are in `f3set-model` |
| **VideoBadminton** — [repo](https://github.com/qilimk/VideoBadminton) | **No LICENSE file** | |
| **SwingNet hitting-event detection** (2023) — [repo](https://github.com/wish44165/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection) | **No LICENSE file** | |
| **BadmintonDB** — [repo](https://github.com/kwban/badminton-db) | **No LICENSE file** | One of BST's benchmarks |
| **RichardPinter/badminton_shot_type** — [repo](https://github.com/RichardPinter/badminton_shot_type) | **No LICENSE file** | |
| **dengty2002-afk/badminton-data-analysis** — [LICENSE.md](https://github.com/dengty2002-afk/badminton-data-analysis/blob/main/LICENSE.md) | Explicitly "has not yet been designated… does not grant an additional blanket license" | MonoTrack-derived parts fall under the Adobe Research License (non-commercial research) |
| **Good-Badminton** — [README](https://github.com/seafood-rice/Good-Badminton) | Original code Apache-2.0; fork's new code **CC BY-NC 4.0** | Allowed for this non-commercial project |
| TemPose-TF-ASF, DSTA / Fine-Badminton, BFMD, Baek & Yun, TactiPlay, Shot2Tactic-Caption | — | **No code or licence found** in searches, e.g. "I found no GitHub or code release for Fine-Badminton or DSTA" — [search summary](https://arxiv.org/pdf/2605.23355) |

**Portability notes**
- BST `bst_infer.py` sets `device = 'cuda' if torch.cuda.is_available() else 'cpu'`, so out of the box it runs on CPU on a Mac. Weights load with `torch.load(..., weights_only=True)` — [bst_infer.py](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/stroke_classification/main_on_shuttleset/bst_infer.py)
- BST model code uses only standard layers: Linear, Conv1d/Conv2d, BatchNorm1d/2d, LayerNorm, GELU/ReLU, Softmax, einsum, one_hot, topk (from a grep of `stroke_classification/`) — [repo](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/tree/main/stroke_classification)
- Good-Badminton ships RTMPose / RTMO / YOLOX as ONNX and lists "Apple Silicon Mac (MPS)" as a supported runtime — [README](https://github.com/seafood-rice/Good-Badminton)
- The dengty2002 pipeline was validated only on an RTX 5060 with PyTorch 2.13 + cu130, and says other GPUs need PyTorch / ONNX Runtime adjustments — [README](https://github.com/dengty2002-afk/badminton-data-analysis)
- UMEG-Net's distilled RGB student uses VideoMAEv2 or RegNetY(+TSM) backbones and pre-extracted frames at about 224 px height. It was tested on Ubuntu 22.04 only — [README](https://github.com/LZYAndy/UMEG-Net)

### Inferences
- BST should run on MPS by changing the device string to `mps`. Its small size (TemPose-TF-class models are about 1.7–2 M params per the ASF paper) and plain ops also make ONNX and CoreML export plausible. This is **unverified** and needs a CPU-vs-MPS NaN/parity check, per the project's MPS convention.
- Under the project licence policy:
  - **Usable:** BST (MIT), TrackNetV3 (MIT), CoachAI / ShuttleSet code (MIT), Good-Badminton (Apache / CC BY-NC).
  - **Reference only (re-implement from the paper):** UMEG-Net, F³Set / F³ED, the SwingNet hitting-event repo.
  - **Avoid:** VideoBadminton and BadmintonDB code/data, unless the authors grant a licence.

### Gaps
- I found no published MPS, CoreML or ONNX benchmarks for BST, TemPose, UMEG-Net or DSTA.
- BST weights on Google Drive carry no separate licence statement. The repo's MIT licence presumably covers them, but this is not explicit.
- ShuttleSet's data / video-usage terms are not in its README (the videos are third-party YouTube BWF content).
- I found no code for TemPose-TF-ASF or DSTA, so their licences are unknown.
