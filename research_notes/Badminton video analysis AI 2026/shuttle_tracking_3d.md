# Shuttlecock detection/tracking and monocular 3D shuttle trajectory reconstruction (2025 – Oct 2026)

> **Method note for the report writer (read first).** In this research session the egress proxy blocked arXiv, alphaXiv, Springer, CVF Open Access, Semantic Scholar, Hugging Face and the project pages (`connect_rejected` / DNS failures). So **numbers attributed to papers come from search-engine extracts of those pages**, not from my own reading of the PDFs. I mark them **[paper, self-reported, via search extract]**. **GitHub repos could be cloned directly**, so README, LICENSE and code facts marked **[repo, read directly]** are first-hand. Third-party re-measurements are marked **[independent]**. Secondary summaries (Lacuna, themoonlight, emergentmind, aggregator leaderboards) are marked **[secondary, unverified]**. Earlier baselines (pre-2025) are labeled **(earlier)**.
>
> Project context: DJI ultrawide, sports hall with **white walls** (white shuttle on white wall = low contrast), current stack TrackNetV3 + WASB (TrackNetV4 noted), MonoTrack-style physics fit for 3D/speed shown as „Schätzung", runtime Apple Silicon M4 Max / PyTorch MPS. Any license is OK **except no license grant**.

---

## Q1. What new shuttle / small-fast-object tracking models appeared in 2025–2026, and how do they compare to TrackNetV3/WASB?

### Takeaway
The main 2025–2026 additions are TrackNetV4 (ICASSP 2025), **two different papers both called "TrackNetV5"** (an arXiv preprint from a Shanghai company, and an ICIP 2026 paper from the TrackNetV3 group), BlurBall (CVPRW 2026), TOTNet (CVIU 2026) and YOLO-style real-time detectors. On the standard broadcast Shuttlecock Trajectory Dataset, they all report F1 at most about 1 point above TrackNetV3 (≈97.5–98.6 → ≈98–98.5). Those gains are smaller than the spread caused by evaluation protocol, and one independent re-implementation found **no ordering gain from V2→V5**. No 2025–26 paper reports results on amateur hall footage with white walls.

### Cited Findings
**Baselines (earlier, for reference)**
- (earlier) TrackNetV3 (Chen & Wang, ACM MM Asia 2023): background estimate as auxiliary input, mixup augmentation, InpaintNet trajectory rectification. README table: Accuracy 97.51%, Precision 97.79%, Recall 99.33%, F1 98.56%, 25.11 FPS, vs TrackNetV2 F1 97.03% at 27.70 FPS and YOLOv7 F1 68.00%. **[repo, read directly; numbers self-reported]**. Hardware for the FPS figure is not given in the table — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- (earlier) WASB (NTT, BMVC 2023): HRNet-based multi-frame heatmap detector with per-sport weights (soccer, tennis, badminton, volleyball, basketball). It ships re-trained weights for DeepBall, BallSeg, TrackNetV2, ResTrackNetV2 and MonoTrack, so baselines can be compared like for like — [nttcom/WASB-SBDT MODEL_ZOO](https://github.com/nttcom/WASB-SBDT). Aggregator leaderboards put WASB 1–2 F1 points above TrackNetV2/MonoTrack on unlabeled splits (e.g., 93.1 vs 90.9 vs 90.5) **[secondary, unverified; dataset per row unclear]** — [hyper.ai SOTA page](https://vercel.hyper.ai/en/sota/tasks/sports-ball-detection-and-tracking/benchmark/sports-ball-detection-and-tracking-on)

**TrackNetV4 (Raj, Wang, Gedeon; arXiv 2409.14543, Sep 2024; ICASSP 2025)**
- Method: learnable motion attention maps from frame differencing, modulated by a "motion prompt layer" and fused with visual features. The authors describe it as plug-and-play for any heatmap tracker — [arXiv 2409.14543](https://arxiv.org/abs/2409.14543); venue ICASSP 2025 per [Griffith project page](https://time.griffith.edu.au/paper-sites/tracknet-v4/)
- Paper-reported shuttlecock F1 gains are small: TrackNetV2 90.6→91.4, TrackNetV3 97.5→97.9 **[paper, self-reported, via alphaXiv summary]** — [alphaXiv](https://www.alphaxiv.org/abs/2409.14543)
- The official repo's `docs/RESULT.md` shows slightly different badminton numbers: TrackNetV3 (3-in-3-out) F1 96.8 vs TrackNetV3 (+Motion) F1 97.1, both at 15.0 FPS (10,836 test frames); TrackNetV2 87.7 vs (+Motion) 87.8 at about 161 FPS. **[repo, read directly]**. **Conflict:** these do not match the 97.5→97.9 quoted from the paper summary — [TrackNetV4/TrackNetV4](https://github.com/TrackNetV4/TrackNetV4)

**"TrackNetV5" #1: Residual-Driven Spatio-Temporal Refinement + Motion Direction Decoupling (arXiv 2512.02789, Dec 2025)**
- From Shanghai Code Zero Sports Technology Co. The SDK README calls it a "tennis ball tracking algorithm". MDD splits frame-to-frame change into positive and negative polarity channels; R-STR treats the decoder output as a coarse draft and refines it — [arXiv 2512.02789](https://arxiv.org/abs/2512.02789); [codelancera-offical/TrackNetV5-SDK](https://github.com/codelancera-offical/TrackNetV5-SDK) **[repo, read directly]**
- Claims F1 0.9859, accuracy 0.9733, precision 0.9923 on the TrackNetV2 (shuttlecock) dataset, a 73.9% reduction of false negatives (to 344), +2.78% F1 over TrackNetV4, and F1 0.9878 on a proprietary "Loveall" dataset. MDD ablation: F1 0.9677→0.9695 **[paper, self-reported, via search extract]** — [arXiv HTML](https://arxiv.org/html/2512.02789v1); [themoonlight review](https://www.themoonlight.io/en/review/tracknetv5-residual-driven-spatio-temporal-refinement-and-motion-direction-decoupling-for-fast-object-tracking)

**"TrackNetV5" #2: Robust Shuttlecock Tracking via Motion Prompts and Spatiotemporal Attentive Fusion (Chang, Wang, Huang; ICIP 2026)**
- A separate work by the TrackNetV3 group (Yu-Shuen Wang is a co-author on both). Uses R(2+1)D spatiotemporal blocks, a gated residual motion prompt layer and spatial-channel-gate fusion. A per-rally median background is fed as a 4th input channel (`bg_mode=subtract_concat`, required, not optional). The project page says the gate "suppresses background" compared with TrackNetV4's static frame differencing — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5) **[repo, read directly]**
- Project page table (4 px tolerance, RTX 5080): TrackNetV4 11.34M params, F1 97.99, 118.81 FPS; TrackNetV5 10.09M, F1 98.47, 60.70 FPS; **TrackNetV5-Lite 2.58M params, F1 98.06, 135.18 FPS** **[paper/project, self-reported; read directly from repo `index.html`]** — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5)
- The repo's own retrained checkpoints, measured on the TrackNetV4 protocol (10,836 "played" frames, 4 px, threshold 0.5): Lite F1 98.094 (98.363 at threshold 0.35); full F1 98.113 (98.331 at 0.30). The authors say these are reproductions, not the paper's numbers. They pin the protocol by running the released TrackNetV3 weights, which give **F1 97.59 vs the 97.5 reported by TrackNetV4** **[repo, read directly; authors' own re-measurement]** — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5)
- On one demo rally the page says "within 4 px — TrackNetV3 76.7%, TrackNetV5-Lite 97.8%". That is a single qualitative showcase rally, not a benchmark **[self-reported]** — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5)

**Independent re-implementation of V1–V5 (AnInsomniacy/tracknet-series-pytorch, May 2026)**
- 30-epoch checkpoints, all 12,658 test frames, 4 px tolerance: V2 F1 0.8051; V3 tracker+rectifier F1 0.8094 (highest recall 0.8019, precision 0.8171); V4 F1 0.8016; **V5 F1 0.7732** (precision 0.8788 from more FP1). The authors stress these "should not be presented as paper-level reproduction results" **[independent, read directly]** — [AnInsomniacy/tracknet-series-pytorch EVALUATION_RESULTS](https://github.com/AnInsomniacy/tracknet-series-pytorch)

**BlurBall (Gossard, Radovic, Ziegler, Zell; arXiv 2509.18387, Sep 2025; v3 Mar 2026; CVPRW 2026 CVsports)**
- Table-tennis ball detector built on WASB: HRNet multi-frame model plus Squeeze-and-Excitation attention. It jointly predicts ball position, blur orientation and blur extent. A new labeling convention puts the ball at the **center of the blur streak**, which the authors say improves accuracy across detectors. Comes with a 64k-frame dataset — [cogsys-tuebingen/blurball](https://github.com/cogsys-tuebingen/blurball) **[repo, read directly]**; [arXiv 2509.18387](https://arxiv.org/abs/2509.18387)
- Comparison table on table tennis (authors' own runs of the baselines): TrackNetV3 F1 93.74 (Front) / 95.93 (Mid); WASB step=3 94.23 / 95.77; WASB step=1 95.58 / 96.00. BlurBall's own row was not in the retrieved text. TrackNetV3 has no AP because its rectifier gives no confidence **[paper, via search extract]** — [arXiv HTML](https://arxiv.org/html/2509.18387v1)
- BlurBall is "sensitive to duplicated frames (common in re-encoded online videos)", and its inference script dedups frames automatically **[repo, read directly]** — [cogsys-tuebingen/blurball](https://github.com/cogsys-tuebingen/blurball)

**TOTNet (Xu, Agrahari Baniya, Wells, Bouadjenek, Dazeley, Aryal; Deakin; arXiv 2508.09650, Aug 2025; CVIU 2026)**
- 3D convolutions, a visibility-weighted loss and occlusion augmentation that masks the ball region during training. Aimed at offline analytics. Evaluated on four datasets: tennis, badminton, table tennis and a new TTA set — [arXiv 2508.09650](https://arxiv.org/html/2508.09650); [CVIU](https://www.sciencedirect.com/science/article/abs/pii/S107731422600024X)
- Reports RMSE falling from 37.30 to 7.19 and fully-occluded-frame accuracy rising from 0.63 to 0.80. Tennis partially-occluded RMSE 105.73→63.41. Badminton is described only as "best in both RMSE and accuracy across visibility levels"; exact badminton numbers were not retrievable **[paper, self-reported, via search extract]** — [alphaXiv FR](https://www.alphaxiv.org/fr/abs/2508.09650); [CVIU author PDF](https://rbouadjenek.github.io/assets/pdf/YCVIU_104657.pdf)

**Other 2025–26 detectors**
- YO-CSA-T (Lai, Shi, Zhu; arXiv 2501.06472, Jan 2025): YOLOv8s with contextual and spatial attention for a badminton robot, using stereo 3D. ONNX-accelerated: 73.94% mAP at 5.82 ms/frame. mAP@0.75 82.67%→89.37% vs plain YOLOv8s. Test data: 12 clips at 160 fps **[paper, self-reported, via search extract]** — [arXiv 2501.06472](https://arxiv.org/pdf/2501.06472)
- One-Shot Badminton Shuttle Detection for Mobile Robots (ETH RSL; arXiv 2603.06691, Mar 2026): fine-tuned YOLOv8 reaches F1 0.86 (custom metric) in environments similar to training and **0.70 in unseen environments**. Performance depends heavily on shuttle size and background texture — [arXiv 2603.06691](https://arxiv.org/html/2603.06691v2)
- TrajectoryNet++ (GitHub, Jul 2025): TrackNetV2 backbone plus MSPEM (multi-scale surround prior) and CKAM attention, with InpaintNet. Claims superiority under occlusion, blur and lighting changes, but its comparison is only a figure, with no numeric table in the README and no paper found **[repo, read directly]** — [Yinshuisiyuan23670/TrajectoryNetPP](https://github.com/Yinshuisiyuan23670/TrajectoryNetPP)
- A 2025 Procedia CS comparison reports that TrackNetV3 handles motion blur and occlusion better than YOLOv9 — [Procedia CS (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1877050925016709/pdf?md5=5c0a86280bebabf332cf3e75c6d8167b&pid=1-s2.0-S1877050925016709-main.pdf)
- A Springer SIVP paper (2025/26) proposes a lightweight attention-guided multi-scale fusion network for real-time shuttle detection and names motion blur and background interference as core difficulties (no numbers retrieved) — [Springer SIVP](https://link.springer.com/article/10.1007/s11760-025-05056-w)

### Inferences
- On the broadcast benchmark, TrackNetV3 (F1 ≈97.5–98.6 depending on protocol) is close to saturation. The successors add ≤1 F1 point. The independent re-implementation shows protocol and training choices (all frames vs "played" interval, rectifier on/off, threshold sweep, epochs) shift F1 by more than 15 points, which dwarfs the architecture deltas. **Paper-to-paper F1 comparisons are only meaningful when the protocol is pinned**, as gino6178 did by re-scoring TrackNetV3 weights.
- The most interesting new design for this project is the ICIP 2026 TrackNetV5-Lite: 2.58M params, explicit median-background channel and gated motion prompt. It is small, fast, and targets clutter suppression. Its license blocks use (see Q5).
- Heatmap/temporal models (TrackNet family, WASB, BlurBall) still beat single-frame YOLO detectors on tiny, blurred shuttles. YOLO variants win on deployment simplicity (ONNX/CoreML export).

### Gaps
- No 2025–26 tracker reports results on **amateur hall footage**, ultrawide/fisheye cameras or white-wall backgrounds. All headline badminton numbers are on the 26-match broadcast dataset.
- Could not retrieve BlurBall's own F1 row, TOTNet's per-badminton numbers, or the full TrackNetV5 (2512.02789) results table, because the primary PDFs were blocked.
- No published head-to-head of WASB vs TrackNetV4/V5 on the shuttlecock dataset under one protocol was found.
- Searched for an event-camera **tracker** and a WASB 2025/26 update; found neither, apart from the event-camera speed paper in Q3.

---

## Q2. Which methods address occlusion, motion blur, low contrast / background clutter (white-on-white) and broadcast→amateur domain shift? Any fine-tuning / self-training recipes?

### Takeaway
2025–26 work attacks blur (BlurBall's blur-center labels and blur prediction), occlusion (TOTNet's occlusion augmentation and visibility-weighted loss; TrackNetV3's InpaintNet) and clutter (median-background input channels; motion and frame-difference prompts in TrackNetV4/V5). **No paper specifically targets a white shuttle against white walls**, and **no shuttle-specific pseudo-labeling or self-training recipe was found**. Domain-shift evidence is sobering: a fine-tuned YOLO falls from F1 0.86 to 0.70 in unseen environments.

### Cited Findings
- **Background modelling (most relevant to white walls):** TrackNetV3 "leverages an estimated background as auxiliary data to locate the shuttlecock in spite of the fluctuating visual interferences". It uses the median image of the match, or of the rally if no match median exists, with `--max_sample_num` and `--video_range` controls for background estimation **[repo, read directly]** — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- ICIP-2026 TrackNetV5 makes the per-rally median a mandatory 4th channel (`subtract_concat`). Its gated motion prompt "suppresses background", unlike TrackNetV4's static frame differencing **[repo, read directly]** — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5)
- **Motion cues:** TrackNetV4 motion attention from frame differencing — [arXiv 2409.14543](https://arxiv.org/abs/2409.14543). TrackNetV5 (2512.02789) adds polarity-decoupled positive/negative frame differences (MDD) — [arXiv 2512.02789](https://arxiv.org/html/2512.02789v1)
- **Motion blur:** BlurBall labels the ball at the blur-streak center and regresses blur length and orientation. TT3D (arXiv 2504.10035, 2025) uses the blur info to segment trajectories, since blur relates to velocity — [cogsys-tuebingen/blurball](https://github.com/cogsys-tuebingen/blurball); [TT3D arXiv 2504.10035](https://arxiv.org/pdf/2504.10035)
- **Color/appearance domain shift (direct evidence):** TT3D notes that BlurBall was trained only on white balls and **cannot detect orange ones** — [TT3D arXiv 2504.10035](https://arxiv.org/pdf/2504.10035)
- **Occlusion:** TOTNet's occlusion augmentation (masking the ball region) plus a visibility-weighted loss raises fully-occluded accuracy from 0.63 to 0.80 **[self-reported]** — [arXiv 2508.09650](https://arxiv.org/html/2508.09650). TrackNetV3's InpaintNet rectifier and TrajectoryNet++ fill missing segments — [TrackNetV3](https://github.com/qaz812345/TrackNetV3); [TrajectoryNetPP](https://github.com/Yinshuisiyuan23670/TrajectoryNetPP)
- **Clutter as a stated failure mode:** the ETH mobile-robot paper says performance "depends heavily on shuttlecock size and background texture complexity". F1 is 0.86 in seen environments and 0.70 in unseen ones across 11 indoor and outdoor backgrounds — [arXiv 2603.06691](https://arxiv.org/html/2603.06691v2)
- **Amateur footage via data diversity, not adaptation:** a GitHub YOLOv8 project reports 76.4% mAP@50 on a mixed BWF-broadcast + amateur test set and claims it generalizes to amateur footage without extra fine-tuning **[repo claim, not peer-reviewed, unverified]** — [ZAYN005/badminton-shuttlecock-detection](https://github.com/ZAYN005/badminton-shuttlecock-detection)
- **Fine-tuning recipe (published):** TrackNetV4 initializes from pretrained TrackNetV2/V3 weights, explores LR 1e-3/1e-4/1e-5, trains 30 epochs, and selects the best validation checkpoint. Inputs are 512×288, 3 frames in, 3 heatmaps out **[repo, read directly]** — [TrackNetV4 docs/RESULT.md](https://github.com/TrackNetV4/TrackNetV4)
- ICIP-2026 TrackNetV5 recipe: seq_len 5, Adam LR 1e-3, 30 epochs, batch 10 (Lite), mixup α 0.5, bf16 AMP, deep-supervision weight 0.25, trained on train+val **[repo, read directly]** — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5)
- **Semi-automatic annotation:** the ETH paper's labeling pipeline produces labels from stationary-camera footage. It is the closest 2026 example of cheap labeling for a new domain — [arXiv 2603.06691](https://arxiv.org/html/2603.06691v2)
- **Self-training (earlier, not shuttle-specific):** Vandeghen et al. (CVPRW 2022) train a teacher on few labels, have it pseudo-label unlabeled soccer video, train a student weighted by confidence, and report substantially improved ball detection — [arXiv 2204.06859](https://arxiv.org/pdf/2204.06859)
- **Frame duplication:** BlurBall (multi-frame MIMO) is sensitive to duplicated frames from re-encoding and dedups before inference **[repo, read directly]** — [cogsys-tuebingen/blurball](https://github.com/cogsys-tuebingen/blurball)

### Inferences
- For white shuttle vs white wall, a **static camera plus median-background channel** is likely the strongest cue: the shuttle is a small moving residual against a stable background. Explicit motion and difference prompts (V4/V5) attack the same problem. This favors keeping or strengthening TrackNetV3's background input and its temporal window over swapping to a single-frame detector. Not verified on white-wall footage.
- The BlurBall "white-ball only" failure, plus ETH's 0.86→0.70 drop, suggest broadcast-trained weights will degrade on DJI hall footage. **Fine-tuning on a few hundred to a few thousand labeled club frames is likely required**, and a teacher-student pseudo-label loop (Vandeghen-style, confidence-filtered on TrackNet heatmaps, with InpaintNet/trajectory consistency as the filter) is a plausible but unpublished recipe for shuttles.
- Blur-center labeling (BlurBall) is a cheap convention change that could also help the user's own annotations, because the indoor shuttle often appears as a streak.

### Gaps
- **No paper or repo found on white-shuttle-vs-white-wall or low-contrast shuttle detection.** The project's first-class requirement is unaddressed in the literature.
- No published broadcast→amateur domain-adaptation study for shuttle tracking (UDA, self-training, test-time adaptation).
- No results on ultrawide/fisheye-undistorted input.

---

## Q3. State of monocular 3D shuttle trajectory reconstruction (MonoTrack successors, drag models, learned lifting, landing in/out, speed): what accuracy is realistic and what are the limits?

### Takeaway
The only new badminton-specific monocular 3D method found is **FreeTrack** (Machine Vision and Applications, Oct 2026). It drops drag-physics fitting for a per-shot **vertical motion plane** anchored at the hit event and ground projection, and claims landing errors fall by about 10× (e.g., Clear 170 cm → 7.8 cm). Learned 2D→3D "uplifting" trained on synthetic physics data is now strong in **table tennis** (WACV 2026), but not yet in badminton. Real-world 3D ground truth is almost never available, so most badminton 3D numbers are on synthetic data or are reprojection errors, and speed estimates from monocular 2D can overestimate peak speed badly.

### Cited Findings
**MonoTrack (earlier, CVPRW 2022, Liu & Wang)**
- Per-shot nonlinear optimization of a drag-only physics model. It needs court geometry, pose and 2D shuttle points, and is evaluated on 10k simulated trajectories — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf); [arXiv 2204.01899](https://arxiv.org/pdf/2204.01899)
- On real data there is **no 3D ground truth**, so only 2D reprojection error is reported. End-to-end error is about 4× that of the GT-hit version, with a max of 37.1 px (~5% of image size). The authors say error correlates strongly with flight time **[paper, via search extract]** — [ar5iv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- Synthetic 3D error falls from 14.9 cm to 8.0 cm with domain priors; HitNet hit-detection F1 is 0.946 vs 0.699 for derivative-based detection **[secondary, unverified — Lacuna summary]** — [Lacuna](https://lacuna.tiptreesystems.com/work/monotrack-shuttle-trajectory-reconstruction-from-monocular-badminton-video/wrk_5222592f6d03c81bc5d0472cd0224fbd)

**FreeTrack (Yu, Jiang, Kuang, Xiao; Machine Vision and Applications vol. 37, art. 167, published 2 Oct 2026)**
- Argues that "physics-prior methods inherently under-fit" high-drag, non-parabolic badminton flights that vary by stroke type. It assumes each flight lies in a vertical "Motion Plane" and reconstructs that plane per segment ("Temporal-Spatial Anchoring") from detected striking events and estimated ground-projection points — [Springer MVA](https://link.springer.com/article/10.1007/s00138-026-01929-3)
- Claims landing-point errors drop by an order of magnitude on complex strokes, e.g., **Clear 170 cm → 7.8 cm** **[paper, self-reported, via search extract; baseline identity not confirmed]** — [Springer MVA](https://link.springer.com/article/10.1007/s00138-026-01929-3)
- Authors listed as Kun Yu, Changlong Jiang, Haohong Kuang, Yang Xiao (per a ResearchGate listing quoted by search) **[secondary]** — [Springer MVA](https://link.springer.com/article/10.1007/s00138-026-01929-3)

**Learned 2D→3D lifting (table tennis / tennis; transferable designs)**
- Uplifting Table Tennis (Kienzle, Ludwig, Lorenz, Satoh, Lienhart; WACV 2026): two stages. 2D front-end detectors are trained on the new TTHQ dataset; a back-end **uplifting transformer is trained only on physically correct synthetic data** (MuJoCo) and generalizes zero-shot to real broadcast video. Scale: 140k synthetic trajectories, 4.7M 2D positions. Robustness tricks: randomly drop 5% of ball/table keypoints and sample 20–60 FPS during training. Model chosen as a compromise between spin F1 and m2DRE (numeric values not retrieved) — [arXiv 2511.20250](https://arxiv.org/pdf/2511.20250); [WACV supplemental](https://openaccess.thecvf.com/content/WACV2026/supplemental/Kienzle_Uplifting_Table_Tennis_WACV_2026_supplemental.pdf); repo [KieDani/UpliftingTableTennis](https://github.com/KieDani/UpliftingTableTennis) **[repo, read directly]**
- Predecessor: "Towards ball spin and trajectory analysis in table tennis broadcast videos via physically grounded synthetic-to-real transfer" (CVPRW 2025) with the TTST dataset — [ar5iv 2504.19863](https://ar5iv.labs.arxiv.org/html/2504.19863)
- TT3D (2025): table-tennis full-rally 3D reconstruction using BlurBall detections plus blur cues — [arXiv 2504.10035](https://arxiv.org/pdf/2504.10035)
- (earlier) SynthNet (Ertner, Konglevoll, Ibh, Graßhof; MMSports '24): tennis. A GRU detects hits/bounces; a fully connected network trained purely on synthetic data reconstructs 3D — [ITU Pure](https://pure.itu.dk/en/publications/synthnet-leveraging-synthetic-data-for-3d-trajectory-estimation-f/); [PDF](https://stellagrasshof.com/assets/pdf/2024_ertner_mmisport.pdf). A DTU/ITU thesis summary reportedly describes a badminton "TrajTrans" model that predicts initial 3D conditions from 2D, trained on filtered synthetic shots **[secondary, could not verify architecture or numbers]** — [ITU PhD defence PDF (M. Ibh)](https://en.itu.dk/-/media/EN/Research/PhD-Programme/PhD-defences/2025/Magnus-Ibh.pdf)

**Speed estimation**
- Monocular phone system (Huang, arXiv 2509.05334, Sep 2025): uses a **2D planar assumption** and is validated on only 20 smashes against a radar gun. "Peak speed" heavily overestimates (radar 132 vs 214.6 km/h; 119 vs 261.2). "Speed at net" tracks radar much more closely (128.6 vs 132; 178.2 vs 144; 57.9 vs 57). Stated limits: small sample, sub-maximal smashes (radar limits), cross-court shots underestimated, dependence on stable video and correct scale input **[paper, self-reported, via search extract]** — [arXiv 2509.05334](https://arxiv.org/html/2509.05334v1)
- Event cameras (Washida et al., MIZUNO; cs.CV, 28 May 2026): **two synchronized event cameras**, 125 smashes from 5 players, high-speed-camera reference. Shuttle speed bias −1.00 m/s (95% CI −2.46 to 0.46), impact-time bias ~1.84 ms, impact location estimable in 116/124 trials (93.5%), no proportional bias **[self-reported]** — [chatpaper listing](https://chatpaper.com/paper/287583)
- Physics context: shuttle drag coefficient ≈0.5–0.6 (bluff body); differentiating a measured trajectory twice to get drag is noise-sensitive — [ResearchGate: Aerodynamic properties of shuttlecocks](https://www.researchgate.net/publication/242463186_Measurements_of_Aerodynamic_Properties_of_Badminton_Shuttlecocks). World-class jump-smash shuttle speeds 56.8–64.9 m/s, attributed by a search summary to "a 2001 biomechanics paper" **[secondary, unverified; the linked ISBS volume 38 is from 2020, so the source mapping is uncertain]** — [ISBS / NMU commons](https://commons.nmu.edu/isbs/vol38/iss1/122)

**Stereo / multi-view (for context, not monocular)**
- YO-CSA-T (2025): stereo 3D plus trajectory prediction for a robot. The authors **could not measure 3D accuracy because 3D GT is "highly costly"** and used smoothness proxies instead — [arXiv 2501.06472](https://arxiv.org/pdf/2501.06472)
- BadmintonBot (GitHub, 2026): 2× Blackfly S at 100 FPS hardware-sync, triangulation, EKF landing prediction **[repo, read directly]** — [BadmintonBot-Capstone/badminton-bot](https://github.com/BadmintonBot-Capstone/badminton-bot)
- (earlier) Princeton thesis (2016): 3D error 0.034 court-length (~48 cm on 13.4 m) — [Princeton DataSpace](https://dataspace.princeton.edu/handle/88435/dsp01td96k495p)

### Inferences
- **Realistic accuracy for a monocular amateur setup:** landing points at roughly decimetre-to-metre level today. FreeTrack's ~8 cm on Clears is a best case with accurate hit events and ground projections on its own data. MonoTrack's ~8–15 cm figures are **synthetic**. Real-world 3D GT is essentially never reported, so any landing in/out call from one ultrawide camera should stay labeled „Schätzung" with an uncertainty band, especially near lines.
- FreeTrack's argument that drag-physics fits under-fit badminton is a direct challenge to the project's MonoTrack-style fit. A cheap hedge is to compare **both** (physics fit vs vertical-plane + hit/ground anchors) on the same shots and report the disagreement as uncertainty. This is an inference, not a published recipe.
- For speed: frame-to-frame "peak" speed from monocular 2D is unreliable (+60–120% overestimates in the phone study). Speed from a fitted initial-velocity model, or a speed sampled at a stable segment (e.g., at the net), is more defensible. The ±1 m/s accuracy is only shown with dual event cameras.
- A table-tennis-style **synthetic-trained uplifting transformer** (shuttle drag physics in simulation; random drop-outs and FPS jitter) is the most promising unexplored direction for badminton. The WACV 2026 code is GPL-3.0, so it is license-compatible for this non-distributed app.

### Gaps
- No public badminton monocular 3D method with **real measured 3D ground truth** was found (2025–26).
- FreeTrack: code/tool release and Badminton3D availability unconfirmed ("will release"). Its baseline for the 170 cm figure, its camera setup (broadcast vs amateur), and its speed accuracy were not retrievable.
- No dedicated monocular **in/out line-call accuracy** study for badminton was found.
- Could not verify the badminton "TrajTrans" thesis claims.

---

## Q4. Are there new shuttle datasets (2D or 3D, amateur footage, multi-view ground truth)?

### Takeaway
New 2025–26 datasets are mostly **2D or semantic**: ETH egocentric 20.5k frames, RacketVision (~435k frames across 3 racket sports), and table-tennis blur/occlusion sets. The only 3D shuttle resources are **promised (Badminton3D, with FreeTrack-estimated rather than measured 3D)** or **access-gated (a 247-trajectory stereo/Vicon set)**. No public amateur-hall multi-view 3D shuttle ground truth exists that I could find.

### Cited Findings
- (earlier, still the benchmark) Shuttlecock Trajectory Dataset (TrackNetV2/V3): 26 broadcast matches split into rallies, 1280×720 frames, per-frame visibility and X,Y. The test set has 29 rallies; the TrackNetV4 protocol scores 10,836 "played" frames via `drop_frame.json` — [gino6178/TrackNetV5 README](https://github.com/gino6178/TrackNetV5); [HackMD dataset page](https://hackmd.io/@TUIK/rJkRW54cU)
- **Badminton3D** (promised with FreeTrack, Oct 2026): videos, 2D shuttle annotations, striking-moment annotations and **3D trajectories estimated by FreeTrack** (not measured) — [Springer MVA](https://link.springer.com/article/10.1007/s00138-026-01929-3)
- **Stereo/Vicon set** referenced by BadmintonBot: 247 trajectories at 120 FPS, stereo video, 2D annotations, 3D CSVs, camera configs, Vicon ground truth (3.6 cm mean error). Public access, origin and license are unclear (open request in issue #11). The repo's own cameras run at 100 FPS, a possible inconsistency — [BadmintonBot issue #11](https://github.com/BadmintonBot-Capstone/badminton-bot/issues/11); [repo](https://github.com/BadmintonBot-Capstone/badminton-bot)
- ETH egocentric shuttle set (Mar 2026): 20,510 semi-automatically annotated frames, 11 indoor/outdoor backgrounds, 3 difficulty tiers — [arXiv 2603.06691](https://arxiv.org/html/2603.06691v2)
- RacketVision (arXiv 2511.17045; AAAI 2026): table tennis, tennis and badminton at 1080p, ~461 games, ~435k frames, ~88k annotations, with ball position plus first racket-pose labels. Benchmarks ball tracking, racket pose and trajectory forecasting. License not found — [arXiv 2511.17045](https://arxiv.org/html/2511.17045v2); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37362)
- BlurBall dataset: 64k real table-tennis frames with ball position and blur length/orientation (Nextcloud) — [cogsys-tuebingen/blurball](https://github.com/cogsys-tuebingen/blurball)
- TTA (TOTNet): table tennis, 2,396 occluded instances (998 fully occluded), "most occlusion-rich public dataset" per the authors — [CVIU author PDF](https://rbouadjenek.github.io/assets/pdf/YCVIU_104657.pdf)
- TrackNetV4 "new tennis dataset" (test 8,472 frames) — [TrackNetV4 docs/RESULT.md](https://github.com/TrackNetV4/TrackNetV4)
- Non-tracking badminton sets (2025–26): FineBadminton (3,215 rally clips, 33,325 strokes, multi-level semantics) — [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554); BFMD full-match dense shot captioning — [arXiv 2603.25533](https://arxiv.org/pdf/2603.25533); BadmintonGRF (multi-view video + mocap ~120 FPS, 1,732 impact trials, ground-reaction force) — [arXiv 2605.01876](https://arxiv.org/pdf/2605.01876); Ultralytics broadcast shuttle set, 8,053 images — [Ultralytics platform](https://platform.ultralytics.com/posathorn/datasets/shuttlecock-dataset)

### Inferences
- For the white-wall hall problem, **the club's own labeled DJI footage will be the most valuable dataset**. Nothing public matches its domain.
- Multi-view mocap sets built for biomechanics (BadmintonGRF) might contain shuttle-visible multi-view footage that could be triangulated for 3D validation. This is unverified and would need checking.

### Gaps
- No public **amateur-hall** shuttle dataset with **measured multi-view 3D GT** was found.
- License terms for RacketVision, Badminton3D and the BadmintonBot stereo set are unknown.

---

## Q5. Which have code + weights, under which licenses, and how portable are they to PyTorch MPS / CoreML / ONNX?

### Takeaway
**Usable under the project's rule (an explicit license):** TrackNetV3 (MIT, code + weights), WASB (MIT, per-sport weights incl. badminton), BlurBall (MIT, weights on Nextcloud, but table tennis), TrackNetV4 official (MIT, but TensorFlow/Keras and **weight links are placeholders**), and Uplifting Table Tennis (GPL-3.0, torch.hub weights). **Blocked (no license grant or proprietary):** both "TrackNetV5"s, plus the unofficial V1–V5 PyTorch port, the TrackNetV4 PyTorch port and TrajectoryNet++. Only TrackNetV3's `predict.py` (and the unlicensed series port) has an MPS code path. No CoreML/ONNX export of any TrackNet variant was found.

### Cited Findings
| Project | License (as read) | Code | Weights | Framework / device notes |
|---|---|---|---|---|
| (earlier) TrackNetV3 | **MIT**, explicitly covering code **and** Google-Drive checkpoints | yes | yes (TrackNet + InpaintNet ckpts) | PyTorch; `predict.py` selects `torch.device("mps")` when `torch.backends.mps.is_available()` — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) **[read directly]** |
| (earlier) WASB-SBDT | **MIT** (NTT 2023) | yes | yes, per sport incl. badminton (Google Drive) | PyTorch; Dockerfile `nvcr.io/nvidia/tensorrt:21.06`; no MPS path found by grep (CUDA-oriented) — [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) **[read directly]** |
| BlurBall (2025/26) | **MIT** (LICENSE.md inherited from WASB) | yes | yes (BlurBall + WASB/TrackNetV2/ResTrackNetV2/BallSeg/DeepBall/MonoTrack, table-tennis-trained) | PyTorch/Hydra; CUDA-oriented, no MPS path found; dedups duplicate frames — [cogsys-tuebingen/blurball](https://github.com/cogsys-tuebingen/blurball) **[read directly]** |
| TrackNetV4 official | **MIT** (2025) | yes | **No real links**: `docs/RESULT.md` "Download" links point to `#` | **TensorFlow/Keras** (`.keras`); integrates with V2, model code for V3 — [TrackNetV4/TrackNetV4](https://github.com/TrackNetV4/TrackNetV4) **[read directly]** |
| TrackNetV4 PyTorch port | **no LICENSE file** | yes | unclear | forked from the series repo — [AIKnowU/tracknet-v4-pytorch](https://github.com/AIKnowU/tracknet-v4-pytorch) **[read directly]** |
| TrackNetV5 (arXiv 2512.02789) SDK | **Proprietary**: "All rights reserved… provided for technical exchange and academic study only" | yes | **No**: "weights and training datasets are proprietary… not open to the public" | PyTorch 2.9 + CUDA 12.6 — [codelancera-offical/TrackNetV5-SDK](https://github.com/codelancera-offical/TrackNetV5-SDK) **[read directly]** |
| TrackNetV5 (ICIP 2026, Chang/Wang/Huang) | **no LICENSE file** in repo (CC BY-SA 4.0 notice is the project-website footer, not the code) | yes | yes, `releases/TrackNetV5_Lite_best.pt`, `TrackNetV5_best.pt` | PyTorch 2.7 cu128 on RTX 5080; bf16 AMP training; CUDA-oriented — [gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5) **[read directly]** |
| tracknet-series-pytorch (V1–V5) | **no LICENSE file** | yes | README says "pre-trained models available"; artifacts excluded from git | Has MPS auto-device (`tracknet/utils/device.py`) and "mac_smoke" configs — [AnInsomniacy/tracknet-series-pytorch](https://github.com/AnInsomniacy/tracknet-series-pytorch) **[read directly]** |
| TrajectoryNet++ | **no LICENSE file** (TrackNetV3-derived) | yes | not evident | PyTorch — [Yinshuisiyuan23670/TrajectoryNetPP](https://github.com/Yinshuisiyuan23670/TrajectoryNetPP) **[read directly]** |
| TOTNet | n/a | **no public code found** | n/a | — [arXiv 2508.09650](https://arxiv.org/html/2508.09650) |
| Uplifting Table Tennis (WACV 2026) | **GPL-3.0** | yes | yes via `torch.hub`; also BlurBall-pretrained Segformer++ | ViTPose util has an MPS check — [KieDani/UpliftingTableTennis](https://github.com/KieDani/UpliftingTableTennis) **[read directly]** |
| FreeTrack / Badminton3D | n/a | "will release" (unconfirmed) | n/a | — [Springer MVA](https://link.springer.com/article/10.1007/s00138-026-01929-3) |

- ONNX: YO-CSA-T reports an ONNX-accelerated YOLO pipeline at 5.82 ms/frame — [arXiv 2501.06472](https://arxiv.org/pdf/2501.06472). A TensorRT-INT8 TrackNetV3 fork exists (CUDA only) — [nickluo/TrackNetV3](https://github.com/nickluo/TrackNetV3)
- CoreML: Ultralytics documents CoreML export for YOLO models — [Ultralytics CoreML docs](https://docs.ultralytics.com/de/integrations/coreml). A late-2025 Apple developer-forum post reports the ONNX→CoreML ingestion path broken in current tooling **[user report, unverified]** — [Apple Developer Forums](https://developer.apple.com/forums/topics/machine-learning-and-ai/machine-learning-topic-core-ml?page=3)

### Inferences
- Under the project's license rule, the **practical 2D upgrade path is still TrackNetV3 + WASB** (both MIT with weights), optionally adding **BlurBall's blur-aware ideas** (MIT, but needs retraining on shuttles). TrackNetV4's motion-attention idea is MIT-licensed but has to be re-implemented and retrained in PyTorch. The official code is TF/Keras and ships no weights. The project could write its own clean-room PyTorch version; the unlicensed ports cannot be copied.
- The ICIP-2026 TrackNetV5-Lite is the best size/accuracy trade-off on paper. With **no LICENSE file it counts as "no license grant" and should be avoided** unless the authors add one. Asking them would be one email. The other TrackNetV5 is proprietary.
- TrackNet-family models are plain 2D CNN/U-Nets at a fixed 512×288 input, so a CoreML fixed-shape export via `coremltools` from PyTorch is technically plausible. This is untested here, and the project's CLAUDE.md rule is to benchmark it against MPS. ICIP-V5's R(2+1)D (3D-conv) blocks and bf16 training may need fp32 and MPS-fallback checks.

### Gaps
- No MPS or CoreML benchmark (FPS, numerical parity) was found for any TrackNet/WASB/BlurBall model. All published FPS figures are on NVIDIA GPUs (e.g., RTX 5080 for ICIP-V5) or unspecified hardware.
- The WASB weights' license is presumably MIT like the repo, but the MODEL_ZOO does not state a separate weights license. TrackNetV3 states it explicitly.
- RacketVision code/data license and FreeTrack release status remain unverified.
