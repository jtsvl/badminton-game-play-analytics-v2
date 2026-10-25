# Own footage, not newer models, decides accuracy

The research from 2025 to October 2026 mostly **confirms the pipeline's architecture, and few of the upgrades it offers are licence-clean and likely to move the repo's numbers**. TrackNetV3 with a background channel is still the best shuttle tracker the repo is allowed to use. Its successors, TrackNetV4 and two unrelated papers that both call themselves "TrackNetV5", add under one F1 point on broadcast footage. Each of them either ships no PyTorch weights or has no usable licence. BST is still the reference stroke model with MIT code and weights. RTMPose-m still matches or beats the new real-time pose models. YOLO plus BoT-SORT with court gating is still what racket-sport papers use. The strongest tactical-analysis systems all follow the repo's own pattern: computer vision writes a structured event log and a language model only talks about it. They do this because zero-shot video-language models (VLMs) score only **about 25–39 % on fine-grained badminton questions**.

The more important finding is a gap. **No 2025–26 paper tests anything on amateur hall footage, ultrawide lenses, or a white shuttle against white walls.** The club's own labelled clips are therefore the only evidence that counts. Against that background, the research exposes risks in how the repo measures, not in which models it picked:

- The shuttle tracker runs at **12 fps in production but was trained at 30 fps**.
- Its 0.953 F1 is in-sample (scored on the same clip it was trained on).
- The failed stroke gate (0.40) fits a field-wide pattern: hit detection, not stroke classification, is the weak link.
- The spec still lists two "usable" datasets that have no licence.

Industry news touches the code in only one place. BWF's **3×15 scoring starts 4 January 2027**, and the web score parser already handles it.

> **Read this before trusting any number below.** The research proxy blocked most primary paper sites: arXiv, CVF Open Access, Hugging Face, Semantic Scholar, Springer, alphaXiv and BWF's own pages. GitHub was readable, so licences, READMEs and repo result tables are first-hand. **Nearly every paper figure in this report is self-reported by its authors and was taken from search-engine extracts, not from the PDFs.** Several of them differ between versions of the same paper; those conflicts are flagged. Use these numbers to decide what to try, not as facts to build on, and open the paper before acting on any single figure. Industry facts also come from search summaries of news pages. The only independent re-measurements in this report are the ones explicitly labelled as such.

## Perception models gained single points on broadcast TV

### Shuttle trackers have hit a benchmark ceiling

On the standard broadcast shuttle dataset, TrackNetV3 already reports **F1 98.56 %**. Its MIT licence explicitly covers the checkpoints, and `predict.py` already has an MPS code path ([TrackNetV3](https://github.com/qaz812345/TrackNetV3)).

**TrackNetV4** (ICASSP 2025) adds learned motion-attention maps built from frame differences. Its paper summary claims 97.5 → 97.9 F1 over V3 ([alphaXiv](https://www.alphaxiv.org/abs/2409.14543)), but its own repo shows 96.8 → 97.1. The repo is MIT-licensed, but it is TensorFlow/Keras and its weight download links point nowhere ([TrackNetV4](https://github.com/TrackNetV4/TrackNetV4)).

The two "TrackNetV5" papers are unrelated:
- **The arXiv paper from a Shanghai company** is proprietary, and its weights are private ([SDK](https://github.com/codelancera-offical/TrackNetV5-SDK)).
- **The ICIP 2026 paper from the TrackNetV3 group** is the most interesting design for a white-wall hall. It has a **2.58 M-parameter Lite model at F1 98.06**, a mandatory per-rally median-background channel, and a gate meant to suppress the background. However, its repo has **no LICENSE file** ([gino6178/TrackNetV5](https://github.com/gino6178/TrackNetV5)).

The sobering data point is an **independent** re-implementation of V1–V5 under one shared protocol. There, V3 with its rectifier scored best (F1 0.809) and V5 scored worst (0.773) ([tracknet-series-pytorch](https://github.com/AnInsomniacy/tracknet-series-pytorch)). Evaluation choices move F1 by more than 15 points, which swamps every architecture gain claimed since 2023.

The robustness ideas are more useful than the new architectures:
- **BlurBall** (CVPRW 2026, MIT) labels the ball at the **centre of its blur streak** and predicts the blur itself. It also removes repeated frames before inference, because multi-frame models break on them ([BlurBall](https://github.com/cogsys-tuebingen/blurball)).
- **TOTNet** masks the ball during training. It reports fully-occluded accuracy rising from 0.63 to 0.80, but has no public code ([arXiv 2508.09650](https://arxiv.org/html/2508.09650)).

Domain shift is the consistent warning. A fine-tuned YOLOv8 shuttle detector falls from **F1 0.86 to 0.70 in environments it has not seen** ([ETH, arXiv 2603.06691](https://arxiv.org/html/2603.06691v2)). A table-tennis detector trained on white balls cannot see orange ones ([TT3D](https://arxiv.org/pdf/2504.10035)). **No paper or repo addresses a white shuttle against white walls**, and no shuttle-specific self-training recipe exists. The closest is a soccer scheme from 2022, where a "teacher" model trained on a few labels creates pseudo-labels for unlabelled video and a "student" model trains on them ([Vandeghen et al.](https://arxiv.org/pdf/2204.06859)).

### Monocular 3D still lacks real ground truth

The one new badminton 3D method is **FreeTrack** (Machine Vision and Applications, 2 Oct 2026). It argues that drag-physics fits like MonoTrack under-fit badminton's high-drag flights. Instead, it fits each shot to a vertical plane anchored at the hit and at the shuttle's projection onto the ground. It claims Clear landing errors fall from **170 cm to 7.8 cm**, with these caveats ([Springer MVA](https://link.springer.com/article/10.1007/s00138-026-01929-3)):
- The extracts did not identify which baseline produced the 170 cm.
- The code and its Badminton3D dataset are only promised.
- That dataset's 3D trajectories are FreeTrack's own estimates, not measurements.

MonoTrack itself never had real 3D ground truth, only 2D reprojection error ([ar5iv](https://ar5iv.labs.arxiv.org/html/2204.01899)). Table tennis is further ahead. A WACV 2026 transformer trained only on synthetic physics turns 2D tracks into 3D, zero-shot, on broadcast video. It was trained with random dropouts and frame-rate jitter between 20 and 60 FPS, and is released under GPL-3.0, which the repo's rule allows ([arXiv 2511.20250](https://arxiv.org/pdf/2511.20250); [repo](https://github.com/KieDani/UpliftingTableTennis)).

Speed is where single-camera systems mislead most:
- A phone system validated on 20 smashes reported **peak speeds of 214.6 and 261.2 km/h where radar read 132 and 119**. Speed at the net tracked radar far better ([arXiv 2509.05334](https://arxiv.org/html/2509.05334v1)).
- Radar itself reads about 9.7 % low compared with high-speed video ([IJRSS](https://revistaseug.ugr.es/index.php/IJRSS/article/view/33259)).
- Accurate speed (bias −1.00 m/s) has only been shown with two synchronised event cameras ([chatpaper listing](https://chatpaper.com/paper/287583)).
- BWF-approved line calling uses rigs like Reveal Lens's **12 synchronised high-speed cameras** ([SoyaCincau](https://soyacincau.com/2026/08/25/malaysian-ai-badminton-review-system-reveal-lens-antica-singapore-international-challenge/)).

### Detection, tracking, pose and court work improved on benchmarks unlike a hall

New detectors gain 1–5 points on the COCO benchmark, mostly on medium and large objects. That is not where two to four players on one court fail.
- **YOLO26** (January 2026) needs no NMS post-processing step (removing duplicate boxes), which makes CoreML export simpler. It uses the same AGPL licence the repo already accepts ([Ultralytics](https://github.com/ultralytics/ultralytics); [docs](https://docs.ultralytics.com/models/yolo26/)).
- **RF-DETR** (ICLR 2026) is Apache-2.0 for its N–L sizes and has an official native CoreML export ([RF-DETR](https://github.com/roboflow/rf-detr); [CoreML cookbook](https://rfdetr.roboflow.com/latest/cookbooks/export-coreml/)). One user reports that an upsampling operation falls back from MPS to the CPU ([Roboflow forum](https://discuss.roboflow.com/t/can-rf-detr-run-with-mac-silicon/10580)).
- A review of DEIMv2 notes that the latest generation's gains leave small-object accuracy roughly flat ([Moonlight](https://www.themoonlight.io/review/real-time-object-detection-meets-dinov3)). The far-court player problem is untouched.

**Tracking.** The only independent benchmark run under the same conditions puts **BoT-SORT at 73.8 HOTA** (the standard tracking score) on SportsMOT. That is ahead of ByteTrack and OC-SORT, and behind only McByte at 76.5, which propagates segmentation masks between frames ([Roboflow trackers](https://github.com/roboflow/trackers)). Gains above that come from three directions:
- **Offline tracklet association.** GTA (MIT) adds 3.83 % HOTA on top of Deep-EIoU, reaching 81.0 ([arXiv 2411.08216](https://arxiv.org/pdf/2411.08216); [gta-link](https://github.com/sjc042/gta-link)).
- **Keypoint cues.** These lift CAMELTrack (Apache-2.0) from 66.1 to 69.3 HOTA on DanceTrack ([CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack)).
- **SAM 3 mask hand-offs.** These reach a self-reported 87.2 HOTA, but they are built on Deep-EIoU, which still has no licence ([arXiv 2606.13033](https://arxiv.org/pdf/2606.13033); [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)). SAM 3 also requires CUDA ≥ 12.6 ([sam3](https://github.com/facebookresearch/sam3)).

SportsMOT contains no racket sport, so none of these numbers transfers directly.

**Pose.** The field moved toward heavy foundation models and one-stage real-time models. Neither displaces RTMPose-m's **75.8 COCO AP** ([arXiv 2303.07399](https://arxiv.org/abs/2303.07399)):
- YOLO26-pose-x reaches 71.6, RF-DETR Keypoint 71.8, and DETRPose-X 73.3 ([DETRPose](https://github.com/SebastianJanampa/DETRPose)).
- Sapiens2 is built for NVIDIA GPUs (CUDA) first and has a custom licence ([sapiens2](https://github.com/facebookresearch/sapiens2)).
- MMPose's last tagged release is from January 2024 ([mmpose](https://github.com/open-mmlab/mmpose)). rtmlib, the wrapper the repo already uses, is Apache-2.0 and still runs the whole RTM family plus ViTPose ([rtmlib](https://github.com/Tau-J/rtmlib)).
- Motion blur remains unsolved beyond training on synthetically blurred images, which improved baseball pose estimators by 29.2 % on average ([paper](https://astro.paperswithcode.com/paper/mitigating-motion-blur-for-robust-3d-baseball)).

**Court calibration.** Research here is still led by soccer, and SoccerNet dropped its calibration challenge in 2025 ([SoccerNet 2025](https://arxiv.org/pdf/2508.19182)). The proven recipe for racket courts is a keypoint-heatmap network followed by homography refinement. TennisCourtDetector reaches a 1.83 px median error that way, but it has no licence ([TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)). Two pieces matter for the DJI lens:
- **AnyCalib** (ICCV 2025) recovers the camera's lens parameters (intrinsics), including fisheye models, from a single image ([ICCV](https://openaccess.thecvf.com//content/ICCV2025/html/Tirado-Garin_AnyCalib_On-Manifold_Learning_for_Model-Agnostic_Single-View_Camera_Calibration_ICCV_2025_paper.html)).
- **PnLCalib** added lens-distortion optimisation in March 2026, under GPL-2.0 ([PnLCalib](https://github.com/mguti97/PnLCalib)).

**No published method picks the target court out of several adjacent ones.** The locked decision (auto-propose, user picks the court, drag the corners) is therefore still the state of practice.

## Understanding moved to hit timing, rally context and LLMs over logs

### BST still leads stroke classification, but only with perfect hit timing

BST (CVPRW 2026, MIT, public weights) remains the reference model that combines pose, shuttle and court position. Its own results workbook shows how little the architecture matters with the full dataset ([result_table.xlsx](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/results_and_images/result_table.xlsx)):

| ShuttleSet setting | BST-CG-AP (acc / macro-F1) | TemPose-TF (acc / macro-F1) |
|---|---|---|
| 35 classes, full data | 0.770 / 0.704 | 0.768 / 0.703 |
| 25 merged classes, full data | 0.825 / 0.798 | 0.819 / 0.794 |
| 35 classes, 25 % of data (best BST variant, BST-CG) | 0.644 / 0.527 | 0.582 / 0.437 |

Even at full data, the worst class sits at F1 0.24. The gap between models opens with less data, and that is the property a small club needs.

Two details explain why BST's numbers do not transfer to a club pipeline:
- Every training and test clip is cut around ShuttleSet's **annotated hit frame** ([gen_my_dataset.py](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/ShuttleSet/gen_my_dataset.py)).
- The shuttle input is normalised by video resolution, so it is tied to the broadcast camera ([BST README](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)).

The README also settles a contradiction in the repo's specs. ShuttleSet natively annotates 18 shot types plus backhand and around-head flags ([ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)). BST derives its own 35-class (34 + None) and 25-class merged label sets from that, so **both spec statements are true at different levels**.

The one 2026 successor, TemPose-TF-ASF, reports 0.854 / 0.761 by feeding in the predicted types of the neighbouring strokes. Its protocol clearly differs from BST's, and no code is released ([arXiv 2605.02558](https://arxiv.org/html/2605.02558v1)). For doubles, Baek & Yun report (in the abstract only) that pose-based models transfer from singles to doubles "2× better" than vision-only ones ([arXiv 2508.13507](https://www.arxiv.org/abs/2508.13507)). There is no benchmarked method for deciding which partner hit the shuttle.

### Hit detection is now the field's weakest link

The only measured comparison on real non-broadcast footage is a third-party test on 174 hand-labelled hits, scored within ±5 frames:

| Method | F1 |
|---|---|
| Hand-written rules | **67.7 %** |
| SwingNet | 68.6 % |
| Fusion of both | 70.8 % |

The fusion also produced 34 % more hits than actually happened. The test's authors give a rule for how errors carry through: end-to-end precision on typed strokes is 0.618 × the classifier's accuracy on correctly detected hits ([ASSESSMENT.md](https://github.com/dengty2002-afk/badminton-data-analysis/blob/main/experiments/swingnet/downstream-impact-20260913/ASSESSMENT.md)).

The strongest new method, **UMEG-Net** (AAAI 2026), builds one graph from player skeletons, shuttle and court-corner keypoints. It learns from few examples and beats the best skeleton-only baseline by 2.1 % event-F1 on ShuttleSet, but its repo has no licence file ([UMEG-Net](https://github.com/LZYAndy/UMEG-Net); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37681)).

Rally segmentation is easier on broadcast footage than on amateur footage:
- About 95 % on broadcast, using a serve detector plus a "no shuttle motion" end rule ([See et al.](https://research.monash.edu/en/publications/a-comparative-analysis-of-deep-learning-models-and-gradient-compu/)).
- Only **75 % on amateur rallies** in TactiPlay, the one amateur-focused system ([arXiv 2607.27125](https://arxiv.org/pdf/2607.27125)).

No badminton paper on detecting hits from audio exists. The nearest is combined audio and video hit detection in padel ([CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.pdf)).

### VLMs fail zero-shot; fine-tuned small models and vision logs win

Five racket-sport benchmarks give the same ordering: zero-shot open models score lowest, zero-shot frontier models next, and fine-tuned 7–8B open models best.
- **FBBench (FineBadminton):** Gemini 2.5 Pro reaches **38.6 %** and small open models stay **below 25 %**. A fine-tuned Qwen2.5-VL-7B beats both at 42.1 % ([arXiv 2508.07554](https://arxiv.org/pdf/2508.07554)).
- **DoublesEval:** Qwen3-VL scores 29.0 % and climbs to 37.2 % when it reranks its answers for self-consistency ([arXiv 2608.24439](https://arxiv.org/html/2608.24439v1)).
- **Tennis TRACE:** zero-shot Gemini-3.1-Pro scores 18.9, against 44.8 for a fine-tuned Qwen3-VL-8B ([arXiv 2608.12920](https://arxiv.org/pdf/2608.12920)).

Questions that need measurements are the worst. CourtSI finds distance measurement the weakest skill ([arXiv 2603.09896](https://arxiv.org/html/2603.09896)). In SportR, GPT-5, Claude 4 and Gemini 2.5 Pro all stay under 7 % IoU at locating things in the image ([VisLang Lab](https://vislang.ai/publications/xia2026sportr)). Dense frame sampling centred on the hit matters as much as model size ([ActionAtlas](https://arxiv.org/pdf/2410.05774)). Qwen3.5-4B, the base of the repo's VideoJev pilot, publishes only general video scores, none of them sports-specific ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)).

The tactical systems that work avoid having a VLM watch the video:
- **COACH** reports that a 3B Flan-T5-XL reading ShuttleSet-derived logs beats Gemini 2.5 Pro ([arXiv 2512.01853](https://arxiv.org/html/2512.01853v1)).
- **The CVPR 2026 Findings debate paper** states outright that VLMs lack the badminton knowledge to read tactical intent, so it converts footage into text logs first ([CVF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhang_From_Alignment_to_Reason_Multi-Agent_Debate_for_Tactical_Badminton_Video_CVPRF_2026_paper.pdf)).

One adjacent study adds a warning for labelling. AI pre-annotations cut labelling time by about 35 %, but they pull human labels toward what the model predicted ([arXiv 2510.21798](https://arxiv.org/html/2510.21798)).

### New datasets are broadcast-only and licence-tangled

2025–26 added large badminton datasets, all elite broadcast footage:

| Dataset | Size | Licence |
|---|---|---|
| FineBadminton ([GitHub](https://github.com/iLearn-Lab/MM25-FineBadminton)) | 33,325 strokes | Repo Apache-2.0; data-card licence unverified |
| BFMD ([BFMD](https://github.com/Ning-D/BFMD)) | 16,751 hits, singles and doubles | **No licence file**; BWF video marked "do not redistribute" |
| DShuttleNet ([arXiv 2610.14016](https://arxiv.org/html/2610.14016)) | 54,141 doubles shots, on Kaggle | Unverified; found in only one search, so verify it exists first |

**ShuttleSet remains the one large stroke dataset with a clean licence**: MIT, 36,492 strokes, with a homography for each match ([LICENSE](https://github.com/wywyWang/CoachAI-Projects/blob/main/LICENSE)). The two datasets the repo's spec names as "usable", VideoBadminton and BadmintonDB, have **no licence file** ([VideoBadminton](https://github.com/qilimk/VideoBadminton); [BadmintonDB](https://github.com/kwban/badminton-db)). No public amateur-hall dataset with measured 3D shuttle positions exists.

## Industry money flows to officiating rigs and phone apps

BWF's technology spending in 2025–26 went to officiating and TV production, not player analytics:
- Members approved **3×15 scoring** (best of three games to 15, extending to a 21-point cap at deuce) on 25 April 2026. It takes effect **4 January 2027** ([BWF](https://bwfbadminton.com/news-single/2026/04/25/bwf-members-approve-3x15-scoring-system)).
- BWF now lists five approved instant-review vendors, including **GroundTouch GmbH of Hamburg** ([BWF](https://corporate.bwfbadminton.com/events/approved-instant-review-system/)). Stupa's AI system was certified at 99 % accuracy against a 98 % threshold, under a test protocol that has not been published ([IT Brief](https://itbrief.in/story/stupa-wins-bwf-approval-for-badminton-review-system)).
- The Infront deal, running through 2034, promises "data-driven graphics" from 2027 without naming AI ([SVG Europe](https://www.svgeurope.org/blog/headlines/infront-to-provide-major-production-upgrade-as-part-of-renewed-badminton-world-federation-deal/)).
- The best template for broadcast-grade per-player stats is Stupa Cast at the 2025 European Championships. It showed serve success rate, rally count, movement areas, error distribution and winner placement ([Broadcast Now](https://www.broadcastnow.co.uk/production/stupa-sports-unveils-badminton-data-driven-insights/5206141.article)).

The repo's score parser (`web/src/lib/score.ts`) reads any "a:b" game score and counts game wins by comparing the two numbers, so 3×15 needs no change. Only a future rule check, such as "a game ends at 21", would need to take the target points as a parameter.

**Consumer products.** Most consumer apps now offer the same feature set: automatic rally clipping, shot classification, heatmaps, pose feedback on short clips, and an LLM coach chat on top. Prices run from €2.99/month at Peak ([Peak](https://badmintonpeak.com/en)) to $28/month at GGAB ([GGAB](https://app.getgoodatbadminton.com/)). **None of them publishes independently tested accuracy.** BadPro+'s claim of ">99 %" across 28 shot types is marketing ([BadPro+](https://badproplus.com/)). An academic YOLOv8 app prototype managed only 37.6 % shuttle detection, even on broadcast rallies ([MatchMotion](https://www.researchgate.net/publication/404597971_MATCHMOTION_AN_AI-POWERED_MOBILE_APPLICATION_FOR_AUTOMATED_BADMINTON_VIDEO_ANALYSIS_USING_YOLOV8_OBJECT_DETECTION)). SwingVision still supports only tennis and pickleball ([App Store](https://apps.apple.com/us/app/swingvision-%ED%85%8C%EB%8B%88%EC%8A%A4-%ED%94%BC%ED%81%B4%EB%B3%BC/id989461317?l=ko)).

**Funding.** Rounds are small and regional. Machaxi raised $1.5 M in India ([Entrepreneur India](https://india.entrepreneur.com/news-and-trends/machaxi-secures-15-million-to-expand-ai-powered-badminton/493070)). SportVision raised a round led by Hillhouse; its AI cameras installed in Chinese badminton halls are the closest commercial analogue to a fixed club camera ([PEdaily](https://news.pedaily.cn/202605/564067.shtml)).

**Open source.** The closest open-source analogue is **Good-Badminton**, with about 1k stars since June 2026. It covers pose, shuttle, court mapping and heatmaps, and its hit analysis is still marked experimental ([Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)). A fork's README describes the original code as Apache-2.0, but this was not confirmed on the original repo ([fork](https://github.com/seafood-rice/Good-Badminton)).

No product found in Europe targets fixed club cameras with per-member reports that track players over time.

## Stage by stage, most choices hold and four spec claims need fixing

| Stage | Research confirms | Worth considering (licence-OK, Mac-runnable) | Gap or risk exposed |
|---|---|---|---|
| **0 Decode & undistort** | Undistort → detect → homography is the standard order | AnyCalib as a one-time check of the DJI lens parameters; PnLCalib's distortion refinement as a reference | Undistortion is still a no-op on footage whose court lines visibly bow; the 12 fps frame cache is below every research dataset |
| **1 Court calibration** | Auto-propose, user picks the court, drag corners: still the state of practice | Own 20–30-keypoint heatmap model + RANSAC fit to the BWF court template (re-implement the TennisCourtDetector recipe); default to the largest central court outline | Auto-detect is not built. A 4-point fit has zero error by construction, so reproject all court lines to measure the real error |
| **2 Player detection** | YOLO11m is adequate; new detectors gain mainly on medium/large objects | YOLO26 (same AGPL licence and API, no NMS step, simpler CoreML export); RF-DETR-S/M (Apache, native CoreML) as a quick A/B test | Far-court recall (~90 %) is the real limit, and no new detector targets it; split the frame into tiles or fine-tune on own footage |
| **3 Tracking & identity** | BoT-SORT is near the top of simple trackers; court-half priors beat appearance-based re-identification when players wear the same club kit | gta-link offline pass (MIT, already named in the spec); CAMELTrack-style keypoint cue (Apache) | Deep-EIoU and the SAM 3 state of the art built on it stay off-limits; no racket-sport tracking benchmark exists |
| **4 Pose** | RTMPose-m (75.8 AP) still at or above new real-time models; rtmlib is the right wrapper | RTMW-l/x or ViTPose-L run only on the gated player crops; training on synthetically blurred frames | Wrist keypoints blur during smashes; the ankle-based foot point measured better but is still off by default |
| **5 Shuttle (2D)** | TrackNetV3 + median background + own-footage fine-tune; "V4/V5 are incremental" holds | InpaintNet (MIT, already vendored); clean-room re-implementation of a V4-style motion channel; blur-centre labels; occlusion masking; teacher–student pseudo-labels (unpublished for shuttles) | No white-wall research exists; F1 0.953 is in-sample; 12 fps in production vs 30 fps training; both V5s blocked |
| **5b 3D, speed, landing** | Not computing shuttle speed; labelling everything 3D „Schätzung" | Speed near the net or a fitted launch velocity; disagreement between the physics fit and the vertical-plane fit as an uncertainty band; validate with 240 fps phone video | 2D peak speed overshoots radar by 60 %+; certified line calls need about 12 high-speed cameras |
| **6 Rally & hit** | The rule-based rally segmenter is a sensible amateur baseline | Learned hit detector over pose + shuttle + court keypoints (re-implement the UMEG-Net idea, or the spec's HitNet-style GRU); audio as an unproven experiment | Hit detection is the field's weak link; the repo's hit recall and rally boundaries are unmeasured |
| **7 Stroke** | BST with pose + shuttle + court inputs; coarse 7-class taxonomy | BST's ShuttleSet checkpoint; shuttle input normalised to the court or camera instead of the image; a second pass that uses the neighbouring strokes (ASF idea) | Benchmarks assume ground-truth hits; the v2 spec file is missing; the VLM pilot will likely land below 0.35 |
| **8 Insights & tactics** | Vision writes a structured log; language models never produce numbers | Stupa Cast's metric set; an LLM layer that reads only the Postgres rows (needs your sign-off) | Doubles pair stats have no writer; heatmaps are a coarse 12×12 grid |
| **Data & labels** | Own labelled club footage is the only in-domain evidence; ShuttleSet (MIT) is clean | FineBadminton (check its data card); DShuttleNet for doubles (verify it exists and its licence) | The spec lists VideoBadminton and BadmintonDB as usable, but neither has a licence; BFMD has none either |

**Four spec statements need correcting:**

1. **TrackNetV4 "now MIT".** `30-ml-pipeline.md` says TrackNetV4 is "now MIT". The licence is right, but the official code is TensorFlow/Keras with no downloadable weights. Using it means writing and retraining a clean-room PyTorch version; the unlicensed PyTorch ports cannot be copied.
2. **"Usable" seed datasets.** The same spec lists VideoBadminton and BadmintonDB as "usable" seed datasets. Both have no licence file, which is the one category locked decision #6 forbids.
3. **PnLCalib exclusion.** The spec excludes PnLCalib for being GPL-2.0, but decision #6 allows GPL. The real reason to skip it is its soccer court template, and its new distortion-refinement code is a legitimate reference.
4. **Class-count dispute.** The stroke specs disagree on 18 versus 35/25 classes. As shown above, both are true at different levels.

CourtKeyNet, the spec's planned court auto-assist, appeared in the research only as a 10-star repo from 2025 ([CourtKeyNet](https://github.com/adithyanraj03/CourtKeyNet)). Its licence and weights were not verified, so check them before use, as with any other dependency.

**Upgrades with the best effort-to-evidence ratio** reuse what is already vendored or licensed:
- **InpaintNet.** TrackNetV3's gap-filling network is MIT, ships checkpoints, and sits in the vendored checkout without the MPS patch. It fills short gaps the way layer 4 of the spec intends. Every position it invents must carry the `interp` provenance that `overlay.py` currently drops.
- **YOLO26.** A one-line model swap under a licence the repo already accepts.
- **gta-link.** MIT, and already named in the spec as the offline re-entry pass.
- **BST's ShuttleSet weights.** MIT, and already planned for v2.

The most hyped new pieces are blocked by licence, need NVIDIA GPUs (CUDA-first), or both: TrackNetV5-Lite, UMEG-Net, SAM 3-based tracking and Sapiens2. For TrackNetV5-Lite, a single email asking the authors to add a licence would unblock it.

## The research exposes measurement risks, not architecture risks

### Risk 1: frame rate

The largest risk is frame rate. Every badminton dataset in the research with a stated rate runs at 25–30 fps or higher: BFMD at 30 fps ([BFMD](https://github.com/Ning-D/BFMD)), and Fine-Badminton resampled to 25 fps ([arXiv 2605.23355](https://arxiv.org/pdf/2605.23355)). Production runs every stage at 12 fps, while the promoted shuttle fine-tune was trained and scored at 30 fps (`specs/features/stroke-classification.md`, `shuttle-landing-inout.md`).

The repo's own symptoms fit this picture:
- 21–27 % shuttle coverage on Session 2.
- A smash swing that lasts one or two frames.
- 0 of 23 landings located.
- Stroke windows of 16 frames, where BST's benchmarks use 30.

The table-tennis 3D work deliberately trains at 20–60 FPS so the model tolerates changes in frame rate ([arXiv 2511.20250](https://arxiv.org/pdf/2511.20250)). The repo does the opposite by accident.

**Suggested experiment:** re-ingest one Session-2 game at 30 fps (about 10 GB per 7-minute game, per `40-worker-contract.md`), then re-measure shuttle coverage, landings and stroke recall. This looks like the cheapest experiment with the largest likely payoff, and the repo already lists it as a separately measured experiment.

### Risk 2: the shuttle score has not been tested on a new session

The shuttle F1 of 0.953 comes from splitting rallies within one labelled clip. Two findings suggest the cross-session number will be markedly lower:
- the ETH detector's drop from 0.86 to 0.70 in environments it had not seen;
- the table-tennis detector that only works on the ball colour it was trained on.

Until one Session-2 clip is labelled and scored against the repo's own F1 ≥ 0.80 gate, every statistic derived from the shuttle rests on an untested assumption.

### Risk 3: the stroke gate failure is expected

The stroke gate failure (0.40 against a 0.70 gate) is what the literature predicts. It is not a sign that BST is the wrong model.

| | BST's benchmark numbers (0.77–0.83) | The repo's training run |
|---|---|---|
| Hit timing | Ground-truth hit clips | Machine-detected hits |
| Data | Tens of thousands of strokes | 695 hits, trained from scratch |
| Format | Singles | Doubles |
| Camera | Broadcast geometry | Ultrawide behind the baseline, 12 fps |
| Result | — | 0.945 on training data vs 0.402 held out |

The third-party error-propagation rule means even a perfect classifier is capped by hit detection. The repo's ~94 % hit precision cannot be compared with the 62 % strict precision above: it was measured loosely on hits the machine proposed, and its recall is unknown.

The research-backed levers are the ones already sketched for the missing v2 spec:
- start from BST's ShuttleSet checkpoint;
- train on coarse classes;
- add a second pass that uses the neighbouring strokes;
- fix hit timing first.

The v2 spec file itself still needs to be written.

### Risk 4: the VideoJev pilot will probably score low

Expect the VideoJev-4B pilot to land in its `closed` tier (below 0.35). The closest measured results for small open models are below 25 % on FBBench and 27–32 % on DoublesEval. Both are on broadcast footage, which these models handle better than an ultrawide hall.

The pilot's protocol already matches what the research recommends: a closed label set, clips centred on the hit, preset frame rates, an option to abstain, and separate dev and eval games. If it qualifies as a labelling assistant, have labellers periodically label without its suggestions. That protects the answer key from the pull toward the model's predictions. No independent numbers exist for VideoJev itself.

### Risk 5: single-camera 3D, licences and Apple Silicon

Single-camera 3D and line calls should stay estimates. BWF certification needs multi-camera high-speed rigs and a 98 % bar; one ultrawide camera at 12 fps located 0 of 23 landings. If shuttle speed is built:
- speed near the net, or from a fitted launch velocity, is defensible;
- peak frame-to-frame speed is not;
- a few 240 fps phone slow-motion smashes would give an in-club error bar, which no commercial app publishes.

Across all stages, licensing and Apple Silicon add friction to every model swap:
- The most attractive new code has no licence grant: ICIP TrackNetV5, UMEG-Net, the BFMD annotations, and trackers built on Deep-EIoU.
- Custom licences (SAM, Sapiens2, DEIMv2, RF-DETR's paid PML tiers) each need one careful reading.
- **No MPS or CoreML benchmark exists for any model in this report.** Each candidate needs the repo's CPU-versus-MPS check for silent NaNs. BST, for example, hard-codes CUDA-or-CPU ([bst_infer.py](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/stroke_classification/main_on_shuttleset/bst_infer.py)).

## Conclusion

The research changes the question from "which model is newest" to "which numbers has the club actually measured". On broadcast benchmarks the field is saturated; on club footage it is silent. That makes BC Rheinbach's labelled DJI clips, its honest gaps and its „Schätzung" labels the real asset. No paper, product or open-source project measures a white shuttle against a white wall, and no consumer app publishes a table of which shots it confuses with which (a confusion matrix). A small published club benchmark would put the project ahead of the market on transparency at almost no model cost. For example: about 200 labelled shots, plus one labelled shuttle clip per session.

My reading of the evidence, ordered by expected payoff per hour of work:

1. A 30 fps re-ingest experiment.
2. One Session-2 shuttle label set, to get a cross-session F1.
3. Lens undistortion, with AnyCalib as a check, before any further geometry work.
4. A learned hit detector before any further stroke work, then BST started from the ShuttleSet checkpoint.
5. Only after those, the cheap model swaps: YOLO26, gta-link, InpaintNet, and a clean-room motion channel.

The direction of the field also argues that any future LLM layer should read the Postgres aggregates, never the video, so every number it states stays traceable. In that design, vision writes the log and language models only describe it. Each of these is a scope decision for you, and none should start without a spec.
