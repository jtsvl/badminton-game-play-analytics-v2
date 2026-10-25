# VLMs / LLMs / foundation models for badminton and racket-sport video understanding, plus badminton datasets & benchmarks (2025 – Oct 2026)

*How these notes were sourced: the sandbox proxy blocked direct fetches of arXiv, Hugging Face, alphaXiv, CVF open access and OpenAlex. Paper numbers therefore come from search-engine extracts of the primary papers (arXiv HTML/PDF, ACM, CVF, ACL Anthology). Repo and license facts come from direct GitHub page fetches where marked "(GitHub fetched)". Every number below is **self-reported by the paper's authors** unless marked otherwise. I found no independent replication of any badminton VLM number. Several papers have version-to-version discrepancies, and those are flagged where they occur.*

---

## Q1. How well do general VLMs (Gemini 2.5/3, GPT-5-class, Qwen2.5-VL/Qwen3-VL/Qwen3.5, InternVL3, …) perform on fine-grained badminton / racket-sport understanding, which benchmarks measure it, and what do they get wrong?

### Takeaway
Zero-shot, every general VLM is weak on fine-grained badminton:
- On FineBadminton's FBBench, the best commercial model (Gemini 2.5 Pro, with the paper's keyframe strategy) reaches about **38.6 %** overall multiple-choice accuracy, and small open models start **below 25 %**.
- On the 2026 doubles benchmark DoublesEval, open VLMs (Qwen2.5-VL, Qwen3-VL, VideoLLaMA3, Molmo2) cluster at about **27–32 %**.

The consistent pattern across badminton, tennis and multi-sport benchmarks is that a **fine-tuned 7–8B open model beats zero-shot frontier models**. The main failure modes are:
- temporal grounding of fast strokes
- frame sampling that is too sparse
- player-role attribution
- spatial and metric reasoning (distance, position)
- multi-step causal or tactical chains

### Cited Findings

#### Badminton-specific benchmarks
- **FineBadminton / FBBench (ACM MM 2025, arXiv 2508.07554, Aug 2025).** The authors say fine-grained analysis of fast sports is hard for MLLMs mainly because few datasets have rich domain annotations. FBBench tests spatio-temporal reasoning and tactical comprehension, and the authors report that "current MLLMs still perform poorly" — [arXiv 2508.07554](https://arxiv.org/abs/2508.07554); [ResearchGate record](https://www.researchgate.net/publication/396940660_FineBadminton_A_Multi-Level_Dataset_for_Fine-Grained_Badminton_Video_Understanding)
- FBBench has **2,563 QA pairs** in four task families: **Count, Action, Position, Cognition** — [arXiv PDF 2508.07554](https://arxiv.org/pdf/2508.07554)
- Commercial model ranking on FBBench: **Gemini 2.5 Pro > Doubao 1.5 Pro > GPT-4.1**. Gemini 2.5 Pro with strategy "S" scored 932 points, which the paper equates to **38.6 % accuracy** overall. Strategy S consistently beat uniform frame sampling at the same frame count. S appears to be the paper's hit-centric keyframe selection, but the extract did not name it explicitly — [arXiv PDF 2508.07554](https://arxiv.org/pdf/2508.07554)
- Small open-source models scored **below 25 %** zero-shot on FBBench. After domain fine-tuning, **Qwen2.5-VL-7B reached 42.06 %** multiple-choice accuracy, beating every commercial model tested, including Gemini 2.5 Pro with S. The paper also proposes hit-centric keyframe selection plus "coordinate-guided condensation" as a baseline — [arXiv PDF 2508.07554](https://arxiv.org/pdf/2508.07554); [alphaXiv (German summary)](https://www.alphaxiv.org/de/abs/2508.07554)
- An automated (non-peer) review flagged that the FBBench numbers need a clearly documented train/test split to be trustworthy. This matters because the fine-tuned model is trained on the same dataset family the benchmark is built from — [Pith review](https://pith.science/paper/2508.07554)
- I found no GPT-4o or InternVL3 FBBench number in the extracts. The commercial lineup centres on GPT-4.1 — [arXiv PDF 2508.07554](https://arxiv.org/pdf/2508.07554)
- **DoublesEval (arXiv 2608.24439, Aug 2026).** It is built from 60 curated rallies from 4 professional doubles matches (BWF World Tour Super 750/1000) and has about 9.6K structured instances. Four open-source VLMs were evaluated zero-shot. **No Gemini or GPT numbers are reported** — [arXiv 2608.24439](https://arxiv.org/abs/2608.24439); [arXiv HTML](https://arxiv.org/html/2608.24439v1)
- DoublesEval zero-shot overall accuracy:

  | Model | Overall |
  |---|---|
  | Qwen3-VL | **29.00 %** (L1 35 / L2 29 / L3 25 / L4 27) |
  | Qwen2.5-VL | **27.57 %** |
  | VideoLLaMA3 | **32.30 %** |
  | Molmo2 | **31.31 %** |

  The text describes a "narrow band" of **28.65–32.35 %**, which does not exactly match the table values. This is probably a version or rounding inconsistency — [arXiv HTML 2608.24439v1](https://arxiv.org/html/2608.24439v1)
- DoublesEval's named recurring failure modes are **player role attribution, spatial localization, and cross-segment causal reasoning**. The authors argue that VLMs describe scenes well but fail at coordinated roles and spatio-temporal dependencies — [arXiv 2608.24439](https://arxiv.org/abs/2608.24439)
- DoublesEval's test-time **TacticCheck** reranks candidate answers using the model's own lower-level tactical predictions. It raised Qwen3-VL from **29.00 to 37.20 (+8.2)** and gave "consistent gains across all evaluated models" — [arXiv HTML 2608.24439v1](https://arxiv.org/html/2608.24439v1)
- **CourtSI / CourtSI-Bench** (arXiv 2603.09896, Mar 2026; Fudan / Shanghai AI Lab). It covers spatial intelligence on broadcast badminton, tennis and table tennis: more than **1M QA pairs**, of which **3,686** are human-verified bench QA. Of **25 VLMs**, the best still trails humans, **especially on distance measurement**. Models trained on generic spatial benchmarks transfer poorly. Fine-tuning **Qwen3-VL-8B** on CourtSI added **+23.5 pp** and generalized to unseen pickleball (CourtSI-Ext) — [arXiv HTML 2603.09896](https://arxiv.org/html/2603.09896); [HF paper page](https://huggingface.co/papers/2603.09896); [project page](https://visionary-laboratory.github.io/CourtSI/)

#### Multi-sport benchmarks that include badminton
- **SportR (ICLR 2026, arXiv 2511.06499).** It covers basketball, soccer, table tennis, badminton and American football, but **badminton appears only in the image split**. Video covers 4 sports and excludes badminton. Sizes: 4,789 images and 2,052 clips in v3, versus 5,017 and 2,101 in the OpenReview version. It has 6,841 human chain-of-thought explanations, more than 20K QA pairs, 50 foul types and 12 tactics — [arXiv HTML v3](https://arxiv.org/html/2511.06499v3); [OpenReview PDF](https://openreview.net/pdf/b380aed35e31abdd346ca5071a6060df8f37f8bc.pdf); [VisLang Lab](https://vislang.ai/publications/xia2026sportr)
- SportR zero-shot results: **GPT-5, Claude 4 and Gemini 2.5 Pro all scored below 7 % IoU on visual grounding**, and foul classification reached about **44 %**. Qwen2.5-VL-7B with SFT + GRPO improved image foul classification from **14.4 % to 50.7 %**, but grounding IoU only reached about **10 %** — [VisLang Lab](https://vislang.ai/publications/xia2026sportr); [Paper note](https://en.papernotes.org/ICLR2026/vlm_reasoning/sportr_a_benchmark_for_multimodal_large_language_model_reasoning_in_sports/)
- **RefereeBench (arXiv 2604.15736, Apr 2026).** It covers 11 sports including badminton, with 925 videos and 6,475 QA pairs. The best models (Doubao-Seed-1.8, Gemini-3-Pro) reach about **60 %**, Gemini-3-Flash 60.6 and Claude-4.5-Sonnet 47.6. The best open model, **Qwen3-VL, reaches about 47 %**. Difficulty "varies substantially" by sport, but no badminton-only number was extracted — [arXiv HTML 2604.15736v1](https://arxiv.org/html/2604.15736v1)
- **SPORTU (ICLR 2025; earlier, arXiv Oct 2024).**
  - SPORTU-text has 900 MCQs; GPT-4o scored 71 %.
  - SPORTU-video, overall: the arXiv abstract gives Claude-3.5-Sonnet **69.52 %**, while a secondary summary gives Qwen2-VL-72B **70.94 %**. The sources conflict.
  - SPORTU-video, hard questions: Claude-3.5-Sonnet 52.6 % (arXiv) versus GPT-4o 57.8 % (ICLR version).
  - I found no badminton breakdown.
  — [arXiv 2410.08474](https://arxiv.org/abs/2410.08474); [ICLR 2025 anthology](https://mlanthology.org/iclr/2025/xia2025iclr-sportu/)
- **ActionAtlas (NeurIPS 2024; earlier).** It has 934 videos, 580 actions and 56 sports. GPT-4o scored **45.52 %**, and the best model in the NeurIPS abstract scored **48.73 %**, against chance of about 20 %. The best open model, **Qwen2-VL-7B, scored about 30.24 %**. Non-expert humans given action descriptions scored 61.64 %. The authors state that **a high frame-sampling rate is essential**, and that Gemini's default video sampling lacks it — [arXiv 2410.05774](https://arxiv.org/pdf/2410.05774); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f8643596721dbac71d67f89497323efe-Abstract.html)
- **VideoNet (CVPR 2026, arXiv 2605.02834).** Domain-specific action recognition across 1,000 actions and 37 domains (arXiv v2), or 1,087 actions and 38 domains (CVPR PDF).
  - Multiple-choice: **Gemini 3.1 Pro 69.9 %** versus **Qwen3-VL-8B 45.0 %**.
  - Binary (chance 50 %): Qwen3-VL-8B **59.2–59.3 %**; GPT-5.4 72.4 % at 0-shot and 76.3 % at 3-shot.
  - Few-shot: Qwen gained about **+7.0 %** from 3 in-context examples, Gemini **declined −4.8 %**, and non-expert humans gained +13.6 %.
  — [arXiv HTML 2605.02834v2](https://arxiv.org/html/2605.02834v2); [author page](https://tanu.sh/research/videonet); [GitHub](https://github.com/RAIVNLab/VideoNet)
- VideoNet's **Sports** subset: GPT-5.4 71.5 % versus Qwen3-VL-8B 58.1 % (binary-style). This comes from a third-party leaderboard and was **not confirmed against the paper** — [benchmarklist](https://benchmarklist.com/benchmarks/videonet_a_large_scale_dataset_for_domain_specific_action_recognition/)

#### Tennis and table-tennis benchmarks (closest analogues)
- **TennisTV (arXiv 2509.15602, Sep 2025).** It has 8 tasks from stroke level to rally level, 2,527 human-verified questions and 17 MLLMs. The authors conclude that frame-sampling density must be tuned per task and that **temporal grounding is the bottleneck**. A third-party review flags the automated question-generation pipeline as a bias risk — [arXiv 2509.15602](https://arxiv.org/pdf/2509.15602); [Pith review](https://pith.science/paper/2509.15602)
- **TennisVAR (arXiv 2608.12920, Aug 2026).** On its TRACE benchmark (total metric):

  | Model | Setting | Total |
  |---|---|---|
  | Gemini-3-Pro | zero-shot | **17.89** |
  | Gemini-3.1-Pro | zero-shot | **18.87** |
  | Qwen2.5-VL-7B | SFT | **42.71** |
  | Qwen3-VL-8B | SFT | **44.83** |
  | TennisVAR | stroke-evidence grounded | **57.11** |

  The authors say general MLLMs "often fail to organize temporally distributed events into a coherent tactical chain" — [arXiv 2608.12920](https://arxiv.org/pdf/2608.12920); [arXiv HTML](https://arxiv.org/html/2608.12920)
- **TennisExpert (arXiv 2603.13397, Mar 2026).** It is built on Qwen3-VL-8B with a semantic parser and memory. Zero-shot "LLM Total" on its TennisVL benchmark:

  | Model | LLM Total |
  |---|---|
  | Gemini-3-Pro | 59.89 |
  | GPT-5.2 | 56.59 |
  | Qwen3.5-Plus | 52.55 |
  | Gemini-3-Flash | 52.50 |
  | Claude-4.5-Sonnet | 50.75 |

  The authors claim the 8B TennisExpert "significantly outperforms" all of them, including InternVL and Qwen3-VL. TennisExpert's own total was not extracted — [arXiv 2603.13397](https://arxiv.org/pdf/2603.13397)
- A June 2025 NUS B.Comp dissertation assessed MLLMs on tennis action classification and sequence identification — [arXiv 2507.02904](https://arxiv.org/abs/2507.02904)
  - A ResearchGate record for that title also claims that, in badminton, LLM feedback was "completely irrelevant" and "fake", and worse than university coaches — [ResearchGate](https://www.researchgate.net/publication/393477193_Enhancing_Sports_Strategy_with_Video_Analytics_and_Data_Mining_Assessing_the_effectiveness_of_Multimodal_LLMs_in_tennis_video_analysis)
  - The arXiv abstract is tennis-only, so this badminton claim is **unverified and possibly misattributed**.

#### General action-recognition evidence relevant to small models
- Qwen2.5-VL-7B zero-shot on the COIN procedural dataset (750 actions) was the best MLLM baseline but "significantly trails" a fine-tuned discriminative method — [arXiv 2603.02546](https://arxiv.org/pdf/2603.02546)
- Library versions matter: a Hugging Face Transformers issue reports a persisting accuracy regression for Qwen2.5-VL-7B-Instruct in v4.56.2 — [GitHub issue #41180](https://github.com/huggingface/transformers/issues/41180)

#### Qwen3.5-4B (the user's pilot model)
- The official model card reports these general video benchmark scores. **None of them are sports-specific or fine-grained action benchmarks.**

  | Benchmark | Score |
  |---|---|
  | Video-MME (with subtitles) | 83.5 |
  | Video-MME (without subtitles) | 76.9 |
  | VideoMMMU | 74.1 |
  | MLVU | 82.8 |
  | MVBench | 71.2 |
  | LVBench | 66.4 |
  | MMVU | 64.9 |

  The card claims the Qwen3.5 family outperforms Qwen3-VL — [HF model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- Third-party claims, not verified:
  - In 4-bit the 4B needs about 3 GB — [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)
  - A deployer (Overshoot) says the 4B retains about 96 % of the 27B's video-understanding score — [HF discussion](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/43)
  - A blog notes high hallucination rates for the 4B on niche factual text queries — [Medium](https://medium.com/@milesk_33/qwen-3-5-small-models-0-8b-9b-the-architecture-shift-that-lets-edge-ai-finally-compete-ab3b807273ff)

### Inferences
- Across five independent racket-sport benchmarks (FBBench, DoublesEval, TennisVAR/TRACE, TennisExpert/TennisVL, CourtSI) the ordering is the same: **zero-shot open < zero-shot frontier < fine-tuned 7–8B open**. All of these numbers are self-reported, mostly in the authors' own benchmark (home-field advantage), so the size of the gap is probably inflated. The direction is consistent, though.
- Fine-grained badminton accuracy for zero-shot VLMs (about 25–39 % on multiple choice) is far below what a stroke classifier needs. A **4B model zero-shot, on amateur footage it has never seen, will likely do worse than the sub-25 % small-model FBBench figure suggests**, because FBBench is broadcast footage and the domain shift adds error. This is an inference; no direct measurement exists.
- General video leaderboards (Video-MME, MVBench) do not predict fine-grained sports performance. Qwen3.5-4B's strong card numbers should not be read as evidence of badminton stroke competence.
- Three sources (ActionAtlas, TennisTV, FineBadminton strategy S) agree on frame sampling: **dense, hit-centred sampling** is a first-order lever, often bigger than model choice. Feeding a VLM a short clip centred on the detected hit frame is likely to matter more than the model's size.
- Metric and spatial outputs (distances, positions, speeds) are where VLMs are worst: CourtSI distance measurement, and SportR grounding below 10 % IoU. That supports keeping geometry in the CV pipeline (homography) and never asking a VLM for numbers.

### Gaps
- No badminton-specific number exists for **Qwen3.5 (any size), InternVL3/3.5, Gemini 3, or GPT-5** on a badminton benchmark. DoublesEval tested open models only, and FBBench predates these models.
- Per-sport badminton breakdowns for RefereeBench, SportR (image split) and CourtSI were not retrievable from the extracts.
- FBBench per-task-family scores (Count / Action / Position / Cognition) were not retrievable. That leaves unknown which badminton skills fail most, for example stroke type versus landing position.
- DoublesEval's exact model sizes (e.g., Qwen3-VL-8B vs 32B) were not confirmed.
- No independent replication of any badminton VLM number was found.
- Full-text access was blocked (arXiv/HF/CVF via proxy), so table-level detail and footnotes could not be checked.

---

## Q2. Which badminton-specific multimodal or LLM systems were published 2025–2026 (tactical reasoning, coaching feedback, commentary, shot forecasting, win probability, CoachAI challenge)?

### Takeaway
The dominant and best-performing design in 2025–26 badminton work is **"CV pipeline → structured event log → LLM for language and reasoning"**, not end-to-end VLMs. Examples:
- COACH (AAAI 2026)
- the CVPR 2026 Findings multi-agent-debate paper
- Court-to-Conversation (KBS 2026)
- TactiPlay (2026, amateur players)
- the ERA 2026 LLM-coaching paper

Several of these explicitly argue that VLMs lack badminton knowledge. Captioning work (Shot2Tactic-Caption, BFMD) uses trained encoder-decoders. Shot forecasting has **no LLM-based ShuttleNet successor**, and I found no CoachAI challenge after IJCAI 2023.

### Cited Findings

#### Tactical reasoning / QA / summarization
- **COACH (AAAI 2026, arXiv 2512.01853).** A reconfigurable multi-agent system for sports video understanding: a shared backbone with role specialization through intent-driven orchestration and structured chain-of-thought fine-tuning. Its dataset, **COACH-Data**, covers **22 badminton matches and about 19,000 strokes**, built on ShuttleSet. The reported backbone is **Flan-T5-XL**. The authors claim it "significantly outperform[s]" **Gemini 2.5 Pro** on QA and summarization; exact scores were not extracted. A third-party note says validation is badminton-only and the backbone has limited capacity for complex tactics — [arXiv HTML 2512.01853v1](https://arxiv.org/html/2512.01853v1); [Paper note](https://en.papernotes.org/AAAI2026/multi_agent/coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor/)
- **"From Alignment to Reason: Multi-Agent Debate for Tactical Badminton Video"** (CVPR 2026 Findings; Yi-Xiang Zhang & Yu-Shuen Wang, NYCU). The authors argue VLMs **lack the badminton-specific knowledge to understand tactical intent**. Their pipeline:
  1. CV tools convert footage into structured text game logs.
  2. An LLM performs retrieval over those logs.
  3. A multi-agent debate drafts and revises narratives and checks them against the original events.

  They report that this semantic search "clearly" beats state-of-the-art VLM-based systems, with grounded explanations. No numbers were extracted — [CVF open access PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhang_From_Alignment_to_Reason_Multi-Agent_Debate_for_Tactical_Badminton_Video_CVPRF_2026_paper.pdf)
- **"Court to conversation"** (Bharadwaj & Srinivasa, *Knowledge-Based Systems* vol. 333, Jan 2026). Single-camera video goes through YOLO and MediaPipe pose into spatio-temporal events and kinematics. These form a queryable knowledge base, and a **RAG-enhanced LLM** answers natural-language coach queries. It defines an **"out-of-position"** vulnerability KPI. A third-party systematic review lists LangChain + FAISS and **mAP 0.94**, unverified against the paper — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125020659); [systematic review (ResearchGate)](https://www.researchgate.net/publication/402667183_Making_the_World's_Fastest_Racket_Sport_even_Better_A_Systematic_Review_of_Artificial_Intelligence-based_Objective_Player_Performance_Assessment_in_Badminton)
- **TactiPlay (arXiv 2607.27125, Jul 2026, cs.HC; Qiaoyi Chen, Xiaojuan Ma et al.).** The system **targets amateur players**: amateurs record matches but tools give only aggregate statistics.
  - It uses a taxonomy of performance issues derived from **national-level athletes' annotations**, with rally-level, video-anchored review linked to court visualizations.
  - A formative study had n = 8 players. A within-subjects study had n = 16 against a reports-plus-statistics baseline, and TactiPlay produced "more frequent, concrete, actionable, and appropriate reflections".
  - Per an introduction snippet, a **VLM classifies shot type and handside** and **GPT-4.1 generates reports**. This is unverified; the full text was not opened.
  - An automated Pith review (non-peer) suspects the headline statistics overstate precision by counting nested reflection entries as independent.

  — [arXiv 2607.27125](https://arxiv.org/abs/2607.27125); [Pith](https://pith.science/paper/2607.27125)
- **DoublesEval's TacticCheck** (above): test-time consistency checking against lower-level predictions, worth +8.2 points on Qwen3-VL — [arXiv HTML 2608.24439v1](https://arxiv.org/html/2608.24439v1)

#### Captioning / commentary
- **Shot2Tactic-Caption (ACM MMSports '25; arXiv 2510.14617; Ding, Fujii, Tamaki).**
  - Dual-branch shot-level and tactic-level captioning, with a visual encoder, a spatio-temporal Transformer encoder and a Transformer decoder.
  - A **Tactic Unit Detector** identifies tactic units, types and states (Interrupt / Resume). A secondary summary says units span 5, 7 or 9 shots, with 9 tactic types and 5 states.
  - A ResNet50 spatio-temporal encoder worked best, and **shot-wise prompt structuring** improved tactic captions.
  - The dataset has 5,494 shot captions and 544 tactic captions. No metric values were extracted.

  — [ACM DL](https://dl.acm.org/doi/10.1145/3728423.3759408); [ResearchGate](https://www.researchgate.net/publication/396542492_Shot2Tactic-Caption_Multi-Scale_Captioning_of_Badminton_Videos_for_Tactical_Understanding); [awesomepapers (arXiv 2510.14617)](https://awesomepapers.io/computer-vision/papers/2510.14617)
- **BFMD (CVPR 2026 CVsports workshop; arXiv 2603.25533; Ding, Fujii, Tamaki).** A VideoMAE-based multimodal captioning framework with a **Semantic Feedback** module that uses shot semantics to guide captions. Multimodal input plus semantic feedback beats RGB-only baselines (qualitative claim; the README has no metrics). The models are trained captioners, not general VLMs — [arXiv 2603.25533](https://arxiv.org/pdf/2603.25533); [GitHub (fetched)](https://github.com/Ning-D/BFMD)
- **Dedicated badminton MLLM commentary generation: none found.** The nearest work is a general pause-aware real-time commentary decoding paper (LREC 2026), which notes that prompting-based MLLMs ignore *when* to speak — [ACL Anthology LREC 2026](https://aclanthology.org/2026.lrec-1.722/)
- **BADGE (2024; earlier).** LLM-based badminton match report generation and evaluation — [arXiv 2406.18116](https://arxiv.org/pdf/2406.18116)

#### Coaching feedback
- **ERA 2026 (Electronic Research Archive).** An explainable stroke-quality framework: an mST-GCN++ skeleton classifier reaches **94.34 % on 11 stroke categories**, followed by hierarchical fuzzy inference. An **LLM receives stroke type, performance indices and the assessment result and writes personalized feedback**. No evaluation of feedback quality was seen in the extracts — [AIMS Press](https://www.aimspress.com/article/doi/10.3934/era.2026333)
- **BadminSense (arXiv 2603.21825, 2026).** Smartwatch stroke evaluation. Its improvement advice is still **rule-based**, and LLM context-aware advice is listed as future work — [arXiv 2603.21825](https://arxiv.org/pdf/2603.21825)
- Adjacent racket-sport coaching work:
  - **Talking Tennis**: 3D biomechanics converted into LLM feedback — [arXiv 2510.03921](https://arxiv.org/pdf/2510.03921)
  - **T3Set (KDD 2025)**: table-tennis IMU data paired with coach suggestions, for an LLM virtual coach — [ACM DL](https://dl.acm.org/doi/10.1145/3711896.3737407)
- **Hallucination risk, earlier work.** Qiu, "The Impact of LLM Hallucinations on Motor Skill Learning: A Case Study in Badminton", *IEEE Access* vol. 12 (2024). Design: 80 novice freshmen over 16 weeks, ChatGPT/New Bing self-learning versus expert-taught control. Both groups improved; the between-group and hallucination findings were not retrievable — [DOAJ / IEEE Access](https://doaj.org/article/2d9ace2f22d0419990139972b4507b7c)

#### Shot forecasting / movement / win probability
- **No 2025–26 LLM- or VLM-based ShuttleNet successor was found.** The ShuttleNet line is a 2022 Transformer seq2seq model that fuses rally progress and player style — [arXiv 2112.01044](https://arxiv.org/pdf/2112.01044)
  - **RallyTemPose (CVPRW 2024; earlier)** forecasts strokes from skeleton poses, court position and player embeddings — [CVF](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Ibh_A_Stroke_of_Genius_Predicting_the_Next_Move_in_Badminton_CVPRW_2024_paper.pdf)
- 2026 non-LLM forecasting work:
  - **SPAIT** (movement forecasting) reports about a **20.13 % ADE improvement** over prior badminton trajectory predictors — [Springer J. Big Data](https://link.springer.com/article/10.1186/s40537-026-01407-7)
  - **KNN-HMM** real-time stroke prediction argues that Transformers are too costly for live streaming — [Springer J. Big Data](https://link.springer.com/article/10.1186/s40537-026-01396-7)
  - A **DRL + game-tree** method values each shot's contribution to winning the rally; the values correlate with match results and rankings — [SCITEPRESS 2026](https://www.scitepress.org/publishedPapers/2026/143176/pdf/index.html)
- **Win-probability baseline (earlier).** ShuttleSet frames per-stroke win probability as predicting the rally outcome, evaluated with AUC, accuracy and Brier score, with the latest 10 matches as test set. This is the "shot influence" line of Wang et al. 2021/22 — [arXiv 2306.04948](https://arxiv.org/pdf/2306.04948); [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.04948)
- **CoachAI Badminton Challenge: only the IJCAI 2023 edition was found.**
  - The stroke-forecasting track used ShuttleSet22 (train **30,172 strokes / 2,888 rallies**).
  - It had about 95–100 participants, and 16–17 teams submitted final results; the two sources differ.
  - The winner, **Intro_to_AI_team8** (NYCU), used "Advanced ShuttleNet".

  — [arXiv 2307.13715](https://arxiv.org/pdf/2307.13715); [ShuttleSet22, IJCAI 2024](https://www.ijcai.org/proceedings/2024/1042)
- The CoachAI-Projects repo lists projects through RallyNet (ECML PKDD 2024) plus a 2026 NewsAgent LLM-agent benchmark (not badminton). It shows **no new badminton challenge or dataset after 2024** — [GitHub CoachAI-Projects (fetched)](https://github.com/wywyWang/CoachAI-Projects)

#### Survey
- **"A Survey of Large Models in Sports"** (Findings of ACL 2026; arXiv 2608.14377). It reviews 241 core papers and notes that badminton is among the better-resourced sports for datasets. It cites BadmintonDB and FineBadminton — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1851/); [arXiv 2608.14377](https://arxiv.org/abs/2608.14377)

### Inferences
- The most credible path to "tactical insight" text for an amateur-club app is the pattern used by COACH, Debate, Court-to-Conversation and TactiPlay: **deterministic CV extracts events, and an LLM only verbalizes or answers over a structured log**. This keeps every number traceable, which fits the project's "plausible-but-wrong is worse than absent" rule.
- TactiPlay is the only 2025–26 system aimed at **amateur** players, and its evaluation is a user study (n = 16) of reflection quality, not label accuracy. It shows the UX value of video-anchored, rally-level review. It does not validate VLM stroke-labelling accuracy on amateur footage.
- COACH reportedly beats Gemini 2.5 Pro with a Flan-T5-XL (about 3B) backbone over structured ShuttleSet data. This suggests small local models are adequate for *language over structured stats*, as opposed to *perception from pixels*.
- Shot forecasting remains a sequence-model problem (ShuttleNet-style). No evidence was found that LLMs improve it.

### Gaps
- No quantitative results were extracted for COACH, the CVPR 2026 debate paper, BFMD captioning, or Shot2Tactic-Caption (BLEU/CIDEr etc.).
- "ShuttleFlow" (a normalizing-flow next-shot model, reportedly *Machine Learning* 2025) surfaced only as a citation and is **unverified**.
- No evaluation of LLM feedback **factual accuracy or hallucination rate** in badminton coaching was found beyond the 2024 IEEE Access design (results not retrieved).
- No CoachAI challenge for 2024–2026 was found. It may not exist, or it may be hosted elsewhere.
- No badminton rally-outcome or win-probability model using LLMs or VLMs was found for 2025–26.

---

## Q3. Which new badminton datasets appeared 2025–2026, and which earlier ones are still relevant? (Size, annotations, footage type, license, access)

### Takeaway
2025–26 added several broadcast-footage badminton datasets aimed at multimodal or tactical work:
- **FineBadminton** (singles, MLLM-assisted multi-level labels + FBBench QA)
- **BFMD** (full matches, singles + doubles, dense captions)
- **DShuttleNet** (doubles, 54K shots, loss reasons)
- **DoublesEval** (doubles VLM diagnostic; data not yet released)
- **Fine-Badminton (TAL)**
- **CourtSI** (spatial QA)
- **Shot2Tactic-Caption**

**Almost all are pro broadcast footage (BWF / YouTube). Amateur footage remains essentially unrepresented**; VideoBadminton (2024, collegiate players) is the closest.

Licensing is the weak spot. Only **ShuttleSet / ShuttleSet22 (MIT)** and **MultiSenseBadminton (CC BY 4.0 article / CC0 data per Figshare)** have clear grants. Several new datasets have **no license file** (BFMD) or unverified terms, and nearly all depend on YouTube/BWF video that cannot be redistributed.

### Cited Findings

#### Catalogue — NEW (2025 – Oct 2026)

| Dataset (venue, date) | Size | Annotations | Footage | License / access (as verified) |
|---|---|---|---|---|
| **FineBadminton** (ACM MM 2025; arXiv 2508.07554, Aug 2025) | 120 singles matches ("past five years"), 3,215 rally clips, **33,325 strokes**; FBBench **2,563 QA** | 3-level hierarchy (Foundational Actions → Tactical Semantics → Decision Evaluation); MLLM/LLM-proposed then expert-refined | Pro singles; broadcast implied, not explicitly confirmed | GitHub repo **Apache-2.0** (fetched); data on HF `Moujuruo/Finebadminton-20K` + `Moujuruo/FineBadmintonBenchmark`; **HF dataset-card license not verified**. The "20K" name does not match the 33,325-stroke count |
| **Shot2Tactic-Caption** (MMSports '25; arXiv 2510.14617) | 5,494 shot captions, 544 tactic captions | Shot + tactic captions, tactic types/states | Broadcast (presumed; same group as BFMD) | Not found |
| **COACH-Data** (AAAI 2026; arXiv 2512.01853, Dec 2025) | 22 matches, ~19K strokes | QA/summarization data built on ShuttleSet | Broadcast (ShuttleSet) | Release / license not found |
| **CourtSI / CourtSI-Bench** (arXiv 2603.09896, Mar 2026) | >1M QA; bench 3,686 human-verified QA | Spatial QA (distance, position, etc.) for badminton, tennis, table tennis | Broadcast | Not found |
| **BFMD** (CVPRW 2026; arXiv 2603.25533, Mar 2026) | 19 matches, 20+ h, 1,687 rallies, 16,751 hits (paper); **released**: 12 MS + 7 MD, 1,058 rallies, 11,301 shots | Match segments, rally events, shot types, shuttle trajectories, COCO-17 pose, court geometry, player boxes, per-hit captions | **Broadcast, BWF TV YouTube**, 1280×720@30 fps | Annotations on Google Drive; videos re-downloaded with yt-dlp script; **no LICENSE file (404) and none listed**; README: "Video copyright belongs to BWF TV; do not redistribute" |
| **BadmintonGRF** (arXiv 2605.01876, May 2026) | 8 synchronized RGB views ~120 FPS + force plates + mocap | Ground-reaction force, pose | Lab capture | License not found |
| **Fine-Badminton (TAL)** (IEEE TMM / arXiv 2605.23355, May 2026) | 31 matches, 2,104 rallies, **27,597 actions**, 29 stroke classes | Temporal action localization | YouTube, resampled 25 fps | Release / license not found. **Name collides with FineBadminton** |
| **TactiPlay** (arXiv 2607.27125, Jul 2026) | Not a dataset release; expert taxonomy from national-level athletes | Performance-issue taxonomy | **Amateur** matches (user study) | No data release found |
| **DoublesEval** (arXiv 2608.24439, Aug 2026) | 60 rallies from 4 pro doubles matches; ~9.6K instances | 4-level tactical QA (L1–L4) | Broadcast, BWF Super 750/1000 | Repo **Apache-2.0** but README: "Code will be soon available"; **no data posted** (fetched) |
| **DShuttleNet** (arXiv 2610.14016; listing dated Sep–Oct 2026) | **58 matches, 4,483 rallies, 54,141 shots**, BWF World Tour 2022–2025 | Synchronized positions of all 4 players; expert "Loss Reasons" (6 types); new CCEI / TLAI coordination metrics | Elite broadcast doubles | Hosted on **Kaggle** (DOI 10.34740/kaggle/dsv/16097938); **Kaggle license field not verified** |
| **SportR** (ICLR 2026; multi-sport) | ~4.8–5.0K images (badminton is in the image split only), ~2.1K videos | Fouls, tactics, CoT rationales, grounding | Mixed | HF media is **gated** ("Request access") |
| **RefereeBench** (arXiv 2604.15736, Apr 2026; multi-sport) | 925 videos, 6,475 QA, 11 sports incl. badminton | Officiating QA | Mixed | Not found |

Sources for table rows:
- FineBadminton: [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554); [GitHub iLearn-Lab/MM25-FineBadminton (fetched)](https://github.com/iLearn-Lab/MM25-FineBadminton); [HF dataset](https://huggingface.co/datasets/Moujuruo/Finebadminton-20K); [HF benchmark](https://huggingface.co/datasets/Moujuruo/FineBadmintonBenchmark)
- Shot2Tactic-Caption: [ACM DL](https://dl.acm.org/doi/10.1145/3728423.3759408)
- COACH-Data: [arXiv 2512.01853](https://arxiv.org/html/2512.01853v1)
- CourtSI: [arXiv 2603.09896](https://arxiv.org/html/2603.09896)
- BFMD: [arXiv 2603.25533](https://arxiv.org/pdf/2603.25533); [GitHub Ning-D/BFMD (fetched)](https://github.com/Ning-D/BFMD); [CVF](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Ding_BFMD_A_Full-Match_Badminton_Dense_Dataset_for_Dense_Shot_Captioning_CVPRW_2026_paper.pdf)
- BadmintonGRF: [arXiv 2605.01876](https://arxiv.org/abs/2605.01876)
- Fine-Badminton (TAL): [arXiv 2605.23355](https://arxiv.org/pdf/2605.23355)
- TactiPlay: [arXiv 2607.27125](https://arxiv.org/abs/2607.27125)
- DoublesEval: [arXiv 2608.24439](https://arxiv.org/abs/2608.24439); [GitHub Chengjt1999/DoublesEval (fetched)](https://github.com/Chengjt1999/DoublesEval)
- DShuttleNet: [arXiv 2610.14016](https://arxiv.org/abs/2610.14016); [Kaggle DOI](https://doi.org/10.34740/kaggle/dsv/16097938)
- SportR: [arXiv 2511.06499](https://arxiv.org/html/2511.06499v3); [HF gated](https://huggingface.co/datasets/haotianxia/SportR)
- RefereeBench: [arXiv 2604.15736](https://arxiv.org/html/2604.15736v1)

Also new in 2025, but code rather than data: **"Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models"** (arXiv 2508.13507, Aug 2025). ViT-Pose keypoints from ShuttleSet are embedded with contrastive ST-GCN, a custom multi-object tracker handles ID switches, and a Transformer detects shots in doubles. Code and pretrained weights are released on GitHub — [arXiv 2508.13507](https://arxiv.org/pdf/2508.13507)

#### Catalogue — EARLIER but still relevant (clearly pre-2025)

| Dataset (year) | Size | Annotations | Footage | License |
|---|---|---|---|---|
| **ShuttleSet** (KDD 2023) | 44 matches, 104 sets, 3,685 rallies, **36,492 strokes**, 27 top MS/WS players, 2018–2021 | 18 shot types, hit time/frame, score, backhand / around-head flags, landing area & coords, player positions, rally-end reason, winner; per-match homography + court corners | Pro broadcast | Repo **MIT License, "Copyright (c) 2022 Wei-Yao Wang"** (LICENSE fetched); dataset README asks for citation, no extra clause; annotations only (video inclusion not confirmed) |
| **ShuttleSet22** (IJCAI 2023 challenge / IJCAI 2024 demo) | Train 30,172 strokes / 2,888 rallies (+ val/test); 2022 matches | Stroke-level, as ShuttleSet | Pro broadcast | Same MIT repo (`CoachAI-Challenge-IJCAI2023` folder) |
| **VideoBadminton** (arXiv 2403.12385, 2024) | **7,822 clips**, 18 BWF-defined actions, 19 athletes, ~145 min, ~1.31 GB zip | Action class; racket trajectory & player position (per aggregator) | **Collegiate / non-broadcast** (National Central University team) | **License not found**; GitHub repo path not confirmed |
| **MultiSenseBadminton** (Scientific Data 2024) | 25 participants | Video, skeleton, eye gaze, foot pressure, arm/leg EMG, 17-tracker IMU | Lab / training | Article **CC BY 4.0**; Figshare data reportedly **CC0**. The sources conflict, and an institutional page says "all rights reserved unless otherwise indicated" |
| **SPORTU** (ICLR 2025, arXiv 2024) | 900 text MCQ + video set | Multi-sport rules / strategy QA | Mixed | Not checked |
| **ActionAtlas** (NeurIPS 2024) | 934 videos, 580 actions, 56 sports | Domain-specialized action MCQ | YouTube | Not checked |

Sources for earlier datasets:
- ShuttleSet: [GitHub ShuttleSet (fetched)](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet); [LICENSE (fetched)](https://github.com/wywyWang/CoachAI-Projects/blob/main/LICENSE); [arXiv 2306.04948](https://arxiv.org/pdf/2306.04948)
- ShuttleSet22: [arXiv 2306.15664](https://arxiv.org/pdf/2306.15664)
- VideoBadminton: [arXiv 2403.12385](https://arxiv.org/html/2403.12385v2); [hyper.ai listing](https://hyper.ai/en/datasets/30582)
- MultiSenseBadminton: [Nature Sci Data](https://www.nature.com/articles/s41597-024-03144-z); [Figshare](https://springernature.figshare.com/collections/MultiSenseBadminton_Wearable_Sensor_Based_Biomechanical_Dataset_for_Evaluation_of_Badminton_Performance/6725706/1)
- SPORTU: [arXiv 2410.08474](https://arxiv.org/abs/2410.08474)
- ActionAtlas: [arXiv 2410.05774](https://arxiv.org/pdf/2410.05774)

#### Other notes
- BadmintonDB (player-specific match analysis and prediction) is cited by the 2026 sports survey; its size and license were not checked — [arXiv HTML 2608.14377v1](https://arxiv.org/html/2608.14377v1)
- Several Roboflow Universe "badminton" detection sets are CC BY 4.0. They are unrelated to stroke or VLM work — [Roboflow Universe example](https://universe.roboflow.com/badmintoncv/badminton-cv)

### Inferences
- **Under the project's license rule** (anything OK except no license grant):
  - **ShuttleSet / ShuttleSet22 (MIT) are clean.**
  - **FineBadminton and DoublesEval code are Apache-2.0** at repo level, but the dataset terms need checking on the HF / Kaggle cards.
  - **BFMD currently has *no license grant* for its annotations** (no LICENSE file, none listed), so it falls in the project's hard-avoid category until the authors add one.
  - VideoBadminton and Fine-Badminton (TAL) have unknown terms. Treat them as "no grant" until verified.
- **Footage domain:** every large 2025–26 badminton set is elite broadcast with a high, behind-baseline camera. The club's footage is amateur DJI ultrawide with white walls. Models or fine-tunes trained on these sets need a **held-out amateur validation set** before any metric is shown. VideoBadminton (collegiate players) is the closest public proxy.
- **Doubles** (P3+ in the plan) now has public stroke-level data: DShuttleNet (54K shots, 4-player positions) and BFMD's 7 MD matches. DoublesEval provides a VLM diagnostic once its data is released.
- FineBadminton's labelling pipeline is exactly the weak-label pattern the project is piloting, and the "20K" HF release name suggests the public release may be a subset. Both are worth checking before use.

### Gaps
- License fields on the HF dataset cards (FineBadminton, FBBench), Kaggle (DShuttleNet), and the VideoBadminton repo could not be read, because HF and Kaggle are blocked by the sandbox proxy and the VideoBadminton repo URL was not found.
- FineBadminton's footage source (broadcast vs other) and the meaning of "20K" in the HF dataset name were not confirmed.
- BFMD per-match doubles video size and any separate annotation license were not stated.
- DShuttleNet's exact posting date is inconsistent: the search said a Sep 2026 listing, while ID 2610 implies Oct 2026.
- I found no public **amateur-footage** badminton stroke dataset from 2025–26, and no German-club or club-level dataset.
- No 3D / multi-view *tactical* badminton dataset was found for 2025–26 apart from BadmintonGRF (lab biomechanics). "Beyond the Single Camera: Agentic Multi-View Reasoning in Sports Video Understanding" (arXiv 2607.11844) surfaced but was not inspected — [arXiv HTML 2607.11844](https://arxiv.org/html/2607.11844)
- No benchmark named "BadmintonQA" exists in any search result. The nearest are FBBench, DoublesEval, and the badminton parts of SportR, RefereeBench and CourtSI.

---

## Q4. Practical guidance: how are teams using VLMs as weak labelers / pre-labelers for sports actions, and what accuracy is achievable with small (≤8B) local models vs large API models?

### Takeaway
The only badminton paper that uses MLLMs as labelers (FineBadminton) uses them to **propose** multi-level labels on top of a **CV structural parse (tracking + hit detection)**, with **expert refinement** of every proposal. It does not trust them raw.

Published evidence gives:
- **zero-shot ≤8B models: about 25–45 %** on fine-grained sports multiple choice
  - FBBench small models <25 %
  - DoublesEval about 27–32 %
  - ActionAtlas Qwen2-VL-7B about 30 %
  - VideoNet Qwen3-VL-8B 45 % MC / 59 % binary
- **frontier APIs: about 39–70 %** depending on the benchmark
- **a fine-tuned 7–8B model can match or beat frontier zero-shot**: FBBench 42 % vs 38.6 %; TRACE 44.8 vs 18.9; CourtSI +23.5 pp; SportR foul classification 14 → 51 %

No study measures pre-labeling accuracy, speed-up or agreement for VLM badminton stroke labels on amateur footage.

### Cited Findings
- FineBadminton's pipeline: foundational video models parse structure (player/ball tracking, **hit detection**). MLLMs/LLMs then propose multi-level annotations, which experts review and refine. The MLLM step turns structured data into text narratives — [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554)
- On the same benchmark:
  - small open models zero-shot: **<25 %**
  - Gemini 2.5 Pro with hit-centric strategy S: **38.6 %**
  - Qwen2.5-VL-7B fine-tuned: **42.06 %**

  — [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554)
- Hit-centric keyframe selection (strategy S) improved commercial models over uniform sampling at an equal frame budget — [arXiv 2508.07554](https://arxiv.org/pdf/2508.07554). ActionAtlas finds a high frame rate essential for fine actions — [arXiv 2410.05774](https://arxiv.org/pdf/2410.05774). TennisTV finds the optimal sampling density is task-dependent — [arXiv 2509.15602](https://arxiv.org/pdf/2509.15602)
- **Few-shot in-context examples:** they help Qwen3-VL-8B (+7.0 %) but hurt Gemini (−4.8 %) on domain-specific actions. Per the VideoNet authors (via a secondary summary), one in-context example helped more than extra frame sampling — [arXiv 2605.02834](https://arxiv.org/html/2605.02834v2)
- **Small vs large, zero-shot, domain actions:**

  | Benchmark | Small open model | Frontier API model | Chance |
  |---|---|---|---|
  | ActionAtlas | Qwen2-VL-7B ~30.2 % | GPT-4o 45.5 % | ~21 % |
  | VideoNet (MC) | Qwen3-VL-8B 45.0 % | Gemini 3.1 Pro 69.9 % | — |
  | VideoNet (binary) | Qwen3-VL-8B 59.3 % | GPT-5.4 72.4 % | 50 % |

  — [ActionAtlas](https://arxiv.org/pdf/2410.05774); [VideoNet author page](https://tanu.sh/research/videonet)
- **Fine-tuning closes the gap:**
  - SportR: Qwen2.5-VL-7B SFT + GRPO raised foul classification from 14.4 to 50.7 % — [Paper note](https://en.papernotes.org/ICLR2026/vlm_reasoning/sportr_a_benchmark_for_multimodal_large_language_model_reasoning_in_sports/)
  - CourtSI: Qwen3-VL-8B +23.5 pp — [arXiv 2603.09896](https://arxiv.org/html/2603.09896)
  - TennisVAR/TRACE: SFT Qwen3-VL-8B 44.83 vs zero-shot Gemini-3.1-Pro 18.87 — [arXiv 2608.12920](https://arxiv.org/pdf/2608.12920)
- **Consistency checks help.** TacticCheck reranks answers using the model's own lower-level predictions, worth +8.2 points on Qwen3-VL — [arXiv HTML 2608.24439v1](https://arxiv.org/html/2608.24439v1)
- **Pre-annotation human factors (adjacent, not a VLM and not sports).** On ambiguous video, AI pre-annotations cut labelling time by about **35 %** for most of 18 annotators. The authors warn that suggestions can **pull human labels toward the model's priors**, so agreement rises while validity falls — [arXiv 2510.21798](https://arxiv.org/html/2510.21798)
- **VLM-as-oracle pseudo-labels.** One corpus used Qwen3-VL-32B as the oracle for temporal-importance labels on action recognition; **four different oracles agreed only moderately** — [arXiv 2610.16009](https://arxiv.org/html/2610.16009). LEViL distils VLM-generated pseudo-label spaces for UCF101/HMDB51, with no human-in-the-loop measurement — [arXiv 2606.21358](https://arxiv.org/pdf/2606.21358)
- **Pipeline designs that avoid VLM perception for tactics:** COACH (Flan-T5-XL over ShuttleSet-derived logs) beats Gemini 2.5 Pro on QA and summarization (self-reported) — [arXiv 2512.01853](https://arxiv.org/html/2512.01853v1). The CVPR 2026 debate paper replaces VLM perception with CV event logs plus LLM reasoning — [CVF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhang_From_Alignment_to_Reason_Multi-Agent_Debate_for_Tactical_Badminton_Video_CVPRF_2026_paper.pdf)
- **Qwen3.5-4B specifics** (vendor card; general video benchmarks only): Video-MME 76.9 without subtitles, MVBench 71.2 — [HF model card](https://huggingface.co/Qwen/Qwen3.5-4B). It runs in about 3 GB at 4-bit (third-party) — [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)

### Inferences
- **Realistic expectation for a 4B local VLM as a zero-shot badminton stroke labeler:** the closest measured analogues for ≤8B models are FBBench (<25 %) and DoublesEval (27–32 %), both on broadcast footage. A 4B model on amateur ultrawide footage should be assumed **unreliable as an unsupervised labeler**. It is plausible as a **proposal generator inside a human-review loop**, and only with:
  1. a **closed label set** (ShuttleSet's 18 shot types or a coarser merge)
  2. **hit-centred short clips** from the existing CV hit detector, not whole rallies
  3. an explicit **"unsicher / abstain"** option
  4. per-class precision/recall measured against a small **human-labelled amateur hold-out set** before any label feeds stats
- The strongest evidence-backed lever is **fine-tuning (LoRA/SFT) a 7–8B or smaller VLM on domain data** (ShuttleSet strokes, FineBadminton if its license permits). Self-reported gains of +17 to +36 points over zero-shot recur across four papers. Fine-tuning on broadcast data then still needs amateur-domain validation.
- **Coarse labels** (forehand/backhand, overhead vs underarm, serve vs non-serve) are likely much more attainable than the full 18-way taxonomy. This is inferred from the coarse-vs-fine pattern in ActionAtlas and VideoNet; no badminton-specific measurement exists.
- **Anchoring risk:** if reviewers see VLM pre-labels, agreement with the model can rise while correctness falls (arXiv 2510.21798). Periodically labelling blind, without suggestions, protects the hold-out set.
- **Self-consistency and constraint checks** (TacticCheck-style) are cheap on a local model and showed +8 points. For example: a stroke label must be compatible with the hit height or position from the CV pipeline, or several sampled answers must agree.
- **Model routing:** use local small VLMs for perception proposals, and if text output is needed, an LLM over structured stats (COACH / Court-to-Conversation pattern). Frontier API VLMs gain only about 10–25 points over small models zero-shot on fine-grained sports, and they still do not reach label-grade accuracy.

### Gaps
- **No published measurement** of VLM pre-labeling accuracy, annotator speed-up, or inter-annotator agreement for **badminton strokes**, and none for **amateur** racket-sport footage.
- **No numbers for Qwen3.5-4B (or any ≤4B model)** on any sports fine-grained benchmark.
- No public study of Apple Silicon (MPS / MLX / llama.cpp) VLM throughput for sports labelling was searched within this scope.
- No source quantified how much VLM labels improve a downstream stroke classifier versus training on ShuttleSet alone.
