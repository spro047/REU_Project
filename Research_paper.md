# Literature Survey: AI-Based Assessment in Surgical Suturing

## Project Context

This literature review supports the development of an **AI-powered framework for evaluating suture quality**, combining computer vision with expert-annotated suture image data. The survey identifies current approaches, benchmark datasets, validation gaps, and methodological patterns across 7 key publications spanning 2017–2026.

---

## 1. Summary Table

| # | Author(s) | Year | Title | Journal | Key Contribution | Pros | Limitations |
|---|-----------|------|-------|---------|------------------|------|-------------|
| 1 | Khalil S, Shah HA, Bednarik R | 2025 | Improving microsurgical suture training with automated phase recognition and skill assessment via deep learning | Computers in Biology and Medicine | LRCN-based phase recognition (T-LRCN: 83.85% accuracy) + dual metric (confidence + duration) for skill assessment; 11 surgeons, 132 sutures in 7 microsurgical phases | Novel skipping window strategy for class imbalance; dual metric skill assessment; good cross-dataset generalization | Small dataset; challenges with visually similar phases; microsurgery-specific (may not generalize to open/laparoscopic) |
| 2 | Spagnulo R, Marzola F, Corso F et al. | 2026 | From precision to strength: computer vision for suture quality assessment—an ex vivo pilot study | Surgical Endoscopy | Direct correlation between spatial metrics (spacing irregularity) and mechanical burst pressure (p < 0.05); 3 platforms (laparoscopy, dVRK, Flex); 15 participants | Novel link between visual suture spacing and mechanical strength; multi-platform comparison | Very small sample (N=45 procedures); XGBoost model failed (R² = -0.42); pilot study only |
| 3 | Pan M, Wang S, Li J et al. | 2023 | An Automated Skill Assessment Framework Based on Visual Motion Signals and a Deep Neural Network in Robot-Assisted Minimally Invasive Surgery | Sensors (MDPI) | KCF visual tracking + ResNet for RAMIS skill assessment; 92.04% (2-level) and 84.80% (3-level) accuracy; process time 3-5 seconds | No additional sensors needed (purely vision-based); near real-time processing; tested on JIGSAWS benchmark | Tested only on JIGSAWS dataset; limited real-world OR validation; 2D vision limitations |
| 4 | Kankanamge D, Wijeweera C, Ong Z et al. | 2025 | Artificial intelligence based assessment of minimally invasive surgical skills using standardised objective metrics – A narrative review | American Journal of Surgery | Systematic review of 24 citations; AI accuracy range 63-100% for predicting SOM scores; OSATS and GEARS most common SOMs (8/24 each) | Comprehensive review of AI + SOM integration; identifies heterogeneity as key barrier | High heterogeneity prevents meta-analysis; no standardized comparison framework; narrative (not systematic) review |
| 5 | Boal MWE, Anastasiou D, Tesfai F et al. | 2024 | Evaluation of objective tools and artificial intelligence in robotic surgery technical skills assessment: a systematic review | BJS (British Journal of Surgery) | 247 studies analyzed; 60% of AI methods >90% accuracy in lab (67-100% in real surgery); Messick's validity framework applied | Largest systematic review (247 studies); up-to-date; rigorous validity framework (Messick + OCEBM) | Most tools not validated for accreditation; lab-to-OR accuracy gap; AI studies lack methodological quality tools |
| 6 | Vedula SS, Ishii M, Hager GD | 2017 | Objective Assessment of Surgical Technical Skill and Competency in the Operating Room | Annual Review of Biomedical Engineering | OCASE-T framework defined; 45 publications reviewed; problem space for computer-aided surgical skill evaluation | Established foundational OCASE-T paradigm; comprehensive problem definition; key reference for the field | Outdated (2017); most studies simulation-based (not OR); no uniform validation methodology |
| 7 | IJgosse WM, Leijte E, Ganni S et al. | 2020 | Competency assessment tool for laparoscopic suturing: development and reliability evaluation (LS-CAT) | Surgical Endoscopy | LS-CAT with 4 task areas; excellent inter-rater reliability (Cohen's Kappa 0.75-0.87); 4-point ordinal scale across 36 videos | Excellent reliability; specific to suturing (unlike generic tools); Delphi expert consensus | Small sample (N=36); laparoscopic only; ordinal scale limits granularity |

### Papers Not Included (Scanned PDF)

- **Paper 10**: Unable to extract text — scanned/image-based PDF. Could not include in analysis.

### Other Files in Research Papers Directory

- **Paper-4 directory**: Contains supplementary articles from the same *American Journal of Surgery* issue (e.g., economics of emergency laparoscopy, COVID-19 effects on providers) — unrelated to suture assessment.
- **Paper-6 directory**: Contains journal metadata/cover pages from *Yonsei Medical Journal* — not a research paper.

---

## 2. Detailed Analysis Per Paper

### Paper 1 — Khalil et al. (2025)
**"Improving microsurgical suture training with automated phase recognition and skill assessment via deep learning"**

- **Method**: Three LRCN variants — ConvLSTM, T-LRCN (temporal), Hybrid LRCN (spatial + temporal)
- **Dataset**: 11 surgeons (6 novices, 5 experts), 132 microsurgical sutures annotated into 7 phases
- **Key finding**: T-LRCN achieved 83.85% testing accuracy for phase recognition
- **Innovation**: Skipping window strategy to handle class imbalance between phases; combined confidence scores with phase duration for skill level classification
- **Relevance to this project**: Demonstrates that deep learning can effectively segment suturing procedures and differentiate skill levels from video data alone

### Paper 2 — Spagnulo et al. (2026)
**"From precision to strength: computer vision for suture quality assessment"**

- **Method**: Extracted spatial features (distance variability between stitches) from video; correlated with mechanical burst pressure
- **Platforms**: Conventional laparoscopy, daVinci Research Kit (dVRK), Flex robotic endoscope
- **Key finding**: Spacing irregularity negatively correlated with burst pressure (p < 0.05) — more uniform spacing = stronger suture
- **Relevance**: Provides a direct link between visual stitch placement and mechanical strength — foundational for vision-based quality assessment
- **Limitation note**: XGBoost model had poor predictive performance (R² = -0.42), suggesting spatial features alone are insufficient

### Paper 3 — Pan et al. (2023)
**"An Automated Skill Assessment Framework Based on Visual Motion Signals and a Deep Neural Network"**

- **Method**: Kernel Correlation Filter (KCF) for surgical instrument tip tracking → ResNet classification
- **Dataset**: JIGSAWS (public benchmark for robotic surgery skill assessment)
- **Key finding**: 92.04% (2-level) / 84.80% (3-level) accuracy in 3-5 seconds processing time
- **Innovation**: No external sensors needed — purely vision-based using endoscopic video
- **Relevance**: Proves that 2D visual features can replace kinematic sensors for skill assessment

### Paper 4 — Kankanamge et al. (2025)
**"AI based assessment of MIS skills using standardised objective metrics"**

- **Scope**: 24 citations analyzing AI systems that predict SOM scores (OSATS, GEARS, etc.)
- **Key finding**: AI accuracy 63-100% — wide range indicates heterogeneity in methods and validation
- **Critical insight**: Stratifying for SOM use did NOT reduce heterogeneity — suggests fundamental differences in study design, not metric choice
- **Relevance**: Key issues identified: small datasets, overfitting, lack of external validation — all relevant to our project

### Paper 5 — Boal et al. (2024)
**"Evaluation of objective tools and AI in robotic surgery technical skills assessment"**

- **Scope**: 247 studies — largest systematic review in this domain
- **Findings**: 8 global rating scales, 26 procedure-specific tools, 3 error-based methods, 10 simulators, 28 APM studies, 53 AI studies
- **Lab vs. OR gap**: 60% of AI methods >90% accuracy in lab, dropping to 67-100% in real surgery
- **Relevance**: Provides the most comprehensive landscape analysis; confirms that AI tools are not yet validated for accreditation

### Paper 6 (Paper 7 in folder) — Vedula, Ishii, & Hager (2017)
**"Objective Assessment of Surgical Technical Skill and Competency in the OR"**

- **Scope**: 45 publications; introduced the OCASE-T (Objective Computer-Aided Surgical Skill Evaluation) framework
- **Key contribution**: Defined the problem space, data sources (kinematic, video, system events), representations, and algorithm categories
- **Limitation**: Predates modern deep learning advances; most studies were simulation-based
- **Relevance**: Foundational framework that later work builds upon

### Paper 7 (Paper 8 in folder) — IJgosse et al. (2020)
**"Competency assessment tool for laparoscopic suturing: development and reliability evaluation (LS-CAT)"**

- **Method**: Delphi consensus with 12+ experts → 4-task evaluation with 4-point ordinal scale
- **Key finding**: Excellent inter-observer reliability (Cohen's Kappa 0.75-0.87, p < 0.001)
- **Core metrics**: Instrument handling (2 tasks) + Tissue handling (2 tasks)
- **Relevance**: Only suturing-specific competency assessment tool found; excellent reliability makes it a strong candidate for AI training ground truth

---

## 3. Cross-Cutting Themes

### 3.1 Benchmark Datasets
| Dataset | Used By | Type | Size |
|---------|---------|------|------|
| JIGSAWS | Pan et al. (2023) | Robotic surgery kinematics + video | ~100-200 trials |
| Custom microsurgical | Khalil et al. (2025) | Microsurgical video | 132 sutures |
| Custom porcine | Spagnulo et al. (2026) | Laparoscopic/robotic video | 45 procedures |

**Gap**: No publicly available suture-specific image dataset with expert quality ratings exists — making our dataset uniquely valuable.

### 3.2 Common Methodologies
- **Feature extraction**: Most papers use either kinematic sensors or visual tracking to extract motion features
- **Deep learning**: CNN/ResNet for spatial features; LSTM/TCN for temporal; hybrid models emerging
- **Validation**: Cross-validation common; external validation rare; OR validation very rare

### 3.3 Validation Standards

| Standard | Description | Papers Using It |
|----------|-------------|----------------|
| Messick's validity | 5 aspects (content, response process, internal structure, relations to other variables, consequences) | Boal et al. (2024) |
| Cohen's Kappa | Inter-rater reliability (beyond chance) | IJgosse et al. (2020) |
| Accuracy (%) | Classification performance | Khalil, Pan, Kankanamge, Boal |

**Gap**: No standard validation framework exists — Boal et al. (2024) advocate for Messick's framework as a solution.

### 3.4 Key Research Gaps Identified

1. **No suture-specific AI assessment** — Most work focuses on general surgical skill, not suture quality specifically
2. **Lab-to-OR translation** — Accuracy drops significantly from lab to operating room
3. **Small datasets** — All studies use small, custom datasets; no large public suture dataset
4. **No validation for accreditation** — No AI tool is validated for high-stakes assessment
5. **Lack of standard metrics** — No unified framework for comparing AI assessment methods
6. **Scanned image challenges** — Real-world suture photos (our dataset) differ from controlled video frames

---

## 4. Proposed Methodology — Addressing Research Limitations

Based on the limitations identified across all 7 papers, we design a methodology that systematically addresses each gap.

### 4.1 Limitations Addressed

| Limitation | Source Papers | How We Address It |
|------------|---------------|-------------------|
| Small datasets (N < 150) | Khalil, Spagnulo, IJgosse | 1,648 images — the largest suture-specific dataset |
| Video/kinematic sensor dependency | Pan, Khalil, Vedula | Static image analysis — no temporal data or sensors needed |
| Binary novice/expert classification | Khalil, Pan, IJgosse | 6 continuous quality metrics (score 1-10) per image |
| No validation for accreditation | Boal, Kankanamge | Uncertainty quantification with confidence intervals and rejection thresholds |
| Lab-to-OR accuracy drop | Boal | Application cohort (432 images) simulates real-world deployment |
| Black-box predictions | All papers | Per-metric Grad-CAM heatmaps for interpretability |
| No standardized validation framework | Vedula, Kankanamge | Messick's 5-aspect validity framework applied throughout |
| Domain specificity (single surgery type) | Khalil, IJgosse | General suture quality on synthetic skin — domain-agnostic |
| Class imbalance | Khalil, Our dataset | Weighted sampling + data augmentation for minority groups |
| Single outcome metric | Spagnulo (burst pressure only) | 6-dimensional multi-task output capturing all quality dimensions |

### 4.2 Framework Overview — Suture-Q AI

```
┌─────────────────────────────────────────────────────────┐
│                    Suture-Q AI Framework                  │
├─────────────────────────────────────────────────────────┤
│  Input: Single suture image (PNG, top-down view)         │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  Backbone: EfficientNet-B3 / Swin-Tiny         │      │
│  │  (Pretrained on ImageNet → fine-tuned)         │      │
│  └──────────────┬─────────────────────────────────┘      │
│                 │                                        │
│    ┌────────────┼────────────┬────────────┬──────┐       │
│    │            │            │            │      │       │
│  ┌─▼──┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐ ┌▼────┐       │
│  │Over-│  │ ISD  │  │ Slack │  │Posi- │ │Angu-│        │
│  │all  │  │ regr. │  │ regr. │  │tion  │ │lation│        │
│  │regr.│  │       │  │       │  │regr. │ │regr. │        │
│  └──┬──┘  └───┬───┘  └───┬───┘  └───┬───┘ └──┬───┘       │
│     └─────────┼──────────┼──────────┼─────────┘            │
│               │          │          │                      │
│     ┌─────────▼──────────▼──────────▼──────────┐          │
│     │     Uncertainty (MC Dropout × 10)         │          │
│     │     → Mean + 95% CI per metric            │          │
│     └────────────────┬─────────────────────────┘          │
│                      │                                    │
│     ┌────────────────▼─────────────────────────┐          │
│     │     Explainability (Grad-CAM)              │          │
│     │     → 6 heatmaps showing region importance │          │
│     └───────────────────────────────────────────┘          │
│                                                          │
│  Output: 6 quality scores + confidence intervals + heatmaps│
└─────────────────────────────────────────────────────────┘
```

### 4.3 Components

#### 4.3.1 Data Strategy
- **Scale**: 1,648 images (1,216 annotated) — addresses the small-dataset gap
- **Augmentation**: Rotation (±15°), horizontal flip, brightness jitter, random crop — simulates real-world variability
- **Weighted sampling**: Inverse frequency weighting for Doctor 7 (7 images) to prevent class imbalance from biasing the model
- **Cohort design**: Training (1,010) for learning, Validation (206) for hyperparameter tuning, Application (432) for unseen-data testing — mirrors real-world deployment pipeline

#### 4.3.2 Multi-Task Architecture
- **Backbone**: EfficientNet-B3 (balanced accuracy/efficiency) or Swin-Tiny (better spatial reasoning for fine-grained suture assessment)
- **Heads**: 6 separate regression heads, each outputting a score in [1, 10]
- **Shared representations**: Lower layers learn general suture features (thread, tissue, needle); upper layers specialize per metric
- **Why multi-task**: Improves generalization by sharing information across related metrics (e.g., Slack and Width both involve thread tension)

#### 4.3.3 Uncertainty Quantification
- **Method**: Monte Carlo Dropout — 10 stochastic forward passes at inference
- **Output**: Predictive mean + 95% confidence interval per metric
- **Rejection rule**: If confidence interval width > 3.0 (on 1-10 scale), flag for human expert review
- **Why**: No previous suture AI provides uncertainty estimates — essential for accreditation-grade assessment (Boal's key concern)

#### 4.3.4 Explainability
- **Method**: Grad-CAM per metric — generates a heatmap highlighting image regions that most influence each score
- **Clinical value**: A trainee can see *why* Slack scored low (heatmap on loose thread) vs why Width scored high (consistent stitch sizes)
- **Why**: Addresses the black-box criticism applicable to all prior AI assessment systems

#### 4.3.5 Validation Using Messick's Framework
Following Boal et al. (2024), we apply all 5 aspects of Messick's validity:

| Aspect | Implementation |
|--------|---------------|
| **Content** | 6 metrics defined by surgical experts; annotation guidelines standardized |
| **Response process** | Inter-rater reliability analysis on a subset of double-annotated images |
| **Internal structure** | Correlation analysis, factor analysis, metric independence tests |
| **Relations to other variables** | Cross-cohort consistency (Train vs Validation); comparison with known experience levels |
| **Consequences** | Error analysis on Application cohort; confusion matrix analysis for borderline cases |

### 4.4 Comparison with Prior Work

| Aspect | Khalil (2025) | Spagnulo (2026) | Pan (2023) | IJgosse (2020) | **Suture-Q AI (Ours)** |
|--------|---------------|-----------------|------------|----------------|------------------------|
| Sample size | 132 sutures | 45 procedures | JIGSAWS (~200) | 36 videos | **1,648 images** |
| Assessment type | Phase + skill | Burst pressure | Novice/expert | 4-pt ordinal | **6 continuous metrics** |
| Input | Video | Video + pressure | Video | Video | **Single static image** |
| Uncertainty | No | No | No | No | **Yes (MC Dropout)** |
| Explainability | No | No | No | No | **Yes (Grad-CAM)** |
| Validation framework | None | None | None | Cohen's Kappa | **Messick (5 aspects)** |
| Real-world test | No | No | No | No | **Yes (Application cohort)** |

### 4.5 Key Innovations

1. **First static-image suture quality model** — All prior work requires video or kinematic sensors
2. **Largest suture-specific dataset** — 10× larger than the next largest study
3. **Uncertainty-aware predictions** — Enables accreditation-grade confidence assessment
4. **Interpretable per-metric heatmaps** — Trainees see *why* each score was assigned
5. **Standardized validation** — Messick framework enables comparison with future work
6. **Multi-task learning** — 6 correlated metrics improve each other through shared representations

---

## 5. Relevance to Our Project

### 4.1 Directly Applicable Findings

| Finding | Source | Application |
|---------|--------|-------------|
| Slack correlates most with Overall quality (r=0.722) | Dataset analysis | Primary feature for AI model |
| Spacing irregularity linked to mechanical weakness | Spagnulo et al. (2026) | ISD metric validation |
| 2D vision features can replace sensors | Pan et al. (2023) | Our static image approach is viable |
| LS-CAT provides suturing-specific ground truth model | IJgosse et al. (2020) | Potential comparator for our 6-metric scoring |

### 4.2 Methodological Positioning

Our project differs from existing work in two key ways:

1. **Static image analysis** (not video) — Most prior work uses video/kinematics; we assess quality from single photographs
2. **Expert-annotated quality metrics** (6 dimensions) — More granular than binary novice/expert labels
3. **Largest suture-specific dataset** (1,648 images, 1,216 annotated) — Addresses the small-dataset gap

### 4.3 Challenges We Must Address

- **No temporal information** — Can't use motion-based features; must rely purely on spatial/visual cues
- **Class imbalance** — Some doctors have very few images (Doctor 7: 7 images)
- **Score distribution** — Most scores cluster in middle range (4-7), limiting discriminability
- **Standard metric alignment** — Our 6 metrics (Overall, ISD, Slack, Position, Angulation, Width) don't directly map to OSATS/GEARS/LS-CAT

---

## 5. References

1. Khalil, S., Shah, H.A., & Bednarik, R. (2025). Improving microsurgical suture training with automated phase recognition and skill assessment via deep learning. *Computers in Biology and Medicine, 192*, 110238.
2. Spagnulo, R., Marzola, F., Corso, F., et al. (2026). From precision to strength: computer vision for suture quality assessment—an ex vivo pilot study. *Surgical Endoscopy, 40*, 1913–1924.
3. Pan, M., Wang, S., Li, J., et al. (2023). An Automated Skill Assessment Framework Based on Visual Motion Signals and a Deep Neural Network in Robot-Assisted Minimally Invasive Surgery. *Sensors, 23*, 4496.
4. Kankanamge, D., Wijeweera, C., Ong, Z., et al. (2025). Artificial intelligence based assessment of minimally invasive surgical skills using standardised objective metrics – A narrative review. *The American Journal of Surgery, 241*, 116074.
5. Boal, M.W.E., Anastasiou, D., Tesfai, F., et al. (2024). Evaluation of objective tools and artificial intelligence in robotic surgery technical skills assessment: a systematic review. *BJS, 111*(1), znad331.
6. Vedula, S.S., Ishii, M., & Hager, G.D. (2017). Objective Assessment of Surgical Technical Skill and Competency in the Operating Room. *Annual Review of Biomedical Engineering, 19*, 301–325.
7. IJgosse, W.M., Leijte, E., Ganni, S., et al. (2020). Competency assessment tool for laparoscopic suturing: development and reliability evaluation (LS-CAT). *Surgical Endoscopy, 34*, 2947–2953.

---

*Generated July 2026 — D:\REU\research_papers*
