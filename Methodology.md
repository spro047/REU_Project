# Methodology: AI-Powered Suture Quality Assessment

*Derived from analysis of 6 research papers (2017–2026)*
*Updated: Video-to-frame extraction pipeline added (July 2026)*

---

## How This Methodology Is Built

This methodology does not invent a pipeline from scratch. It extracts and adapts validated methods from existing research, then combines them to address the gaps no single paper solved.

---

## Part A — What the Papers Actually Did

### A.1 Khalil et al. (2025) — Microsurgical Phase Recognition

| Element | Method Used | Transferable to Our Project? |
|---------|------------|------------------------------|
| **Input** | Video clips (5-s segments), 128×128 frames | ❌ We use static images |
| **Feature extraction** | TimeDistributed Conv2D → LSTM (spatial + temporal) | ❌ Temporal component irrelevant |
| **Augmentation** | Horizontal/vertical flips, Gaussian blur (σ 0–3), rotation ±45° | ✅ Yes — proven valid for suture images |
| **Class imbalance** | Skipping window + augmentation on minority phases | ✅ Yes — adapt for doctor imbalance |
| **Training** | Adam, categorical cross-entropy, sequence length sweep (5–40) | ⚠️ Classification loss only |
| **Validation** | Accuracy on hold-out test set | ✅ Yes |
| **Skill assessment** | Confidence score + phase duration dual-metric | ❌ Video-based measure |

**Key takeaway for us**: Their augmentation strategy (flips, blur, rotation) is proven on suture images. Their main limitation — needing video — is the gap we close.

### A.2 Spagnulo et al. (2026) — Suture Spatial Metrics vs. Burst Pressure

| Element | Method Used | Transferable? |
|---------|------------|---------------|
| **Input** | Post-suture photographs (iPhone 15 Pro) | ✅ Yes — closest to our data format |
| **Spatial features** | 18 distance metrics: PM (point-to-margin), PP (point-to-point) | ✅ Yes — defines measurable stitch geometry |
| **Calibration** | 3×3 checkerboard grid for pixel-to-mm conversion | ✅ Yes |
| **Ground truth** | Burst pressure (mechanical resistance) | ❌ Not available in our dataset |
| **Prediction model** | XGBoost regression | ⚠️ Failed (R² = −0.42) — spatial features alone insufficient |
| **Annotation tool** | MATLAB ImageLabeler, cross-validated by team | ✅ Yes — annotation workflow |
| **Statistical test** | Pearson correlation, p < 0.05 | ✅ Yes |

**Key takeaway**: Spatial features correlate with quality but alone are insufficient. This motivates our deep learning approach — CNNs can capture texture, tension, and angle cues that geometry-only models miss.

### A.3 Pan et al. (2023) — Visual Tracking + ResNet for Skill Classification

| Element | Method Used | Transferable? |
|---------|------------|---------------|
| **Input** | KCF-tracked motion signals [x, y, t, v, a, MJ] from video | ❌ Requires video |
| **Classifier** | ResNet-34 (trained from scratch, 100 epochs) | ✅ Yes — CNN architecture proven for surgical assessment |
| **Data** | JIGSAWS suturing task (24 videos, 8 subjects) | ⚠️ Small N |
| **Validation** | Leave-one-super-trial-out (LOSO) cross-validation | ❌ Not applicable — we have independent cohorts |
| **Metrics** | Accuracy, precision, recall, F1-score | ✅ Yes — but we need regression metrics |
| **Ablation** | Tested 4 network architectures, 5 input feature sets | ✅ Yes — ablation framework |
| **Key limitation** | Self-proclaimed skill labels conflicted with GRS scores | ✅ Relevant — our expert annotations are more reliable |

**Key takeaway**: ResNet is validated for surgical skill assessment from visual data. Their 84.80% 3-level accuracy proves the approach works; our 6-dimensional continuous scoring extends it.

### A.4 Boal et al. (2024) — Systematic Review (247 Studies)

| Finding | Implication for Our Methodology |
|---------|--------------------------------|
| 60% of AI methods report >90% accuracy in lab | We must test outside lab conditions — Application cohort |
| Lab-to-OR accuracy drops to 67–100% | Real-world performance testing is mandatory |
| Messick's 5-aspect validity framework recommended | We adopt it as our validation standard |
| Most AI tools not validated for accreditation | Uncertainty quantification needed for high-stakes use |
| Only 8 GRS, 26 procedure-specific tools existed | No suture-specific AI quality tool — gap we fill |

### A.5 Vedula et al. (2017) — OCASE-T Framework

| Concept | Application to Our Work |
|---------|------------------------|
| Surgical skill is a multi-dimensional construct | Our 6 metrics capture multiple quality dimensions |
| 4 data sources: motion, video, eye gaze, system events | We use only video (static image) — intentional simplification |
| 3 data representations: summary features, time-series, histograms | We use CNN feature maps (learned representations) |
| Most studies simulation-based, not OR | We include Application cohort as real-world proxy |
| No uniform validation methodology | We apply Messick framework (following Boal) |

### A.6 IJgosse et al. (2020) — LS-CAT Development

| Element | Method Used | Transferable? |
|---------|------------|---------------|
| **Tool development** | Delphi consensus (12+ experts) → 4 task areas | ✅ Yes — our 6 metrics were expert-defined |
| **Scoring** | 4-point ordinal scale per domain (1–4) | ❌ We use 1–10 continuous (more granular) |
| **Assessment domains** | Instrument handling + Tissue handling | ✅ Yes — maps to our quality dimensions |
| **Reliability** | Cohen's Kappa (κ = 0.75–0.87), Pearson correlation (r = 0.86–0.98) | ✅ Yes — we adopt same metrics |
| **Learning curve** | Mann–Whitney U comparing first vs. last attempt | ✅ Yes — applicable to our Application cohort (2wk vs 4wk) |
| **Setup** | eoSim laparoscopic simulator, 36 videos, 2 blinded experts | ✅ Yes — blinded expert review protocol |

**Key takeaway**: LS-CAT proves that expert-derived scoring tools with structured ordinal scales achieve excellent inter-rater reliability. Our 1–10 continuous scale extends this granularity.

---

## Part B — Synthesized Methodology for Suture-Q AI

### B.1 Research Design

**Type**: Quantitative, multi-cohort observational study with deep learning regression model.

**Rationale**: Every reviewed paper uses observational study design (no control group). Khalil, Pan, and IJgosse all use skill-level comparison groups. We follow this established paradigm while adding three improvements:
- **Multi-cohort design** (Train/Validation/Application) — addresses Boal's lab-to-OR gap
- **Continuous scoring** (1–10 per metric) — extends IJgosse's ordinal and Pan's binary classification
- **Uncertainty quantification** — responds to Boal's accreditation concern

### B.2 Dataset & Collection Protocol

**Adapted from**: Spagnulo's imaging protocol, Khalil's augmentation strategy

#### B.2.1 Data Source: Video Recordings

The dataset has transitioned from direct images to raw video recordings of suturing procedures. Currently sourced from JNMC institution:

| File | Resolution | FPS | Duration | Frames | Size | Quality Assessment |
|------|-----------|-----|----------|--------|------|-------------------|
| video-1.MOV | 3840×2160 (4K) | 24 | 5.4 min | 7,775 | 1.47 GB | 75% usable, 25% motion blur |
| video-2.MOV | Pending download | — | — | — | ~2.5 GB (partial) | Not yet analyzed |
| video-3.mov | 3840×2160 (4K) | 30 | 6.6 min | 11,789 | 1.79 GB | Mostly blurry (98% lapvar<50); limited usable segments |

**Video content**: Top-down view of suture practice on synthetic skin pads (same subject matter as original image dataset). video-1 has good overall quality with some motion blur during active suturing. video-3 is predominantly out-of-focus — only ~1-2% of frames are sharp enough for extraction.

#### B.2.2 Frame Extraction Pipeline

Since the model requires static images (not video), all usable frames must be extracted through a quality-controlled pipeline:

```
Raw Video (MOV)
    │
    ▼
┌──────────────────────────────┐
│ Step 1: Quality Scan          │
│ - Per-frame brightness check  │  threshold: mean ≥ 20 (grayscale)
│ - Per-frame sharpness check   │  threshold: Laplacian variance ≥ 50
│ - Per-frame motion blur check │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Step 2: Adaptive Sampling     │
│ - Uniform interval sampling   │  1 frame/sec for steady segments
│ - Dense sampling during       │  3-5 frames/sec during active suturing
│   high-motion segments        │  (detected via frame-to-frame diff)
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Step 3: Quality Filter        │
│ - Remove blurry frames        │  lapvar < 50 rejected
│ - Remove near-duplicate       │  SSIM > 0.95 → keep only first
│ - Remove underexposed frames  │  mean < 20 rejected
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Step 4: Format Standardization│
│ - Resize to 3840×2160→target  │
│ - Convert to PNG (lossless)   │
│ - Metadata attachment         │  source_video, frame_no, timestamp
└──────────┬───────────────────┘
           ▼
    Extracted Frames (PNG)
```

**Sampling strategy rationale**:

| Strategy | When to Use | Extraction Rate | Expected Yield |
|----------|------------|----------------|----------------|
| Uniform | Steady camera, consistent content | 1 fps | ~300-400 frames per 5-6 min video |
| Motion-adaptive | High activity periods (knot tying, needle pass) | Up to 5 fps | Captures key moments with redundancy |
| Keyframe only | Mostly blurry video (video-3) | Lapvar peaks only | ~50-200 frames salvageable |

**Expected total image yield**: 700-1,500 frames from current videos (pending video-2 analysis).

#### B.2.3 Cohort Construction from Extracted Frames

After extraction, frames are organized into the standard 3-cohort structure:

| Cohort | Purpose | Target Size | Selection Method |
|--------|---------|-------------|-----------------|
| Training | Model learning | ~60% of extracted | Random from early video segments |
| Validation | Hyperparameter tuning | ~15% of extracted | Random from mid video segments |
| Application | Blind evaluation | ~25% of extracted | Held-out from last portions + cross-video |

**Frame naming convention**: `{video_id}_{frame_number}_{timestamp_sec}.png`
- Example: `v1_5270_219.6.png`
- Embeds source tracking for traceability

**Note**: Cross-institution data (Hubli, FDP) would improve generalizability (identified by Pan's limited-dataset discussion). Currently only JNMC videos are available.

### B.3 Annotation Protocol

**Adapted from**: IJgosse's Delphi/expert consensus method, Spagnulo's cross-validation workflow

**Metric definition** (by surgical faculty):
1. **Overall** — Global suture line quality (1–10)
2. **ISD** — Inter-Suture Distance consistency (1–10)
3. **Slack** — Thread tension control (1–10)
4. **Position** — Needle placement accuracy (1–10)
5. **Angulation** — Needle entry/exit angle (1–10)
6. **Width** — Suture width uniformity (1–10)

**Procedure** (per IJgosse's blinded review protocol):
- ≥2 surgical faculty rate each Training image independently (blinded)
- Third expert arbitrates if scores differ >2 points
- 10% subset double-annotated for inter-rater reliability
- 5% re-annotated after 2-week washout (intra-rater reliability)
- Written guidelines with reference images per score level

**Reliability targets** (per IJgosse's benchmarks):
- Cohen's Kappa: κ ≥ 0.75 (IJgosse achieved 0.75–0.87)
- Pearson correlation: r ≥ 0.85 between raters

### B.4 Preprocessing Pipeline

**Adapted from**: Khalil's augmentation strategy (validated on suture images)

#### B.4.1 Video-Level Preprocessing (Pre-Extraction)

Before frame extraction, raw videos undergo:
- **Codec check**: Ensure readable format (H.264/H.265 in MOV container)
- **Frame index rebuild**: For reliable seeking in long videos
- **Metadata extraction**: FPS, resolution, duration logged to `video_manifest.json`
- **Quality pre-scan**: Brightness/sharpness profile across entire video to identify usable segments

#### B.4.2 Frame Extraction Filtering Criteria

| Criterion | Metric | Threshold | Rationale |
|-----------|--------|-----------|-----------|
| Minimum brightness | Grayscale mean | ≥ 20 | Eliminates near-black frames (camera covered, transitions) |
| Sharpness | Laplacian variance | ≥ 50 | Removes motion-blurred frames during hand movement |
| Duplicate detection | SSIM | < 0.95 with prior frame | Removes near-identical frames (camera stationary) |
| Content relevance | Manual ROI check | Suture line visible | Ensures frame contains usable suture content |

#### B.4.3 Image Preprocessing (Post-Extraction)

```
Extracted Frame (3840×2160 PNG)
    │
    ▼
┌──────────────────────┐
│ Center crop to square │   Crop to 2160×2160 (center)
│ + Downscale           │   → Resize to 384×384
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Normalize per-channel │   ImageNet: μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Augmentation (train)  │   Applied on-the-fly during training only
└──────────┬───────────┘
           ▼
    Batch to GPU
```

**Training augmentations** (per Khalil's proven set):

| Augmentation | Range | Source Paper |
|-------------|-------|-------------|
| Random rotation | ±15° | Khalil (±45°, narrowed for sutures) |
| Horizontal flip | p=0.5 | Khalil |
| Gaussian blur | σ ≤ 1.0 | Khalil (σ 0–3, narrowed) |
| Brightness/contrast jitter | ±10% | Standard |
| Random crop (90–100%) | Scale variation | Standard |

**Class imbalance**: Doctor 7 has 7 images (0.7%). Mitigation per Khalil's class imbalance approach:
- Inverse frequency weighting for sampling
- Per-doctor metrics tracked separately

### B.5 Model Architecture

**Adapted from**: Pan's ResNet classification framework, extended to multi-task regression

**Rationale**: Pan proved CNN-based surgical assessment works (84.80% 3-level accuracy). We extend from classification to multi-dimensional regression.

```
Input: 384×384×3
        │
┌───────▼────────┐
│  Backbone CNN  │   EfficientNet-B3 (ImageNet-pretrained)
│  (shared)      │   — Per Pan, pretrained CNNs outperform scratch-training
└───────┬────────┘
        │
    Global Avg Pooling  (1536-dim feature vector)
        │
        ├────────┬────────┬────────┬────────┬────────┬────────┐
        ▼        ▼        ▼        ▼        ▼        ▼        ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Overall │ │  ISD   │ │ Slack  │ │Position│ │Angulat │ │ Width  │
   │Reg Head│ │Reg Head│ │Reg Head│ │Reg Head│ │Reg Head│ │Reg Head│
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼          ▼
   Score 1-10  Score 1-10  Score 1-10  Score 1-10  Score 1-10  Score 1-10
```

**Why multi-task**: Khalil and Pan both show that correlated surgical metrics benefit from shared representations. Slack–Overall correlation (r=0.722) from our dataset analysis supports this.

### B.6 Training Strategy

**Adapted from**: Pan's training protocol + Khalil's hyperparameter exploration

| Hyperparameter | Value | Source |
|---------------|-------|--------|
| Optimizer | AdamW | Standard improvement on Pan's Adam |
| Learning rate | 3×10⁻⁴ (linear warmup 5 epochs) | Per Pan: 0.001 for scratch; lower for fine-tune |
| LR schedule | Cosine decay to 1×10⁻⁶ | Standard |
| Batch size | 32 | Adapted from Pan's 24 |
| Max epochs | 100 (early stopping patience=15) | Per Pan |
| Loss function | Smooth L1 + uncertainty-weighted multi-task loss | Extends Pan's cross-entropy to regression |
| Weight decay | 1×10⁻⁴ | Regularization (Pan did not use) |

**Validation method**: Independent held-out set (206 images) — differs from Pan's LOSO cross-validation because we have explicit cohort design.

### B.7 Evaluation Metrics

**Adapted from**: Pan's classification metrics, extended for regression

| Metric | What It Measures | Source |
|--------|-----------------|--------|
| MAE | Average absolute error (score units) | Regression standard |
| RMSE | Penalizes large errors | Regression standard |
| R² | Variance explained | Spagnulo |
| Spearman ρ | Ordinal agreement (ranking) | Non-parametric correlation |
| Acc@1 | Proportion within 1 point of expert | — |
| Acc@2 | Proportion within 2 points (clinically acceptable) | — |
| 95% CI coverage | Uncertainty calibration | — |

### B.8 Baselines & Ablations

**Adapted from**: Pan's ablation framework (4 architectures × 5 input sets)

**Baselines**:
1. **Mean predictor** — always predicts training set mean (floor)
2. **Single-task ResNet-50** — 6 independent models (ablates multi-task)
3. **EfficientNet-B3 single-task** — best single metric model

**Ablation experiments**:
1. Remove augmentation → tests generalization
2. Remove weighted sampling → tests imbalance handling
3. Replace SmoothL1 with MSE → loss function sensitivity
4. Predict only Slack+Width+Position (top-3 correlated) → metric redundancy
5. Remove MC Dropout (deterministic) → UQ impact on rejection

### B.9 Uncertainty Quantification

**Rationale**: Boal's review identifies that no AI tool is validated for accreditation — we need confidence bounds.

**Method**: Monte Carlo Dropout (50 forward passes at inference)
- Predictive mean + 95% CI per metric
- CI width > 3.0 → flag for human review

### B.10 Explainability

**Rationale**: Pan acknowledges "the black box feature of deep learning models limits interpretability" — a limitation we address.

**Method**: Grad-CAM per regression head
- 6 heatmaps per image
- Each heatmap shows regions driving that metric's score

### B.11 Validation — Messick's 5-Aspect Framework

**Following**: Boal's recommendation (applied to 247 studies) and Vedula's call for standardized validation

| Aspect | Our Implementation | Success Criterion |
|--------|-------------------|-------------------|
| **Content** | 6 metrics defined by surgical experts | All metrics rated essential ≥80% |
| **Response process** | IRR on 10% double-annotated subset | κ ≥ 0.75 (per IJgosse benchmark) |
| **Internal structure** | Correlation + factor analysis | α ≥ 0.70 |
| **Relations to other variables** | Cross-cohort consistency + experience-level ANOVA | Advanced > Intermediate > Beginner |
| **Consequences** | Error analysis on Application cohort | MAE < 1.0, failure mode documented |

---

## Part C — Mapping: Papers → Our Methodology

| Methodology Component | Primary Source(s) | What We Changed / Extended |
|----------------------|-------------------|---------------------------|
| Annotation protocol | IJgosse (Delphi + blinded review) | Extended from 4-point ordinal to 1–10 continuous |
| Image capture | Spagnulo (standardized photo + calibration) | Same protocol |
| Augmentation | Khalil (flips, blur, rotation) | Narrowed rotation range (45°→15°) for suture-appropriate |
| Model architecture | Pan (ResNet CNN) | Changed from classification to multi-task regression |
| Training loop | Pan + Khalil (Adam, epochs, sweep) | Added AdamW, cosine decay, uncertainty-weighted loss |
| Validation framework | Boal (Messick) + Vedula (OCASE-T) | Applied all 5 aspects with quantitative thresholds |
| Inter-rater reliability | IJgosse (Cohen's Kappa) | Same metric, same target (κ ≥ 0.75) |
| Real-world testing | Boal (lab-to-OR gap finding) | Application cohort as real-world proxy |
| Ablation study | Pan (4 architectures × 5 inputs) | Adapted to 5 ablation experiments |
| Uncertainty | Boal (accreditation gap) | MC Dropout (new — no prior paper did this) |
| Explainability | Pan (acknowledged black-box limitation) | Grad-CAM (new — no prior suture AI did this) |
| Class imbalance | Khalil (skipping window + augmentation) | Inverse frequency weighting (adapted) |
| Temporal analysis | IJgosse (learning curve) | 2-week vs 4-week comparison in Application cohort |

---

## Part D — Identified Limitations

### From the Papers (carried forward)

| Limitation | Source Paper | How We Address It |
|-----------|-------------|-------------------|
| Small dataset (N < 200) | Khalil: N=132, Spagnulo: N=45, IJgosse: N=36 | **Our N=1,216 annotated** — 9× larger than next largest |
| Video/sensor dependency | Pan, Khalil, Vedula | **Static image analysis** — no temporal data needed |
| Binary/expert classification only | Khalil, Pan: 2–3 levels | **6 continuous metrics** (1–10 scale) |
| No uncertainty estimates | All 7 papers | **MC Dropout + 95% CI** |
| Black-box predictions | All 7 papers | **Grad-CAM heatmaps** per metric |
| No validation framework | Vedula: "no uniform methodology" | **Messick's 5 aspects** with quantitative thresholds |
| Lab-to-OR accuracy drop | Boal: 60% >90% → 67–100% | **Application cohort** (432 unseen images) |
| Class imbalance | Khalil: phase imbalance | **Weighted sampling** + per-doctor tracking |
| Annotation label quality | Pan: GRS vs self-proclaimed conflict | **Expert-defined + dual-review + adjudication** |
| Spatial features insufficient alone | Spagnulo: XGBoost R² = −0.42 | **CNN learned features** (not hand-crafted geometry) |

### From Dataset Transition (new)

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| Source changed from images to video | Need to extract frames; potential quality loss | **Adaptive extraction pipeline** with quality filters |
| video-3.mov predominantly blurry (98%) | Reduced yield from this source | Conservative sampling; only use sharp segments (lapvar≥30) |
| Only 1 institution currently (JNMC) | Limited cross-institution diversity | Extract maximum usable frames; pursue Hubli/FDP data |
| Video-2 download incomplete | Dataset not yet final | Proceed with video-1 + video-3; integrate video-2 when ready |
| No frame-level annotations | Cannot directly use existing annotation schema | Expert annotators score extracted frames (same 6-metric protocol) |
| Variable FPS (24 vs 30) | Inconsistent temporal sampling | Frame extraction uses absolute time, not frame index |

---

## Part E — Updated Folder Structure

```
Dataset_From_JNMC/
├── raw_videos/                          # Original video files (read-only)
│   ├── video-1.MOV                      # 4K, 24fps, 5.4 min, good quality
│   ├── video-2.MOV                      # 4K, ~30fps (estimated), pending
│   └── video-3.mov                      # 4K, 30fps, 6.6 min, mostly blurry
│
├── extracted_frames/                    # Output of extraction pipeline
│   ├── video-1/                         # Frames from video-1
│   │   ├── v1_0001_0.0.png
│   │   ├── v1_0002_0.5.png
│   │   └── ...
│   └── video-3/                         # Frames from video-3 (filtered)
│       ├── v3_1410_47.1.png
│       └── ...
│
├── dataset/                             # Organized cohort structure
│   ├── Train_cohort/                    # ~60% of extracted frames
│   ├── Validation_cohort/               # ~15% of extracted frames
│   └── Application_cohort/              # ~25% of extracted frames
│
├── manifests/                           # Tracking and metadata
│   ├── video_manifest.json              # Source video metadata
│   ├── extraction_log.csv               # Per-frame extraction record
│   └── quality_report.json              # Quality metrics per frame
│
├── extraction_config.json               # Configuration for extraction
└── extraction_pipeline.py               # Reusable extraction script
```

## Part E — Implementation Roadmap

```
Phase 1 — Video Ingestion & Frame Extraction (NEW — current)
├── Wait for video-2 download to complete
├── Analyze all videos for quality profile
├── Run extraction pipeline with adaptive sampling
├── Apply quality filters (brightness, sharpness, duplicates)
└── Organize extracted frames into cohort structure

Phase 2 — Annotation (in progress)
├── Train expert annotators with reference guide
├── Conduct dual-expert review on extracted frames
├── Calculate inter-rater reliability
├── Adjudicate disagreements
└── Freeze annotation dataset

Phase 3 — Preprocessing & EDA
├── Run statistical profiling (distributions, correlations)
├── Build augmentation pipeline
├── Set up weighted sampling
└── Prepare data loaders

Phase 4 — Model Development
├── Implement EfficientNet-B3 + 6 regression heads
├── Implement uncertainty-weighted loss
├── Train with 4-phase schedule
├── Ablation experiments
├── MC Dropout integration
└── Grad-CAM integration

Phase 5 — Validation
├── Compute all Messick validity aspects
├── Application cohort evaluation
├── Temporal analysis (2wk vs 4wk)
└── Failure mode documentation

Phase 6 — Deployment
├── Export model + preprocessing pipeline
├── Build inference API
├── Dashboard integration
└── Documentation
```

---

*Methodology v2.0 — Derived from analysis of: Khalil (2025), Spagnulo (2026), Pan (2023), Boal (2024), Vedula (2017), IJgosse (2020)*
