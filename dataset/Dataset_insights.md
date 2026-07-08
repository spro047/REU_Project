# Dataset Insights: Framework for AI in Quality Control of Suturing

## Project Overview

This dataset supports research on an **AI-powered framework for evaluating suture quality** among medical trainees. The images capture suturing practice on synthetic skin pads, with expert annotations rating multiple quality dimensions.

---

## Glossary: Simple explanations for everyone

Before diving into the data, here's what the technical terms mean in plain language:

### What is Suturing?

**Suturing** (also called **stitching**) is the medical technique of closing a wound or cut in the skin using thread and a needle. Think of it like sewing, but on a patient's skin. Doctors and medical students practice this skill on artificial skin pads (pink rubber-like pads with grid lines) before doing it on real patients.

### The 6 Quality Metrics (Rated 1-10)

| Term | What it really means | Real-life example |
|------|---------------------|-------------------|
| **Overall** | "How good does this entire suture line look overall?" | Like grading an entire painting — not perfect, but pretty good |
| **ISD** (Inter-Suture Distance) | "Are the stitches evenly spaced apart?" | Like soldiers standing in a line — some are perfectly spaced, others are too close or too far apart |
| **Slack** | "Is the thread tight enough, or is it loose and floppy?" | Like a guitar string — too loose sounds bad, too tight breaks. Just right is perfect |
| **Position** | "Did the needle go in and come out at the right spot?" | Like parking a car — you want it centered in the spot, not crooked or too far to one side |
| **Angulation** | "Is the needle going in at the right angle?" | Like throwing a ball — the angle matters. Too steep or too flat won't work well |
| **Width** | "Are all the stitches the same width/size?" | Like handwriting — consistent letters look professional, messy sizes look sloppy |

### Other Terms You'll See

| Term | Simple Explanation |
|------|-------------------|
| **Cohort** | A group of images grouped together for a specific purpose (like "Class A" vs "Class B") |
| **Training Set** | The practice exam — images used to teach the AI what good vs bad sutures look like |
| **Validation Set** | The quiz — images used to check if the AI learned correctly |
| **Application Set** | The final exam — real-world images to test the AI in actual use |
| **Annotation** | The expert's grade/score for each image |
| **Correlation** | How closely two things are related. High correlation = they usually go together |
| **Outlier** | An unusual score that's very different from most others (like a student who scores 2/10 when everyone else gets 6-8) |
| **Standard Deviation (SD)** | How spread out the scores are. Small SD = everyone scored similarly. Large SD = big range of scores |
| **Skewness** | Whether scores lean toward high or low. "Left-skewed" means most scores are on the higher end |

### What the Numbers Mean

- **Score 1-3**: Needs significant improvement (poor quality)
- **Score 4-6**: Getting there but has issues (moderate quality)  
- **Score 7-9**: Good quality with minor issues
- **Score 10**: Excellent — nearly perfect

### Why This Matters

This research helps create an **AI assistant** that can automatically evaluate suture quality, giving medical students instant feedback on their practice. Instead of waiting for a teacher to check their work, students can get immediate, consistent grading — like having a personal coach available 24/7.

---

## 1. Dataset Structure

### 1.1 Three-Cohort Design

| Cohort | Purpose | Images | Annotated |
|--------|---------|--------|-----------|
| **Train** | Model training | 1,010 | Yes |
| **Validation** | Hyperparameter tuning | 206 | Yes |
| **Application** | Real-world deployment testing | 432 | No (structure only) |
| **Total** | - | **1,648** | **1,216** |

### 1.2 Image Content

- **Subject**: Suture lines on pink synthetic skin pads with grid markings
- **Format**: PNG images
- **Resolution**: Standard practice pad photography
- **View**: Top-down perspective showing full suture line

---

## 2. Naming Convention & Metadata

### 2.1 Train/Validation Format
```
Image_XXXX_XX_X_X_XX.png
  │      │   │ │ │ └── Parameter 4 (likely suture count or complexity)
  │      │   │ │ └──── Parameter 3 (experience level: 0=beginner, 1=intermediate, 2=advanced)
  │      │   │ └────── Parameter 2 (always 0 - possibly unused)
  │      │   └──────── Doctor/Subject ID (4, 5, 7, 8, 10)
  │      └──────────── Sequential ID
  └─────────────────── Prefix
```

### 2.2 Application Cohort Format
```
XXXX_Doc_XX_Itr_XX.png
  │      │    └── Iteration number (01-16)
  │      └─────── Doctor number (01-10)
  └────────────── Sequential ID
```

### 2.3 Application Cohort Time Points

| Timepoint | Doctors | Images per Doctor | Total |
|-----------|---------|-------------------|-------|
| 2-week | 10 | 16 | 160 |
| 4-week | 10 | 16 | 160 |
| Resident | 7 | 16 | 112 |

---

## 3. Annotation Schema

Each image receives 6 quality ratings on a **1-10 scale**:

| Metric | Description | Range | Training Mean (±SD) |
|--------|-------------|-------|---------------------|
| **Overall** | Global quality assessment | 1-10 | 5.44 ± 1.57 |
| **ISD** | Inter-Suture Distance consistency | 2-9 | 6.05 ± 1.15 |
| **Slack** | Thread tension/slack control | 0-10 | 6.29 ± 1.64 |
| **Position** | Suture placement accuracy | 1-9 | 5.78 ± 1.48 |
| **Angulation** | Needle entry/exit angles | 1-9 | 6.34 ± 1.31 |
| **Width** | Suture width uniformity | 3-9 | 7.09 ± 1.07 |

---

## 4. Key Statistical Insights

### 4.1 Score Distributions

```
Overall:   ████████████████████░░░░ 5.44 (Moderate - bell curve centered at 5-6)
ISD:       ████████████████████░░░░ 6.05 (Slightly above average)
Slack:     █████████████████████░░░ 6.29 (Above average, left-skewed)
Position:  ███████████████████░░░░░ 5.78 (Moderate, left-skewed)
Angulation:████████████████████░░░░ 6.34 (Above average)
Width:     ██████████████████████░░ 7.09 (Best performing metric)
```

### 4.2 Performance Ranking by Metric

1. **Width** (7.09) - Trainees perform best on maintaining uniform width
2. **Angulation** (6.34) - Good angle control overall
3. **Slack** (6.29) - Moderate tension control
4. **ISD** (6.05) - Spacing consistency is challenging
5. **Position** (5.78) - Placement accuracy needs improvement
6. **Overall** (5.44) - Global quality indicates room for growth

### 4.3 Variability Analysis

| Metric | Coefficient of Variation | Interpretation |
|--------|--------------------------|----------------|
| Width | 0.151 | Low - Consistent performance |
| ISD | 0.190 | Low - Stable metric |
| Angulation | 0.206 | Moderate |
| Position | 0.256 | Moderate |
| Slack | 0.261 | Moderate |
| Overall | 0.289 | Moderate - Most variable |

### 4.4 Score Distribution Shape

| Metric | Skewness | Interpretation |
|--------|----------|----------------|
| Overall | -0.186 | Near-symmetric |
| ISD | -0.227 | Near-symmetric |
| Slack | -0.479 | Left-skewed (more high scores) |
| Position | -1.010 | Strongly left-skewed |
| Angulation | -0.637 | Left-skewed |
| Width | -0.613 | Left-skewed |

**Key Finding**: Most metrics are left-skewed, indicating trainees tend to score in the moderate-to-good range, with fewer poor performances.

---

## 5. Correlation Analysis

### 5.1 Strongest Correlations

| Metric Pair | Correlation | Interpretation |
|-------------|-------------|----------------|
| **Overall ↔ Slack** | 0.722 | Tension control strongly predicts overall quality |
| **Overall ↔ Width** | 0.617 | Width uniformity is key to perceived quality |
| **Overall ↔ Position** | 0.546 | Placement accuracy matters significantly |
| **Slack ↔ Position** | 0.523 | These skills co-develop |
| **Angulation ↔ Width** | 0.500 | Technical precision cluster |

### 5.2 Weakest Correlations

| Metric Pair | Correlation | Interpretation |
|-------------|-------------|----------------|
| **ISD ↔ Slack** | 0.026 | Nearly independent skills |
| **ISD ↔ Position** | 0.154 | Spacing vs placement are distinct |
| **Overall ↔ ISD** | 0.162 | Spacing contributes least to overall score |

### 5.3 Implications for AI Model

- **Primary predictors of overall quality**: Slack, Width, Position
- **Independent skill dimension**: ISD (spacing) appears to be a separate competency
- **Feature engineering opportunity**: Slack + Width + Position could form a "core quality" composite

---

## 6. Outlier Analysis

| Metric | Outliers (>2 SD from mean) | Percentage |
|--------|---------------------------|------------|
| Slack | 72 | 7.1% |
| Overall | 63 | 6.2% |
| Position | 49 | 4.9% |
| Angulation | 37 | 3.7% |
| Width | 21 | 2.1% |
| ISD | 16 | 1.6% |

**Insight**: Slack and Overall have the most outliers, suggesting these are the most variable skills and may benefit from targeted feedback.

---

## 7. Subject/Doctor Distribution

### 7.1 Training Set

| Doctor ID | Images | Percentage |
|-----------|--------|------------|
| 10 | 371 | 36.7% |
| 8 | 221 | 21.9% |
| 4 | 214 | 21.2% |
| 5 | 197 | 19.5% |
| 7 | 7 | 0.7% |

**Class Imbalance**: Doctor 7 has significantly fewer samples (0.7%), which may affect model generalization.

### 7.2 Validation Set

| Doctor ID | Images | Percentage |
|-----------|--------|------------|
| 10 | 41 | 19.9% |
| 4 | 55 | 26.7% |
| 5 | 53 | 25.7% |
| 8 | 51 | 24.8% |
| 7 | 6 | 2.9% |

---

## 8. Experience Level Distribution

| Level | Code | Train Count | Val Count |
|-------|------|-------------|-----------|
| Beginner | 0 | 309 (30.6%) | 79 (38.3%) |
| Intermediate | 1 | 504 (49.9%) | 99 (48.1%) |
| Advanced | 2 | 197 (19.5%) | 23 (11.2%) |
| Other | 3-6 | 0 | 5 (2.4%) |

**Note**: The experience level coding suggests a progression model. The validation set has more beginner samples proportionally.

---

## 9. Critical Insights for AI Framework

### 9.1 Model Design Recommendations

1. **Multi-task Learning**: Predict all 6 metrics simultaneously, leveraging inter-metric correlations
2. **Attention to Slack**: Since Slack has highest correlation with Overall (0.722), prioritize this feature
3. **Composite Scoring**: Consider a weighted composite of Slack + Width + Position as primary quality indicator
4. **Handle ISD Separately**: Since ISD is nearly independent, consider a separate sub-model or head

### 9.2 Data Augmentation Opportunities

- The 2-week vs 4-week temporal data in Application cohort enables **learning curve modeling**
- Doctor-level variation enables **personalized feedback systems**
- The 16 iterations per doctor enables **intra-subject variability analysis**

### 9.3 Potential Challenges

1. **Class Imbalance**: Doctor 7 severely underrepresented
2. **Score Ceiling Effect**: Width scores cluster at 7-8, limiting discriminative power
3. **Annotator Variability**: Left-skewed distributions may indicate lenient grading
4. **Cross-cohort Generalization**: Need to validate if Train/Val patterns hold for Application cohort

---

## 10. Summary Statistics

| Statistic | Train | Validation |
|-----------|-------|------------|
| Total Images | 1,010 | 206 |
| Unique Doctors | 5 | 5 |
| Mean Overall | 5.44 | 5.36 |
| Std Overall | 1.57 | 1.48 |
| Best Metric | Width (7.09) | Angulation (6.64) |
| Worst Metric | Overall (5.44) | Overall (5.36) |
| Most Correlated Pair | Overall-Slack (0.722) | - |

---

*Analysis Date: July 2026*
*Dataset Location: D:\REU\dataset*
