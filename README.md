# AI-Powered Suture Quality Assessment Framework

**Dataset Insights & Literature Survey Dashboard** — an interactive analysis of suture quality metrics across 1,648 medical training images, with a comprehensive review of 7 research papers (2017–2026).

---

## 📂 Repository Structure

```
REU/
├── dataset/
│   ├── index.html              # Interactive dashboard (brutalist cream+black design)
│   ├── Dataset_insights.md     # Dataset analysis with plain-language glossary
│   ├── Train_cohort/           # Training images (excluded from git)
│   ├── Validation_cohort/      # Validation images (excluded from git)
│   └── Application_cohort/     # Application images (excluded from git)
├── Research_paper.md           # Literature survey + methodology framework
├── main_Doc.docx               # Project proposal document
└── README.md                   # This file
```

## 🔍 What This Project Does

1. **Analyzes** a dataset of 1,648 suture images rated by experts across 6 quality metrics (Overall, ISD, Slack, Position, Angulation, Width)
2. **Visualizes** distributions, correlations, and variability through a Chart.js dashboard (brutalist aesthetic, cream + black palette)
3. **Surveys** 7 research papers on AI-based surgical skill assessment, identifying gaps and limitations
4. **Proposes** **Suture-Q AI** — a multi-source, interpretable, Messick-validated framework addressing all identified limitations

## 📊 Dashboard Features

- 3 cohort breakdowns (Train / Validation / Application)
- 6 interactive charts (bar, scatter, radar, donut, horizontal bar)
- Animated flowchart of the proposed methodology (anime.js)
- Scroll-reveal animations on all chart sections
- Literature Survey with collapsible paper comparison table

## 📚 Papers Reviewed

| # | Author(s) | Year | Focus |
|---|-----------|------|-------|
| 1 | Khalil et al. | 2025 | Microsurgical suture phase recognition (LRCN, 83.85% accuracy) |
| 2 | Spagnulo et al. | 2026 | Computer vision suture spacing vs. mechanical strength |
| 3 | Pan et al. | 2023 | RAMIS skill assessment (ResNet, 92.04% accuracy) |
| 4 | Kankanamge et al. | 2025 | Systematic review: AI + objective metrics integration |
| 5 | Boal et al. | 2024 | Systematic review: 247 robotic surgery assessment studies |
| 6 | Vedula et al. | 2017 | OCASE-T framework for OR skill assessment |
| 7 | IJgosse et al. | 2020 | LS-CAT competency tool for laparoscopic suturing |

## 🧪 Proposed Framework: Suture-Q AI

A 5-stage pipeline addressing all gaps identified in the literature:
1. **Multi-Source Data Acquisition** — RGB + depth + motion + force sensors
2. **Multi-Granularity Feature Extraction** — stitch-level + sequence-level features
3. **Ensemble Classification** — hybrid model for >2 skill levels
4. **Interpretable Output** — SHAP-based metric attribution
5. **Messick Validation Suite** — 5-source validity framework

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Chart.js, anime.js)
- **Fonts**: Inter, Space Mono
- **Analysis**: Python (NumPy, statistics)
- **Design**: Brutalist aesthetic, light cream background, zero-radius sharp edges

## 🚀 Getting Started

Open `dataset/index.html` in any modern browser — no build step required. All dependencies loaded from CDN:

```html
<!-- Chart.js for visualizations -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<!-- anime.js for scroll-reveal and flowchart animations -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js"></script>
```

## 📝 Note on Dataset Images

The raw suture images (Train: ~498 MB, Validation: ~74 MB, Application: ~213 MB) are not included in this repository due to size constraints. Contact the dataset authors or refer to the project proposal (`main_Doc.docx`) for data access.
