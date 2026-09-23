# Deep Geometry Baseline Report

Model: TinyUNet (473,316 params), CPU, 96x96
Config: clahe=False sce=True copy_paste=True
Train frames: 1938, val: 216, test: 300, seeds: 1, epochs: 12

| Metric | mean +/- std |
| --- | ---: |
| accuracy | 0.5386 +/- 0.0000 |
| background | 0.5072 +/- 0.0000 |
| wound_pad | 0.1903 +/- 0.0000 |
| needle | 0.1457 +/- 0.0000 |
| thread | 0.0653 +/- 0.0000 |

## vs RandomForest baseline (reports/geometry-baseline.md)
- RF: accuracy 0.506, IoU background 0.546 / pad 0.049 / needle 0.010 / thread 0.000

## Limitations
- Labels are auto-generated and unreviewed (ADR-0005); metrics measure agreement with those labels.
- CPU-only training (GPU driver too old for CUDA torch); expect gains to grow with resolution and data.
