# Deep Geometry Baseline Report

Model: TinyUNet (473,316 params), CPU, 96x96
Train frames: 360 (capped 400), val: 40, test: 60
Best val mean IoU: 0.2466
Pixel accuracy (test): 0.3929

| Class | IoU |
| --- | ---: |
| background | 0.4304 |
| wound_pad | 0.0685 |
| needle | 0.0373 |
| thread | 0.0256 |

## vs RandomForest baseline (reports/geometry-baseline.md)
- RF: accuracy 0.506, IoU background 0.546 / pad 0.049 / needle 0.010 / thread 0.000

## Limitations
- Labels are auto-generated and unreviewed (ADR-0005); metrics measure agreement with those labels.
- CPU-only training (GPU driver too old for CUDA torch); expect gains to grow with resolution and data.
