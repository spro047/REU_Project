# Deep Geometry Baseline Report

Model: TinyUNet (473,316 params), CPU, 96x96
Config: clahe=False sce=True copy_paste=True
Train frames: 360 (capped 400), val: 40, test: 60
Best val mean IoU: 0.2573
Pixel accuracy (test): 0.4189

| Class | IoU |
| --- | ---: |
| background | 0.4573 |
| wound_pad | 0.0525 |
| needle | 0.0265 |
| thread | 0.0276 |

## vs RandomForest baseline (reports/geometry-baseline.md)
- RF: accuracy 0.506, IoU background 0.546 / pad 0.049 / needle 0.010 / thread 0.000

## Accuracy upgrades and seed variance (unit 3)

Implemented from `Docs/research/accuracy-improvements.md`: Symmetric Cross Entropy loss (alpha=1.0, beta=0.1, class-weighted CE term), Copy-Paste augmentation (p=0.5), optional CLAHE preprocessing. All three are configurable via `--no-sce`, `--no-copy-paste`, `--clahe`.

Seeded comparison (torch.manual_seed(0), 360 train frames, 12 epochs, 60 test frames):

| Config | accuracy | pad IoU | needle IoU | thread IoU |
| --- | ---: | ---: | ---: | ---: |
| SCE + Copy-Paste (default) | 0.419 | 0.052 | 0.026 | 0.028 |
| SCE + Copy-Paste + CLAHE | 0.541 | 0.069 | 0.001 | 0.026 |

**Honest finding:** before seeding, single runs showed pad IoU up to 0.185 and needle up to 0.079, but those were seed luck. With a fixed seed the upgrades are within noise of the original weighted-CE model (pad 0.069 / needle 0.037 / thread 0.026), and the two configs trade classes against each other. At 360 training frames and 60 test frames, run-to-run variance exceeds the effect of any single upgrade.

## Limitations
- Labels are auto-generated and unreviewed (ADR-0005); metrics measure agreement with those labels.
- CPU-only training (GPU driver too old for CUDA torch); expect gains to grow with resolution and data.
- Single-seed results; 60 test frames. Reliable comparisons need multiple seeds (mean +/- std) and more training data - that is the recommended next unit.
