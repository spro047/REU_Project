# Geometry Baseline Report

Model: RandomForest pixel classifier (HSV, 96x96)
Train frames sampled: 60 (capped at 800000 pixels)
Test frames evaluated: 40
Pixel accuracy: 0.5058

| Class | IoU |
| --- | ---: |
| background | 0.5459 |
| wound_pad | 0.0490 |
| needle | 0.0097 |
| thread | 0.0002 |

## Limitations
- Labels are auto-generated and unreviewed (ADR-0005); metrics measure agreement with those labels, not clinical truth.
- Baseline is the comparison point for the deep model unit.
