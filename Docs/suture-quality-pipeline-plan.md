# Suture Quality Video ML Pipeline Plan

## 1. Objective and scope

Build a reproducible pilot pipeline that accepts fixed-camera surgical-training videos, extracts a frame dataset, uses local CVAT annotations, trains a staged multi-task computer-vision system, and renders an output video with stitch-level feedback after stitch completion.

The system is for research and educational feedback. It must not be presented as clinical guidance, certification, or accreditation. The current nine videos are a feasibility dataset; broader claims require additional operators, procedures, and external validation.

## 2. Confirmed output

For each completed stitch, the rendered video should show:

- wound/pad region and relevant object overlays;
- stitch phase/event status;
- predicted Overall, ISD, Slack, Position, Angulation, and Width scores on a 1-10 scale;
- a clearly separate model-confidence value;
- a stitch identifier and timestamp.

At the end, report stitch-level results aggregated into a suture-line summary. During evaluation only, optionally show expert reference scores and prediction error. Do not label the predicted 1-10 value "accuracy".

## 3. Phase 0: repository and video audit

1. Inventory all nine videos, codecs, resolution, frame rate, duration, orientation, and file integrity.
2. Record video ID, doctor/operator ID if available, viewpoint, lighting, wound/pad visibility, occlusions, blur, active-tool count, and approximate stitch count.
3. Define inclusion and exclusion rules before annotation. Initial scope is fixed or near-fixed camera, visible wound/pad, and one active stitch at a time.
4. Produce `data/video_manifest.csv` and an audit report. Do not use the old `index.html` estimates as measured ground truth; verify current counts from the files.

Acceptance criteria: every source video has an audit row, unusable segments are timestamped with reasons, and operator metadata is either recorded or explicitly marked unknown.

## 4. Phase 1: frame dataset construction

1. Decode videos without re-encoding the source.
2. Extract a baseline sample at 5 FPS.
3. Preserve additional 2-second event-centered windows around needle entry, needle exit, knot formation, and stitch completion once intervals are identified.
4. Apply quality filtering for unreadable blur, severe occlusion, duplicate frames, and invalid decoding, while retaining excluded-frame counts.
5. Store frames with stable IDs derived from video ID, timestamp, and frame index. Keep source video and timestamp in a manifest.
6. Use a lossless or minimally compressed working format and record preprocessing, resize, normalization, and augmentation parameters.

Acceptance criteria: the dataset can be regenerated from the manifest and extraction configuration; no frame exists without a source-video/frame reference; duplicate and excluded-frame statistics are reported.

## 5. Phase 2: CVAT annotation workflow

Create a local/self-hosted CVAT project with versioned labels and an annotation guide approved by the clinical expert.

### Annotation targets

- `wound_pad`: polygon on key frames, propagated/tracked between reviewed keyframes;
- `needle`: box or polygon track;
- `thread`: polyline track where visible;
- `entry_point`: point/keypoint;
- `exit_point`: point/keypoint;
- `stitch`: stitch ID and location;
- `knot`: box or keypoints where visible;
- temporal events: `needle_entry`, `needle_exit`, `knot_formation`, `stitch_completion`;
- quality attributes on the stitch: `overall`, `isd`, `slack`, `position`, `angulation`, `width`, each 1-10;
- metadata: video ID, timestamp, stitch ID, annotator ID, visibility/occlusion, and review status.

Use CVAT tracks and interpolation for temporal objects where appropriate. Use event intervals plus a representative key frame. Dense geometry and quality scoring are required on key frames, not blindly on every near-duplicate frame.

### Human and AI-assisted passes

1. A primary expert manually labels a seed set.
2. A second expert reviews a defined subset for agreement.
3. Train or connect an initial detector/segmenter only after seed review.
4. Generate AI-assisted pre-annotations in CVAT.
5. The primary expert corrects every generated annotation used for training.
6. Freeze an annotation version and export it with the CVAT task configuration and schema.

Acceptance criteria: all training annotations have human review status, every stitch has a stable ID, event intervals are valid and ordered, and the annotation export can be parsed into the project dataset format.

## 6. Phase 3: scoring rubric and label quality

Formalize the `index.html` definitions into an expert-approved rubric:

- Overall: global suture-line or stitch quality;
- ISD: spacing consistency;
- Slack: thread tension control;
- Position: entry/exit placement;
- Angulation: needle entry/exit angle;
- Width: stitch size consistency.

Record operational 1-10 anchors, visibility rules, and what to do when a metric cannot be judged. Keep measured geometry separate from expert ratings. Use normalized image coordinates and degrees for angles unless a scale marker enables physical calibration.

Use one primary expert for the full set and a second expert on a subset. Report agreement and adjudication rules before training. Do not train on unresolved disagreements.

Acceptance criteria: the rubric contains reproducible anchors, missing/occluded values have a defined status, and agreement statistics are available for the reviewed subset.

## 7. Phase 4: dataset freeze and split

Create a manifest linking each frame, key frame, event interval, stitch ID, expert scores, geometry, source video, operator, and annotation version. Split by video, and by operator when metadata exists. Never randomly split adjacent frames from one recording across train and test.

The split must be decided before model tuning. Report counts by video, operator, stitch, event, and score distribution. Because the collection is small, use pilot language and avoid claiming clinical generalization.

Acceptance criteria: a clean frozen dataset version exists; the split is reproducible from the manifest; no source video or operator leaks across evaluation groups.

## 8. Phase 5: model development

Implement the smallest verifiable vertical slice first:

1. key-frame wound/needle/stitch geometry detection;
2. stitch tracking or stable stitch association;
3. one event recognizer for stitch completion;
4. one quality regression baseline;
5. rendered overlay on a held-out video.

Expand to the full staged multi-task system only after this slice works:

- object/segmentation and keypoint model;
- tracking across the fixed-camera sequence;
- temporal event recognition for entry, exit, knot formation, and completion;
- regression heads for ISD, Slack, Position, Angulation, Width, and learned Overall;
- confidence estimation and uncertainty calibration;
- optional explainability heatmaps after core measurements are validated.

Use GPU-first training/inference and benchmark CPU separately. Keep augmentations physically plausible: limited rotation, crop, brightness, and horizontal flip only where they do not change the meaning of entry/exit or handedness. Record all seeds, versions, checkpoints, and configurations.

## 9. Phase 6: evaluation

Evaluate on held-out videos/operators using task-specific metrics:

- detection/segmentation: precision, recall, IoU, and mAP where applicable;
- keypoints/geometry: normalized coordinate error and angle error;
- events: class precision/recall/F1 and temporal IoU;
- scores: MAE, RMSE, rank correlation, and expert agreement such as ICC where appropriate;
- confidence: calibration error and coverage of uncertainty intervals;
- runtime: decoded/processed FPS, latency, hardware, and resolution.

Compare at least a simple image-only baseline with the proposed staged temporal pipeline. Add ablations for event context, geometry inputs, and learned Overall versus component-only inputs. Report confidence intervals or bootstrap intervals where sample size permits.

## 10. Phase 7: output-video renderer

For each input video, run inference in timestamp order. Maintain the current stitch state. Do not finalize a stitch score until the stitch-completion event is detected. Render the score on the wound/pad region with a visible stitch ID and timestamp. Preserve the original frame timing and write a machine-readable sidecar containing every prediction, confidence, event, and model version.

The final demonstration must include one held-out video and one failure/uncertainty example. A score must never be shown without indicating whether it is a prediction, expert reference, or evaluation error.

## 11. Research-paper deliverables

The methods package should include:

- video inventory and technical audit;
- extraction and quality-filtering configuration;
- CVAT label schema and annotation guide;
- hybrid annotation and reviewer workflow;
- inter-rater agreement procedure;
- frozen dataset manifest and split policy;
- model architecture and training configuration;
- baseline and ablation definitions;
- evaluation metrics and held-out results;
- output-video examples;
- limitations: small pilot sample, fixed viewpoint, annotation subjectivity, occlusion, lack of physical calibration, and no clinical validation.

## 12. Suggested repository layout

```text
data/
  raw/                 # source videos, unchanged
  frames/              # generated frame dataset, normally ignored by git
  annotations/         # versioned CVAT exports and schema
  manifests/           # video and frame manifests
configs/               # extraction, labels, training, and evaluation configs
src/
  extraction/          # video audit and frame extraction
  annotation/          # CVAT import/export and validation
  geometry/             # normalized measurements
  events/               # temporal event handling
  models/               # detection, temporal, and regression models
  inference/            # ordered video inference and stitch state
  rendering/            # overlay video and sidecar output
  evaluation/           # metrics, splits, and reports
tests/
reports/
  dataset-audit.md
  annotation-quality.md
  evaluation.md
```

## 13. Definition of done for the first milestone

The first milestone is complete when:

1. all nine videos are audited;
2. a reproducible 5 FPS plus event-window extraction run exists;
3. extracted frames have stable manifest IDs;
4. the CVAT project and annotation guide are versioned;
5. seed annotations and reviewed AI-assisted annotations exist;
6. each annotated stitch has geometry, event, and quality-label status;
7. expert review agreement is reported;
8. the frozen pilot dataset and grouped split are documented;
9. known limitations are written down.

Model training begins only after these conditions pass review.
