# Specification: Suture Quality Video ML Pipeline

## Problem Statement

The project has fixed-camera surgical-training videos but no reproducible frame dataset, expert-reviewed annotation set, or validated computer-vision workflow for measuring suturing technique. The desired research output is an offline video in which each completed stitch receives understandable feedback on suture quality.

The current collection contains nine videos and should be treated as a pilot feasibility dataset. The project must support research-paper reporting without presenting predictions as clinical guidance or accreditation.

## Solution

Build a reproducible, staged machine-learning pipeline that audits the videos, extracts a frame dataset, uses local/self-hosted CVAT for expert and AI-assisted annotation, freezes a grouped dataset, trains multi-task vision models, and renders an annotated output video.

The output will show a predicted Suture quality score from 1-10 after stitch completion, the six component scores, model confidence, stitch ID, and timestamp. Research evaluation will separately report agreement with expert reference scores and computer-vision accuracy metrics.

## User Stories

1. As a researcher, I want to inventory every source video, so that I know which recordings are usable.
2. As a researcher, I want to record resolution, frame rate, duration, orientation, and integrity, so that preprocessing is reproducible.
3. As a researcher, I want to record operator, viewpoint, lighting, visibility, blur, occlusion, and stitch information, so that dataset limitations are explicit.
4. As a researcher, I want inclusion and exclusion rules before annotation, so that unusable segments are not silently included.
5. As a researcher, I want to extract frames at 5 FPS, so that the dataset has a consistent baseline sample.
6. As a researcher, I want additional frames around needle entry, needle exit, knot formation, and stitch completion, so that important motion is represented.
7. As a researcher, I want stable frame IDs linked to source timestamps, so that every annotation can be traced back to its video.
8. As a researcher, I want duplicate, blurry, occluded, and invalid frames recorded separately, so that filtering decisions can be audited.
9. As an annotator, I want a versioned CVAT project, so that labels and task configuration can be reproduced.
10. As an annotator, I want to mark the wound or pad boundary, so that measurements are relative to the relevant region.
11. As an annotator, I want to mark the needle, so that the system can learn needle localization.
12. As an annotator, I want to mark the visible thread path, so that thread slack can be studied.
13. As an annotator, I want to mark needle entry and exit points, so that position and angulation can be measured.
14. As an annotator, I want to identify stitches and knots, so that each completed stitch has a stable identity.
15. As an annotator, I want to label needle-entry, needle-exit, knot-formation, and stitch-completion intervals, so that temporal events can be learned.
16. As a clinical expert, I want to rate Overall quality, so that the model has a global quality target.
17. As a clinical expert, I want to rate ISD, so that spacing consistency can be evaluated.
18. As a clinical expert, I want to rate Slack, so that thread tension control can be evaluated.
19. As a clinical expert, I want to rate Position, so that entry and exit placement can be evaluated.
20. As a clinical expert, I want to rate Angulation, so that needle entry and exit angle can be evaluated.
21. As a clinical expert, I want to rate Width, so that stitch size consistency can be evaluated.
22. As an annotator, I want to attach scores to representative key frames and stitch intervals, so that labels are detailed without duplicating every near-identical frame.
23. As a researcher, I want a primary expert to label the full set and a second expert to review a subset, so that inter-rater agreement can be measured.
24. As a researcher, I want unresolved annotation disagreements identified, so that uncertain labels are not treated as ground truth.
25. As a researcher, I want AI-assisted pre-annotation only after a manually reviewed seed set exists, so that automated errors do not become unchecked training labels.
26. As an annotator, I want to correct every AI-generated annotation used for training, so that the training set remains human-reviewed.
27. As a researcher, I want measurements represented in normalized image coordinates, so that the pilot does not falsely imply physical calibration.
28. As a researcher, I want operator- and video-level dataset splitting, so that adjacent frames from one recording cannot leak into evaluation.
29. As a researcher, I want a frozen dataset manifest, so that training and evaluation can be repeated.
30. As a model developer, I want a key-frame geometry baseline, so that the first model slice is small and verifiable.
31. As a model developer, I want stitch tracking, so that observations can be associated with one stitch.
32. As a model developer, I want stitch-completion recognition, so that scores are finalized at a meaningful time.
33. As a model developer, I want regression heads for all six metrics, so that the model provides detailed feedback.
34. As a model developer, I want a learned Overall head, so that Overall remains an expert-rated target rather than an unexplained formula.
35. As a researcher, I want image-only and staged temporal baselines, so that the proposed pipeline can be compared fairly.
36. As a researcher, I want detection, event, geometry, regression, calibration, and runtime metrics, so that success is not reduced to one misleading accuracy number.
37. As a user watching the output video, I want feedback after each completed stitch, so that the timing of the score is understandable.
38. As a user watching the output video, I want Suture quality score and Model confidence shown separately, so that confidence is not confused with accuracy.
39. As a researcher, I want expert reference score and prediction error available during evaluation, so that the output can be compared with ground truth.
40. As a researcher, I want a machine-readable prediction sidecar, so that results can be analyzed independently of the rendered video.
41. As a paper author, I want extraction, annotation, split, model, baseline, and limitation documentation, so that the methods are reproducible.
42. As a paper author, I want the project described as a pilot feasibility study, so that conclusions match the small fixed-camera dataset.

## Implementation Decisions

- The primary system boundary is the dataset pipeline: source videos plus extraction and annotation configuration produce a reproducible frame manifest, validated CVAT export, grouped split, and audit report.
- The first milestone is dataset construction and annotation. Model training is blocked until the dataset is frozen and annotation quality is reported.
- Input is offline fixed-camera video. Live camera capture is not part of this specification.
- Frame extraction uses a 5 FPS baseline plus 2-second event-centered windows.
- Key frames receive detailed geometry and quality annotations; event intervals cover temporal events.
- CVAT is local/self-hosted. CVAT tracks and interpolation may be used for temporal objects, followed by human review.
- The annotation schema includes wound/pad polygon, needle shape, thread polyline, entry and exit keypoints, stitch and knot identity, four temporal events, six 1-10 quality attributes, timestamps, operator/video IDs, annotator ID, visibility, and review status.
- Human annotation is the ground truth. The workflow is manual seed annotation, expert review, AI-assisted pre-annotation, expert correction, and annotation freeze.
- Quality labels are expert ratings plus measured geometry where possible. The rubric must define anchors and handling for occlusion or unjudgeable attributes.
- Geometry uses normalized coordinates and degrees for angles unless a future scale marker enables physical calibration.
- The model is a staged multi-task pipeline: object/geometry estimation, tracking, temporal event recognition, and quality regression.
- Regression outputs include ISD, Slack, Position, Angulation, Width, and a learned Overall head.
- The overlay calls the 1-10 prediction Suture quality score. Model confidence and research accuracy are separate concepts.
- A stitch is scored after stitch completion/knot formation, then stitch results may be aggregated into a suture-line summary.
- Training is GPU-first with CPU benchmarking reported separately.
- Data augmentation must preserve the meaning of entry/exit and handedness.
- Evaluation splits are by video and by operator when operator metadata exists. Random frame-level splitting is prohibited.
- Baselines include a simple image-only model and the proposed staged temporal pipeline, with ablations for event context and geometry inputs.
- The output renderer preserves source timing and writes a sidecar containing stitch ID, timestamp, events, scores, confidence, and model version.
- The system is a pilot research and educational prototype, not a clinical, certification, or accreditation tool.

## Testing Decisions

- Tests should verify external behavior at the dataset-pipeline boundary rather than internal implementation details.
- A reproducibility test should run extraction twice with the same configuration and confirm stable frame IDs, timestamps, counts, and manifest values.
- A validation test should reject missing source references, invalid timestamps, malformed CVAT shapes, unknown labels, invalid score ranges, and unordered event intervals.
- A split test should prove that no source video or operator appears in more than one grouped split.
- A schema test should prove that a valid CVAT export becomes the expected frame, stitch, event, geometry, and quality records.
- A scoring test should verify that a stitch is not finalized before stitch completion and that all six component scores remain distinct from confidence.
- A rendering test should verify that the output video preserves frame order/timing and that the sidecar contains one traceable record per prediction.
- Model evaluation tests should verify metric calculations on small known fixtures for IoU, keypoint error, temporal IoU, F1, MAE, RMSE, rank correlation, calibration, and FPS reporting.
- A smoke test should process one short fixture video end to end and produce a manifest, annotation validation result, prediction sidecar, and rendered video.
- The repository currently has no established ML test suite, so new tests should be introduced at the pipeline boundary with small synthetic fixtures before real video runs.

## Out of Scope

- Live camera input or real-time surgical guidance.
- Clinical diagnosis, treatment recommendations, accreditation, or autonomous assessment.
- Claims of generalization beyond the pilot dataset.
- Physical millimeter measurements without camera calibration or a scale marker.
- Fully automatic annotation without expert review.
- Random frame-level splitting across train, validation, and test sets.
- Treating model confidence as prediction accuracy.
- Replacing expert quality ratings with a formula that has not been clinically approved.
- Adding unrelated frontend application features.

## Further Notes

- The existing `Docs/index.html` is a communication page and contains conceptual estimates and literature summaries. Current video counts and dataset statistics must be measured from the source files.
- The existing six metric names are Overall, ISD, Slack, Position, Angulation, and Width.
- The first implementation ticket should audit the nine videos and create the video manifest.
- The project should preserve excluded-frame counts, annotation versions, configuration versions, random seeds, model checkpoints, and hardware details.
- Research reporting should include limitations involving sample size, operator diversity, fixed viewpoint, occlusion, annotation subjectivity, calibration, and lack of clinical validation.
