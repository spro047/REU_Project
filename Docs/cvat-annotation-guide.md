# CVAT Annotation Guide

Pilot annotation guide for the suture-quality dataset. This is a **draft for expert approval**: the scoring rubric and the visibility rules below must be reviewed and approved by the clinical expert before seed annotation starts.

## Purpose and scope

The dataset is a pilot feasibility collection of frames extracted from fixed-camera surgical-training videos. Annotations support two goals:

1. **Temporal events** (needle entry, needle exit, knot formation, stitch completion) used to learn when each stitch is finished.
2. **Geometry and quality** (wound, needle, thread, entry/exit points, six 1-10 quality scores per completed Stitch) used to learn placement and technique.

Labels are defined in `configs/cvat_labels.json` and validated by `scripts/validate_cvat_schema.py`.

## Setup

1. Start the local CVAT server (see deployment notes).
2. Create a project, then import `configs/cvat_labels.json` as the label source.
3. Create one task per video from the extracted frames in `data/frames/<video_id>/`. Frames are already 384x384 PNGs named `frame_000001.png` and up, in timestamp order.
4. Work in **video mode** so tracks and interpolation are available.

## Key frames vs event intervals

- **Key frame**: a representative frame where geometry and quality scores are annotated in full. Choose the frame where a completed stitch is most clearly visible.
- **Event interval**: a temporal span with a start and end frame, annotated as a rectangle **track** spanning the interval.

Rule: annotate geometry and the six quality scores only on key frames. Annotate the four event intervals as tracks across their full span. Dense annotation of every near-identical frame is not required.

## Label instructions

| Label | Shape | Rule |
| --- | --- | --- |
| `wound_pad` | polygon | Draw the wound/artificial skin pad boundary on each key frame. Use track interpolation between key frames, then review every key frame. |
| `needle` | polygon | Track the needle while visible. |
| `thread` | polyline | Trace the visible thread path. Omit portions fully hidden by occlusion. |
| `entry_point` | points | Single keypoint where the needle enters the tissue, on key frames. |
| `exit_point` | points | Single keypoint where the needle exits, on key frames. |
| `stitch` | rect | Draw the box around a **completed** stitch (after knot formation). Set `stitch_id` (`<video_id>-s<NN>`, numbered per video) and the six quality scores. One `stitch` annotation per completed stitch, attached to its key frame. |
| `knot` | rect | Box around a visible knot. |
| `needle_entry` | rect track | Interval from when the needle first touches tissue until it disappears into it. |
| `needle_exit` | rect track | Interval from when the needle reappears until it clears the tissue. |
| `knot_formation` | rect track | Interval from the first wrapping movement until the knot is pulled tight. |
| `stitch_completion` | rect track | Interval ending when the stitch is finished and the knot is set. |

## Score-labeling rules (six quality scores)

Each completed stitch receives six ratings on the 1-10 scale. The anchors below are derived from `Docs/index.html` and are a draft to be approved:

| Score | Meaning |
| --- | --- |
| 1-3 | Needs work |
| 4-6 | Getting there |
| 7-9 | Good quality |
| 10 | Excellent |

| Metric | What is rated |
| --- | --- |
| `overall` | Global quality of the stitch |
| `isd` | Inter-Suture Distance: consistency of spacing to adjacent stitches |
| `slack` | Thread tension control (loose vs floppy vs tight) |
| `position` | Entry/exit placement accuracy relative to the wound |
| `angulation` | Needle entry/exit angle |
| `width` | Stitch size consistency |

A stitch receives its scores **only after `stitch_completion`**; do not score while it is still being formed.

## Visibility and occlusion rules

- Set the `visibility` attribute on every annotation: `visible`, `partially_occluded`, or `occluded`.
- If an object cannot be seen at all, do not annotate it on that frame; mark the stitch `visibility` accordingly.
- If a quality metric cannot be judged (for example, slack is invisible because the thread is occluded), leave that metric at its default and set the stitch `review_status` to `disputed` with a comment. Unresolved disputes are excluded from training ground truth.

## Review workflow

1. A primary clinical expert annotates the full set.
2. A second expert reviews a defined subset (20% of stitches) independently.
3. Agreement is measured (ICC for scores, Cohen's Kappa for events where appropriate).
4. Disagreements are adjudicated by discussion; the result is recorded.
5. `review_status` is set to `agreed` for resolved annotations and `disputed` for unresolved ones. Only `reviewed`/`agreed` annotations enter the frozen dataset.

## Metadata

- `stitch_id`: `<video_id>-s<NN>`, numbered per video.
- `annotator_id`: stable identifier per annotator.
- `review_status`: `pending` → `reviewed` → `agreed` / `disputed`.
- Every annotation carries the frame timestamp from the frame name, and the source video from the task.

## Export

Export the task annotations from CVAT. Keep the export, the schema, and the annotation version together so the frozen dataset can be reproduced. See `docs/specs/suture-quality-pipeline.md` for the dataset freeze and split requirements.