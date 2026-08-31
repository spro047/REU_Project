# Frame Extraction Report

Extraction tool: ffmpeg -vf fps=5.0,scale=384:384
Sample rate: 5.0 fps
Sharpness threshold: 50.0
Duplicate mean-abs-diff threshold: 2.0
Source: Dataset_From_JNMC

| Status | Count |
| --- | ---: |
| selected | 19145 |
| excluded_blurred | 42 |
| excluded_duplicate | 2221 |
| excluded_decode_failure | 0 |

Occlusion exclusion requires the geometry model from a later ticket and is not detected yet.
Event-centered windows will be added once CVAT annotations exist.
