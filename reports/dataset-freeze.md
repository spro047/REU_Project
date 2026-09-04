# Dataset Freeze Report

## Splits
| Split | Frames | Annotated |
| --- | ---: | ---: |
| train | 11818 | 2364 |
| val | 3525 | 706 |
| test | 6065 | 1213 |

## Per video
| Video | Split | Frames |
| --- | --- | ---: |
| video-1 | train | 1620 |
| video-10 | train | 2962 |
| video-2 | train | 4143 |
| video-3 | train | 1969 |
| video-5 | train | 1124 |
| video-6 | val | 2913 |
| video-7 | val | 612 |
| video-8 | test | 1625 |
| video-9 | test | 4440 |

## Frame status
| Status | Count |
| --- | ---: |
| selected | 19145 |
| excluded_duplicate | 2221 |
| excluded_blurred | 42 |

## Limitations
- Split is by video only (operator metadata is unknown).
- Annotations are machine-generated and unreviewed (ADR-0005, ADR-0006).
- Only geometry labels exist; quality scores and events are not annotated.
