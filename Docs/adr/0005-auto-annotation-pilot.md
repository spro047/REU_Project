# Auto-Annotation Pilot Without Expert Ground Truth

The project originally required a clinical expert to create seed annotations. No expert is available, so seed annotations are produced by an automated computer-vision pipeline instead, reviewed by the project owner, and explicitly treated as **non-expert, heuristic labels** rather than ground truth.

Consequences recorded:
- Quality scores derived from pixel geometry are approximate and must not be presented as expert ratings.
- Annotations are imported into CVAT as pre-annotations for human correction.
- The dataset freeze and any model training must state that labels are auto-generated and unreviewed by a clinician; research claims about scoring accuracy are out of scope until expert review exists.