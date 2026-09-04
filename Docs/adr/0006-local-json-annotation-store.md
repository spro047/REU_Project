# Local JSON Annotation Store Without CVAT

CVAT is not required for pipeline-generated annotations. Annotations are stored as versioned JSON per video under the local annotation store (`data/annotations/<video>.json`) with frames, shapes (polygon/box/polyline), and attributes. Review happens through rendered preview contact sheets, not the CVAT UI.

CVAT remains optional for later human review or correction; the local JSON is the canonical store the model pipeline consumes. The label vocabulary from the CVAT schema remains the reference for shape/attribute names.