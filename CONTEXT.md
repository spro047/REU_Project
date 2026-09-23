# Suture Quality Assessment Context

This project studies a computer-vision assistant for assessing suturing technique in fixed-camera surgical-training videos. It is a pilot feasibility study, not a clinical decision or accreditation system.

## Language

**Stitch**:
A single completed suturing unit, considered from needle entry through stitch completion/knot formation.
_Avoid_: Suture line, unless referring to the complete sequence of stitches.

**Suture line**:
The complete ordered sequence of stitches closing the wound or artificial skin pad.
_Avoid_: Stitch, when referring to the whole sequence.

**Suture quality score**:
An expert- or model-produced 1-10 rating of stitch or suture-line quality. It is not the same as model accuracy or confidence.
_Avoid_: Accuracy score, when referring to the predicted 1-10 rating.

**Model confidence**:
The model's estimated certainty in a prediction, reported separately from the suture quality score.
_Avoid_: Accuracy.

**Inter-Suture Distance (ISD)**:
The consistency of spacing between adjacent stitches.
_Avoid_: Stitch gap, unless used only as an informal explanation.

**Slack**:
The degree of thread looseness or tension-control quality visible in the stitch.
_Avoid_: Thread accuracy.

**Position**:
The placement accuracy of needle entry and exit relative to the intended wound location.
_Avoid_: Location score.

**Angulation**:
The angle of needle entry and exit.
_Avoid_: Needle direction, when a measured angle is intended.

**Width**:
The consistency of stitch size or span.
_Avoid_: Stitch length, unless the annotation rubric defines them as identical.

**Stitch completion**:
The temporal event after which a stitch is considered complete and eligible for scoring.
_Avoid_: Prediction complete.

**Event interval**:
A temporal span covering an event such as needle entry, needle exit, knot formation, or stitch completion.
_Avoid_: Event frame, when the event has a duration.

**Key frame**:
A representative frame used for detailed geometry and quality annotation.
_Avoid_: Every frame, unless dense annotation is explicitly intended.

**Pilot dataset**:
The versioned frame-and-annotation collection used to test feasibility and measurement reliability before claims of broader generalization.
_Avoid_: Clinical dataset.
