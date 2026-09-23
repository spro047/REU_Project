# Staged Video Model

The model will be a staged multi-task pipeline rather than one end-to-end video model: object/geometry estimation feeds tracking, temporal event recognition, and quality regression. This makes the pilot easier to audit and reduces data demands while preserving a path to later temporal modeling.
