"""Validate the CVAT label schema used for the pilot dataset.

Run with: uv run python -m scripts.validate_cvat_schema configs/cvat_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REQUIRED_LABELS: Final[frozenset[str]] = frozenset(
    {
        "wound_pad",
        "needle",
        "thread",
        "entry_point",
        "exit_point",
        "stitch",
        "knot",
        "needle_entry",
        "needle_exit",
        "knot_formation",
        "stitch_completion",
    }
)
ALLOWED_TYPES: Final[frozenset[str]] = frozenset(
    {
        "rectangle",
        "polygon",
        "polyline",
        "points",
        "ellipse",
        "cuboid",
        "skeleton",
        "tag",
        "mask",
    }
)
ALLOWED_INPUT_TYPES: Final[frozenset[str]] = frozenset(
    {"text", "number", "select", "radio", "checkbox"}
)
STITCH_QUALITY_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"overall", "isd", "slack", "position", "angulation", "width"}
)
STITCH_METADATA_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"stitch_id", "annotator_id", "review_status", "visibility"}
)


@dataclass(frozen=True, slots=True)
class SchemaError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _attributes_of(label: dict[str, object]) -> list[dict[str, object]]:
    attributes = label.get("attributes", [])
    if not isinstance(attributes, list):
        raise SchemaError(f"attributes of label {label.get('name')!r} must be a list")
    return attributes


def _attribute_names(label: dict[str, object]) -> set[str]:
    names: set[str] = set()
    for attribute in _attributes_of(label):
        if not isinstance(attribute, dict):
            raise SchemaError(
                f"attribute of label {label.get('name')!r} must be an object"
            )
        raw_name = attribute.get("name")
        if not isinstance(raw_name, str):
            raise SchemaError(
                f"attribute of label {label.get('name')!r} must have a string name"
            )
        names.add(raw_name)
    return names


def validate_schema(data: Mapping[str, object]) -> None:
    """Validate a parsed CVAT label schema, raising SchemaError on any violation."""
    raw_labels = data.get("labels")
    if not isinstance(raw_labels, list):
        raise SchemaError("schema must contain a 'labels' list")
    labels: dict[str, dict[str, object]] = {}
    for label in raw_labels:
        if not isinstance(label, dict):
            raise SchemaError("each label must be an object")
        name = label.get("name")
        shape_type = label.get("type")
        if not isinstance(name, str) or not name:
            raise SchemaError("each label must have a non-empty string name")
        if not isinstance(shape_type, str) or shape_type not in ALLOWED_TYPES:
            raise SchemaError(f"label {name!r} has invalid type {shape_type!r}")
        for attribute in _attributes_of(label):
            input_type = attribute.get("input_type")
            if not isinstance(input_type, str) or input_type not in ALLOWED_INPUT_TYPES:
                raise SchemaError(
                    f"attribute {attribute.get('name')!r} has invalid input_type {input_type!r}"
                )
            values = attribute.get("values")
            if not values:
                raise SchemaError(
                    f"attribute {attribute.get('name')!r} must have non-empty values"
                )
            if input_type == "number" and (
                not isinstance(values, list) or len(values) != 3
            ):
                raise SchemaError(
                    f"number attribute {attribute.get('name')!r} needs exactly three values (min, max, step)"
                )
        labels[name] = label
    missing = REQUIRED_LABELS - set(labels)
    if missing:
        raise SchemaError(f"missing required labels: {', '.join(sorted(missing))}")
    stitch = labels["stitch"]
    missing_quality = STITCH_QUALITY_ATTRIBUTES - _attribute_names(stitch)
    if missing_quality:
        raise SchemaError(
            f"stitch label missing quality attributes: {', '.join(sorted(missing_quality))}"
        )
    missing_metadata = STITCH_METADATA_ATTRIBUTES - _attribute_names(stitch)
    if missing_metadata:
        raise SchemaError(
            f"stitch label missing metadata attributes: {', '.join(sorted(missing_metadata))}"
        )


def main() -> int:
    """Validate the schema file given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema", type=Path)
    args = parser.parse_args()
    with args.schema.open(encoding="utf-8") as handle:
        data = json.load(handle)
    validate_schema(data)
    print(f"Schema valid: {args.schema}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SchemaError as error:
        print(f"Schema invalid: {error}", file=sys.stderr)
        raise SystemExit(1) from error
