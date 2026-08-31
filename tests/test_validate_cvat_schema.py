from scripts.validate_cvat_schema import SchemaError, validate_schema

Attribute = dict[str, object]
Label = dict[str, object]

QUALITY_ATTRIBUTES = ["overall", "isd", "slack", "position", "angulation", "width"]


def build_attribute(
    name: str, input_type: str, values: list[str] | None = None
) -> Attribute:
    return {"name": name, "input_type": input_type, "values": values or []}


def build_valid_schema() -> dict[str, list[Label]]:
    labels: list[Label] = []
    for name in ["wound_pad", "needle", "thread", "entry_point", "exit_point", "knot"]:
        labels.append({"name": name, "type": "polygon"})
    stitch_attributes = [
        *[build_attribute(name, "number") for name in QUALITY_ATTRIBUTES],
        *[build_attribute(name, "text") for name in ["stitch_id", "annotator_id"]],
        build_attribute(
            "review_status", "select", ["pending", "reviewed", "agreed", "disputed"]
        ),
        build_attribute(
            "visibility", "select", ["visible", "partially_occluded", "occluded"]
        ),
    ]
    labels.append({"name": "stitch", "type": "rect", "attributes": stitch_attributes})
    for name in ["needle_entry", "needle_exit", "knot_formation", "stitch_completion"]:
        labels.append({"name": name, "type": "rect"})
    return {"labels": labels}


def stitch_attributes(schema: dict[str, list[Label]]) -> list[Attribute]:
    stitch = next(label for label in schema["labels"] if label["name"] == "stitch")
    raw = stitch["attributes"]
    assert isinstance(raw, list)
    attributes: list[Attribute] = []
    for item in raw:
        assert isinstance(item, dict)
        attributes.append(item)
    return attributes


def test_valid_schema_passes() -> None:
    validate_schema(build_valid_schema())


def test_missing_required_label_fails() -> None:
    schema = build_valid_schema()
    schema["labels"] = [
        label for label in schema["labels"] if label["name"] != "stitch"
    ]
    try:
        validate_schema(schema)
    except SchemaError as error:
        assert "stitch" in str(error)
    else:
        raise AssertionError("expected SchemaError")


def test_stitch_missing_quality_attribute_fails() -> None:
    schema = build_valid_schema()
    stitch = next(label for label in schema["labels"] if label["name"] == "stitch")
    stitch["attributes"] = [
        a for a in stitch_attributes(schema) if a["name"] != "slack"
    ]
    try:
        validate_schema(schema)
    except SchemaError as error:
        assert "slack" in str(error)
    else:
        raise AssertionError("expected SchemaError")


def test_stitch_missing_metadata_attribute_fails() -> None:
    schema = build_valid_schema()
    stitch = next(label for label in schema["labels"] if label["name"] == "stitch")
    stitch["attributes"] = [
        a for a in stitch_attributes(schema) if a["name"] != "stitch_id"
    ]
    try:
        validate_schema(schema)
    except SchemaError as error:
        assert "stitch_id" in str(error)
    else:
        raise AssertionError("expected SchemaError")


def test_invalid_shape_type_fails() -> None:
    schema = build_valid_schema()
    schema["labels"][0]["type"] = "hexagon"
    try:
        validate_schema(schema)
    except SchemaError as error:
        assert "hexagon" in str(error)
    else:
        raise AssertionError("expected SchemaError")


def test_select_attribute_without_values_fails() -> None:
    schema = build_valid_schema()
    stitch = next(label for label in schema["labels"] if label["name"] == "stitch")
    stitch["attributes"] = [
        a
        if a["name"] != "review_status"
        else build_attribute("review_status", "select")
        for a in stitch_attributes(schema)
    ]
    try:
        validate_schema(schema)
    except SchemaError as error:
        assert "review_status" in str(error)
    else:
        raise AssertionError("expected SchemaError")
