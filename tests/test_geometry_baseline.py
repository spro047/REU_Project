import numpy as np

from scripts.geometry_baseline import (
    build_class_mask,
    class_iou,
    pixel_accuracy,
)


def make_shape(label: str, kind: str, points: list[list[float]]) -> dict[str, object]:
    return {"label": label, "kind": kind, "points": points, "attributes": {}}


def test_build_class_mask_fills_polygon_box_and_line() -> None:
    shapes = [
        make_shape(
            "wound_pad",
            "polygon",
            [[0.0, 0.0], [384.0, 0.0], [384.0, 384.0], [0.0, 384.0]],
        ),
        make_shape("needle", "box", [[20.0, 20.0], [80.0, 60.0]]),
        make_shape("thread", "polyline", [[10.0, 10.0], [300.0, 300.0]]),
    ]
    mask = build_class_mask(shapes, 96)
    assert mask[48, 10] == 1
    assert mask[10, 16] == 2
    assert mask[50, 50] == 3


def test_pixel_accuracy_and_iou() -> None:
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[0:2, 0:2] = 1
    prediction = np.zeros((4, 4), dtype=np.uint8)
    prediction[0:2, 0:2] = 1
    assert pixel_accuracy(prediction, truth) == 1.0
    assert class_iou(prediction, truth, 1) == 1.0
    prediction[3, 3] = 1
    assert class_iou(prediction, truth, 1) < 1.0
