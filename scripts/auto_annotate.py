# /// script
# dependencies = ["opencv-python>=4.10"]
# ///
"""Generate heuristic annotations from extracted frames as local JSON.

Run with: uv run python -m scripts.auto_annotate --video video-7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

KEY_FRAME_EVERY: int = 20
PRESENCE_EVERY: int = 5
PIPELINE_VERSION: str = "1"
PREVIEW_GRID: tuple[int, int] = (6, 6)


@dataclass(frozen=True, slots=True)
class Shape:
    label: str
    kind: str
    points: list[list[float]]
    attributes: dict[str, str]


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    index: int
    name: str
    shapes: list[Shape]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a mask of only the largest connected component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return (labels == largest).astype(np.uint8)


def pad_region(frame: np.ndarray) -> np.ndarray:
    """Segment the surgical pad as the largest non-background color region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    _, labels, _ = cv2.kmeans(
        pixels, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1,
        cv2.KMEANS_PP_CENTERS,
    )
    label_map = labels.reshape(hsv.shape[:2])
    areas = [int((label_map == c).sum()) for c in range(3)]
    background = int(np.argmax(areas))
    candidates = [c for c in range(3) if c != background]
    if not candidates:
        return np.zeros_like(hsv[:, :, 0])
    pad_class = max(candidates, key=lambda c: areas[c])
    return largest_component((label_map == pad_class).astype(np.uint8))


def region_polygon(mask: np.ndarray) -> list[list[float]]:
    """Return a downsampled polygon around the largest contour of a mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500:
        return []
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return []
    return [[float(point[0][0]), float(point[0][1])] for point in approx]


def needle_shape(frame: np.ndarray, gray: np.ndarray) -> Shape | None:
    """Return a bounding box for the brightest elongated metallic region."""
    mask = largest_component((gray > 170).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 300 or max(w, h) / max(1, min(w, h)) < 1.5:
        return None
    return Shape(
        "needle",
        "box",
        [[float(x), float(y)], [float(x + w), float(y + h)]],
        {"visibility": "visible"},
    )


def thread_shape(gray: np.ndarray, pad_mask: np.ndarray) -> Shape | None:
    """Return a coarse polyline along the principal axis of dark thread pixels inside the pad."""
    mask = largest_component(((gray < 80) & (pad_mask > 0)).astype(np.uint8))
    points = np.argwhere(mask > 0)
    if points.shape[0] < 100:
        return None
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    _, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, 1]
    projections = centered @ axis
    min_p = center + projections.min() * axis
    max_p = center + projections.max() * axis
    return Shape(
        "thread",
        "polyline",
        [[float(min_p[1]), float(min_p[0])], [float(max_p[1]), float(max_p[0])]],
        {"visibility": "visible"},
    )


def analyze_frame(path: Path) -> tuple[list[Shape], bool]:
    """Run the heuristic detectors on one frame and report detected shapes."""
    frame = cv2.imread(str(path))
    if frame is None:
        return [], False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pad_mask = pad_region(frame)
    shapes: list[Shape] = []
    polygon = region_polygon(pad_mask)
    if polygon:
        shapes.append(Shape("wound_pad", "polygon", polygon, {"visibility": "visible"}))
    needle = needle_shape(frame, gray)
    if needle is not None:
        shapes.append(needle)
    if polygon:
        thread = thread_shape(gray, pad_mask)
        if thread is not None:
            shapes.append(thread)
    return shapes, needle is not None


def draw_shapes(frame: np.ndarray, shapes: list[Shape]) -> np.ndarray:
    """Draw shapes onto a frame for preview."""
    for shape in shapes:
        if shape.kind == "box":
            (x0, y0), (x1, y1) = shape.points
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)
            cv2.putText(
                frame,
                shape.label,
                (int(x0), max(10, int(y0) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )
        elif shape.kind == "polygon":
            pts = np.array(shape.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 1)
        elif shape.kind == "polyline":
            pts = np.array(shape.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (0, 0, 255), 1)
    return frame


def write_preview(records: list[FrameAnnotation], frame_dir: Path, output_path: Path) -> None:
    """Write one contact-sheet PNG per video with drawn annotations."""
    rows, cols = PREVIEW_GRID
    cell = 192
    sheet = np.full((rows * cell, cols * cell, 3), 30, dtype=np.uint8)
    for position, record in enumerate(records[: rows * cols]):
        frame = cv2.imread(str(frame_dir / record.name))
        if frame is None:
            continue
        frame = cv2.resize(frame, (cell, cell))
        annotated = draw_shapes(frame, record.shapes)
        row, col = divmod(position, cols)
        sheet[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = annotated
        cv2.putText(
            sheet,
            f"#{record.index}",
            (col * cell + 4, row * cell + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = cv2.imwrite(str(output_path), sheet)


def write_json(records: list[FrameAnnotation], video_id: str, output_path: Path) -> None:
    """Write the annotation store as versioned JSON."""
    payload = {
        "video_id": video_id,
        "pipeline": {"name": "auto_annotate", "version": PIPELINE_VERSION},
        "reviewed": False,
        "frames": [asdict(record) for record in records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> int:
    """Run heuristic annotation for one video into the local store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations"))
    args = parser.parse_args()
    frame_dir = args.frames_dir / args.video
    paths = tuple(sorted(frame_dir.glob("frame_*.png")))
    records: list[FrameAnnotation] = []
    needle_presence: list[bool] = []
    for index, path in enumerate(paths):
        shapes, has_needle = analyze_frame(path)
        needle_presence.append(has_needle)
        if index % KEY_FRAME_EVERY == 0 or index % PRESENCE_EVERY == 0:
            records.append(FrameAnnotation(index, path.name, shapes))
    write_json(records, args.video, args.output / f"{args.video}.json")
    write_preview(records, frame_dir, args.output / "previews" / f"{args.video}.png")
    pad_count = sum(1 for r in records if any(s.label == "wound_pad" for s in r.shapes))
    needle_count = sum(1 for r in records if any(s.label == "needle" for s in r.shapes))
    thread_count = sum(1 for r in records if any(s.label == "thread" for s in r.shapes))
    transitions = sum(1 for a, b in zip(needle_presence, needle_presence[1:]) if a != b)
    print(
        f"video={args.video} annotated_frames={len(records)} pad={pad_count} "
        f"needle={needle_count} thread={thread_count} transitions={transitions}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())