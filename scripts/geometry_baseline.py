# /// script
# dependencies = ["opencv-python>=4.10", "scikit-learn>=1.5", "numpy>=1.26"]
# ///
"""Train and evaluate the pixel-level geometry baseline on the frozen dataset.

Run with: uv run python -m scripts.geometry_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES: tuple[str, ...] = ("background", "wound_pad", "needle", "thread")
BACKGROUND, WOUND_PAD, NEEDLE, THREAD = 0, 1, 2, 3
GRID_SIZE: int = 96
MAX_PIXELS_PER_FRAME: int = 1500
MAX_TRAIN_PIXELS: int = 800_000


def build_class_mask(shapes: list[dict[str, object]], size: int) -> np.ndarray:
    """Return a per-pixel class id mask from annotated shapes."""
    mask = np.zeros((size, size), dtype=np.uint8)
    scale = size / 384.0
    for shape in shapes:
        label = str(shape.get("label", ""))
        kind = str(shape.get("kind", ""))
        points = shape.get("points", [])
        if not isinstance(points, list):
            continue
        if kind == "polygon" and label == "wound_pad":
            pts = np.array(
                [[p[0] * scale, p[1] * scale] for p in points], dtype=np.int32
            ).reshape(-1, 1, 2)
            _ = cv2.fillPoly(mask, [pts], WOUND_PAD)
        elif kind == "box" and label == "needle":
            (x0, y0), (x1, y1) = points
            _ = cv2.rectangle(
                mask,
                (int(x0 * scale), int(y0 * scale)),
                (int(x1 * scale), int(y1 * scale)),
                NEEDLE,
                -1,
            )
        elif kind == "polyline" and label == "thread":
            (x0, y0), (x1, y1) = points
            _ = cv2.line(
                mask,
                (int(x0 * scale), int(y0 * scale)),
                (int(x1 * scale), int(y1 * scale)),
                THREAD,
                2,
            )
    return mask


def load_annotation(
    video_id: str, frame_name: str, annotations_dir: Path
) -> list[dict[str, object]]:
    """Return the shapes for one frame from the annotation store."""
    path = annotations_dir / f"{video_id}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for frame in payload.get("frames", []):
        if str(frame.get("name")) == frame_name:
            return frame.get("shapes", [])
    return []


def pixel_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Return the fraction of pixels classified correctly."""
    return float((prediction == truth).mean())


def class_iou(prediction: np.ndarray, truth: np.ndarray, class_id: int) -> float:
    """Return intersection-over-union for one class."""
    pred = prediction == class_id
    truth_mask = truth == class_id
    intersection = int((pred & truth_mask).sum())
    union = int((pred | truth_mask).sum())
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def sample_pixels(
    feature: np.ndarray, mask: np.ndarray, max_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced per-pixel training samples from one frame."""
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for class_id in range(4):
        indices = np.argwhere(mask == class_id)
        if indices.shape[0] == 0:
            continue
        chosen = indices[
            np.random.default_rng(class_id + mask.sum()).choice(
                indices.shape[0], min(max_per_class, indices.shape[0]), replace=False
            )
        ]
        rows.append(feature[chosen[:, 0], chosen[:, 1]])
        columns.append(np.full(chosen.shape[0], class_id, dtype=np.uint8))
    if not rows:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    return np.concatenate(rows), np.concatenate(columns)


def frame_path(video_id: str, frame_name: str, frames_dir: Path) -> Path:
    """Return the frame image path from its manifest output path name."""
    return frames_dir / video_id / frame_name


def predict_frame(model, frame: np.ndarray) -> np.ndarray:
    """Classify every pixel of one frame at GRID_SIZE resolution."""
    resized = cv2.resize(frame, (GRID_SIZE, GRID_SIZE))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    labels = model.predict(hsv)
    return labels.reshape(GRID_SIZE, GRID_SIZE)


def overlay_preview(frame: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Draw the predicted class mask on the full-resolution frame."""
    color_map = {WOUND_PAD: (255, 0, 0), NEEDLE: (0, 255, 0), THREAD: (0, 0, 255)}
    upscaled = cv2.resize(
        prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    overlay = frame.copy()
    for class_id, color in color_map.items():
        overlay[upscaled == class_id] = color
    return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)


def main() -> int:
    """Train the baseline on the train split and evaluate on the test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dataset_manifest.csv")
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("data/annotations")
    )
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/geometry-baseline.md")
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("data/annotations/previews/baseline-test.png"),
    )
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "selected"]
    train_frames = [
        row for row in rows if row["split"] == "train" and row["annotated"] == "True"
    ]
    test_frames = [
        row for row in rows if row["split"] == "test" and row["annotated"] == "True"
    ]
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in train_frames[:150]:
        path = frame_path(
            row["video_id"], Path(row["output_path"]).name, args.frames_dir
        )
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        small = cv2.resize(frame, (GRID_SIZE, GRID_SIZE))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        shapes = load_annotation(
            row["video_id"], Path(row["output_path"]).name, args.annotations_dir
        )
        mask = build_class_mask(shapes, GRID_SIZE)
        pixel_features, pixel_labels = sample_pixels(hsv, mask, MAX_PIXELS_PER_FRAME)
        if pixel_features.shape[0]:
            features.append(pixel_features)
            labels.append(pixel_labels)
        if sum(f.shape[0] for f in features) > MAX_TRAIN_PIXELS:
            break
    if not features:
        raise RuntimeError("no training pixels collected")
    train_x = np.concatenate(features)
    train_y = np.concatenate(labels)
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=0
    )
    model.fit(train_x, train_y)
    truth_pixels: list[np.ndarray] = []
    predicted_pixels: list[np.ndarray] = []
    preview_frames: list[np.ndarray] = []
    for row in test_frames[:40]:
        path = frame_path(
            row["video_id"], Path(row["output_path"]).name, args.frames_dir
        )
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        shapes = load_annotation(
            row["video_id"], Path(row["output_path"]).name, args.annotations_dir
        )
        truth = build_class_mask(shapes, GRID_SIZE)
        prediction = predict_frame(model, frame)
        truth_pixels.append(truth)
        predicted_pixels.append(prediction)
        if len(preview_frames) < 12:
            preview_frames.append(overlay_preview(frame, prediction))
    truth_all = np.stack(truth_pixels)
    predicted_all = np.stack(predicted_pixels)
    accuracy = pixel_accuracy(predicted_all, truth_all)
    lines = [
        "# Geometry Baseline Report",
        "",
        f"Model: RandomForest pixel classifier (HSV, {GRID_SIZE}x{GRID_SIZE})",
        f"Train frames sampled: {min(len(train_frames), 60)} (capped at {MAX_TRAIN_PIXELS} pixels)",
        f"Test frames evaluated: {len(truth_pixels)}",
        f"Pixel accuracy: {accuracy:.4f}",
        "",
        "| Class | IoU |",
        "| --- | ---: |",
        *[
            f"| {name} | {class_iou(predicted_all, truth_all, class_id):.4f} |"
            for class_id, name in enumerate(CLASS_NAMES)
        ],
        "",
        "## Limitations",
        "- Labels are auto-generated and unreviewed (ADR-0005); metrics measure agreement with those labels, not clinical truth.",
        "- Baseline is the comparison point for the deep model unit.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if preview_frames:
        rows_grid, cols_grid = 3, 4
        cell = 192
        sheet = np.full((rows_grid * cell, cols_grid * cell, 3), 30, dtype=np.uint8)
        for position, frame in enumerate(preview_frames[: rows_grid * cols_grid]):
            row, col = divmod(position, cols_grid)
            sheet[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = (
                cv2.resize(frame, (cell, cell))
            )
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        _ = cv2.imwrite(str(args.preview), sheet)
    print(
        f"accuracy={accuracy:.4f} "
        + " ".join(
            f"{name}={class_iou(predicted_all, truth_all, c):.3f}"
            for c, name in enumerate(CLASS_NAMES)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
