# /// script
# dependencies = ["opencv-python>=4.10", "torch>=2.0", "numpy>=1.26"]
# ///
"""Train and evaluate a small U-Net geometry model on the frozen dataset (CPU).

Run with: uv run python -m scripts.deep_geometry
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from scripts.geometry_baseline import (
    CLASS_NAMES,
    GRID_SIZE,
    build_class_mask,
    class_iou,
    frame_path,
    load_annotation,
    pixel_accuracy,
)

TRAIN_FRAMES_CAP: int = 400
VAL_FRACTION: float = 0.1
EPOCHS: int = 12
BATCH_SIZE: int = 16
BASE_CHANNELS: int = 32
TEST_FRAMES_CAP: int = 60


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert a class-id mask to a long tensor."""
    return torch.from_numpy(mask.astype(np.int64))


class TinyUNet(nn.Module):
    """A compact U-Net for 4-class segmentation at low resolution."""

    def __init__(
        self, in_channels: int = 3, num_classes: int = 4, base: int = BASE_CHANNELS
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                self._block(in_channels, base),
                self._block(base, base * 2),
                self._block(base * 2, base * 2),
            ]
        )
        self.bridge = self._block(base * 2, base * 2)
        self.decoder = nn.ModuleList(
            [
                self._block(base * 4, base * 2),
                self._block(base * 4, base * 2),
                self._block(base * 3, base),
            ]
        )
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.bridge(x)
        for index, block in enumerate(self.decoder):
            skip = skips[len(skips) - index - 1]
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)


def load_frame_tensor(video_id: str, frame_name: str, frames_dir: Path) -> torch.Tensor:
    """Load one frame as a normalized (3, H, W) tensor."""
    frame = cv2.imread(str(frame_path(video_id, frame_name, frames_dir)))
    if frame is None:
        raise RuntimeError(f"could not read frame {video_id}/{frame_name}")
    resized = cv2.resize(frame, (GRID_SIZE, GRID_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def class_weights(masks: list[np.ndarray]) -> torch.Tensor:
    """Return inverse-frequency class weights from training masks."""
    counts = np.zeros(4, dtype=np.float64)
    for mask in masks:
        counts += np.bincount(mask.ravel(), minlength=4)
    total = counts.sum()
    weights = np.where(counts > 0, total / (4 * counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def mean_iou(predictions: list[np.ndarray], truths: list[np.ndarray]) -> float:
    """Return the mean per-class IoU across all provided frames."""
    sums = np.zeros(4)
    for prediction, truth in zip(predictions, truths):
        for class_id in range(4):
            sums[class_id] += class_iou(prediction, truth, class_id)
    return float(sums.mean() / max(1, len(predictions)))


def main() -> int:
    """Train the U-Net on the train split and evaluate on the test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dataset_manifest.csv")
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("data/annotations")
    )
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--report", type=Path, default=Path("reports/geometry-deep.md"))
    parser.add_argument(
        "--preview", type=Path, default=Path("data/annotations/previews/deep-test.png")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/models/geometry_unet.pt")
    )
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "selected"]
    train_rows = [
        row for row in rows if row["split"] == "train" and row["annotated"] == "True"
    ]
    test_rows = [
        row for row in rows if row["split"] == "test" and row["annotated"] == "True"
    ]
    train_rows = train_rows[:TRAIN_FRAMES_CAP]
    split_point = max(1, int(len(train_rows) * (1 - VAL_FRACTION)))
    train_split, val_split = train_rows[:split_point], train_rows[split_point:]

    def build_inputs(
        frame_rows: list[dict[str, str]],
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        inputs: list[torch.Tensor] = []
        masks: list[np.ndarray] = []
        for row in frame_rows:
            frame_name = Path(row["output_path"]).name
            inputs.append(
                load_frame_tensor(row["video_id"], frame_name, args.frames_dir)
            )
            shapes = load_annotation(row["video_id"], frame_name, args.annotations_dir)
            masks.append(build_class_mask(shapes, GRID_SIZE))
        return inputs, masks

    train_inputs, train_masks = build_inputs(train_split)
    val_inputs, val_masks = build_inputs(val_split)
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_masks))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(train_inputs))
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            batch = torch.stack([train_inputs[index] for index in indices])
            target = torch.stack(
                [mask_to_tensor(train_masks[index]) for index in indices]
            )
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
        model.eval()
        val_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for frame, mask in zip(val_inputs, val_masks):
                logits = model(frame.unsqueeze(0))
                val_predictions.append(logits.argmax(dim=1)[0].numpy())
        score = mean_iou(val_predictions, val_masks)
        print(f"epoch={epoch + 1}/{EPOCHS} val_mean_iou={score:.4f}")
        if score > best_score:
            best_score = score
            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
    assert best_state is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.checkpoint)
    model.load_state_dict(best_state)
    model.eval()
    test_truths: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    preview_frames: list[np.ndarray] = []
    with torch.no_grad():
        for row in test_rows[:TEST_FRAMES_CAP]:
            frame_name = Path(row["output_path"]).name
            frame = cv2.imread(
                str(frame_path(row["video_id"], frame_name, args.frames_dir))
            )
            if frame is None:
                continue
            logits = model(
                load_frame_tensor(
                    row["video_id"], frame_name, args.frames_dir
                ).unsqueeze(0)
            )
            prediction = logits.argmax(dim=1)[0].numpy()
            shapes = load_annotation(row["video_id"], frame_name, args.annotations_dir)
            truth = build_class_mask(shapes, GRID_SIZE)
            test_predictions.append(prediction)
            test_truths.append(truth)
            if len(preview_frames) < 12:
                upscaled = cv2.resize(
                    prediction.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                overlay = frame.copy()
                overlay[upscaled == 1] = (255, 0, 0)
                overlay[upscaled == 2] = (0, 255, 0)
                overlay[upscaled == 3] = (0, 0, 255)
                preview_frames.append(cv2.addWeighted(frame, 0.6, overlay, 0.4, 0))
    accuracy = pixel_accuracy(np.stack(test_predictions), np.stack(test_truths))
    lines = [
        "# Deep Geometry Baseline Report",
        "",
        f"Model: TinyUNet ({sum(p.numel() for p in model.parameters()):,} params), CPU, {GRID_SIZE}x{GRID_SIZE}",
        f"Train frames: {len(train_split)} (capped {TRAIN_FRAMES_CAP}), val: {len(val_split)}, test: {len(test_truths)}",
        f"Best val mean IoU: {best_score:.4f}",
        f"Pixel accuracy (test): {accuracy:.4f}",
        "",
        "| Class | IoU |",
        "| --- | ---: |",
        *[
            f"| {name} | {class_iou(np.stack(test_predictions), np.stack(test_truths), class_id):.4f} |"
            for class_id, name in enumerate(CLASS_NAMES)
        ],
        "",
        "## vs RandomForest baseline (reports/geometry-baseline.md)",
        "- RF: accuracy 0.506, IoU background 0.546 / pad 0.049 / needle 0.010 / thread 0.000",
        "",
        "## Limitations",
        "- Labels are auto-generated and unreviewed (ADR-0005); metrics measure agreement with those labels.",
        "- CPU-only training (GPU driver too old for CUDA torch); expect gains to grow with resolution and data.",
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
            f"{name}={class_iou(np.stack(test_predictions), np.stack(test_truths), c):.3f}"
            for c, name in enumerate(CLASS_NAMES)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
