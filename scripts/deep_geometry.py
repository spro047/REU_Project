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

VAL_FRACTION: float = 0.1
BATCH_SIZE: int = 16
BASE_CHANNELS: int = 32
COPY_PASTE_PROBABILITY: float = 0.5
SCE_ALPHA: float = 1.0
SCE_BETA: float = 0.1


def clahe_frame(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB to even out illumination."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    adjusted = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(
        lightness
    )
    return cv2.cvtColor(cv2.merge([adjusted, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def symmetric_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = SCE_ALPHA,
    beta: float = SCE_BETA,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Noise-robust Symmetric Cross Entropy loss."""
    ce = torch.nn.functional.cross_entropy(logits, targets, weight=weight)
    probs = torch.nn.functional.softmax(logits, dim=1)
    one_hot = (
        torch.nn.functional.one_hot(targets, num_classes=logits.shape[1])
        .permute(0, 3, 1, 2)
        .float()
    )
    reverse = -(one_hot * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
    return alpha * ce + beta * reverse


def copy_paste(
    batch: torch.Tensor,
    masks: torch.Tensor,
    rng: np.random.Generator,
    probability: float = COPY_PASTE_PROBABILITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste a foreground patch from one sample onto another in the batch."""
    if batch.shape[0] < 2 or rng.random() > probability:
        return batch, masks
    source_index = int(rng.integers(batch.shape[0]))
    target_index = int(rng.integers(batch.shape[0] - 1))
    if target_index >= source_index:
        target_index += 1
    source_mask = masks[source_index]
    candidate_classes = [
        class_id for class_id in (2, 3) if int((source_mask == class_id).sum()) > 0
    ]
    if not candidate_classes:
        return batch, masks
    class_id = candidate_classes[int(rng.integers(len(candidate_classes)))]
    pixels = (source_mask == class_id).nonzero()
    if pixels.shape[0] == 0:
        return batch, masks
    min_row = int(pixels[:, 0].min())
    max_row = int(pixels[:, 0].max())
    min_col = int(pixels[:, 1].min())
    max_col = int(pixels[:, 1].max())
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    rows = masks.shape[1]
    cols = masks.shape[2]
    if height >= rows or width >= cols:
        return batch, masks
    offset_row = int(rng.integers(0, rows - height))
    offset_col = int(rng.integers(0, cols - width))
    patch = source_mask[min_row : max_row + 1, min_col : max_col + 1] == class_id
    target_pixels = masks[
        target_index, offset_row : offset_row + height, offset_col : offset_col + width
    ]
    target_pixels[patch] = class_id
    for channel in range(3):
        source_patch = batch[
            source_index, channel, min_row : max_row + 1, min_col : max_col + 1
        ]
        batch[
            target_index,
            channel,
            offset_row : offset_row + height,
            offset_col : offset_col + width,
        ][patch] = source_patch[patch]
    return batch, masks


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


def load_frame_tensor(
    video_id: str, frame_name: str, frames_dir: Path, use_clahe: bool = True
) -> torch.Tensor:
    """Load one frame as a normalized (3, H, W) tensor."""
    frame = cv2.imread(str(frame_path(video_id, frame_name, frames_dir)))
    if frame is None:
        raise RuntimeError(f"could not read frame {video_id}/{frame_name}")
    resized = cv2.resize(frame, (GRID_SIZE, GRID_SIZE))
    if use_clahe:
        resized = clahe_frame(resized)
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


def run_seed(
    args: argparse.Namespace,
    seed: int,
    train_split: list[dict[str, str]],
    val_split: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> tuple[dict[str, float], list[np.ndarray], dict[str, torch.Tensor]]:
    """Train and evaluate one seed; return metrics, previews, and best state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_inputs: list[torch.Tensor] = []
    train_masks: list[np.ndarray] = []
    val_inputs: list[torch.Tensor] = []
    val_masks: list[np.ndarray] = []
    for row in train_split:
        frame_name = Path(row["output_path"]).name
        train_inputs.append(
            load_frame_tensor(
                row["video_id"], frame_name, args.frames_dir, use_clahe=args.clahe
            )
        )
        train_masks.append(
            build_class_mask(
                load_annotation(row["video_id"], frame_name, args.annotations_dir),
                GRID_SIZE,
            )
        )
    for row in val_split:
        frame_name = Path(row["output_path"]).name
        val_inputs.append(
            load_frame_tensor(
                row["video_id"], frame_name, args.frames_dir, use_clahe=args.clahe
            )
        )
        val_masks.append(
            build_class_mask(
                load_annotation(row["video_id"], frame_name, args.annotations_dir),
                GRID_SIZE,
            )
        )
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = class_weights(train_masks)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(train_inputs))
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            batch = torch.stack([train_inputs[index] for index in indices])
            target = torch.stack(
                [mask_to_tensor(train_masks[index]) for index in indices]
            )
            if not args.no_copy_paste:
                batch, target = copy_paste(batch, target, rng)
            optimizer.zero_grad()
            logits = model(batch)
            loss = (
                symmetric_cross_entropy(logits, target, weight=weights)
                if not args.no_sce
                else nn.CrossEntropyLoss(weight=weights)(logits, target)
            )
            loss.backward()
            optimizer.step()
        model.eval()
        val_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for frame, mask in zip(val_inputs, val_masks):
                logits = model(frame.unsqueeze(0))
                val_predictions.append(logits.argmax(dim=1)[0].numpy())
        score = mean_iou(val_predictions, val_masks)
        print(f"seed={seed} epoch={epoch + 1}/{args.epochs} val_mean_iou={score:.4f}")
        if score > best_score:
            best_score = score
            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    test_truths: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    preview_frames: list[np.ndarray] = []
    with torch.no_grad():
        for row in test_rows[: args.test_cap]:
            frame_name = Path(row["output_path"]).name
            frame = cv2.imread(
                str(frame_path(row["video_id"], frame_name, args.frames_dir))
            )
            if frame is None:
                continue
            logits = model(
                load_frame_tensor(
                    row["video_id"],
                    frame_name,
                    args.frames_dir,
                    use_clahe=args.clahe,
                ).unsqueeze(0)
            )
            prediction = logits.argmax(dim=1)[0].numpy()
            truth = build_class_mask(
                load_annotation(row["video_id"], frame_name, args.annotations_dir),
                GRID_SIZE,
            )
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
    prediction_all = np.stack(test_predictions)
    truth_all = np.stack(test_truths)
    metrics = {"accuracy": pixel_accuracy(prediction_all, truth_all)}
    for class_id, name in enumerate(CLASS_NAMES):
        metrics[name] = class_iou(prediction_all, truth_all, class_id)
    return metrics, preview_frames, best_state


def main() -> int:
    """Train the U-Net across seeds and evaluate on the test split."""
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
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--no-sce", action="store_true")
    parser.add_argument("--no-copy-paste", action="store_true")
    parser.add_argument("--train-cap", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--test-cap", type=int, default=300)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "selected"]
    train_rows = [
        row for row in rows if row["split"] == "train" and row["annotated"] == "True"
    ]
    test_rows = [
        row for row in rows if row["split"] == "test" and row["annotated"] == "True"
    ]
    if args.train_cap > 0:
        train_rows = train_rows[: args.train_cap]
    split_point = max(1, int(len(train_rows) * (1 - VAL_FRACTION)))
    train_split, val_split = train_rows[:split_point], train_rows[split_point:]
    all_metrics: list[dict[str, float]] = []
    final_state: dict[str, torch.Tensor] | None = None
    preview_frames: list[np.ndarray] = []
    for seed in range(args.seeds):
        metrics, seed_previews, state = run_seed(
            args, seed, train_split, val_split, test_rows
        )
        all_metrics.append(metrics)
        final_state = state
        preview_frames = seed_previews
    assert final_state is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_state, args.checkpoint)
    mean: dict[str, float] = {}
    spread: dict[str, float] = {}
    for key in all_metrics[0]:
        values = np.array([metrics[key] for metrics in all_metrics])
        mean[key] = float(values.mean())
        spread[key] = float(values.std())
    lines = [
        "# Deep Geometry Baseline Report",
        "",
        f"Model: TinyUNet ({sum(p.numel() for p in TinyUNet().parameters()):,} params), CPU, {GRID_SIZE}x{GRID_SIZE}",
        f"Config: clahe={args.clahe} sce={not args.no_sce} copy_paste={not args.no_copy_paste}",
        f"Train frames: {len(train_split)}, val: {len(val_split)}, test: {args.test_cap}, seeds: {args.seeds}, epochs: {args.epochs}",
        "",
        "| Metric | mean +/- std |",
        "| --- | ---: |",
        *[
            f"| {name} | {mean[name]:.4f} +/- {spread[name]:.4f} |"
            for name in ("accuracy", *CLASS_NAMES)
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
    print(f"accuracy={mean['accuracy']:.4f} +/- {spread['accuracy']:.4f}")
    for name in CLASS_NAMES:
        print(f"{name}={mean[name]:.4f} +/- {spread[name]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
