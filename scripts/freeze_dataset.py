# /// script
# dependencies = ["opencv-python>=4.10"]
# ///
"""Freeze the pilot dataset: join frames with annotations and emit grouped splits.

Run with: uv run python -m scripts.freeze_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scripts.audit_videos import discover_videos


@dataclass(frozen=True, slots=True)
class FreezeError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class FrameRow:
    video_id: str
    frame_id: str
    timestamp_seconds: float
    output_path: str
    status: str


@dataclass(frozen=True, slots=True)
class DatasetRow:
    video_id: str
    frame_id: str
    timestamp_seconds: float
    output_path: str
    status: str
    split: str
    annotated: bool
    reviewed: bool


def assign_splits(
    video_ids: tuple[str, ...], train_count: int, val_count: int, test_count: int
) -> dict[str, str]:
    """Assign videos to train/val/test deterministically by sorted order."""
    ordered = tuple(sorted(video_ids))
    if train_count + val_count + test_count != len(ordered):
        raise FreezeError(
            f"split counts {train_count}+{val_count}+{test_count} do not sum to {len(ordered)} videos"
        )
    assignment: dict[str, str] = {}
    for index, video_id in enumerate(ordered):
        if index < train_count:
            assignment[video_id] = "train"
        elif index < train_count + val_count:
            assignment[video_id] = "val"
        else:
            assignment[video_id] = "test"
    return assignment


def load_annotation_shapes(annotations_dir: Path, video_id: str) -> dict[str, int]:
    """Return frame name -> shape count from a video's annotation JSON."""
    path = annotations_dir / f"{video_id}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(frame["name"]): len(frame["shapes"]) for frame in payload.get("frames", [])
    }


def build_rows(
    frames: tuple[FrameRow, ...],
    shapes: dict[str, int],
    splits: dict[str, str],
) -> tuple[DatasetRow, ...]:
    """Join frame rows with split and annotation presence."""
    rows: list[DatasetRow] = []
    for row in frames:
        annotated = shapes.get(Path(row.output_path).name, 0) > 0
        rows.append(
            DatasetRow(
                row.video_id,
                row.frame_id,
                row.timestamp_seconds,
                row.output_path,
                row.status,
                splits[row.video_id],
                annotated,
                False,
            )
        )
    return tuple(rows)


def write_outputs(
    rows: tuple[DatasetRow, ...], manifest_path: Path, report_path: Path
) -> None:
    """Write the frozen dataset manifest and summary report."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DatasetRow.__dataclass_fields__)
        _ = writer.writeheader()
        _ = writer.writerows(
            {field: getattr(row, field) for field in DatasetRow.__dataclass_fields__}
            for row in rows
        )
    status_counts = Counter(row.status for row in rows)
    split_counts = Counter(row.split for row in rows)
    annotated_counts = Counter(row.split for row in rows if row.annotated)
    by_video: dict[str, Counter[str]] = {}
    for row in rows:
        by_video.setdefault(row.video_id, Counter())[row.split] += 1
    lines = [
        "# Dataset Freeze Report",
        "",
        "## Splits",
        "| Split | Frames | Annotated |",
        "| --- | ---: | ---: |",
        *[
            f"| {split} | {split_counts[split]} | {annotated_counts[split]} |"
            for split in ("train", "val", "test")
        ],
        "",
        "## Per video",
        "| Video | Split | Frames |",
        "| --- | --- | ---: |",
        *[
            f"| {video} | {next(iter(counts))} | {counts[next(iter(counts))]} |"
            for video, counts in by_video.items()
        ],
        "",
        "## Frame status",
        "| Status | Count |",
        "| --- | ---: |",
        *[f"| {status} | {count} |" for status, count in status_counts.most_common()],
        "",
        "## Limitations",
        "- Split is by video only (operator metadata is unknown).",
        "- Annotations are machine-generated and unreviewed (ADR-0005, ADR-0006).",
        "- Only geometry labels exist; quality scores and events are not annotated.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Freeze the pilot dataset with grouped splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/frame_manifest.csv")
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("data/annotations")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/dataset_manifest.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/dataset-freeze.md")
    )
    parser.add_argument("--train", type=int, default=5)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        rows = tuple(
            FrameRow(
                str(row["video_id"]),
                str(row["frame_id"]),
                float(row["timestamp_seconds"]),
                str(row["output_path"]),
                str(row["status"]),
            )
            for row in csv.DictReader(handle)
        )
    videos = tuple(
        path.stem.lower() for path in discover_videos(Path("Dataset_From_JNMC"))
    )
    splits = assign_splits(videos, args.train, args.val, args.test)
    shape_map = {
        video_id: load_annotation_shapes(args.annotations_dir, video_id)
        for video_id in videos
    }
    dataset_rows = tuple(
        DatasetRow(
            row.video_id,
            row.frame_id,
            row.timestamp_seconds,
            row.output_path,
            row.status,
            splits[row.video_id],
            shape_map[row.video_id].get(Path(row.output_path).name, 0) > 0,
            False,
        )
        for row in rows
    )
    write_outputs(dataset_rows, args.output, args.report)
    print(
        f"Froze {len(dataset_rows)} frames across {len(videos)} videos; "
        f"train={sum(1 for r in dataset_rows if r.split == 'train')} "
        f"val={sum(1 for r in dataset_rows if r.split == 'val')} "
        f"test={sum(1 for r in dataset_rows if r.split == 'test')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
