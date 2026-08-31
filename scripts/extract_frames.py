# /// script
# dependencies = ["opencv-python>=4.10"]
# ///
"""Extract a quality-filtered frame dataset at 5 FPS using ffmpeg.

Run with: uv run scripts/extract_frames.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from scripts.audit_videos import discover_videos

FRAME_SIZE: int = 384
DUPLICATE_CHECK_SIZE: int = 64


@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame_id: str
    video_id: str
    source_frame_index: int
    timestamp_seconds: float
    output_path: str
    status: str
    sharpness: float | None
    duplicate_of: str | None


def find_ffmpeg() -> str:
    """Return the ffmpeg executable path or raise a typed error."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


def extract_with_ffmpeg(
    ffmpeg: str, video_path: Path, output_dir: Path, sample_rate: float
) -> tuple[Path, ...]:
    """Extract scaled frames at the requested rate and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_rate},scale={FRAME_SIZE}:{FRAME_SIZE}",
        str(output_dir / "frame_%06d.png"),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return tuple(sorted(output_dir.glob("frame_*.png")))


def source_fps_of(video_path: Path) -> float:
    """Return the source video frame rate from container metadata."""
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    _ = capture.release()
    return fps


def source_frame_index_for(
    output_index: int, source_fps: float, sample_rate: float
) -> int:
    """Map a 0-based extracted frame to its source frame index."""
    return round(output_index * source_fps / sample_rate)


def frame_id_for(video_id: str, frame_index: int, timestamp: float) -> str:
    """Return a stable frame identifier traceable to its source."""
    return f"{video_id}__f{frame_index:06d}__t{timestamp:.3f}"


def sharpness_of(gray) -> float:
    """Return Laplacian variance as a blur heuristic for one grayscale image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_abs_diff(a, b) -> float:
    """Return the mean absolute difference between two grayscale images."""
    return float(cv2.absdiff(a, b).mean())


def classify_frames(
    paths: tuple[Path, ...],
    video_id: str,
    source_fps: float,
    sample_rate: float,
    sharpness_threshold: float,
    duplicate_threshold: float,
) -> tuple[FrameRecord, ...]:
    """Classify extracted frames as selected or excluded and build records."""
    records: list[FrameRecord] = []
    previous: None | object = None
    previous_id = ""
    for output_index, path in enumerate(paths):
        timestamp = output_index / sample_rate
        frame_index = source_frame_index_for(output_index, source_fps, sample_rate)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            records.append(
                FrameRecord(
                    frame_id_for(video_id, frame_index, timestamp),
                    video_id,
                    frame_index,
                    timestamp,
                    str(path),
                    "excluded_decode_failure",
                    None,
                    None,
                )
            )
            continue
        sharpness = sharpness_of(gray)
        if sharpness < sharpness_threshold:
            records.append(
                FrameRecord(
                    frame_id_for(video_id, frame_index, timestamp),
                    video_id,
                    frame_index,
                    timestamp,
                    str(path),
                    "excluded_blurred",
                    sharpness,
                    None,
                )
            )
            continue
        small = cv2.resize(gray, (DUPLICATE_CHECK_SIZE, DUPLICATE_CHECK_SIZE))
        if (
            previous is not None
            and mean_abs_diff(small, previous) < duplicate_threshold
        ):
            records.append(
                FrameRecord(
                    frame_id_for(video_id, frame_index, timestamp),
                    video_id,
                    frame_index,
                    timestamp,
                    str(path),
                    "excluded_duplicate",
                    sharpness,
                    previous_id,
                )
            )
            continue
        records.append(
            FrameRecord(
                frame_id_for(video_id, frame_index, timestamp),
                video_id,
                frame_index,
                timestamp,
                str(path),
                "selected",
                sharpness,
                None,
            )
        )
        previous = small
        previous_id = records[-1].frame_id
    return tuple(records)


def write_outputs(
    records: tuple[FrameRecord, ...],
    manifest_path: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    """Write the frame manifest and extraction summary report."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FrameRecord.__dataclass_fields__)
        _ = writer.writeheader()
        _ = writer.writerows(
            {
                field: getattr(record, field)
                for field in FrameRecord.__dataclass_fields__
            }
            for record in records
        )
    counts = {
        "selected": 0,
        "excluded_blurred": 0,
        "excluded_duplicate": 0,
        "excluded_decode_failure": 0,
    }
    for record in records:
        counts[record.status] += 1
    lines = [
        "# Frame Extraction Report",
        "",
        f"Extraction tool: ffmpeg -vf fps={args.sample_rate},scale=384:384",
        f"Sample rate: {args.sample_rate} fps",
        f"Sharpness threshold: {args.sharpness_threshold}",
        f"Duplicate mean-abs-diff threshold: {args.duplicate_threshold}",
        f"Source: {args.source}",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        *[f"| {status} | {count} |" for status, count in counts.items()],
        "",
        "Occlusion exclusion requires the geometry model from a later ticket and is not detected yet.",
        "Event-centered windows will be added once CVAT annotations exist.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the frame extraction command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Dataset_From_JNMC"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/frame_manifest.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/frame-extraction.md")
    )
    parser.add_argument("--sample-rate", type=float, default=5.0)
    parser.add_argument("--sharpness-threshold", type=float, default=50.0)
    parser.add_argument("--duplicate-threshold", type=float, default=2.0)
    args = parser.parse_args()
    ffmpeg = find_ffmpeg()
    records: list[FrameRecord] = []
    for path in discover_videos(args.source):
        video_id = path.stem.lower()
        source_fps = source_fps_of(path)
        frames = extract_with_ffmpeg(
            ffmpeg, path, args.output_dir / video_id, args.sample_rate
        )
        records.extend(
            classify_frames(
                frames,
                video_id,
                source_fps,
                args.sample_rate,
                args.sharpness_threshold,
                args.duplicate_threshold,
            )
        )
    write_outputs(tuple(records), args.manifest, args.report, args)
    print(
        f"Extracted {sum(1 for r in records if r.status == 'selected')} frames; wrote {args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
