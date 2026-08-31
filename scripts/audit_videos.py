# /// script
# dependencies = ["opencv-python>=4.10"]
# ///

"""Audit fixed-camera source videos and write reproducible manifests.

Run with: uv run scripts/audit_videos.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2

VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mov", ".mp4", ".m4v", ".avi"})
SAMPLE_COUNT: Final[int] = 20


@dataclass(frozen=True, slots=True)
class SegmentObservation:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    readable: bool
    brightness: float | None
    sharpness: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class VideoAudit:
    video_id: str
    filename: str
    source_path: str
    file_size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    orientation: str
    readable: bool
    operator_id: str
    stitch_count_status: str
    usable_segment_count: int
    excluded_segment_count: int
    notes: str


def video_id_for(path: Path) -> str:
    """Return a stable video ID based on the filename stem."""
    return path.stem.lower()


def orientation_for(width: int, height: int) -> str:
    """Classify a frame orientation without assuming camera calibration."""
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def sample_indices(frame_count: int) -> tuple[int, ...]:
    """Return deterministic frame indices spread across a video."""
    if frame_count <= 0:
        return ()
    sample_count = min(SAMPLE_COUNT, frame_count)
    return (
        tuple(
            sorted(
                {
                    round(index * (frame_count - 1) / (sample_count - 1))
                    for index in range(sample_count)
                }
            )
        )
        if sample_count > 1
        else (0,)
    )


def observe_video(path: Path) -> tuple[VideoAudit, tuple[SegmentObservation, ...]]:
    """Read metadata and deterministic frame samples from one video."""
    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    observations: list[SegmentObservation] = []
    for frame_index in sample_indices(frame_count):
        _ = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        is_read, frame = capture.read()
        timestamp = frame_index / fps if fps > 0 else 0.0
        if not is_read:
            observations.append(
                SegmentObservation(
                    video_id_for(path),
                    frame_index,
                    timestamp,
                    False,
                    None,
                    None,
                    "frame could not be decoded",
                )
            )
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        observations.append(
            SegmentObservation(
                video_id_for(path),
                frame_index,
                timestamp,
                True,
                float(gray.mean()),
                float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "usable sample",
            )
        )
    _ = capture.release()
    readable = bool(
        width > 0
        and height > 0
        and fps > 0
        and frame_count > 0
        and observations
        and all(item.readable for item in observations)
    )
    usable_count = sum(item.readable for item in observations)
    excluded_count = len(observations) - usable_count
    notes = "" if readable else "metadata or sampled-frame read failure"
    audit = VideoAudit(
        video_id_for(path),
        path.name,
        str(path),
        path.stat().st_size,
        width,
        height,
        fps,
        frame_count,
        duration,
        orientation_for(width, height),
        readable,
        "unknown",
        "unknown",
        usable_count,
        excluded_count,
        notes,
    )
    return audit, tuple(observations)


def discover_videos(source_dir: Path) -> tuple[Path, ...]:
    """Find supported video files in deterministic filename order."""
    return tuple(
        sorted(
            (
                path
                for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )
    )


def write_outputs(
    audits: tuple[VideoAudit, ...],
    observations: tuple[SegmentObservation, ...],
    manifest_path: Path,
    segments_path: Path,
    report_path: Path,
) -> None:
    """Write machine-readable manifests and a human-readable audit report."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VideoAudit.__dataclass_fields__)
        _ = writer.writeheader()
        _ = writer.writerows(
            {field: getattr(audit, field) for field in VideoAudit.__dataclass_fields__}
            for audit in audits
        )
    with segments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SegmentObservation.__dataclass_fields__
        )
        _ = writer.writeheader()
        _ = writer.writerows(
            {
                field: getattr(item, field)
                for field in SegmentObservation.__dataclass_fields__
            }
            for item in observations
        )
    total_frames = sum(audit.frame_count for audit in audits)
    lines = [
        "# Video Audit Report",
        "",
        f"Audited videos: {len(audits)}",
        f"Total source frames reported: {total_frames}",
        "",
        "## Video inventory",
        "",
        "| Video | Resolution | FPS | Duration (s) | Readable | Sampled usable | Sampled excluded | Operator | Stitch count |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {audit.filename} | {audit.width}x{audit.height} | {audit.fps:.3f} | {audit.duration_seconds:.2f} | {'yes' if audit.readable else 'no'} | {audit.usable_segment_count} | {audit.excluded_segment_count} | {audit.operator_id} | {audit.stitch_count_status} |"
        for audit in audits
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Sampled frames are deterministic observations spread across each video. Operator IDs and stitch counts are marked unknown because no source metadata was provided. Usability here means that metadata is valid and every sampled frame decoded; clinical quality and annotation suitability require expert review.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the video audit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Dataset_From_JNMC"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/video_manifest.csv")
    )
    parser.add_argument(
        "--segments", type=Path, default=Path("data/manifests/video_segments.csv")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/video-audit.md"))
    args = parser.parse_args()
    audits: list[VideoAudit] = []
    observations: list[SegmentObservation] = []
    for path in discover_videos(args.source):
        audit, sampled = observe_video(path)
        audits.append(audit)
        observations.extend(sampled)
    write_outputs(
        tuple(audits), tuple(observations), args.manifest, args.segments, args.report
    )
    print(f"Audited {len(audits)} videos; wrote {args.manifest} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
