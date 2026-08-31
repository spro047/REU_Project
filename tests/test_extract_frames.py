import subprocess
from pathlib import Path

import numpy as np

from scripts.extract_frames import (
    classify_frames,
    extract_with_ffmpeg,
    find_ffmpeg,
    frame_id_for,
    sharpness_of,
    source_fps_of,
    source_frame_index_for,
)


def write_synthetic_video(
    path: Path, duration: float = 2.0, rate: float = 30.0
) -> None:
    subprocess.run(
        [
            find_ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=64x64:rate={rate}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_source_frame_index_maps_extracted_to_source() -> None:
    assert source_frame_index_for(0, 30.0, 5.0) == 0
    assert source_frame_index_for(1, 30.0, 5.0) == 6
    assert source_frame_index_for(9, 30.0, 5.0) == 54


def test_sharpness_ranks_noise_above_uniform() -> None:
    noise = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    uniform = np.full((64, 64), 128, dtype=np.uint8)
    assert sharpness_of(noise) > sharpness_of(uniform)


def test_extract_is_traceable_and_regenerable(tmp_path: Path) -> None:
    video = tmp_path / "video-1.mp4"
    write_synthetic_video(video)
    source_fps = source_fps_of(video)
    first = classify_frames(
        extract_with_ffmpeg(find_ffmpeg(), video, tmp_path / "frames", 5.0),
        "video-1",
        source_fps,
        5.0,
        1.0,
        2.0,
    )
    second = classify_frames(
        extract_with_ffmpeg(find_ffmpeg(), video, tmp_path / "frames2", 5.0),
        "video-1",
        source_fps,
        5.0,
        1.0,
        2.0,
    )
    selected = [record for record in first if record.status == "selected"]
    assert len(selected) == 10
    assert all(
        record.frame_id
        == frame_id_for("video-1", record.source_frame_index, record.timestamp_seconds)
        for record in selected
    )
    assert [(r.frame_id, r.source_frame_index, r.status) for r in first] == [
        (r.frame_id, r.source_frame_index, r.status) for r in second
    ]
