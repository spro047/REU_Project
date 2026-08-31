from pathlib import Path

from scripts.audit_videos import (
    SegmentObservation,
    VideoAudit,
    orientation_for,
    sample_indices,
    write_outputs,
)


def test_sample_indices_are_deterministic_and_bounded() -> None:
    assert sample_indices(1) == (0,)
    assert sample_indices(4) == (0, 1, 2, 3)
    assert sample_indices(100)[0] == 0
    assert sample_indices(100)[-1] == 99


def test_orientation_uses_frame_dimensions() -> None:
    assert orientation_for(1920, 1080) == "landscape"
    assert orientation_for(1080, 1920) == "portrait"
    assert orientation_for(100, 100) == "square"


def test_write_outputs_preserves_audit_and_segment_fields(tmp_path: Path) -> None:
    audit = VideoAudit(
        "video-1",
        "video-1.MOV",
        "Dataset_From_JNMC/video-1.MOV",
        10,
        1920,
        1080,
        30.0,
        300,
        10.0,
        "landscape",
        True,
        "unknown",
        "unknown",
        2,
        0,
        "",
    )
    observation = SegmentObservation(
        "video-1", 0, 0.0, True, 120.0, 40.0, "usable sample"
    )
    manifest = tmp_path / "manifest.csv"
    segments = tmp_path / "segments.csv"
    report = tmp_path / "report.md"
    write_outputs((audit,), (observation,), manifest, segments, report)
    assert "video_id" in manifest.read_text(encoding="utf-8")
    assert "usable sample" in segments.read_text(encoding="utf-8")
    assert "video-1.MOV" in report.read_text(encoding="utf-8")
