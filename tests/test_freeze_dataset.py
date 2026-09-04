from scripts.freeze_dataset import (
    DatasetRow,
    FrameRow,
    assign_splits,
    build_rows,
)


def sample_videos() -> tuple[str, ...]:
    return (
        "video-1",
        "video-10",
        "video-2",
        "video-3",
        "video-5",
        "video-6",
        "video-7",
        "video-8",
        "video-9",
    )


def test_assign_splits_is_deterministic_and_complete() -> None:
    first = assign_splits(sample_videos(), 5, 2, 2)
    second = assign_splits(sample_videos(), 5, 2, 2)
    assert first == second
    assert sorted(first) == sorted(sample_videos())
    assert len([s for s in first.values() if s == "train"]) == 5
    assert len([s for s in first.values() if s == "val"]) == 2
    assert len([s for s in first.values() if s == "test"]) == 2


def test_build_rows_links_annotations_and_split() -> None:
    frames = (
        FrameRow(
            "video-1",
            "video-1__f000000__t0.000",
            0.0,
            "data/frames/video-1/frame_000001.png",
            "selected",
        ),
        FrameRow(
            "video-1",
            "video-1__f000006__t0.200",
            0.2,
            "data/frames/video-1/frame_000002.png",
            "selected",
        ),
        FrameRow(
            "video-1",
            "video-1__f000012__t0.400",
            0.4,
            "data/frames/video-1/frame_000003.png",
            "excluded_blurred",
        ),
    )
    shapes = {"frame_000001.png": 2, "frame_000003.png": 0}
    rows = build_rows(frames, shapes, {"video-1": "train"})
    assert rows[0].split == "train"
    assert rows[0].annotated is True
    assert rows[1].annotated is False
    assert rows[2].annotated is False
    assert isinstance(rows[0], DatasetRow)
