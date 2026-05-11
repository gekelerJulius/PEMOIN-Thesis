import numpy as np

from pemoin.data.contracts import SemanticSegment, SemanticsData
from pemoin.visualization.harmonized_overlay_grid import (
    _composite_grid_with_mask,
    _draw_projected_polyline_segments,
    _road_mask_from_semantics,
)


def test_grid_overlay_projection_uses_blender_convention(monkeypatch):
    captured = []

    def _capture(_image, polylines, isClosed, color, thickness, lineType):  # noqa: N803
        captured.append(np.asarray(polylines[0]).reshape(-1, 2))

    monkeypatch.setattr("pemoin.visualization.harmonized_overlay_grid.cv2.polylines", _capture)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    points_world = np.array(
        [
            [0.0, 1.0, -5.0],
            [0.0, 0.0, -5.0],
            [0.0, -1.0, -5.0],
        ],
        dtype=np.float32,
    )
    intrinsics = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    _draw_projected_polyline_segments(
        image,
        points_world,
        w2c=np.eye(4, dtype=np.float32),
        intrinsics=intrinsics,
        width=100,
        height=100,
        color_bgr=(0, 255, 0),
        thickness=1,
    )

    assert len(captured) == 1
    np.testing.assert_array_equal(
        captured[0],
        np.array(
            [
                [50, 30],
                [50, 50],
                [50, 70],
            ],
            dtype=np.int32,
        ),
    )


def test_composite_grid_with_mask_applies_grid_only_on_road_pixels():
    base = np.full((3, 3, 3), 10, dtype=np.uint8)
    grid = np.zeros_like(base)
    grid[0, 1] = np.array([0, 255, 0], dtype=np.uint8)
    grid[1, 1] = np.array([0, 200, 0], dtype=np.uint8)
    road_mask = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    output = _composite_grid_with_mask(base, grid, road_mask)

    np.testing.assert_array_equal(output[0, 1], grid[0, 1])
    np.testing.assert_array_equal(output[1, 1], base[1, 1])
    np.testing.assert_array_equal(output[2, 2], base[2, 2])


def test_road_mask_from_semantics_prefers_label_ids():
    label_ids = np.array(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=np.int32,
    )
    semantics = SemanticsData(
        frame_index=0,
        segments=[
            SemanticSegment(segment_id=11, label="road", score=1.0, mask=np.ones((2, 2), dtype=bool), label_id=1),
            SemanticSegment(segment_id=12, label="car", score=1.0, mask=np.ones((2, 2), dtype=bool), label_id=2),
        ],
        label_ids=label_ids,
        segment_ids=np.full((2, 2), 999, dtype=np.int32),
    )

    mask = _road_mask_from_semantics(semantics, road_labels=("road",))

    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        ),
    )


def test_road_mask_from_semantics_falls_back_to_segment_ids():
    segment_ids = np.array(
        [
            [10, 11],
            [11, 10],
        ],
        dtype=np.int32,
    )
    semantics = SemanticsData(
        frame_index=3,
        segments=[
            SemanticSegment(segment_id=10, label="road", score=1.0, mask=np.ones((2, 2), dtype=bool)),
            SemanticSegment(segment_id=11, label="person", score=1.0, mask=np.ones((2, 2), dtype=bool)),
        ],
        segment_ids=segment_ids,
    )

    mask = _road_mask_from_semantics(semantics, road_labels=("road",))

    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        ),
    )


def test_road_mask_from_semantics_returns_empty_when_no_matching_labels():
    semantics = SemanticsData(
        frame_index=5,
        segments=[
            SemanticSegment(segment_id=1, label="sidewalk", score=1.0, mask=np.ones((2, 2), dtype=bool), label_id=7),
        ],
        label_ids=np.full((2, 2), 7, dtype=np.int32),
    )

    mask = _road_mask_from_semantics(semantics, road_labels=("road",))

    assert not np.any(mask)


def test_composite_grid_with_mask_rejects_shape_mismatch():
    base = np.zeros((2, 2, 3), dtype=np.uint8)
    grid = np.zeros((2, 2, 3), dtype=np.uint8)
    road_mask = np.zeros((3, 2), dtype=bool)

    try:
        _composite_grid_with_mask(base, grid, road_mask)
    except ValueError as exc:
        assert "road_mask must match image spatial shape" in str(exc)
        assert "mask=(3, 2)" in str(exc)
        assert "image=(2, 2)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for road_mask shape mismatch.")
