"""Tests for road pixel selection and backprojection."""

import numpy as np
import pytest

from pemoin.data.contracts import ResourceStore, SemanticsAuxData
from pemoin.providers.geometry_fusion.utils.road_pixel_selection import (
    RoadPixelSelection,
    RoadPixelSelectionError,
    select_road_pixels,
)


def _make_mock_semantics(h: int, w: int, road_fraction: float = 0.5, frame_index: int = 0):
    """Create a mock SemanticsData with road labels in the bottom half."""
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class MockSegment:
        label_id: int
        label: str

    @dataclass
    class MockSemanticsData:
        frame_index: int
        label_ids: np.ndarray
        segment_ids: Optional[np.ndarray] = None
        segments: list = field(default_factory=list)
        metadata: dict = field(default_factory=dict)

    label_ids = np.zeros((h, w), dtype=np.int32)
    road_start = int(h * (1 - road_fraction))
    label_ids[road_start:, :] = 1  # Road class

    segments = [
        MockSegment(label_id=0, label="background"),
        MockSegment(label_id=1, label="road"),
    ]

    return MockSemanticsData(
        frame_index=frame_index,
        label_ids=label_ids,
        segments=segments,
    )


class TestSelectRoadPixels:
    def test_basic_selection(self, tmp_path):
        """Select road pixels from synthetic data."""
        store = ResourceStore("road_pixels_basic", root=tmp_path)
        h, w = 480, 640
        K = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Flat depth at 10m
        depth = np.full((h, w), 10.0, dtype=np.float32)
        semantics = _make_mock_semantics(h, w, road_fraction=0.5)

        result = select_road_pixels(
            resources=store,
            depth=depth,
            semantics=semantics,
            K=K,
            road_labels=("road",),
            conf_thresh=0.5,
            roi_bottom_frac=0.5,
            z_max_m=50.0,
            min_points=10,
        )

        assert isinstance(result, RoadPixelSelection)
        assert result.points_cam.shape[1] == 3
        assert result.weights.shape[0] == result.points_cam.shape[0]
        assert result.pixel_uv.shape[0] == result.points_cam.shape[0]
        assert result.points_cam.shape[0] > 0

    def test_depth_filtering(self, tmp_path):
        """Pixels with depth beyond z_max_m should be excluded."""
        store = ResourceStore("road_pixels_depth_filter", root=tmp_path)
        h, w = 100, 100
        K = np.array([
            [50.0, 0.0, 50.0],
            [0.0, 50.0, 50.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        depth = np.full((h, w), 100.0, dtype=np.float32)  # All beyond z_max
        semantics = _make_mock_semantics(h, w, road_fraction=1.0)

        with pytest.raises(RuntimeError, match="insufficient road pixels"):
            select_road_pixels(
                resources=store,
                depth=depth,
                semantics=semantics,
                K=K,
                road_labels=("road",),
                z_max_m=30.0,
                min_points=10,
            )

    def test_backprojection_correctness(self, tmp_path):
        """Verify that backprojected points use PEMOIN's Blender camera convention."""
        store = ResourceStore("road_pixels_backproject", root=tmp_path)
        h, w = 100, 100
        fx, fy, cx, cy = 50.0, 50.0, 50.0, 50.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        depth = np.full((h, w), 5.0, dtype=np.float32)
        semantics = _make_mock_semantics(h, w, road_fraction=1.0)

        result = select_road_pixels(
            resources=store,
            depth=depth,
            semantics=semantics,
            K=K,
            road_labels=("road",),
            conf_thresh=0.5,
            roi_bottom_frac=1.0,
            z_max_m=10.0,
            min_points=1,
        )

        # All Z should be -depth in Blender convention.
        np.testing.assert_allclose(result.points_cam[:, 2], -5.0, atol=1e-5)
        # Points at (cx, cy) should have X=0, Y=0
        center_mask = (result.pixel_uv[:, 0] == cx) & (result.pixel_uv[:, 1] == cy)
        if np.any(center_mask):
            center_pts = result.points_cam[center_mask]
            np.testing.assert_allclose(center_pts[:, 0], 0.0, atol=1e-5)
            np.testing.assert_allclose(center_pts[:, 1], 0.0, atol=1e-5)

        # Bottom-image road pixels should lie below the camera => negative Y in Blender convention.
        bottom_idx = int(np.argmax(result.pixel_uv[:, 1]))
        assert float(result.points_cam[bottom_idx, 1]) < 0.0

    def test_label_resolution_error_reports_configured_and_available_labels(self, tmp_path):
        store = ResourceStore("road_pixels_label_error", root=tmp_path)
        h, w = 32, 32
        K = np.array([
            [16.0, 0.0, 16.0],
            [0.0, 16.0, 16.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        semantics = _make_mock_semantics(h, w, road_fraction=1.0)
        semantics.segments[1].label = "roads"

        with pytest.raises(RoadPixelSelectionError, match="configured road_labels=\\['road'\\]"):
            select_road_pixels(
                resources=store,
                depth=depth,
                semantics=semantics,
                K=K,
                road_labels=("road",),
                conf_thresh=0.5,
                roi_bottom_frac=1.0,
                z_max_m=10.0,
                min_points=1,
            )

    def test_segformer_probability_artifact_alias_is_accepted(self, tmp_path):
        store = ResourceStore("road_pixels_prob_alias", root=tmp_path)
        h, w = 32, 32
        K = np.array([
            [16.0, 0.0, 16.0],
            [0.0, 16.0, 16.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        semantics = _make_mock_semantics(h, w, road_fraction=1.0)
        probs = np.zeros((2, h, w), dtype=np.float32)
        probs[1] = 1.0
        store.save_semantics_aux(
            SemanticsAuxData(
                frame_index=0,
                class_probabilities=probs,
                class_ids=np.array([0, 1], dtype=np.int32),
                metadata={"source": "unit-test"},
            )
        )

        result = select_road_pixels(
            resources=store,
            depth=depth,
            semantics=semantics,
            K=K,
            road_labels=("road",),
            conf_thresh=0.5,
            roi_bottom_frac=1.0,
            z_max_m=10.0,
            min_points=1,
        )

        assert isinstance(result, RoadPixelSelection)
        assert result.points_cam.shape[0] > 0
