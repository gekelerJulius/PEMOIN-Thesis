from __future__ import annotations

import numpy as np
import pytest

from pemoin.data.contracts import ResourceKind, SemanticsData, SemanticSegment
from pemoin.utils.geometry_validation import (
    GeometryValidationConfig,
    GeometryValidationError,
    _validate_depth_consistency,
)


class _FakeStore:
    def __init__(self, semantics: SemanticsData | None) -> None:
        self._semantics = semantics

    def has(self, kind: ResourceKind) -> bool:
        return kind == ResourceKind.SEMANTICS_2D and self._semantics is not None

    def load_semantics2d(self, frame_index: int) -> SemanticsData:
        if self._semantics is None:
            raise RuntimeError("missing semantics")
        return self._semantics


def _identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _intrinsics(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [float(width), 0.0, float(width) / 2.0],
            [0.0, float(height), float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_depth_positive_ratio_uses_non_sky_mask_when_available() -> None:
    height, width = 20, 20
    depth = np.ones((height, width), dtype=np.float32)
    depth[:2, :] = 0.0
    label_ids = np.full((height, width), 2, dtype=np.int32)
    label_ids[:2, :] = 1
    semantics = SemanticsData(
        frame_index=0,
        segments=[
            SemanticSegment(segment_id=1, label="sky", score=1.0, mask=np.zeros((0, 0), dtype=bool), label_id=1),
            SemanticSegment(segment_id=2, label="road", score=1.0, mask=np.zeros((0, 0), dtype=bool), label_id=2),
        ],
        label_ids=label_ids,
        metadata={"semantic_roles": {"sky": ["sky"]}},
    )
    cfg = GeometryValidationConfig(min_positive_depth_ratio=0.95)
    result = _validate_depth_consistency(
        _FakeStore(semantics),
        depth,
        _identity_pose(),
        _identity_pose(),
        _intrinsics(width, height),
        (height, width),
        0,
        cfg,
        False,
    )
    assert result.positive_depth_ratio >= 0.95


def test_depth_positive_ratio_without_semantics_still_fails() -> None:
    height, width = 20, 20
    depth = np.ones((height, width), dtype=np.float32)
    depth[:2, :] = 0.0
    cfg = GeometryValidationConfig(min_positive_depth_ratio=0.95)
    with pytest.raises(GeometryValidationError, match="positive finite values within full_frame"):
        _validate_depth_consistency(
            _FakeStore(None),
            depth,
            _identity_pose(),
            _identity_pose(),
            _intrinsics(width, height),
            (height, width),
            0,
            cfg,
            False,
        )
