from __future__ import annotations

import pytest

from pemoin.utils.geometry_validation import (
    GeometryValidationError,
    _validate_intrinsics_image_bounds,
)


def test_validate_intrinsics_image_bounds_allows_wide_unity_camera() -> None:
    _validate_intrinsics_image_bounds(
        fx=360.00003,
        fy=360.27338,
        cx=540.0,
        cy=219.0,
        shape=(438, 1080),
    )


def test_validate_intrinsics_image_bounds_rejects_implausibly_small_focal_length() -> None:
    with pytest.raises(GeometryValidationError, match="Unusual fx"):
        _validate_intrinsics_image_bounds(
            fx=200.0,
            fy=360.27338,
            cx=540.0,
            cy=219.0,
            shape=(438, 1080),
        )
