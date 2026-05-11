"""Canonical plane representation and operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    """Plane `n^T x + d = 0` with normalized `n`.

    `height_at_camera(camera_center)` is defined as:
    `n^T c + d`, i.e. signed camera-to-plane height along plane normal.
    """

    normal: np.ndarray
    offset: float

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-8:
            raise ValueError("Plane normal is degenerate.")
        object.__setattr__(self, "normal", (n / norm).astype(np.float32))
        object.__setattr__(self, "offset", float(self.offset) / norm)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be Nx3, got {pts.shape}.")
        return (pts @ self.normal + float(self.offset)).astype(np.float32)

    def height_at_camera(self, camera_center: np.ndarray) -> float:
        c = np.asarray(camera_center, dtype=np.float32).reshape(3)
        return float(self.normal @ c + float(self.offset))

    def enforce_normal_orientation(
        self,
        *,
        camera_center: np.ndarray,
        target_height_m: float,
    ) -> "Plane":
        """Flip sign if it better matches a target camera height."""
        current = self.height_at_camera(camera_center)
        flipped = -current
        if abs(flipped - target_height_m) < abs(current - target_height_m):
            return Plane(normal=-self.normal, offset=-float(self.offset))
        return self

    @classmethod
    def from_height_anchor(
        cls,
        *,
        normal: np.ndarray,
        camera_center: np.ndarray,
        plane_height_at_camera_m: float,
    ) -> "Plane":
        """Construct a plane with known `normal` and camera-height anchor."""
        n = np.asarray(normal, dtype=np.float32).reshape(3)
        c = np.asarray(camera_center, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-8:
            raise ValueError("Plane normal is degenerate.")
        n = n / norm
        d = float(plane_height_at_camera_m) - float(n @ c)
        return cls(normal=n, offset=d)

