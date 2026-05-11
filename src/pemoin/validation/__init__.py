"""Validation modules for runtime data consistency."""

from __future__ import annotations

from typing import Any

from .policy import (
    AdaptiveValidationContext,
    ValidationPolicySettings,
    resolve_effective_sampling_fps,
)

__all__ = [
    "GeometryConsistencyValidationSettings",
    "GeometryConsistencyValidationResult",
    "ValidationPolicySettings",
    "AdaptiveValidationContext",
    "resolve_effective_sampling_fps",
    "validate_depth_pose_intrinsics_consistency",
]


def __getattr__(name: str) -> Any:
    if name in {
        "GeometryConsistencyValidationSettings",
        "GeometryConsistencyValidationResult",
        "validate_depth_pose_intrinsics_consistency",
    }:
        from .depth_pose_consistency import (
            GeometryConsistencyValidationResult,
            GeometryConsistencyValidationSettings,
            validate_depth_pose_intrinsics_consistency,
        )

        exports = {
            "GeometryConsistencyValidationSettings": GeometryConsistencyValidationSettings,
            "GeometryConsistencyValidationResult": GeometryConsistencyValidationResult,
            "validate_depth_pose_intrinsics_consistency": validate_depth_pose_intrinsics_consistency,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
