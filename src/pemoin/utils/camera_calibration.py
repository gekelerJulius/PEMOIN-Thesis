from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_INTRINSICS_NORMALIZATION_VERSION = 1
DEFAULT_BLENDER_SENSOR_WIDTH_MM = 36.0
DEFAULT_BLENDER_SENSOR_HEIGHT_MM = 24.0
DEFAULT_BLENDER_MAX_FOCAL_RESIDUAL_PX = 1.0
DEFAULT_BLENDER_MAX_PRINCIPAL_POINT_RESIDUAL_PX = 0.5


class IntrinsicsValidationError(ValueError):
    """Raised when pinhole intrinsics metadata or geometry is inconsistent."""


class BlenderCameraParityError(ValueError):
    """Raised when Blender camera settings cannot reproduce the target intrinsics."""


@dataclass(frozen=True)
class ResolvedIntrinsicsResolution:
    width: float
    height: float
    source: str
    heuristic_used: bool


@dataclass(frozen=True)
class BlenderCameraSolution:
    sensor_fit: str
    lens_mm: float
    shift_x: float
    shift_y: float
    sensor_width_mm: float
    sensor_height_mm: float
    pixel_aspect_x: float
    pixel_aspect_y: float
    effective_matrix: np.ndarray
    focal_residual_px: float
    principal_point_residual_px: float

    def diagnostics_payload(self, target_matrix: np.ndarray) -> dict[str, Any]:
        target = np.asarray(target_matrix, dtype=np.float64)
        effective = np.asarray(self.effective_matrix, dtype=np.float64)
        diff = effective - target
        return {
            "sensor_fit": self.sensor_fit,
            "lens_mm": float(self.lens_mm),
            "shift_x": float(self.shift_x),
            "shift_y": float(self.shift_y),
            "sensor_width_mm": float(self.sensor_width_mm),
            "sensor_height_mm": float(self.sensor_height_mm),
            "pixel_aspect_x": float(self.pixel_aspect_x),
            "pixel_aspect_y": float(self.pixel_aspect_y),
            "target_matrix": target.tolist(),
            "effective_matrix": effective.tolist(),
            "matrix_delta": diff.tolist(),
            "focal_residual_px": float(self.focal_residual_px),
            "principal_point_residual_px": float(self.principal_point_residual_px),
        }


def _coerce_resolution_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    first = float(value[0])
    second = float(value[1])
    if first <= 0.0 or second <= 0.0:
        return None
    return first, second


def _coerce_positive_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def resolve_intrinsics_resolution(
    metadata: Mapping[str, object] | None,
    matrix: np.ndarray,
    *,
    frame_shape: Sequence[int] | None = None,
    allow_principal_point_fallback: bool = True,
) -> ResolvedIntrinsicsResolution:
    info = metadata or {}
    width = _coerce_positive_float(info.get("width"))
    height = _coerce_positive_float(info.get("height"))
    if width is not None and height is not None:
        return ResolvedIntrinsicsResolution(
            width=width,
            height=height,
            source="metadata.width_height",
            heuristic_used=False,
        )

    for key in ("resolution", "working_resolution", "reference_resolution"):
        pair = _coerce_resolution_pair(info.get(key))
        if pair is not None:
            pair_height, pair_width = pair
            return ResolvedIntrinsicsResolution(
                width=pair_width,
                height=pair_height,
                source=f"metadata.{key}",
                heuristic_used=False,
            )

    if frame_shape is not None and len(frame_shape) >= 2:
        frame_height = float(frame_shape[0])
        frame_width = float(frame_shape[1])
        if frame_height > 0.0 and frame_width > 0.0:
            return ResolvedIntrinsicsResolution(
                width=frame_width,
                height=frame_height,
                source="frame_shape",
                heuristic_used=False,
            )

    if not allow_principal_point_fallback:
        raise IntrinsicsValidationError(
            "Intrinsics metadata is missing explicit dimensions and principal-point fallback is disabled."
        )

    width = float(np.asarray(matrix, dtype=np.float64)[0, 2] * 2.0)
    height = float(np.asarray(matrix, dtype=np.float64)[1, 2] * 2.0)
    if width <= 0.0 or height <= 0.0:
        raise IntrinsicsValidationError(
            "Unable to infer intrinsics dimensions from the principal point fallback."
        )
    return ResolvedIntrinsicsResolution(
        width=width,
        height=height,
        source="principal_point_fallback",
        heuristic_used=True,
    )


def validate_and_normalize_intrinsics(
    matrix: np.ndarray,
    metadata: Mapping[str, object] | None,
    *,
    frame_shape: Sequence[int] | None = None,
    allow_principal_point_fallback: bool = True,
    fail_on_heuristic: bool = False,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    intrinsics = np.asarray(matrix, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise IntrinsicsValidationError(
            f"Intrinsics matrix must have shape (3, 3); received {intrinsics.shape!r}."
        )
    if not np.all(np.isfinite(intrinsics)):
        raise IntrinsicsValidationError("Intrinsics matrix contains non-finite values.")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise IntrinsicsValidationError(
            f"Intrinsics focal lengths must be positive; received fx={fx:.6f}, fy={fy:.6f}."
        )

    info = dict(metadata or {})
    resolved = resolve_intrinsics_resolution(
        info,
        intrinsics,
        frame_shape=frame_shape,
        allow_principal_point_fallback=allow_principal_point_fallback,
    )
    width = int(round(resolved.width))
    height = int(round(resolved.height))
    if width <= 0 or height <= 0:
        raise IntrinsicsValidationError(
            f"Resolved intrinsics dimensions must be positive; received {width}x{height}."
        )

    if fail_on_heuristic and resolved.heuristic_used:
        raise IntrinsicsValidationError(
            "Intrinsics relied on principal-point dimension fallback while strict validation is enabled."
        )

    # Validate any preexisting explicit fields against the resolved dimensions before canonicalizing.
    explicit_width = _coerce_positive_float(info.get("width"))
    explicit_height = _coerce_positive_float(info.get("height"))
    if explicit_width is not None and int(round(explicit_width)) != width:
        raise IntrinsicsValidationError(
            f"Intrinsics width metadata mismatch: resolved {width}, explicit {explicit_width}."
        )
    if explicit_height is not None and int(round(explicit_height)) != height:
        raise IntrinsicsValidationError(
            f"Intrinsics height metadata mismatch: resolved {height}, explicit {explicit_height}."
        )

    for key in ("resolution", "working_resolution", "reference_resolution"):
        pair = _coerce_resolution_pair(info.get(key))
        if pair is None:
            continue
        pair_height, pair_width = pair
        if int(round(pair_width)) != width or int(round(pair_height)) != height:
            raise IntrinsicsValidationError(
                f"Intrinsics metadata field '{key}' does not match resolved dimensions {width}x{height}."
            )

    if frame_shape is not None and len(frame_shape) >= 2:
        frame_height = int(frame_shape[0])
        frame_width = int(frame_shape[1])
        if frame_width != width or frame_height != height:
            raise IntrinsicsValidationError(
                "Intrinsics dimensions do not match the associated frame shape: "
                f"intrinsics={width}x{height}, frame={frame_width}x{frame_height}."
            )

    # Principal point is allowed to lie slightly outside the image due to float rounding, but not materially.
    if cx < -0.5 or cx > float(width) + 0.5:
        raise IntrinsicsValidationError(
            f"Principal point cx={cx:.6f} lies outside the resolved width {width}."
        )
    if cy < -0.5 or cy > float(height) + 0.5:
        raise IntrinsicsValidationError(
            f"Principal point cy={cy:.6f} lies outside the resolved height {height}."
        )

    canonical = dict(info)
    canonical["width"] = width
    canonical["height"] = height
    canonical["resolution"] = [height, width]
    canonical["reference_resolution"] = [height, width]
    canonical["working_resolution"] = [height, width]
    canonical.setdefault("input_resolution", [float(height), float(width)])
    canonical["intrinsics_resolution_source"] = resolved.source
    canonical["intrinsics_resolution_was_heuristic"] = bool(resolved.heuristic_used)
    canonical["intrinsics_normalization_version"] = int(
        canonical.get("intrinsics_normalization_version", DEFAULT_INTRINSICS_NORMALIZATION_VERSION)
    )
    canonical["intrinsics_validation_passed"] = True
    report = {
        "width": width,
        "height": height,
        "resolution_source": resolved.source,
        "heuristic_used": bool(resolved.heuristic_used),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    return intrinsics, canonical, report


def _normalize_pixel_aspect_ratio(ratio: float) -> tuple[float, float]:
    if ratio <= 0.0 or not np.isfinite(ratio):
        raise BlenderCameraParityError(
            f"Pixel-aspect ratio must be positive and finite; received {ratio!r}."
        )
    if ratio >= 1.0:
        return 1.0, ratio
    return 1.0 / ratio, 1.0


def effective_intrinsics_from_blender(
    *,
    width: int,
    height: int,
    lens_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    sensor_fit: str,
    shift_x: float,
    shift_y: float,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise BlenderCameraParityError(
            f"Render dimensions must be positive; received {width}x{height}."
        )
    if lens_mm <= 0.0:
        raise BlenderCameraParityError(f"Camera lens must be positive; received {lens_mm!r}.")
    ratio = float(pixel_aspect_y) / float(pixel_aspect_x)
    sensor_fit_upper = str(sensor_fit).upper()
    if sensor_fit_upper == "VERTICAL":
        if sensor_height_mm <= 0.0:
            raise BlenderCameraParityError(
                f"Camera sensor_height must be positive; received {sensor_height_mm!r}."
            )
        fx = float(lens_mm) * ratio * float(height) / float(sensor_height_mm)
        fy = float(lens_mm) * float(height) / float(sensor_height_mm)
        cx = float(width) * 0.5 - float(shift_x) * ratio * float(height)
        cy = float(height) * 0.5 + float(shift_y) * float(height)
    else:
        if sensor_width_mm <= 0.0:
            raise BlenderCameraParityError(
                f"Camera sensor_width must be positive; received {sensor_width_mm!r}."
            )
        fx = float(lens_mm) * float(width) / float(sensor_width_mm)
        fy = fx / ratio
        cx = float(width) * 0.5 - float(shift_x) * float(width)
        cy = float(height) * 0.5 + float(shift_y) * float(width) / ratio
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def solve_blender_camera_for_intrinsics(
    target_matrix: np.ndarray,
    *,
    width: int,
    height: int,
    sensor_width_mm: float = DEFAULT_BLENDER_SENSOR_WIDTH_MM,
    sensor_height_mm: float = DEFAULT_BLENDER_SENSOR_HEIGHT_MM,
    max_focal_residual_px: float = DEFAULT_BLENDER_MAX_FOCAL_RESIDUAL_PX,
    max_principal_point_residual_px: float = DEFAULT_BLENDER_MAX_PRINCIPAL_POINT_RESIDUAL_PX,
) -> BlenderCameraSolution:
    target = np.asarray(target_matrix, dtype=np.float32)
    if target.shape != (3, 3):
        raise BlenderCameraParityError(
            f"Target intrinsics matrix must have shape (3, 3); received {target.shape!r}."
        )
    fx = float(target[0, 0])
    fy = float(target[1, 1])
    cx = float(target[0, 2])
    cy = float(target[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise BlenderCameraParityError(
            f"Target intrinsics focal lengths must be positive; received fx={fx:.6f}, fy={fy:.6f}."
        )

    aspect_ratio = fx / fy
    pixel_aspect_x, pixel_aspect_y = _normalize_pixel_aspect_ratio(aspect_ratio)
    ratio = pixel_aspect_y / pixel_aspect_x
    candidates: list[BlenderCameraSolution] = []

    horizontal_lens = fx * float(sensor_width_mm) / float(width)
    horizontal_shift_x = (float(width) * 0.5 - cx) / float(width)
    horizontal_shift_y = (cy - float(height) * 0.5) * ratio / float(width)
    horizontal_effective = effective_intrinsics_from_blender(
        width=width,
        height=height,
        lens_mm=horizontal_lens,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        sensor_fit="HORIZONTAL",
        shift_x=horizontal_shift_x,
        shift_y=horizontal_shift_y,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )
    candidates.append(
        _build_blender_candidate(
            sensor_fit="HORIZONTAL",
            lens_mm=horizontal_lens,
            shift_x=horizontal_shift_x,
            shift_y=horizontal_shift_y,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
            effective=horizontal_effective,
            target=target,
        )
    )

    vertical_lens = fy * float(sensor_height_mm) / float(height)
    vertical_shift_x = (float(width) * 0.5 - cx) / (ratio * float(height))
    vertical_shift_y = (cy - float(height) * 0.5) / float(height)
    vertical_effective = effective_intrinsics_from_blender(
        width=width,
        height=height,
        lens_mm=vertical_lens,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        sensor_fit="VERTICAL",
        shift_x=vertical_shift_x,
        shift_y=vertical_shift_y,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )
    candidates.append(
        _build_blender_candidate(
            sensor_fit="VERTICAL",
            lens_mm=vertical_lens,
            shift_x=vertical_shift_x,
            shift_y=vertical_shift_y,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
            effective=vertical_effective,
            target=target,
        )
    )

    best = min(
        candidates,
        key=lambda item: (
            item.focal_residual_px + item.principal_point_residual_px,
            0 if (width >= height and item.sensor_fit == "HORIZONTAL") else 1,
        ),
    )
    if best.focal_residual_px > max_focal_residual_px:
        raise BlenderCameraParityError(
            "Blender camera focal parity failed: "
            f"residual={best.focal_residual_px:.6f}px threshold={max_focal_residual_px:.6f}px."
        )
    if best.principal_point_residual_px > max_principal_point_residual_px:
        raise BlenderCameraParityError(
            "Blender camera principal-point parity failed: "
            f"residual={best.principal_point_residual_px:.6f}px "
            f"threshold={max_principal_point_residual_px:.6f}px."
        )
    return best


def _build_blender_candidate(
    *,
    sensor_fit: str,
    lens_mm: float,
    shift_x: float,
    shift_y: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
    effective: np.ndarray,
    target: np.ndarray,
) -> BlenderCameraSolution:
    focal_residual_px = float(
        max(
            abs(float(effective[0, 0]) - float(target[0, 0])),
            abs(float(effective[1, 1]) - float(target[1, 1])),
        )
    )
    principal_point_residual_px = float(
        max(
            abs(float(effective[0, 2]) - float(target[0, 2])),
            abs(float(effective[1, 2]) - float(target[1, 2])),
        )
    )
    return BlenderCameraSolution(
        sensor_fit=sensor_fit,
        lens_mm=float(lens_mm),
        shift_x=float(shift_x),
        shift_y=float(shift_y),
        sensor_width_mm=float(sensor_width_mm),
        sensor_height_mm=float(sensor_height_mm),
        pixel_aspect_x=float(pixel_aspect_x),
        pixel_aspect_y=float(pixel_aspect_y),
        effective_matrix=np.asarray(effective, dtype=np.float32),
        focal_residual_px=focal_residual_px,
        principal_point_residual_px=principal_point_residual_px,
    )
