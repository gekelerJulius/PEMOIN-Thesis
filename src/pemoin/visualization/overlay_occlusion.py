"""Depth-aware pedestrian occlusion helpers for overlay compositing."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - some Blender envs may not provide cv2
    cv2 = None

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover - blender env may not provide imageio
    imageio = None


@dataclass(frozen=True)
class EdgeTreatmentSettings:
    """Controls boundary-only alpha and color treatment for inserted pedestrians."""

    enabled: bool = True
    boundary_band_px: int = 4
    feather_radius_px: float = 2.0
    feather_strength: float = 0.35
    blur_enabled: bool = True
    blur_radius_px: float = 1.5
    blur_strength: float = 0.25
    despill_enabled: bool = True
    despill_strength: float = 0.25
    regrain_enabled: bool = True
    regrain_strength: float = 0.12
    tiny_object_disable_feather: bool = True
    tiny_object_disable_blur: bool = True
    tiny_object_disable_despill: bool = True
    tiny_object_disable_regrain: bool = True
    tiny_object_max_boundary_fraction: float = 0.25
    tiny_object_disable_all_below_short_side_px: int = 20
    tiny_object_disable_all_below_visible_pixels: int = 256
    disable_when_boundary_fraction_above: float = 0.6


@dataclass(frozen=True)
class TemporalOcclusionSettings:
    """Temporal stabilization controls for borderline occlusion frames."""

    enabled: bool = True
    base_hysteresis_margin_m: float = 0.02
    state_flip_persist_frames: int = 2
    edge_exit_hold_frames: int = 2
    max_single_frame_visible_area_drop_ratio: float = 0.5


@dataclass(frozen=True)
class OcclusionSettings:
    """Controls per-pixel pedestrian-vs-scene depth occlusion."""

    default_front_margin_m: float = 0.03
    relative_margin: float = 0.01
    alpha_presence_threshold: float = 0.05
    alpha_visible_threshold: float = 0.05
    contact_plane_band_m: float = 0.025
    contact_patch_radius_m: float = 0.30
    contact_coplanar_tolerance_m: float = 0.03
    edge_treatment: EdgeTreatmentSettings = field(default_factory=EdgeTreatmentSettings)
    temporal_stabilization: TemporalOcclusionSettings = field(
        default_factory=TemporalOcclusionSettings
    )


@dataclass(frozen=True)
class OcclusionFrameDiagnostics:
    frame_index: int
    pedestrian_pixels: int
    visible_pixels: int
    occluded_pixels: int
    visible_ratio: float
    min_scene_depth_m: float | None
    max_scene_depth_m: float | None
    min_ped_depth_m: float | None
    max_ped_depth_m: float | None
    median_depth_margin_m: float | None
    ped_depth_mode: str = "per_pixel"
    support_depth_m: float | None = None
    semantics_available: bool = False
    traversable_ground_pixels: int = 0
    contact_candidate_pixels: int = 0
    contact_override_pixels: int = 0
    ground_exempt_candidate_pixels: int = 0
    ground_exempt_pixels: int = 0
    boundary_pixels: int = 0
    feathered_pixels: int = 0
    despill_pixels: int = 0
    blurred_pixels: int = 0
    regrained_pixels: int = 0
    estimated_background_noise_sigma: float | None = None


@dataclass(frozen=True)
class EdgeTreatmentDebug:
    boundary_mask: np.ndarray
    outer_ring_mask: np.ndarray
    pre_composite_rgb: np.ndarray
    post_composite_rgb: np.ndarray
    strength_heatmap: np.ndarray


@dataclass(frozen=True)
class TinyVisibleStats:
    short_side_px: int
    long_side_px: int
    pixel_count: int


@dataclass
class TemporalOcclusionState:
    previous_visible_mask: np.ndarray | None = None
    pending_visible_mask: np.ndarray | None = None
    pending_frames: int = 0
    edge_hold_remaining: int = 0


def _kernel_size_from_radius(radius_px: float) -> int:
    radius = int(np.ceil(max(0.0, float(radius_px))))
    return max(1, radius * 2 + 1)


def _ellipse_kernel(radius_px: float) -> np.ndarray:
    if cv2 is not None:
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_kernel_size_from_radius(radius_px), _kernel_size_from_radius(radius_px)),
        )
    size = _kernel_size_from_radius(radius_px)
    yy, xx = np.indices((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    radius = max(center, 1.0)
    mask = ((xx - center) ** 2 + (yy - center) ** 2) <= radius**2
    return mask.astype(np.uint8)


def _dilate_mask(mask: np.ndarray, radius_px: float) -> np.ndarray:
    src = np.asarray(mask, dtype=np.uint8)
    if src.size == 0:
        return np.asarray(mask, dtype=bool)
    kernel = _ellipse_kernel(radius_px)
    if cv2 is not None:
        return cv2.dilate(src, kernel, iterations=1) > 0
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(src, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    out = np.zeros_like(src, dtype=bool)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            window = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            out[y, x] = bool(np.any(window[kernel > 0]))
    return out


def _erode_mask(mask: np.ndarray, radius_px: float) -> np.ndarray:
    src = np.asarray(mask, dtype=np.uint8)
    if src.size == 0:
        return np.asarray(mask, dtype=bool)
    kernel = _ellipse_kernel(radius_px)
    if cv2 is not None:
        return cv2.erode(src, kernel, iterations=1) > 0
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(src, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    out = np.zeros_like(src, dtype=bool)
    kernel_count = int(np.count_nonzero(kernel))
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            window = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            out[y, x] = bool(np.count_nonzero(window[kernel > 0]) == kernel_count)
    return out


def _gaussian_blur(image: np.ndarray, radius_px: float) -> np.ndarray:
    sigma = max(float(radius_px), 0.0)
    if sigma <= 1e-6:
        return np.asarray(image, dtype=np.float32)
    arr = np.asarray(image, dtype=np.float32)
    if cv2 is not None:
        return cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    kernel_size = _kernel_size_from_radius(radius_px)
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords**2) / max(2.0 * sigma * sigma, 1e-6))
    kernel /= np.sum(kernel)
    out = arr.copy()
    for axis in (0, 1):
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis, out)
    return out.astype(np.float32)


def _estimate_background_noise_sigma(background_rgb: np.ndarray, ring_mask: np.ndarray) -> float:
    ring = np.asarray(ring_mask, dtype=bool)
    if not np.any(ring):
        return 0.0
    bg = np.asarray(background_rgb, dtype=np.float32)
    luminance = 0.2126 * bg[:, :, 0] + 0.7152 * bg[:, :, 1] + 0.0722 * bg[:, :, 2]
    smooth = _gaussian_blur(luminance, 1.0)
    residual = (luminance - smooth)[ring]
    if residual.size == 0:
        return 0.0
    mad = float(np.median(np.abs(residual - np.median(residual))))
    sigma = 1.4826 * mad
    return max(0.0, sigma)


def _unpremultiply_rgb(premul_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgb = np.zeros_like(premul_rgb, dtype=np.float32)
    valid = alpha > 1e-6
    rgb[valid] = premul_rgb[valid] / alpha[valid, None]
    return np.clip(rgb, 0.0, 255.0)


def _prepare_outer_ring(mask: np.ndarray, boundary_radius: int) -> np.ndarray:
    outer = _dilate_mask(mask, max(1, boundary_radius)) & (~mask)
    if np.any(outer):
        return outer
    return _dilate_mask(mask, max(2, boundary_radius * 2)) & (~mask)


def _visible_mask_stats(mask: np.ndarray) -> TinyVisibleStats | None:
    visible = np.asarray(mask, dtype=bool)
    if not np.any(visible):
        return None
    ys, xs = np.where(visible)
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    short_side = min(height, width)
    long_side = max(height, width)
    return TinyVisibleStats(
        short_side_px=int(short_side),
        long_side_px=int(long_side),
        pixel_count=int(np.count_nonzero(visible)),
    )


def _apply_boundary_edge_treatment(
    *,
    background_rgb: np.ndarray,
    pedestrian_rgb: np.ndarray,
    visible_alpha: np.ndarray,
    settings: EdgeTreatmentSettings,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, EdgeTreatmentDebug | None, dict[str, float]]:
    bg = np.asarray(background_rgb, dtype=np.float32)
    ped_rgb = np.asarray(pedestrian_rgb, dtype=np.float32)
    alpha = np.clip(np.asarray(visible_alpha, dtype=np.float32), 0.0, 1.0)
    pre_composite = ped_rgb * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])
    visible_mask = alpha > 1e-6
    if (not settings.enabled) or (not np.any(visible_mask)):
        return pre_composite, alpha, None, {
            "boundary_pixels": 0.0,
            "feathered_pixels": 0.0,
            "despill_pixels": 0.0,
            "blurred_pixels": 0.0,
            "regrained_pixels": 0.0,
            "estimated_background_noise_sigma": 0.0,
        }

    mask_stats = _visible_mask_stats(visible_mask)
    boundary_radius = max(1, int(settings.boundary_band_px))
    tiny_object_mode = False
    if mask_stats is not None and mask_stats.short_side_px > 0:
        max_boundary_fraction = float(
            np.clip(settings.tiny_object_max_boundary_fraction, 0.0, 1.0)
        )
        max_boundary_radius = max(
            1,
            int(np.floor(float(mask_stats.short_side_px) * max_boundary_fraction)),
        )
        if boundary_radius > max_boundary_radius:
            boundary_radius = max_boundary_radius
        if (
            mask_stats.short_side_px <= int(settings.tiny_object_disable_all_below_short_side_px)
            or mask_stats.pixel_count <= int(settings.tiny_object_disable_all_below_visible_pixels)
            or mask_stats.short_side_px <= 4
            or mask_stats.pixel_count <= 16
        ):
            tiny_object_mode = True
    inner_mask = _erode_mask(visible_mask, boundary_radius)
    boundary_mask = visible_mask & (~inner_mask)
    if not np.any(boundary_mask):
        return pre_composite, alpha, None, {
            "boundary_pixels": 0.0,
            "feathered_pixels": 0.0,
            "despill_pixels": 0.0,
            "blurred_pixels": 0.0,
            "regrained_pixels": 0.0,
            "estimated_background_noise_sigma": 0.0,
        }
    boundary_fraction = float(np.count_nonzero(boundary_mask)) / float(
        max(int(np.count_nonzero(visible_mask)), 1)
    )
    if boundary_fraction > float(settings.disable_when_boundary_fraction_above):
        return pre_composite, alpha, None, {
            "boundary_pixels": float(np.count_nonzero(boundary_mask)),
            "feathered_pixels": 0.0,
            "despill_pixels": 0.0,
            "blurred_pixels": 0.0,
            "regrained_pixels": 0.0,
            "estimated_background_noise_sigma": 0.0,
        }
    outer_ring = _prepare_outer_ring(visible_mask, boundary_radius)
    boundary_weight = boundary_mask.astype(np.float32)
    strength_heatmap = np.zeros_like(alpha, dtype=np.float32)

    treated_alpha = alpha.copy()
    feathered_pixels = 0
    feather_enabled = (
        settings.feather_radius_px > 0.0
        and settings.feather_strength > 0.0
        and not (tiny_object_mode and settings.tiny_object_disable_feather)
    )
    if feather_enabled:
        blurred_alpha = _gaussian_blur(alpha, settings.feather_radius_px)
        feather_alpha = (1.0 - float(settings.feather_strength)) * treated_alpha + float(
            settings.feather_strength
        ) * blurred_alpha
        feather_alpha = np.clip(feather_alpha, 0.0, alpha)
        treated_alpha[boundary_mask] = feather_alpha[boundary_mask]
        feathered_pixels = int(np.count_nonzero(boundary_mask))
        strength_heatmap[boundary_mask] = np.maximum(
            strength_heatmap[boundary_mask], float(settings.feather_strength)
        )

    ring_rgb = bg[outer_ring]
    if ring_rgb.size == 0:
        ring_rgb = bg[~visible_mask]
    bg_mean = (
        np.mean(ring_rgb, axis=0).astype(np.float32)
        if ring_rgb.size > 0
        else np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    )
    noise_sigma = _estimate_background_noise_sigma(bg, outer_ring)

    treated_rgb = ped_rgb.copy()
    despill_pixels = 0
    despill_enabled = (
        settings.despill_enabled
        and settings.despill_strength > 0.0
        and not (tiny_object_mode and settings.tiny_object_disable_despill)
    )
    if despill_enabled:
        edge_alpha_weight = np.clip(1.0 - treated_alpha, 0.0, 1.0)
        despill_weight = (
            boundary_weight
            * np.clip(0.35 + 0.65 * edge_alpha_weight, 0.0, 1.0)
            * float(settings.despill_strength)
        )
        treated_rgb = treated_rgb * (1.0 - despill_weight[:, :, None]) + bg_mean[None, None, :] * despill_weight[
            :, :, None
        ]
        despill_pixels = int(np.count_nonzero(despill_weight > 1e-5))
        strength_heatmap = np.maximum(strength_heatmap, despill_weight)

    treated_premul = treated_rgb * treated_alpha[:, :, None]
    blurred_pixels = 0
    blur_enabled = (
        settings.blur_enabled
        and settings.blur_radius_px > 0.0
        and settings.blur_strength > 0.0
        and not (tiny_object_mode and settings.tiny_object_disable_blur)
    )
    if blur_enabled:
        if cv2 is not None:
            filtered = cv2.bilateralFilter(
                treated_premul.astype(np.float32),
                d=-1,
                sigmaColor=max(4.0, 16.0 * float(settings.blur_radius_px)),
                sigmaSpace=max(1.0, 2.0 * float(settings.blur_radius_px)),
            )
        else:
            filtered = _gaussian_blur(treated_premul, settings.blur_radius_px)
        blur_weight = boundary_weight * float(settings.blur_strength)
        treated_premul = treated_premul * (1.0 - blur_weight[:, :, None]) + filtered * blur_weight[:, :, None]
        blurred_pixels = int(np.count_nonzero(blur_weight > 1e-5))
        strength_heatmap = np.maximum(strength_heatmap, blur_weight)

    regrained_pixels = 0
    regrain_enabled = (
        settings.regrain_enabled
        and settings.regrain_strength > 0.0
        and noise_sigma > 1e-6
        and not (tiny_object_mode and settings.tiny_object_disable_regrain)
    )
    if regrain_enabled:
        rng = np.random.default_rng(int(random_seed))
        grain = rng.normal(0.0, noise_sigma, size=alpha.shape).astype(np.float32)
        grain_weight = boundary_weight * float(settings.regrain_strength)
        grain_rgb = grain[:, :, None] * grain_weight[:, :, None]
        treated_premul = np.clip(
            treated_premul + grain_rgb * treated_alpha[:, :, None],
            0.0,
            255.0,
        )
        regrained_pixels = int(np.count_nonzero(grain_weight > 1e-5))
        strength_heatmap = np.maximum(strength_heatmap, grain_weight)

    treated_rgb = _unpremultiply_rgb(treated_premul, treated_alpha)
    post_composite = treated_premul + bg * (1.0 - treated_alpha[:, :, None])
    debug = EdgeTreatmentDebug(
        boundary_mask=boundary_mask,
        outer_ring_mask=outer_ring,
        pre_composite_rgb=np.clip(np.rint(pre_composite), 0.0, 255.0).astype(np.uint8),
        post_composite_rgb=np.clip(np.rint(post_composite), 0.0, 255.0).astype(np.uint8),
        strength_heatmap=np.clip(strength_heatmap, 0.0, 1.0),
    )
    return post_composite, treated_alpha, debug, {
        "boundary_pixels": float(np.count_nonzero(boundary_mask)),
        "feathered_pixels": float(feathered_pixels),
        "despill_pixels": float(despill_pixels),
        "blurred_pixels": float(blurred_pixels),
        "regrained_pixels": float(regrained_pixels),
        "estimated_background_noise_sigma": float(noise_sigma),
    }


def load_depth_npz(path: Path) -> np.ndarray:
    """Load depth array from an NPZ file with `depth` key."""
    if not path.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "depth" not in data.files:
            raise ValueError(f"Depth file missing `depth` key: {path}")
        depth = np.asarray(data["depth"], dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got {depth.shape} at {path}")
    return depth


def load_mask_png(path: Path, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a single-channel mask PNG as boolean."""
    if not path.exists():
        raise FileNotFoundError(f"Occlusion mask not found: {path}")
    mask_img: np.ndarray
    if imageio is not None:
        mask_img = np.asarray(imageio.imread(path))
    else:  # pragma: no cover
        try:
            import bpy  # type: ignore
        except Exception as exc:
            raise RuntimeError("imageio unavailable and bpy not found for PNG loading.") from exc
        image = bpy.data.images.load(str(path))
        try:
            width, height = int(image.size[0]), int(image.size[1])
            rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))
        finally:
            bpy.data.images.remove(image)
        rgba = np.flipud(rgba)
        mask_img = np.clip(np.rint(rgba[:, :, 0] * 255.0), 0.0, 255.0).astype(np.uint8)
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    mask = np.asarray(mask_img, dtype=np.uint8) > 0
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"Mask shape mismatch for {path}: got {mask.shape}, expected {expected_shape}."
        )
    return mask


def compute_visible_pedestrian_mask(
    *,
    ped_alpha: np.ndarray,
    ped_depth_m: np.ndarray,
    scene_depth_m: np.ndarray,
    settings: OcclusionSettings,
    ped_world_points: np.ndarray | None = None,
    traversable_ground_mask: np.ndarray | None = None,
    support_anchor_world: np.ndarray | None = None,
    support_plane_normal: np.ndarray | None = None,
    support_plane_offset: float | None = None,
) -> tuple[np.ndarray, OcclusionFrameDiagnostics]:
    """Compute where pedestrian pixels are visible in front of scene depth."""
    alpha = np.asarray(ped_alpha, dtype=np.float32)
    ped_depth = np.asarray(ped_depth_m, dtype=np.float32)
    scene_depth = np.asarray(scene_depth_m, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError(f"ped_alpha must be 2D, got {alpha.shape}.")
    if ped_depth.shape != alpha.shape:
        raise ValueError(
            f"ped_depth shape {ped_depth.shape} must match ped_alpha shape {alpha.shape}."
        )
    if scene_depth.shape != alpha.shape:
        raise ValueError(
            f"scene_depth shape {scene_depth.shape} must match ped_alpha shape {alpha.shape}."
        )
    if ped_world_points is not None and tuple(ped_world_points.shape) != tuple(alpha.shape) + (3,):
        raise ValueError(
            "ped_world_points must be HxWx3 matching ped_alpha, got "
            f"{ped_world_points.shape} for alpha {alpha.shape}."
        )
    if traversable_ground_mask is not None and traversable_ground_mask.shape != alpha.shape:
        raise ValueError(
            "traversable_ground_mask shape "
            f"{traversable_ground_mask.shape} must match ped_alpha shape {alpha.shape}."
        )

    ped_presence = alpha > float(settings.alpha_presence_threshold)
    ped_count = int(np.count_nonzero(ped_presence))

    valid_scene = np.isfinite(scene_depth) & (scene_depth > 0.0)
    valid_ped = np.isfinite(ped_depth) & (ped_depth > 0.0)

    invalid_scene_on_ped = ped_presence & (~valid_scene)
    if np.any(invalid_scene_on_ped):
        idx = np.argwhere(invalid_scene_on_ped)[0]
        raise ValueError(
            "Scene depth invalid at pedestrian pixels. "
            f"First invalid pixel (v={int(idx[0])}, u={int(idx[1])})."
        )

    margin = np.maximum(
        float(settings.default_front_margin_m),
        float(settings.relative_margin) * scene_depth,
    )
    strict_visible = valid_ped & (ped_depth < (scene_depth - margin))
    visible = ped_presence & strict_visible

    if traversable_ground_mask is None:
        raise ValueError(
            "Traversable-ground semantics are required for pedestrian overlay occlusion."
        )
    semantics_available = True
    traversable_ground_pixels = 0
    contact_candidate_pixels = 0
    contact_override_pixels = 0
    ground_exempt_candidate_pixels = 0
    ground_exempt_pixels = 0

    if (
        ped_world_points is not None
        and support_anchor_world is not None
        and support_plane_normal is not None
        and support_plane_offset is not None
    ):
        ground_mask = np.asarray(traversable_ground_mask, dtype=bool)
        traversable_ground_pixels = int(np.count_nonzero(ped_presence & ground_mask))
        normal = np.asarray(support_plane_normal, dtype=np.float32).reshape(3)
        n_norm = float(np.linalg.norm(normal))
        if n_norm > 1e-8:
            normal = normal / n_norm
            anchor = np.asarray(support_anchor_world, dtype=np.float32).reshape(3)
            world_valid = np.isfinite(ped_world_points).all(axis=2)
            signed_plane_dist = np.sum(
                np.asarray(ped_world_points, dtype=np.float32) * normal[None, None, :],
                axis=2,
            ) + float(support_plane_offset)
            xy_dist = np.linalg.norm(
                np.asarray(ped_world_points[:, :, :2], dtype=np.float32) - anchor[:2][None, None, :],
                axis=2,
            )
            contact_candidates = (
                ped_presence
                & ground_mask
                & valid_ped
                & world_valid
                & (np.abs(signed_plane_dist) <= float(settings.contact_plane_band_m))
                & (xy_dist <= float(settings.contact_patch_radius_m))
            )
            contact_candidate_pixels = int(np.count_nonzero(contact_candidates))
            contact_visible = (
                contact_candidates
                & valid_scene
                & (ped_depth <= (scene_depth + float(settings.contact_coplanar_tolerance_m)))
            )
            visible = visible | contact_visible
            contact_override_pixels = int(np.count_nonzero(contact_visible & (~strict_visible)))
    else:
        traversable_ground_pixels = int(
            np.count_nonzero(ped_presence & np.asarray(traversable_ground_mask, dtype=bool))
        )

    ground_mask = np.asarray(traversable_ground_mask, dtype=bool)
    ground_exempt_candidates = ped_presence & ground_mask & valid_ped
    ground_exempt_candidate_pixels = int(np.count_nonzero(ground_exempt_candidates))
    before_ground_exempt = visible.copy()
    visible = visible | ground_exempt_candidates
    ground_exempt_pixels = int(
        np.count_nonzero(ground_exempt_candidates & (~before_ground_exempt))
    )

    visible_count = int(np.count_nonzero(visible))
    occluded_count = int(np.count_nonzero(ped_presence & (~visible)))

    ped_scene = scene_depth[ped_presence]
    ped_depth_vals = ped_depth[ped_presence]
    ped_margin_vals = margin[ped_presence]

    diag = OcclusionFrameDiagnostics(
        frame_index=-1,
        pedestrian_pixels=ped_count,
        visible_pixels=visible_count,
        occluded_pixels=occluded_count,
        visible_ratio=(0.0 if ped_count == 0 else float(visible_count) / float(ped_count)),
        min_scene_depth_m=(None if ped_count == 0 else float(np.min(ped_scene))),
        max_scene_depth_m=(None if ped_count == 0 else float(np.max(ped_scene))),
        min_ped_depth_m=(None if ped_count == 0 else float(np.min(ped_depth_vals))),
        max_ped_depth_m=(None if ped_count == 0 else float(np.max(ped_depth_vals))),
        median_depth_margin_m=(None if ped_count == 0 else float(np.median(ped_margin_vals))),
        semantics_available=bool(semantics_available),
        traversable_ground_pixels=int(traversable_ground_pixels),
        contact_candidate_pixels=int(contact_candidate_pixels),
        contact_override_pixels=int(contact_override_pixels),
        ground_exempt_candidate_pixels=int(ground_exempt_candidate_pixels),
        ground_exempt_pixels=int(ground_exempt_pixels),
    )
    return visible, diag


def _touches_image_edge(mask: np.ndarray) -> bool:
    visible = np.asarray(mask, dtype=bool)
    if not np.any(visible):
        return False
    return bool(
        np.any(visible[0, :])
        or np.any(visible[-1, :])
        or np.any(visible[:, 0])
        or np.any(visible[:, -1])
    )


def _stabilize_visible_mask_temporally(
    *,
    visible_mask: np.ndarray,
    ambiguous_mask: np.ndarray,
    ped_presence: np.ndarray,
    settings: TemporalOcclusionSettings,
    temporal_state: TemporalOcclusionState | None,
) -> np.ndarray:
    current_presence = np.asarray(ped_presence, dtype=bool)
    current = np.asarray(visible_mask, dtype=bool) & current_presence
    if (
        temporal_state is None
        or not settings.enabled
        or temporal_state.previous_visible_mask is None
        or temporal_state.previous_visible_mask.shape != current.shape
    ):
        if temporal_state is not None:
            temporal_state.previous_visible_mask = current.copy()
            temporal_state.pending_visible_mask = None
            temporal_state.pending_frames = 0
            temporal_state.edge_hold_remaining = 0
        return current

    previous = np.asarray(temporal_state.previous_visible_mask, dtype=bool) & current_presence
    prev_visible = int(np.count_nonzero(previous))
    current_visible = int(np.count_nonzero(current))
    if prev_visible == 0 and current_visible > 0:
        temporal_state.previous_visible_mask = current.copy()
        temporal_state.pending_visible_mask = None
        temporal_state.pending_frames = 0
        temporal_state.edge_hold_remaining = 0
        return current
    hold_due_to_edge_exit = (
        prev_visible > 0
        and _touches_image_edge(previous)
        and current_visible
        < int(round((1.0 - float(settings.max_single_frame_visible_area_drop_ratio)) * prev_visible))
    )
    if hold_due_to_edge_exit:
        temporal_state.edge_hold_remaining = max(
            int(temporal_state.edge_hold_remaining),
            int(settings.edge_exit_hold_frames),
        )
    if temporal_state.edge_hold_remaining > 0 and current_visible < prev_visible:
        temporal_state.edge_hold_remaining -= 1
        temporal_state.previous_visible_mask = previous.copy()
        temporal_state.pending_visible_mask = None
        temporal_state.pending_frames = 0
        return previous.copy()
    temporal_state.previous_visible_mask = current.copy()
    temporal_state.pending_visible_mask = None
    temporal_state.pending_frames = 0
    return current


def compose_depth_occluded_rgba(
    *,
    background_rgb: np.ndarray,
    pedestrian_rgba: np.ndarray,
    scene_depth_m: np.ndarray,
    ped_depth_m: np.ndarray,
    settings: OcclusionSettings,
    ped_world_points: np.ndarray | None = None,
    traversable_ground_mask: np.ndarray | None = None,
    support_anchor_world: np.ndarray | None = None,
    support_plane_normal: np.ndarray | None = None,
    support_plane_offset: float | None = None,
    temporal_state: TemporalOcclusionState | None = None,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, OcclusionFrameDiagnostics]:
    """Return composited RGB, visible-ped mask, and diagnostics."""
    bg = np.asarray(background_rgb, dtype=np.float32)
    ped = np.asarray(pedestrian_rgba, dtype=np.float32)
    if bg.ndim != 3 or bg.shape[2] < 3:
        raise ValueError(f"background_rgb must be HxWx3, got {bg.shape}")
    if ped.ndim != 3 or ped.shape[2] < 4:
        raise ValueError(f"pedestrian_rgba must be HxWx4, got {ped.shape}")
    if bg.shape[:2] != ped.shape[:2]:
        raise ValueError(f"Shape mismatch bg={bg.shape[:2]} ped={ped.shape[:2]}")

    alpha = np.clip(ped[:, :, 3] / 255.0, 0.0, 1.0)
    visible_mask, diag = compute_visible_pedestrian_mask(
        ped_alpha=alpha,
        ped_depth_m=ped_depth_m,
        scene_depth_m=scene_depth_m,
        settings=settings,
        ped_world_points=ped_world_points,
        traversable_ground_mask=traversable_ground_mask,
        support_anchor_world=support_anchor_world,
        support_plane_normal=support_plane_normal,
        support_plane_offset=support_plane_offset,
    )
    ped_presence = alpha > float(settings.alpha_presence_threshold)
    valid_scene = np.isfinite(scene_depth_m) & (scene_depth_m > 0.0)
    valid_ped = np.isfinite(ped_depth_m) & (ped_depth_m > 0.0)
    margin = np.maximum(
        float(settings.default_front_margin_m),
        float(settings.relative_margin) * np.asarray(scene_depth_m, dtype=np.float32),
    )
    signed_front_margin = (
        np.asarray(scene_depth_m, dtype=np.float32)
        - np.asarray(ped_depth_m, dtype=np.float32)
        - margin
    )
    ambiguous_mask = (
        ped_presence
        & valid_scene
        & valid_ped
        & (np.abs(signed_front_margin) <= float(settings.temporal_stabilization.base_hysteresis_margin_m))
    )
    visible_mask = _stabilize_visible_mask_temporally(
        visible_mask=visible_mask,
        ambiguous_mask=ambiguous_mask,
        ped_presence=ped_presence,
        settings=settings.temporal_stabilization,
        temporal_state=temporal_state,
    )

    pedestrian_pixels = int(np.count_nonzero(ped_presence))
    effective_alpha = alpha * visible_mask.astype(np.float32)
    out_rgb, _, _, edge_stats = _apply_boundary_edge_treatment(
        background_rgb=bg[:, :, :3],
        pedestrian_rgb=ped[:, :, :3],
        visible_alpha=effective_alpha,
        settings=settings.edge_treatment,
        random_seed=int(random_seed),
    )
    out_rgb_u8 = np.clip(np.rint(out_rgb), 0.0, 255.0).astype(np.uint8)
    visible_binary = effective_alpha > float(settings.alpha_visible_threshold)
    visible_pixels = int(np.count_nonzero(visible_binary))
    diag = OcclusionFrameDiagnostics(
        **{
            **asdict(diag),
            "visible_pixels": int(visible_pixels),
            "occluded_pixels": int(max(pedestrian_pixels - visible_pixels, 0)),
            "visible_ratio": (
                0.0
                if pedestrian_pixels == 0
                else float(visible_pixels) / float(pedestrian_pixels)
            ),
            "boundary_pixels": int(edge_stats["boundary_pixels"]),
            "feathered_pixels": int(edge_stats["feathered_pixels"]),
            "despill_pixels": int(edge_stats["despill_pixels"]),
            "blurred_pixels": int(edge_stats["blurred_pixels"]),
            "regrained_pixels": int(edge_stats["regrained_pixels"]),
            "estimated_background_noise_sigma": float(edge_stats["estimated_background_noise_sigma"]),
        }
    )
    return out_rgb_u8, visible_binary, diag


def write_occlusion_mask_png(path: Path, visible_mask: np.ndarray) -> None:
    mask = np.asarray(visible_mask, dtype=bool)
    image = (mask.astype(np.uint8) * 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.imwrite(path, image)
        return
    try:  # pragma: no cover
        import bpy  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio unavailable and bpy not found for PNG writing.") from exc
    h, w = image.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    val = image.astype(np.float32) / 255.0
    rgba[:, :, 0] = val
    rgba[:, :, 1] = val
    rgba[:, :, 2] = val
    rgba[:, :, 3] = 1.0
    out = bpy.data.images.new(name=f"occ_mask_{path.stem}", width=w, height=h, alpha=True)
    try:
        out.pixels = np.flipud(rgba).reshape(-1).tolist()
        out.filepath_raw = str(path)
        out.file_format = "PNG"
        out.save()
    finally:
        bpy.data.images.remove(out)


def write_occlusion_debug_png(
    path: Path,
    background_rgb: np.ndarray,
    visible_mask: np.ndarray,
    *,
    contact_candidate_mask: np.ndarray | None = None,
    contact_override_mask: np.ndarray | None = None,
    occluded_mask: np.ndarray | None = None,
) -> None:
    bg = np.asarray(background_rgb, dtype=np.uint8)
    mask = np.asarray(visible_mask, dtype=bool)
    if bg.ndim != 3 or bg.shape[2] < 3:
        raise ValueError(f"background_rgb must be HxWx3, got {bg.shape}")
    debug = bg[:, :, :3].copy()
    if contact_candidate_mask is not None:
        candidates = np.asarray(contact_candidate_mask, dtype=bool)
        debug[candidates] = np.asarray([255, 210, 80], dtype=np.uint8)
    if occluded_mask is not None:
        occluded = np.asarray(occluded_mask, dtype=bool)
        debug[occluded] = np.asarray([220, 70, 70], dtype=np.uint8)
    debug[mask] = np.asarray([40, 220, 80], dtype=np.uint8)
    if contact_override_mask is not None:
        overrides = np.asarray(contact_override_mask, dtype=bool)
        debug[overrides] = np.asarray([80, 220, 255], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.imwrite(path, debug)
        return
    try:  # pragma: no cover
        import bpy  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio unavailable and bpy not found for PNG writing.") from exc
    h, w = debug.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = debug.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    out = bpy.data.images.new(name=f"occ_debug_{path.stem}", width=w, height=h, alpha=True)
    try:
        out.pixels = np.flipud(rgba).reshape(-1).tolist()
        out.filepath_raw = str(path)
        out.file_format = "PNG"
        out.save()
    finally:
        bpy.data.images.remove(out)


def write_edge_treatment_debug_artifacts(
    path: Path,
    background_rgb: np.ndarray,
    debug: EdgeTreatmentDebug,
) -> None:
    bg = np.asarray(background_rgb, dtype=np.uint8)
    boundary_mask = np.asarray(debug.boundary_mask, dtype=bool)
    outer_ring_mask = np.asarray(debug.outer_ring_mask, dtype=bool)
    strength = np.clip(np.asarray(debug.strength_heatmap, dtype=np.float32), 0.0, 1.0)

    boundary_vis = bg.copy()
    boundary_vis[outer_ring_mask] = np.asarray([255, 210, 80], dtype=np.uint8)
    boundary_vis[boundary_mask] = np.asarray([80, 220, 255], dtype=np.uint8)

    diff = np.mean(
        np.abs(
            np.asarray(debug.post_composite_rgb, dtype=np.float32)
            - np.asarray(debug.pre_composite_rgb, dtype=np.float32)
        ),
        axis=2,
    )
    diff_norm = np.clip(diff / max(float(np.max(diff)), 1.0), 0.0, 1.0)
    compare = bg.astype(np.float32)
    compare[:, :, 0] = np.clip(compare[:, :, 0] + 180.0 * diff_norm, 0.0, 255.0)
    compare[:, :, 1] = np.clip(compare[:, :, 1] * (1.0 - 0.35 * diff_norm), 0.0, 255.0)
    compare[:, :, 2] = np.clip(compare[:, :, 2] * (1.0 - 0.55 * diff_norm), 0.0, 255.0)
    compare = compare.astype(np.uint8)

    heat = np.zeros_like(bg, dtype=np.uint8)
    heat[:, :, 0] = np.clip(np.rint(strength * 255.0), 0.0, 255.0).astype(np.uint8)
    heat[:, :, 1] = np.clip(np.rint(strength * 170.0), 0.0, 255.0).astype(np.uint8)

    for suffix, image in (
        ("_edge_boundary.png", boundary_vis),
        ("_edge_compare.png", compare),
        ("_edge_strength.png", heat),
    ):
        target = path.with_name(f"{path.stem}{suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if imageio is not None:
            imageio.imwrite(target, image)
            continue
        try:  # pragma: no cover
            import bpy  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("imageio unavailable and bpy not found for PNG writing.") from exc
        h, w = image.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = image.astype(np.float32) / 255.0
        rgba[:, :, 3] = 1.0
        out = bpy.data.images.new(name=f"edge_debug_{target.stem}", width=w, height=h, alpha=True)
        try:
            out.pixels = np.flipud(rgba).reshape(-1).tolist()
            out.filepath_raw = str(target)
            out.file_format = "PNG"
            out.save()
        finally:
            bpy.data.images.remove(out)


def write_occlusion_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Sequence[OcclusionFrameDiagnostics],
) -> tuple[Path, Path]:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "occlusion_diagnostics.json"
    csv_path = vis_dir / "occlusion_diagnostics.csv"

    total_ped = int(sum(int(d.pedestrian_pixels) for d in diagnostics))
    total_visible = int(sum(int(d.visible_pixels) for d in diagnostics))
    total_occluded = int(sum(int(d.occluded_pixels) for d in diagnostics))
    visible_ratio = 0.0 if total_ped == 0 else float(total_visible) / float(total_ped)

    payload = {
        "frames": len(diagnostics),
        "total_pedestrian_pixels": total_ped,
        "total_visible_pixels": total_visible,
        "total_occluded_pixels": total_occluded,
        "visible_ratio": visible_ratio,
        "per_frame": [asdict(d) for d in diagnostics],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame_index",
                "pedestrian_pixels",
                "visible_pixels",
                "occluded_pixels",
                "visible_ratio",
                "min_scene_depth_m",
                "max_scene_depth_m",
                "min_ped_depth_m",
                "max_ped_depth_m",
                "median_depth_margin_m",
                "ped_depth_mode",
                "support_depth_m",
                "semantics_available",
                "traversable_ground_pixels",
                "contact_candidate_pixels",
                "contact_override_pixels",
                "ground_exempt_candidate_pixels",
                "ground_exempt_pixels",
                "boundary_pixels",
                "feathered_pixels",
                "despill_pixels",
                "blurred_pixels",
                "regrained_pixels",
                "estimated_background_noise_sigma",
            )
        )
        for d in diagnostics:
            writer.writerow(
                (
                    int(d.frame_index),
                    int(d.pedestrian_pixels),
                    int(d.visible_pixels),
                    int(d.occluded_pixels),
                    float(d.visible_ratio),
                    "" if d.min_scene_depth_m is None else float(d.min_scene_depth_m),
                    "" if d.max_scene_depth_m is None else float(d.max_scene_depth_m),
                    "" if d.min_ped_depth_m is None else float(d.min_ped_depth_m),
                    "" if d.max_ped_depth_m is None else float(d.max_ped_depth_m),
                    "" if d.median_depth_margin_m is None else float(d.median_depth_margin_m),
                    str(d.ped_depth_mode),
                    "" if d.support_depth_m is None else float(d.support_depth_m),
                    int(bool(d.semantics_available)),
                    int(d.traversable_ground_pixels),
                    int(d.contact_candidate_pixels),
                    int(d.contact_override_pixels),
                    int(d.ground_exempt_candidate_pixels),
                    int(d.ground_exempt_pixels),
                    int(d.boundary_pixels),
                    int(d.feathered_pixels),
                    int(d.despill_pixels),
                    int(d.blurred_pixels),
                    int(d.regrained_pixels),
                    "" if d.estimated_background_noise_sigma is None else float(d.estimated_background_noise_sigma),
                )
            )
    return json_path, csv_path
