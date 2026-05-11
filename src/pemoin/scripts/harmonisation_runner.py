"""Run Harmonizer on occlusion-aware overlay frames and masks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

def _ensure_harmonizer_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    harmonizer_root = repo_root / "tools" / "Harmonizer"
    if str(harmonizer_root) not in sys.path:
        sys.path.insert(0, str(harmonizer_root))


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


class HarmonizerProtocol(Protocol):
    def predict_arguments(
        self, comp_tensor: torch.Tensor, mask_tensor: torch.Tensor
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor:
        ...

    def restore_image(
        self,
        comp_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        arguments: torch.Tensor,
    ) -> list[torch.Tensor]:
        ...


@dataclass(frozen=True, slots=True)
class CropBounds:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_xyxy(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True, slots=True)
class FrameCropDiagnostics:
    frame_index: int
    status: str
    image_size: list[int]
    mask_pixel_count: int
    bbox_xyxy: list[int] | None
    expanded_bbox_xyxy: list[int] | None
    crop_xyxy: list[int] | None
    crop_size: list[int] | None
    bbox_expansion_scale: float
    min_crop_size_ratio: float
    model_ran: bool
    max_frame_coverage_ratio: float | None = None
    containment_margin_px: int | None = None
    crop_coverage_ratio: float | None = None
    actor_fully_contained_in_crop: bool | None = None
    visible_mask_outside_crop_pixels: int = 0
    crop_insufficient_reason: str | None = None
    oversized_actor_behavior_applied: str | None = None
    full_frame_fallback_used: bool = False
    learned_model_eligible: bool | None = None
    eligibility_reason: str | None = None
    fallback_reason: str | None = None
    color_match_applied: bool = False
    color_match_skip_reason: str | None = None
    ring_pixel_count_raw: int = 0
    ring_pixel_count_filtered: int = 0
    sky_filtered_pixel_count: int = 0
    outlier_filtered_pixel_count: int = 0
    foreground_pixel_count: int = 0
    color_match_debug: dict[str, Any] | None = None
    temporal_smoothing_applied: bool = False
    temporal_reset_applied: bool = False
    temporal_reset_reason: str | None = None
    raw_harmonizer_arguments: dict[str, float] | None = None
    smoothed_harmonizer_arguments: dict[str, float] | None = None
    raw_color_match_parameters: dict[str, float] | None = None
    smoothed_color_match_parameters: dict[str, float] | None = None
    fallback_used: bool = False
    fallback_mode: str | None = None
    fallback_transform: dict[str, list[float]] | None = None
    crop_iou_with_previous: float | None = None
    mask_area_ratio_with_previous: float | None = None
    centroid_jump_fraction: float | None = None
    pre_harmonization_masked_luma: float | None = None
    post_harmonization_masked_luma: float | None = None
    local_ring_luma: float | None = None
    rejected_candidate_masked_luma: float | None = None
    postcheck_rejected: bool = False
    postcheck_reason: str | None = None
    recovery_mode: str | None = None
    recovery_strength: float | None = None
    recovery_continuity_bound_source: str | None = None
    span_id: int | None = None
    span_policy: str | None = None
    appearance_application_mode: str | None = None
    reference_frame_for_track: int | None = None
    propagated_parameters_used: bool = False
    track_downgrade_reason: str | None = None
    track_id: int | None = None
    used_for_reference_estimation: bool = False
    applied_parameter_source: str | None = None
    adaptive_effect_strength: float | None = None
    adaptive_color_match_strength: float | None = None
    adaptive_harmonizer_strength: float | None = None
    adaptive_preserve_contrast: float | None = None
    adaptive_preserve_saturation: float | None = None
    adaptive_local_support_ratio: float | None = None
    adaptive_tiny_subject_score: float | None = None
    adaptive_synthetic_scene_score: float | None = None


@dataclass(frozen=True, slots=True)
class ColorMatchSettings:
    enabled: bool = False
    color_space: str = "lab"
    ring_inner_px: int = 10
    ring_outer_px: int = 40
    exclude_top_band: bool = True
    top_band_reference: str = "mask_top"
    top_band_px: int = 12
    use_semantics_for_sky_filter: bool = True
    outlier_rejection: str = "robust_percentile"
    luminance_match: str = "mean_std"
    luminance_strength: float = 0.60
    chroma_match: str = "mean_only"
    chroma_strength: float = 0.30
    prefer_pedestrian_reference: bool = True
    pedestrian_reference_weight: float = 0.65
    fallback_scene_reference_weight: float = 0.35
    saturation_attenuation_strength: float = 0.35
    contrast_attenuation_strength: float = 0.25
    min_pedestrian_reference_pixels: int = 48
    min_ring_pixels: int = 256
    fallback_behavior: str = "skip_and_continue"
    write_diagnostics: bool = True


@dataclass(frozen=True, slots=True)
class ColorMatchDiagnostics:
    applied: bool
    skip_reason: str | None
    ring_pixel_count_raw: int
    ring_pixel_count_filtered: int
    sky_filtered_pixel_count: int
    outlier_filtered_pixel_count: int
    foreground_pixel_count: int
    debug: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TemporalSmoothingSettings:
    enabled: bool = True
    mode: str = "parameter_ema"
    appearance_alpha: float = 0.85
    tonal_alpha: float = 0.92
    color_match_alpha: float = 0.85
    warmup_mode: str = "seed_from_first_valid"
    reset_on_empty_mask: bool = True
    reset_on_copy_through: bool = False
    reset_on_harmonizer_failure: bool = True
    reset_on_crop_iou_below: float = 0.25
    reset_on_mask_area_ratio_low: float = 0.5
    reset_on_mask_area_ratio_high: float = 2.0
    reset_on_centroid_jump_fraction: float = 0.25
    fallback_mode: str = "affine_rgb_gain_bias"
    write_diagnostics: bool = True


@dataclass(frozen=True, slots=True)
class EligibilitySettings:
    min_visible_mask_pixels_for_model: int = 48
    min_visible_bbox_short_side_px_for_model: int = 6
    max_crop_coverage_ratio_for_model: float = 0.70
    max_crop_coverage_mask_pixels_threshold: int = 128


@dataclass(frozen=True, slots=True)
class CorrectionClampSettings:
    min_foreground_pixels_for_luminance_scale: int = 64
    min_foreground_luminance_std_for_scale: float = 2.0
    luminance_delta_clamp_small_mask: float = 18.0
    luminance_delta_clamp_model: float = 28.0
    luminance_std_ratio_clamp_low: float = 0.75
    luminance_std_ratio_clamp_high: float = 1.25
    chroma_shift_clamp: float = 6.0


@dataclass(frozen=True, slots=True)
class PostcheckSettings:
    max_ring_overshoot_luma: float = 6.0
    max_small_mask_brighten_luma: float = 24.0


@dataclass(frozen=True, slots=True)
class TinyObjectSettings:
    enabled: bool = False
    max_mask_pixels_for_conservative_path: int = 256
    max_bbox_short_side_px_for_conservative_path: int = 20
    skip_color_match_below_mask_pixels: int = 256


@dataclass(frozen=True, slots=True)
class AdaptiveSettings:
    enabled: bool = False
    profile_bias: float = 0.0
    min_effect_strength: float = 0.2
    low_support_weight: float = 0.75
    tiny_subject_weight: float = 0.6
    synthetic_scene_weight: float = 0.5
    no_local_support_color_match_strength_cap: float = 0.35
    no_local_support_harmonizer_strength_cap: float = 0.5
    backfilled_parameter_strength_scale: float = 0.35
    interpolated_parameter_strength_scale: float = 0.6
    synthetic_contrast_preservation: float = 0.85
    synthetic_saturation_preservation: float = 0.8


@dataclass(frozen=True, slots=True)
class SceneStyleMetrics:
    edge_density: float
    saturation_fraction: float
    quantized_color_ratio: float
    synthetic_score: float


@dataclass(frozen=True, slots=True)
class AdaptivePolicy:
    effect_strength: float
    color_match_strength: float
    harmonizer_strength: float
    preserve_contrast: float
    preserve_saturation: float
    local_support_ratio: float
    tiny_subject_score: float
    synthetic_scene_score: float


@dataclass(frozen=True, slots=True)
class OversizedActorSettings:
    max_frame_coverage_ratio: float = 0.85
    containment_margin_px: int = 8
    reject_when_actor_exceeds_crop: bool = True
    oversized_actor_behavior: str = "full_mask_affine_or_copy"
    full_frame_affine_min_mask_pixels: int = 512


@dataclass(frozen=True, slots=True)
class ColorMatchParameters:
    luminance_mean_delta: float
    luminance_std_ratio: float
    chroma_a_shift: float
    chroma_b_shift: float
    chroma_scale: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "luminance_mean_delta": float(self.luminance_mean_delta),
            "luminance_std_ratio": float(self.luminance_std_ratio),
            "chroma_a_shift": float(self.chroma_a_shift),
            "chroma_b_shift": float(self.chroma_b_shift),
            "chroma_scale": float(self.chroma_scale),
        }


@dataclass(frozen=True, slots=True)
class AffineColorTransform:
    gain: np.ndarray
    bias: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "gain": [float(v) for v in self.gain.tolist()],
            "bias": [float(v) for v in self.bias.tolist()],
        }


@dataclass(frozen=True, slots=True)
class TemporalContinuityStats:
    crop_iou: float | None
    mask_area_ratio: float | None
    centroid_jump_fraction: float | None


@dataclass(frozen=True, slots=True)
class TemporalCropStats:
    crop_bounds: CropBounds
    mask_pixel_count: int
    centroid_xy: tuple[float, float]
    crop_diagonal: float


@dataclass(frozen=True, slots=True)
class TrackSpanFrameAnalysis:
    frame_index: int
    mask_pixel_count: int
    bbox: CropBounds
    crop_bounds: CropBounds
    stable_reference_eligible: bool


@dataclass(frozen=True, slots=True)
class TrackSpanDecision:
    track_id: int
    span_id: int
    frame_indices: tuple[int, ...]
    policy: str
    reference_frame_indices: tuple[int, ...]
    seed_reference_frame_index: int | None
    downgrade_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessFrameArtifacts:
    frame_index: int
    smoothed_arguments: dict[str, float] | None = None
    smoothed_color_match_parameters: ColorMatchParameters | None = None
    fallback_transform: AffineColorTransform | None = None
    used_for_reference_estimation: bool = False


@dataclass(slots=True)
class TemporalState:
    argument_ema: dict[str, torch.Tensor] = field(default_factory=dict)
    color_match_ema: ColorMatchParameters | None = None
    affine_transform_ema: AffineColorTransform | None = None
    previous_crop_stats: TemporalCropStats | None = None
    tiny_object_mode: bool = False
    tiny_object_release_streak: int = 0
    last_accepted_masked_luma: float | None = None
    reset_count: int = 0
    fallback_count: int = 0

    def reset(self) -> None:
        self.argument_ema.clear()
        self.color_match_ema = None
        self.affine_transform_ema = None
        self.previous_crop_stats = None
        self.tiny_object_mode = False
        self.tiny_object_release_streak = 0
        self.last_accepted_masked_luma = None
        self.reset_count += 1


@dataclass(frozen=True, slots=True)
class PostcheckEvaluation:
    rejected: bool
    reason: str | None
    before_luma: float | None
    after_luma: float | None
    ring_luma: float | None
    continuity_bound_source: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    recovered_rgb: np.ndarray
    mode: str
    strength: float
    evaluation: PostcheckEvaluation


DEFAULT_FILTER_NAMES = (
    "temperature",
    "brightness",
    "contrast",
    "saturation",
    "highlight",
    "shadow",
)
TONAL_FILTER_NAMES = {"highlight", "shadow"}


def _clip_unit(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _estimate_scene_style_metrics(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
) -> SceneStyleMetrics:
    rgb = np.asarray(crop_rgb, dtype=np.uint8)
    if rgb.size == 0:
        return SceneStyleMetrics(0.0, 0.0, 1.0, 0.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    edge_density = float(np.mean(grad > 18.0))
    rgb_f = np.asarray(rgb, dtype=np.float32)
    saturation_fraction = float(
        np.mean((rgb_f.max(axis=-1) - rgb_f.min(axis=-1)) > 28.0)
    )
    quantized = (rgb // 32).reshape(-1, 3)
    unique_ratio = float(
        len({(int(r), int(g), int(b)) for r, g, b in quantized})
        / max(int(quantized.shape[0]), 1)
    )
    support_fraction = float(np.mean(np.asarray(crop_mask, dtype=np.uint8) > 0))
    synthetic_score = _clip_unit(
        0.55 * (1.0 - min(unique_ratio / 0.22, 1.0))
        + 0.25 * min(saturation_fraction / 0.35, 1.0)
        + 0.20 * (1.0 - min(edge_density / 0.18, 1.0))
        + 0.10 * (1.0 - min(support_fraction / 0.35, 1.0))
    )
    return SceneStyleMetrics(
        edge_density=float(edge_density),
        saturation_fraction=float(saturation_fraction),
        quantized_color_ratio=float(unique_ratio),
        synthetic_score=float(synthetic_score),
    )


def _compute_adaptive_policy(
    *,
    adaptive_settings: AdaptiveSettings,
    scene_style_metrics: SceneStyleMetrics,
    mask_pixel_count: int,
    bbox: CropBounds | None,
    color_match_settings: ColorMatchSettings,
    ring_pixel_count_filtered: int,
    applied_parameter_source: str | None,
) -> AdaptivePolicy:
    if not adaptive_settings.enabled:
        return AdaptivePolicy(
            effect_strength=1.0,
            color_match_strength=1.0,
            harmonizer_strength=1.0,
            preserve_contrast=0.0,
            preserve_saturation=0.0,
            local_support_ratio=1.0,
            tiny_subject_score=0.0,
            synthetic_scene_score=scene_style_metrics.synthetic_score,
        )
    bbox_short_side = 0.0 if bbox is None else float(min(bbox.width, bbox.height))
    tiny_by_mask = 1.0 - min(mask_pixel_count / 2048.0, 1.0)
    tiny_by_bbox = 1.0 - min(bbox_short_side / 48.0, 1.0)
    tiny_subject_score = _clip_unit(max(tiny_by_mask, tiny_by_bbox))
    local_support_ratio = _clip_unit(
        ring_pixel_count_filtered / max(float(color_match_settings.min_ring_pixels), 1.0)
    )
    preserve = _clip_unit(
        adaptive_settings.profile_bias
        + adaptive_settings.synthetic_scene_weight * scene_style_metrics.synthetic_score
        + adaptive_settings.tiny_subject_weight * tiny_subject_score
        + adaptive_settings.low_support_weight * (1.0 - local_support_ratio)
    )
    source_scale = 1.0
    if applied_parameter_source == "backfilled_from_future":
        source_scale = adaptive_settings.backfilled_parameter_strength_scale
    elif applied_parameter_source == "interpolated":
        source_scale = adaptive_settings.interpolated_parameter_strength_scale
    no_support_color_cap = (
        adaptive_settings.no_local_support_color_match_strength_cap
        if ring_pixel_count_filtered <= 0
        else 1.0
    )
    no_support_harmonizer_cap = (
        adaptive_settings.no_local_support_harmonizer_strength_cap
        if ring_pixel_count_filtered <= 0
        else 1.0
    )
    base_strength = max(adaptive_settings.min_effect_strength, 1.0 - 0.65 * preserve)
    color_match_strength = min(base_strength * source_scale, no_support_color_cap)
    harmonizer_strength = min(base_strength * source_scale, no_support_harmonizer_cap)
    return AdaptivePolicy(
        effect_strength=float(base_strength),
        color_match_strength=float(color_match_strength),
        harmonizer_strength=float(harmonizer_strength),
        preserve_contrast=float(
            _clip_unit(preserve * adaptive_settings.synthetic_contrast_preservation)
        ),
        preserve_saturation=float(
            _clip_unit(preserve * adaptive_settings.synthetic_saturation_preservation)
        ),
        local_support_ratio=float(local_support_ratio),
        tiny_subject_score=float(tiny_subject_score),
        synthetic_scene_score=float(scene_style_metrics.synthetic_score),
    )


def _scale_color_match_parameters(
    parameters: ColorMatchParameters,
    *,
    strength: float,
) -> ColorMatchParameters:
    strength = _clip_unit(strength)
    return ColorMatchParameters(
        luminance_mean_delta=float(parameters.luminance_mean_delta) * strength,
        luminance_std_ratio=1.0 + (float(parameters.luminance_std_ratio) - 1.0) * strength,
        chroma_a_shift=float(parameters.chroma_a_shift) * strength,
        chroma_b_shift=float(parameters.chroma_b_shift) * strength,
        chroma_scale=1.0 + (float(parameters.chroma_scale) - 1.0) * strength,
    )


def _blend_masked_rgb_strength(
    base_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    alpha = alpha * _clip_unit(strength)
    result = np.asarray(base_rgb, dtype=np.float32).copy()
    candidate = np.asarray(candidate_rgb, dtype=np.float32)
    result = result * (1.0 - alpha[..., None]) + candidate * alpha[..., None]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _load_harmonizer_model_module():
    _ensure_harmonizer_on_path()
    from src import model  # noqa: E402

    return model


def _load_resource_store(run_dir: Path):
    _ensure_repo_src_on_path()
    from pemoin.data.contracts import ResourceStore  # noqa: E402

    return ResourceStore(run_dir.name, root=run_dir.parent)


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected image shape (H, W, 3+), got {arr.shape}.")
    arr = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _build_frame_index_map(frame_dir: Path, *, suffix: str = ".png") -> dict[int, Path]:
    frame_map: dict[int, Path] = {}
    for frame_path in sorted(frame_dir.glob(f"*{suffix}")):
        match = re.search(r"(\d+)$", frame_path.stem)
        if not match:
            continue
        frame_map[int(match.group(1))] = frame_path
    return frame_map


def _harmonizer_filter_names(harmonizer: HarmonizerProtocol, count: int) -> list[str]:
    filter_types = getattr(harmonizer, "filter_types", None)
    if isinstance(filter_types, list) and len(filter_types) == count:
        names: list[str] = []
        for item in filter_types:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name:
                names.append(name.strip().lower())
            else:
                names.append(str(item).strip().lower())
        return names
    return list(DEFAULT_FILTER_NAMES[:count])


def _normalize_arguments(arguments: Any) -> list[torch.Tensor]:
    if isinstance(arguments, torch.Tensor):
        return [arguments]
    if not isinstance(arguments, (list, tuple)):
        raise TypeError(f"Unsupported harmonizer argument type: {type(arguments)!r}.")
    normalized = list(arguments)
    if not normalized:
        raise ValueError("Harmonizer returned no arguments.")
    tensors = [arg for arg in normalized if isinstance(arg, torch.Tensor)]
    if len(tensors) != len(normalized):
        raise TypeError("Harmonizer arguments must all be tensors.")
    return tensors


def _argument_scalar(argument: torch.Tensor) -> float:
    detached = argument.detach().reshape(-1)
    if detached.numel() == 0:
        raise ValueError("Harmonizer argument tensor is empty.")
    return float(detached[0].item())


def _named_argument_dict(
    filter_names: list[str],
    arguments: list[torch.Tensor],
) -> dict[str, float]:
    return {
        name: _argument_scalar(argument)
        for name, argument in zip(filter_names, arguments)
    }


def _predict_harmonizer_arguments(
    harmonizer: HarmonizerProtocol,
    *,
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    device: str,
) -> tuple[list[torch.Tensor], list[str]]:
    comp_tensor = tf.to_tensor(Image.fromarray(crop_rgb))[None, ...]
    mask_tensor = tf.to_tensor(Image.fromarray(crop_mask, mode="L"))[None, ...]

    if device == "cuda":
        comp_tensor = comp_tensor.cuda()
        mask_tensor = mask_tensor.cuda()

    with torch.no_grad():
        arguments = _normalize_arguments(harmonizer.predict_arguments(comp_tensor, mask_tensor))
    return arguments, _harmonizer_filter_names(harmonizer, len(arguments))


def _restore_harmonized_crop(
    harmonizer: HarmonizerProtocol,
    *,
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    arguments: list[torch.Tensor],
    device: str,
) -> np.ndarray:
    comp_tensor = tf.to_tensor(Image.fromarray(crop_rgb))[None, ...]
    mask_tensor = tf.to_tensor(Image.fromarray(crop_mask, mode="L"))[None, ...]

    if device == "cuda":
        comp_tensor = comp_tensor.cuda()
        mask_tensor = mask_tensor.cuda()

    with torch.no_grad():
        harmonized = harmonizer.restore_image(comp_tensor, mask_tensor, arguments)[-1]

    harmonized_np = np.transpose(harmonized[0].detach().cpu().numpy(), (1, 2, 0)) * 255.0
    return np.clip(np.rint(harmonized_np), 0.0, 255.0).astype(np.uint8)


def _ema_tensor(previous: torch.Tensor | None, current: torch.Tensor, alpha: float) -> torch.Tensor:
    if previous is None:
        return current.detach().clone()
    return (float(alpha) * previous) + ((1.0 - float(alpha)) * current.detach())


def _smooth_harmonizer_arguments(
    raw_arguments: list[torch.Tensor],
    *,
    filter_names: list[str],
    temporal_state: TemporalState,
    temporal_settings: TemporalSmoothingSettings,
) -> list[torch.Tensor]:
    smoothed: list[torch.Tensor] = []
    for name, argument in zip(filter_names, raw_arguments):
        alpha = (
            temporal_settings.tonal_alpha
            if name in TONAL_FILTER_NAMES
            else temporal_settings.appearance_alpha
        )
        previous = temporal_state.argument_ema.get(name)
        ema_value = _ema_tensor(previous, argument, alpha)
        temporal_state.argument_ema[name] = ema_value.detach().clone()
        smoothed.append(ema_value)
    return smoothed


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mask_bbox(mask: np.ndarray) -> CropBounds | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return None
    return CropBounds(
        left=int(xs.min()),
        top=int(ys.min()),
        right=int(xs.max()) + 1,
        bottom=int(ys.max()) + 1,
    )


def _clamp_crop_bounds(
    *,
    center_x: float,
    center_y: float,
    target_width: int,
    target_height: int,
    image_width: int,
    image_height: int,
) -> CropBounds:
    width = min(int(target_width), int(image_width))
    height = min(int(target_height), int(image_height))
    if width <= 0 or height <= 0:
        raise ValueError(
            "Crop dimensions must be positive after clamping, got "
            f"width={width}, height={height}."
        )

    left = int(round(center_x - (width / 2.0)))
    top = int(round(center_y - (height / 2.0)))
    right = left + width
    bottom = top + height

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    return CropBounds(left=left, top=top, right=right, bottom=bottom)


def _compute_local_crop(
    mask: np.ndarray,
    *,
    bbox_expansion_scale: float,
    min_crop_size_ratio: float | None = None,
    min_crop_size_px: int | None = None,
    max_frame_coverage_ratio: float = 0.85,
) -> tuple[CropBounds | None, CropBounds | None, CropBounds | None]:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None, None, None

    center_x = (bbox.left + bbox.right) / 2.0
    center_y = (bbox.top + bbox.bottom) / 2.0
    if min_crop_size_px is not None:
        min_crop_width = max(1, int(min_crop_size_px))
        min_crop_height = max(1, int(min_crop_size_px))
    else:
        if min_crop_size_ratio is None:
            raise ValueError("min_crop_size_ratio or min_crop_size_px must be provided.")
        min_crop_width = max(
            1,
            int(np.ceil(float(mask.shape[1]) * float(min_crop_size_ratio))),
        )
        min_crop_height = max(
            1,
            int(np.ceil(float(mask.shape[0]) * float(min_crop_size_ratio))),
        )
    target_width = max(
        int(np.ceil(float(bbox.width) * float(bbox_expansion_scale))),
        min_crop_width,
    )
    target_height = max(
        int(np.ceil(float(bbox.height) * float(bbox_expansion_scale))),
        min_crop_height,
    )
    target_width = min(
        target_width,
        max(1, int(np.ceil(float(mask.shape[1]) * float(max_frame_coverage_ratio)))),
    )
    target_height = min(
        target_height,
        max(1, int(np.ceil(float(mask.shape[0]) * float(max_frame_coverage_ratio)))),
    )
    expanded = _clamp_crop_bounds(
        center_x=center_x,
        center_y=center_y,
        target_width=target_width,
        target_height=target_height,
        image_width=int(mask.shape[1]),
        image_height=int(mask.shape[0]),
    )
    return bbox, expanded, expanded


def _crop_contains_bbox(
    crop_bounds: CropBounds | None,
    bbox: CropBounds | None,
    *,
    margin_px: int,
) -> bool:
    if crop_bounds is None or bbox is None:
        return True
    margin = int(max(0, margin_px))
    return (
        bbox.left >= (crop_bounds.left + margin)
        and bbox.top >= (crop_bounds.top + margin)
        and bbox.right <= (crop_bounds.right - margin)
        and bbox.bottom <= (crop_bounds.bottom - margin)
    )


def _count_visible_mask_outside_crop(mask: np.ndarray, crop_bounds: CropBounds | None) -> int:
    visible = np.asarray(mask, dtype=np.uint8) > 0
    if crop_bounds is None:
        return int(np.count_nonzero(visible))
    outside = visible.copy()
    outside[crop_bounds.top : crop_bounds.bottom, crop_bounds.left : crop_bounds.right] = False
    return int(np.count_nonzero(outside))


def _crop_rgb(image: np.ndarray, bounds: CropBounds) -> np.ndarray:
    return np.asarray(image[bounds.top : bounds.bottom, bounds.left : bounds.right, :], dtype=np.uint8)


def _crop_mask(mask: np.ndarray, bounds: CropBounds) -> np.ndarray:
    return np.asarray(mask[bounds.top : bounds.bottom, bounds.left : bounds.right], dtype=np.uint8)


def _paste_crop(base_image: np.ndarray, crop_image: np.ndarray, bounds: CropBounds) -> np.ndarray:
    result = np.array(base_image, copy=True)
    result[bounds.top : bounds.bottom, bounds.left : bounds.right, :] = crop_image
    return result


def _crop_iou(left: CropBounds, right: CropBounds) -> float:
    inter_left = max(left.left, right.left)
    inter_top = max(left.top, right.top)
    inter_right = min(left.right, right.right)
    inter_bottom = min(left.bottom, right.bottom)
    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)
    intersection = float(inter_width * inter_height)
    area_left = float(left.width * left.height)
    area_right = float(right.width * right.height)
    union = area_left + area_right - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _build_temporal_crop_stats(
    mask: np.ndarray,
    crop_bounds: CropBounds,
) -> TemporalCropStats:
    centroid = _mask_centroid(mask)
    if centroid is None:
        raise ValueError("Temporal crop stats require a non-empty mask.")
    return TemporalCropStats(
        crop_bounds=crop_bounds,
        mask_pixel_count=int(np.count_nonzero(mask)),
        centroid_xy=centroid,
        crop_diagonal=float(np.hypot(float(crop_bounds.width), float(crop_bounds.height))),
    )


def _compute_temporal_continuity(
    previous: TemporalCropStats | None,
    current: TemporalCropStats,
) -> TemporalContinuityStats:
    if previous is None:
        return TemporalContinuityStats(
            crop_iou=None,
            mask_area_ratio=None,
            centroid_jump_fraction=None,
        )
    crop_iou = _crop_iou(previous.crop_bounds, current.crop_bounds)
    mask_area_ratio = float(current.mask_pixel_count) / max(float(previous.mask_pixel_count), 1.0)
    centroid_jump = float(
        np.hypot(
            current.centroid_xy[0] - previous.centroid_xy[0],
            current.centroid_xy[1] - previous.centroid_xy[1],
        )
    )
    diag = max(previous.crop_diagonal, current.crop_diagonal, 1.0)
    return TemporalContinuityStats(
        crop_iou=float(crop_iou),
        mask_area_ratio=float(mask_area_ratio),
        centroid_jump_fraction=float(centroid_jump / diag),
    )


def _maybe_reset_temporal_state(
    temporal_state: TemporalState,
    current_stats: TemporalCropStats,
    temporal_settings: TemporalSmoothingSettings,
    *,
    ignore_mask_area_ratio: bool = False,
) -> tuple[bool, str | None, TemporalContinuityStats]:
    continuity = _compute_temporal_continuity(temporal_state.previous_crop_stats, current_stats)
    reason: str | None = None
    if continuity.crop_iou is not None and continuity.crop_iou < temporal_settings.reset_on_crop_iou_below:
        reason = "crop_iou_below_threshold"
    elif (
        not ignore_mask_area_ratio
        and continuity.mask_area_ratio is not None
        and continuity.mask_area_ratio < temporal_settings.reset_on_mask_area_ratio_low
    ):
        reason = "mask_area_ratio_below_threshold"
    elif (
        not ignore_mask_area_ratio
        and continuity.mask_area_ratio is not None
        and continuity.mask_area_ratio > temporal_settings.reset_on_mask_area_ratio_high
    ):
        reason = "mask_area_ratio_above_threshold"
    elif (
        continuity.centroid_jump_fraction is not None
        and continuity.centroid_jump_fraction
        > temporal_settings.reset_on_centroid_jump_fraction
    ):
        reason = "centroid_jump_fraction_above_threshold"
    if reason is not None:
        temporal_state.reset()
        continuity = TemporalContinuityStats(
            crop_iou=continuity.crop_iou,
            mask_area_ratio=continuity.mask_area_ratio,
            centroid_jump_fraction=continuity.centroid_jump_fraction,
        )
        return True, reason, continuity
    return False, None, continuity


def _build_ring_mask(
    mask: np.ndarray,
    *,
    ring_inner_px: int,
    ring_outer_px: int,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    kernel_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * int(ring_outer_px) + 1, 2 * int(ring_outer_px) + 1),
    )
    outer = cv2.dilate(mask_bool.astype(np.uint8), kernel_outer) > 0
    if ring_inner_px <= 0:
        inner = mask_bool
    else:
        kernel_inner = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(ring_inner_px) + 1, 2 * int(ring_inner_px) + 1),
        )
        inner = cv2.dilate(mask_bool.astype(np.uint8), kernel_inner) > 0
    return outer & (~inner) & (~mask_bool)


def _label_map_from_segments(segments, *, use_label_id: bool) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for seg in segments:
        key = getattr(seg, "label_id", None) if use_label_id else getattr(seg, "segment_id", None)
        if key is None:
            continue
        label_map[int(key)] = str(getattr(seg, "label", "")).strip().lower()
    return label_map


def _semantic_role_tokens(role: str, *, metadata: dict[str, Any] | None) -> set[str]:
    _ensure_repo_src_on_path()
    from pemoin.providers.semantic_roles import resolve_semantic_role_labels  # noqa: E402

    return {
        str(token).strip().lower()
        for token in resolve_semantic_role_labels(role, metadata=metadata)
        if str(token).strip()
    }


def _sky_mask_from_semantics(semantics) -> np.ndarray:
    sky_tokens = _semantic_role_tokens("sky", metadata=getattr(semantics, "metadata", None))
    if not sky_tokens:
        return np.zeros_like(np.asarray(semantics.segment_ids), dtype=bool)

    if getattr(semantics, "label_ids", None) is not None:
        ids = np.asarray(semantics.label_ids, dtype=np.int32)
        label_map = _label_map_from_segments(semantics.segments, use_label_id=True)
    elif getattr(semantics, "segment_ids", None) is not None:
        ids = np.asarray(semantics.segment_ids, dtype=np.int32)
        label_map = _label_map_from_segments(semantics.segments, use_label_id=False)
    else:
        return np.zeros((0, 0), dtype=bool)
    sky_ids = {idx for idx, label in label_map.items() if label in sky_tokens}
    if not sky_ids:
        return np.zeros_like(ids, dtype=bool)
    return np.isin(ids, list(sky_ids))


def _pedestrian_mask_from_semantics(semantics) -> np.ndarray:
    if getattr(semantics, "label_ids", None) is not None:
        ids = np.asarray(semantics.label_ids, dtype=np.int32)
        label_map = _label_map_from_segments(semantics.segments, use_label_id=True)
    elif getattr(semantics, "segment_ids", None) is not None:
        ids = np.asarray(semantics.segment_ids, dtype=np.int32)
        label_map = _label_map_from_segments(semantics.segments, use_label_id=False)
    else:
        return np.zeros((0, 0), dtype=bool)
    pedestrian_ids = {
        idx
        for idx, label in label_map.items()
        if any(marker in label for marker in ("person", "pedestrian", "human"))
    }
    if not pedestrian_ids:
        return np.zeros_like(ids, dtype=bool)
    return np.isin(ids, list(pedestrian_ids))


def _robust_inlier_mask(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((values.shape[0],), dtype=bool)
    low = np.percentile(values, 5.0, axis=0)
    high = np.percentile(values, 95.0, axis=0)
    return np.all((values >= low[None, :]) & (values <= high[None, :]), axis=1)


def _robust_center(values: np.ndarray) -> np.ndarray:
    return np.median(values, axis=0)


def _robust_spread(values: np.ndarray) -> np.ndarray:
    q25 = np.percentile(values, 25.0, axis=0)
    q75 = np.percentile(values, 75.0, axis=0)
    return (q75 - q25) / 1.349


def _clamp_color_match_parameters(
    parameters: ColorMatchParameters,
    *,
    clamp_settings: CorrectionClampSettings,
    small_mask_mode: bool,
    allow_luminance_scale: bool,
) -> ColorMatchParameters:
    luminance_delta_limit = (
        clamp_settings.luminance_delta_clamp_small_mask
        if small_mask_mode
        else clamp_settings.luminance_delta_clamp_model
    )
    luminance_std_ratio = (
        1.0
        if not allow_luminance_scale
        else float(
            np.clip(
                parameters.luminance_std_ratio,
                clamp_settings.luminance_std_ratio_clamp_low,
                clamp_settings.luminance_std_ratio_clamp_high,
            )
        )
    )
    chroma_limit = float(clamp_settings.chroma_shift_clamp)
    return ColorMatchParameters(
        luminance_mean_delta=float(
            np.clip(
                parameters.luminance_mean_delta,
                -luminance_delta_limit,
                luminance_delta_limit,
            )
        ),
        luminance_std_ratio=luminance_std_ratio,
        chroma_a_shift=float(
            np.clip(parameters.chroma_a_shift, -chroma_limit, chroma_limit)
        ),
        chroma_b_shift=float(
            np.clip(parameters.chroma_b_shift, -chroma_limit, chroma_limit)
        ),
        chroma_scale=float(np.clip(parameters.chroma_scale, 0.55, 1.15)),
    )


def _masked_luma(image_rgb: np.ndarray, mask: np.ndarray) -> float | None:
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(mask_bool):
        return None
    rgb = np.asarray(image_rgb, dtype=np.float32)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return float(luma[mask_bool].mean())


def _ring_luma(image_rgb: np.ndarray, ring_mask: np.ndarray) -> float | None:
    ring = np.asarray(ring_mask, dtype=bool)
    if not np.any(ring):
        return None
    rgb = np.asarray(image_rgb, dtype=np.float32)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return float(luma[ring].mean())


def _postcheck_harmonized_crop(
    *,
    pre_harmonization_rgb: np.ndarray,
    harmonized_rgb: np.ndarray,
    crop_mask: np.ndarray,
    ring_filtered: np.ndarray,
    postcheck_settings: PostcheckSettings,
    small_mask_mode: bool,
) -> tuple[bool, str | None, float | None, float | None, float | None]:
    before_luma = _masked_luma(pre_harmonization_rgb, crop_mask)
    after_luma = _masked_luma(harmonized_rgb, crop_mask)
    ring_luma = _ring_luma(pre_harmonization_rgb, ring_filtered)
    if before_luma is None or after_luma is None:
        return False, None, before_luma, after_luma, ring_luma
    if ring_luma is not None and after_luma > (ring_luma + postcheck_settings.max_ring_overshoot_luma):
        return True, "ring_overshoot", before_luma, after_luma, ring_luma
    brighten_limit = (
        postcheck_settings.max_small_mask_brighten_luma
        if small_mask_mode
        else float("inf")
    )
    if small_mask_mode and (after_luma - before_luma) > brighten_limit:
        return True, "small_mask_brightness_increase", before_luma, after_luma, ring_luma
    if ring_luma is not None:
        local_gap = max(ring_luma - before_luma, 0.0)
        adaptive_limit = min(postcheck_settings.max_small_mask_brighten_luma, 0.75 * local_gap + 6.0)
        if (after_luma - before_luma) > adaptive_limit:
            return True, "local_gap_brightness_increase", before_luma, after_luma, ring_luma
    return False, None, before_luma, after_luma, ring_luma


def _temporal_continuity_limit(
    *,
    before_luma: float | None,
    temporal_state: TemporalState | None,
    postcheck_settings: PostcheckSettings,
    small_mask_mode: bool,
) -> tuple[float | None, str | None]:
    if before_luma is None or temporal_state is None:
        return None, None
    previous_luma = temporal_state.last_accepted_masked_luma
    if previous_luma is None:
        return None, None
    margin = (
        min(float(postcheck_settings.max_small_mask_brighten_luma) * 0.35, 8.0)
        if small_mask_mode
        else 12.0
    )
    limit = max(float(before_luma), float(previous_luma) + float(margin))
    return float(limit), "previous_accepted_masked_luma"


def _evaluate_postcheck_candidate(
    *,
    pre_harmonization_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    crop_mask: np.ndarray,
    ring_filtered: np.ndarray,
    postcheck_settings: PostcheckSettings,
    small_mask_mode: bool,
    temporal_state: TemporalState | None = None,
    use_continuity_bound: bool = False,
) -> PostcheckEvaluation:
    (
        rejected,
        reason,
        before_luma,
        after_luma,
        ring_luma,
    ) = _postcheck_harmonized_crop(
        pre_harmonization_rgb=pre_harmonization_rgb,
        harmonized_rgb=candidate_rgb,
        crop_mask=crop_mask,
        ring_filtered=ring_filtered,
        postcheck_settings=postcheck_settings,
        small_mask_mode=small_mask_mode,
    )
    continuity_source = None
    if not rejected and use_continuity_bound:
        continuity_limit, continuity_source = _temporal_continuity_limit(
            before_luma=before_luma,
            temporal_state=temporal_state,
            postcheck_settings=postcheck_settings,
            small_mask_mode=small_mask_mode,
        )
        if (
            continuity_limit is not None
            and after_luma is not None
            and after_luma > float(continuity_limit)
        ):
            rejected = True
            reason = "continuity_brightness_overshoot"
    return PostcheckEvaluation(
        rejected=bool(rejected),
        reason=reason,
        before_luma=before_luma,
        after_luma=after_luma,
        ring_luma=ring_luma,
        continuity_bound_source=continuity_source,
    )


def _derive_lab_color_match_parameters(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    bbox: CropBounds,
    color_match_settings: ColorMatchSettings,
    clamp_settings: CorrectionClampSettings,
    small_mask_mode: bool,
    full_frame_sky_mask_crop: np.ndarray | None = None,
    full_frame_pedestrian_mask_crop: np.ndarray | None = None,
) -> tuple[ColorMatchParameters | None, ColorMatchDiagnostics, np.ndarray, np.ndarray]:
    fg_mask = np.asarray(crop_mask, dtype=np.uint8) > 0
    foreground_count = int(np.count_nonzero(fg_mask))
    if not np.any(fg_mask):
        return (
            None,
            ColorMatchDiagnostics(
                applied=False,
                skip_reason="empty_foreground_mask",
                ring_pixel_count_raw=0,
                ring_pixel_count_filtered=0,
                sky_filtered_pixel_count=0,
                outlier_filtered_pixel_count=0,
                foreground_pixel_count=foreground_count,
                debug=None,
            ),
            np.zeros_like(fg_mask, dtype=bool),
            np.zeros_like(fg_mask, dtype=bool),
        )

    ring_raw = _build_ring_mask(
        fg_mask.astype(np.uint8) * 255,
        ring_inner_px=color_match_settings.ring_inner_px,
        ring_outer_px=color_match_settings.ring_outer_px,
    )
    filtered_ring = np.array(ring_raw, copy=True)
    sky_filtered_pixel_count = 0

    if color_match_settings.exclude_top_band and color_match_settings.top_band_reference == "mask_top":
        top_limit = min(filtered_ring.shape[0], int(bbox.top) + int(color_match_settings.top_band_px))
        filtered_ring[:top_limit, :] = False
    if full_frame_sky_mask_crop is not None and full_frame_sky_mask_crop.shape == filtered_ring.shape:
        sky_overlap = filtered_ring & np.asarray(full_frame_sky_mask_crop, dtype=bool)
        sky_filtered_pixel_count = int(np.count_nonzero(sky_overlap))
        filtered_ring = filtered_ring & (~np.asarray(full_frame_sky_mask_crop, dtype=bool))

    crop_lab = cv2.cvtColor(np.asarray(crop_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    raw_count = int(np.count_nonzero(filtered_ring))
    if raw_count < int(color_match_settings.min_ring_pixels):
        return (
            None,
            ColorMatchDiagnostics(
                applied=False,
                skip_reason="insufficient_ring_pixels",
                ring_pixel_count_raw=int(np.count_nonzero(ring_raw)),
                ring_pixel_count_filtered=raw_count,
                sky_filtered_pixel_count=sky_filtered_pixel_count,
                outlier_filtered_pixel_count=0,
                foreground_pixel_count=foreground_count,
                debug=None,
            ),
            ring_raw,
            filtered_ring,
        )

    bg_values = crop_lab[filtered_ring]
    inlier_mask = _robust_inlier_mask(bg_values)
    if color_match_settings.outlier_rejection != "robust_percentile":
        raise ValueError(
            f"Unsupported color-match outlier rejection: {color_match_settings.outlier_rejection}."
        )
    outlier_filtered_pixel_count = int(bg_values.shape[0] - int(np.count_nonzero(inlier_mask)))
    if np.count_nonzero(inlier_mask) < int(color_match_settings.min_ring_pixels):
        return (
            None,
            ColorMatchDiagnostics(
                applied=False,
                skip_reason="insufficient_ring_inliers",
                ring_pixel_count_raw=int(np.count_nonzero(ring_raw)),
                ring_pixel_count_filtered=int(np.count_nonzero(inlier_mask)),
                sky_filtered_pixel_count=sky_filtered_pixel_count,
                outlier_filtered_pixel_count=outlier_filtered_pixel_count,
                foreground_pixel_count=foreground_count,
                debug=None,
            ),
            ring_raw,
            filtered_ring,
        )
    bg_values = bg_values[inlier_mask]

    fg_values = crop_lab[fg_mask]
    fg_mean = _robust_center(fg_values)
    fg_std = _robust_spread(fg_values)
    bg_mean = _robust_center(bg_values)
    bg_std = _robust_spread(bg_values)
    pedestrian_reference_values = None
    pedestrian_reference_pixel_count = 0
    if (
        color_match_settings.prefer_pedestrian_reference
        and full_frame_pedestrian_mask_crop is not None
        and full_frame_pedestrian_mask_crop.shape == fg_mask.shape
    ):
        pedestrian_mask = np.asarray(full_frame_pedestrian_mask_crop, dtype=bool) & (~fg_mask)
        pedestrian_mask = pedestrian_mask & (~np.asarray(full_frame_sky_mask_crop, dtype=bool)) if (
            full_frame_sky_mask_crop is not None and full_frame_sky_mask_crop.shape == fg_mask.shape
        ) else pedestrian_mask
        pedestrian_reference_pixel_count = int(np.count_nonzero(pedestrian_mask))
        if pedestrian_reference_pixel_count >= int(color_match_settings.min_pedestrian_reference_pixels):
            pedestrian_reference_values = crop_lab[pedestrian_mask]
    target_mean = np.asarray(bg_mean, dtype=np.float32)
    target_std = np.asarray(bg_std, dtype=np.float32)
    if pedestrian_reference_values is not None and pedestrian_reference_values.shape[0] > 0:
        ped_mean = _robust_center(pedestrian_reference_values)
        ped_std = _robust_spread(pedestrian_reference_values)
        reference_weight = float(color_match_settings.pedestrian_reference_weight)
        scene_weight = float(color_match_settings.fallback_scene_reference_weight)
        total_weight = max(reference_weight + scene_weight, 1e-6)
        ped_share = reference_weight / total_weight
        scene_share = scene_weight / total_weight
        target_mean = (
            np.asarray(bg_mean, dtype=np.float32) * scene_share
            + np.asarray(ped_mean, dtype=np.float32) * ped_share
        )
        target_std = (
            np.asarray(bg_std, dtype=np.float32) * scene_share
            + np.asarray(ped_std, dtype=np.float32) * ped_share
        )
    allow_luminance_scale = (
        foreground_count >= int(clamp_settings.min_foreground_pixels_for_luminance_scale)
        and float(fg_std[0]) >= float(clamp_settings.min_foreground_luminance_std_for_scale)
    )
    fg_chroma = np.linalg.norm(np.asarray(fg_values[:, 1:3], dtype=np.float32) - 128.0, axis=1)
    bg_chroma = np.linalg.norm(np.asarray(bg_values[:, 1:3], dtype=np.float32) - 128.0, axis=1)
    reference_chroma = bg_chroma
    if pedestrian_reference_values is not None and pedestrian_reference_values.shape[0] > 0:
        ped_chroma = np.linalg.norm(
            np.asarray(pedestrian_reference_values[:, 1:3], dtype=np.float32) - 128.0,
            axis=1,
        )
        reference_chroma = (
            (1.0 - float(color_match_settings.pedestrian_reference_weight)) * bg_chroma.mean()
            + float(color_match_settings.pedestrian_reference_weight) * ped_chroma.mean()
        )
    else:
        reference_chroma = float(np.mean(reference_chroma))
    fg_chroma_mean = float(np.mean(fg_chroma))
    target_chroma_scale = 1.0
    if fg_chroma_mean > 1e-6:
        target_chroma_scale = float(
            np.clip(
                (1.0 - float(color_match_settings.saturation_attenuation_strength))
                + float(color_match_settings.saturation_attenuation_strength)
                * (float(reference_chroma) / fg_chroma_mean),
                0.55,
                1.10,
            )
        )
    params = _clamp_color_match_parameters(
        ColorMatchParameters(
            luminance_mean_delta=float(target_mean[0] - fg_mean[0]),
            luminance_std_ratio=float(
                max(
                    float(
                        fg_std[0]
                        + float(color_match_settings.contrast_attenuation_strength)
                        * (target_std[0] - fg_std[0])
                    ),
                    1e-6,
                )
                / max(float(fg_std[0]), 1e-6)
            ),
            chroma_a_shift=float(target_mean[1] - fg_mean[1]),
            chroma_b_shift=float(target_mean[2] - fg_mean[2]),
            chroma_scale=target_chroma_scale,
        ),
        clamp_settings=clamp_settings,
        small_mask_mode=small_mask_mode,
        allow_luminance_scale=allow_luminance_scale,
    )

    debug = None
    if color_match_settings.write_diagnostics:
        debug = {
            "lab_foreground_mean_before": [float(v) for v in fg_mean.tolist()],
            "lab_foreground_std_before": [float(v) for v in fg_std.tolist()],
            "lab_background_mean": [float(v) for v in bg_mean.tolist()],
            "lab_background_std": [float(v) for v in bg_std.tolist()],
            "lab_target_mean": [float(v) for v in target_mean.tolist()],
            "lab_target_std": [float(v) for v in target_std.tolist()],
            "allow_luminance_scale": bool(allow_luminance_scale),
            "small_mask_mode": bool(small_mask_mode),
            "luminance_strength": float(color_match_settings.luminance_strength),
            "chroma_strength": float(color_match_settings.chroma_strength),
            "pedestrian_reference_pixel_count": int(pedestrian_reference_pixel_count),
            "prefer_pedestrian_reference": bool(color_match_settings.prefer_pedestrian_reference),
            "pedestrian_reference_weight": float(color_match_settings.pedestrian_reference_weight),
            "fallback_scene_reference_weight": float(color_match_settings.fallback_scene_reference_weight),
            "saturation_attenuation_strength": float(color_match_settings.saturation_attenuation_strength),
            "contrast_attenuation_strength": float(color_match_settings.contrast_attenuation_strength),
            "raw_parameters": params.as_dict(),
        }
    return (
        params,
        ColorMatchDiagnostics(
            applied=True,
            skip_reason=None,
            ring_pixel_count_raw=int(np.count_nonzero(ring_raw)),
            ring_pixel_count_filtered=int(bg_values.shape[0]),
            sky_filtered_pixel_count=sky_filtered_pixel_count,
            outlier_filtered_pixel_count=outlier_filtered_pixel_count,
            foreground_pixel_count=foreground_count,
            debug=debug,
        ),
        ring_raw,
        filtered_ring,
    )


def _apply_color_match_parameters(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    color_match_settings: ColorMatchSettings,
    parameters: ColorMatchParameters,
) -> np.ndarray:
    fg_mask = np.asarray(crop_mask, dtype=np.uint8) > 0
    if not np.any(fg_mask):
        return crop_rgb
    crop_lab = cv2.cvtColor(np.asarray(crop_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    corrected = crop_lab.copy()
    fg_pixels = corrected[fg_mask]

    if color_match_settings.luminance_match != "mean_std":
        raise ValueError(
            f"Unsupported color-match luminance match: {color_match_settings.luminance_match}."
        )
    fg_mean = fg_pixels.mean(axis=0)
    fg_std = fg_pixels.std(axis=0)
    fg_l = fg_pixels[:, 0]
    fg_l_std = max(float(fg_std[0]), 1e-6)
    target_mean = float(fg_mean[0]) + float(parameters.luminance_mean_delta)
    target_std = fg_l_std * float(parameters.luminance_std_ratio)
    if abs(float(parameters.luminance_std_ratio) - 1.0) <= 1e-6:
        target_l = fg_l + (target_mean - float(fg_mean[0]))
    else:
        target_l = ((fg_l - float(fg_mean[0])) / fg_l_std) * target_std + target_mean
    fg_pixels[:, 0] = fg_l + float(color_match_settings.luminance_strength) * (target_l - fg_l)

    if color_match_settings.chroma_match != "mean_only":
        raise ValueError(
            f"Unsupported color-match chroma match: {color_match_settings.chroma_match}."
        )
    for channel, shift in (
        (1, float(parameters.chroma_a_shift)),
        (2, float(parameters.chroma_b_shift)),
    ):
        fg_pixels[:, channel] = fg_pixels[:, channel] + (
            float(color_match_settings.chroma_strength) * shift
        )
    chroma_scale = float(getattr(parameters, "chroma_scale", 1.0))
    if abs(chroma_scale - 1.0) > 1e-6:
        fg_pixels[:, 1:3] = 128.0 + (fg_pixels[:, 1:3] - 128.0) * chroma_scale

    corrected[fg_mask] = fg_pixels
    corrected[:, :, 0] = np.clip(corrected[:, :, 0], 0.0, 255.0)
    corrected[:, :, 1:] = np.clip(corrected[:, :, 1:], 0.0, 255.0)
    corrected_rgb = cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2RGB)
    corrected_rgb[~fg_mask] = crop_rgb[~fg_mask]
    return corrected_rgb


def _apply_lab_color_match(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    bbox: CropBounds,
    color_match_settings: ColorMatchSettings,
    clamp_settings: CorrectionClampSettings,
    small_mask_mode: bool,
    full_frame_sky_mask_crop: np.ndarray | None = None,
    full_frame_pedestrian_mask_crop: np.ndarray | None = None,
) -> tuple[np.ndarray, ColorMatchDiagnostics, np.ndarray, np.ndarray]:
    params, diag, ring_raw, ring_filtered = _derive_lab_color_match_parameters(
        crop_rgb,
        crop_mask,
        bbox=bbox,
        color_match_settings=color_match_settings,
        clamp_settings=clamp_settings,
        small_mask_mode=small_mask_mode,
        full_frame_sky_mask_crop=full_frame_sky_mask_crop,
        full_frame_pedestrian_mask_crop=full_frame_pedestrian_mask_crop,
    )
    if params is None:
        return crop_rgb, diag, ring_raw, ring_filtered
    return (
        _apply_color_match_parameters(
            crop_rgb,
            crop_mask,
            color_match_settings=color_match_settings,
            parameters=params,
        ),
        diag,
        ring_raw,
        ring_filtered,
    )


def _smooth_color_match_parameters(
    raw_parameters: ColorMatchParameters,
    *,
    temporal_state: TemporalState,
    temporal_settings: TemporalSmoothingSettings,
) -> ColorMatchParameters:
    previous = temporal_state.color_match_ema
    alpha = float(temporal_settings.color_match_alpha)
    if previous is None:
        smoothed = raw_parameters
    else:
        smoothed = ColorMatchParameters(
            luminance_mean_delta=(
                alpha * previous.luminance_mean_delta
                + (1.0 - alpha) * raw_parameters.luminance_mean_delta
            ),
            luminance_std_ratio=(
                alpha * previous.luminance_std_ratio
                + (1.0 - alpha) * raw_parameters.luminance_std_ratio
            ),
            chroma_a_shift=(
                alpha * previous.chroma_a_shift
                + (1.0 - alpha) * raw_parameters.chroma_a_shift
            ),
            chroma_b_shift=(
                alpha * previous.chroma_b_shift
                + (1.0 - alpha) * raw_parameters.chroma_b_shift
            ),
        )
    temporal_state.color_match_ema = smoothed
    return smoothed


def _estimate_affine_color_transform(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    mask: np.ndarray,
) -> AffineColorTransform | None:
    fg_mask = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(fg_mask):
        return None
    source = np.asarray(source_rgb, dtype=np.float32)[fg_mask]
    target = np.asarray(target_rgb, dtype=np.float32)[fg_mask]
    gain = np.ones((3,), dtype=np.float32)
    bias = np.zeros((3,), dtype=np.float32)
    for channel in range(3):
        x = source[:, channel]
        y = target[:, channel]
        var_x = float(np.var(x))
        if var_x <= 1e-6:
            gain[channel] = 1.0
            bias[channel] = float(np.mean(y) - np.mean(x))
            continue
        cov_xy = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
        gain[channel] = cov_xy / var_x
        bias[channel] = float(np.mean(y) - gain[channel] * np.mean(x))
    return AffineColorTransform(gain=gain, bias=bias)


def _smooth_affine_color_transform(
    transform: AffineColorTransform,
    *,
    temporal_state: TemporalState,
    temporal_settings: TemporalSmoothingSettings,
) -> AffineColorTransform:
    previous = temporal_state.affine_transform_ema
    alpha = float(temporal_settings.appearance_alpha)
    if previous is None:
        smoothed = transform
    else:
        smoothed = AffineColorTransform(
            gain=(alpha * previous.gain) + ((1.0 - alpha) * transform.gain),
            bias=(alpha * previous.bias) + ((1.0 - alpha) * transform.bias),
        )
    temporal_state.affine_transform_ema = smoothed
    return smoothed


def _apply_affine_color_transform(
    source_rgb: np.ndarray,
    mask: np.ndarray,
    transform: AffineColorTransform,
) -> np.ndarray:
    result = np.asarray(source_rgb, dtype=np.float32).copy()
    fg_mask = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(fg_mask):
        return np.asarray(source_rgb, dtype=np.uint8)
    result[fg_mask] = (
        result[fg_mask] * transform.gain.reshape(1, 3)
        + transform.bias.reshape(1, 3)
    )
    result = np.clip(np.rint(result), 0.0, 255.0).astype(np.uint8)
    result[~fg_mask] = np.asarray(source_rgb, dtype=np.uint8)[~fg_mask]
    return result


def _blend_masked_rgb(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    result = np.asarray(source_rgb, dtype=np.uint8).copy()
    fg_mask = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(fg_mask):
        return result
    src = np.asarray(source_rgb, dtype=np.float32)
    tgt = np.asarray(target_rgb, dtype=np.float32)
    blended = np.clip(
        np.rint(((1.0 - float(alpha)) * src) + (float(alpha) * tgt)),
        0.0,
        255.0,
    ).astype(np.uint8)
    result[fg_mask] = blended[fg_mask]
    return result


def _attenuate_affine_color_transform(
    transform: AffineColorTransform,
    *,
    alpha: float,
) -> AffineColorTransform:
    gain = 1.0 + ((transform.gain - 1.0) * float(alpha))
    bias = transform.bias * float(alpha)
    return AffineColorTransform(gain=gain.astype(np.float32), bias=bias.astype(np.float32))


def _recover_postcheck_rejected_crop(
    *,
    pre_harmonization_rgb: np.ndarray,
    rejected_harmonized_rgb: np.ndarray,
    crop_mask: np.ndarray,
    ring_filtered: np.ndarray,
    postcheck_settings: PostcheckSettings,
    small_mask_mode: bool,
    temporal_state: TemporalState | None = None,
    use_continuity_bound: bool = False,
) -> RecoveryDecision | None:
    baseline = _evaluate_postcheck_candidate(
        pre_harmonization_rgb=pre_harmonization_rgb,
        candidate_rgb=pre_harmonization_rgb,
        crop_mask=crop_mask,
        ring_filtered=ring_filtered,
        postcheck_settings=postcheck_settings,
        small_mask_mode=small_mask_mode,
        temporal_state=temporal_state,
        use_continuity_bound=use_continuity_bound,
    )
    if baseline.rejected:
        return None

    best_rgb = np.asarray(pre_harmonization_rgb, dtype=np.uint8)
    best_eval = baseline
    best_alpha = 0.0
    lo = 0.0
    hi = 1.0
    for _ in range(12):
        alpha = (lo + hi) / 2.0
        candidate = _blend_masked_rgb(
            pre_harmonization_rgb,
            rejected_harmonized_rgb,
            crop_mask,
            alpha=alpha,
        )
        evaluation = _evaluate_postcheck_candidate(
            pre_harmonization_rgb=pre_harmonization_rgb,
            candidate_rgb=candidate,
            crop_mask=crop_mask,
            ring_filtered=ring_filtered,
            postcheck_settings=postcheck_settings,
            small_mask_mode=small_mask_mode,
            temporal_state=temporal_state,
            use_continuity_bound=use_continuity_bound,
        )
        if evaluation.rejected:
            hi = alpha
            continue
        lo = alpha
        best_alpha = alpha
        best_rgb = candidate
        best_eval = evaluation
    if best_alpha > 1e-3:
        return RecoveryDecision(
            recovered_rgb=best_rgb,
            mode="bounded_postcheck_blend",
            strength=float(best_alpha),
            evaluation=best_eval,
        )

    estimated_transform = _estimate_affine_color_transform(
        pre_harmonization_rgb,
        rejected_harmonized_rgb,
        crop_mask,
    )
    if estimated_transform is None:
        return None
    best_rgb = np.asarray(pre_harmonization_rgb, dtype=np.uint8)
    best_eval = baseline
    best_alpha = 0.0
    lo = 0.0
    hi = 1.0
    for _ in range(12):
        alpha = (lo + hi) / 2.0
        attenuated = _attenuate_affine_color_transform(estimated_transform, alpha=alpha)
        candidate = _apply_affine_color_transform(
            pre_harmonization_rgb,
            crop_mask,
            attenuated,
        )
        evaluation = _evaluate_postcheck_candidate(
            pre_harmonization_rgb=pre_harmonization_rgb,
            candidate_rgb=candidate,
            crop_mask=crop_mask,
            ring_filtered=ring_filtered,
            postcheck_settings=postcheck_settings,
            small_mask_mode=small_mask_mode,
            temporal_state=temporal_state,
            use_continuity_bound=use_continuity_bound,
        )
        if evaluation.rejected:
            hi = alpha
            continue
        lo = alpha
        best_alpha = alpha
        best_rgb = candidate
        best_eval = evaluation
    if best_alpha > 1e-3:
        return RecoveryDecision(
            recovered_rgb=best_rgb,
            mode="bounded_postcheck_affine",
            strength=float(best_alpha),
            evaluation=best_eval,
        )
    return None


def _diagnostics_root(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}_diagnostics"


def _append_frame_diagnostics(
    diagnostics_dir: Path,
    diagnostics: FrameCropDiagnostics,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with (diagnostics_dir / "frames.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(diagnostics), sort_keys=True))
        handle.write("\n")


def _write_diagnostics_summary(
    diagnostics_dir: Path,
    *,
    temporal_state: TemporalState | None = None,
) -> None:
    frames_path = diagnostics_dir / "frames.jsonl"
    if not frames_path.exists():
        return
    records = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "frame_count": len(records),
        "harmonized_frame_count": sum(
            str(record.get("status", "")).startswith("harmonized_") for record in records
        ),
        "copy_through_frame_count": sum(
            str(record.get("status", "")).startswith("copied_through") for record in records
        ),
        "temporal_reset_frame_count": sum(
            bool(record.get("temporal_reset_applied", False)) for record in records
        ),
        "fallback_frame_count": sum(bool(record.get("fallback_used", False)) for record in records),
    }
    if temporal_state is not None:
        summary["temporal_state"] = {
            "reset_count": int(temporal_state.reset_count),
            "fallback_count": int(temporal_state.fallback_count),
        }
    (diagnostics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tint_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    result = np.asarray(image_rgb, dtype=np.float32).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if np.any(mask_bool):
        overlay = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
        result[mask_bool] = (
            (1.0 - float(alpha)) * result[mask_bool]
            + float(alpha) * overlay.reshape(3)
        )
    return np.clip(np.rint(result), 0.0, 255.0).astype(np.uint8)


def _write_crop_debug_overlay(
    debug_dir: Path,
    *,
    frame_idx: int,
    overlay_rgb: np.ndarray,
    bbox: CropBounds | None,
    crop: CropBounds | None,
    ring_raw: np.ndarray | None = None,
    ring_filtered: np.ndarray | None = None,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_rgb = np.asarray(overlay_rgb, dtype=np.uint8)
    if crop is not None and ring_raw is not None and ring_raw.shape != debug_rgb.shape[:2]:
        expanded = np.zeros(debug_rgb.shape[:2], dtype=bool)
        expanded[crop.top : crop.bottom, crop.left : crop.right] = np.asarray(ring_raw, dtype=bool)
        ring_raw = expanded
    if crop is not None and ring_filtered is not None and ring_filtered.shape != debug_rgb.shape[:2]:
        expanded = np.zeros(debug_rgb.shape[:2], dtype=bool)
        expanded[crop.top : crop.bottom, crop.left : crop.right] = np.asarray(ring_filtered, dtype=bool)
        ring_filtered = expanded
    if ring_raw is not None:
        debug_rgb = _tint_mask(debug_rgb, ring_raw, (80, 120, 255), 0.28)
    if ring_filtered is not None:
        debug_rgb = _tint_mask(debug_rgb, ring_filtered, (96, 220, 96), 0.45)
    image = Image.fromarray(debug_rgb)
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        draw.rectangle(bbox.as_xyxy(), outline=(255, 192, 0), width=2)
    if crop is not None:
        draw.rectangle(crop.as_xyxy(), outline=(64, 220, 96), width=3)
    image.save(debug_dir / f"{frame_idx:06d}.png")


def _copy_through_with_diagnostics(
    *,
    frame_idx: int,
    overlay_rgb: np.ndarray,
    output_dir: Path,
    diagnostics_dir: Path | None,
    bbox_expansion_scale: float,
    min_crop_size_ratio: float,
    max_frame_coverage_ratio: float | None,
    containment_margin_px: int | None,
    fallback_reason: str,
    write_crop_debug_overlays: bool,
    temporal_reset_applied: bool = False,
    temporal_reset_reason: str | None = None,
) -> None:
    _save_rgb_image(output_dir / f"{frame_idx:06d}.png", overlay_rgb)
    diagnostics = FrameCropDiagnostics(
        frame_index=int(frame_idx),
        status="copied_through_empty_mask",
        image_size=[int(overlay_rgb.shape[1]), int(overlay_rgb.shape[0])],
        mask_pixel_count=0,
        bbox_xyxy=None,
        expanded_bbox_xyxy=None,
        crop_xyxy=None,
        crop_size=None,
        bbox_expansion_scale=float(bbox_expansion_scale),
        min_crop_size_ratio=float(min_crop_size_ratio),
        max_frame_coverage_ratio=max_frame_coverage_ratio,
        containment_margin_px=containment_margin_px,
        model_ran=False,
        fallback_reason=fallback_reason,
        temporal_reset_applied=temporal_reset_applied,
        temporal_reset_reason=temporal_reset_reason,
    )
    if diagnostics_dir is not None:
        _append_frame_diagnostics(diagnostics_dir, diagnostics)
        if write_crop_debug_overlays:
            _write_crop_debug_overlay(
                diagnostics_dir / "debug_overlays",
                frame_idx=frame_idx,
                overlay_rgb=overlay_rgb,
                bbox=None,
                crop=None,
                ring_raw=None,
                ring_filtered=None,
            )


def _load_semantics_for_frame(run_dir: Path, frame_idx: int):
    try:
        store = _load_resource_store(run_dir)
        return store.load_semantics2d(int(frame_idx))
    except Exception:
        return None


def _crop_coverage_ratio(bounds: CropBounds, image_shape: tuple[int, int]) -> float:
    image_area = float(max(int(image_shape[0]) * int(image_shape[1]), 1))
    return float((bounds.width * bounds.height) / image_area)


def _threshold_visible_mask(mask_gray: np.ndarray, mask_threshold: float) -> np.ndarray:
    return (
        np.asarray(mask_gray, dtype=np.uint8)
        >= int(round(float(mask_threshold) * 255.0))
    ).astype(np.uint8) * 255


def _model_eligibility_reason(
    *,
    mask_pixel_count: int,
    bbox: CropBounds,
    crop_bounds: CropBounds,
    image_shape: tuple[int, int],
    eligibility_settings: EligibilitySettings,
) -> str | None:
    if mask_pixel_count < int(eligibility_settings.min_visible_mask_pixels_for_model):
        return "small_visible_mask"
    if min(bbox.width, bbox.height) < int(
        eligibility_settings.min_visible_bbox_short_side_px_for_model
    ):
        return "tiny_visible_bbox"
    crop_ratio = _crop_coverage_ratio(crop_bounds, image_shape)
    if (
        crop_ratio > float(eligibility_settings.max_crop_coverage_ratio_for_model)
        and mask_pixel_count
        < int(eligibility_settings.max_crop_coverage_mask_pixels_threshold)
    ):
        return "crop_too_global_for_mask"
    return None


def _tiny_object_conservative_reason(
    *,
    enabled: bool,
    mask_pixel_count: int,
    bbox: CropBounds,
    tiny_object_settings: TinyObjectSettings,
) -> str | None:
    if not enabled:
        return None
    if mask_pixel_count <= int(tiny_object_settings.max_mask_pixels_for_conservative_path):
        return "tiny_object_conservative_mask"
    if min(bbox.width, bbox.height) <= int(
        tiny_object_settings.max_bbox_short_side_px_for_conservative_path
    ):
        return "tiny_object_conservative_bbox"
    return None


def _resolve_temporal_tiny_object_mode(
    *,
    conservative_reason: str | None,
    mask_pixel_count: int,
    bbox: CropBounds,
    tiny_object_settings: TinyObjectSettings,
    temporal_state: TemporalState | None,
) -> str | None:
    if not tiny_object_settings.enabled or temporal_state is None:
        return conservative_reason
    activation_mask_threshold = max(
        int(tiny_object_settings.max_mask_pixels_for_conservative_path) + 32,
        int(np.ceil(float(tiny_object_settings.max_mask_pixels_for_conservative_path) * 1.1)),
    )
    activation_bbox_threshold = max(
        int(tiny_object_settings.max_bbox_short_side_px_for_conservative_path) + 2,
        int(
            np.ceil(
                float(tiny_object_settings.max_bbox_short_side_px_for_conservative_path) * 1.1
            )
        ),
    )
    bbox_short_side = min(int(bbox.width), int(bbox.height))
    if conservative_reason is not None:
        temporal_state.tiny_object_mode = True
        temporal_state.tiny_object_release_streak = 0
        return conservative_reason
    if mask_pixel_count <= activation_mask_threshold or bbox_short_side <= activation_bbox_threshold:
        temporal_state.tiny_object_mode = True
        temporal_state.tiny_object_release_streak = 0
        return "tiny_object_temporal_hold"
    if not temporal_state.tiny_object_mode:
        temporal_state.tiny_object_release_streak = 0
        return None

    release_mask_threshold = max(
        int(tiny_object_settings.max_mask_pixels_for_conservative_path) + 64,
        int(np.ceil(float(tiny_object_settings.max_mask_pixels_for_conservative_path) * 1.5)),
    )
    release_bbox_threshold = max(
        int(tiny_object_settings.max_bbox_short_side_px_for_conservative_path) + 4,
        int(
            np.ceil(
                float(tiny_object_settings.max_bbox_short_side_px_for_conservative_path) * 1.25
            )
        ),
    )
    if mask_pixel_count > release_mask_threshold and bbox_short_side > release_bbox_threshold:
        temporal_state.tiny_object_release_streak += 1
        if temporal_state.tiny_object_release_streak >= 2:
            temporal_state.tiny_object_mode = False
            temporal_state.tiny_object_release_streak = 0
            return None
    else:
        temporal_state.tiny_object_release_streak = 0
    return "tiny_object_temporal_hold"


def _analyze_frame_for_track_policy(
    *,
    frame_idx: int,
    mask_gray: np.ndarray,
    mask_threshold: float,
    bbox_expansion_scale: float,
    min_crop_size_ratio: float | None = None,
    min_crop_size_px: int | None = None,
    max_frame_coverage_ratio: float = 0.85,
    eligibility_settings: EligibilitySettings,
    tiny_object_settings: TinyObjectSettings,
) -> TrackSpanFrameAnalysis | None:
    mask = _threshold_visible_mask(mask_gray, mask_threshold)
    bbox, _, crop_bounds = _compute_local_crop(
        mask,
        bbox_expansion_scale=bbox_expansion_scale,
        min_crop_size_ratio=min_crop_size_ratio,
        min_crop_size_px=min_crop_size_px,
        max_frame_coverage_ratio=max_frame_coverage_ratio,
    )
    if bbox is None or crop_bounds is None:
        return None
    mask_pixel_count = int(np.count_nonzero(mask))
    conservative_reason = _tiny_object_conservative_reason(
        enabled=bool(tiny_object_settings.enabled),
        mask_pixel_count=mask_pixel_count,
        bbox=bbox,
        tiny_object_settings=tiny_object_settings,
    )
    eligibility_reason = None
    if conservative_reason is None:
        eligibility_reason = _model_eligibility_reason(
            mask_pixel_count=mask_pixel_count,
            bbox=bbox,
            crop_bounds=crop_bounds,
            image_shape=mask.shape[:2],
            eligibility_settings=eligibility_settings,
        )
    return TrackSpanFrameAnalysis(
        frame_index=int(frame_idx),
        mask_pixel_count=int(mask_pixel_count),
        bbox=bbox,
        crop_bounds=crop_bounds,
        stable_reference_eligible=(
            conservative_reason is None and eligibility_reason is None
        ),
    )


def _discover_track_span_policies(
    *,
    mask_frames: dict[int, Path],
    mask_threshold: float,
    bbox_expansion_scale: float,
    min_crop_size_ratio: float | None = None,
    min_crop_size_px: int | None = None,
    max_frame_coverage_ratio: float = 0.85,
    eligibility_settings: EligibilitySettings,
    tiny_object_settings: TinyObjectSettings,
) -> dict[int, TrackSpanDecision]:
    decisions: dict[int, TrackSpanDecision] = {}
    spans: list[list[TrackSpanFrameAnalysis]] = []
    current_span: list[TrackSpanFrameAnalysis] = []
    span_id = 0

    def _flush_span(items: list[TrackSpanFrameAnalysis]) -> None:
        if not items:
            return
        spans.append(list(items))

    previous_visible_idx: int | None = None
    for frame_idx in sorted(mask_frames):
        mask_gray = np.asarray(Image.open(mask_frames[frame_idx]).convert("L"), dtype=np.uint8)
        analysis = _analyze_frame_for_track_policy(
            frame_idx=int(frame_idx),
            mask_gray=mask_gray,
            mask_threshold=mask_threshold,
            bbox_expansion_scale=bbox_expansion_scale,
            min_crop_size_ratio=min_crop_size_ratio,
            min_crop_size_px=min_crop_size_px,
            max_frame_coverage_ratio=max_frame_coverage_ratio,
            eligibility_settings=eligibility_settings,
            tiny_object_settings=tiny_object_settings,
        )
        if analysis is None:
            _flush_span(current_span)
            current_span = []
            previous_visible_idx = None
            continue
        if previous_visible_idx is not None and int(frame_idx) != (previous_visible_idx + 1):
            _flush_span(current_span)
            current_span = []
        current_span.append(analysis)
        previous_visible_idx = int(frame_idx)

    _flush_span(current_span)
    global_reference_frames = tuple(
        int(item.frame_index)
        for span in spans
        for item in span
        if item.stable_reference_eligible
    )
    policy = (
        "learned_track_harmonization"
        if len(global_reference_frames) >= 1
        else "conservative_track_harmonization"
    )
    downgrade_reason = None if global_reference_frames else "no_stable_reference_frame"
    seed_reference = None if not global_reference_frames else int(global_reference_frames[0])
    for span in spans:
        decision = TrackSpanDecision(
            track_id=0,
            span_id=int(span_id),
            frame_indices=tuple(int(item.frame_index) for item in span),
            policy=policy,
            reference_frame_indices=global_reference_frames,
            seed_reference_frame_index=seed_reference,
            downgrade_reason=downgrade_reason,
        )
        for item in span:
            decisions[int(item.frame_index)] = decision
        span_id += 1
    return decisions


def _tensors_from_named_argument_dict(
    arguments: dict[str, float],
    *,
    device: str,
) -> tuple[list[torch.Tensor], list[str]]:
    filter_names = [name for name in DEFAULT_FILTER_NAMES if name in arguments]
    if not filter_names:
        filter_names = [str(name) for name in arguments.keys()]
    tensors = [
        torch.tensor([[float(arguments[name])]], dtype=torch.float32, device=device)
        for name in filter_names
    ]
    return tensors, filter_names


def _interpolate_scalar_series(
    reference_values: dict[int, float],
    target_frames: list[int],
) -> tuple[dict[int, float], dict[int, str]]:
    if not reference_values:
        return {}, {}
    ordered_refs = sorted((int(frame), float(value)) for frame, value in reference_values.items())
    ref_frames = [frame for frame, _ in ordered_refs]
    ref_vals = [value for _, value in ordered_refs]
    values: dict[int, float] = {}
    sources: dict[int, str] = {}
    for frame_idx in target_frames:
        frame = int(frame_idx)
        if frame in reference_values:
            values[frame] = float(reference_values[frame])
            sources[frame] = "direct_reference"
            continue
        if frame < ref_frames[0]:
            values[frame] = float(ref_vals[0])
            sources[frame] = "backfilled_from_future"
            continue
        if frame > ref_frames[-1]:
            values[frame] = float(ref_vals[-1])
            sources[frame] = "forward_propagated"
            continue
        for left_idx in range(len(ref_frames) - 1):
            left_frame = ref_frames[left_idx]
            right_frame = ref_frames[left_idx + 1]
            if left_frame <= frame <= right_frame:
                alpha = float(frame - left_frame) / float(max(right_frame - left_frame, 1))
                values[frame] = float((1.0 - alpha) * ref_vals[left_idx] + alpha * ref_vals[left_idx + 1])
                sources[frame] = "interpolated"
                break
    return values, sources


def _triangular_smooth_track_values(
    values_by_frame: dict[int, float],
    *,
    window_radius: int = 2,
) -> dict[int, float]:
    frames = sorted(int(frame) for frame in values_by_frame)
    if len(frames) <= 2 or window_radius <= 0:
        return {int(frame): float(values_by_frame[int(frame)]) for frame in frames}
    smoothed: dict[int, float] = {}
    for index, frame in enumerate(frames):
        weighted_sum = 0.0
        total_weight = 0.0
        for neighbor_index in range(
            max(0, index - window_radius),
            min(len(frames), index + window_radius + 1),
        ):
            distance = abs(neighbor_index - index)
            weight = float(window_radius + 1 - distance)
            neighbor_frame = frames[neighbor_index]
            weighted_sum += weight * float(values_by_frame[neighbor_frame])
            total_weight += weight
        smoothed[frame] = float(weighted_sum / max(total_weight, 1e-6))
    return smoothed


def _fit_track_argument_curve(
    reference_arguments: dict[int, dict[str, float]],
    target_frames: list[int],
) -> tuple[dict[int, dict[str, float]], dict[int, str]]:
    if not reference_arguments:
        return {}, {}
    names = sorted({name for params in reference_arguments.values() for name in params.keys()})
    per_name_values: dict[str, dict[int, float]] = {}
    source_votes: dict[int, dict[str, int]] = {int(frame): {} for frame in target_frames}
    for name in names:
        series = {
            int(frame): float(params[name])
            for frame, params in reference_arguments.items()
            if name in params
        }
        interpolated, sources = _interpolate_scalar_series(series, target_frames)
        smoothed = _triangular_smooth_track_values(interpolated)
        per_name_values[name] = smoothed
        for frame, source in sources.items():
            source_votes[int(frame)][source] = source_votes[int(frame)].get(source, 0) + 1
    result: dict[int, dict[str, float]] = {}
    source_by_frame: dict[int, str] = {}
    for frame in target_frames:
        frame_int = int(frame)
        result[frame_int] = {
            name: float(per_name_values[name][frame_int])
            for name in names
            if frame_int in per_name_values[name]
        }
        votes = source_votes.get(frame_int, {})
        source_by_frame[frame_int] = max(
            votes.items(),
            key=lambda item: (item[1], item[0] == "direct_reference", item[0]),
        )[0] if votes else "none"
    return result, source_by_frame


def _fit_track_color_parameter_curve(
    reference_parameters: dict[int, ColorMatchParameters],
    target_frames: list[int],
) -> dict[int, ColorMatchParameters]:
    if not reference_parameters:
        return {}
    fields = (
        "luminance_mean_delta",
        "luminance_std_ratio",
        "chroma_a_shift",
        "chroma_b_shift",
        "chroma_scale",
    )
    per_field: dict[str, dict[int, float]] = {}
    for field_name in fields:
        series = {
            int(frame): float(getattr(params, field_name))
            for frame, params in reference_parameters.items()
        }
        interpolated, _ = _interpolate_scalar_series(series, target_frames)
        per_field[field_name] = _triangular_smooth_track_values(interpolated)
    return {
        int(frame): ColorMatchParameters(
            luminance_mean_delta=float(per_field["luminance_mean_delta"][int(frame)]),
            luminance_std_ratio=float(per_field["luminance_std_ratio"][int(frame)]),
            chroma_a_shift=float(per_field["chroma_a_shift"][int(frame)]),
            chroma_b_shift=float(per_field["chroma_b_shift"][int(frame)]),
            chroma_scale=float(per_field["chroma_scale"][int(frame)]),
        )
        for frame in target_frames
    }


def process_frame(
    *,
    frame_idx: int,
    overlay_rgb: np.ndarray,
    mask_gray: np.ndarray,
    run_dir: Path | None,
    output_dir: Path,
    harmonizer: HarmonizerProtocol,
    device: str,
    mask_threshold: float,
    bbox_expansion_scale: float,
    min_crop_size_ratio: float | None = None,
    min_crop_size_px: int | None = None,
    empty_mask_behavior: str,
    eligibility_settings: EligibilitySettings,
    color_match_settings: ColorMatchSettings,
    correction_clamp_settings: CorrectionClampSettings,
    postcheck_settings: PostcheckSettings,
    tiny_object_settings: TinyObjectSettings = TinyObjectSettings(),
    adaptive_settings: AdaptiveSettings = AdaptiveSettings(),
    diagnostics_dir: Path | None,
    write_crop_debug_overlays: bool,
    oversized_actor_settings: OversizedActorSettings = OversizedActorSettings(),
    temporal_settings: TemporalSmoothingSettings | None = None,
    temporal_state: TemporalState | None = None,
    span_id: int | None = None,
    span_policy: str | None = None,
    reference_frame_for_track: int | None = None,
    track_downgrade_reason: str | None = None,
    force_propagated_arguments: dict[str, float] | None = None,
    force_propagated_color_match_parameters: ColorMatchParameters | None = None,
    track_id: int | None = None,
    used_for_reference_estimation: bool = False,
    applied_parameter_source: str | None = None,
) -> ProcessFrameArtifacts:
    if overlay_rgb.shape[:2] != mask_gray.shape[:2]:
        raise ValueError(
            f"Overlay/mask shape mismatch at frame {frame_idx}: "
            f"overlay={overlay_rgb.shape[:2]} mask={mask_gray.shape[:2]}"
        )

    mask = (mask_gray >= int(round(float(mask_threshold) * 255.0))).astype(np.uint8) * 255
    effective_min_crop_size_ratio = (
        float(min_crop_size_ratio) if min_crop_size_ratio is not None else 0.0
    )
    bbox, expanded_bbox, crop_bounds = _compute_local_crop(
        mask,
        bbox_expansion_scale=bbox_expansion_scale,
        min_crop_size_ratio=min_crop_size_ratio,
        min_crop_size_px=min_crop_size_px,
        max_frame_coverage_ratio=float(oversized_actor_settings.max_frame_coverage_ratio),
    )
    temporal_reset_applied = False
    temporal_reset_reason: str | None = None
    if crop_bounds is None:
        if empty_mask_behavior != "copy_through":
            raise ValueError(
                f"Unsupported empty_mask_behavior={empty_mask_behavior!r} at frame {frame_idx}."
            )
        if (
            temporal_settings is not None
            and temporal_settings.enabled
            and temporal_state is not None
            and temporal_settings.reset_on_empty_mask
        ):
            temporal_state.reset()
            temporal_reset_applied = True
            temporal_reset_reason = "empty_visible_mask"
        _copy_through_with_diagnostics(
            frame_idx=frame_idx,
            overlay_rgb=overlay_rgb,
            output_dir=output_dir,
            diagnostics_dir=diagnostics_dir,
            bbox_expansion_scale=bbox_expansion_scale,
            min_crop_size_ratio=effective_min_crop_size_ratio,
            max_frame_coverage_ratio=float(oversized_actor_settings.max_frame_coverage_ratio),
            containment_margin_px=int(oversized_actor_settings.containment_margin_px),
            fallback_reason="empty_visible_mask",
            write_crop_debug_overlays=write_crop_debug_overlays,
            temporal_reset_applied=temporal_reset_applied,
            temporal_reset_reason=temporal_reset_reason,
        )
        return ProcessFrameArtifacts(frame_index=int(frame_idx))

    mask_pixel_count = int(np.count_nonzero(mask))
    crop_coverage_ratio = _crop_coverage_ratio(crop_bounds, overlay_rgb.shape[:2])
    visible_mask_outside_crop_pixels = _count_visible_mask_outside_crop(mask, crop_bounds)
    actor_fully_contained_in_crop = _crop_contains_bbox(
        crop_bounds,
        bbox,
        margin_px=0,
    )
    crop_insufficient_reason: str | None = None
    if (
        bool(oversized_actor_settings.reject_when_actor_exceeds_crop)
        and visible_mask_outside_crop_pixels > 0
    ):
        crop_insufficient_reason = "visible_mask_exceeds_crop"
    conservative_reason = (
        None
        if bbox is None
        else _tiny_object_conservative_reason(
            enabled=bool(tiny_object_settings.enabled),
            mask_pixel_count=mask_pixel_count,
            bbox=bbox,
            tiny_object_settings=tiny_object_settings,
        )
    )
    crop_rgb = _crop_rgb(overlay_rgb, crop_bounds)
    crop_mask = _crop_mask(mask, crop_bounds)
    scene_style_metrics = _estimate_scene_style_metrics(crop_rgb, crop_mask)
    continuity = TemporalContinuityStats(
        crop_iou=None,
        mask_area_ratio=None,
        centroid_jump_fraction=None,
    )
    if temporal_settings is not None and temporal_settings.enabled and temporal_state is not None:
        tiny_object_temporal_mode_active = bool(
            conservative_reason is not None or temporal_state.tiny_object_mode
        )
        current_crop_stats = _build_temporal_crop_stats(crop_mask, crop_bounds)
        (
            temporal_reset_applied,
            temporal_reset_reason,
            continuity,
        ) = _maybe_reset_temporal_state(
            temporal_state,
            current_crop_stats,
            temporal_settings,
            ignore_mask_area_ratio=tiny_object_temporal_mode_active,
        )
    else:
        current_crop_stats = None
    if bbox is not None:
        conservative_reason = _resolve_temporal_tiny_object_mode(
            conservative_reason=conservative_reason,
            mask_pixel_count=mask_pixel_count,
            bbox=bbox,
            tiny_object_settings=tiny_object_settings,
            temporal_state=temporal_state,
        )
    eligibility_reason = conservative_reason
    if eligibility_reason is None and crop_insufficient_reason is not None:
        eligibility_reason = "crop_insufficient_for_actor"
    if eligibility_reason is None and bbox is not None:
        eligibility_reason = _model_eligibility_reason(
            mask_pixel_count=mask_pixel_count,
            bbox=bbox,
            crop_bounds=crop_bounds,
            image_shape=overlay_rgb.shape[:2],
            eligibility_settings=eligibility_settings,
        )
    learned_model_eligible = eligibility_reason is None
    small_mask_mode = not learned_model_eligible
    color_match_diag = ColorMatchDiagnostics(
        applied=False,
        skip_reason="disabled",
        ring_pixel_count_raw=0,
        ring_pixel_count_filtered=0,
        sky_filtered_pixel_count=0,
        outlier_filtered_pixel_count=0,
        foreground_pixel_count=int(np.count_nonzero(crop_mask)),
        debug=None,
    )
    ring_raw = np.zeros_like(crop_mask, dtype=bool)
    ring_filtered = np.zeros_like(crop_mask, dtype=bool)
    raw_color_parameters: ColorMatchParameters | None = None
    smoothed_color_parameters: ColorMatchParameters | None = None
    pre_harmonization_rgb = crop_rgb
    adaptive_policy = _compute_adaptive_policy(
        adaptive_settings=adaptive_settings,
        scene_style_metrics=scene_style_metrics,
        mask_pixel_count=mask_pixel_count,
        bbox=bbox,
        color_match_settings=color_match_settings,
        ring_pixel_count_filtered=0,
        applied_parameter_source=applied_parameter_source,
    )
    color_match_enabled = bool(color_match_settings.enabled) and crop_insufficient_reason is None
    if (
        color_match_enabled
        and (
            (
                conservative_reason is not None
                and force_propagated_color_match_parameters is None
            )
            or (
                bool(tiny_object_settings.enabled)
                and mask_pixel_count
                <= int(tiny_object_settings.skip_color_match_below_mask_pixels)
                and force_propagated_color_match_parameters is None
            )
        )
    ):
        color_match_enabled = False
        color_match_diag = ColorMatchDiagnostics(
            applied=False,
            skip_reason=(
                "tiny_object_temporal_conservative_skip"
                if conservative_reason == "tiny_object_temporal_hold"
                else "tiny_object_conservative_skip"
            ),
            ring_pixel_count_raw=0,
            ring_pixel_count_filtered=0,
            sky_filtered_pixel_count=0,
            outlier_filtered_pixel_count=0,
            foreground_pixel_count=int(np.count_nonzero(crop_mask)),
            debug=None,
        )
    if force_propagated_color_match_parameters is not None:
        adaptive_policy = _compute_adaptive_policy(
            adaptive_settings=adaptive_settings,
            scene_style_metrics=scene_style_metrics,
            mask_pixel_count=mask_pixel_count,
            bbox=bbox,
            color_match_settings=color_match_settings,
            ring_pixel_count_filtered=0,
            applied_parameter_source=applied_parameter_source,
        )
        smoothed_color_parameters = _scale_color_match_parameters(
            force_propagated_color_match_parameters,
            strength=adaptive_policy.color_match_strength,
        )
        color_match_diag = ColorMatchDiagnostics(
            applied=True,
            skip_reason=None,
            ring_pixel_count_raw=0,
            ring_pixel_count_filtered=0,
            sky_filtered_pixel_count=0,
            outlier_filtered_pixel_count=0,
            foreground_pixel_count=int(np.count_nonzero(crop_mask)),
            debug={
                "mode": "track_propagated",
                "adaptive_color_match_strength": float(adaptive_policy.color_match_strength),
            },
        )
        pre_harmonization_rgb = _apply_color_match_parameters(
            crop_rgb,
            crop_mask,
            color_match_settings=color_match_settings,
            parameters=smoothed_color_parameters,
        )
    elif color_match_enabled:
        local_bbox = CropBounds(
            left=max(0, int(bbox.left - crop_bounds.left)) if bbox is not None else 0,
            top=max(0, int(bbox.top - crop_bounds.top)) if bbox is not None else 0,
            right=min(crop_bounds.width, int(bbox.right - crop_bounds.left)) if bbox is not None else crop_bounds.width,
            bottom=min(crop_bounds.height, int(bbox.bottom - crop_bounds.top)) if bbox is not None else crop_bounds.height,
        )
        sky_mask_crop = None
        pedestrian_mask_crop = None
        if (
            run_dir is not None
            and (
                color_match_settings.use_semantics_for_sky_filter
                or color_match_settings.prefer_pedestrian_reference
            )
        ):
            semantics = _load_semantics_for_frame(run_dir, frame_idx)
            if semantics is not None:
                full_sky_mask = _sky_mask_from_semantics(semantics)
                if full_sky_mask.size and tuple(full_sky_mask.shape) == tuple(mask.shape):
                    sky_mask_crop = np.asarray(
                        full_sky_mask[crop_bounds.top : crop_bounds.bottom, crop_bounds.left : crop_bounds.right],
                        dtype=bool,
                    )
                full_pedestrian_mask = _pedestrian_mask_from_semantics(semantics)
                if full_pedestrian_mask.size and tuple(full_pedestrian_mask.shape) == tuple(mask.shape):
                    pedestrian_mask_crop = np.asarray(
                        full_pedestrian_mask[
                            crop_bounds.top : crop_bounds.bottom,
                            crop_bounds.left : crop_bounds.right,
                        ],
                        dtype=bool,
                    )
        raw_color_parameters, color_match_diag, ring_raw, ring_filtered = _derive_lab_color_match_parameters(
            crop_rgb,
            crop_mask,
            bbox=local_bbox,
            color_match_settings=replace(
                color_match_settings,
                contrast_attenuation_strength=(
                    float(color_match_settings.contrast_attenuation_strength)
                    * (1.0 - adaptive_policy.preserve_contrast)
                ),
                saturation_attenuation_strength=(
                    float(color_match_settings.saturation_attenuation_strength)
                    * (1.0 - adaptive_policy.preserve_saturation)
                ),
            ),
            clamp_settings=correction_clamp_settings,
            small_mask_mode=small_mask_mode,
            full_frame_sky_mask_crop=sky_mask_crop,
            full_frame_pedestrian_mask_crop=pedestrian_mask_crop,
        )
        adaptive_policy = _compute_adaptive_policy(
            adaptive_settings=adaptive_settings,
            scene_style_metrics=scene_style_metrics,
            mask_pixel_count=mask_pixel_count,
            bbox=bbox,
            color_match_settings=color_match_settings,
            ring_pixel_count_filtered=color_match_diag.ring_pixel_count_filtered,
            applied_parameter_source=applied_parameter_source,
        )
        if raw_color_parameters is not None:
            raw_color_parameters = _scale_color_match_parameters(
                raw_color_parameters,
                strength=adaptive_policy.color_match_strength,
            )
            smoothed_color_parameters = raw_color_parameters
            if temporal_settings is not None and temporal_settings.enabled and temporal_state is not None:
                smoothed_color_parameters = _smooth_color_match_parameters(
                    raw_color_parameters,
                    temporal_state=temporal_state,
                    temporal_settings=temporal_settings,
                )
            pre_harmonization_rgb = _apply_color_match_parameters(
                crop_rgb,
                crop_mask,
                color_match_settings=color_match_settings,
                parameters=smoothed_color_parameters,
            )
        elif (
            temporal_settings is not None
            and temporal_settings.enabled
            and temporal_state is not None
            and temporal_state.color_match_ema is not None
            and not temporal_reset_applied
        ):
            smoothed_color_parameters = _scale_color_match_parameters(
                temporal_state.color_match_ema,
                strength=adaptive_policy.color_match_strength,
            )
            pre_harmonization_rgb = _apply_color_match_parameters(
                crop_rgb,
                crop_mask,
                color_match_settings=color_match_settings,
                parameters=smoothed_color_parameters,
            )
    raw_arguments: dict[str, float] | None = None
    smoothed_arguments: dict[str, float] | None = None
    fallback_used = False
    fallback_reason: str | None = None
    fallback_mode: str | None = None
    fallback_transform: AffineColorTransform | None = None
    recovery_mode: str | None = None
    recovery_strength: float | None = None
    recovery_continuity_bound_source: str | None = None
    oversized_actor_behavior_applied: str | None = None
    full_frame_fallback_used = False
    harmonized_crop = pre_harmonization_rgb
    postcheck_rejected = False
    postcheck_reason: str | None = None
    pre_harmonization_masked_luma: float | None = None
    post_harmonization_masked_luma: float | None = None
    rejected_candidate_masked_luma: float | None = None
    local_ring_luma: float | None = None
    status = "harmonized_local_crop"
    propagated_arguments = force_propagated_arguments
    if (
        propagated_arguments is None
        and span_policy == "learned_track_harmonization"
        and temporal_state is not None
        and temporal_state.argument_ema
        and not learned_model_eligible
        and crop_insufficient_reason is None
    ):
        propagated_arguments = {
            name: _argument_scalar(value)
            for name, value in temporal_state.argument_ema.items()
        }

    if crop_insufficient_reason is not None:
        fallback_used = True
        fallback_reason = "crop_insufficient_for_actor"
        oversized_actor_behavior_applied = str(
            oversized_actor_settings.oversized_actor_behavior
        )
        if (
            oversized_actor_settings.oversized_actor_behavior == "full_mask_affine_or_copy"
            and temporal_state is not None
            and temporal_state.affine_transform_ema is not None
            and mask_pixel_count
            >= int(oversized_actor_settings.full_frame_affine_min_mask_pixels)
        ):
            fallback_transform = temporal_state.affine_transform_ema
            harmonized_full = _apply_affine_color_transform(
                overlay_rgb,
                mask,
                fallback_transform,
            )
            _save_rgb_image(output_dir / f"{frame_idx:06d}.png", harmonized_full)
            status = "harmonized_full_mask_affine_crop_insufficient"
            fallback_mode = "full_mask_affine_from_track_ema"
            full_frame_fallback_used = True
            pre_harmonization_masked_luma = _masked_luma(overlay_rgb, mask)
            post_harmonization_masked_luma = _masked_luma(harmonized_full, mask)
        else:
            harmonized_full = np.asarray(overlay_rgb, dtype=np.uint8)
            _save_rgb_image(output_dir / f"{frame_idx:06d}.png", harmonized_full)
            status = "copied_through_crop_insufficient"
            fallback_mode = "copy_through"
            full_frame_fallback_used = True
            pre_harmonization_masked_luma = _masked_luma(overlay_rgb, mask)
            post_harmonization_masked_luma = pre_harmonization_masked_luma
    elif force_propagated_arguments is not None and span_policy == "learned_track_harmonization":
        restore_arguments, filter_names = _tensors_from_named_argument_dict(
            force_propagated_arguments,
            device=device,
        )
        smoothed_arguments = _named_argument_dict(filter_names, restore_arguments)
        raw_arguments = None
        harmonized_crop = _restore_harmonized_crop(
            harmonizer,
            crop_rgb=pre_harmonization_rgb,
            crop_mask=crop_mask,
            arguments=restore_arguments,
            device=device,
        )
        harmonized_crop = _blend_masked_rgb_strength(
            pre_harmonization_rgb,
            harmonized_crop,
            crop_mask,
            strength=adaptive_policy.harmonizer_strength,
        )
        evaluation = _evaluate_postcheck_candidate(
            pre_harmonization_rgb=pre_harmonization_rgb,
            candidate_rgb=harmonized_crop,
            crop_mask=crop_mask,
            ring_filtered=ring_filtered,
            postcheck_settings=postcheck_settings,
            small_mask_mode=small_mask_mode,
            temporal_state=temporal_state,
            use_continuity_bound=True,
        )
        postcheck_rejected = evaluation.rejected
        postcheck_reason = evaluation.reason
        pre_harmonization_masked_luma = evaluation.before_luma
        post_harmonization_masked_luma = evaluation.after_luma
        rejected_candidate_masked_luma = evaluation.after_luma
        local_ring_luma = evaluation.ring_luma
        if postcheck_rejected:
            recovery = _recover_postcheck_rejected_crop(
                pre_harmonization_rgb=pre_harmonization_rgb,
                rejected_harmonized_rgb=harmonized_crop,
                crop_mask=crop_mask,
                ring_filtered=ring_filtered,
                postcheck_settings=postcheck_settings,
                small_mask_mode=small_mask_mode,
                temporal_state=temporal_state,
                use_continuity_bound=True,
            )
            fallback_used = True
            fallback_mode = "bounded_masked_correction"
            if recovery is not None:
                harmonized_crop = recovery.recovered_rgb
                recovery_mode = recovery.mode
                recovery_strength = recovery.strength
                recovery_continuity_bound_source = recovery.evaluation.continuity_bound_source
                post_harmonization_masked_luma = recovery.evaluation.after_luma
                local_ring_luma = recovery.evaluation.ring_luma
                status = "harmonized_track_recovered_blend"
                if recovery.mode == "bounded_postcheck_affine":
                    status = "harmonized_track_recovered_affine"
            else:
                harmonized_crop = pre_harmonization_rgb
                status = "harmonized_track_conservative"
                post_harmonization_masked_luma = pre_harmonization_masked_luma
        else:
            status = "harmonized_track_applied"
        fallback_reason = None
    elif learned_model_eligible:
        try:
            predicted_arguments, filter_names = _predict_harmonizer_arguments(
                harmonizer,
                crop_rgb=pre_harmonization_rgb,
                crop_mask=crop_mask,
                device=device,
            )
            raw_arguments = _named_argument_dict(filter_names, predicted_arguments)
            restore_arguments = predicted_arguments
            if temporal_settings is not None and temporal_settings.enabled and temporal_state is not None:
                restore_arguments = _smooth_harmonizer_arguments(
                    predicted_arguments,
                    filter_names=filter_names,
                    temporal_state=temporal_state,
                    temporal_settings=temporal_settings,
                )
                smoothed_arguments = _named_argument_dict(filter_names, restore_arguments)
            else:
                smoothed_arguments = raw_arguments
            try:
                harmonized_crop = _restore_harmonized_crop(
                    harmonizer,
                    crop_rgb=pre_harmonization_rgb,
                    crop_mask=crop_mask,
                    arguments=restore_arguments,
                    device=device,
                )
                harmonized_crop = _blend_masked_rgb_strength(
                    pre_harmonization_rgb,
                    harmonized_crop,
                    crop_mask,
                    strength=adaptive_policy.harmonizer_strength,
                )
            except Exception:
                if temporal_settings is None or temporal_state is None:
                    raise
                raw_harmonized_crop = _restore_harmonized_crop(
                    harmonizer,
                    crop_rgb=pre_harmonization_rgb,
                    crop_mask=crop_mask,
                    arguments=predicted_arguments,
                    device=device,
                )
                estimated_transform = _estimate_affine_color_transform(
                    pre_harmonization_rgb,
                    raw_harmonized_crop,
                    crop_mask,
                )
                if estimated_transform is None:
                    raise
                fallback_transform = _smooth_affine_color_transform(
                    estimated_transform,
                    temporal_state=temporal_state,
                    temporal_settings=temporal_settings,
                )
                harmonized_crop = _apply_affine_color_transform(
                    pre_harmonization_rgb,
                    crop_mask,
                    fallback_transform,
                )
                harmonized_crop = _blend_masked_rgb_strength(
                    pre_harmonization_rgb,
                    harmonized_crop,
                    crop_mask,
                    strength=adaptive_policy.harmonizer_strength,
                )
                fallback_used = True
                fallback_mode = temporal_settings.fallback_mode
                temporal_state.fallback_count += 1
        except Exception:
            if (
                temporal_settings is not None
                and temporal_settings.enabled
                and temporal_state is not None
                and temporal_settings.reset_on_harmonizer_failure
            ):
                temporal_state.reset()
            raise
        evaluation = _evaluate_postcheck_candidate(
            pre_harmonization_rgb=pre_harmonization_rgb,
            candidate_rgb=harmonized_crop,
            crop_mask=crop_mask,
            ring_filtered=ring_filtered,
            postcheck_settings=postcheck_settings,
            small_mask_mode=small_mask_mode,
            temporal_state=temporal_state,
            use_continuity_bound=span_policy == "learned_track_harmonization",
        )
        postcheck_rejected = evaluation.rejected
        postcheck_reason = evaluation.reason
        pre_harmonization_masked_luma = evaluation.before_luma
        post_harmonization_masked_luma = evaluation.after_luma
        rejected_candidate_masked_luma = evaluation.after_luma
        local_ring_luma = evaluation.ring_luma
        if postcheck_rejected:
            fallback_used = True
            fallback_mode = "bounded_masked_correction"
            if span_policy == "learned_track_harmonization":
                recovery = _recover_postcheck_rejected_crop(
                    pre_harmonization_rgb=pre_harmonization_rgb,
                    rejected_harmonized_rgb=harmonized_crop,
                    crop_mask=crop_mask,
                    ring_filtered=ring_filtered,
                    postcheck_settings=postcheck_settings,
                    small_mask_mode=small_mask_mode,
                    temporal_state=temporal_state,
                    use_continuity_bound=True,
                )
                if recovery is not None:
                    harmonized_crop = recovery.recovered_rgb
                    recovery_mode = recovery.mode
                    recovery_strength = recovery.strength
                    recovery_continuity_bound_source = recovery.evaluation.continuity_bound_source
                    post_harmonization_masked_luma = recovery.evaluation.after_luma
                    local_ring_luma = recovery.evaluation.ring_luma
                    status = "harmonized_track_recovered_blend"
                    if recovery.mode == "bounded_postcheck_affine":
                        status = "harmonized_track_recovered_affine"
                else:
                    harmonized_crop = pre_harmonization_rgb
                    status = "harmonized_track_conservative"
                    post_harmonization_masked_luma = pre_harmonization_masked_luma
            else:
                harmonized_crop = pre_harmonization_rgb
                status = "fallback_postcheck_rejected"
                post_harmonization_masked_luma = pre_harmonization_masked_luma
        elif span_policy == "learned_track_harmonization":
            status = "harmonized_track_applied"
    else:
        fallback_used = True
        fallback_mode = "bounded_masked_correction"
        fallback_reason = f"fallback_{eligibility_reason}"
        status = (
            "harmonized_track_conservative"
            if span_policy == "conservative_track_harmonization"
            else fallback_reason
        )
        pre_harmonization_masked_luma = _masked_luma(pre_harmonization_rgb, crop_mask)
        post_harmonization_masked_luma = pre_harmonization_masked_luma
        local_ring_luma = _ring_luma(pre_harmonization_rgb, ring_filtered)

    if temporal_state is not None and current_crop_stats is not None:
        temporal_state.previous_crop_stats = current_crop_stats
        if fallback_used and postcheck_rejected:
            temporal_state.fallback_count += 1
        if status in {
            "harmonized_local_crop",
            "harmonized_track_applied",
            "harmonized_track_recovered_blend",
            "harmonized_track_recovered_affine",
            "harmonized_full_mask_affine_crop_insufficient",
        }:
            temporal_state.last_accepted_masked_luma = post_harmonization_masked_luma
    if crop_insufficient_reason is None:
        harmonized_full = _paste_crop(overlay_rgb, harmonized_crop, crop_bounds)
        _save_rgb_image(output_dir / f"{frame_idx:06d}.png", harmonized_full)
        if (
            temporal_state is not None
            and temporal_settings is not None
            and temporal_settings.enabled
            and status in {
                "harmonized_local_crop",
                "harmonized_track_applied",
                "harmonized_track_recovered_blend",
                "harmonized_track_recovered_affine",
            }
        ):
            estimated_transform = _estimate_affine_color_transform(
                pre_harmonization_rgb,
                harmonized_crop,
                crop_mask,
            )
            if estimated_transform is not None:
                _smooth_affine_color_transform(
                    estimated_transform,
                    temporal_state=temporal_state,
                    temporal_settings=temporal_settings,
                )

    application_mode = None
    if status == "harmonized_local_crop":
        application_mode = "learned_local"
    elif status == "harmonized_track_applied":
        application_mode = "track_applied"
    elif status == "harmonized_track_recovered_blend":
        application_mode = "track_recovered_blend"
    elif status == "harmonized_track_recovered_affine":
        application_mode = "track_recovered_affine"
    elif status == "harmonized_track_conservative":
        application_mode = "track_conservative"
    elif status == "harmonized_full_mask_affine_crop_insufficient":
        application_mode = "full_mask_affine_fallback"
    elif str(status).startswith("fallback_"):
        application_mode = "fallback"

    if diagnostics_dir is not None:
        diagnostics = FrameCropDiagnostics(
            frame_index=int(frame_idx),
            status=status,
            image_size=[int(overlay_rgb.shape[1]), int(overlay_rgb.shape[0])],
            mask_pixel_count=mask_pixel_count,
            bbox_xyxy=None if bbox is None else bbox.as_xyxy(),
            expanded_bbox_xyxy=None if expanded_bbox is None else expanded_bbox.as_xyxy(),
            crop_xyxy=crop_bounds.as_xyxy(),
            crop_size=[crop_bounds.width, crop_bounds.height],
            crop_coverage_ratio=float(crop_coverage_ratio),
            bbox_expansion_scale=float(bbox_expansion_scale),
            min_crop_size_ratio=float(effective_min_crop_size_ratio),
            max_frame_coverage_ratio=float(oversized_actor_settings.max_frame_coverage_ratio),
            containment_margin_px=int(oversized_actor_settings.containment_margin_px),
            model_ran=bool(learned_model_eligible),
            actor_fully_contained_in_crop=bool(actor_fully_contained_in_crop),
            visible_mask_outside_crop_pixels=int(visible_mask_outside_crop_pixels),
            crop_insufficient_reason=crop_insufficient_reason,
            oversized_actor_behavior_applied=oversized_actor_behavior_applied,
            full_frame_fallback_used=bool(full_frame_fallback_used),
            learned_model_eligible=bool(learned_model_eligible),
            eligibility_reason=eligibility_reason,
            fallback_reason=fallback_reason,
            color_match_applied=color_match_diag.applied,
            color_match_skip_reason=color_match_diag.skip_reason,
            ring_pixel_count_raw=color_match_diag.ring_pixel_count_raw,
            ring_pixel_count_filtered=color_match_diag.ring_pixel_count_filtered,
            sky_filtered_pixel_count=color_match_diag.sky_filtered_pixel_count,
            outlier_filtered_pixel_count=color_match_diag.outlier_filtered_pixel_count,
            foreground_pixel_count=color_match_diag.foreground_pixel_count,
            color_match_debug=color_match_diag.debug,
            temporal_smoothing_applied=bool(
                temporal_settings is not None and temporal_settings.enabled
            ),
            temporal_reset_applied=temporal_reset_applied,
            temporal_reset_reason=temporal_reset_reason,
            raw_harmonizer_arguments=raw_arguments,
            smoothed_harmonizer_arguments=smoothed_arguments,
            raw_color_match_parameters=(
                None if raw_color_parameters is None else raw_color_parameters.as_dict()
            ),
            smoothed_color_match_parameters=(
                None
                if smoothed_color_parameters is None
                else smoothed_color_parameters.as_dict()
            ),
            fallback_used=fallback_used,
            fallback_mode=fallback_mode,
            fallback_transform=(
                None if fallback_transform is None else fallback_transform.as_dict()
            ),
            crop_iou_with_previous=continuity.crop_iou,
            mask_area_ratio_with_previous=continuity.mask_area_ratio,
            centroid_jump_fraction=continuity.centroid_jump_fraction,
            pre_harmonization_masked_luma=pre_harmonization_masked_luma,
            post_harmonization_masked_luma=post_harmonization_masked_luma,
            local_ring_luma=local_ring_luma,
            rejected_candidate_masked_luma=rejected_candidate_masked_luma,
            postcheck_rejected=bool(postcheck_rejected),
            postcheck_reason=postcheck_reason,
            recovery_mode=recovery_mode,
            recovery_strength=recovery_strength,
            recovery_continuity_bound_source=recovery_continuity_bound_source,
            span_id=span_id,
            span_policy=span_policy,
            appearance_application_mode=application_mode,
            reference_frame_for_track=reference_frame_for_track,
            propagated_parameters_used=bool(
                status == "harmonized_track_applied"
                or status == "harmonized_track_recovered_blend"
                or status == "harmonized_track_recovered_affine"
                or force_propagated_color_match_parameters is not None
            ),
            track_downgrade_reason=track_downgrade_reason,
            track_id=track_id,
            used_for_reference_estimation=bool(used_for_reference_estimation),
            applied_parameter_source=applied_parameter_source,
            adaptive_effect_strength=float(adaptive_policy.effect_strength),
            adaptive_color_match_strength=float(adaptive_policy.color_match_strength),
            adaptive_harmonizer_strength=float(adaptive_policy.harmonizer_strength),
            adaptive_preserve_contrast=float(adaptive_policy.preserve_contrast),
            adaptive_preserve_saturation=float(adaptive_policy.preserve_saturation),
            adaptive_local_support_ratio=float(adaptive_policy.local_support_ratio),
            adaptive_tiny_subject_score=float(adaptive_policy.tiny_subject_score),
            adaptive_synthetic_scene_score=float(adaptive_policy.synthetic_scene_score),
        )
        _append_frame_diagnostics(diagnostics_dir, diagnostics)
        if write_crop_debug_overlays:
            _write_crop_debug_overlay(
                diagnostics_dir / "debug_overlays",
                frame_idx=frame_idx,
                overlay_rgb=overlay_rgb,
                bbox=bbox,
                crop=crop_bounds,
                ring_raw=None if not color_match_settings.enabled else ring_raw,
                ring_filtered=None if not color_match_settings.enabled else ring_filtered,
            )

    return ProcessFrameArtifacts(
        frame_index=int(frame_idx),
        smoothed_arguments=smoothed_arguments,
        smoothed_color_match_parameters=smoothed_color_parameters,
        fallback_transform=fallback_transform,
        used_for_reference_estimation=bool(
            used_for_reference_estimation
            and learned_model_eligible
            and crop_insufficient_reason is None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonise occluded overlay frames.")
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.05)
    parser.add_argument("--mode", type=str, default="local_crop")
    parser.add_argument("--bbox-expansion-scale", type=float, default=2.5)
    parser.add_argument("--min-crop-size-ratio", type=float, default=0.30)
    parser.add_argument("--max-frame-coverage-ratio", type=float, default=0.85)
    parser.add_argument("--containment-margin-px", type=int, default=8)
    parser.add_argument("--reject-when-actor-exceeds-crop", action="store_true")
    parser.add_argument("--oversized-actor-behavior", type=str, default="full_mask_affine_or_copy")
    parser.add_argument("--full-frame-affine-min-mask-pixels", type=int, default=512)
    parser.add_argument("--mask-source", type=str, default="visible_occlusion")
    parser.add_argument("--empty-mask-behavior", type=str, default="copy_through")
    parser.add_argument("--eligibility-min-visible-mask-pixels-for-model", type=int, default=48)
    parser.add_argument("--eligibility-min-visible-bbox-short-side-px-for-model", type=int, default=6)
    parser.add_argument("--eligibility-max-crop-coverage-ratio-for-model", type=float, default=0.70)
    parser.add_argument("--eligibility-max-crop-coverage-mask-pixels-threshold", type=int, default=128)
    parser.add_argument("--color-match-enabled", action="store_true")
    parser.add_argument("--color-match-color-space", type=str, default="lab")
    parser.add_argument("--color-match-ring-inner-px", type=int, default=10)
    parser.add_argument("--color-match-ring-outer-px", type=int, default=40)
    parser.add_argument("--color-match-exclude-top-band", action="store_true")
    parser.add_argument("--color-match-top-band-reference", type=str, default="mask_top")
    parser.add_argument("--color-match-top-band-px", type=int, default=12)
    parser.add_argument("--color-match-use-semantics-for-sky-filter", action="store_true")
    parser.add_argument("--color-match-outlier-rejection", type=str, default="robust_percentile")
    parser.add_argument("--color-match-luminance-match", type=str, default="mean_std")
    parser.add_argument("--color-match-luminance-strength", type=float, default=0.60)
    parser.add_argument("--color-match-chroma-match", type=str, default="mean_only")
    parser.add_argument("--color-match-chroma-strength", type=float, default=0.30)
    parser.add_argument("--color-match-prefer-pedestrian-reference", action="store_true")
    parser.add_argument("--color-match-pedestrian-reference-weight", type=float, default=0.65)
    parser.add_argument("--color-match-fallback-scene-reference-weight", type=float, default=0.35)
    parser.add_argument("--color-match-saturation-attenuation-strength", type=float, default=0.35)
    parser.add_argument("--color-match-contrast-attenuation-strength", type=float, default=0.25)
    parser.add_argument("--color-match-min-pedestrian-reference-pixels", type=int, default=48)
    parser.add_argument("--color-match-min-ring-pixels", type=int, default=256)
    parser.add_argument("--color-match-fallback-behavior", type=str, default="skip_and_continue")
    parser.add_argument("--color-match-write-diagnostics", action="store_true")
    parser.add_argument("--correction-clamps-min-foreground-pixels-for-luminance-scale", type=int, default=64)
    parser.add_argument("--correction-clamps-min-foreground-luminance-std-for-scale", type=float, default=2.0)
    parser.add_argument("--correction-clamps-luminance-delta-clamp-small-mask", type=float, default=18.0)
    parser.add_argument("--correction-clamps-luminance-delta-clamp-model", type=float, default=28.0)
    parser.add_argument("--correction-clamps-luminance-std-ratio-clamp-low", type=float, default=0.75)
    parser.add_argument("--correction-clamps-luminance-std-ratio-clamp-high", type=float, default=1.25)
    parser.add_argument("--correction-clamps-chroma-shift-clamp", type=float, default=6.0)
    parser.add_argument("--write-crop-diagnostics", action="store_true")
    parser.add_argument("--write-crop-debug-overlays", action="store_true")
    parser.add_argument("--temporal-smoothing-enabled", action="store_true")
    parser.add_argument("--temporal-smoothing-mode", type=str, default="parameter_ema")
    parser.add_argument("--temporal-smoothing-appearance-alpha", type=float, default=0.85)
    parser.add_argument("--temporal-smoothing-tonal-alpha", type=float, default=0.92)
    parser.add_argument("--temporal-smoothing-color-match-alpha", type=float, default=0.85)
    parser.add_argument("--temporal-smoothing-warmup-mode", type=str, default="seed_from_first_valid")
    parser.add_argument("--temporal-smoothing-reset-on-empty-mask", action="store_true")
    parser.add_argument("--temporal-smoothing-reset-on-copy-through", action="store_true")
    parser.add_argument("--temporal-smoothing-reset-on-harmonizer-failure", action="store_true")
    parser.add_argument("--temporal-smoothing-reset-on-crop-iou-below", type=float, default=0.25)
    parser.add_argument("--temporal-smoothing-reset-on-mask-area-ratio-low", type=float, default=0.5)
    parser.add_argument("--temporal-smoothing-reset-on-mask-area-ratio-high", type=float, default=2.0)
    parser.add_argument("--temporal-smoothing-reset-on-centroid-jump-fraction", type=float, default=0.25)
    parser.add_argument("--temporal-smoothing-fallback-mode", type=str, default="affine_rgb_gain_bias")
    parser.add_argument("--temporal-smoothing-write-diagnostics", action="store_true")
    parser.add_argument("--postcheck-max-ring-overshoot-luma", type=float, default=6.0)
    parser.add_argument("--postcheck-max-small-mask-brighten-luma", type=float, default=24.0)
    parser.add_argument("--tiny-object-enabled", action="store_true")
    parser.add_argument("--tiny-object-max-mask-pixels-for-conservative-path", type=int, default=256)
    parser.add_argument("--tiny-object-max-bbox-short-side-px-for-conservative-path", type=int, default=20)
    parser.add_argument("--tiny-object-skip-color-match-below-mask-pixels", type=int, default=256)
    parser.add_argument("--adaptive-enabled", action="store_true")
    parser.add_argument("--adaptive-profile-bias", type=float, default=0.0)
    parser.add_argument("--adaptive-min-effect-strength", type=float, default=0.2)
    parser.add_argument("--adaptive-low-support-weight", type=float, default=0.75)
    parser.add_argument("--adaptive-tiny-subject-weight", type=float, default=0.6)
    parser.add_argument("--adaptive-synthetic-scene-weight", type=float, default=0.5)
    parser.add_argument("--adaptive-no-local-support-color-match-strength-cap", type=float, default=0.35)
    parser.add_argument("--adaptive-no-local-support-harmonizer-strength-cap", type=float, default=0.5)
    parser.add_argument("--adaptive-backfilled-parameter-strength-scale", type=float, default=0.35)
    parser.add_argument("--adaptive-interpolated-parameter-strength-scale", type=float, default=0.6)
    parser.add_argument("--adaptive-synthetic-contrast-preservation", type=float, default=0.85)
    parser.add_argument("--adaptive-synthetic-saturation-preservation", type=float, default=0.8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    overlay_dir = args.overlay_dir.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()

    if not overlay_dir.exists():
        raise FileNotFoundError(f"Overlay directory not found: {overlay_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not pretrained.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.mode != "local_crop":
        raise ValueError(f"Unsupported harmonisation mode: {args.mode}.")
    if args.mask_source != "visible_occlusion":
        raise ValueError(f"Unsupported mask source: {args.mask_source}.")
    if args.bbox_expansion_scale <= 1.0:
        raise ValueError("--bbox-expansion-scale must be > 1.0.")
    if not 0.0 < args.min_crop_size_ratio <= 1.0:
        raise ValueError("--min-crop-size-ratio must be in (0, 1].")
    if not 0.0 < args.max_frame_coverage_ratio <= 1.0:
        raise ValueError("--max-frame-coverage-ratio must be in (0, 1].")
    if args.containment_margin_px < 0:
        raise ValueError("--containment-margin-px must be >= 0.")
    if args.oversized_actor_behavior != "full_mask_affine_or_copy":
        raise ValueError(
            "Unsupported oversized actor behavior: "
            f"{args.oversized_actor_behavior}."
        )
    if args.full_frame_affine_min_mask_pixels < 1:
        raise ValueError("--full-frame-affine-min-mask-pixels must be >= 1.")
    if args.eligibility_min_visible_mask_pixels_for_model < 1:
        raise ValueError("--eligibility-min-visible-mask-pixels-for-model must be >= 1.")
    if args.eligibility_min_visible_bbox_short_side_px_for_model < 1:
        raise ValueError("--eligibility-min-visible-bbox-short-side-px-for-model must be >= 1.")
    if not 0.0 < args.eligibility_max_crop_coverage_ratio_for_model <= 1.0:
        raise ValueError("--eligibility-max-crop-coverage-ratio-for-model must be in (0, 1].")
    if args.eligibility_max_crop_coverage_mask_pixels_threshold < 1:
        raise ValueError("--eligibility-max-crop-coverage-mask-pixels-threshold must be >= 1.")
    if args.write_crop_debug_overlays and not args.write_crop_diagnostics:
        raise ValueError("--write-crop-debug-overlays requires --write-crop-diagnostics.")
    if args.color_match_color_space != "lab":
        raise ValueError(f"Unsupported color-match color space: {args.color_match_color_space}.")
    if args.color_match_ring_inner_px < 0:
        raise ValueError("--color-match-ring-inner-px must be >= 0.")
    if args.color_match_ring_outer_px <= args.color_match_ring_inner_px:
        raise ValueError("--color-match-ring-outer-px must be > --color-match-ring-inner-px.")
    if args.color_match_top_band_reference != "mask_top":
        raise ValueError(
            f"Unsupported color-match top-band reference: {args.color_match_top_band_reference}."
        )
    if not 0.0 <= args.color_match_luminance_strength <= 1.0:
        raise ValueError("--color-match-luminance-strength must be in [0, 1].")
    if not 0.0 <= args.color_match_chroma_strength <= 1.0:
        raise ValueError("--color-match-chroma-strength must be in [0, 1].")
    if not 0.0 <= args.color_match_pedestrian_reference_weight <= 1.0:
        raise ValueError("--color-match-pedestrian-reference-weight must be in [0, 1].")
    if not 0.0 <= args.color_match_fallback_scene_reference_weight <= 1.0:
        raise ValueError("--color-match-fallback-scene-reference-weight must be in [0, 1].")
    if not 0.0 <= args.color_match_saturation_attenuation_strength <= 1.0:
        raise ValueError("--color-match-saturation-attenuation-strength must be in [0, 1].")
    if not 0.0 <= args.color_match_contrast_attenuation_strength <= 1.0:
        raise ValueError("--color-match-contrast-attenuation-strength must be in [0, 1].")
    if args.color_match_min_pedestrian_reference_pixels < 1:
        raise ValueError("--color-match-min-pedestrian-reference-pixels must be >= 1.")
    if args.color_match_min_ring_pixels < 1:
        raise ValueError("--color-match-min-ring-pixels must be >= 1.")
    if args.correction_clamps_min_foreground_pixels_for_luminance_scale < 1:
        raise ValueError("--correction-clamps-min-foreground-pixels-for-luminance-scale must be >= 1.")
    if args.correction_clamps_min_foreground_luminance_std_for_scale <= 0.0:
        raise ValueError("--correction-clamps-min-foreground-luminance-std-for-scale must be > 0.")
    if args.correction_clamps_luminance_delta_clamp_small_mask <= 0.0:
        raise ValueError("--correction-clamps-luminance-delta-clamp-small-mask must be > 0.")
    if args.correction_clamps_luminance_delta_clamp_model <= 0.0:
        raise ValueError("--correction-clamps-luminance-delta-clamp-model must be > 0.")
    if (
        args.correction_clamps_luminance_std_ratio_clamp_low <= 0.0
        or args.correction_clamps_luminance_std_ratio_clamp_high <= 0.0
        or args.correction_clamps_luminance_std_ratio_clamp_low
        > args.correction_clamps_luminance_std_ratio_clamp_high
    ):
        raise ValueError(
            "--correction-clamps-luminance-std-ratio-clamp-low/high must be a positive increasing pair."
        )
    if args.correction_clamps_chroma_shift_clamp <= 0.0:
        raise ValueError("--correction-clamps-chroma-shift-clamp must be > 0.")
    if args.color_match_outlier_rejection != "robust_percentile":
        raise ValueError(
            f"Unsupported color-match outlier rejection: {args.color_match_outlier_rejection}."
        )
    if args.color_match_luminance_match != "mean_std":
        raise ValueError(
            f"Unsupported color-match luminance match: {args.color_match_luminance_match}."
        )
    if args.color_match_chroma_match != "mean_only":
        raise ValueError(
            f"Unsupported color-match chroma match: {args.color_match_chroma_match}."
        )
    if args.color_match_fallback_behavior != "skip_and_continue":
        raise ValueError(
            "Unsupported color-match fallback behavior: "
            f"{args.color_match_fallback_behavior}."
        )
    if args.temporal_smoothing_mode != "parameter_ema":
        raise ValueError(
            "Unsupported temporal smoothing mode: "
            f"{args.temporal_smoothing_mode}."
        )
    if args.temporal_smoothing_warmup_mode != "seed_from_first_valid":
        raise ValueError(
            "Unsupported temporal smoothing warmup mode: "
            f"{args.temporal_smoothing_warmup_mode}."
        )
    for value, key in (
        (args.temporal_smoothing_appearance_alpha, "--temporal-smoothing-appearance-alpha"),
        (args.temporal_smoothing_tonal_alpha, "--temporal-smoothing-tonal-alpha"),
        (args.temporal_smoothing_color_match_alpha, "--temporal-smoothing-color-match-alpha"),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{key} must be in (0, 1).")
    if args.temporal_smoothing_reset_on_crop_iou_below <= 0.0:
        raise ValueError("--temporal-smoothing-reset-on-crop-iou-below must be > 0.")
    if (
        args.temporal_smoothing_reset_on_mask_area_ratio_low <= 0.0
        or args.temporal_smoothing_reset_on_mask_area_ratio_high <= 0.0
        or args.temporal_smoothing_reset_on_mask_area_ratio_low
        >= args.temporal_smoothing_reset_on_mask_area_ratio_high
    ):
        raise ValueError(
            "--temporal-smoothing-reset-on-mask-area-ratio-low/high "
            "must be a positive increasing pair."
        )
    if args.temporal_smoothing_reset_on_centroid_jump_fraction <= 0.0:
        raise ValueError(
            "--temporal-smoothing-reset-on-centroid-jump-fraction must be > 0."
        )
    if args.temporal_smoothing_fallback_mode != "affine_rgb_gain_bias":
        raise ValueError(
            "Unsupported temporal smoothing fallback mode: "
            f"{args.temporal_smoothing_fallback_mode}."
        )
    if args.postcheck_max_ring_overshoot_luma < 0.0:
        raise ValueError("--postcheck-max-ring-overshoot-luma must be >= 0.")
    if args.postcheck_max_small_mask_brighten_luma < 0.0:
        raise ValueError("--postcheck-max-small-mask-brighten-luma must be >= 0.")
    if args.tiny_object_max_mask_pixels_for_conservative_path < 1:
        raise ValueError(
            "--tiny-object-max-mask-pixels-for-conservative-path must be >= 1."
        )
    if args.tiny_object_max_bbox_short_side_px_for_conservative_path < 1:
        raise ValueError(
            "--tiny-object-max-bbox-short-side-px-for-conservative-path must be >= 1."
        )
    if args.tiny_object_skip_color_match_below_mask_pixels < 1:
        raise ValueError(
            "--tiny-object-skip-color-match-below-mask-pixels must be >= 1."
        )
    for value, key in (
        (args.adaptive_profile_bias, "--adaptive-profile-bias"),
        (args.adaptive_min_effect_strength, "--adaptive-min-effect-strength"),
        (args.adaptive_low_support_weight, "--adaptive-low-support-weight"),
        (args.adaptive_tiny_subject_weight, "--adaptive-tiny-subject-weight"),
        (args.adaptive_synthetic_scene_weight, "--adaptive-synthetic-scene-weight"),
        (
            args.adaptive_no_local_support_color_match_strength_cap,
            "--adaptive-no-local-support-color-match-strength-cap",
        ),
        (
            args.adaptive_no_local_support_harmonizer_strength_cap,
            "--adaptive-no-local-support-harmonizer-strength-cap",
        ),
        (
            args.adaptive_backfilled_parameter_strength_scale,
            "--adaptive-backfilled-parameter-strength-scale",
        ),
        (
            args.adaptive_interpolated_parameter_strength_scale,
            "--adaptive-interpolated-parameter-strength-scale",
        ),
        (
            args.adaptive_synthetic_contrast_preservation,
            "--adaptive-synthetic-contrast-preservation",
        ),
        (
            args.adaptive_synthetic_saturation_preservation,
            "--adaptive-synthetic-saturation-preservation",
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = _diagnostics_root(output_dir) if args.write_crop_diagnostics else None
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        frames_jsonl = diagnostics_dir / "frames.jsonl"
        if frames_jsonl.exists():
            frames_jsonl.unlink()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    color_match_settings = ColorMatchSettings(
        enabled=bool(args.color_match_enabled),
        color_space=str(args.color_match_color_space),
        ring_inner_px=int(args.color_match_ring_inner_px),
        ring_outer_px=int(args.color_match_ring_outer_px),
        exclude_top_band=bool(args.color_match_exclude_top_band),
        top_band_reference=str(args.color_match_top_band_reference),
        top_band_px=int(args.color_match_top_band_px),
        use_semantics_for_sky_filter=bool(args.color_match_use_semantics_for_sky_filter),
        outlier_rejection=str(args.color_match_outlier_rejection),
        luminance_match=str(args.color_match_luminance_match),
        luminance_strength=float(args.color_match_luminance_strength),
        chroma_match=str(args.color_match_chroma_match),
        chroma_strength=float(args.color_match_chroma_strength),
        prefer_pedestrian_reference=bool(args.color_match_prefer_pedestrian_reference),
        pedestrian_reference_weight=float(args.color_match_pedestrian_reference_weight),
        fallback_scene_reference_weight=float(args.color_match_fallback_scene_reference_weight),
        saturation_attenuation_strength=float(args.color_match_saturation_attenuation_strength),
        contrast_attenuation_strength=float(args.color_match_contrast_attenuation_strength),
        min_pedestrian_reference_pixels=int(args.color_match_min_pedestrian_reference_pixels),
        min_ring_pixels=int(args.color_match_min_ring_pixels),
        fallback_behavior=str(args.color_match_fallback_behavior),
        write_diagnostics=bool(args.color_match_write_diagnostics),
    )
    eligibility_settings = EligibilitySettings(
        min_visible_mask_pixels_for_model=int(
            args.eligibility_min_visible_mask_pixels_for_model
        ),
        min_visible_bbox_short_side_px_for_model=int(
            args.eligibility_min_visible_bbox_short_side_px_for_model
        ),
        max_crop_coverage_ratio_for_model=float(
            args.eligibility_max_crop_coverage_ratio_for_model
        ),
        max_crop_coverage_mask_pixels_threshold=int(
            args.eligibility_max_crop_coverage_mask_pixels_threshold
        ),
    )
    correction_clamp_settings = CorrectionClampSettings(
        min_foreground_pixels_for_luminance_scale=int(
            args.correction_clamps_min_foreground_pixels_for_luminance_scale
        ),
        min_foreground_luminance_std_for_scale=float(
            args.correction_clamps_min_foreground_luminance_std_for_scale
        ),
        luminance_delta_clamp_small_mask=float(
            args.correction_clamps_luminance_delta_clamp_small_mask
        ),
        luminance_delta_clamp_model=float(
            args.correction_clamps_luminance_delta_clamp_model
        ),
        luminance_std_ratio_clamp_low=float(
            args.correction_clamps_luminance_std_ratio_clamp_low
        ),
        luminance_std_ratio_clamp_high=float(
            args.correction_clamps_luminance_std_ratio_clamp_high
        ),
        chroma_shift_clamp=float(args.correction_clamps_chroma_shift_clamp),
    )
    temporal_settings = TemporalSmoothingSettings(
        enabled=bool(args.temporal_smoothing_enabled),
        mode=str(args.temporal_smoothing_mode),
        appearance_alpha=float(args.temporal_smoothing_appearance_alpha),
        tonal_alpha=float(args.temporal_smoothing_tonal_alpha),
        color_match_alpha=float(args.temporal_smoothing_color_match_alpha),
        warmup_mode=str(args.temporal_smoothing_warmup_mode),
        reset_on_empty_mask=bool(args.temporal_smoothing_reset_on_empty_mask),
        reset_on_copy_through=bool(args.temporal_smoothing_reset_on_copy_through),
        reset_on_harmonizer_failure=bool(args.temporal_smoothing_reset_on_harmonizer_failure),
        reset_on_crop_iou_below=float(args.temporal_smoothing_reset_on_crop_iou_below),
        reset_on_mask_area_ratio_low=float(args.temporal_smoothing_reset_on_mask_area_ratio_low),
        reset_on_mask_area_ratio_high=float(args.temporal_smoothing_reset_on_mask_area_ratio_high),
        reset_on_centroid_jump_fraction=float(
            args.temporal_smoothing_reset_on_centroid_jump_fraction
        ),
        fallback_mode=str(args.temporal_smoothing_fallback_mode),
        write_diagnostics=bool(args.temporal_smoothing_write_diagnostics),
    )
    tiny_object_settings = TinyObjectSettings(
        enabled=bool(args.tiny_object_enabled),
        max_mask_pixels_for_conservative_path=int(
            args.tiny_object_max_mask_pixels_for_conservative_path
        ),
        max_bbox_short_side_px_for_conservative_path=int(
            args.tiny_object_max_bbox_short_side_px_for_conservative_path
        ),
        skip_color_match_below_mask_pixels=int(
            args.tiny_object_skip_color_match_below_mask_pixels
        ),
    )
    adaptive_settings = AdaptiveSettings(
        enabled=bool(args.adaptive_enabled),
        profile_bias=float(args.adaptive_profile_bias),
        min_effect_strength=float(args.adaptive_min_effect_strength),
        low_support_weight=float(args.adaptive_low_support_weight),
        tiny_subject_weight=float(args.adaptive_tiny_subject_weight),
        synthetic_scene_weight=float(args.adaptive_synthetic_scene_weight),
        no_local_support_color_match_strength_cap=float(
            args.adaptive_no_local_support_color_match_strength_cap
        ),
        no_local_support_harmonizer_strength_cap=float(
            args.adaptive_no_local_support_harmonizer_strength_cap
        ),
        backfilled_parameter_strength_scale=float(
            args.adaptive_backfilled_parameter_strength_scale
        ),
        interpolated_parameter_strength_scale=float(
            args.adaptive_interpolated_parameter_strength_scale
        ),
        synthetic_contrast_preservation=float(
            args.adaptive_synthetic_contrast_preservation
        ),
        synthetic_saturation_preservation=float(
            args.adaptive_synthetic_saturation_preservation
        ),
    )
    postcheck_settings = PostcheckSettings(
        max_ring_overshoot_luma=float(args.postcheck_max_ring_overshoot_luma),
        max_small_mask_brighten_luma=float(args.postcheck_max_small_mask_brighten_luma),
    )
    oversized_actor_settings = OversizedActorSettings(
        max_frame_coverage_ratio=float(args.max_frame_coverage_ratio),
        containment_margin_px=int(args.containment_margin_px),
        reject_when_actor_exceeds_crop=bool(args.reject_when_actor_exceeds_crop),
        oversized_actor_behavior=str(args.oversized_actor_behavior),
        full_frame_affine_min_mask_pixels=int(args.full_frame_affine_min_mask_pixels),
    )
    temporal_state = TemporalState()

    model = _load_harmonizer_model_module()
    harmonizer = model.Harmonizer()
    harmonizer.load_state_dict(torch.load(pretrained, map_location=device), strict=True)
    harmonizer.eval()
    if device == "cuda":
        harmonizer = harmonizer.cuda()

    overlay_frames = _build_frame_index_map(overlay_dir)
    mask_frames = _build_frame_index_map(mask_dir)
    if not overlay_frames:
        raise ValueError(f"No overlay frames found in {overlay_dir}")
    if not mask_frames:
        raise ValueError(f"No occlusion mask frames found in {mask_dir}")

    missing_masks = [idx for idx in sorted(overlay_frames) if idx not in mask_frames]
    if missing_masks:
        preview = ", ".join(str(idx) for idx in missing_masks[:10])
        raise FileNotFoundError(
            "Missing occlusion masks for overlay indices: "
            f"{preview}{'...' if len(missing_masks) > 10 else ''}"
        )

    span_policies = _discover_track_span_policies(
        mask_frames=mask_frames,
        mask_threshold=args.mask_threshold,
        bbox_expansion_scale=args.bbox_expansion_scale,
        min_crop_size_ratio=args.min_crop_size_ratio,
        max_frame_coverage_ratio=float(oversized_actor_settings.max_frame_coverage_ratio),
        eligibility_settings=eligibility_settings,
        tiny_object_settings=tiny_object_settings,
    )

    reference_frames = sorted(
        {
            int(reference_frame)
            for decision in span_policies.values()
            if decision.policy == "learned_track_harmonization"
            for reference_frame in decision.reference_frame_indices
        }
    )
    fitted_arguments_by_frame: dict[int, dict[str, float]] = {}
    fitted_color_parameters_by_frame: dict[int, ColorMatchParameters] = {}
    parameter_source_by_frame: dict[int, str] = {}
    successful_reference_frames: set[int] = set()
    if reference_frames:
        reference_state = TemporalState()
        reference_argument_estimates: dict[int, dict[str, float]] = {}
        reference_color_estimates: dict[int, ColorMatchParameters] = {}
        with tempfile.TemporaryDirectory(prefix="pemoin_harmonization_estimate_") as tmpdir:
            tmp_output_dir = Path(tmpdir)
            for frame_idx in reference_frames:
                overlay_rgb = np.asarray(Image.open(overlay_frames[frame_idx]).convert("RGB"))
                mask_gray = np.asarray(Image.open(mask_frames[frame_idx]).convert("L"), dtype=np.uint8)
                decision = span_policies[int(frame_idx)]
                artifacts = process_frame(
                    frame_idx=frame_idx,
                    overlay_rgb=overlay_rgb,
                    mask_gray=mask_gray,
                    run_dir=run_dir,
                    output_dir=tmp_output_dir,
                    harmonizer=harmonizer,
                    device=device,
                    mask_threshold=args.mask_threshold,
                    bbox_expansion_scale=args.bbox_expansion_scale,
                    min_crop_size_ratio=args.min_crop_size_ratio,
                    oversized_actor_settings=oversized_actor_settings,
                    empty_mask_behavior=args.empty_mask_behavior,
                    eligibility_settings=eligibility_settings,
                    color_match_settings=color_match_settings,
                    correction_clamp_settings=correction_clamp_settings,
                    postcheck_settings=postcheck_settings,
                    tiny_object_settings=tiny_object_settings,
                    adaptive_settings=adaptive_settings,
                    diagnostics_dir=None,
                    write_crop_debug_overlays=False,
                    temporal_settings=temporal_settings,
                    temporal_state=reference_state,
                    span_id=int(decision.span_id),
                    span_policy=str(decision.policy),
                    reference_frame_for_track=decision.seed_reference_frame_index,
                    track_downgrade_reason=decision.downgrade_reason,
                    track_id=int(decision.track_id),
                    used_for_reference_estimation=True,
                    applied_parameter_source="direct_reference",
                )
                if artifacts.smoothed_arguments:
                    reference_argument_estimates[int(frame_idx)] = dict(artifacts.smoothed_arguments)
                if artifacts.smoothed_color_match_parameters is not None:
                    reference_color_estimates[int(frame_idx)] = artifacts.smoothed_color_match_parameters
        successful_reference_frames = set(int(frame) for frame in reference_argument_estimates)
        if successful_reference_frames:
            resolved_reference_frames = tuple(sorted(successful_reference_frames))
            resolved_seed_reference = int(resolved_reference_frames[0])
            for frame_idx, decision in list(span_policies.items()):
                span_policies[frame_idx] = TrackSpanDecision(
                    track_id=int(decision.track_id),
                    span_id=int(decision.span_id),
                    frame_indices=decision.frame_indices,
                    policy=str(decision.policy),
                    reference_frame_indices=resolved_reference_frames,
                    seed_reference_frame_index=resolved_seed_reference,
                    downgrade_reason=decision.downgrade_reason,
                )
        visible_track_frames = sorted(int(frame) for frame in span_policies.keys())
        fitted_arguments_by_frame, parameter_source_by_frame = _fit_track_argument_curve(
            reference_argument_estimates,
            visible_track_frames,
        )
        fitted_color_parameters_by_frame = _fit_track_color_parameter_curve(
            reference_color_estimates,
            visible_track_frames,
        )
        if not fitted_arguments_by_frame:
            for frame_idx, decision in list(span_policies.items()):
                span_policies[frame_idx] = TrackSpanDecision(
                    track_id=int(decision.track_id),
                    span_id=int(decision.span_id),
                    frame_indices=decision.frame_indices,
                    policy="conservative_track_harmonization",
                    reference_frame_indices=tuple(),
                    seed_reference_frame_index=None,
                    downgrade_reason="no_postcheck_safe_reference_frame",
                )

    for frame_idx in tqdm(sorted(overlay_frames), desc="harmonize", unit="frame"):
        overlay_rgb = np.asarray(Image.open(overlay_frames[frame_idx]).convert("RGB"))
        mask_gray = np.asarray(Image.open(mask_frames[frame_idx]).convert("L"), dtype=np.uint8)
        span = span_policies.get(int(frame_idx))
        process_frame(
            frame_idx=frame_idx,
            overlay_rgb=overlay_rgb,
            mask_gray=mask_gray,
            run_dir=run_dir,
            output_dir=output_dir,
            harmonizer=harmonizer,
            device=device,
            mask_threshold=args.mask_threshold,
            bbox_expansion_scale=args.bbox_expansion_scale,
            min_crop_size_ratio=args.min_crop_size_ratio,
            oversized_actor_settings=oversized_actor_settings,
            empty_mask_behavior=args.empty_mask_behavior,
            eligibility_settings=eligibility_settings,
            color_match_settings=color_match_settings,
            correction_clamp_settings=correction_clamp_settings,
            postcheck_settings=postcheck_settings,
            tiny_object_settings=tiny_object_settings,
            adaptive_settings=adaptive_settings,
            diagnostics_dir=diagnostics_dir,
            write_crop_debug_overlays=args.write_crop_debug_overlays,
            temporal_settings=temporal_settings,
            temporal_state=temporal_state,
            span_id=None if span is None else int(span.span_id),
            span_policy=None if span is None else str(span.policy),
            reference_frame_for_track=(
                None if span is None else span.seed_reference_frame_index
            ),
            track_downgrade_reason=None if span is None else span.downgrade_reason,
            force_propagated_arguments=(
                None
                if span is None or span.policy != "learned_track_harmonization"
                else fitted_arguments_by_frame.get(int(frame_idx))
            ),
            force_propagated_color_match_parameters=(
                None
                if span is None or span.policy != "learned_track_harmonization"
                else fitted_color_parameters_by_frame.get(int(frame_idx))
            ),
            track_id=None if span is None else int(span.track_id),
            used_for_reference_estimation=(
                False
                if span is None
                else int(frame_idx) in successful_reference_frames
            ),
            applied_parameter_source=(
                None
                if span is None or span.policy != "learned_track_harmonization"
                else parameter_source_by_frame.get(int(frame_idx))
            ),
        )
    if diagnostics_dir is not None:
        _write_diagnostics_summary(diagnostics_dir, temporal_state=temporal_state)


if __name__ == "__main__":
    main()
