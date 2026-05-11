"""
Pipeline harmonisation step using the bundled Harmonizer model.

This invokes tools/Harmonizer to harmonize occlusion-correct overlay frames
using precomputed visible-pedestrian masks.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from pemoin.data.contracts import ResourceStore

LOG = logging.getLogger(__name__)


class HarmonisationMode(str, Enum):
    LOCAL_CROP = "local_crop"


class HarmonisationMaskSource(str, Enum):
    VISIBLE_OCCLUSION = "visible_occlusion"


class HarmonisationEmptyMaskBehavior(str, Enum):
    COPY_THROUGH = "copy_through"


class HarmonisationColorSpace(str, Enum):
    LAB = "lab"


class HarmonisationOutlierRejection(str, Enum):
    ROBUST_PERCENTILE = "robust_percentile"


class HarmonisationLuminanceMatch(str, Enum):
    MEAN_STD = "mean_std"


class HarmonisationChromaMatch(str, Enum):
    MEAN_ONLY = "mean_only"


class HarmonisationColorMatchFallbackBehavior(str, Enum):
    SKIP_AND_CONTINUE = "skip_and_continue"


class HarmonisationTemporalSmoothingMode(str, Enum):
    PARAMETER_EMA = "parameter_ema"


class HarmonisationFallbackMode(str, Enum):
    AFFINE_RGB_GAIN_BIAS = "affine_rgb_gain_bias"


class HarmonisationOversizedActorBehavior(str, Enum):
    FULL_MASK_AFFINE_OR_COPY = "full_mask_affine_or_copy"


@dataclass(frozen=True, slots=True)
class HarmonisationEligibilitySettings:
    min_visible_mask_pixels_for_model: int = 48
    min_visible_bbox_short_side_px_for_model: int = 6
    max_crop_coverage_ratio_for_model: float = 0.70
    max_crop_coverage_mask_pixels_threshold: int = 128

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "HarmonisationEligibilitySettings":
        data = mapping if isinstance(mapping, Mapping) else {}
        settings = cls(
            min_visible_mask_pixels_for_model=int(
                data.get("min_visible_mask_pixels_for_model", 48)
            ),
            min_visible_bbox_short_side_px_for_model=int(
                data.get("min_visible_bbox_short_side_px_for_model", 6)
            ),
            max_crop_coverage_ratio_for_model=float(
                data.get("max_crop_coverage_ratio_for_model", 0.70)
            ),
            max_crop_coverage_mask_pixels_threshold=int(
                data.get("max_crop_coverage_mask_pixels_threshold", 128)
            ),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.min_visible_mask_pixels_for_model < 1:
            raise ValueError(
                "harmonisation.eligibility.min_visible_mask_pixels_for_model must be >= 1."
            )
        if self.min_visible_bbox_short_side_px_for_model < 1:
            raise ValueError(
                "harmonisation.eligibility.min_visible_bbox_short_side_px_for_model must be >= 1."
            )
        if not 0.0 < self.max_crop_coverage_ratio_for_model <= 1.0:
            raise ValueError(
                "harmonisation.eligibility.max_crop_coverage_ratio_for_model must be in (0, 1]."
            )
        if self.max_crop_coverage_mask_pixels_threshold < 1:
            raise ValueError(
                "harmonisation.eligibility.max_crop_coverage_mask_pixels_threshold must be >= 1."
            )


@dataclass(frozen=True, slots=True)
class HarmonisationCorrectionClampSettings:
    min_foreground_pixels_for_luminance_scale: int = 64
    min_foreground_luminance_std_for_scale: float = 2.0
    luminance_delta_clamp_small_mask: float = 18.0
    luminance_delta_clamp_model: float = 28.0
    luminance_std_ratio_clamp: tuple[float, float] = (0.75, 1.25)
    chroma_shift_clamp: float = 6.0

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "HarmonisationCorrectionClampSettings":
        data = mapping if isinstance(mapping, Mapping) else {}
        ratio_clamp = data.get("luminance_std_ratio_clamp", [0.75, 1.25])
        if not isinstance(ratio_clamp, Sequence) or len(ratio_clamp) != 2:
            raise ValueError(
                "harmonisation.correction_clamps.luminance_std_ratio_clamp must be a two-item sequence."
            )
        settings = cls(
            min_foreground_pixels_for_luminance_scale=int(
                data.get("min_foreground_pixels_for_luminance_scale", 64)
            ),
            min_foreground_luminance_std_for_scale=float(
                data.get("min_foreground_luminance_std_for_scale", 2.0)
            ),
            luminance_delta_clamp_small_mask=float(
                data.get("luminance_delta_clamp_small_mask", 18.0)
            ),
            luminance_delta_clamp_model=float(
                data.get("luminance_delta_clamp_model", 28.0)
            ),
            luminance_std_ratio_clamp=(float(ratio_clamp[0]), float(ratio_clamp[1])),
            chroma_shift_clamp=float(data.get("chroma_shift_clamp", 6.0)),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.min_foreground_pixels_for_luminance_scale < 1:
            raise ValueError(
                "harmonisation.correction_clamps.min_foreground_pixels_for_luminance_scale must be >= 1."
            )
        if self.min_foreground_luminance_std_for_scale <= 0.0:
            raise ValueError(
                "harmonisation.correction_clamps.min_foreground_luminance_std_for_scale must be > 0."
            )
        if self.luminance_delta_clamp_small_mask <= 0.0:
            raise ValueError(
                "harmonisation.correction_clamps.luminance_delta_clamp_small_mask must be > 0."
            )
        if self.luminance_delta_clamp_model <= 0.0:
            raise ValueError(
                "harmonisation.correction_clamps.luminance_delta_clamp_model must be > 0."
            )
        low, high = self.luminance_std_ratio_clamp
        if low <= 0.0 or high <= 0.0 or low > high:
            raise ValueError(
                "harmonisation.correction_clamps.luminance_std_ratio_clamp must be a positive increasing pair."
            )
        if self.chroma_shift_clamp <= 0.0:
            raise ValueError(
                "harmonisation.correction_clamps.chroma_shift_clamp must be > 0."
            )


@dataclass(frozen=True, slots=True)
class HarmonisationPostcheckSettings:
    max_ring_overshoot_luma: float = 6.0
    max_small_mask_brighten_luma: float = 24.0

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "HarmonisationPostcheckSettings":
        data = mapping if isinstance(mapping, Mapping) else {}
        settings = cls(
            max_ring_overshoot_luma=float(data.get("max_ring_overshoot_luma", 6.0)),
            max_small_mask_brighten_luma=float(
                data.get("max_small_mask_brighten_luma", 24.0)
            ),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.max_ring_overshoot_luma < 0.0:
            raise ValueError(
                "harmonisation.postcheck.max_ring_overshoot_luma must be >= 0."
            )
        if self.max_small_mask_brighten_luma < 0.0:
            raise ValueError(
                "harmonisation.postcheck.max_small_mask_brighten_luma must be >= 0."
            )


@dataclass(frozen=True, slots=True)
class HarmonisationColorMatchingSettings:
    enabled: bool = True
    color_space: HarmonisationColorSpace = HarmonisationColorSpace.LAB
    ring_inner_px: int = 10
    ring_outer_px: int = 40
    exclude_top_band: bool = True
    top_band_reference: str = "mask_top"
    top_band_px: int = 12
    use_semantics_for_sky_filter: bool = True
    outlier_rejection: HarmonisationOutlierRejection = HarmonisationOutlierRejection.ROBUST_PERCENTILE
    luminance_match: HarmonisationLuminanceMatch = HarmonisationLuminanceMatch.MEAN_STD
    luminance_strength: float = 0.60
    chroma_match: HarmonisationChromaMatch = HarmonisationChromaMatch.MEAN_ONLY
    chroma_strength: float = 0.30
    prefer_pedestrian_reference: bool = True
    pedestrian_reference_weight: float = 0.65
    fallback_scene_reference_weight: float = 0.35
    saturation_attenuation_strength: float = 0.35
    contrast_attenuation_strength: float = 0.25
    min_pedestrian_reference_pixels: int = 48
    min_ring_pixels: int = 256
    fallback_behavior: HarmonisationColorMatchFallbackBehavior = (
        HarmonisationColorMatchFallbackBehavior.SKIP_AND_CONTINUE
    )
    write_diagnostics: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HarmonisationColorMatchingSettings":
        settings = cls(
            enabled=bool(mapping.get("enabled", True)),
            color_space=HarmonisationColorSpace(
                str(mapping.get("color_space", HarmonisationColorSpace.LAB.value))
            ),
            ring_inner_px=int(mapping.get("ring_inner_px", 10)),
            ring_outer_px=int(mapping.get("ring_outer_px", 40)),
            exclude_top_band=bool(mapping.get("exclude_top_band", True)),
            top_band_reference=str(mapping.get("top_band_reference", "mask_top")),
            top_band_px=int(mapping.get("top_band_px", 12)),
            use_semantics_for_sky_filter=bool(
                mapping.get("use_semantics_for_sky_filter", True)
            ),
            outlier_rejection=HarmonisationOutlierRejection(
                str(
                    mapping.get(
                        "outlier_rejection",
                        HarmonisationOutlierRejection.ROBUST_PERCENTILE.value,
                    )
                )
            ),
            luminance_match=HarmonisationLuminanceMatch(
                str(
                    mapping.get(
                        "luminance_match",
                        HarmonisationLuminanceMatch.MEAN_STD.value,
                    )
                )
            ),
            luminance_strength=float(mapping.get("luminance_strength", 0.60)),
            chroma_match=HarmonisationChromaMatch(
                str(
                    mapping.get(
                        "chroma_match",
                        HarmonisationChromaMatch.MEAN_ONLY.value,
                    )
                )
            ),
            chroma_strength=float(mapping.get("chroma_strength", 0.30)),
            prefer_pedestrian_reference=bool(mapping.get("prefer_pedestrian_reference", True)),
            pedestrian_reference_weight=float(mapping.get("pedestrian_reference_weight", 0.65)),
            fallback_scene_reference_weight=float(
                mapping.get("fallback_scene_reference_weight", 0.35)
            ),
            saturation_attenuation_strength=float(
                mapping.get("saturation_attenuation_strength", 0.35)
            ),
            contrast_attenuation_strength=float(
                mapping.get("contrast_attenuation_strength", 0.25)
            ),
            min_pedestrian_reference_pixels=int(
                mapping.get("min_pedestrian_reference_pixels", 48)
            ),
            min_ring_pixels=int(mapping.get("min_ring_pixels", 256)),
            fallback_behavior=HarmonisationColorMatchFallbackBehavior(
                str(
                    mapping.get(
                        "fallback_behavior",
                        HarmonisationColorMatchFallbackBehavior.SKIP_AND_CONTINUE.value,
                    )
                )
            ),
            write_diagnostics=bool(mapping.get("write_diagnostics", True)),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.ring_inner_px < 0:
            raise ValueError("harmonisation.color_matching.ring_inner_px must be >= 0.")
        if self.ring_outer_px <= self.ring_inner_px:
            raise ValueError(
                "harmonisation.color_matching.ring_outer_px must be > ring_inner_px."
            )
        if self.top_band_reference != "mask_top":
            raise ValueError(
                "harmonisation.color_matching.top_band_reference must be 'mask_top'."
            )
        if self.top_band_px < 0:
            raise ValueError("harmonisation.color_matching.top_band_px must be >= 0.")
        if not 0.0 <= self.luminance_strength <= 1.0:
            raise ValueError(
                "harmonisation.color_matching.luminance_strength must be in [0, 1]."
            )
        if not 0.0 <= self.chroma_strength <= 1.0:
            raise ValueError(
                "harmonisation.color_matching.chroma_strength must be in [0, 1]."
            )
        if not 0.0 <= self.pedestrian_reference_weight <= 1.0:
            raise ValueError(
                "harmonisation.color_matching.pedestrian_reference_weight must be in [0, 1]."
            )
        if not 0.0 <= self.fallback_scene_reference_weight <= 1.0:
            raise ValueError(
                "harmonisation.color_matching.fallback_scene_reference_weight must be in [0, 1]."
            )
        if not 0.0 <= self.saturation_attenuation_strength <= 1.0:
            raise ValueError(
                "harmonisation.color_matching.saturation_attenuation_strength must be in [0, 1]."
            )
        if not 0.0 <= self.contrast_attenuation_strength <= 1.0:
            raise ValueError(
                "harmonisation.color_matching.contrast_attenuation_strength must be in [0, 1]."
            )
        if self.min_pedestrian_reference_pixels < 1:
            raise ValueError(
                "harmonisation.color_matching.min_pedestrian_reference_pixels must be >= 1."
            )
        if self.min_ring_pixels < 1:
            raise ValueError(
                "harmonisation.color_matching.min_ring_pixels must be >= 1."
            )


@dataclass(frozen=True, slots=True)
class HarmonisationTemporalSmoothingSettings:
    enabled: bool = True
    mode: HarmonisationTemporalSmoothingMode = HarmonisationTemporalSmoothingMode.PARAMETER_EMA
    appearance_alpha: float = 0.85
    tonal_alpha: float = 0.92
    color_match_alpha: float = 0.85
    warmup_mode: str = "seed_from_first_valid"
    reset_on_empty_mask: bool = True
    reset_on_copy_through: bool = False
    reset_on_harmonizer_failure: bool = True
    reset_on_crop_iou_below: float = 0.25
    reset_on_mask_area_ratio_outside: tuple[float, float] = (0.5, 2.0)
    reset_on_centroid_jump_fraction: float = 0.25
    fallback_mode: HarmonisationFallbackMode = HarmonisationFallbackMode.AFFINE_RGB_GAIN_BIAS
    write_diagnostics: bool = True

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "HarmonisationTemporalSmoothingSettings":
        data = mapping if isinstance(mapping, Mapping) else {}
        area_ratio = data.get("reset_on_mask_area_ratio_outside", [0.5, 2.0])
        if not isinstance(area_ratio, Sequence) or len(area_ratio) != 2:
            raise ValueError(
                "harmonisation.temporal_smoothing.reset_on_mask_area_ratio_outside "
                "must be a two-item sequence."
            )
        settings = cls(
            enabled=bool(data.get("enabled", True)),
            mode=HarmonisationTemporalSmoothingMode(
                str(
                    data.get(
                        "mode",
                        HarmonisationTemporalSmoothingMode.PARAMETER_EMA.value,
                    )
                )
            ),
            appearance_alpha=float(data.get("appearance_alpha", 0.85)),
            tonal_alpha=float(data.get("tonal_alpha", 0.92)),
            color_match_alpha=float(data.get("color_match_alpha", 0.85)),
            warmup_mode=str(data.get("warmup_mode", "seed_from_first_valid")),
            reset_on_empty_mask=bool(data.get("reset_on_empty_mask", True)),
            reset_on_copy_through=bool(data.get("reset_on_copy_through", False)),
            reset_on_harmonizer_failure=bool(
                data.get("reset_on_harmonizer_failure", True)
            ),
            reset_on_crop_iou_below=float(data.get("reset_on_crop_iou_below", 0.25)),
            reset_on_mask_area_ratio_outside=(
                float(area_ratio[0]),
                float(area_ratio[1]),
            ),
            reset_on_centroid_jump_fraction=float(
                data.get("reset_on_centroid_jump_fraction", 0.25)
            ),
            fallback_mode=HarmonisationFallbackMode(
                str(
                    data.get(
                        "fallback_mode",
                        HarmonisationFallbackMode.AFFINE_RGB_GAIN_BIAS.value,
                    )
                )
            ),
            write_diagnostics=bool(data.get("write_diagnostics", True)),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.mode != HarmonisationTemporalSmoothingMode.PARAMETER_EMA:
            raise ValueError(
                "harmonisation.temporal_smoothing.mode must be 'parameter_ema'."
            )
        for field_name, value in (
            ("appearance_alpha", self.appearance_alpha),
            ("tonal_alpha", self.tonal_alpha),
            ("color_match_alpha", self.color_match_alpha),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"harmonisation.temporal_smoothing.{field_name} must be in (0, 1)."
                )
        if self.warmup_mode != "seed_from_first_valid":
            raise ValueError(
                "harmonisation.temporal_smoothing.warmup_mode must be "
                "'seed_from_first_valid'."
            )
        if self.reset_on_crop_iou_below <= 0.0:
            raise ValueError(
                "harmonisation.temporal_smoothing.reset_on_crop_iou_below must be > 0."
            )
        low, high = self.reset_on_mask_area_ratio_outside
        if low <= 0.0 or high <= 0.0 or low >= high:
            raise ValueError(
                "harmonisation.temporal_smoothing.reset_on_mask_area_ratio_outside "
                "must be an increasing positive pair."
            )
        if self.reset_on_centroid_jump_fraction <= 0.0:
            raise ValueError(
                "harmonisation.temporal_smoothing.reset_on_centroid_jump_fraction "
                "must be > 0."
            )


@dataclass(frozen=True, slots=True)
class HarmonisationTinyObjectSettings:
    enabled: bool = True
    max_mask_pixels_for_conservative_path: int = 256
    max_bbox_short_side_px_for_conservative_path: int = 20
    skip_color_match_below_mask_pixels: int = 256

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "HarmonisationTinyObjectSettings":
        data = mapping if isinstance(mapping, Mapping) else {}
        settings = cls(
            enabled=bool(data.get("enabled", True)),
            max_mask_pixels_for_conservative_path=int(
                data.get("max_mask_pixels_for_conservative_path", 256)
            ),
            max_bbox_short_side_px_for_conservative_path=int(
                data.get("max_bbox_short_side_px_for_conservative_path", 20)
            ),
            skip_color_match_below_mask_pixels=int(
                data.get("skip_color_match_below_mask_pixels", 256)
            ),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.max_mask_pixels_for_conservative_path < 1:
            raise ValueError(
                "harmonisation.tiny_object.max_mask_pixels_for_conservative_path must be >= 1."
            )
        if self.max_bbox_short_side_px_for_conservative_path < 1:
            raise ValueError(
                "harmonisation.tiny_object.max_bbox_short_side_px_for_conservative_path must be >= 1."
            )
        if self.skip_color_match_below_mask_pixels < 1:
            raise ValueError(
                "harmonisation.tiny_object.skip_color_match_below_mask_pixels must be >= 1."
            )


@dataclass(frozen=True, slots=True)
class HarmonisationAdaptiveSettings:
    enabled: bool = True
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

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "HarmonisationAdaptiveSettings":
        data = mapping if isinstance(mapping, Mapping) else {}
        settings = cls(
            enabled=bool(data.get("enabled", True)),
            profile_bias=float(data.get("profile_bias", 0.0)),
            min_effect_strength=float(data.get("min_effect_strength", 0.2)),
            low_support_weight=float(data.get("low_support_weight", 0.75)),
            tiny_subject_weight=float(data.get("tiny_subject_weight", 0.6)),
            synthetic_scene_weight=float(data.get("synthetic_scene_weight", 0.5)),
            no_local_support_color_match_strength_cap=float(
                data.get("no_local_support_color_match_strength_cap", 0.35)
            ),
            no_local_support_harmonizer_strength_cap=float(
                data.get("no_local_support_harmonizer_strength_cap", 0.5)
            ),
            backfilled_parameter_strength_scale=float(
                data.get("backfilled_parameter_strength_scale", 0.35)
            ),
            interpolated_parameter_strength_scale=float(
                data.get("interpolated_parameter_strength_scale", 0.6)
            ),
            synthetic_contrast_preservation=float(
                data.get("synthetic_contrast_preservation", 0.85)
            ),
            synthetic_saturation_preservation=float(
                data.get("synthetic_saturation_preservation", 0.8)
            ),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        for field_name, value in (
            ("profile_bias", self.profile_bias),
            ("min_effect_strength", self.min_effect_strength),
            ("low_support_weight", self.low_support_weight),
            ("tiny_subject_weight", self.tiny_subject_weight),
            ("synthetic_scene_weight", self.synthetic_scene_weight),
            (
                "no_local_support_color_match_strength_cap",
                self.no_local_support_color_match_strength_cap,
            ),
            (
                "no_local_support_harmonizer_strength_cap",
                self.no_local_support_harmonizer_strength_cap,
            ),
            (
                "backfilled_parameter_strength_scale",
                self.backfilled_parameter_strength_scale,
            ),
            (
                "interpolated_parameter_strength_scale",
                self.interpolated_parameter_strength_scale,
            ),
            (
                "synthetic_contrast_preservation",
                self.synthetic_contrast_preservation,
            ),
            (
                "synthetic_saturation_preservation",
                self.synthetic_saturation_preservation,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"harmonisation.adaptive.{field_name} must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class HarmonisationSettings:
    enabled: bool = False
    conda_env: Optional[str] = "harmonizer"
    pretrained_path: str = "tools/Harmonizer/pretrained/harmonizer.pth"
    overlay_dir: str = "artifacts/blender/overlayed_frames"
    occlusion_mask_dir: str = "artifacts/blender/occlusion_masks"
    output_dir: str = "artifacts/harmonisation/harmonized_overlays"
    mask_threshold: float = 0.05
    mode: HarmonisationMode = HarmonisationMode.LOCAL_CROP
    bbox_expansion_scale: float = 2.5
    min_crop_size_ratio: float = 0.30
    max_frame_coverage_ratio: float = 0.85
    containment_margin_px: int = 8
    reject_when_actor_exceeds_crop: bool = True
    oversized_actor_behavior: HarmonisationOversizedActorBehavior = (
        HarmonisationOversizedActorBehavior.FULL_MASK_AFFINE_OR_COPY
    )
    full_frame_affine_min_mask_pixels: int = 512
    mask_source: HarmonisationMaskSource = HarmonisationMaskSource.VISIBLE_OCCLUSION
    empty_mask_behavior: HarmonisationEmptyMaskBehavior = HarmonisationEmptyMaskBehavior.COPY_THROUGH
    write_crop_diagnostics: bool = True
    write_crop_debug_overlays: bool = False
    eligibility: HarmonisationEligibilitySettings = HarmonisationEligibilitySettings()
    color_matching: HarmonisationColorMatchingSettings = HarmonisationColorMatchingSettings()
    correction_clamps: HarmonisationCorrectionClampSettings = (
        HarmonisationCorrectionClampSettings()
    )
    temporal_smoothing: HarmonisationTemporalSmoothingSettings = (
        HarmonisationTemporalSmoothingSettings()
    )
    postcheck: HarmonisationPostcheckSettings = HarmonisationPostcheckSettings()
    tiny_object: HarmonisationTinyObjectSettings = HarmonisationTinyObjectSettings()
    adaptive: HarmonisationAdaptiveSettings = HarmonisationAdaptiveSettings()
    device: Optional[str] = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HarmonisationSettings":
        settings = cls(
            enabled=bool(mapping.get("enabled", False)),
            conda_env=(
                str(mapping["conda_env"]) if mapping.get("conda_env") is not None else None
            ),
            pretrained_path=str(
                mapping.get("pretrained_path", "tools/Harmonizer/pretrained/harmonizer.pth")
            ),
            overlay_dir=str(
                mapping.get("overlay_dir", "artifacts/blender/overlayed_frames")
            ),
            occlusion_mask_dir=str(
                mapping.get("occlusion_mask_dir", "artifacts/blender/occlusion_masks")
            ),
            output_dir=str(
                mapping.get(
                    "output_dir",
                    "artifacts/harmonisation/harmonized_overlays",
                )
            ),
            mask_threshold=float(mapping.get("mask_threshold", 0.05)),
            mode=HarmonisationMode(str(mapping.get("mode", HarmonisationMode.LOCAL_CROP.value))),
            bbox_expansion_scale=float(mapping.get("bbox_expansion_scale", 2.5)),
            min_crop_size_ratio=float(mapping.get("min_crop_size_ratio", 0.30)),
            max_frame_coverage_ratio=float(mapping.get("max_frame_coverage_ratio", 0.85)),
            containment_margin_px=int(mapping.get("containment_margin_px", 8)),
            reject_when_actor_exceeds_crop=bool(mapping.get("reject_when_actor_exceeds_crop", True)),
            oversized_actor_behavior=HarmonisationOversizedActorBehavior(
                str(
                    mapping.get(
                        "oversized_actor_behavior",
                        HarmonisationOversizedActorBehavior.FULL_MASK_AFFINE_OR_COPY.value,
                    )
                )
            ),
            full_frame_affine_min_mask_pixels=int(
                mapping.get("full_frame_affine_min_mask_pixels", 512)
            ),
            mask_source=HarmonisationMaskSource(
                str(mapping.get("mask_source", HarmonisationMaskSource.VISIBLE_OCCLUSION.value))
            ),
            empty_mask_behavior=HarmonisationEmptyMaskBehavior(
                str(
                    mapping.get(
                        "empty_mask_behavior",
                        HarmonisationEmptyMaskBehavior.COPY_THROUGH.value,
                    )
                )
            ),
            write_crop_diagnostics=bool(mapping.get("write_crop_diagnostics", True)),
            write_crop_debug_overlays=bool(mapping.get("write_crop_debug_overlays", False)),
            eligibility=HarmonisationEligibilitySettings.from_mapping(
                mapping.get("eligibility", {})
                if isinstance(mapping.get("eligibility", {}), Mapping)
                else {}
            ),
            color_matching=HarmonisationColorMatchingSettings.from_mapping(
                mapping.get("color_matching", {})
                if isinstance(mapping.get("color_matching", {}), Mapping)
                else {}
            ),
            correction_clamps=HarmonisationCorrectionClampSettings.from_mapping(
                mapping.get("correction_clamps", {})
                if isinstance(mapping.get("correction_clamps", {}), Mapping)
                else {}
            ),
            temporal_smoothing=HarmonisationTemporalSmoothingSettings.from_mapping(
                mapping.get("temporal_smoothing", {})
                if isinstance(mapping.get("temporal_smoothing", {}), Mapping)
                else {}
            ),
            postcheck=HarmonisationPostcheckSettings.from_mapping(
                mapping.get("postcheck", {})
                if isinstance(mapping.get("postcheck", {}), Mapping)
                else {}
            ),
            tiny_object=HarmonisationTinyObjectSettings.from_mapping(
                mapping.get("tiny_object", {})
                if isinstance(mapping.get("tiny_object", {}), Mapping)
                else {}
            ),
            adaptive=HarmonisationAdaptiveSettings.from_mapping(
                mapping.get("adaptive", {})
                if isinstance(mapping.get("adaptive", {}), Mapping)
                else {}
            ),
            device=(str(mapping["device"]) if mapping.get("device") is not None else None),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.bbox_expansion_scale <= 1.0:
            raise ValueError("harmonisation.bbox_expansion_scale must be > 1.0.")
        if not 0.0 < self.min_crop_size_ratio <= 1.0:
            raise ValueError("harmonisation.min_crop_size_ratio must be in (0, 1].")
        if not 0.0 < self.max_frame_coverage_ratio <= 1.0:
            raise ValueError("harmonisation.max_frame_coverage_ratio must be in (0, 1].")
        if self.containment_margin_px < 0:
            raise ValueError("harmonisation.containment_margin_px must be >= 0.")
        if self.full_frame_affine_min_mask_pixels < 1:
            raise ValueError("harmonisation.full_frame_affine_min_mask_pixels must be >= 1.")
        if self.write_crop_debug_overlays and not self.write_crop_diagnostics:
            raise ValueError(
                "harmonisation.write_crop_debug_overlays requires "
                "harmonisation.write_crop_diagnostics=true."
            )


def run_harmonisation(
    run_dir: Path,
    settings: HarmonisationSettings,
    *,
    env: Optional[MutableMapping[str, str]] = None,
) -> Path:
    """
    Run Harmonizer on occlusion-correct overlay frames and masks.

    Args:
        run_dir: outputs/<run> directory.
        settings: Harmonisation settings.
        env: Optional environment variables for the subprocess.

    Returns:
        Output directory containing harmonized overlay frames.
    """
    if not settings.enabled:
        raise ValueError("Harmonisation is disabled but run_harmonisation() was called.")

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "src" / "pemoin" / "scripts" / "harmonisation_runner.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Harmonisation runner not found: {script_path}")

    overlay_dir = (run_dir / settings.overlay_dir).resolve()
    occlusion_mask_dir = (run_dir / settings.occlusion_mask_dir).resolve()
    output_dir = (run_dir / settings.output_dir).resolve()
    pedestrian_frames_dir = ResourceStore.blender_artifact_dir_for(
        run_dir,
        "pedestrian_frames",
    ).resolve()

    if not pedestrian_frames_dir.exists():
        raise FileNotFoundError(
            "Pedestrian frames directory not found: "
            f"{pedestrian_frames_dir} (expected by harmonisation). "
            "This directory is produced by Blender scene composition; "
            "enable runtime.settings.blender_scene.enabled=true for this profile "
            "or disable runtime.settings.harmonisation.enabled."
        )
    if not overlay_dir.exists():
        raise FileNotFoundError(
            f"Overlay frames directory not found: {overlay_dir} (expected by harmonisation)"
        )
    if not occlusion_mask_dir.exists():
        raise FileNotFoundError(
            "Occlusion mask directory not found: "
            f"{occlusion_mask_dir} (expected by harmonisation). "
            "This directory is produced by Blender scene composition; "
            "enable runtime.settings.blender_scene.enabled=true for this profile "
            "or disable runtime.settings.harmonisation.enabled."
        )

    pretrained_path = Path(settings.pretrained_path)
    if not pretrained_path.is_absolute():
        pretrained_path = (repo_root / pretrained_path).resolve()
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Harmonizer checkpoint not found: {pretrained_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "python",
        str(script_path),
        "--overlay-dir",
        str(overlay_dir),
        "--mask-dir",
        str(occlusion_mask_dir),
        "--output-dir",
        str(output_dir),
        "--run-dir",
        str(run_dir.resolve()),
        "--pretrained",
        str(pretrained_path),
        "--mask-threshold",
        str(settings.mask_threshold),
        "--mode",
        settings.mode.value,
        "--bbox-expansion-scale",
        str(settings.bbox_expansion_scale),
        "--min-crop-size-ratio",
        str(settings.min_crop_size_ratio),
        "--max-frame-coverage-ratio",
        str(settings.max_frame_coverage_ratio),
        "--containment-margin-px",
        str(settings.containment_margin_px),
        "--oversized-actor-behavior",
        settings.oversized_actor_behavior.value,
        "--full-frame-affine-min-mask-pixels",
        str(settings.full_frame_affine_min_mask_pixels),
        "--mask-source",
        settings.mask_source.value,
        "--empty-mask-behavior",
        settings.empty_mask_behavior.value,
    ]
    if settings.reject_when_actor_exceeds_crop:
        cmd.append("--reject-when-actor-exceeds-crop")
    eligibility = settings.eligibility
    cmd.extend(
        [
            "--eligibility-min-visible-mask-pixels-for-model",
            str(eligibility.min_visible_mask_pixels_for_model),
            "--eligibility-min-visible-bbox-short-side-px-for-model",
            str(eligibility.min_visible_bbox_short_side_px_for_model),
            "--eligibility-max-crop-coverage-ratio-for-model",
            str(eligibility.max_crop_coverage_ratio_for_model),
            "--eligibility-max-crop-coverage-mask-pixels-threshold",
            str(eligibility.max_crop_coverage_mask_pixels_threshold),
        ]
    )
    color_matching = settings.color_matching
    if color_matching.enabled:
        cmd.extend(
            [
                "--color-match-enabled",
                "--color-match-color-space",
                color_matching.color_space.value,
                "--color-match-ring-inner-px",
                str(color_matching.ring_inner_px),
                "--color-match-ring-outer-px",
                str(color_matching.ring_outer_px),
                "--color-match-top-band-reference",
                color_matching.top_band_reference,
                "--color-match-top-band-px",
                str(color_matching.top_band_px),
                "--color-match-outlier-rejection",
                color_matching.outlier_rejection.value,
                "--color-match-luminance-match",
                color_matching.luminance_match.value,
                "--color-match-luminance-strength",
                str(color_matching.luminance_strength),
                "--color-match-chroma-match",
                color_matching.chroma_match.value,
                "--color-match-chroma-strength",
                str(color_matching.chroma_strength),
                "--color-match-pedestrian-reference-weight",
                str(color_matching.pedestrian_reference_weight),
                "--color-match-fallback-scene-reference-weight",
                str(color_matching.fallback_scene_reference_weight),
                "--color-match-saturation-attenuation-strength",
                str(color_matching.saturation_attenuation_strength),
                "--color-match-contrast-attenuation-strength",
                str(color_matching.contrast_attenuation_strength),
                "--color-match-min-pedestrian-reference-pixels",
                str(color_matching.min_pedestrian_reference_pixels),
                "--color-match-min-ring-pixels",
                str(color_matching.min_ring_pixels),
                "--color-match-fallback-behavior",
                color_matching.fallback_behavior.value,
            ]
        )
        if color_matching.prefer_pedestrian_reference:
            cmd.append("--color-match-prefer-pedestrian-reference")
        if color_matching.exclude_top_band:
            cmd.append("--color-match-exclude-top-band")
        if color_matching.use_semantics_for_sky_filter:
            cmd.append("--color-match-use-semantics-for-sky-filter")
        if color_matching.write_diagnostics:
            cmd.append("--color-match-write-diagnostics")
    correction_clamps = settings.correction_clamps
    cmd.extend(
        [
            "--correction-clamps-min-foreground-pixels-for-luminance-scale",
            str(correction_clamps.min_foreground_pixels_for_luminance_scale),
            "--correction-clamps-min-foreground-luminance-std-for-scale",
            str(correction_clamps.min_foreground_luminance_std_for_scale),
            "--correction-clamps-luminance-delta-clamp-small-mask",
            str(correction_clamps.luminance_delta_clamp_small_mask),
            "--correction-clamps-luminance-delta-clamp-model",
            str(correction_clamps.luminance_delta_clamp_model),
            "--correction-clamps-luminance-std-ratio-clamp-low",
            str(correction_clamps.luminance_std_ratio_clamp[0]),
            "--correction-clamps-luminance-std-ratio-clamp-high",
            str(correction_clamps.luminance_std_ratio_clamp[1]),
            "--correction-clamps-chroma-shift-clamp",
            str(correction_clamps.chroma_shift_clamp),
        ]
    )
    if settings.write_crop_diagnostics:
        cmd.append("--write-crop-diagnostics")
    if settings.write_crop_debug_overlays:
        cmd.append("--write-crop-debug-overlays")
    temporal = settings.temporal_smoothing
    if temporal.enabled:
        cmd.extend(
            [
                "--temporal-smoothing-enabled",
                "--temporal-smoothing-mode",
                temporal.mode.value,
                "--temporal-smoothing-appearance-alpha",
                str(temporal.appearance_alpha),
                "--temporal-smoothing-tonal-alpha",
                str(temporal.tonal_alpha),
                "--temporal-smoothing-color-match-alpha",
                str(temporal.color_match_alpha),
                "--temporal-smoothing-warmup-mode",
                temporal.warmup_mode,
                "--temporal-smoothing-reset-on-crop-iou-below",
                str(temporal.reset_on_crop_iou_below),
                "--temporal-smoothing-reset-on-mask-area-ratio-low",
                str(temporal.reset_on_mask_area_ratio_outside[0]),
                "--temporal-smoothing-reset-on-mask-area-ratio-high",
                str(temporal.reset_on_mask_area_ratio_outside[1]),
                "--temporal-smoothing-reset-on-centroid-jump-fraction",
                str(temporal.reset_on_centroid_jump_fraction),
                "--temporal-smoothing-fallback-mode",
                temporal.fallback_mode.value,
            ]
        )
        if temporal.reset_on_empty_mask:
            cmd.append("--temporal-smoothing-reset-on-empty-mask")
        if temporal.reset_on_copy_through:
            cmd.append("--temporal-smoothing-reset-on-copy-through")
        if temporal.reset_on_harmonizer_failure:
            cmd.append("--temporal-smoothing-reset-on-harmonizer-failure")
        if temporal.write_diagnostics:
            cmd.append("--temporal-smoothing-write-diagnostics")
    if settings.device:
        cmd.extend(["--device", settings.device])
    postcheck = settings.postcheck
    cmd.extend(
        [
            "--postcheck-max-ring-overshoot-luma",
            str(postcheck.max_ring_overshoot_luma),
            "--postcheck-max-small-mask-brighten-luma",
            str(postcheck.max_small_mask_brighten_luma),
        ]
    )
    tiny_object = settings.tiny_object
    if tiny_object.enabled:
        cmd.extend(
            [
                "--tiny-object-enabled",
                "--tiny-object-max-mask-pixels-for-conservative-path",
                str(tiny_object.max_mask_pixels_for_conservative_path),
                "--tiny-object-max-bbox-short-side-px-for-conservative-path",
                str(tiny_object.max_bbox_short_side_px_for_conservative_path),
                "--tiny-object-skip-color-match-below-mask-pixels",
                str(tiny_object.skip_color_match_below_mask_pixels),
            ]
        )
    adaptive = settings.adaptive
    if adaptive.enabled:
        cmd.extend(
            [
                "--adaptive-enabled",
                "--adaptive-profile-bias",
                str(adaptive.profile_bias),
                "--adaptive-min-effect-strength",
                str(adaptive.min_effect_strength),
                "--adaptive-low-support-weight",
                str(adaptive.low_support_weight),
                "--adaptive-tiny-subject-weight",
                str(adaptive.tiny_subject_weight),
                "--adaptive-synthetic-scene-weight",
                str(adaptive.synthetic_scene_weight),
                "--adaptive-no-local-support-color-match-strength-cap",
                str(adaptive.no_local_support_color_match_strength_cap),
                "--adaptive-no-local-support-harmonizer-strength-cap",
                str(adaptive.no_local_support_harmonizer_strength_cap),
                "--adaptive-backfilled-parameter-strength-scale",
                str(adaptive.backfilled_parameter_strength_scale),
                "--adaptive-interpolated-parameter-strength-scale",
                str(adaptive.interpolated_parameter_strength_scale),
                "--adaptive-synthetic-contrast-preservation",
                str(adaptive.synthetic_contrast_preservation),
                "--adaptive-synthetic-saturation-preservation",
                str(adaptive.synthetic_saturation_preservation),
            ]
        )

    if settings.conda_env:
        cmd = [*_default_env_launcher(settings.conda_env), *cmd]

    LOG.info("Running harmonisation: %s", " ".join(cmd))
    process = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Harmonisation failed.\n"
            f"stdout:\n{process.stdout}\n\n"
            f"stderr:\n{process.stderr}"
        )

    return output_dir


def _default_env_launcher(env_name: str) -> Sequence[str]:
    available = [name for name in ("micromamba", "mamba", "conda") if shutil.which(name)]
    for candidate in available:
        detected = _find_env_launcher_for_manager(candidate, env_name)
        if detected is not None:
            return detected
    if available:
        return (available[0], "run", "-n", env_name)
    raise RuntimeError(
        f"Unable to run harmoniser inside environment '{env_name}': "
        "none of micromamba/mamba/conda was found on PATH."
    )


def _parse_env_listing(env_list_output: str) -> tuple[set[str], dict[str, list[str]]]:
    names: set[str] = set()
    path_matches: dict[str, list[str]] = {}
    for raw_line in env_list_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in line.split() if part != "*"]
        if not parts:
            continue
        first = parts[0].strip()
        if first.lower() in {"name", "envs", "environments"}:
            continue
        if not first.startswith("/"):
            names.add(first)
        last = parts[-1].strip()
        if "/" in last:
            base = Path(last).name
            path_matches.setdefault(base, []).append(last)
    return names, path_matches


def _find_env_launcher_for_manager(manager: str, env_name: str) -> Optional[tuple[str, ...]]:
    probe = subprocess.run(
        [manager, "env", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    names, path_matches = _parse_env_listing(f"{probe.stdout}\n{probe.stderr}")
    if env_name in names:
        return (manager, "run", "-n", env_name)
    if env_name in path_matches and path_matches[env_name]:
        return (manager, "run", "-p", path_matches[env_name][0])
    return None
