from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pemoin.data.contracts import ResourceStore, SemanticSegment, SemanticsData
from pemoin.scripts import harmonisation_runner


class _StubHarmonizer:
    def predict_arguments(self, comp_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=comp_tensor.dtype, device=comp_tensor.device)

    def restore_image(
        self,
        comp_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        arguments: torch.Tensor,
    ) -> list[torch.Tensor]:
        brightened = torch.clamp(comp_tensor + 0.25, 0.0, 1.0)
        return [brightened]


class _SequenceHarmonizer:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def predict_arguments(self, comp_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return [torch.tensor([[value]], dtype=comp_tensor.dtype, device=comp_tensor.device) for _ in range(6)]

    def restore_image(
        self,
        comp_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        arguments,
    ) -> list[torch.Tensor]:
        brightness = float(arguments[1].detach().reshape(-1)[0].item())
        result = torch.clamp(comp_tensor + brightness, 0.0, 1.0)
        return [result]


def _default_color_match_settings(**overrides) -> harmonisation_runner.ColorMatchSettings:
    base = {
        "enabled": True,
        "color_space": "lab",
        "ring_inner_px": 2,
        "ring_outer_px": 6,
        "exclude_top_band": True,
        "top_band_reference": "mask_top",
        "top_band_px": 2,
        "use_semantics_for_sky_filter": True,
        "outlier_rejection": "robust_percentile",
        "luminance_match": "mean_std",
        "luminance_strength": 0.60,
        "chroma_match": "mean_only",
        "chroma_strength": 0.30,
        "min_ring_pixels": 8,
        "fallback_behavior": "skip_and_continue",
        "write_diagnostics": True,
    }
    base.update(overrides)
    return harmonisation_runner.ColorMatchSettings(**base)


def _default_temporal_settings(**overrides) -> harmonisation_runner.TemporalSmoothingSettings:
    base = {
        "enabled": True,
        "mode": "parameter_ema",
        "appearance_alpha": 0.8,
        "tonal_alpha": 0.9,
        "color_match_alpha": 0.8,
        "warmup_mode": "seed_from_first_valid",
        "reset_on_empty_mask": True,
        "reset_on_copy_through": False,
        "reset_on_harmonizer_failure": True,
        "reset_on_crop_iou_below": 0.25,
        "reset_on_mask_area_ratio_low": 0.5,
        "reset_on_mask_area_ratio_high": 2.0,
        "reset_on_centroid_jump_fraction": 0.25,
        "fallback_mode": "affine_rgb_gain_bias",
        "write_diagnostics": True,
    }
    base.update(overrides)
    return harmonisation_runner.TemporalSmoothingSettings(**base)


def _default_eligibility_settings(**overrides) -> harmonisation_runner.EligibilitySettings:
    base = {
        "min_visible_mask_pixels_for_model": 48,
        "min_visible_bbox_short_side_px_for_model": 6,
        "max_crop_coverage_ratio_for_model": 0.70,
        "max_crop_coverage_mask_pixels_threshold": 128,
    }
    base.update(overrides)
    return harmonisation_runner.EligibilitySettings(**base)


def _default_correction_clamps(**overrides) -> harmonisation_runner.CorrectionClampSettings:
    base = {
        "min_foreground_pixels_for_luminance_scale": 64,
        "min_foreground_luminance_std_for_scale": 2.0,
        "luminance_delta_clamp_small_mask": 18.0,
        "luminance_delta_clamp_model": 28.0,
        "luminance_std_ratio_clamp_low": 0.75,
        "luminance_std_ratio_clamp_high": 1.25,
        "chroma_shift_clamp": 6.0,
    }
    base.update(overrides)
    return harmonisation_runner.CorrectionClampSettings(**base)


def _default_postcheck_settings(**overrides) -> harmonisation_runner.PostcheckSettings:
    base = {
        "max_ring_overshoot_luma": 6.0,
        "max_small_mask_brighten_luma": 24.0,
    }
    base.update(overrides)
    return harmonisation_runner.PostcheckSettings(**base)


def _default_oversized_actor_settings(
    **overrides,
) -> harmonisation_runner.OversizedActorSettings:
    base = {
        "max_frame_coverage_ratio": 0.85,
        "containment_margin_px": 8,
        "reject_when_actor_exceeds_crop": True,
        "oversized_actor_behavior": "full_mask_affine_or_copy",
        "full_frame_affine_min_mask_pixels": 512,
    }
    base.update(overrides)
    return harmonisation_runner.OversizedActorSettings(**base)


def _default_adaptive_settings(**overrides) -> harmonisation_runner.AdaptiveSettings:
    base = {
        "enabled": True,
        "profile_bias": 0.0,
        "min_effect_strength": 0.2,
        "low_support_weight": 0.75,
        "tiny_subject_weight": 0.6,
        "synthetic_scene_weight": 0.5,
        "no_local_support_color_match_strength_cap": 0.35,
        "no_local_support_harmonizer_strength_cap": 0.5,
        "backfilled_parameter_strength_scale": 0.35,
        "interpolated_parameter_strength_scale": 0.6,
        "synthetic_contrast_preservation": 0.85,
        "synthetic_saturation_preservation": 0.8,
    }
    base.update(overrides)
    return harmonisation_runner.AdaptiveSettings(**base)


def _save_frame_semantics(
    run_dir: Path,
    *,
    frame_idx: int,
    label_ids: np.ndarray,
    segments: list[SemanticSegment],
    metadata: dict | None = None,
) -> None:
    store = ResourceStore(run_dir.name, root=run_dir.parent)
    store.save_semantics2d(
        SemanticsData(
            frame_index=frame_idx,
            frame_id=str(frame_idx),
            segments=segments,
            segment_ids=label_ids.copy(),
            label_ids=label_ids,
            metadata=dict(metadata or {}),
        )
    )


def test_compute_local_crop_expands_bbox_and_respects_min_size() -> None:
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 25:30] = 255

    bbox, expanded_bbox, crop = harmonisation_runner._compute_local_crop(
        mask,
        bbox_expansion_scale=2.5,
        min_crop_size_px=24,
        max_frame_coverage_ratio=0.85,
    )

    assert bbox is not None
    assert bbox.as_xyxy() == [25, 10, 30, 20]
    assert expanded_bbox is not None
    assert crop is not None
    assert crop.width >= 24
    assert crop.height >= 24
    assert crop.left >= 0 and crop.top >= 0
    assert crop.right <= mask.shape[1]
    assert crop.bottom <= mask.shape[0]


def test_compute_local_crop_clamps_to_image_bounds() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:5, 0:6] = 255

    _, _, crop = harmonisation_runner._compute_local_crop(
        mask,
        bbox_expansion_scale=4.0,
        min_crop_size_px=40,
        max_frame_coverage_ratio=0.85,
    )

    assert crop is not None
    assert crop.as_xyxy() == [0, 0, 28, 28]


def test_process_frame_copies_through_when_visible_mask_empty(tmp_path: Path) -> None:
    overlay_rgb = np.full((16, 20, 3), 80, dtype=np.uint8)
    mask_gray = np.zeros((16, 20), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"

    harmonisation_runner.process_frame(
        frame_idx=7,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=run_dir,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=12,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(
            max_ring_overshoot_luma=999.0,
            max_small_mask_brighten_luma=999.0,
        ),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=True,
    )

    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000007.png").convert("RGB"))
    assert np.array_equal(written, overlay_rgb)

    records = (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 1
    payload = json.loads(records[0])
    assert payload["status"] == "copied_through_empty_mask"
    assert payload["fallback_reason"] == "empty_visible_mask"
    assert payload["model_ran"] is False
    assert (diagnostics_dir / "debug_overlays" / "000007.png").exists()


def test_process_frame_harmonizes_only_local_crop_and_preserves_full_frame_shape(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    overlay_rgb[:, :] = [10, 20, 30]
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[8:12, 10:14] = 255
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"

    harmonisation_runner.process_frame(
        frame_idx=3,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=run_dir,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=1,
            min_visible_bbox_short_side_px_for_model=1,
        ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(
            max_ring_overshoot_luma=999.0,
            max_small_mask_brighten_luma=999.0,
        ),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
    )

    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000003.png").convert("RGB"))
    assert written.shape == overlay_rgb.shape
    assert not np.array_equal(written, overlay_rgb)

    records = (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(records[0])
    crop_left, crop_top, crop_right, crop_bottom = payload["crop_xyxy"]
    untouched = written.copy()
    untouched[crop_top:crop_bottom, crop_left:crop_right, :] = overlay_rgb[
        crop_top:crop_bottom,
        crop_left:crop_right,
        :,
    ]
    assert np.array_equal(untouched, overlay_rgb)
    assert payload["status"] == "harmonized_local_crop"
    assert payload["model_ran"] is True
    assert payload["color_match_applied"] is False


def test_process_frame_copies_through_when_actor_exceeds_crop_without_safe_full_frame_fallback(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((40, 40, 3), [40, 50, 60], dtype=np.uint8)
    mask_gray = np.zeros((40, 40), dtype=np.uint8)
    mask_gray[2:38, 4:36] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"

    harmonisation_runner.process_frame(
        frame_idx=11,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=1,
            min_visible_bbox_short_side_px_for_model=1,
        ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        oversized_actor_settings=_default_oversized_actor_settings(
            max_frame_coverage_ratio=0.4,
            full_frame_affine_min_mask_pixels=2048,
        ),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
    )

    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000011.png").convert("RGB"))
    assert np.array_equal(written, overlay_rgb)

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["status"] == "copied_through_crop_insufficient"
    assert payload["model_ran"] is False
    assert payload["eligibility_reason"] == "crop_insufficient_for_actor"
    assert payload["crop_insufficient_reason"] == "visible_mask_exceeds_crop"
    assert payload["full_frame_fallback_used"] is True
    assert payload["fallback_mode"] == "copy_through"
    assert payload["visible_mask_outside_crop_pixels"] > 0


def test_process_frame_uses_full_mask_affine_fallback_when_actor_exceeds_crop_and_track_transform_exists(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((40, 40, 3), [40, 50, 60], dtype=np.uint8)
    mask_gray = np.zeros((40, 40), dtype=np.uint8)
    mask_gray[2:38, 4:36] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    temporal_state = harmonisation_runner.TemporalState(
        affine_transform_ema=harmonisation_runner.AffineColorTransform(
            gain=np.ones((3,), dtype=np.float32),
            bias=np.asarray([18.0, 0.0, 0.0], dtype=np.float32),
        )
    )

    harmonisation_runner.process_frame(
        frame_idx=12,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=1,
            min_visible_bbox_short_side_px_for_model=1,
        ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        oversized_actor_settings=_default_oversized_actor_settings(
            max_frame_coverage_ratio=0.4,
            full_frame_affine_min_mask_pixels=32,
        ),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(),
        temporal_state=temporal_state,
    )

    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000012.png").convert("RGB"))
    mask = mask_gray > 0
    np.testing.assert_array_equal(written[~mask], overlay_rgb[~mask])
    assert np.all(written[mask, 0] > overlay_rgb[mask, 0])

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["status"] == "harmonized_full_mask_affine_crop_insufficient"
    assert payload["full_frame_fallback_used"] is True
    assert payload["fallback_mode"] == "full_mask_affine_from_track_ema"
    assert payload["oversized_actor_behavior_applied"] == "full_mask_affine_or_copy"
    assert payload["crop_insufficient_reason"] == "visible_mask_exceeds_crop"


def test_process_frame_rejects_overlay_mask_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Overlay/mask shape mismatch"):
        harmonisation_runner.process_frame(
            frame_idx=1,
            overlay_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            mask_gray=np.zeros((7, 8), dtype=np.uint8),
            run_dir=tmp_path,
            output_dir=tmp_path / "harmonized_overlays",
            harmonizer=_StubHarmonizer(),
            device="cpu",
            mask_threshold=0.05,
            bbox_expansion_scale=2.5,
            min_crop_size_px=8,
            empty_mask_behavior="copy_through",
            eligibility_settings=_default_eligibility_settings(),
            color_match_settings=_default_color_match_settings(enabled=False),
            correction_clamp_settings=_default_correction_clamps(),
            postcheck_settings=_default_postcheck_settings(),
            diagnostics_dir=None,
            write_crop_debug_overlays=False,
        )


def test_apply_lab_color_match_shifts_foreground_toward_local_ring_stats() -> None:
    crop_rgb = np.full((18, 18, 3), [40, 70, 90], dtype=np.uint8)
    crop_rgb[6:12, 6:12] = [160, 110, 90]
    crop_mask = np.zeros((18, 18), dtype=np.uint8)
    crop_mask[6:12, 6:12] = 255

    corrected, diag, ring_raw, ring_filtered = harmonisation_runner._apply_lab_color_match(
        crop_rgb,
        crop_mask,
        bbox=harmonisation_runner.CropBounds(left=6, top=6, right=12, bottom=12),
        color_match_settings=_default_color_match_settings(),
        clamp_settings=_default_correction_clamps(),
        small_mask_mode=False,
        full_frame_sky_mask_crop=None,
    )

    assert diag.applied is True
    assert diag.skip_reason is None
    assert np.count_nonzero(ring_raw) >= np.count_nonzero(ring_filtered) > 0

    orig_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    corrected_lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_mask = crop_mask > 0
    bg_mask = ring_filtered
    orig_delta_l = abs(float(orig_lab[fg_mask, 0].mean()) - float(orig_lab[bg_mask, 0].mean()))
    corrected_delta_l = abs(
        float(corrected_lab[fg_mask, 0].mean()) - float(corrected_lab[bg_mask, 0].mean())
    )
    orig_delta_a = abs(float(orig_lab[fg_mask, 1].mean()) - float(orig_lab[bg_mask, 1].mean()))
    corrected_delta_a = abs(
        float(corrected_lab[fg_mask, 1].mean()) - float(corrected_lab[bg_mask, 1].mean())
    )
    assert corrected_delta_l < orig_delta_l
    assert corrected_delta_a < orig_delta_a
    np.testing.assert_array_equal(corrected[~fg_mask], crop_rgb[~fg_mask])


def test_apply_lab_color_match_uses_semantics_to_filter_sky_ring(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    crop_rgb = np.full((24, 24, 3), [90, 100, 110], dtype=np.uint8)
    crop_rgb[0:8, :] = [140, 180, 245]
    crop_rgb[10:14, 10:14] = [180, 120, 110]
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[10:14, 10:14] = 255

    label_ids = np.zeros((24, 24), dtype=np.int32)
    label_ids[0:8, :] = 2
    _save_frame_semantics(
        run_dir,
        frame_idx=5,
        label_ids=label_ids,
        segments=[
            SemanticSegment(segment_id=1, label="road", score=1.0, mask=label_ids == 0, label_id=0),
            SemanticSegment(segment_id=2, label="sky", score=1.0, mask=label_ids == 2, label_id=2),
        ],
        metadata={"semantic_roles": {"sky": ["sky"]}},
    )

    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    harmonisation_runner.process_frame(
        frame_idx=5,
        overlay_rgb=crop_rgb,
        mask_gray=mask_gray,
        run_dir=run_dir,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=14,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(),
        color_match_settings=_default_color_match_settings(exclude_top_band=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=True,
    )

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["color_match_applied"] is True
    assert payload["sky_filtered_pixel_count"] > 0
    assert payload["ring_pixel_count_filtered"] >= 8
    assert (diagnostics_dir / "debug_overlays" / "000005.png").exists()


def test_apply_lab_color_match_skips_when_ring_too_sparse() -> None:
    crop_rgb = np.full((10, 10, 3), [100, 100, 100], dtype=np.uint8)
    crop_mask = np.zeros((10, 10), dtype=np.uint8)
    crop_mask[3:7, 3:7] = 255

    corrected, diag, _, _ = harmonisation_runner._apply_lab_color_match(
        crop_rgb,
        crop_mask,
        bbox=harmonisation_runner.CropBounds(left=3, top=3, right=7, bottom=7),
        color_match_settings=_default_color_match_settings(min_ring_pixels=100),
        clamp_settings=_default_correction_clamps(),
        small_mask_mode=False,
        full_frame_sky_mask_crop=None,
    )

    assert diag.applied is False
    assert diag.skip_reason in {"insufficient_ring_pixels", "insufficient_ring_inliers"}
    np.testing.assert_array_equal(corrected, crop_rgb)


def test_apply_lab_color_match_uses_pedestrian_reference_to_soften_chroma() -> None:
    crop_rgb = np.full((18, 18, 3), [70, 80, 90], dtype=np.uint8)
    crop_rgb[6:12, 6:12] = [210, 90, 70]
    crop_mask = np.zeros((18, 18), dtype=np.uint8)
    crop_mask[6:12, 6:12] = 255
    pedestrian_reference_mask = np.zeros((18, 18), dtype=bool)
    pedestrian_reference_mask[2:5, 2:6] = True

    corrected, diag, _, ring_filtered = harmonisation_runner._apply_lab_color_match(
        crop_rgb,
        crop_mask,
        bbox=harmonisation_runner.CropBounds(left=6, top=6, right=12, bottom=12),
        color_match_settings=_default_color_match_settings(
            pedestrian_reference_weight=0.8,
            saturation_attenuation_strength=0.8,
            contrast_attenuation_strength=0.4,
            min_pedestrian_reference_pixels=4,
        ),
        clamp_settings=_default_correction_clamps(),
        small_mask_mode=False,
        full_frame_sky_mask_crop=None,
        full_frame_pedestrian_mask_crop=pedestrian_reference_mask,
    )

    assert diag.applied is True
    assert diag.debug is not None
    assert diag.debug["pedestrian_reference_pixel_count"] >= 4
    orig_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    corrected_lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_mask = crop_mask > 0
    orig_chroma = np.linalg.norm(orig_lab[fg_mask, 1:3] - 128.0, axis=1).mean()
    corrected_chroma = np.linalg.norm(corrected_lab[fg_mask, 1:3] - 128.0, axis=1).mean()
    assert corrected_chroma < orig_chroma
    assert np.count_nonzero(ring_filtered) > 0


def test_process_frame_temporally_smooths_harmonizer_arguments(tmp_path: Path) -> None:
    overlay_rgb = np.full((20, 20, 3), 50, dtype=np.uint8)
    mask_gray = np.zeros((20, 20), dtype=np.uint8)
    mask_gray[6:14, 6:14] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    temporal_state = harmonisation_runner.TemporalState()
    harmonizer = _SequenceHarmonizer([0.0, 0.5])

    harmonisation_runner.process_frame(
        frame_idx=1,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=harmonizer,
        device="cpu",
        mask_threshold=0.05,
            bbox_expansion_scale=2.5,
            min_crop_size_px=10,
            empty_mask_behavior="copy_through",
            eligibility_settings=_default_eligibility_settings(
                max_crop_coverage_ratio_for_model=0.8,
            ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(appearance_alpha=0.8, tonal_alpha=0.8),
        temporal_state=temporal_state,
    )
    harmonisation_runner.process_frame(
        frame_idx=2,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=harmonizer,
        device="cpu",
        mask_threshold=0.05,
            bbox_expansion_scale=2.5,
            min_crop_size_px=10,
            empty_mask_behavior="copy_through",
            eligibility_settings=_default_eligibility_settings(
                max_crop_coverage_ratio_for_model=0.8,
            ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(appearance_alpha=0.8, tonal_alpha=0.8),
        temporal_state=temporal_state,
    )

    records = [
        json.loads(line)
        for line in (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["raw_harmonizer_arguments"]["brightness"] == pytest.approx(0.0)
    assert records[0]["smoothed_harmonizer_arguments"]["brightness"] == pytest.approx(0.0)
    assert records[1]["raw_harmonizer_arguments"]["brightness"] == pytest.approx(0.5)
    assert records[1]["smoothed_harmonizer_arguments"]["brightness"] == pytest.approx(0.1)


def test_process_frame_resets_temporal_state_on_large_crop_jump(tmp_path: Path) -> None:
    overlay_rgb = np.full((32, 32, 3), 70, dtype=np.uint8)
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    temporal_state = harmonisation_runner.TemporalState()
    harmonizer = _SequenceHarmonizer([0.1, 0.6])

    first_mask = np.zeros((32, 32), dtype=np.uint8)
    first_mask[4:12, 4:12] = 255
    second_mask = np.zeros((32, 32), dtype=np.uint8)
    second_mask[20:28, 20:28] = 255

    harmonisation_runner.process_frame(
        frame_idx=1,
        overlay_rgb=overlay_rgb,
        mask_gray=first_mask,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=harmonizer,
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(appearance_alpha=0.8),
        temporal_state=temporal_state,
    )
    harmonisation_runner.process_frame(
        frame_idx=2,
        overlay_rgb=overlay_rgb,
        mask_gray=second_mask,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=harmonizer,
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(appearance_alpha=0.8),
        temporal_state=temporal_state,
    )

    records = [
        json.loads(line)
        for line in (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[1]["temporal_reset_applied"] is True
    assert records[1]["temporal_reset_reason"] == "crop_iou_below_threshold"
    assert records[1]["smoothed_harmonizer_arguments"]["brightness"] == pytest.approx(0.6)


def test_process_frame_skips_learned_model_for_tiny_visible_mask(tmp_path: Path) -> None:
    overlay_rgb = np.full((24, 24, 3), 80, dtype=np.uint8)
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[10:12, 10:12] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"

    harmonisation_runner.process_frame(
        frame_idx=4,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
    )

    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000004.png").convert("RGB"))
    np.testing.assert_array_equal(written, overlay_rgb)
    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["model_ran"] is False
    assert payload["eligibility_reason"] == "small_visible_mask"
    assert payload["status"] == "fallback_small_visible_mask"


def test_process_frame_holds_tiny_object_conservative_mode_across_adjacent_frames(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((24, 24, 3), 80, dtype=np.uint8)
    first_mask = np.zeros((24, 24), dtype=np.uint8)
    first_mask[10:13, 10:13] = 255
    second_mask = np.zeros((24, 24), dtype=np.uint8)
    second_mask[9:14, 9:14] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    temporal_state = harmonisation_runner.TemporalState()

    for frame_idx, mask_gray in ((4, first_mask), (5, second_mask)):
        harmonisation_runner.process_frame(
            frame_idx=frame_idx,
            overlay_rgb=overlay_rgb,
            mask_gray=mask_gray,
            run_dir=tmp_path,
            output_dir=output_dir,
            harmonizer=_StubHarmonizer(),
            device="cpu",
            mask_threshold=0.05,
            bbox_expansion_scale=2.5,
            min_crop_size_px=10,
            empty_mask_behavior="copy_through",
            eligibility_settings=_default_eligibility_settings(
                min_visible_mask_pixels_for_model=4,
                min_visible_bbox_short_side_px_for_model=3,
            ),
            color_match_settings=_default_color_match_settings(enabled=True),
            correction_clamp_settings=_default_correction_clamps(),
            postcheck_settings=_default_postcheck_settings(),
            tiny_object_settings=harmonisation_runner.TinyObjectSettings(
                enabled=True,
                max_mask_pixels_for_conservative_path=16,
                max_bbox_short_side_px_for_conservative_path=3,
                skip_color_match_below_mask_pixels=16,
            ),
            diagnostics_dir=diagnostics_dir,
            write_crop_debug_overlays=False,
            temporal_settings=_default_temporal_settings(),
            temporal_state=temporal_state,
        )

    records = [
        json.loads(line)
        for line in (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["model_ran"] is False
    assert records[0]["eligibility_reason"] == "tiny_object_conservative_mask"
    assert records[0]["color_match_applied"] is False
    assert records[1]["model_ran"] is False
    assert records[1]["eligibility_reason"] == "tiny_object_temporal_hold"
    assert records[1]["status"] == "fallback_tiny_object_temporal_hold"
    assert records[1]["color_match_applied"] is False
    assert records[1]["color_match_skip_reason"] == "tiny_object_temporal_conservative_skip"


def test_process_frame_enters_temporal_tiny_hold_near_threshold_before_model_runs(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((32, 32, 3), 80, dtype=np.uint8)
    mask_gray = np.zeros((32, 32), dtype=np.uint8)
    mask_gray[8:25, 8:24] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    temporal_state = harmonisation_runner.TemporalState()

    harmonisation_runner.process_frame(
        frame_idx=4,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=4,
            min_visible_bbox_short_side_px_for_model=3,
        ),
        color_match_settings=_default_color_match_settings(enabled=True),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(),
        tiny_object_settings=harmonisation_runner.TinyObjectSettings(
            enabled=True,
            max_mask_pixels_for_conservative_path=256,
            max_bbox_short_side_px_for_conservative_path=12,
            skip_color_match_below_mask_pixels=256,
        ),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(),
        temporal_state=temporal_state,
    )

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["model_ran"] is False
    assert payload["eligibility_reason"] == "tiny_object_temporal_hold"
    assert payload["status"] == "fallback_tiny_object_temporal_hold"


def test_process_frame_rejects_overbright_learned_result_with_postcheck(tmp_path: Path) -> None:
    overlay_rgb = np.full((24, 24, 3), 100, dtype=np.uint8)
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[6:14, 8:16] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"

    harmonisation_runner.process_frame(
        frame_idx=8,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(),
        color_match_settings=_default_color_match_settings(enabled=True),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(max_ring_overshoot_luma=0.0),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
    )

    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000008.png").convert("RGB"))
    np.testing.assert_array_equal(written, overlay_rgb)
    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["model_ran"] is True
    assert payload["postcheck_rejected"] is True
    assert payload["postcheck_reason"] == "ring_overshoot"
    assert payload["status"] == "fallback_postcheck_rejected"


def test_discover_track_span_policies_uses_single_learned_policy_for_span(
    tmp_path: Path,
) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    empty = np.zeros((24, 24), dtype=np.uint8)
    tiny = np.zeros((24, 24), dtype=np.uint8)
    tiny[10:13, 10:13] = 255
    stable = np.zeros((24, 24), dtype=np.uint8)
    stable[6:18, 7:19] = 255
    harmonisation_runner.Image.fromarray(empty).save(mask_dir / "000000.png")
    harmonisation_runner.Image.fromarray(tiny).save(mask_dir / "000001.png")
    harmonisation_runner.Image.fromarray(tiny).save(mask_dir / "000002.png")
    harmonisation_runner.Image.fromarray(stable).save(mask_dir / "000003.png")
    harmonisation_runner.Image.fromarray(stable).save(mask_dir / "000004.png")

    decisions = harmonisation_runner._discover_track_span_policies(
        mask_frames=harmonisation_runner._build_frame_index_map(mask_dir),
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=32,
            min_visible_bbox_short_side_px_for_model=6,
        ),
        tiny_object_settings=harmonisation_runner.TinyObjectSettings(
            enabled=True,
            max_mask_pixels_for_conservative_path=16,
            max_bbox_short_side_px_for_conservative_path=3,
            skip_color_match_below_mask_pixels=16,
        ),
    )

    assert decisions[1].policy == "learned_track_harmonization"
    assert decisions[1].span_id == decisions[2].span_id == decisions[3].span_id == decisions[4].span_id
    assert decisions[1].seed_reference_frame_index == 3
    assert decisions[4].reference_frame_indices == (3, 4)


def test_process_frame_uses_track_propagation_instead_of_fallback_for_tiny_frame(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((24, 24, 3), 60, dtype=np.uint8)
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[10:13, 10:13] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"

    harmonisation_runner.process_frame(
        frame_idx=9,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=32,
            min_visible_bbox_short_side_px_for_model=6,
        ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(
            max_ring_overshoot_luma=999.0,
            max_small_mask_brighten_luma=999.0,
        ),
        tiny_object_settings=harmonisation_runner.TinyObjectSettings(
            enabled=True,
            max_mask_pixels_for_conservative_path=16,
            max_bbox_short_side_px_for_conservative_path=3,
            skip_color_match_below_mask_pixels=16,
        ),
        adaptive_settings=_default_adaptive_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        span_id=2,
        span_policy="learned_track_harmonization",
        reference_frame_for_track=12,
        force_propagated_arguments={"brightness": 0.12},
        track_id=0,
        applied_parameter_source="backfilled_from_future",
    )

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000009.png").convert("RGB"))
    assert payload["status"] == "harmonized_track_applied"
    assert payload["propagated_parameters_used"] is True
    assert payload["span_policy"] == "learned_track_harmonization"
    assert payload["applied_parameter_source"] == "backfilled_from_future"
    assert payload["adaptive_harmonizer_strength"] < 0.5
    assert payload["adaptive_color_match_strength"] < 0.4
    assert not np.array_equal(written, overlay_rgb)


def test_process_frame_recovers_rejected_tracked_tiny_frame_with_bounded_blend(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((24, 24, 3), 60, dtype=np.uint8)
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[10:13, 10:13] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    temporal_state = harmonisation_runner.TemporalState(
        last_accepted_masked_luma=76.0,
    )

    harmonisation_runner.process_frame(
        frame_idx=10,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=32,
            min_visible_bbox_short_side_px_for_model=6,
        ),
        color_match_settings=_default_color_match_settings(enabled=False),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(
            max_ring_overshoot_luma=999.0,
            max_small_mask_brighten_luma=20.0,
        ),
        tiny_object_settings=harmonisation_runner.TinyObjectSettings(
            enabled=True,
            max_mask_pixels_for_conservative_path=16,
            max_bbox_short_side_px_for_conservative_path=3,
            skip_color_match_below_mask_pixels=16,
        ),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        temporal_settings=_default_temporal_settings(),
        temporal_state=temporal_state,
        adaptive_settings=_default_adaptive_settings(enabled=False),
        span_id=2,
        span_policy="learned_track_harmonization",
        reference_frame_for_track=12,
        force_propagated_arguments={"brightness": 0.12},
        track_id=0,
        applied_parameter_source="interpolated",
    )

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    written = np.asarray(harmonisation_runner.Image.open(output_dir / "000010.png").convert("RGB"))
    assert payload["postcheck_rejected"] is True
    assert payload["postcheck_reason"] == "small_mask_brightness_increase"
    assert payload["status"] == "harmonized_track_recovered_blend"
    assert payload["recovery_mode"] == "bounded_postcheck_blend"
    assert payload["recovery_strength"] > 0.0
    assert payload["rejected_candidate_masked_luma"] > payload["post_harmonization_masked_luma"]
    assert payload["post_harmonization_masked_luma"] > payload["pre_harmonization_masked_luma"]
    assert payload["propagated_parameters_used"] is True
    assert not np.array_equal(written, overlay_rgb)


def test_process_frame_adaptive_color_match_preserves_more_identity_without_local_support(
    tmp_path: Path,
) -> None:
    overlay_rgb = np.full((24, 24, 3), 100, dtype=np.uint8)
    overlay_rgb[9:15, 9:15] = [140, 70, 70]
    mask_gray = np.zeros((24, 24), dtype=np.uint8)
    mask_gray[10:13, 10:13] = 255
    output_dir = tmp_path / "harmonized_overlays"
    diagnostics_dir = tmp_path / "harmonized_overlays_diagnostics"
    propagated = harmonisation_runner.ColorMatchParameters(
        luminance_mean_delta=18.0,
        luminance_std_ratio=0.8,
        chroma_a_shift=6.0,
        chroma_b_shift=4.0,
        chroma_scale=0.7,
    )

    harmonisation_runner.process_frame(
        frame_idx=11,
        overlay_rgb=overlay_rgb,
        mask_gray=mask_gray,
        run_dir=tmp_path,
        output_dir=output_dir,
        harmonizer=_StubHarmonizer(),
        device="cpu",
        mask_threshold=0.05,
        bbox_expansion_scale=2.5,
        min_crop_size_px=10,
        empty_mask_behavior="copy_through",
        eligibility_settings=_default_eligibility_settings(
            min_visible_mask_pixels_for_model=32,
            min_visible_bbox_short_side_px_for_model=6,
        ),
        color_match_settings=_default_color_match_settings(enabled=True, min_ring_pixels=12),
        correction_clamp_settings=_default_correction_clamps(),
        postcheck_settings=_default_postcheck_settings(
            max_ring_overshoot_luma=999.0,
            max_small_mask_brighten_luma=999.0,
        ),
        tiny_object_settings=harmonisation_runner.TinyObjectSettings(
            enabled=True,
            max_mask_pixels_for_conservative_path=16,
            max_bbox_short_side_px_for_conservative_path=3,
            skip_color_match_below_mask_pixels=16,
        ),
        adaptive_settings=_default_adaptive_settings(),
        diagnostics_dir=diagnostics_dir,
        write_crop_debug_overlays=False,
        span_id=2,
        span_policy="learned_track_harmonization",
        reference_frame_for_track=12,
        force_propagated_arguments={"brightness": 0.05},
        force_propagated_color_match_parameters=propagated,
        track_id=0,
        applied_parameter_source="backfilled_from_future",
    )

    payload = json.loads(
        (diagnostics_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["color_match_applied"] is True
    assert payload["ring_pixel_count_filtered"] == 0
    assert payload["adaptive_color_match_strength"] < 0.4
    assert payload["adaptive_harmonizer_strength"] < 0.5
    assert payload["adaptive_synthetic_scene_score"] >= 0.0


def test_fit_track_argument_curve_backfills_early_frames_and_interpolates_middle() -> None:
    fitted, sources = harmonisation_runner._fit_track_argument_curve(
        {
            4: {"brightness": 0.2, "contrast": 0.4},
            8: {"brightness": 0.6, "contrast": 0.2},
        },
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    assert sources[1] == "backfilled_from_future"
    assert sources[4] == "direct_reference"
    assert sources[6] == "interpolated"
    assert sources[9] == "forward_propagated"
    assert fitted[1]["brightness"] == pytest.approx(fitted[2]["brightness"])
    assert fitted[9]["contrast"] < fitted[4]["contrast"]
    assert fitted[9]["contrast"] > 0.19
