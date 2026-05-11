from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from pemoin.utils import harmonisation
from pemoin.utils.harmonisation import HarmonisationSettings, run_harmonisation


def test_harmonisation_settings_default_to_disabled_when_missing_block() -> None:
    settings = HarmonisationSettings.from_mapping({})
    assert settings.enabled is False


def test_harmonisation_settings_honor_explicit_enablement() -> None:
    settings = HarmonisationSettings.from_mapping({"enabled": True})
    assert settings.enabled is True


def test_harmonisation_settings_default_to_local_crop_defaults() -> None:
    settings = HarmonisationSettings.from_mapping({"enabled": True})
    assert settings.mode.value == "local_crop"
    assert settings.bbox_expansion_scale == pytest.approx(2.5)
    assert settings.min_crop_size_ratio == pytest.approx(0.30)
    assert settings.max_frame_coverage_ratio == pytest.approx(0.85)
    assert settings.containment_margin_px == 8
    assert settings.reject_when_actor_exceeds_crop is True
    assert settings.oversized_actor_behavior.value == "full_mask_affine_or_copy"
    assert settings.full_frame_affine_min_mask_pixels == 512
    assert settings.mask_source.value == "visible_occlusion"
    assert settings.empty_mask_behavior.value == "copy_through"
    assert settings.write_crop_diagnostics is True
    assert settings.write_crop_debug_overlays is False
    assert settings.eligibility.min_visible_mask_pixels_for_model == 48
    assert settings.eligibility.min_visible_bbox_short_side_px_for_model == 6
    assert settings.color_matching.enabled is True
    assert settings.color_matching.color_space.value == "lab"
    assert settings.color_matching.ring_inner_px == 10
    assert settings.color_matching.ring_outer_px == 40
    assert settings.color_matching.use_semantics_for_sky_filter is True
    assert settings.color_matching.luminance_strength == pytest.approx(0.60)
    assert settings.color_matching.chroma_strength == pytest.approx(0.30)
    assert settings.color_matching.prefer_pedestrian_reference is True
    assert settings.color_matching.pedestrian_reference_weight == pytest.approx(0.65)
    assert settings.color_matching.saturation_attenuation_strength == pytest.approx(0.35)
    assert settings.color_matching.contrast_attenuation_strength == pytest.approx(0.25)
    assert settings.correction_clamps.min_foreground_pixels_for_luminance_scale == 64
    assert settings.correction_clamps.luminance_delta_clamp_small_mask == pytest.approx(18.0)
    assert settings.postcheck.max_ring_overshoot_luma == pytest.approx(6.0)
    assert settings.temporal_smoothing.enabled is True
    assert settings.temporal_smoothing.mode.value == "parameter_ema"
    assert settings.temporal_smoothing.appearance_alpha == pytest.approx(0.85)
    assert settings.temporal_smoothing.tonal_alpha == pytest.approx(0.92)
    assert settings.temporal_smoothing.color_match_alpha == pytest.approx(0.85)
    assert settings.temporal_smoothing.reset_on_crop_iou_below == pytest.approx(0.25)
    assert settings.temporal_smoothing.reset_on_mask_area_ratio_outside == pytest.approx((0.5, 2.0))
    assert settings.tiny_object.enabled is True
    assert settings.temporal_smoothing.reset_on_copy_through is False
    assert settings.tiny_object.max_mask_pixels_for_conservative_path == 256
    assert settings.tiny_object.max_bbox_short_side_px_for_conservative_path == 20
    assert settings.tiny_object.skip_color_match_below_mask_pixels == 256
    assert settings.adaptive.enabled is True
    assert settings.adaptive.profile_bias == pytest.approx(0.0)
    assert settings.adaptive.min_effect_strength == pytest.approx(0.2)
    assert settings.adaptive.no_local_support_color_match_strength_cap == pytest.approx(0.35)
    assert settings.adaptive.no_local_support_harmonizer_strength_cap == pytest.approx(0.5)


def test_harmonisation_settings_reject_invalid_crop_configuration() -> None:
    with pytest.raises(ValueError, match="bbox_expansion_scale"):
        HarmonisationSettings.from_mapping({"bbox_expansion_scale": 1.0})

    with pytest.raises(ValueError, match="min_crop_size_ratio"):
        HarmonisationSettings.from_mapping({"min_crop_size_ratio": 0})
    with pytest.raises(ValueError, match="max_frame_coverage_ratio"):
        HarmonisationSettings.from_mapping({"max_frame_coverage_ratio": 1.5})
    with pytest.raises(ValueError, match="containment_margin_px"):
        HarmonisationSettings.from_mapping({"containment_margin_px": -1})
    with pytest.raises(ValueError, match="full_frame_affine_min_mask_pixels"):
        HarmonisationSettings.from_mapping({"full_frame_affine_min_mask_pixels": 0})

    with pytest.raises(ValueError, match="write_crop_debug_overlays requires"):
        HarmonisationSettings.from_mapping(
            {
                "write_crop_diagnostics": False,
                "write_crop_debug_overlays": True,
            }
        )
    with pytest.raises(ValueError, match="ring_outer_px"):
        HarmonisationSettings.from_mapping(
            {"color_matching": {"ring_inner_px": 20, "ring_outer_px": 20}}
        )
    with pytest.raises(ValueError, match="luminance_strength"):
        HarmonisationSettings.from_mapping(
            {"color_matching": {"luminance_strength": 1.5}}
        )
    with pytest.raises(ValueError, match="pedestrian_reference_weight"):
        HarmonisationSettings.from_mapping(
            {"color_matching": {"pedestrian_reference_weight": 1.5}}
        )
    with pytest.raises(ValueError, match="min_visible_mask_pixels_for_model"):
        HarmonisationSettings.from_mapping(
            {"eligibility": {"min_visible_mask_pixels_for_model": 0}}
        )
    with pytest.raises(ValueError, match="luminance_std_ratio_clamp"):
        HarmonisationSettings.from_mapping(
            {
                "correction_clamps": {
                    "luminance_std_ratio_clamp": [1.25, 0.75],
                }
            }
        )
    with pytest.raises(ValueError, match="appearance_alpha"):
        HarmonisationSettings.from_mapping(
            {"temporal_smoothing": {"appearance_alpha": 1.0}}
        )
    with pytest.raises(ValueError, match="reset_on_mask_area_ratio_outside"):
        HarmonisationSettings.from_mapping(
            {"temporal_smoothing": {"reset_on_mask_area_ratio_outside": [2.0, 1.0]}}
        )
    with pytest.raises(ValueError, match="max_mask_pixels_for_conservative_path"):
        HarmonisationSettings.from_mapping(
            {"tiny_object": {"max_mask_pixels_for_conservative_path": 0}}
        )
    with pytest.raises(ValueError, match="harmonisation.adaptive.profile_bias"):
        HarmonisationSettings.from_mapping({"adaptive": {"profile_bias": 1.5}})


def test_run_harmonisation_missing_pedestrian_frames_has_actionable_hint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    original_dir = run_dir / "standard" / "frames"
    original_dir.mkdir(parents=True)

    settings = HarmonisationSettings(enabled=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        run_harmonisation(run_dir, settings)

    message = str(excinfo.value)
    assert "Pedestrian frames directory not found" in message
    assert "runtime.settings.blender_scene.enabled=true" in message
    assert "disable runtime.settings.harmonisation.enabled" in message


def test_run_harmonisation_passes_local_crop_args_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "artifacts" / "blender" / "pedestrian_frames").mkdir(parents=True)
    (run_dir / "artifacts" / "blender" / "overlayed_frames").mkdir(parents=True)
    (run_dir / "artifacts" / "blender" / "occlusion_masks").mkdir(parents=True)
    pretrained = tmp_path / "harmonizer.pth"
    pretrained.write_bytes(b"stub")

    launched: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        launched["cmd"] = list(cmd)
        launched["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(harmonisation, "_default_env_launcher", lambda env_name: ("conda", "run", "-n", env_name))
    monkeypatch.setattr(harmonisation.subprocess, "run", _fake_run)

    settings = HarmonisationSettings(
        enabled=True,
        pretrained_path=str(pretrained),
        write_crop_debug_overlays=True,
    )

    output_dir = run_harmonisation(run_dir, settings)

    assert output_dir == (
        run_dir / "artifacts" / "harmonisation" / "harmonized_overlays"
    ).resolve()
    cmd = launched["cmd"]
    assert "--run-dir" in cmd and str(run_dir.resolve()) in cmd
    assert "--mode" in cmd and "local_crop" in cmd
    assert "--bbox-expansion-scale" in cmd and "2.5" in cmd
    assert "--min-crop-size-ratio" in cmd and "0.3" in cmd
    assert "--max-frame-coverage-ratio" in cmd and "0.85" in cmd
    assert "--containment-margin-px" in cmd and "8" in cmd
    assert "--reject-when-actor-exceeds-crop" in cmd
    assert "--oversized-actor-behavior" in cmd and "full_mask_affine_or_copy" in cmd
    assert "--full-frame-affine-min-mask-pixels" in cmd and "512" in cmd
    assert "--eligibility-min-visible-mask-pixels-for-model" in cmd and "48" in cmd
    assert "--mask-source" in cmd and "visible_occlusion" in cmd
    assert "--empty-mask-behavior" in cmd and "copy_through" in cmd
    assert "--color-match-enabled" in cmd
    assert "--color-match-color-space" in cmd and "lab" in cmd
    assert "--color-match-ring-inner-px" in cmd and "10" in cmd
    assert "--color-match-ring-outer-px" in cmd and "40" in cmd
    assert "--color-match-exclude-top-band" in cmd
    assert "--color-match-use-semantics-for-sky-filter" in cmd
    assert "--color-match-luminance-strength" in cmd and "0.6" in cmd
    assert "--color-match-chroma-strength" in cmd and "0.3" in cmd
    assert "--color-match-prefer-pedestrian-reference" in cmd
    assert "--color-match-pedestrian-reference-weight" in cmd and "0.65" in cmd
    assert "--color-match-saturation-attenuation-strength" in cmd and "0.35" in cmd
    assert "--color-match-contrast-attenuation-strength" in cmd and "0.25" in cmd
    assert "--correction-clamps-luminance-delta-clamp-small-mask" in cmd and "18.0" in cmd
    assert "--postcheck-max-ring-overshoot-luma" in cmd and "6.0" in cmd
    assert "--write-crop-diagnostics" in cmd
    assert "--write-crop-debug-overlays" in cmd
    assert "--temporal-smoothing-enabled" in cmd
    assert "--temporal-smoothing-mode" in cmd and "parameter_ema" in cmd
    assert "--temporal-smoothing-appearance-alpha" in cmd and "0.85" in cmd
    assert "--temporal-smoothing-tonal-alpha" in cmd and "0.92" in cmd
    assert "--temporal-smoothing-color-match-alpha" in cmd and "0.85" in cmd
    assert "--temporal-smoothing-reset-on-empty-mask" in cmd
    assert "--temporal-smoothing-reset-on-harmonizer-failure" in cmd
    assert "--temporal-smoothing-reset-on-crop-iou-below" in cmd and "0.25" in cmd
    assert "--tiny-object-enabled" in cmd
    assert "--temporal-smoothing-reset-on-copy-through" not in cmd
    assert "--tiny-object-max-mask-pixels-for-conservative-path" in cmd and "256" in cmd
    assert "--tiny-object-max-bbox-short-side-px-for-conservative-path" in cmd and "20" in cmd
    assert "--tiny-object-skip-color-match-below-mask-pixels" in cmd and "256" in cmd
    assert "--adaptive-enabled" in cmd
    assert "--adaptive-profile-bias" in cmd and "0.0" in cmd
    assert "--adaptive-min-effect-strength" in cmd and "0.2" in cmd
    assert "--adaptive-no-local-support-color-match-strength-cap" in cmd and "0.35" in cmd
    assert "--adaptive-no-local-support-harmonizer-strength-cap" in cmd and "0.5" in cmd


def test_harmonisation_env_launcher_uses_prefix_when_env_only_matches_path(monkeypatch) -> None:
    monkeypatch.setattr(harmonisation.shutil, "which", lambda name: "/usr/bin/micromamba")

    def _fake_find(manager: str, env_name: str):
        if manager == "micromamba" and env_name == "harmonizer":
            return ("micromamba", "run", "-p", "/home/juli/.local/share/mamba/envs/harmonizer")
        return None

    monkeypatch.setattr(harmonisation, "_find_env_launcher_for_manager", _fake_find)

    launcher = harmonisation._default_env_launcher("harmonizer")
    assert launcher == (
        "micromamba",
        "run",
        "-p",
        "/home/juli/.local/share/mamba/envs/harmonizer",
    )
