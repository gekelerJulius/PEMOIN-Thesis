from __future__ import annotations

import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from pemoin.visualization.overlay_compositor import (
    compose_overlay_frame_with_occlusion,
    compose_shadow_on_background,
)
from pemoin.visualization.overlay_occlusion import (
    _apply_boundary_edge_treatment,
    EdgeTreatmentSettings,
    OcclusionSettings,
    TemporalOcclusionState,
    compose_depth_occluded_rgba,
    compute_visible_pedestrian_mask,
)


def _base_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.zeros((3, 3), dtype=np.float32)
    alpha[1, 1] = 1.0
    ped_depth = np.zeros((3, 3), dtype=np.float32)
    ped_depth[1, 1] = 5.02
    scene_depth = np.full((3, 3), 5.0, dtype=np.float32)
    ped_world = np.full((3, 3, 3), np.nan, dtype=np.float32)
    ped_world[1, 1] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return alpha, ped_depth, scene_depth, ped_world


def test_contact_ground_override_preserves_coplanar_foot_pixels() -> None:
    alpha, ped_depth, scene_depth, ped_world = _base_inputs()
    traversable = np.zeros((3, 3), dtype=bool)
    traversable[1, 1] = True
    visible, diag = compute_visible_pedestrian_mask(
        ped_alpha=alpha,
        ped_depth_m=ped_depth,
        scene_depth_m=scene_depth,
        settings=OcclusionSettings(),
        ped_world_points=ped_world,
        traversable_ground_mask=traversable,
        support_anchor_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        support_plane_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        support_plane_offset=0.0,
    )

    assert bool(visible[1, 1]) is True
    assert diag.contact_candidate_pixels == 1
    assert diag.contact_override_pixels == 1
    assert diag.occluded_pixels == 0


def test_non_ground_pixels_still_use_strict_depth_occlusion() -> None:
    alpha, ped_depth, scene_depth, ped_world = _base_inputs()
    traversable = np.zeros((3, 3), dtype=bool)
    visible, diag = compute_visible_pedestrian_mask(
        ped_alpha=alpha,
        ped_depth_m=ped_depth,
        scene_depth_m=scene_depth,
        settings=OcclusionSettings(),
        ped_world_points=ped_world,
        traversable_ground_mask=traversable,
        support_anchor_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        support_plane_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        support_plane_offset=0.0,
    )

    assert bool(visible[1, 1]) is False
    assert diag.contact_candidate_pixels == 0
    assert diag.contact_override_pixels == 0
    assert diag.occluded_pixels == 1


def test_missing_semantics_raises_for_overlay_occlusion() -> None:
    alpha, ped_depth, scene_depth, ped_world = _base_inputs()

    with pytest.raises(ValueError, match="Traversable-ground semantics are required"):
        compute_visible_pedestrian_mask(
            ped_alpha=alpha,
            ped_depth_m=ped_depth,
            scene_depth_m=scene_depth,
            settings=OcclusionSettings(),
            ped_world_points=ped_world,
            traversable_ground_mask=None,
            support_anchor_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            support_plane_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            support_plane_offset=0.0,
        )


def test_ground_semantics_preserve_full_pedestrian_silhouette() -> None:
    alpha = np.ones((6, 1), dtype=np.float32)
    ped_depth = np.full((6, 1), 5.02, dtype=np.float32)
    scene_depth = np.full((6, 1), 5.0, dtype=np.float32)
    ped_world = np.zeros((6, 1, 3), dtype=np.float32)
    traversable = np.ones((6, 1), dtype=bool)

    visible, diag = compute_visible_pedestrian_mask(
        ped_alpha=alpha,
        ped_depth_m=ped_depth,
        scene_depth_m=scene_depth,
        settings=OcclusionSettings(),
        ped_world_points=ped_world,
        traversable_ground_mask=traversable,
    )

    assert visible[:, 0].tolist() == [True, True, True, True, True, True]
    assert diag.ground_exempt_candidate_pixels == 6
    assert diag.ground_exempt_pixels == 6
    assert diag.occluded_pixels == 0


def test_ground_semantics_only_exempt_ground_labeled_pixels() -> None:
    alpha = np.ones((6, 1), dtype=np.float32)
    ped_depth = np.full((6, 1), 5.02, dtype=np.float32)
    scene_depth = np.full((6, 1), 5.0, dtype=np.float32)
    ped_world = np.zeros((6, 1, 3), dtype=np.float32)
    traversable = np.asarray([[False], [False], [True], [True], [False], [False]], dtype=bool)

    visible, diag = compute_visible_pedestrian_mask(
        ped_alpha=alpha,
        ped_depth_m=ped_depth,
        scene_depth_m=scene_depth,
        settings=OcclusionSettings(),
        ped_world_points=ped_world,
        traversable_ground_mask=traversable,
    )

    assert visible[:, 0].tolist() == [False, False, True, True, False, False]
    assert diag.ground_exempt_candidate_pixels == 2
    assert diag.ground_exempt_pixels == 2


def test_edge_treatment_only_changes_boundary_pixels() -> None:
    bg = np.full((9, 9, 3), 40, dtype=np.uint8)
    bg[0::2, :, :] += 3
    ped = np.zeros((9, 9, 4), dtype=np.uint8)
    ped[2:7, 2:7, :3] = 200
    ped[2:7, 2:7, 3] = 255
    scene_depth = np.full((9, 9), 10.0, dtype=np.float32)
    ped_depth = np.full((9, 9), 5.0, dtype=np.float32)
    out, visible, diag = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=OcclusionSettings(
            edge_treatment=EdgeTreatmentSettings(
                boundary_band_px=1,
                tiny_object_disable_all_below_short_side_px=4,
                tiny_object_disable_all_below_visible_pixels=16,
                disable_when_boundary_fraction_above=1.0,
            )
        ),
        traversable_ground_mask=np.zeros((9, 9), dtype=bool),
    )

    assert visible[2:7, 2:7].all()
    assert np.array_equal(out[3:6, 3:6], np.full((3, 3, 3), 200, dtype=np.uint8))
    assert diag.boundary_pixels > 0
    assert diag.feathered_pixels > 0
    assert diag.blurred_pixels > 0
    assert diag.despill_pixels > 0
    assert diag.estimated_background_noise_sigma is not None


def test_edge_treatment_uses_visible_mask_not_raw_alpha() -> None:
    bg = np.full((9, 9, 3), 30, dtype=np.uint8)
    ped = np.zeros((9, 9, 4), dtype=np.uint8)
    ped[2:7, 2:7, :3] = 180
    ped[2:7, 2:7, 3] = 255
    scene_depth = np.full((9, 9), 10.0, dtype=np.float32)
    scene_depth[2:5, 2:7] = 4.0
    ped_depth = np.full((9, 9), 5.0, dtype=np.float32)

    out, visible, diag = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=OcclusionSettings(),
        traversable_ground_mask=np.zeros((9, 9), dtype=bool),
    )

    assert not visible[2:5, 2:7].any()
    assert visible[5:7, 2:7].all()
    assert np.array_equal(out[2:5, 2:7], bg[2:5, 2:7])
    assert diag.boundary_pixels > 0


def test_edge_treatment_can_be_disabled() -> None:
    bg = np.full((7, 7, 3), 25, dtype=np.uint8)
    ped = np.zeros((7, 7, 4), dtype=np.uint8)
    ped[1:6, 1:6, :3] = 150
    ped[1:6, 1:6, 3] = 255
    scene_depth = np.full((7, 7), 10.0, dtype=np.float32)
    ped_depth = np.full((7, 7), 5.0, dtype=np.float32)

    out, _, diag = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=OcclusionSettings(
            edge_treatment=EdgeTreatmentSettings(
                tiny_object_disable_all_below_short_side_px=4,
                tiny_object_disable_all_below_visible_pixels=16,
                disable_when_boundary_fraction_above=1.0,
            )
        ),
        traversable_ground_mask=np.zeros((7, 7), dtype=bool),
    )
    out_disabled, _, diag_disabled = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=OcclusionSettings(
            edge_treatment=EdgeTreatmentSettings(enabled=False)
        ),
        traversable_ground_mask=np.zeros((7, 7), dtype=bool),
    )

    assert diag_disabled.boundary_pixels == 0
    assert np.array_equal(out_disabled[2:5, 2:5], np.full((3, 3, 3), 150, dtype=np.uint8))
    assert not np.array_equal(out, out_disabled)


def test_edge_treatment_runs_for_alpha_only_composition_inputs() -> None:
    bg = np.full((7, 7, 3), 20, dtype=np.uint8)
    ped_rgb = np.full((7, 7, 3), 190, dtype=np.uint8)
    visible_alpha = np.zeros((7, 7), dtype=np.float32)
    visible_alpha[1:6, 1:6] = 1.0

    out_rgb, out_alpha, _, stats = _apply_boundary_edge_treatment(
        background_rgb=bg,
        pedestrian_rgb=ped_rgb,
        visible_alpha=visible_alpha,
        settings=EdgeTreatmentSettings(
            boundary_band_px=1,
            tiny_object_disable_all_below_short_side_px=4,
            tiny_object_disable_all_below_visible_pixels=16,
            disable_when_boundary_fraction_above=1.0,
        ),
        random_seed=7,
    )

    assert out_rgb.shape == (7, 7, 3)
    assert out_alpha.shape == (7, 7)
    assert stats["boundary_pixels"] > 0
    assert stats["feathered_pixels"] > 0


def test_temporal_edge_hold_diagnostics_do_not_count_stale_pixels() -> None:
    bg = np.full((5, 5, 3), 20, dtype=np.uint8)
    ped = np.zeros((5, 5, 4), dtype=np.uint8)
    ped[:, 0, :3] = 180
    ped[:, 0, 3] = 255
    empty_ped = np.zeros_like(ped)
    scene_depth = np.full((5, 5), 10.0, dtype=np.float32)
    ped_depth = np.full((5, 5), 5.0, dtype=np.float32)
    state = TemporalOcclusionState()
    settings = OcclusionSettings()

    compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=settings,
        temporal_state=state,
        traversable_ground_mask=np.zeros((5, 5), dtype=bool),
    )
    _, visible_mask, diag = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=empty_ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=settings,
        temporal_state=state,
        traversable_ground_mask=np.zeros((5, 5), dtype=bool),
    )

    assert int(np.count_nonzero(visible_mask)) == 0
    assert diag.visible_pixels == 0
    assert diag.occluded_pixels == 0


def test_edge_treatment_regrain_activates_when_background_has_local_noise() -> None:
    rng = np.random.default_rng(3)
    bg = np.full((11, 11, 3), 50, dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + rng.integers(-8, 9, size=(11, 11, 3)), 0, 255).astype(
        np.uint8
    )
    ped_rgb = np.full((11, 11, 3), 190, dtype=np.uint8)
    visible_alpha = np.zeros((11, 11), dtype=np.float32)
    visible_alpha[2:9, 2:9] = 1.0

    _, _, _, stats = _apply_boundary_edge_treatment(
        background_rgb=bg,
        pedestrian_rgb=ped_rgb,
        visible_alpha=visible_alpha,
        settings=EdgeTreatmentSettings(
            boundary_band_px=1,
            tiny_object_disable_all_below_short_side_px=4,
            tiny_object_disable_all_below_visible_pixels=16,
            disable_when_boundary_fraction_above=1.0,
        ),
        random_seed=5,
    )

    assert stats["estimated_background_noise_sigma"] > 0.0
    assert stats["regrained_pixels"] > 0


def test_edge_treatment_disables_destructive_ops_for_tiny_visible_masks() -> None:
    bg = np.full((7, 7, 3), 40, dtype=np.uint8)
    ped_rgb = np.full((7, 7, 3), 200, dtype=np.uint8)
    visible_alpha = np.zeros((7, 7), dtype=np.float32)
    visible_alpha[3, 3] = 1.0
    visible_alpha[3, 4] = 1.0

    _, _, _, stats = _apply_boundary_edge_treatment(
        background_rgb=bg,
        pedestrian_rgb=ped_rgb,
        visible_alpha=visible_alpha,
        settings=EdgeTreatmentSettings(boundary_band_px=4),
        random_seed=9,
    )

    assert stats["boundary_pixels"] > 0
    assert stats["feathered_pixels"] == 0
    assert stats["blurred_pixels"] == 0
    assert stats["despill_pixels"] == 0
    assert stats["regrained_pixels"] == 0


def test_edge_treatment_bypasses_when_boundary_dominates_visible_actor() -> None:
    bg = np.full((15, 15, 3), 40, dtype=np.uint8)
    ped_rgb = np.full((15, 15, 3), 200, dtype=np.uint8)
    visible_alpha = np.zeros((15, 15), dtype=np.float32)
    visible_alpha[3:12, 6:9] = 1.0

    _, _, _, stats = _apply_boundary_edge_treatment(
        background_rgb=bg,
        pedestrian_rgb=ped_rgb,
        visible_alpha=visible_alpha,
        settings=EdgeTreatmentSettings(
            boundary_band_px=2,
            disable_when_boundary_fraction_above=0.5,
        ),
        random_seed=5,
    )

    assert stats["boundary_pixels"] > 0
    assert stats["feathered_pixels"] == 0
    assert stats["blurred_pixels"] == 0
    assert stats["despill_pixels"] == 0
    assert stats["regrained_pixels"] == 0


def test_temporal_occlusion_stabilization_holds_small_edge_exit_drop() -> None:
    bg = np.full((10, 10, 3), 25, dtype=np.uint8)
    ped = np.zeros((10, 10, 4), dtype=np.uint8)
    ped[2:8, 7:10, :3] = 160
    ped[2:8, 7:10, 3] = 255
    scene_depth = np.full((10, 10), 10.0, dtype=np.float32)
    ped_depth = np.full((10, 10), 9.992, dtype=np.float32)
    ped_depth[:, 9] = 10.02
    settings = OcclusionSettings(
        default_front_margin_m=0.001,
        relative_margin=0.0,
        temporal_stabilization=OcclusionSettings().temporal_stabilization,
    )
    temporal_state = TemporalOcclusionState()

    _, visible_first, diag_first = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=settings,
        traversable_ground_mask=np.zeros((10, 10), dtype=bool),
        temporal_state=temporal_state,
    )
    assert diag_first.visible_pixels > 0

    ped_depth_second = ped_depth.copy()
    ped_depth_second[:, 8:] = 10.02
    _, visible_second, diag_second = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth_second,
        settings=settings,
        traversable_ground_mask=np.zeros((10, 10), dtype=bool),
        temporal_state=temporal_state,
    )

    assert diag_second.visible_pixels > 0
    assert diag_second.visible_pixels < diag_first.visible_pixels
    assert int(np.count_nonzero(visible_second)) > 0


def test_temporal_occlusion_stabilization_accepts_first_nonempty_appearance() -> None:
    bg = np.full((8, 8, 3), 25, dtype=np.uint8)
    ped = np.zeros((8, 8, 4), dtype=np.uint8)
    ped[2:6, 5:8, :3] = 160
    ped[2:6, 5:8, 3] = 255
    scene_depth = np.full((8, 8), 10.0, dtype=np.float32)
    ped_depth = np.full((8, 8), 9.99, dtype=np.float32)
    settings = OcclusionSettings(default_front_margin_m=0.001, relative_margin=0.0)
    temporal_state = TemporalOcclusionState()

    empty_ped = np.zeros_like(ped)
    empty_depth = np.zeros_like(scene_depth)
    _, empty_visible, empty_diag = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=empty_ped,
        scene_depth_m=scene_depth,
        ped_depth_m=empty_depth,
        settings=settings,
        traversable_ground_mask=np.zeros((8, 8), dtype=bool),
        temporal_state=temporal_state,
    )
    assert empty_diag.visible_pixels == 0
    assert int(np.count_nonzero(empty_visible)) == 0

    _, visible, diag = compose_depth_occluded_rgba(
        background_rgb=bg,
        pedestrian_rgba=ped,
        scene_depth_m=scene_depth,
        ped_depth_m=ped_depth,
        settings=settings,
        traversable_ground_mask=np.zeros((8, 8), dtype=bool),
        temporal_state=temporal_state,
    )

    assert diag.visible_pixels > 0
    assert int(np.count_nonzero(visible)) > 0


def test_shadow_composition_darkens_background_before_pedestrian_overlay() -> None:
    bg = np.full((3, 3, 3), 200, dtype=np.uint8)
    shadow = np.zeros((3, 3, 4), dtype=np.uint8)
    shadow[1, 1, 3] = 128

    out = compose_shadow_on_background(
        background_rgb=bg,
        shadow_rgba=shadow,
        opacity=1.0,
        blur_radius_px=0.0,
        tint_rgb=(0.0, 0.0, 0.0),
    )

    assert np.array_equal(out[0, 0], bg[0, 0])
    assert int(out[1, 1, 0]) < 200
    assert np.array_equal(out[1, 1], out[1, 1, 0] * np.ones(3, dtype=np.uint8))


def test_shadow_composition_supports_tint_and_blur_controls() -> None:
    bg = np.full((5, 5, 3), 180, dtype=np.uint8)
    shadow = np.zeros((5, 5, 4), dtype=np.uint8)
    shadow[2, 2, 3] = 255

    out = compose_shadow_on_background(
        background_rgb=bg,
        shadow_rgba=shadow,
        opacity=0.5,
        blur_radius_px=1.0,
        tint_rgb=(0.2, 0.1, 0.0),
    )

    assert int(out[2, 2, 0]) < 180
    assert int(out[2, 2, 0]) >= int(out[2, 2, 1]) >= int(out[2, 2, 2])
    assert int(out[2, 1, 0]) < 180


def test_compose_overlay_frame_with_occlusion_resizes_low_res_render_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        bg = np.full((8, 12, 3), 20, dtype=np.uint8)
        ped = np.zeros((4, 6, 4), dtype=np.uint8)
        ped[1:3, 2:4, :3] = 200
        ped[1:3, 2:4, 3] = 255
        shadow = np.zeros((4, 6, 4), dtype=np.uint8)
        shadow[2:, 1:5, 3] = 180
        scene_depth = np.full((8, 12), 10.0, dtype=np.float32)
        ped_depth = np.full((4, 6), 5.0, dtype=np.float32)
        traversable = np.ones((4, 6), dtype=bool)

        bg_path = root / "bg.png"
        ped_path = root / "ped.png"
        shadow_path = root / "shadow.png"
        scene_depth_path = root / "scene_depth.npz"
        ped_depth_path = root / "ped_depth.npz"
        mask_path = root / "mask.png"

        imageio.imwrite(bg_path, bg)
        imageio.imwrite(ped_path, ped)
        imageio.imwrite(shadow_path, shadow)
        np.savez_compressed(scene_depth_path, depth=scene_depth)
        np.savez_compressed(ped_depth_path, depth=ped_depth)

        out_rgb, visible_mask, diag = compose_overlay_frame_with_occlusion(
            frame_idx=0,
            original_frame_path=bg_path,
            pedestrian_rgba_path=ped_path,
            scene_depth_path=scene_depth_path,
            pedestrian_depth_path=ped_depth_path,
            settings=OcclusionSettings(),
            mask_output_path=mask_path,
            traversable_ground_mask=traversable,
            shadow_rgba_path=shadow_path,
        )

        assert out_rgb.shape == bg.shape
        assert visible_mask.shape == bg.shape[:2]
        assert diag.visible_pixels > 0


def test_compose_overlay_frame_with_occlusion_resizes_odd_dimension_render_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        bg = np.full((281, 500, 3), 20, dtype=np.uint8)
        ped = np.zeros((140, 250, 4), dtype=np.uint8)
        ped[50:120, 90:150, :3] = 200
        ped[50:120, 90:150, 3] = 255
        shadow = np.zeros((140, 250, 4), dtype=np.uint8)
        shadow[100:130, 80:170, 3] = 160
        scene_depth = np.full((281, 500), 10.0, dtype=np.float32)
        ped_depth = np.full((140, 250), 5.0, dtype=np.float32)
        traversable = np.zeros((281, 500), dtype=bool)
        traversable[150:, :] = True

        bg_path = root / "bg.png"
        ped_path = root / "ped.png"
        shadow_path = root / "shadow.png"
        scene_depth_path = root / "scene_depth.npz"
        ped_depth_path = root / "ped_depth.npz"
        mask_path = root / "mask.png"

        imageio.imwrite(bg_path, bg)
        imageio.imwrite(ped_path, ped)
        imageio.imwrite(shadow_path, shadow)
        np.savez_compressed(scene_depth_path, depth=scene_depth)
        np.savez_compressed(ped_depth_path, depth=ped_depth)

        out_rgb, visible_mask, diag = compose_overlay_frame_with_occlusion(
            frame_idx=0,
            original_frame_path=bg_path,
            pedestrian_rgba_path=ped_path,
            scene_depth_path=scene_depth_path,
            pedestrian_depth_path=ped_depth_path,
            settings=OcclusionSettings(),
            mask_output_path=mask_path,
            traversable_ground_mask=traversable,
            shadow_rgba_path=shadow_path,
        )

        assert out_rgb.shape == bg.shape
        assert visible_mask.shape == bg.shape[:2]
        assert diag.visible_pixels > 0
