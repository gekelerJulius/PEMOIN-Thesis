from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Vec3 = tuple[float, float, float]
LIGHTING_PRESET_NEUTRAL_HEMISPHERE = "neutral_hemisphere"
ALLOWED_LIGHTING_PRESETS = (LIGHTING_PRESET_NEUTRAL_HEMISPHERE,)
ALLOWED_SHADOW_CUBE_SIZES = ("512", "1024", "2048", "4096")
ALLOWED_RENDER_ENGINES = ("raster",)


@dataclass(frozen=True)
class LightSpec:
    """Concrete Blender light definition."""

    name: str
    kind: Literal["SUN", "AREA", "POINT"]
    energy: float
    rotation_euler_deg: tuple[float, float, float]
    color: tuple[float, float, float]
    role: str = "custom"
    casts_shadow: bool = True
    angle_deg: float | None = None
    area_size: tuple[float, float] | None = None
    location: tuple[float, float, float] | None = None
    placement_mode: str = "world_absolute"
    placement_target: str = "world"
    relative_location: tuple[float, float, float] | None = None
    transport_mode: str | None = None


@dataclass(frozen=True)
class WrapSubjectFillSpec:
    """Blender-side realization controls for subject-relative wrap fills."""

    global_strength_scale: float = 2.0
    wrap_key_role_scale: float = 0.08
    counter_wrap_role_scale: float = 0.035
    sky_fill_role_scale: float = 0.02
    counter_side_lift_bias: float = 0.6
    sky_softness_bias: float = 0.55
    direct_preservation_bias: float = 0.35
    raw_exposure_trim: float = 1.0


@dataclass(frozen=True)
class LightingRigSpec:
    """Scene-wide lighting configuration."""

    enabled: bool = True
    preset: Literal["neutral_hemisphere"] = LIGHTING_PRESET_NEUTRAL_HEMISPHERE
    ambient_world_strength: float = 0.12
    shadow_cube_size: Literal["512", "1024", "2048", "4096"] = "2048"
    wrap_subject_fill: WrapSubjectFillSpec = field(default_factory=WrapSubjectFillSpec)
    lights: tuple[LightSpec, ...] = ()


@dataclass(frozen=True)
class EdgeTreatmentSpec:
    """Boundary-only treatment for pedestrian overlay edges."""

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
class TemporalOcclusionStabilizationSpec:
    """Temporal hysteresis controls for unstable small-actor occlusion."""

    enabled: bool = True
    base_hysteresis_margin_m: float = 0.02
    state_flip_persist_frames: int = 2
    edge_exit_hold_frames: int = 2
    max_single_frame_visible_area_drop_ratio: float = 0.5


@dataclass(frozen=True)
class OcclusionSpec:
    """Contact-aware overlay occlusion configuration."""

    depth_source: Literal["z_pass"] = "z_pass"
    contact_ground_labels: tuple[str, ...] = ("road", "sidewalk")
    default_front_margin_m: float = 0.03
    relative_margin: float = 0.01
    contact_plane_band_m: float = 0.025
    contact_patch_radius_m: float = 0.30
    contact_coplanar_tolerance_m: float = 0.03
    write_debug: bool = True
    edge_treatment: EdgeTreatmentSpec = field(default_factory=EdgeTreatmentSpec)
    temporal_stabilization: TemporalOcclusionStabilizationSpec = field(
        default_factory=TemporalOcclusionStabilizationSpec
    )


@dataclass(frozen=True)
class ShadowSpec:
    """Shadow-catcher render and Python compositing controls."""

    enabled: bool = True
    receiver_patch_size_m: float = 4.0
    map_resolution: Literal["512", "1024", "2048", "4096"] = "1024"
    softness: float = 1.5
    opacity: float = 1.0
    tint_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RawSubjectExposureSpec:
    """Clip-level brightness calibration for raw pedestrian renders."""

    enabled: bool = True
    target_match_strength: float = 0.75
    max_gain: float = 2.5
    validation_tolerance: float = 0.18
    pedestrian_reference_weight: float = 0.7
    min_pedestrian_reference_pixels: int = 48


@dataclass(frozen=True)
class RenderPerformanceSpec:
    """Renderer-side fast-path toggles for pipeline-internal Blender artifacts."""

    persistent_data: bool = True
    fast_png_compression: bool = True
    disable_raytracing: bool = True
    disable_volumetric_shadows: bool = True
    disable_volumetric_lighting: bool = True
    disable_bloom: bool = True
    disable_screen_space_reflections: bool = True
    disable_gtao: bool = True
    disable_motion_blur: bool = True
    disable_high_quality_normals: bool = True


@dataclass(frozen=True)
class SalienceAdaptiveRenderSpec:
    """Protection-first render-cost reductions for only disposable visible frames."""

    enabled: bool = True
    low_salience_resolution_scale: float = 0.85
    protect_below_visible_pixels: int = 10000
    protect_below_bbox_short_side_px: int = 56
    protect_when_center_distance_ratio_below: float = 0.30
    reduce_only_when_boundary_fraction_above: float = 0.24
    reduce_only_near_visibility_transition: bool = True
    shadow_quality_reduction_enabled: bool = True
    fill_light_reduction_enabled: bool = True


@dataclass(frozen=True)
class RenderSpec:
    """Fast raster render-budget controls for Blender outputs."""

    engine: Literal["raster"] = "raster"
    resolution_scale: float = 1.0
    samples: int = 16
    performance: RenderPerformanceSpec = field(default_factory=RenderPerformanceSpec)
    material_policy: Literal[
        "preserve_most_maps",
        "preserve_base_alpha_normal",
        "preserve_base_alpha",
    ] = "preserve_base_alpha_normal"
    dynamic_light_binding: Literal[
        "copy_location_constraint",
        "sparse_keyframes",
        "spawn_only_static",
    ] = "copy_location_constraint"
    salience_adaptive: SalienceAdaptiveRenderSpec = field(
        default_factory=SalienceAdaptiveRenderSpec
    )
    raw_subject_exposure: RawSubjectExposureSpec = field(
        default_factory=RawSubjectExposureSpec
    )


@dataclass(frozen=True)
class SceneSpec:
    """Root configuration for the entire scene."""

    run_dir: Path
    trajectory_path: Path
    output_path: Path | None
    cube_size: float
    collection_name: str
    host_python: Path | None = None
    road_plane_gap: float = 0.05
    mixamo_character_fbx_path: Path | None = None
    mixamo_animation_fbx_path: Path | None = None
    mixamo_asset_root: Path | None = None
    pedestrian_actor_name: str = "Pedestrian01"
    pedestrian_placement_mode: Literal["trajectory_relative", "unity_world_horizontal"] = (
        "trajectory_relative"
    )
    pedestrian_authored_position_x_m: float | None = None
    pedestrian_authored_position_z_m: float | None = None
    pedestrian_authored_heading_yaw_deg: float | None = None
    pedestrian_authoring_to_canonical_transform: tuple[tuple[float, ...], ...] | None = None
    pedestrian_authoring_frame_metadata: dict[str, Any] = field(default_factory=dict)
    pedestrian_resolved_spawn_world: tuple[float, float, float] | None = None
    pedestrian_resolved_forward_world: tuple[float, float, float] | None = None
    pedestrian_resolved_heading_world_deg: float | None = None
    pedestrian_trajectory_t: float = 0.0
    pedestrian_forward_offset_m: float = 5.0
    pedestrian_left_offset_m: float = 2.0
    pedestrian_up_offset_m: float = 0.0
    pedestrian_heading_deg: float = 0.0
    pedestrian_motion_policy: Literal[
        "auto", "stationary_at_spawn", "animation_root_motion", "camera_trajectory_relative"
    ] = "auto"
    mixamo_scene_fps: float | None = None
    mixamo_export_fps: float = 30.0
    mixamo_source_fps: float = 30.0
    mixamo_debug: bool = True
    sampling_fps: float | None = None
    global_plane_range_m: float = 25.0
    global_plane_min_range_m: float = 3.0
    global_plane_frame_window: int = 3
    global_plane_max_points_per_frame: int = 4000
    global_plane_confidence_threshold: float = 0.5
    global_plane_trim_ratio: float = 0.2
    road_labels: tuple[str, ...] = ("road",)
    local_support_radius_m: float = 2.5
    local_support_frame_window: int = 3
    local_support_min_points: int = 10
    local_support_plane_size_m: float = 0.6
    local_support_confidence_threshold: float = 0.0
    local_support_max_radius_m: float = 3.0
    local_support_radius_step_m: float = 0.5
    local_support_snap_to_nearest_road: bool = True
    local_support_snap_radius_m: float = 4.0
    local_support_temporal_hold_frames: int = 6
    local_support_temporal_hold_seconds: float | None = None
    local_support_snap_max_vertical_delta_m: float = 0.2
    local_support_snap_max_radius_ratio: float = 0.5
    local_support_prefilter_vertical_window_m: float = 0.75
    foot_contact_mode: Literal["nearest_plane", "mixamo_phase"] = "mixamo_phase"
    foot_contact_phase_offset: float = 0.0
    foot_contact_gait_cycle_frames: float | None = None
    foot_contact_left_stance_phase_ranges: tuple[tuple[float, float], ...] = ()
    foot_contact_right_stance_phase_ranges: tuple[tuple[float, float], ...] = ()
    foot_contact_auto_calibrate_phase_ranges: bool = True
    foot_contact_plane_mode: Literal["strict", "project", "off"] = "project"
    foot_contact_min_plane_confidence_for_projection: float = 0.35
    foot_contact_max_plane_dist_m: float = 0.08
    max_plane_center_xy_distance_m: float = 8.0
    foot_contact_max_speed_mps: float = 1.8
    foot_contact_min_stance_frames: int = 2
    foot_contact_min_swing_frames: int = 2
    support_anchor_smoothing_mode: Literal["contact_hysteresis", "contact_segment_lock"] = "contact_segment_lock"
    support_anchor_transfer_frames: int = 3
    support_anchor_switch_margin: float = 0.12
    support_anchor_dual_support_height_tol_m: float = 0.035
    support_anchor_max_z_step_m: float = 0.01
    support_anchor_flat_ground_normal_z_min: float = 0.97
    support_anchor_allow_vertical_motion_on_plane_change: bool = True
    support_anchor_plane_change_height_tol_m: float = 0.04
    support_anchor_same_plane_normal_tol_deg: float = 3.0
    support_anchor_same_plane_height_tol_m: float = 0.015
    support_anchor_locked_xy_drift_tol_m: float = 0.02
    trajectory_grounding_transition_frames: int = 4
    trajectory_grounding_max_step_m: float = 0.05
    trajectory_grounding_max_vertical_velocity_mps: float = 0.9
    trajectory_grounding_max_vertical_accel_mps2: float = 2.5
    render: RenderSpec = field(default_factory=RenderSpec)
    lighting: LightingRigSpec | None = None
    shadow: ShadowSpec = field(default_factory=ShadowSpec)
    occlusion: OcclusionSpec = field(default_factory=OcclusionSpec)


@dataclass(frozen=True)
class TrajectorySpec:
    """Configuration for trajectory visualization."""

    cube_size: float
    material_color: Vec3 = (0.8, 0.2, 0.2)


@dataclass(frozen=True)
class RoadPlaneSpec:
    """Configuration for road plane visualization."""

    normal: np.ndarray
    offset: float
    center: np.ndarray
    scale_u: float
    scale_v: float
    frame_index: int
    confidence: float = 1.0
    fit_point_count: int | None = None
    metadata: dict[str, Any] | None = None
    material_color: Vec3 = (0.1, 0.3, 0.8)
    material_alpha: float = 0.5


@dataclass(frozen=True)
class RoadSurfacePipelineResult:
    """Persisted road-plane outputs used for visualization and insertion grounding."""

    global_planes: dict[int, RoadPlaneSpec]


@dataclass(frozen=True)
class SupportSurfaceResolution:
    mode: Literal["local_fit", "persisted_blend", "hold_prev"]
    normal: np.ndarray
    offset: float
    confidence: float
    source_frame_indices: tuple[int, ...]
    local_fit_point_count: int | None
    local_fit_radius_m: float | None
    local_fit_residual_p90_m: float | None
    local_fit_inlier_ratio: float | None
    persisted_blend_candidate_count: int | None
    persisted_blend_disagreement_m: float | None
    held_from_previous: bool
    origin_mode: Literal["local_fit", "persisted_blend"] | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class GroundingDiagnostic:
    frame_index: int
    support_mode: str
    support_confidence: float | None
    support_source_frame_indices: tuple[int, ...]
    support_failure_reason: str | None
    sole_offset_m: float
    chosen_plane_frame_index: int | None
    chosen_plane_normal: np.ndarray | None
    chosen_plane_offset: float | None
    chosen_plane_center: np.ndarray | None
    chosen_plane_center_xy_distance_m: float | None
    selected_support_foot: str
    left_foot_before: np.ndarray | None
    right_foot_before: np.ndarray | None
    left_foot_after: np.ndarray | None
    right_foot_after: np.ndarray | None
    support_point_before: np.ndarray | None
    support_point_after: np.ndarray | None
    pre_correction_signed_distance_m: float | None
    post_correction_signed_distance_m: float | None
    left_post_signed_distance_m: float | None
    right_post_signed_distance_m: float | None
    support_jump_from_prev_deg: float | None
    support_height_jump_from_prev_m: float | None
    support_anchor_shift_from_prev_m: float | None
    dynamic_anchor_shift_limit_m: float | None
    applied_translation_world: np.ndarray
    plane_selection_rejected_for_locality: bool
    missing_left_foot: bool
    missing_right_foot: bool
    no_plane: bool
    selected_plane_source: str | None = None
    authored_root_world: np.ndarray | None = None
    grounded_root_world: np.ndarray | None = None
    root_support_offset_m: float | None = None
    plane_height_at_xy_m: float | None = None
    planned_z_delta_m: float | None = None
    vertical_velocity_mps: float | None = None
    vertical_accel_mps2: float | None = None
    traversal_segment_id: int | None = None
    plane_transition_phase: str | None = None
    visibility_culled: bool = False
    visibility_cull_reason: str | None = None
    frame_requires_support: bool = True
    previous_support_point_before: np.ndarray | None = None
    previous_support_point_after: np.ndarray | None = None
    relock_current_signed_distance_m: float | None = None
    relock_previous_signed_distance_m: float | None = None
    support_origin_mode: str | None = None
    relock_decision_reason: str | None = None
    nearest_persisted_plane_center_xy_distance_m: float | None = None
    effective_persisted_plane_locality_limit_m: float | None = None
    persisted_plane_locality_mode: str | None = None
    support_anchor_policy: str | None = None
    left_support_weight: float | None = None
    right_support_weight: float | None = None
    support_anchor_blended: np.ndarray | None = None
    support_height_raw_m: float | None = None
    support_height_filtered_m: float | None = None
    left_support_confidence: float | None = None
    right_support_confidence: float | None = None
    support_switch_decision: str | None = None
    support_transfer_state: str | None = None
    support_height_clamped: bool = False
    contact_phase: float | None = None
    contact_state_raw: str | None = None
    contact_state_clean: str | None = None
    contact_segment_id: int | None = None
    contact_segment_kind: str | None = None
    plant_lock_source_frame: int | None = None
    plant_target_world: np.ndarray | None = None
    plant_lock_error_m: float | None = None
    support_authority: str | None = None
    same_plane_continuity: bool | None = None
    continuity_break_reason: str | None = None
    plant_lock_xy_error_m: float | None = None
    applied_translation_xy_m: float | None = None
    xy_lock_mode: str | None = None
    xy_lock_clamped: bool = False
    support_state: str = "supported"
    visibility_contract_state: str | None = None


@dataclass(frozen=True, init=False)
class OverlayValidationDiagnostic:
    frame_index: int
    has_visible_pedestrian: bool
    internal_render_shape: tuple[int, int] | None
    overlay_shape: tuple[int, int] | None
    lowest_alpha_u: int | None
    lowest_alpha_v: int | None
    lowest_alpha_row_coverage_px: int
    touches_image_bottom: bool
    left_foot_projected_uv: np.ndarray | None
    right_foot_projected_uv: np.ndarray | None
    left_foot_visible_expected: bool
    right_foot_visible_expected: bool
    support_point_projected_uv: np.ndarray | None
    support_point_projected_visible: bool
    support_point_depth_m: float | None
    scene_depth_at_support_px_m: float | None
    support_point_occluded_by_scene: bool
    selected_foot_projected_uv: np.ndarray | None
    selected_foot_projected_visible: bool
    support_to_left_foot_px: float | None
    support_to_right_foot_px: float | None
    support_to_contact_foot_px: float | None
    contact_foot_comparison_mode: str
    support_to_silhouette_bottom_px: float | None
    support_to_selected_foot_px: float | None
    support_patch_road_fraction: float | None
    support_patch_nonroad_fraction: float | None
    support_patch_size_px: int
    road_region_validation_available: bool
    road_context_search_mode: str
    validation_passed: bool
    failure_reason: str | None
    support_mode: str
    selected_support_foot: str
    warning_flags: tuple[str, ...]
    touches_image_left: bool = False
    touches_image_right: bool = False
    selected_support_foot_expected_visible: bool = False
    contact_validation_state: Literal["verified", "degraded", "unverifiable", "hard_failure"] = "verified"
    contact_validation_trusted: bool = True
    abort_relevant: bool = True

    def __init__(
        self,
        *,
        frame_index: int,
        has_visible_pedestrian: bool,
        internal_render_shape: tuple[int, int] | None = None,
        overlay_shape: tuple[int, int] | None = None,
        lowest_alpha_u: int | None,
        lowest_alpha_v: int | None,
        lowest_alpha_row_coverage_px: int,
        touches_image_bottom: bool,
        left_foot_projected_uv: np.ndarray | None,
        right_foot_projected_uv: np.ndarray | None,
        left_foot_visible_expected: bool,
        right_foot_visible_expected: bool,
        support_point_projected_uv: np.ndarray | None,
        support_point_projected_visible: bool,
        support_point_depth_m: float | None,
        scene_depth_at_support_px_m: float | None,
        support_point_occluded_by_scene: bool,
        selected_foot_projected_uv: np.ndarray | None,
        selected_foot_projected_visible: bool,
        support_to_left_foot_px: float | None,
        support_to_right_foot_px: float | None,
        support_to_contact_foot_px: float | None,
        contact_foot_comparison_mode: str,
        support_to_silhouette_bottom_px: float | None,
        support_to_selected_foot_px: float | None,
        support_patch_road_fraction: float | None,
        support_patch_nonroad_fraction: float | None,
        support_patch_size_px: int,
        road_region_validation_available: bool,
        road_context_search_mode: str,
        validation_passed: bool,
        failure_reason: str | None,
        support_mode: str,
        selected_support_foot: str,
        warning_flags: tuple[str, ...],
        touches_image_left: bool = False,
        touches_image_right: bool = False,
        selected_support_foot_expected_visible: bool = False,
        contact_validation_state: Literal["verified", "degraded", "unverifiable", "hard_failure"] = "verified",
        contact_validation_trusted: bool = True,
        abort_relevant: bool = True,
    ) -> None:
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "has_visible_pedestrian", has_visible_pedestrian)
        object.__setattr__(self, "internal_render_shape", internal_render_shape)
        object.__setattr__(self, "overlay_shape", overlay_shape)
        object.__setattr__(self, "lowest_alpha_u", lowest_alpha_u)
        object.__setattr__(self, "lowest_alpha_v", lowest_alpha_v)
        object.__setattr__(self, "lowest_alpha_row_coverage_px", lowest_alpha_row_coverage_px)
        object.__setattr__(self, "touches_image_bottom", touches_image_bottom)
        object.__setattr__(self, "left_foot_projected_uv", left_foot_projected_uv)
        object.__setattr__(self, "right_foot_projected_uv", right_foot_projected_uv)
        object.__setattr__(self, "left_foot_visible_expected", left_foot_visible_expected)
        object.__setattr__(self, "right_foot_visible_expected", right_foot_visible_expected)
        object.__setattr__(self, "support_point_projected_uv", support_point_projected_uv)
        object.__setattr__(self, "support_point_projected_visible", support_point_projected_visible)
        object.__setattr__(self, "support_point_depth_m", support_point_depth_m)
        object.__setattr__(self, "scene_depth_at_support_px_m", scene_depth_at_support_px_m)
        object.__setattr__(self, "support_point_occluded_by_scene", support_point_occluded_by_scene)
        object.__setattr__(self, "selected_foot_projected_uv", selected_foot_projected_uv)
        object.__setattr__(self, "selected_foot_projected_visible", selected_foot_projected_visible)
        object.__setattr__(self, "support_to_left_foot_px", support_to_left_foot_px)
        object.__setattr__(self, "support_to_right_foot_px", support_to_right_foot_px)
        object.__setattr__(self, "support_to_contact_foot_px", support_to_contact_foot_px)
        object.__setattr__(self, "contact_foot_comparison_mode", contact_foot_comparison_mode)
        object.__setattr__(self, "support_to_silhouette_bottom_px", support_to_silhouette_bottom_px)
        object.__setattr__(self, "support_to_selected_foot_px", support_to_selected_foot_px)
        object.__setattr__(self, "support_patch_road_fraction", support_patch_road_fraction)
        object.__setattr__(self, "support_patch_nonroad_fraction", support_patch_nonroad_fraction)
        object.__setattr__(self, "support_patch_size_px", support_patch_size_px)
        object.__setattr__(self, "road_region_validation_available", road_region_validation_available)
        object.__setattr__(self, "road_context_search_mode", road_context_search_mode)
        object.__setattr__(self, "validation_passed", validation_passed)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "support_mode", support_mode)
        object.__setattr__(self, "selected_support_foot", selected_support_foot)
        object.__setattr__(self, "warning_flags", warning_flags)
        object.__setattr__(self, "touches_image_left", touches_image_left)
        object.__setattr__(self, "touches_image_right", touches_image_right)
        object.__setattr__(
            self,
            "selected_support_foot_expected_visible",
            selected_support_foot_expected_visible,
        )
        object.__setattr__(self, "contact_validation_state", contact_validation_state)
        object.__setattr__(self, "contact_validation_trusted", contact_validation_trusted)
        object.__setattr__(self, "abort_relevant", abort_relevant)


@dataclass(frozen=True)
class MixamoSpec:
    character_fbx: Path
    animation_fbx: Path
    asset_root: Path
    actor_name: str = "MIXAMO_ACTOR"
    root_name: str = "CHAR_ROOT"
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    heading_deg: float = 0.0
    global_scale: float = 1.0


@dataclass(frozen=True)
class ActorSupportContract:
    root_to_support_m: float
    support_samples_used: int


@dataclass(frozen=True)
class RenderVisibilityFrame:
    frame_index: int
    rendered_visible: bool
    rendered_alpha_pixels: int
    projected_visible: bool
    support_state: str
    visibility_contract_state: str | None


@dataclass(frozen=True)
class PersistedPlaneLocalityDecision:
    nearest_xy_distance_m: float
    effective_limit_m: float
    locality_mode: Literal["strict", "bootstrap_relaxed", "rejected", "no_planes"]


@dataclass(frozen=True)
class ContactFrameState:
    frame_index: int
    phase: float | None
    raw_state: str
    clean_state: str
    segment_id: int
    segment_kind: str
    segment_frame_index: int
    segment_length: int
    previous_stance_kind: str | None = None
    next_stance_kind: str | None = None


@dataclass(frozen=True)
class ContactSegment:
    segment_id: int
    kind: str
    frame_indices: tuple[int, ...]


@dataclass
class PlantLockState:
    foot: str
    target_world: np.ndarray
    source_frame_index: int
    support_plane: SupportSurfaceResolution
    segment_id: int


@dataclass(frozen=True)
class SupportAnchorSelection:
    anchor: np.ndarray
    left_weight: float
    right_weight: float
    label: str
    left_confidence: float
    right_confidence: float
    switch_decision: str
    transfer_state: str


@dataclass(frozen=True)
class SupportAnchorHeightFilterResult:
    anchor: np.ndarray
    raw_height: float
    filtered_height: float
    clamped: bool
