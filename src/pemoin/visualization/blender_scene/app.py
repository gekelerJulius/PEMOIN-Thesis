from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import numpy as np
from pemoin.data.contracts import ResourceStore

from pemoin.visualization.pedestrian_placement import (
    minimum_xy_distance_to_trajectory,
    resolve_pedestrian_spawn_world,
    validate_pedestrian_spawn_near_trajectory,
)

from ._blender_api import bpy
from .config import parse_args
from .grounding import (
    _raise_for_grounding_failures,
    _write_dynamic_lighting_anchor_diagnostics,
    _write_grounding_diagnostics,
    _write_road_surface_summary,
    _write_support_surface_diagnostics,
    _write_trajectory_height_profile,
    _write_trajectory_support_segments,
    apply_road_support_to_inserted_pedestrian,
    viz_road_planes,
)
from .lighting import (
    bind_dynamic_subject_lights,
    configure_render_engine,
    configure_scene_lighting,
)
from .logging import log_info
from .mixamo import export_mixamo_root_motion_fbx, insert_mixamo_character
from .overlay import compose_overlay_frames, render_pedestrian
from .scene_setup import (
    add_trajectory_cubes,
    clear_scene,
    create_animated_camera,
    ensure_collection,
    load_intrinsics,
    load_trajectory,
    save_blend,
)
from .specs import SceneSpec, TrajectorySpec


@dataclass(frozen=True)
class CameraSetupResult:
    intrinsics_matrix: np.ndarray
    width: int
    height: int
    intrinsics_metadata: dict[str, object]
    parity_solution: object


@dataclass(frozen=True)
class SpawnResolution:
    resolved_spawn_world_arr: np.ndarray
    trajectory_anchor_world_arr: np.ndarray
    motion_forward_world_arr: np.ndarray
    base_heading_world_deg: float
    spawn_min_distance_m: float


@dataclass(frozen=True)
class RenderOutputPaths:
    pedestrian_frames_dir: Path
    pedestrian_depth_frames_dir: Path
    overlay_frames_dir: Path
    overlay_support_local_grid_dir: Path
    occlusion_masks_dir: Path
    occlusion_debug_dir: Path
    shadow_frames_dir: Path | None = None


def run_scene(argv: list[str]) -> None:
    run_scene_from_spec(parse_args(argv))


def _setup_camera_and_trajectory(
    spec: SceneSpec,
    traj_collection,
) -> tuple[np.ndarray, np.ndarray, CameraSetupResult]:
    c2w, frame_indices = load_trajectory(spec.trajectory_path)
    traj_spec = TrajectorySpec(cube_size=spec.cube_size)
    add_trajectory_cubes(c2w, frame_indices, traj_spec, traj_collection)
    intrinsics_matrix, width, height, intrinsics_metadata = load_intrinsics(
        spec.run_dir,
        frame_indices,
    )
    _camera, parity_solution = create_animated_camera(
        c2w_matrices=c2w,
        frame_indices=frame_indices,
        intrinsics_matrix=intrinsics_matrix,
        width=width,
        height=height,
    )
    fx = float(intrinsics_matrix[0, 0])
    fy = float(intrinsics_matrix[1, 1])
    cx = float(intrinsics_matrix[0, 2])
    cy = float(intrinsics_matrix[1, 2])
    log_info(
        f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, width={width}, height={height}"
    )
    log_info(
        "Blender camera parity: "
        f"fit={parity_solution.sensor_fit} "
        f"focal_residual={parity_solution.focal_residual_px:.6f}px "
        f"principal_point_residual={parity_solution.principal_point_residual_px:.6f}px "
        f"resolution_source={intrinsics_metadata.get('intrinsics_resolution_source')}"
    )
    return c2w, frame_indices, CameraSetupResult(
        intrinsics_matrix=intrinsics_matrix,
        width=width,
        height=height,
        intrinsics_metadata=intrinsics_metadata,
        parity_solution=parity_solution,
    )


def _configure_scene_timing(spec: SceneSpec) -> None:
    if spec.sampling_fps is None:
        return
    scene = bpy.context.scene
    fps = float(spec.sampling_fps)
    if fps <= 0:
        raise ValueError(f"Invalid sampling_fps: {fps}")
    scene.render.fps = max(1, int(round(fps)))
    log_info(f"Setting scene FPS to {scene.render.fps} based on sampling_fps={fps}")
    scene.render.fps_base = scene.render.fps / fps


def _resolve_spawn(
    spec: SceneSpec,
    c2w: np.ndarray,
) -> SpawnResolution:
    if (
        spec.pedestrian_placement_mode == "unity_world_horizontal"
        and spec.pedestrian_resolved_spawn_world is not None
        and spec.pedestrian_resolved_forward_world is not None
        and spec.pedestrian_resolved_heading_world_deg is not None
    ):
        resolved_spawn_world_arr = np.asarray(
            spec.pedestrian_resolved_spawn_world,
            dtype=np.float32,
        ).reshape(3)
        motion_forward_world_arr = np.asarray(
            spec.pedestrian_resolved_forward_world,
            dtype=np.float32,
        ).reshape(3)
        trajectory_anchor_world_arr = np.asarray(
            spec.pedestrian_resolved_spawn_world,
            dtype=np.float32,
        ).reshape(3)
        spawn_min_distance_m = minimum_xy_distance_to_trajectory(
            c2w,
            resolved_spawn_world_arr,
        )
        log_info(
            "Resolved pedestrian spawn from Unity-authored placement: "
            f"authored=(x={float(spec.pedestrian_authored_position_x_m or 0.0):.3f}, "
            f"z={float(spec.pedestrian_authored_position_z_m or 0.0):.3f}, "
            f"yaw={float(spec.pedestrian_authored_heading_yaw_deg or 0.0):.3f}) "
            f"world={tuple(float(v) for v in resolved_spawn_world_arr.tolist())} "
            f"forward={tuple(float(v) for v in motion_forward_world_arr.tolist())} "
            f"heading_world_deg={float(spec.pedestrian_resolved_heading_world_deg):.3f} "
            f"min_xy_to_trajectory={spawn_min_distance_m:.3f}m"
        )
        return SpawnResolution(
            resolved_spawn_world_arr=resolved_spawn_world_arr,
            trajectory_anchor_world_arr=trajectory_anchor_world_arr,
            motion_forward_world_arr=motion_forward_world_arr,
            base_heading_world_deg=float(spec.pedestrian_resolved_heading_world_deg),
            spawn_min_distance_m=float(spawn_min_distance_m),
        )

    spawn_threshold_m = max(10.0, 2.0 * float(spec.global_plane_range_m))
    (
        resolved_spawn_world_arr,
        trajectory_anchor_world_arr,
        motion_forward_world_arr,
        base_heading_world_deg,
    ) = resolve_pedestrian_spawn_world(
        c2w,
        spec.pedestrian_trajectory_t,
        spec.pedestrian_forward_offset_m,
        spec.pedestrian_left_offset_m,
        spec.pedestrian_up_offset_m,
    )
    spawn_min_distance_m = validate_pedestrian_spawn_near_trajectory(
        c2w,
        resolved_spawn_world_arr,
        max_distance_m=spawn_threshold_m,
    )
    resolved_spawn_world = tuple(float(v) for v in resolved_spawn_world_arr.tolist())
    trajectory_anchor_world = tuple(float(v) for v in trajectory_anchor_world_arr.tolist())
    motion_forward_world = tuple(float(v) for v in motion_forward_world_arr.tolist())
    trajectory_heading_world_deg = float(base_heading_world_deg)
    log_info(
        "Resolved pedestrian spawn: "
        f"trajectory_t={float(spec.pedestrian_trajectory_t):.3f} "
        f"anchor={trajectory_anchor_world} forward={motion_forward_world} "
        f"offsets=(fwd={float(spec.pedestrian_forward_offset_m):.3f}, "
        f"left={float(spec.pedestrian_left_offset_m):.3f}, up={float(spec.pedestrian_up_offset_m):.3f}) "
        f"world={resolved_spawn_world} trajectory_heading_world_deg={trajectory_heading_world_deg:.3f} "
        f"pedestrian_heading_offset_deg={float(spec.pedestrian_heading_deg):.3f} "
        f"min_xy_to_trajectory={spawn_min_distance_m:.3f}m threshold={spawn_threshold_m:.3f}m"
    )
    return SpawnResolution(
        resolved_spawn_world_arr=resolved_spawn_world_arr,
        trajectory_anchor_world_arr=trajectory_anchor_world_arr,
        motion_forward_world_arr=motion_forward_world_arr,
        base_heading_world_deg=trajectory_heading_world_deg,
        spawn_min_distance_m=spawn_min_distance_m,
    )


def _render_outputs(
    spec: SceneSpec,
    camera_setup: CameraSetupResult,
    grounding_diagnostics,
) -> RenderOutputPaths:
    pedestrian_frames_dir = render_pedestrian(
        spec,
        render_width=camera_setup.width,
        render_height=camera_setup.height,
        target_intrinsics=camera_setup.intrinsics_matrix,
        parity_solution=camera_setup.parity_solution,
        grounding_diagnostics=grounding_diagnostics,
    )
    pedestrian_depth_frames_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "pedestrian_depth_frames",
    )
    shadow_frames_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "shadow_frames",
    )
    overlay_frames_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "overlayed_frames",
    )
    overlay_support_local_grid_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "overlayed_frames_support_local_grid",
    )
    occlusion_masks_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "occlusion_masks",
    )
    occlusion_debug_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "occlusion_debug",
    )
    return RenderOutputPaths(
        pedestrian_frames_dir=pedestrian_frames_dir,
        pedestrian_depth_frames_dir=pedestrian_depth_frames_dir,
        shadow_frames_dir=shadow_frames_dir,
        overlay_frames_dir=overlay_frames_dir,
        overlay_support_local_grid_dir=overlay_support_local_grid_dir,
        occlusion_masks_dir=occlusion_masks_dir,
        occlusion_debug_dir=occlusion_debug_dir,
    )


def run_scene_from_spec(spec: SceneSpec) -> None:
    clear_scene()
    traj_collection = ensure_collection(spec.collection_name)
    global_plane_collection = ensure_collection("RoadPlanesGlobal")
    c2w, frame_indices, camera_setup = _setup_camera_and_trajectory(spec, traj_collection)
    _configure_scene_timing(spec)
    configure_render_engine(spec)
    spawn = _resolve_spawn(spec, c2w)
    configure_scene_lighting(
        spec.lighting,
        run_dir=spec.run_dir,
        anchor_world=tuple(float(v) for v in spawn.resolved_spawn_world_arr.tolist()),
    )
    motion_direction_parity = insert_mixamo_character(
        spec,
        c2w_matrices=c2w,
        frame_indices=frame_indices,
        spawn_world=tuple(float(v) for v in spawn.resolved_spawn_world_arr.tolist()),
        trajectory_anchor_world=tuple(float(v) for v in spawn.trajectory_anchor_world_arr.tolist()),
        intended_forward_world=tuple(float(v) for v in spawn.motion_forward_world_arr.tolist()),
    )
    if spec.mixamo_animation_fbx_path is not None:
        export_manifest = export_mixamo_root_motion_fbx(
            spec=spec,
            actor_name=spec.pedestrian_actor_name,
            animation_fbx_path=Path(spec.mixamo_animation_fbx_path),
        )
        log_info(
            "Reusable character FBX export written: "
            f"{export_manifest['artifact_path']}"
        )
    road_surface = viz_road_planes(
        c2w=c2w,
        frame_indices=frame_indices,
        global_plane_collection=global_plane_collection,
        spec=spec,
    )
    grounding_diagnostics = apply_road_support_to_inserted_pedestrian(
        spec=spec,
        road_surface=road_surface,
        frame_indices=frame_indices,
        actor_name=spec.pedestrian_actor_name,
    )
    lighting_anchor_diagnostics = bind_dynamic_subject_lights(
        actor_name=spec.pedestrian_actor_name,
        frame_indices=frame_indices,
        binding_mode=str(
            getattr(getattr(spec, "render", None), "dynamic_light_binding", "copy_location_constraint")
        ),
    )
    if lighting_anchor_diagnostics:
        lighting_json = _write_dynamic_lighting_anchor_diagnostics(
            run_dir=spec.run_dir,
            diagnostics=lighting_anchor_diagnostics,
        )
        log_info(
            "Lighting anchor diagnostics written: "
            f"json={lighting_json} entries={len(lighting_anchor_diagnostics)}"
        )
    diag_json, diag_csv = _write_grounding_diagnostics(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    log_info(
        "Grounding diagnostics written: "
        f"json={diag_json} csv={diag_csv} entries={len(grounding_diagnostics)}"
    )
    support_json, support_csv = _write_support_surface_diagnostics(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    log_info(
        "Support-surface diagnostics written: "
        f"json={support_json} csv={support_csv} entries={len(grounding_diagnostics)}"
    )
    trajectory_segments_json = _write_trajectory_support_segments(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    trajectory_height_csv = _write_trajectory_height_profile(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    log_info(
        "Trajectory grounding diagnostics written: "
        f"segments={trajectory_segments_json} height_profile={trajectory_height_csv}"
    )
    _write_road_surface_summary(
        spec=spec,
        trajectory_anchor_world=tuple(float(v) for v in spawn.trajectory_anchor_world_arr.tolist()),
        motion_forward_world=tuple(float(v) for v in spawn.motion_forward_world_arr.tolist()),
        resolved_spawn_world=tuple(float(v) for v in spawn.resolved_spawn_world_arr.tolist()),
        base_heading_world_deg=float(spawn.base_heading_world_deg),
        resolved_heading_world_deg=(
            None
            if motion_direction_parity is None
            else motion_direction_parity.get("resolved_root_yaw_world_deg")
        ),
        spawn_min_distance_to_trajectory_m=spawn.spawn_min_distance_m,
        global_planes=road_surface.global_planes,
        grounding_diagnostics=grounding_diagnostics,
        motion_direction_parity=motion_direction_parity,
    )
    _raise_for_grounding_failures(
        diagnostics=grounding_diagnostics,
        max_residual_m=float(spec.foot_contact_max_plane_dist_m),
        max_plane_center_xy_distance_m=float(spec.max_plane_center_xy_distance_m),
    )
    render_outputs_params = inspect.signature(_render_outputs).parameters
    if "grounding_diagnostics" in render_outputs_params:
        render_outputs = _render_outputs(
            spec,
            camera_setup,
            grounding_diagnostics=grounding_diagnostics,
        )
    else:
        render_outputs = _render_outputs(spec, camera_setup)
    compose_overlay_frames(
        run_dir=spec.run_dir,
        actor_name=spec.pedestrian_actor_name,
        road_labels=spec.road_labels,
        contact_ground_labels=spec.occlusion.contact_ground_labels,
        occlusion_spec=spec.occlusion,
        shadow_spec=spec.shadow,
        grounding_diagnostics=grounding_diagnostics,
        original_frames_dir=spec.run_dir / "standard" / "frames",
        pedestrian_frames_dir=render_outputs.pedestrian_frames_dir,
        pedestrian_depth_frames_dir=render_outputs.pedestrian_depth_frames_dir,
        shadow_frames_dir=render_outputs.shadow_frames_dir,
        output_dir=render_outputs.overlay_frames_dir,
        debug_output_dir=None,
        support_debug_output_dir=None,
        support_local_grid_output_dir=render_outputs.overlay_support_local_grid_dir,
        occlusion_mask_output_dir=render_outputs.occlusion_masks_dir,
        occlusion_debug_output_dir=render_outputs.occlusion_debug_dir,
    )
    if spec.output_path:
        save_blend(spec.output_path)
        log_info(f"Scene saved to {spec.output_path}")
