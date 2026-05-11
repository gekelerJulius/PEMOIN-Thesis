"""Shared filesystem-backed compositor for depth-occluded pedestrian overlays."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from pemoin.utils.resolution import _resize_array

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover - blender env may not provide imageio
    imageio = None

from pemoin.geometry.camera_model import backproject_uv_depth_to_camera, camera_to_world

from .overlay_occlusion import (
    OcclusionFrameDiagnostics,
    OcclusionSettings,
    TemporalOcclusionState,
    _gaussian_blur,
    _apply_boundary_edge_treatment,
    compose_depth_occluded_rgba,
    load_depth_npz,
    write_edge_treatment_debug_artifacts,
    write_occlusion_debug_png,
    write_occlusion_mask_png,
)


def _resize_rgba_frame(frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if tuple(arr.shape[:2]) == tuple(target_shape):
        return arr
    try:
        from PIL import Image

        resample = Image.LANCZOS
        image = Image.fromarray(arr, mode="RGBA")
        resized = image.resize((int(target_shape[1]), int(target_shape[0])), resample=resample)
        return np.asarray(resized, dtype=np.uint8)
    except Exception:
        resized = _resize_array(arr, target_shape, interpolation="bilinear")
        return np.clip(np.rint(resized), 0.0, 255.0).astype(np.uint8)


def _resize_mask(mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.uint8)
    if tuple(arr.shape[:2]) == tuple(target_shape):
        return np.asarray(mask, dtype=bool)
    resized = _resize_array(arr, target_shape, interpolation="nearest")
    return np.asarray(resized, dtype=np.uint8) > 0


def _resize_depth_map(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if tuple(arr.shape[:2]) == tuple(target_shape):
        return arr
    return np.asarray(_resize_array(arr, target_shape, interpolation="bilinear"), dtype=np.float32)


def compose_shadow_on_background(
    *,
    background_rgb: np.ndarray,
    shadow_rgba: np.ndarray,
    opacity: float,
    blur_radius_px: float,
    tint_rgb: tuple[float, float, float],
) -> np.ndarray:
    """Darken or tint the background using a shadow RGBA pass."""
    bg = np.asarray(background_rgb, dtype=np.float32)
    shadow = np.asarray(shadow_rgba, dtype=np.uint8)
    if shadow.ndim != 3 or shadow.shape[2] < 4:
        raise ValueError(f"Shadow frame must be RGBA, got {shadow.shape}.")
    if tuple(shadow.shape[:2]) != tuple(bg.shape[:2]):
        raise ValueError(
            f"Shadow frame shape mismatch: background={bg.shape[:2]} shadow={shadow.shape[:2]}."
        )
    alpha = np.asarray(shadow[:, :, 3], dtype=np.float32) / 255.0
    if blur_radius_px > 0.0:
        alpha = _gaussian_blur(alpha, float(blur_radius_px))
    alpha = np.clip(alpha * float(opacity), 0.0, 1.0)
    tint = np.asarray(tint_rgb, dtype=np.float32).reshape(1, 1, 3) * 255.0
    composite = bg * (1.0 - alpha[:, :, None]) + tint * alpha[:, :, None]
    return np.clip(np.rint(composite), 0.0, 255.0).astype(np.uint8)


def _backproject_pedestrian_depth_to_world(
    *,
    ped_depth_m: np.ndarray,
    ped_rgba: np.ndarray,
    intrinsics_k: np.ndarray,
    camera_to_world_matrix: np.ndarray,
    alpha_threshold: float,
) -> np.ndarray:
    depth = np.asarray(ped_depth_m, dtype=np.float32)
    alpha = np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & (alpha > float(alpha_threshold))
    )
    world = np.full(depth.shape + (3,), np.nan, dtype=np.float32)
    if not np.any(valid):
        return world
    ys, xs = np.where(valid)
    uv = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    cam_points = backproject_uv_depth_to_camera(
        uv,
        depth[ys, xs].astype(np.float32),
        np.asarray(intrinsics_k, dtype=np.float32),
        camera_convention="blender",
    )
    world_points = camera_to_world(
        cam_points,
        np.asarray(camera_to_world_matrix, dtype=np.float32),
    )
    world[ys, xs] = np.asarray(world_points, dtype=np.float32)
    return world


def compose_overlay_frame_with_occlusion(
    *,
    frame_idx: int,
    original_frame_path: Path,
    pedestrian_rgba_path: Path,
    scene_depth_path: Path,
    pedestrian_depth_path: Path,
    settings: OcclusionSettings,
    mask_output_path: Path,
    debug_output_path: Path | None = None,
    force_alpha_only: bool = False,
    intrinsics_k: np.ndarray | None = None,
    camera_to_world_matrix: np.ndarray | None = None,
    traversable_ground_mask: np.ndarray | None = None,
    support_anchor_world: np.ndarray | None = None,
    support_plane_normal: np.ndarray | None = None,
    support_plane_offset: float | None = None,
    temporal_state: TemporalOcclusionState | None = None,
    shadow_rgba_path: Path | None = None,
    shadow_opacity: float = 1.0,
    shadow_blur_radius_px: float = 0.0,
    shadow_tint_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, OcclusionFrameDiagnostics]:
    """Compose a single frame and persist its visible-foreground occlusion mask."""
    if imageio is not None:
        bg = np.asarray(imageio.imread(original_frame_path), dtype=np.uint8)
        ped_rgba = np.asarray(imageio.imread(pedestrian_rgba_path), dtype=np.uint8)
    else:  # pragma: no cover
        try:
            import bpy  # type: ignore
        except Exception as exc:
            raise RuntimeError("imageio unavailable and bpy not found for PNG loading.") from exc

        def _load(path: Path) -> np.ndarray:
            image = bpy.data.images.load(str(path))
            try:
                width, height = int(image.size[0]), int(image.size[1])
                rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))
            finally:
                bpy.data.images.remove(image)
            rgba = np.flipud(rgba)
            rgba = np.clip(np.rint(rgba * 255.0), 0.0, 255.0).astype(np.uint8)
            return rgba

        bg = _load(original_frame_path)
        ped_rgba = _load(pedestrian_rgba_path)
    if bg.ndim == 2:
        bg = np.stack([bg] * 3, axis=-1)
    if ped_rgba.ndim != 3 or ped_rgba.shape[2] < 4:
        raise ValueError(
            f"Pedestrian frame must be RGBA, got {ped_rgba.shape} at {pedestrian_rgba_path}."
        )
    target_shape = tuple(int(v) for v in bg.shape[:2])
    if target_shape != tuple(ped_rgba.shape[:2]):
        ped_rgba = _resize_rgba_frame(ped_rgba, target_shape)
    if shadow_rgba_path is not None:
        if imageio is not None:
            shadow_rgba = np.asarray(imageio.imread(shadow_rgba_path), dtype=np.uint8)
        else:  # pragma: no cover
            try:
                import bpy  # type: ignore
            except Exception as exc:
                raise RuntimeError("imageio unavailable and bpy not found for PNG loading.") from exc

            def _load_shadow(path: Path) -> np.ndarray:
                image = bpy.data.images.load(str(path))
                try:
                    width, height = int(image.size[0]), int(image.size[1])
                    rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))
                finally:
                    bpy.data.images.remove(image)
                rgba = np.flipud(rgba)
                return np.clip(np.rint(rgba * 255.0), 0.0, 255.0).astype(np.uint8)

            shadow_rgba = _load_shadow(shadow_rgba_path)
        if tuple(shadow_rgba.shape[:2]) != target_shape:
            shadow_rgba = _resize_rgba_frame(shadow_rgba, target_shape)
        bg = compose_shadow_on_background(
            background_rgb=bg[:, :, :3],
            shadow_rgba=shadow_rgba,
            opacity=float(shadow_opacity),
            blur_radius_px=float(shadow_blur_radius_px),
            tint_rgb=shadow_tint_rgb,
        )

    scene_depth = None
    ped_depth = None
    if force_alpha_only:
        alpha = np.clip(np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0, 0.0, 1.0)
        presence = alpha > float(settings.alpha_presence_threshold)
        effective_alpha = alpha * presence.astype(np.float32)
        out_rgb_float, _, edge_debug, edge_stats = _apply_boundary_edge_treatment(
            background_rgb=bg[:, :, :3],
            pedestrian_rgb=ped_rgba[:, :, :3],
            visible_alpha=effective_alpha,
            settings=settings.edge_treatment,
            random_seed=int(frame_idx),
        )
        out_rgb = np.clip(np.rint(out_rgb_float), 0.0, 255.0).astype(np.uint8)
        visible_mask = effective_alpha > float(settings.alpha_visible_threshold)
        ped_count = int(np.count_nonzero(presence))
        visible_count = int(np.count_nonzero(visible_mask))
        diag = OcclusionFrameDiagnostics(
            frame_index=int(frame_idx),
            pedestrian_pixels=ped_count,
            visible_pixels=visible_count,
            occluded_pixels=ped_count - visible_count,
            visible_ratio=(0.0 if ped_count == 0 else float(visible_count) / float(ped_count)),
            min_scene_depth_m=None,
            max_scene_depth_m=None,
            min_ped_depth_m=None,
            max_ped_depth_m=None,
            median_depth_margin_m=None,
            boundary_pixels=int(edge_stats["boundary_pixels"]),
            feathered_pixels=int(edge_stats["feathered_pixels"]),
            despill_pixels=int(edge_stats["despill_pixels"]),
            blurred_pixels=int(edge_stats["blurred_pixels"]),
            regrained_pixels=int(edge_stats["regrained_pixels"]),
            estimated_background_noise_sigma=float(edge_stats["estimated_background_noise_sigma"]),
        )
    else:
        scene_depth = load_depth_npz(scene_depth_path)
        ped_depth = load_depth_npz(pedestrian_depth_path)
        if tuple(scene_depth.shape) != tuple(bg.shape[:2]):
            raise ValueError(
                f"Scene depth shape mismatch for frame {frame_idx}: "
                f"depth={scene_depth.shape}, image={bg.shape[:2]}."
            )
        if tuple(ped_depth.shape) != tuple(bg.shape[:2]):
            ped_depth = _resize_depth_map(ped_depth, target_shape)
        traversable_ground_mask = _resize_mask(traversable_ground_mask, target_shape)
        ped_world_points = None
        if intrinsics_k is not None and camera_to_world_matrix is not None:
            ped_world_points = _backproject_pedestrian_depth_to_world(
                ped_depth_m=ped_depth,
                ped_rgba=ped_rgba,
                intrinsics_k=np.asarray(intrinsics_k, dtype=np.float32),
                camera_to_world_matrix=np.asarray(camera_to_world_matrix, dtype=np.float32),
                alpha_threshold=float(settings.alpha_presence_threshold),
            )

        out_rgb, visible_mask, diag = compose_depth_occluded_rgba(
            background_rgb=bg[:, :, :3],
            pedestrian_rgba=ped_rgba,
            scene_depth_m=scene_depth,
            ped_depth_m=ped_depth,
            settings=settings,
            ped_world_points=ped_world_points,
            traversable_ground_mask=traversable_ground_mask,
            support_anchor_world=support_anchor_world,
            support_plane_normal=support_plane_normal,
            support_plane_offset=support_plane_offset,
            temporal_state=temporal_state,
            random_seed=int(frame_idx),
        )
        diag = replace(diag, frame_index=int(frame_idx))
        edge_debug = None

    write_occlusion_mask_png(mask_output_path, visible_mask)
    if debug_output_path is not None:
        alpha = np.clip(np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0, 0.0, 1.0)
        ped_presence = alpha > float(settings.alpha_presence_threshold)
        strict_margin = (
            np.maximum(
                float(settings.default_front_margin_m),
                float(settings.relative_margin) * scene_depth,
            )
            if not force_alpha_only and scene_depth is not None and ped_depth is not None
            else None
        )
        strict_visible = (
            ped_presence
            if force_alpha_only
            else (
                ped_presence
                & np.isfinite(ped_depth)
                & (ped_depth > 0.0)
                & np.isfinite(scene_depth)
                & (scene_depth > 0.0)
                & (ped_depth < (scene_depth - strict_margin))
            )
        )
        contact_override = visible_mask & (~strict_visible)
        write_occlusion_debug_png(
            debug_output_path,
            bg[:, :, :3],
            visible_mask,
            contact_candidate_mask=(
                None
                if traversable_ground_mask is None
                else (ped_presence & np.asarray(traversable_ground_mask, dtype=bool))
            ),
            contact_override_mask=contact_override,
            occluded_mask=(ped_presence & (~visible_mask)),
        )
        if force_alpha_only and edge_debug is not None:
            write_edge_treatment_debug_artifacts(
                debug_output_path,
                bg[:, :, :3],
                edge_debug,
            )
        elif not force_alpha_only:
            post_alpha = np.clip(np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0, 0.0, 1.0)
            post_alpha *= visible_mask.astype(np.float32)
            _, _, composed_edge_debug, _ = _apply_boundary_edge_treatment(
                background_rgb=bg[:, :, :3],
                pedestrian_rgb=ped_rgba[:, :, :3],
                visible_alpha=post_alpha,
                settings=settings.edge_treatment,
                random_seed=int(frame_idx),
            )
            if composed_edge_debug is not None:
                write_edge_treatment_debug_artifacts(
                    debug_output_path,
                    bg[:, :, :3],
                    composed_edge_debug,
                )

    return out_rgb, visible_mask, diag
