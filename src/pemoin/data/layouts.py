"""Canonical standardized resource layouts."""

from __future__ import annotations

from typing import Mapping

from .models import ResourceKind, ResourceLayout

_STANDARD_LAYOUTS: Mapping[ResourceKind, ResourceLayout] = {
    ResourceKind.FRAMES: ResourceLayout(
        subdir="standard/frames",
        pattern="{frame:06d}.png",
        description="RGB frames in display resolution.",
    ),
    ResourceKind.INTRINSICS: ResourceLayout(
        subdir="standard/intrinsics",
        filename="intrinsics.npz",
        description="Camera intrinsics matrix and distortion.",
    ),
    ResourceKind.DEPTH: ResourceLayout(
        subdir="standard/depth",
        pattern="{frame:06d}.npz",
        description="Depth maps with optional confidence.",
    ),
    ResourceKind.TRAJECTORY: ResourceLayout(
        subdir="standard/trajectory",
        filename="poses.npz",
        description="Camera extrinsics per frame.",
    ),
    ResourceKind.TRAJECTORY_MATCH_GRAPH: ResourceLayout(
        subdir="standard/trajectory_match_graph",
        filename="dpvo_match_graph.npz",
        description="Canonical persisted trajectory match graph.",
    ),
    ResourceKind.SEMANTICS_2D: ResourceLayout(
        subdir="standard/semantics_2d",
        pattern="{frame:06d}.npz",
        description="Panoptic segmentation maps and labels.",
    ),
    ResourceKind.SEMANTICS_AUX: ResourceLayout(
        subdir="standard/semantics_aux",
        pattern="{frame:06d}.npz",
        description="Per-frame semantics sidecars such as probabilities and confidence maps.",
    ),
    ResourceKind.POINT_CLOUD_3D: ResourceLayout(
        subdir="standard/point_cloud_3d",
        filename="cloud.npz",
        description="Dense global point cloud with semantic labels and RGB colors.",
    ),
    ResourceKind.CAMERA_HEIGHT: ResourceLayout(
        subdir="standard/camera_height",
        pattern="{frame:06d}.npz",
        description="Camera height per frame in meters.",
    ),
    ResourceKind.ROAD_PLANE: ResourceLayout(
        subdir="standard/road_plane",
        pattern="{frame:06d}.npz",
        description="Road plane estimates per frame.",
    ),
    ResourceKind.ROAD_PLANE_SUPPORT: ResourceLayout(
        subdir="standard/road_plane_support",
        pattern="{frame:06d}.npz",
        description="Persisted road-plane support points and diagnostics per frame.",
    ),
    ResourceKind.DYNAMIC_MASK: ResourceLayout(
        subdir="standard/dynamic_mask",
        pattern="{frame:06d}.png",
        description="Binary dynamic-object masks (255=static, 0=dynamic).",
    ),
    ResourceKind.LIGHTING: ResourceLayout(
        subdir="standard/lighting",
        description="Clip-level lighting contract with JSON metadata and fused HDR envmap.",
    ),
}
