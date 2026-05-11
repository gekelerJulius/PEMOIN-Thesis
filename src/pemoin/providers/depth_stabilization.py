"""Temporal depth stabilization via pose-based backward warping.

Uses bidirectional (forward + backward) passes with inverse-warp reprojection
and robust inverse-depth blending to reduce depth flicker across frames.
Dynamic objects are excluded using existing semantic masks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from pemoin.data.contracts import (
    DepthData,
    ResourceKind,
    ResourceStore,
)

LOG = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Settings
# ------------------------------------------------------------------ #
@dataclass(frozen=True)
class DepthStabilizationSettings:
    """Configuration for temporal depth stabilization."""

    enabled: bool = False
    sigma_inv_depth: float = 0.05
    w_propagated_min: float = 0.0
    w_propagated_max: float = 0.9
    bidirectional_blend: str = "confidence_weighted"  # "mean" or "confidence_weighted"
    use_dynamic_masks: bool = True
    min_valid_propagation_ratio: float = 0.05
    device: str = "cuda"

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> DepthStabilizationSettings:
        if mapping is None:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in mapping.items() if k in known}
        return cls(**kwargs)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def stabilize_depth_sequence(
    resource_store: ResourceStore,
    settings: DepthStabilizationSettings,
) -> None:
    """Stabilize all depth maps in *resource_store* in-place."""
    # 1. Intrinsics
    intrinsics = resource_store.load_intrinsics()
    K = np.asarray(intrinsics.matrix, dtype=np.float32)

    # 2. Frame indices
    frame_indices = resource_store.frame_indices(ResourceKind.DEPTH)
    if len(frame_indices) <= 1:
        LOG.debug("Depth stabilization skipped: ≤1 frame.")
        return

    # 3. Load all depth maps
    depths_raw: dict[int, np.ndarray] = {}
    for idx in frame_indices:
        dd = resource_store.load_depth(idx)
        depths_raw[idx] = dd.depth.astype(np.float32)

    # 4. Load trajectory
    traj_path = resource_store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        traj_frame_indices = data["frame_indices"].astype(int)
        c2w_all = data["camera_to_world"].astype(np.float32)
        w2c_all = (
            data["world_to_camera"].astype(np.float32)
            if "world_to_camera" in data.files
            else None
        )

    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, fi in enumerate(traj_frame_indices):
        fi = int(fi)
        if fi not in depths_raw:
            continue
        c2w = c2w_all[i]
        w2c = w2c_all[i] if w2c_all is not None else np.linalg.inv(c2w)
        poses[fi] = (c2w, w2c)

    # Filter frame_indices to those with both depth and pose
    frame_indices = [fi for fi in frame_indices if fi in poses]
    if len(frame_indices) <= 1:
        LOG.debug("Depth stabilization skipped: ≤1 frame with both depth and pose.")
        return

    # 5. Optionally load dynamic masks
    dynamic_masks: dict[int, np.ndarray] | None = None
    if settings.use_dynamic_masks and resource_store.has(ResourceKind.DYNAMIC_MASK):
        dynamic_masks = {}
        mask_indices = resource_store.frame_indices(ResourceKind.DYNAMIC_MASK)
        for idx in mask_indices:
            if idx in depths_raw:
                dm = resource_store.load_dynamic_mask(idx)
                dynamic_masks[idx] = dm.mask  # True=static
        if not dynamic_masks:
            dynamic_masks = None

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    LOG.info(
        "Depth stabilization: %d frames, device=%s", len(frame_indices), device
    )

    # 6. Forward pass
    forward = _single_pass(
        frame_indices, depths_raw, poses, K, dynamic_masks, settings, device
    )

    # 7. Backward pass
    backward = _single_pass(
        list(reversed(frame_indices)),
        depths_raw,
        poses,
        K,
        dynamic_masks,
        settings,
        device,
    )

    # 8. Bidirectional blend
    blended = _blend_bidirectional(forward, backward, frame_indices, settings)

    # 9. Overwrite depth
    for idx in frame_indices:
        resource_store.save_depth(
            DepthData(
                frame_index=idx,
                depth=blended[idx],
                metadata={"stabilized": True},
            )
        )
    LOG.info("Depth stabilization complete: %d frames updated.", len(frame_indices))


# ------------------------------------------------------------------ #
# Single-direction pass
# ------------------------------------------------------------------ #
def _single_pass(
    frame_indices: list[int],
    depths_raw: dict[int, np.ndarray],
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    dynamic_masks: dict[int, np.ndarray] | None,
    settings: DepthStabilizationSettings,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Run one temporal pass (forward or backward) and return stabilized depths."""
    result: dict[int, np.ndarray] = {}

    for i, fi in enumerate(frame_indices):
        D_raw = depths_raw[fi]
        if i == 0:
            result[fi] = D_raw.copy()
            continue

        prev_fi = frame_indices[i - 1]
        D_prev = result[prev_fi]
        c2w_prev, w2c_prev = poses[prev_fi]
        c2w_curr, w2c_curr = poses[fi]
        static_mask = dynamic_masks.get(fi) if dynamic_masks else None

        D_prop = _propagate_depth(
            D_prev, D_raw, c2w_prev, w2c_prev, c2w_curr, w2c_curr, K,
            static_mask, settings, device,
        )

        # Check propagation validity
        valid_prop = D_prop > 0
        prop_ratio = valid_prop.sum() / max(D_prop.size, 1)
        if prop_ratio < settings.min_valid_propagation_ratio:
            result[fi] = D_raw.copy()
        else:
            result[fi] = _fuse_inverse_depth(D_raw, D_prop, settings)

    return result


# ------------------------------------------------------------------ #
# Depth propagation via backward warping (inverse warp + grid_sample)
# ------------------------------------------------------------------ #
def _propagate_depth(
    depth_prev: np.ndarray,
    depth_curr_raw: np.ndarray,
    c2w_prev: np.ndarray,
    w2c_prev: np.ndarray,
    c2w_curr: np.ndarray,
    w2c_curr: np.ndarray,
    K: np.ndarray,
    static_mask: np.ndarray | None,
    settings: DepthStabilizationSettings,
    device: torch.device,
) -> np.ndarray:
    """Propagate depth from previous frame into current frame via backward warping.

    For each pixel in the *current* frame that has valid raw depth:
    1. Backproject using raw depth → 3D point in current camera
    2. Transform to previous camera frame
    3. Project to sub-pixel coords in previous image
    4. Bilinear-sample previous stabilized depth via grid_sample
    5. Filter to valid sampled depths (grid_sample padding returns 0)
    6. Re-project valid samples back to current camera → propagated depth

    Uses Blender camera convention:
    - Backproject: x = (u-cx)/fx * d, y = -(v-cy)/fy * d, z = -d
    - Project: in_front = z < 0, denom = -z, u = fx*(x/denom)+cx, v = fy*(-y/denom)+cy

    Returns (H, W) propagated depth map; 0 where no data.
    """
    H, W = depth_curr_raw.shape[:2]

    # Move matrices to device
    K_t = torch.from_numpy(K.astype(np.float32)).to(device)
    c2w_prev_t = torch.from_numpy(c2w_prev.astype(np.float32)).to(device)
    w2c_prev_t = torch.from_numpy(w2c_prev.astype(np.float32)).to(device)
    c2w_curr_t = torch.from_numpy(c2w_curr.astype(np.float32)).to(device)
    w2c_curr_t = torch.from_numpy(w2c_curr.astype(np.float32)).to(device)

    fx, fy = K_t[0, 0], K_t[1, 1]
    cx, cy = K_t[0, 2], K_t[1, 2]

    # Build full pixel grid for current frame
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    us_flat = us.reshape(-1)
    vs_flat = vs.reshape(-1)

    d_curr_flat = torch.from_numpy(
        depth_curr_raw.astype(np.float32)
    ).to(device).reshape(-1)

    # Valid mask: positive raw depth on current frame, and optionally static
    valid = d_curr_flat > 0
    if static_mask is not None:
        valid = valid & torch.from_numpy(static_mask.reshape(-1)).to(device)

    if valid.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    us_v = us_flat[valid]
    vs_v = vs_flat[valid]
    d_v = d_curr_flat[valid]

    # --- Step 1: Backproject current pixels to 3D (Blender convention) ---
    x_cam = (us_v - cx) / fx * d_v
    y_cam = -(vs_v - cy) / fy * d_v
    z_cam = -d_v

    # --- Step 2: Current camera → world → previous camera ---
    pts_cam = torch.stack(
        [x_cam, y_cam, z_cam, torch.ones_like(x_cam)], dim=1
    )  # (N, 4)
    pts_world = (c2w_curr_t @ pts_cam.T).T[:, :3]  # (N, 3)

    pts_h = torch.cat(
        [pts_world, torch.ones(pts_world.shape[0], 1, device=device)], dim=1
    )
    pts_prev = (w2c_prev_t @ pts_h.T).T[:, :3]  # (N, 3)

    # --- Step 3: Project into previous image (sub-pixel) ---
    z_prev = pts_prev[:, 2]
    in_front = z_prev < -1e-6
    denom = -z_prev.clamp(max=-1e-6)

    u_prev = fx * (pts_prev[:, 0] / denom) + cx
    v_prev = fy * (-pts_prev[:, 1] / denom) + cy
    # Filter: must be in front and inside image bounds (with 0.5px margin for bilinear)
    inside = (
        in_front
        & (u_prev >= 0)
        & (u_prev <= W - 1)
        & (v_prev >= 0)
        & (v_prev <= H - 1)
    )

    if inside.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    # --- Step 4: Bilinear-sample D_prev_stabilized at (u', v') ---
    # Prepare depth_prev as (1, 1, H, W) tensor for grid_sample
    D_prev_t = torch.from_numpy(
        depth_prev.astype(np.float32)
    ).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # grid_sample expects coords in [-1, 1]; convert from pixel coords
    u_norm = 2.0 * u_prev[inside] / (W - 1) - 1.0
    v_norm = 2.0 * v_prev[inside] / (H - 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

    d_sampled = torch.nn.functional.grid_sample(
        D_prev_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(-1)  # (N,)

    # --- Step 5: Filter to valid sampled depths ---
    sampled_valid = d_sampled > 0

    if sampled_valid.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    # --- Step 6: Re-project valid samples back to current camera ---
    # Backproject from prev camera using *sampled* depth
    u_prev_con = u_prev[inside][sampled_valid]
    v_prev_con = v_prev[inside][sampled_valid]
    d_sampled_con = d_sampled[sampled_valid]

    x_prev_cam = (u_prev_con - cx) / fx * d_sampled_con
    y_prev_cam = -(v_prev_con - cy) / fy * d_sampled_con
    z_prev_cam = -d_sampled_con

    pts_prev_cam = torch.stack(
        [x_prev_cam, y_prev_cam, z_prev_cam, torch.ones_like(x_prev_cam)], dim=1
    )
    pts_world2 = (c2w_prev_t @ pts_prev_cam.T).T[:, :3]

    pts_h2 = torch.cat(
        [pts_world2, torch.ones(pts_world2.shape[0], 1, device=device)], dim=1
    )
    pts_back_curr = (w2c_curr_t @ pts_h2.T).T[:, :3]

    d_prop_values = -pts_back_curr[:, 2]  # positive depth = -z in Blender
    d_prop_valid = d_prop_values > 0

    # Map valid pixels back to their original flat indices in the current frame
    # valid → inside → sampled_valid → d_prop_valid
    valid_indices = torch.where(valid)[0]
    inside_indices = valid_indices[inside]
    sampled_indices = inside_indices[sampled_valid]
    final_indices = sampled_indices[d_prop_valid]

    result = torch.zeros(H * W, device=device)
    result[final_indices] = d_prop_values[d_prop_valid]

    return result.reshape(H, W).cpu().numpy()


# ------------------------------------------------------------------ #
# Robust inverse-depth fusion
# ------------------------------------------------------------------ #
def _fuse_inverse_depth(
    D_raw: np.ndarray,
    D_prop: np.ndarray,
    settings: DepthStabilizationSettings,
) -> np.ndarray:
    """Fuse raw and propagated depth using inverse-depth weighting."""
    H, W = D_raw.shape[:2]
    result = D_raw.copy()

    raw_valid = D_raw > 0
    prop_valid = D_prop > 0
    both = raw_valid & prop_valid

    if both.any():
        inv_raw = np.zeros_like(D_raw)
        inv_prop = np.zeros_like(D_prop)
        inv_raw[raw_valid] = 1.0 / D_raw[raw_valid]
        inv_prop[prop_valid] = 1.0 / D_prop[prop_valid]

        residual = np.abs(inv_prop[both] - inv_raw[both])
        sigma = settings.sigma_inv_depth
        w = np.exp(-0.5 * (residual / sigma) ** 2)
        w = np.clip(w, settings.w_propagated_min, settings.w_propagated_max)

        inv_fused = np.zeros_like(D_raw)
        inv_fused[both] = w * inv_prop[both] + (1.0 - w) * inv_raw[both]

        fused_depth = np.zeros_like(D_raw)
        nonzero = inv_fused > 0
        fused_depth[nonzero] = 1.0 / inv_fused[nonzero]
        result[both] = fused_depth[both]

    # Where only propagated exists, use propagated
    prop_only = prop_valid & ~raw_valid
    if prop_only.any():
        result[prop_only] = D_prop[prop_only]

    return result


# ------------------------------------------------------------------ #
# Bidirectional blending
# ------------------------------------------------------------------ #
def _blend_bidirectional(
    forward: dict[int, np.ndarray],
    backward: dict[int, np.ndarray],
    frame_indices: list[int],
    settings: DepthStabilizationSettings,
) -> dict[int, np.ndarray]:
    """Blend forward and backward pass results."""
    N = len(frame_indices)
    result: dict[int, np.ndarray] = {}

    for i, fi in enumerate(frame_indices):
        D_fwd = forward[fi]
        D_bwd = backward[fi]

        fwd_valid = D_fwd > 0
        bwd_valid = D_bwd > 0
        both = fwd_valid & bwd_valid

        blended = np.zeros_like(D_fwd)

        if settings.bidirectional_blend == "confidence_weighted":
            w_fwd = (i + 1) / N
            w_bwd = (N - i) / N
        else:  # "mean"
            w_fwd = 0.5
            w_bwd = 0.5

        # Where both valid: blend in inverse-depth space
        if both.any():
            inv_fwd = 1.0 / D_fwd[both]
            inv_bwd = 1.0 / D_bwd[both]
            w_total = w_fwd + w_bwd
            inv_blended = (w_fwd * inv_fwd + w_bwd * inv_bwd) / w_total
            blended[both] = 1.0 / inv_blended

        # Where only one valid
        fwd_only = fwd_valid & ~bwd_valid
        bwd_only = bwd_valid & ~fwd_valid
        if fwd_only.any():
            blended[fwd_only] = D_fwd[fwd_only]
        if bwd_only.any():
            blended[bwd_only] = D_bwd[bwd_only]

        result[fi] = blended

    return result
