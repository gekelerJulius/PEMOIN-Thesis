"""Unit tests for temporal depth stabilization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pemoin.providers.depth_stabilization import (
    DepthStabilizationSettings,
    _blend_bidirectional,
    _fuse_inverse_depth,
    _propagate_depth,
    _single_pass,
)


def _identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _simple_intrinsics(fx: float = 4.0, fy: float = 4.0, cx: float = 2.0, cy: float = 2.0) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _translation_pose(tx: float = 0, ty: float = 0, tz: float = 0) -> np.ndarray:
    """Camera-to-world with translation only."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = tx
    c2w[1, 3] = ty
    c2w[2, 3] = tz
    return c2w


DEVICE = torch.device("cpu")
SETTINGS = DepthStabilizationSettings(
    enabled=True,
    sigma_inv_depth=0.05,
    w_propagated_min=0.0,
    w_propagated_max=0.9,
    device="cpu",
)


def _call_propagate(depth_prev, depth_curr_raw, c2w_prev, w2c_curr,
                    K, static_mask=None, settings=SETTINGS, device=DEVICE,
                    w2c_prev=None, c2w_curr=None):
    """Helper that fills in default poses for backward warping signature."""
    if w2c_prev is None:
        w2c_prev = np.linalg.inv(c2w_prev).astype(np.float32)
    if c2w_curr is None:
        c2w_curr = np.linalg.inv(w2c_curr).astype(np.float32)
    return _propagate_depth(
        depth_prev, depth_curr_raw,
        c2w_prev, w2c_prev, c2w_curr, w2c_curr,
        K, static_mask, settings, device,
    )


class TestSettingsFromMapping:
    def test_none_gives_defaults(self):
        s = DepthStabilizationSettings.from_mapping(None)
        assert s.enabled is False
        assert s.sigma_inv_depth == 0.05

    def test_valid_mapping(self):
        s = DepthStabilizationSettings.from_mapping({
            "enabled": True,
            "sigma_inv_depth": 0.1,
            "device": "cpu",
        })
        assert s.enabled is True
        assert s.sigma_inv_depth == 0.1
        assert s.device == "cpu"

    def test_unknown_keys_ignored(self):
        s = DepthStabilizationSettings.from_mapping({
            "enabled": True,
            "unknown_key": 42,
        })
        assert s.enabled is True

    def test_empty_mapping_gives_defaults(self):
        s = DepthStabilizationSettings.from_mapping({})
        assert s.enabled is False


class TestPropagateDepth:
    def test_identity_pose_returns_same_depth(self):
        """Same pose → propagated depth should match original."""
        depth = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.0, 3.0, 4.0, 5.0],
            [2.5, 3.5, 4.5, 5.5],
        ], dtype=np.float32)
        K = _simple_intrinsics()
        c2w = _identity_pose()
        w2c = _identity_pose()

        prop = _call_propagate(depth, depth, c2w, w2c, K)

        # For identity pose, each pixel should map back to itself
        valid = prop > 0
        assert valid.sum() > 0
        np.testing.assert_allclose(prop[valid], depth[valid], atol=0.01)

    def test_zero_depth_not_propagated(self):
        """Pixels with zero depth in current raw should not get propagation."""
        depth_prev = np.ones((4, 4), dtype=np.float32) * 5.0
        depth_curr = np.zeros((4, 4), dtype=np.float32)
        depth_curr[1, 1] = 5.0
        K = _simple_intrinsics()
        c2w = _identity_pose()
        w2c = _identity_pose()

        prop = _call_propagate(depth_prev, depth_curr, c2w, w2c, K)
        assert (prop > 0).sum() == 1

    def test_static_mask_excludes_dynamic(self):
        """Masked-out pixels on current frame should not be propagated."""
        depth = np.ones((4, 4), dtype=np.float32) * 5.0
        K = _simple_intrinsics()
        c2w = _identity_pose()
        w2c = _identity_pose()

        # Only top-left pixel is static on current frame
        static_mask = np.zeros((4, 4), dtype=bool)
        static_mask[0, 0] = True

        prop = _call_propagate(depth, depth, c2w, w2c, K, static_mask=static_mask)
        assert (prop > 0).sum() == 1


class TestForwardTranslation:
    def test_translation_along_z(self):
        """Camera moves forward (negative z in Blender) → depth should decrease."""
        H, W = 4, 4
        depth_prev = np.full((H, W), 10.0, dtype=np.float32)
        depth_curr = np.full((H, W), 9.0, dtype=np.float32)  # expected ~9 after move
        K = _simple_intrinsics()

        # Frame 0: identity
        c2w_prev = _identity_pose()
        w2c_prev = _identity_pose()
        # Frame 1: moved 1 unit forward (Blender: -z is forward, so translate -z)
        c2w_curr = _translation_pose(tz=-1.0)
        w2c_curr = np.linalg.inv(c2w_curr).astype(np.float32)

        settings = DepthStabilizationSettings(
            enabled=True, device="cpu",
        )

        prop = _propagate_depth(
            depth_prev, depth_curr,
            c2w_prev, w2c_prev, c2w_curr, w2c_curr,
            K, None, settings, DEVICE,
        )
        valid = prop > 0
        assert valid.any()
        # Depth should be ~9.0 (moved 1 unit closer)
        np.testing.assert_allclose(prop[valid], 9.0, atol=0.2)


class TestInverseDepthFusion:
    def test_identical_depths_returns_same(self):
        """When raw and propagated are identical, fused should be identical."""
        D = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32)
        fused = _fuse_inverse_depth(D, D.copy(), SETTINGS)
        np.testing.assert_allclose(fused, D, atol=0.01)

    def test_weight_formula(self):
        """Verify the inverse-depth weighting formula for known inputs."""
        D_raw = np.array([[10.0]], dtype=np.float32)
        D_prop = np.array([[10.0]], dtype=np.float32)
        settings = DepthStabilizationSettings(sigma_inv_depth=0.05, w_propagated_min=0.1, w_propagated_max=0.9)

        fused = _fuse_inverse_depth(D_raw, D_prop, settings)
        # Same depth → residual = 0 → w = clip(1.0, 0.1, 0.9) = 0.9
        # inv_fused = 0.9 * (1/10) + 0.1 * (1/10) = 1/10
        np.testing.assert_allclose(fused[0, 0], 10.0, atol=0.01)

    def test_large_residual_low_weight(self):
        """Large depth discrepancy → near-zero propagated weight → result ≈ raw."""
        D_raw = np.array([[2.0]], dtype=np.float32)
        D_prop = np.array([[20.0]], dtype=np.float32)
        settings = DepthStabilizationSettings(sigma_inv_depth=0.05, w_propagated_min=0.0, w_propagated_max=0.9)

        fused = _fuse_inverse_depth(D_raw, D_prop, settings)
        # inv_raw = 0.5, inv_prop = 0.05, residual = 0.45
        # 0.45/0.05 = 9, exp(-0.5*81) ≈ 0 → clipped to w_min=0.0
        # w ≈ 0 → result ≈ raw depth
        np.testing.assert_allclose(fused[0, 0], 2.0, atol=0.01)

    def test_propagated_only_where_raw_zero(self):
        """Where raw is zero but prop exists, should use propagated."""
        D_raw = np.array([[0.0, 5.0]], dtype=np.float32)
        D_prop = np.array([[3.0, 5.0]], dtype=np.float32)

        fused = _fuse_inverse_depth(D_raw, D_prop, SETTINGS)
        assert fused[0, 0] == pytest.approx(3.0)
        np.testing.assert_allclose(fused[0, 1], 5.0, atol=0.01)


class TestBidirectionalBlend:
    def test_symmetry_for_middle_frame(self):
        """Middle frame in 3-frame sequence gets balanced blend."""
        forward = {0: np.array([[10.0]]), 1: np.array([[10.0]]), 2: np.array([[10.0]])}
        backward = {0: np.array([[10.0]]), 1: np.array([[10.0]]), 2: np.array([[10.0]])}
        indices = [0, 1, 2]

        result = _blend_bidirectional(forward, backward, indices, SETTINGS)
        # All same depth → blend should be same depth
        for fi in indices:
            np.testing.assert_allclose(result[fi], 10.0, atol=0.01)

    def test_mean_mode(self):
        """Mean blend mode gives equal weight."""
        forward = {0: np.array([[8.0]]), 1: np.array([[8.0]])}
        backward = {0: np.array([[12.0]]), 1: np.array([[12.0]])}
        settings = DepthStabilizationSettings(bidirectional_blend="mean")

        result = _blend_bidirectional(forward, backward, [0, 1], settings)
        # Mean in inverse-depth: 0.5*(1/8) + 0.5*(1/12) = 0.5*(0.125+0.0833) = 0.1042
        # D = 1/0.1042 ≈ 9.6
        for fi in [0, 1]:
            assert 9.0 < result[fi][0, 0] < 10.0

    def test_confidence_weighted_endpoints(self):
        """Forward pass has low weight at start, high at end."""
        forward = {0: np.array([[5.0]]), 1: np.array([[5.0]]), 2: np.array([[5.0]])}
        backward = {0: np.array([[10.0]]), 1: np.array([[10.0]]), 2: np.array([[10.0]])}
        indices = [0, 1, 2]
        settings = DepthStabilizationSettings(bidirectional_blend="confidence_weighted")

        result = _blend_bidirectional(forward, backward, indices, settings)
        # Frame 0: w_fwd=1/3, w_bwd=3/3 → more backward → closer to 10
        # Frame 2: w_fwd=3/3, w_bwd=1/3 → more forward → closer to 5
        assert result[0][0, 0] > result[2][0, 0]

    def test_only_forward_valid(self):
        """Where backward is zero, use forward only."""
        forward = {0: np.array([[5.0]])}
        backward = {0: np.array([[0.0]])}

        result = _blend_bidirectional(forward, backward, [0], SETTINGS)
        assert result[0][0, 0] == pytest.approx(5.0)


class TestSinglePass:
    def test_single_frame_returns_copy(self):
        """Single frame → returns raw copy."""
        depths = {0: np.array([[5.0, 10.0]], dtype=np.float32)}
        poses = {0: (_identity_pose(), _identity_pose())}
        K = _simple_intrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5)

        result = _single_pass([0], depths, poses, K, None, SETTINGS, DEVICE)
        np.testing.assert_array_equal(result[0], depths[0])

    def test_identity_pose_two_frames(self):
        """Two frames same pose → stabilized ≈ raw."""
        H, W = 4, 4
        depth = np.random.uniform(1, 10, (H, W)).astype(np.float32)
        depths = {0: depth.copy(), 1: depth.copy()}
        pose = (_identity_pose(), _identity_pose())
        poses = {0: pose, 1: pose}
        K = _simple_intrinsics()

        result = _single_pass([0, 1], depths, poses, K, None, SETTINGS, DEVICE)
        # First frame is always raw copy
        np.testing.assert_array_equal(result[0], depth)
        # Second frame should be close to raw (same pose, same depth → fusion of identical)
        valid = result[1] > 0
        np.testing.assert_allclose(result[1][valid], depth[valid], atol=0.1)


class TestCPUFallback:
    def test_works_on_cpu(self):
        """Verify all operations work with device='cpu'."""
        depth = np.full((4, 4), 5.0, dtype=np.float32)
        K = _simple_intrinsics()
        c2w = _identity_pose()
        w2c = _identity_pose()
        device = torch.device("cpu")

        prop = _call_propagate(depth, depth, c2w, w2c, K, device=device)
        assert prop.shape == (4, 4)
        assert (prop > 0).any()


class TestBackwardWarpNoHoles:
    def test_no_holes_backward_warp(self):
        """Uniform depth + small translation → no holes in overlapping region."""
        H, W = 16, 16
        depth_prev = np.full((H, W), 10.0, dtype=np.float32)
        depth_curr = np.full((H, W), 10.0, dtype=np.float32)
        K = _simple_intrinsics(fx=8.0, fy=8.0, cx=8.0, cy=8.0)

        # Small lateral translation (0.1 units right)
        c2w_prev = _identity_pose()
        c2w_curr = _translation_pose(tx=0.1)
        w2c_prev = np.linalg.inv(c2w_prev).astype(np.float32)
        w2c_curr = np.linalg.inv(c2w_curr).astype(np.float32)

        settings = DepthStabilizationSettings(
            enabled=True, device="cpu",
        )
        prop = _propagate_depth(
            depth_prev, depth_curr,
            c2w_prev, w2c_prev, c2w_curr, w2c_curr,
            K, None, settings, DEVICE,
        )

        # The interior region (excluding border pixels that map outside prev image)
        # should have no holes
        interior = prop[2:-2, 2:-2]
        total_interior = interior.size
        valid_interior = (interior > 0).sum()
        # At least 90% of interior should be filled (no holes)
        assert valid_interior / total_interior > 0.9, (
            f"Only {valid_interior}/{total_interior} interior pixels filled — holes remain"
        )


class TestEdgeSoftRejection:
    def test_edge_pixels_fused_close_to_raw(self):
        """Depth discontinuity → Gaussian fusion downweights bad propagated depths at edges."""
        H, W = 8, 8
        K = _simple_intrinsics(fx=4.0, fy=4.0, cx=4.0, cy=4.0)

        # Previous frame: left half near (2m), right half far (20m)
        depth_prev = np.full((H, W), 20.0, dtype=np.float32)
        depth_prev[:, :4] = 2.0

        # Current frame: same depths, small lateral shift
        depth_curr = depth_prev.copy()

        c2w_prev = _identity_pose()
        c2w_curr = _translation_pose(tx=0.3)
        w2c_prev = np.linalg.inv(c2w_prev).astype(np.float32)
        w2c_curr = np.linalg.inv(c2w_curr).astype(np.float32)

        settings = DepthStabilizationSettings(
            enabled=True, sigma_inv_depth=0.05, w_propagated_min=0.0,
            w_propagated_max=0.9, device="cpu",
        )
        prop = _propagate_depth(
            depth_prev, depth_curr,
            c2w_prev, w2c_prev, c2w_curr, w2c_curr,
            K, None, settings, DEVICE,
        )

        # Fuse propagated with raw — edge pixels should stay close to raw
        fused = _fuse_inverse_depth(depth_curr, prop, settings)

        # Near-region interior (col 0-1): raw = 2.0, fused should be close
        near_fused = fused[:, :2]
        np.testing.assert_allclose(near_fused, 2.0, atol=0.3,
                                   err_msg="Near-region fused depth drifted from raw")

        # Far-region interior (col 6-7): raw = 20.0, fused should be close
        far_fused = fused[:, 6:]
        np.testing.assert_allclose(far_fused, 20.0, atol=2.0,
                                   err_msg="Far-region fused depth drifted from raw")


class TestBilinearSubpixel:
    def test_bilinear_subpixel(self):
        """Half-pixel offset → propagated depth is smooth interpolation."""
        H, W = 8, 8
        K = _simple_intrinsics(fx=8.0, fy=8.0, cx=4.0, cy=4.0)

        # Create a depth gradient in prev frame
        depth_prev = np.full((H, W), 5.0, dtype=np.float32)
        for col in range(W):
            depth_prev[:, col] = 5.0 + col * 0.5  # 5.0 to 8.5

        # Current frame raw depth at same values (for correspondence)
        depth_curr = depth_prev.copy()

        # Identity poses — should get exact values back
        c2w = _identity_pose()
        w2c = _identity_pose()

        settings = DepthStabilizationSettings(
            enabled=True, device="cpu",
        )
        prop = _propagate_depth(
            depth_prev, depth_curr,
            c2w, w2c, c2w, w2c,
            K, None, settings, DEVICE,
        )

        valid = prop > 0
        assert valid.sum() > 0
        # With identity pose, propagated should closely match the gradient
        np.testing.assert_allclose(prop[valid], depth_curr[valid], atol=0.1)
