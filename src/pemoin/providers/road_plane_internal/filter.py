"""Temporal filtering for road-plane state."""

from __future__ import annotations

import numpy as np


class SimpleRoadStateFilter:
    """Linear Kalman filter over `[roll, pitch, plane_height_at_camera]`."""

    def __init__(self, settings):
        self._s = settings
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.Q = np.diag(
            [
                settings.state_process_noise_roll,
                settings.state_process_noise_pitch,
                settings.state_process_noise_height,
            ]
        ).astype(np.float32)
        self.R_base = np.diag(
            [
                settings.state_meas_noise_roll,
                settings.state_meas_noise_pitch,
                settings.state_meas_noise_height,
            ]
        ).astype(np.float32)
        self.last_update_accepted = True
        self.last_predict_only = False

    def predict(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.x is None or self.P is None:
            self.last_predict_only = True
            return None, None
        self.P = self.P + self.Q
        self.last_predict_only = True
        return self.x.copy(), np.diag(self.P).copy()

    def ensure_initialized(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x is None or self.P is None:
            self.x = measurement.astype(np.float32)
            self.P = np.eye(3, dtype=np.float32) * 0.05
            self.last_update_accepted = True
            self.last_predict_only = False
        return self.x.copy(), np.diag(self.P).copy()

    def update(
        self,
        measurement: np.ndarray,
        quality_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.ensure_initialized(measurement)
        R = self.R_base * float(max(1.0, quality_scale))

        y = measurement.astype(np.float32) - self.x
        S = self.P + R
        S_inv = np.linalg.pinv(S)
        maha = float(y.T @ S_inv @ y)
        if maha > self._s.state_innovation_gate:
            self.last_update_accepted = False
            self.last_predict_only = True
            return self.x.copy(), np.diag(self.P).copy()

        K = self.P @ S_inv
        self.x = self.x + K @ y
        identity = np.eye(3, dtype=np.float32)
        self.P = (identity - K) @ self.P
        self.last_update_accepted = True
        self.last_predict_only = False
        return self.x.copy(), np.diag(self.P).copy()
