#!/usr/bin/env python3
# Sliding Window Marginalization.
# Fancy memory management for past poses.

import logging
import math
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("sliding_window")


@dataclass
class IMUPreintegration:
    """Preintegrated IMU measurements between two keyframes.

    Instead of propagating the full ESKF state at every IMU sample,
    we accumulate the relative motion (delta position, velocity, rotation)
    between keyframes. This "preintegrated" delta only needs to be
    recomputed when bias estimates change significantly.

    Reference: Forster et al., "On-Manifold Preintegration for
    Real-Time Visual-Inertial Odometry" (2017)
    """
    # Preintegrated deltas
    delta_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_q: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))

    # Jacobians w.r.t. bias (for bias correction without re-integration)
    J_p_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_q_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    # Preintegration covariance (9×9: dp, dv, dtheta)
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))

    # Integration metadata
    dt_sum: float = 0.0
    n_samples: int = 0

    # Bias at time of preintegration (for detecting significant changes)
    ba_ref: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg_ref: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Keyframe:
    """A keyframe in the sliding window."""
    timestamp: float
    position: np.ndarray       # NED (3,)
    velocity: np.ndarray       # NED (3,)
    quaternion: np.ndarray     # [w,x,y,z] (4,)
    accel_bias: np.ndarray     # (3,)
    gyro_bias: np.ndarray      # (3,)
    preintegration: Optional[IMUPreintegration] = None  # IMU to next keyframe
    observations: list = field(default_factory=list)     # visual observations


class SlidingWindowOptimizer:
    """Sliding-window marginalization for VIO/SLAM fusion.

    Maintains a fixed-size window of keyframe poses connected by:
      1. IMU preintegration factors (between consecutive keyframes)
      2. Visual/SLAM observation factors (tie keyframes to landmarks)

    When the window is full:
      1. Marginalize the oldest keyframe using Schur complement
      2. The marginalized information becomes a prior on remaining states
      3. This prior prevents information loss while bounding computation

    State vector for N keyframes:
      x_window = [p1, v1, q1, p2, v2, q2, ..., pN, vN, qN, ba, bg]
      (9 per keyframe + 6 shared biases = 9N + 6)

    The window optimizer runs at keyframe rate (~10-20 Hz), not IMU rate.
    The ESKF continues running at 100 Hz for prediction between keyframes.
    """

    # Maximum keyframes in the sliding window
    MAX_WINDOW_SIZE = 10

    # Minimum keyframes before optimization starts
    MIN_WINDOW_SIZE = 3

    # Bias change threshold for re-preintegration (m/s² and rad/s)
    BIAS_CHANGE_THRESHOLD_ACCEL = 0.05
    BIAS_CHANGE_THRESHOLD_GYRO = 0.005

    # Keyframe selection: minimum temporal spacing
    MIN_KEYFRAME_INTERVAL = 0.1  # seconds

    # Gauss-Newton optimization iterations
    MAX_ITERATIONS = 5
    CONVERGENCE_TOL = 1e-4

    def __init__(self, enable: bool = False):
        self._enabled = enable
        self._keyframes: List[Keyframe] = []
        self._prior_H: Optional[np.ndarray] = None  # marginalization prior Hessian
        self._prior_b: Optional[np.ndarray] = None   # marginalization prior residual
        self._prior_dim = 0

        # Current preintegration accumulator
        self._current_preint = IMUPreintegration()

        # Statistics
        self._keyframe_count = 0
        self._marginalization_count = 0
        self._optimization_count = 0

        if enable:
            log.info(f"Sliding window optimizer enabled "
                     f"(window={self.MAX_WINDOW_SIZE})")

    @property
    def is_active(self) -> bool:
        return self._enabled and len(self._keyframes) >= self.MIN_WINDOW_SIZE

    @property
    def window_size(self) -> int:
        return len(self._keyframes)

    def integrate_imu(self, accel: np.ndarray, gyro: np.ndarray,
                      dt: float, accel_bias: np.ndarray,
                      gyro_bias: np.ndarray):
        """Accumulate IMU measurement into preintegration.

        Called at IMU rate (100 Hz). Accumulates relative motion
        between keyframes without propagating the full state.

        Args:
            accel: bias-corrected accelerometer (m/s²)
            gyro: bias-corrected gyroscope (rad/s)
            dt: time step (seconds)
            accel_bias: current accel bias estimate
            gyro_bias: current gyro bias estimate
        """
        if not self._enabled:
            return

        preint = self._current_preint

        # Store reference bias on first sample
        if preint.n_samples == 0:
            preint.ba_ref = accel_bias.copy()
            preint.bg_ref = gyro_bias.copy()

        # Bias-corrected measurements
        a = accel - preint.ba_ref
        w = gyro - preint.bg_ref

        # Current rotation from preintegrated quaternion
        R_k = self._quat_to_dcm(preint.delta_q)

        # ── Update preintegrated states ────────────────────────
        # delta_p += delta_v * dt + 0.5 * R_k * a * dt²
        preint.delta_p += preint.delta_v * dt + 0.5 * R_k @ a * dt ** 2

        # delta_v += R_k * a * dt
        preint.delta_v += R_k @ a * dt

        # delta_q = delta_q ⊗ exp(w * dt)
        angle = np.linalg.norm(w) * dt
        if angle > 1e-10:
            axis = w / np.linalg.norm(w)
            ha = angle / 2.0
            dq = np.array([math.cos(ha),
                           axis[0] * math.sin(ha),
                           axis[1] * math.sin(ha),
                           axis[2] * math.sin(ha)])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        preint.delta_q = self._quat_mult(preint.delta_q, dq)
        preint.delta_q /= np.linalg.norm(preint.delta_q)

        # ── Update Jacobians w.r.t. bias ───────────────────────
        # These allow correcting the preintegration when bias changes
        # without redoing the full integration
        skew_a = self._skew(a)
        skew_w = self._skew(w)

        # J_v_ba += -R_k * dt
        preint.J_v_ba += -R_k * dt

        # J_v_bg += -R_k * skew(a) * J_q_bg * dt
        preint.J_v_bg += -R_k @ skew_a @ preint.J_q_bg * dt

        # J_p_ba += J_v_ba * dt - 0.5 * R_k * dt²
        preint.J_p_ba += preint.J_v_ba * dt - 0.5 * R_k * dt ** 2

        # J_p_bg += J_v_bg * dt - 0.5 * R_k * skew(a) * J_q_bg * dt²
        preint.J_p_bg += preint.J_v_bg * dt - 0.5 * R_k @ skew_a @ preint.J_q_bg * dt ** 2

        # J_q_bg += -(I - skew(w*dt)) * J_q_bg - I * dt  (approximation)
        preint.J_q_bg = (np.eye(3) - skew_w * dt) @ preint.J_q_bg - np.eye(3) * dt

        # ── Update covariance ──────────────────────────────────
        # Noise injection into preintegration covariance
        accel_var = 0.05 ** 2  # m/s² (from noise params)
        gyro_var = 0.005 ** 2  # rad/s
        Q_preint = np.zeros((9, 9))
        Q_preint[0:3, 0:3] = np.eye(3) * accel_var * dt ** 2 * 0.25  # position noise
        Q_preint[3:6, 3:6] = np.eye(3) * accel_var * dt               # velocity noise
        Q_preint[6:9, 6:9] = np.eye(3) * gyro_var * dt                # rotation noise

        # State transition for covariance
        A = np.eye(9)
        A[0:3, 3:6] = np.eye(3) * dt  # dp/dv
        A[3:6, 6:9] = -R_k @ skew_a * dt  # dv/dtheta
        A[6:9, 6:9] = np.eye(3) - skew_w * dt  # dtheta/dtheta

        preint.cov = A @ preint.cov @ A.T + Q_preint

        preint.dt_sum += dt
        preint.n_samples += 1

    def add_keyframe(self, timestamp: float, position: np.ndarray,
                     velocity: np.ndarray, quaternion: np.ndarray,
                     accel_bias: np.ndarray, gyro_bias: np.ndarray):
        """Add a new keyframe to the sliding window.

        Call this at keyframe rate (~10-20 Hz) from the VIO/SLAM pipeline.

        Args:
            timestamp: keyframe timestamp
            position, velocity, quaternion: ESKF state at keyframe time
            accel_bias, gyro_bias: current bias estimates
        """
        if not self._enabled:
            return

        # Rate limiting
        if (self._keyframes and
                timestamp - self._keyframes[-1].timestamp < self.MIN_KEYFRAME_INTERVAL):
            return

        # Attach preintegration to previous keyframe
        if self._keyframes and self._current_preint.n_samples > 0:
            self._keyframes[-1].preintegration = self._current_preint

        # Reset preintegration for next interval
        self._current_preint = IMUPreintegration()

        # Create new keyframe
        kf = Keyframe(
            timestamp=timestamp,
            position=position.copy(),
            velocity=velocity.copy(),
            quaternion=quaternion.copy(),
            accel_bias=accel_bias.copy(),
            gyro_bias=gyro_bias.copy(),
        )
        self._keyframes.append(kf)
        self._keyframe_count += 1

        # Marginalize if window is full
        if len(self._keyframes) > self.MAX_WINDOW_SIZE:
            self._marginalize_oldest()

        log.debug(f"Keyframe {self._keyframe_count} added at t={timestamp:.2f}s, "
                  f"window={len(self._keyframes)}")

    def _marginalize_oldest(self):
        """Marginalize the oldest keyframe using Schur complement.

        The information (Hessian and residual) from the marginalized keyframe
        is converted into a prior on the second-oldest keyframe.

        Schur complement:
            H_marg = H_rr - H_rm @ H_mm^{-1} @ H_mr
            b_marg = b_r - H_rm @ H_mm^{-1} @ b_m

        where m = marginalized states, r = remaining states.
        """
        if len(self._keyframes) < 2:
            return

        oldest = self._keyframes[0]
        preint = oldest.preintegration

        if preint is None or preint.n_samples == 0:
            # No preintegration data — just remove
            self._keyframes.pop(0)
            return

        # Build local Hessian for the IMU preintegration factor
        # connecting oldest (m) to second-oldest (r)
        # Factor dimension: 15 (9 from each keyframe pose + shared biases)
        # But for simplicity, we build a 9×9 Hessian on pose states only

        # Information from preintegration covariance
        try:
            info = np.linalg.inv(preint.cov + np.eye(9) * 1e-10)
        except np.linalg.LinAlgError:
            log.warning("Marginalization: singular preintegration covariance")
            self._keyframes.pop(0)
            return

        # Partition: H_mm (oldest pose), H_mr (cross), H_rr (second oldest)
        # For a binary factor: H = J.T @ info @ J
        # J_m = I (identity for oldest pose in residual)
        # J_r = -I (negative identity for second oldest)
        # H = [[info, -info], [-info, info]]

        H_mm = info
        H_mr = -info
        H_rm = -info
        H_rr = info

        # Residual computation
        second = self._keyframes[1]
        r_p = (second.position - oldest.position -
               oldest.velocity * preint.dt_sum) - preint.delta_p
        r_v = (second.velocity - oldest.velocity) - preint.delta_v
        # Simplified rotation residual (small angle approximation)
        dq = self._quat_mult(self._quat_inv(oldest.quaternion), second.quaternion)
        r_q = 2.0 * dq[1:4] - 2.0 * preint.delta_q[1:4]  # approximate

        residual = np.concatenate([r_p, r_v, r_q])

        b_m = info @ residual
        b_r = -info @ residual

        # Schur complement marginalization
        try:
            H_mm_inv = np.linalg.inv(H_mm + np.eye(9) * 1e-10)
        except np.linalg.LinAlgError:
            log.warning("Marginalization: singular H_mm")
            self._keyframes.pop(0)
            return

        H_marg = H_rr - H_rm @ H_mm_inv @ H_mr
        b_marg = b_r - H_rm @ H_mm_inv @ b_m

        # Store as prior for next optimization
        self._prior_H = H_marg
        self._prior_b = b_marg
        self._prior_dim = 9

        # Remove oldest keyframe
        self._keyframes.pop(0)
        self._marginalization_count += 1

        log.debug(f"Marginalized keyframe, window={len(self._keyframes)}, "
                  f"prior_dim={self._prior_dim}")

    def get_correction(self) -> Optional[dict]:
        """Get state correction from sliding window optimization.

        Runs a Gauss-Newton optimization over the window and returns
        the correction to the most recent keyframe's state.

        Returns:
            dict with position/velocity/attitude corrections, or None
        """
        if not self.is_active:
            return None

        if len(self._keyframes) < self.MIN_WINDOW_SIZE:
            return None

        # Build and solve the optimization problem
        correction = self._gauss_newton_optimize()
        if correction is not None:
            self._optimization_count += 1

        return correction

    def _gauss_newton_optimize(self) -> Optional[dict]:
        """Gauss-Newton optimization over the sliding window.

        Minimizes the sum of:
          - IMU preintegration residuals between consecutive keyframes
          - Marginalization prior (from previous marginalizations)

        Returns corrections to the most recent keyframe state.
        """
        n = len(self._keyframes)
        state_dim = 9 * n  # 9 per keyframe (dp, dv, dtheta)

        for iteration in range(self.MAX_ITERATIONS):
            H_total = np.zeros((state_dim, state_dim))
            b_total = np.zeros(state_dim)

            # ── IMU preintegration factors ─────────────────────
            for i in range(n - 1):
                kf_i = self._keyframes[i]
                kf_j = self._keyframes[i + 1]
                preint = kf_i.preintegration

                if preint is None or preint.n_samples == 0:
                    continue

                # Preintegration residual
                r_p = (kf_j.position - kf_i.position -
                       kf_i.velocity * preint.dt_sum) - preint.delta_p
                r_v = (kf_j.velocity - kf_i.velocity) - preint.delta_v
                dq = self._quat_mult(self._quat_inv(kf_i.quaternion),
                                     kf_j.quaternion)
                r_q = 2.0 * dq[1:4] - 2.0 * preint.delta_q[1:4]

                residual = np.concatenate([r_p, r_v, r_q])

                # Information matrix
                try:
                    info = np.linalg.inv(preint.cov + np.eye(9) * 1e-10)
                except np.linalg.LinAlgError:
                    continue

                # Jacobians (simplified for pose-only)
                J_i = -np.eye(9)  # d(residual)/d(state_i)
                J_j = np.eye(9)   # d(residual)/d(state_j)

                idx_i = i * 9
                idx_j = (i + 1) * 9

                # Hessian blocks
                H_total[idx_i:idx_i+9, idx_i:idx_i+9] += J_i.T @ info @ J_i
                H_total[idx_i:idx_i+9, idx_j:idx_j+9] += J_i.T @ info @ J_j
                H_total[idx_j:idx_j+9, idx_i:idx_i+9] += J_j.T @ info @ J_i
                H_total[idx_j:idx_j+9, idx_j:idx_j+9] += J_j.T @ info @ J_j

                # Gradient
                b_total[idx_i:idx_i+9] += J_i.T @ info @ residual
                b_total[idx_j:idx_j+9] += J_j.T @ info @ residual

            # ── Marginalization prior ──────────────────────────
            if self._prior_H is not None:
                pdim = self._prior_dim
                H_total[0:pdim, 0:pdim] += self._prior_H
                b_total[0:pdim] += self._prior_b

            # ── Damping (Levenberg-Marquardt) ──────────────────
            damping = 1e-4
            H_total += np.eye(state_dim) * damping

            # ── Solve ──────────────────────────────────────────
            try:
                dx = np.linalg.solve(H_total, -b_total)
            except np.linalg.LinAlgError:
                log.warning("Sliding window: singular Hessian")
                return None

            # Check convergence
            if np.linalg.norm(dx) < self.CONVERGENCE_TOL:
                break

            # Apply corrections to all keyframes
            for i in range(n):
                idx = i * 9
                self._keyframes[i].position += dx[idx:idx+3]
                self._keyframes[i].velocity += dx[idx+3:idx+6]

                dtheta = dx[idx+6:idx+9]
                dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
                dq /= np.linalg.norm(dq)
                self._keyframes[i].quaternion = self._quat_mult(
                    self._keyframes[i].quaternion, dq)
                self._keyframes[i].quaternion /= np.linalg.norm(
                    self._keyframes[i].quaternion)

        # Return correction for the most recent keyframe
        latest = self._keyframes[-1]
        return {
            "position": latest.position.copy(),
            "velocity": latest.velocity.copy(),
            "quaternion": latest.quaternion.copy(),
            "window_size": n,
            "iteration": iteration + 1,
        }

    # ── Quaternion utilities ───────────────────────────────────

    @staticmethod
    def _quat_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quat_inv(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_to_dcm(q):
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "window_size": len(self._keyframes),
            "keyframe_count": self._keyframe_count,
            "marginalization_count": self._marginalization_count,
            "optimization_count": self._optimization_count,
            "has_prior": self._prior_H is not None,
        }
