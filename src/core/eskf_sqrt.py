#!/usr/bin/env python3
# Square-Root ESKF.
# Because negative covariance is mathematically offensive.

import numpy as np
import math
import logging
from typing import Optional
from scipy.linalg import cho_factor, cho_solve, cholesky
from utils.noise import IMUNoiseParams
from core.eskf import (
    ESKF, NOMINAL_DIM, ERROR_DIM, EKFHealth,
    POS, VEL, QUAT, ABIAS, GBIAS, BARO_BIAS_IDX,
    CLK_BIAS_IDX, CLK_DRIFT_IDX, WIND,
    E_POS, E_VEL, E_ATT, E_ABIAS, E_GBIAS,
    E_BARO_BIAS, E_CLK_BIAS, E_CLK_DRIFT, E_WIND,
    CHI2_THRESHOLDS, GRAVITY_NED
)

log = logging.getLogger("eskf_sqrt")


class SquareRootESKF(ESKF):
    """Square-Root ESKF — Cholesky-factored covariance for guaranteed PD.

    Instead of maintaining P directly, we maintain its lower Cholesky
    factor S such that P = S @ S.T. All operations are reformulated
    to work on S directly:

    Predict:
        S_new = QR_lower([ F @ S, sqrt(Q*dt) ])

    Update:
        Uses Carlson's sequential processing or Potter's method
        for numerically stable square-root measurement updates.

    The P property is computed from S only when needed (for output/logging).
    """

    def __init__(self, noise: IMUNoiseParams):
        # Initialize the parent ESKF (which sets up P, Q, etc.)
        super().__init__(noise)

        # Convert P to Cholesky factor
        self.S = np.linalg.cholesky(self.P)  # P = S @ S.T (lower triangular)

        # Pre-compute sqrt(Q_base) for predict step
        self._sqrt_Q_base = self._safe_cholesky(self.Q_base)

        log.info("Square-Root ESKF initialized (Cholesky-factored covariance)")

    @staticmethod
    def _safe_cholesky(M: np.ndarray) -> np.ndarray:
        """Cholesky decomposition with regularization for near-singular matrices."""
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # Add small diagonal to make PD
            eps = 1e-10
            M_reg = M + np.eye(M.shape[0]) * eps
            return np.linalg.cholesky(M_reg)

    @property
    def P(self) -> np.ndarray:
        """Reconstruct P from Cholesky factor (for output/logging only)."""
        return self.S @ self.S.T

    @P.setter
    def P(self, value: np.ndarray):
        """When P is set (e.g., during init), update S accordingly."""
        self.S = self._safe_cholesky(value)

    def predict(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, dt: float):
        """Square-root predict using QR decomposition.

        Instead of:  P_new = F @ P @ F.T + Q * dt
        We compute:  S_new = triu(qr([ F@S, sqrt(Q*dt) ].T)).T

        This is the Householder QR approach to square-root filtering.
        """
        if dt <= 0:
            return

        # Bias compensation and RK4 state propagation (reuse parent logic)
        accel = accel_raw - self.x[ABIAS]
        gyro = gyro_raw - self.x[GBIAS]

        # RK4 integration (same as parent)
        y = np.zeros(10)
        y[0:3] = self.x[POS]
        y[3:6] = self.x[VEL]
        y[6:10] = self.x[QUAT]

        k1 = self._state_derivative(y, accel, gyro)
        k2 = self._state_derivative(y + 0.5 * dt * k1, accel, gyro)
        k3 = self._state_derivative(y + 0.5 * dt * k2, accel, gyro)
        k4 = self._state_derivative(y + dt * k3, accel, gyro)

        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.x[POS] = y_new[0:3]
        self.x[VEL] = y_new[3:6]
        self.x[QUAT] = y_new[6:10]
        self.x[QUAT] /= np.linalg.norm(self.x[QUAT])

        # Bias and state propagation
        tau_a = max(self.noise.accel_bias_tau, 1.0)
        tau_g = max(self.noise.gyro_bias_tau, 1.0)
        self.x[ABIAS] *= (1.0 - dt / tau_a)
        self.x[GBIAS] *= (1.0 - dt / tau_g)
        self.x[ABIAS] = np.clip(self.x[ABIAS],
                                -self.ACCEL_BIAS_LIMIT, self.ACCEL_BIAS_LIMIT)
        self.x[GBIAS] = np.clip(self.x[GBIAS],
                                -self.GYRO_BIAS_LIMIT, self.GYRO_BIAS_LIMIT)
        self.x[BARO_BIAS_IDX] = np.clip(self.x[BARO_BIAS_IDX],
                                        -self.BARO_BIAS_LIMIT, self.BARO_BIAS_LIMIT)
        self.x[CLK_BIAS_IDX] += self.x[CLK_DRIFT_IDX] * dt
        self.x[WIND] = np.clip(self.x[WIND], -self.WIND_LIMIT, self.WIND_LIMIT)

        # ── Square-root covariance propagation ─────────────────
        R_body = self._quat_to_dcm(self.x[QUAT])
        F = self._compute_F(accel, gyro, R_body, dt)

        # Scale process noise by vibration
        sqrt_Q = self._sqrt_Q_base * math.sqrt(self._vibration_scale * dt)

        # Concatenated matrix: [F@S | sqrt_Q] — both are (ERROR_DIM × ERROR_DIM)
        # Then QR decompose to get new S
        FS = F @ self.S
        compound = np.hstack([FS, sqrt_Q])  # (ERROR_DIM × 2*ERROR_DIM)

        # QR decomposition: compound.T = Q_qr @ R_qr
        # S_new = R_qr[0:ERROR_DIM, :].T (lower triangular)
        try:
            _, R_qr = np.linalg.qr(compound.T, mode='reduced')
            # R_qr is upper triangular; we want lower triangular S
            self.S = R_qr[:ERROR_DIM, :ERROR_DIM].T
            # Ensure positive diagonal (convention)
            for i in range(ERROR_DIM):
                if self.S[i, i] < 0:
                    self.S[i, :] = -self.S[i, :]
        except np.linalg.LinAlgError:
            log.warning("SR-ESKF: QR failed in predict, falling back to standard")
            P_fallback = F @ (self.S @ self.S.T) @ F.T + self.Q * dt
            P_fallback = (P_fallback + P_fallback.T) / 2.0
            self.S = self._safe_cholesky(P_fallback)

        self._step_count += 1
        self._check_health()

    def update_external(self, z: np.ndarray, z_pred: np.ndarray,
                        H: np.ndarray, R: np.ndarray,
                        source: str = "external") -> bool:
        """Square-root measurement update using Potter's method.

        Processes each measurement component sequentially (rank-1 updates)
        to maintain the Cholesky factor without explicitly forming P.

        For a scalar measurement z_i:
            f = S.T @ h_i  (where h_i is a column of H.T)
            alpha = f.T @ f + R_ii
            K = S @ f / alpha
            S_new = S - K @ f.T / (1 + sqrt(R_ii/alpha))
        """
        if not self._initialized:
            return False

        m = z.shape[0]
        y = z - z_pred

        # Wrap angles if yaw observation
        if m == 1 and H.shape[1] == ERROR_DIM and H[0, 8] != 0.0:
            y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))

        # Innovation gating (using full P for gating check)
        P_check = self.S @ self.S.T
        S_innov = H @ P_check @ H.T + R
        try:
            S_inv = np.linalg.inv(S_innov)
        except np.linalg.LinAlgError:
            log.warning(f"{source}: singular innovation covariance")
            return False

        nis = float(y @ S_inv @ y)
        chi2_thresh = CHI2_THRESHOLDS.get(m, 3.0 * m)
        if nis > chi2_thresh:
            log.debug(f"{source} rejected: NIS={nis:.2f} > {chi2_thresh}")
            return False

        # Sequential scalar measurement processing (Potter's method)
        S_work = self.S.copy()

        for i in range(m):
            h_i = H[i, :]  # (ERROR_DIM,)
            r_i = R[i, i]  # scalar variance

            # f = S.T @ h_i
            f = S_work.T @ h_i  # (ERROR_DIM,)

            # alpha = f.T @ f + r_i
            alpha = np.dot(f, f) + r_i

            if alpha < 1e-15:
                continue

            # Kalman gain (in square-root form)
            K = S_work @ f / alpha

            # Update state
            self.x[POS] += K[E_POS] * y[i]
            self.x[VEL] += K[E_VEL] * y[i]

            # Attitude correction
            dtheta = K[E_ATT] * y[i]
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            self.x[QUAT] = self._quat_mult(self.x[QUAT], dq)
            self.x[QUAT] /= np.linalg.norm(self.x[QUAT])

            # Bias and extended state corrections
            self.x[ABIAS] += K[E_ABIAS] * y[i]
            self.x[GBIAS] += K[E_GBIAS] * y[i]
            self.x[BARO_BIAS_IDX] += K[E_BARO_BIAS] * y[i]
            self.x[CLK_BIAS_IDX] += K[E_CLK_BIAS] * y[i]
            self.x[CLK_DRIFT_IDX] += K[E_CLK_DRIFT] * y[i]
            self.x[WIND] += K[E_WIND] * y[i]

            # Square-root covariance update (rank-1 downdate)
            beta = 1.0 / (1.0 + math.sqrt(r_i / alpha))
            S_work = S_work - beta * np.outer(K, f)

        # Apply clamping
        self.x[ABIAS] = np.clip(self.x[ABIAS],
                                -self.ACCEL_BIAS_LIMIT, self.ACCEL_BIAS_LIMIT)
        self.x[GBIAS] = np.clip(self.x[GBIAS],
                                -self.GYRO_BIAS_LIMIT, self.GYRO_BIAS_LIMIT)
        self.x[BARO_BIAS_IDX] = np.clip(self.x[BARO_BIAS_IDX],
                                        -self.BARO_BIAS_LIMIT, self.BARO_BIAS_LIMIT)
        self.x[WIND] = np.clip(self.x[WIND], -self.WIND_LIMIT, self.WIND_LIMIT)

        self.S = S_work
        return True

    def _harden_covariance(self):
        """No-op: square-root form is inherently PD.

        This is the whole point — no eigenvalue clamping needed.
        """
        pass

    def scale_process_noise(self, vibration_level: float):
        """Scale Q and recompute sqrt(Q)."""
        scale = max(1.0, min(1.0 + vibration_level * 5.0, 10.0))
        self._vibration_scale = scale
        self.Q = self.Q_base * scale
        self._sqrt_Q_base = self._safe_cholesky(self.Q_base * scale)

    def reset(self):
        """Factory reset."""
        super().reset()
        self.S = self._safe_cholesky(self.P)
        self._sqrt_Q_base = self._safe_cholesky(self.Q_base)
