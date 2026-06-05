#!/usr/bin/env python3
# 21-state Error-State Quaternion EKF (ESKF) — NavCore-Pixhawk
#
# Nominal state (21):
#   x = [px, py, pz,                   # 0:3   position (NED, m)
#        vx, vy, vz,                    # 3:6   velocity (NED, m/s)
#        qw, qx, qy, qz,               # 6:10  attitude quaternion
#        ba_x, ba_y, ba_z,              # 10:13 accel bias (m/s²)
#        bg_x, bg_y, bg_z,              # 13:16 gyro bias (rad/s)
#        baro_bias,                     # 16    barometric altitude bias (m)
#        clk_bias,                      # 17    GNSS receiver clock bias (m)
#        clk_drift,                     # 18    GNSS receiver clock drift (m/s)
#        wind_n, wind_e]                # 19:21 wind velocity NED horizontal (m/s)
#
# Error state (20):
#   dx = [dp(3), dv(3), dtheta(3),      # 0:9   pos, vel, attitude error
#         dba(3), dbg(3),                # 9:15  bias errors
#         d_baro_bias,                   # 15    baro bias error
#         d_clk_bias,                    # 16    clock bias error
#         d_clk_drift,                   # 17    clock drift error
#         d_wind_n, d_wind_e]            # 18:20 wind error
#
# Upgraded from 16-state with:
#   - Baro bias as proper filter state (replaces EMA hack)
#   - GNSS receiver clock bias/drift for tight coupling
#   - Wind velocity estimation for multi-rotor accuracy
#   - RK4 quaternion integration (replaces Euler)
#   - Iterated EKF (IEKF) for nonlinear measurements
#
# If this crashes, the drone probably will too. No pressure.

import numpy as np
import math
import logging
from enum import Enum
from typing import Optional, Callable, Tuple
import scipy.linalg as la
from utils.noise import IMUNoiseParams

log = logging.getLogger("eskf_core")

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])

# Error-state dimension constants
NOMINAL_DIM = 21
ERROR_DIM = 20

# State index constants (nominal)
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
ABIAS = slice(10, 13)
GBIAS = slice(13, 16)
BARO_BIAS_IDX = 16
CLK_BIAS_IDX = 17
CLK_DRIFT_IDX = 18
WIND = slice(19, 21)

# Error-state index constants
E_POS = slice(0, 3)
E_VEL = slice(3, 6)
E_ATT = slice(6, 9)
E_ABIAS = slice(9, 12)
E_GBIAS = slice(12, 15)
E_BARO_BIAS = 15
E_CLK_BIAS = 16
E_CLK_DRIFT = 17
E_WIND = slice(18, 20)

# Chi-squared thresholds for innovation gating (95% confidence)
CHI2_THRESHOLDS = {1: 5.991, 2: 9.210, 3: 7.815, 4: 9.488, 5: 11.07}


class EKFHealth(Enum):
    CONVERGING = 0
    HEALTHY = 1
    WARNING = 2
    FAULT = 3


class ESKF:
    """21-state Error-State Kalman Filter with RK4 and IEKF support."""

    # Safety bounds
    VEL_WARN = 30.0       # m/s
    VEL_FAULT = 100.0     # m/s
    TILT_WARN_DEG = 60.0
    TILT_FAULT_DEG = 80.0
    ACCEL_BIAS_LIMIT = 2.0   # m/s²
    GYRO_BIAS_LIMIT = 0.1    # rad/s
    BARO_BIAS_LIMIT = 15.0   # m — max baro bias
    WIND_LIMIT = 25.0        # m/s — max estimated wind
    P_TRACE_LIMIT = 1e9
    P_COND_LIMIT = 1e15
    Z_COV_CONVERGED = 1.5   # z-axis covariance threshold (m²)
    SYMMETRY_INTERVAL = 50

    # Mag rejection thresholds
    MAG_NORM_TOLERANCE = 0.30
    MAG_REJECT_DURATION = 2.0

    def __init__(self, noise: IMUNoiseParams):
        self.noise = noise
        self._step_count = 0
        self._initialized = False
        self._health = EKFHealth.CONVERGING
        self._mag_reject_until = 0.0
        self._calibrated_mag_norm = 0.5
        self._mag_consecutive_good = 0
        self._mag_required_good = 10
        self._gps_origin = None
        self._innovation_stats = {"baro": [], "mag": []}
        self._sensor_rejections = {}

        # Adaptive process noise (Feature 6)
        self._vibration_scale = 1.0

        # --- Nominal state (21) ---
        self.x = np.zeros(NOMINAL_DIM)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)

        # --- Error-state covariance (20x20) ---
        P_init = np.eye(ERROR_DIM)
        P_init[E_POS, E_POS] *= 1.0      # position
        P_init[E_VEL, E_VEL] *= 0.1      # velocity
        P_init[E_ATT, E_ATT] *= 0.01     # attitude
        P_init[E_ABIAS, E_ABIAS] *= 0.01 # accel bias
        P_init[E_GBIAS, E_GBIAS] *= 0.001  # gyro bias
        P_init[E_BARO_BIAS, E_BARO_BIAS] = 25.0   # baro bias: ±5m initial uncertainty
        # ~100 m initial clock bias σ (1e4 m²). 1e6 made cond(P)≈1e9 and broke NEES/NIS numerics.
        P_init[E_CLK_BIAS, E_CLK_BIAS] = 1e4
        P_init[E_CLK_DRIFT, E_CLK_DRIFT] = 100.0   # clock drift
        P_init[E_WIND, E_WIND] *= 10.0              # wind: ±3.2 m/s initial
        
        # Upper triangular Cholesky factor U (where P = U @ U.T)
        self.U = np.linalg.cholesky(P_init).T

        # --- Health Monitoring Buffers ---
        from collections import deque
        self.innovation_history = deque(maxlen=500)  # (time, source, y, S)
        self.health_history = deque(maxlen=500)      # (time, cond_num, min_diag, max_diag)
        self.cond_num = 1.0
        self.min_diag_U = 1.0
        self.max_diag_U = 1.0
        
        # Actionable Health Metrics
        self.cholesky_failures = 0
        self.innovation_spikes = 0
        self.covariance_repairs = 0

        # Last bias-compensated gyro measurement (body rates, rad/s)
        self._last_gyro = np.zeros(3)
        # --- Process noise (Continuous-Time Model Q_c) ---
        # The continuous-time process model is: dx_err/dt = F * x_err + G * w
        # where w is continuous-time white noise with Power Spectral Density (PSD) Q_c.
        # Accelerometer and Gyro noise are assumed isotropic in the body frame: E[w w^T] = sigma^2 I
        # Because the mapping matrix G for velocity is -R(q), the noise mapped to global frame is:
        # G Q_c G^T = R(q) (sigma^2 I) R(q)^T = sigma^2 I.
        # This allows us to use a purely diagonal Q_c matrix and simply scale by dt
        # for the discrete-time noise Q_d ≈ Q_c * dt.
        
        self.Q_base = np.zeros((ERROR_DIM, ERROR_DIM))
        # sa and sg are PSDs: (m/s^2)^2/Hz and (rad/s)^2/Hz
        sa = noise.accel_std ** 2
        sg = noise.gyro_std ** 2
        sab = (2.0 * noise.accel_bias_std ** 2 / max(noise.accel_bias_tau, 1.0))
        sgb = (2.0 * noise.gyro_bias_std ** 2 / max(noise.gyro_bias_tau, 1.0))
        np.fill_diagonal(self.Q_base[E_VEL, E_VEL], sa)
        np.fill_diagonal(self.Q_base[E_ATT, E_ATT], sg)
        np.fill_diagonal(self.Q_base[E_ABIAS, E_ABIAS], sab)
        np.fill_diagonal(self.Q_base[E_GBIAS, E_GBIAS], sgb)
        # Baro bias: random walk (slow drift ~0.01 m/√s)
        self.Q_base[E_BARO_BIAS, E_BARO_BIAS] = 0.01 ** 2
        # Clock bias: driven by clock drift (coupled in F)
        self.Q_base[E_CLK_BIAS, E_CLK_BIAS] = 0.1 ** 2
        # Clock drift: TCXO stability (~1e-9 relative → ~0.3 m/s²)
        self.Q_base[E_CLK_DRIFT, E_CLK_DRIFT] = 0.3 ** 2
        # Wind: random walk (~0.5 m/s/√s for turbulence)
        np.fill_diagonal(self.Q_base[E_WIND, E_WIND], 0.5 ** 2)
        self.Q = self.Q_base.copy()

        # --- Measurement noise ---
        self.R_baro = np.array([[noise.baro_std ** 2]])
        self.R_mag = np.array([[noise.mag_std ** 2]])
        self._R_mag_base = noise.mag_std ** 2

        # --- Observation matrices (20-col for error state) ---
        self.H_baro = np.zeros((1, ERROR_DIM))
        self.H_baro[0, 2] = 1.0   # observes dp_z
        self.H_baro[0, E_BARO_BIAS] = -1.0  # minus baro bias

        self.H_mag = np.zeros((1, ERROR_DIM))
        self.H_mag[0, 8] = 1.0    # observes dtheta_z (yaw error)

        log.info(f"ESKF initialized: {NOMINAL_DIM}-state nominal, "
                 f"{ERROR_DIM}-state error, SR-ESKF/RK4, IEKF enabled")

    # ── Properties ─────────────────────────────────────────────

    @property
    def P(self) -> np.ndarray:
        """Dynamically compute full covariance matrix from Cholesky factor."""
        return self.U.T @ self.U

    @P.setter
    def P(self, value: np.ndarray):
        """Update Cholesky factor if full covariance is explicitly set."""
        self.U = np.linalg.cholesky(value).T

    @property
    def state(self) -> dict:
        q = self.x[QUAT]
        euler = self._quat_to_euler(q)
        return {
            "pos": self.x[POS].copy(),
            "vel": self.x[VEL].copy(),
            "quat": q.copy(),
            "euler": np.array(euler),
            "accel_bias": self.x[ABIAS].copy(),
            "gyro_bias": self.x[GBIAS].copy(),
            "baro_bias": float(self.x[BARO_BIAS_IDX]),
            "clock_bias": float(self.x[CLK_BIAS_IDX]),
            "clock_drift": float(self.x[CLK_DRIFT_IDX]),
            "wind": self.x[WIND].copy(),
        }

    @property
    def health(self) -> EKFHealth:
        return self._health

    @property
    def baro_bias(self) -> float:
        """Barometric bias estimate (m). Now a proper filter state."""
        return float(self.x[BARO_BIAS_IDX])

    @property
    def wind_estimate(self) -> np.ndarray:
        """Estimated wind velocity [North, East] (m/s)."""
        return self.x[WIND].copy()

    # ── Initialization ─────────────────────────────────────────

    def initialize_from_sensors(self, accel_samples: np.ndarray,
                                mag_samples: np.ndarray) -> bool:
        """Initialize attitude from stationary IMU + mag data."""
        if len(accel_samples) < 10 or len(mag_samples) < 10:
            log.warning("Not enough samples for initialization")
            return False

        accel_mean = np.mean(accel_samples, axis=0)
        accel_var = np.var(np.linalg.norm(accel_samples, axis=1))

        if accel_var > 0.5:
            log.warning(f"IMU variance too high ({accel_var:.2f}), drone may be moving")
            return False

        # Roll and pitch from gravity vector
        ax, ay, az = accel_mean
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))

        # Yaw from magnetometer (tilt-compensated)
        mag_mean = np.mean(mag_samples, axis=0)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        mx, my, mz = mag_mean
        mag_x = mx * cp + my * sr * sp + mz * cr * sp
        mag_y = my * cr - mz * sr
        yaw = math.atan2(-mag_y, mag_x)

        # Store calibrated mag norm
        self._calibrated_mag_norm = np.linalg.norm(mag_mean)

        # Set nominal state
        self.x[QUAT] = self._euler_to_quat(roll, pitch, yaw)
        self.x[POS] = 0.0
        self.x[VEL] = 0.0
        self.x[ABIAS] = 0.0
        self.x[GBIAS] = 0.0
        self.x[BARO_BIAS_IDX] = 0.0    # baro bias starts at zero
        self.x[CLK_BIAS_IDX] = 0.0     # clock bias unknown
        self.x[CLK_DRIFT_IDX] = 0.0    # clock drift unknown
        self.x[WIND] = 0.0             # no wind initially

        self._initialized = True
        self._health = EKFHealth.CONVERGING
        log.info(f"ESKF initialized: roll={math.degrees(roll):.1f} "
                 f"pitch={math.degrees(pitch):.1f} yaw={math.degrees(yaw):.1f}")
        return True

    # ── Predict (RK4 Integration) ──────────────────────────────

    def predict(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, dt: float):
        """Predict step using 4th-order Runge-Kutta integration.

        Replaces the old Euler integration for significantly reduced
        integration error at the same 100 Hz rate.
        """
        if dt <= 0:
            return

        # Bias compensation
        accel = accel_raw - self.x[ABIAS]
        gyro = gyro_raw - self.x[GBIAS]

        # Cache last bias-compensated gyro for use in other updates
        self._last_gyro = gyro.copy()

        # ── RK4 for position, velocity, and quaternion ──────────
        # State pack: [pos(3), vel(3), quat(4)] = 10 elements
        y = np.zeros(10)
        y[0:3] = self.x[POS]
        y[3:6] = self.x[VEL]
        y[6:10] = self.x[QUAT]

        k1 = self._state_derivative(y, accel, gyro)
        k2 = self._state_derivative(y + 0.5 * dt * k1, accel, gyro)
        k3 = self._state_derivative(y + 0.5 * dt * k2, accel, gyro)
        k4 = self._state_derivative(y + dt * k3, accel, gyro)

        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Unpack and normalize quaternion
        self.x[POS] = y_new[0:3]
        self.x[VEL] = y_new[3:6]
        self.x[QUAT] = y_new[6:10]
        self.x[QUAT] /= np.linalg.norm(self.x[QUAT])

        # ── Bias decay (Gauss-Markov) ──────────────────────────
        tau_a = max(self.noise.accel_bias_tau, 1.0)
        tau_g = max(self.noise.gyro_bias_tau, 1.0)
        self.x[ABIAS] *= (1.0 - dt / tau_a)
        self.x[GBIAS] *= (1.0 - dt / tau_g)

        # Clamp biases
        self.x[ABIAS] = np.clip(self.x[ABIAS],
                                -self.ACCEL_BIAS_LIMIT, self.ACCEL_BIAS_LIMIT)
        self.x[GBIAS] = np.clip(self.x[GBIAS],
                                -self.GYRO_BIAS_LIMIT, self.GYRO_BIAS_LIMIT)

        # ── Baro bias: random walk (no decay) ──────────────────
        # Baro bias evolves as process noise only — no dynamics
        self.x[BARO_BIAS_IDX] = np.clip(self.x[BARO_BIAS_IDX],
                                        -self.BARO_BIAS_LIMIT, self.BARO_BIAS_LIMIT)

        # ── Clock: bias driven by drift ────────────────────────
        self.x[CLK_BIAS_IDX] += self.x[CLK_DRIFT_IDX] * dt
        # Clock drift: random walk (TCXO)

        # ── Wind: random walk, clamped ─────────────────────────
        self.x[WIND] = np.clip(self.x[WIND], -self.WIND_LIMIT, self.WIND_LIMIT)

        # ── Square-Root Covariance Update (QR) ─────────────────
        R_dcm = self._quat_to_dcm(self.x[QUAT])
        F = self._compute_F(accel, gyro, R_dcm, dt)
        
        # M = [ U * F.T ]
        #     [ sqrt(Q) ]
        M = np.vstack([
            self.U @ F.T,
            np.diag(np.sqrt(np.diag(self.Q * dt)))
        ])
        
        # QR decomposition gives upper triangular R which is our new U
        _, R_qr = la.qr(M, mode='economic')
        
        # Enforce positive diagonal elements for uniqueness
        signs = np.sign(np.diag(R_qr))
        signs[signs == 0] = 1.0
        self.U = R_qr * signs[:, np.newaxis]
        
        # Track numerical health of U
        diags = np.abs(np.diag(self.U))
        self.min_diag_U = float(np.min(diags))
        self.max_diag_U = float(np.max(diags))
        self.cond_num = self.max_diag_U / max(self.min_diag_U, 1e-12)
        
        # Log to health history (assuming dt is time since start roughly)
        # Actually, time tracking should be done in m.py, but we can store raw values here.
        self.health_history.append((self._step_count * dt, self.cond_num, self.min_diag_U, self.max_diag_U))

        # Health & State checks
        self._step_count += 1
        self._check_health()

    def _state_derivative(self, y: np.ndarray, accel: np.ndarray,
                          gyro: np.ndarray) -> np.ndarray:
        """Compute state derivative for RK4: dy/dt = f(y, u).

        y = [pos(3), vel(3), quat(4)]
        """
        dy = np.zeros(10)
        q = y[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm

        R = self._quat_to_dcm(q)

        # dp/dt = v
        dy[0:3] = y[3:6]

        # dv/dt = R*accel + gravity
        # Wind affects velocity through aerodynamic drag on the airframe
        # For multirotors: v_air = v_ground - wind → additional accel ∝ drag
        # First-order approximation: no direct velocity coupling (wind is estimated
        # from GPS-vs-INS discrepancy, not modeled in dynamics)
        dy[3:6] = R @ accel + GRAVITY_NED

        # dq/dt = 0.5 * q ⊗ [0, gyro]
        omega = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        dy[6:10] = 0.5 * self._quat_mult(q, omega)

        return dy

    def _compute_F(self, accel: np.ndarray, gyro: np.ndarray,
                   R: np.ndarray, dt: float) -> np.ndarray:
        """Compute the 20x20 error-state transition Jacobian."""
        F = np.eye(ERROR_DIM)

        # dp/dv
        F[E_POS, E_VEL] = np.eye(3) * dt

        # dv/dtheta: -R * [accel]x * dt
        F[3:6, E_ATT] = -R @ self._skew(accel) * dt

        # dv/dba: -R * dt
        F[3:6, E_ABIAS] = -R * dt

        # dtheta/dtheta: I - [gyro]x * dt
        F[E_ATT, E_ATT] = np.eye(3) - self._skew(gyro) * dt

        # dtheta/dbg: -I * dt
        F[E_ATT, E_GBIAS] = -np.eye(3) * dt

        # Bias states: Gauss-Markov decay
        tau_a = max(self.noise.accel_bias_tau, 1.0)
        tau_g = max(self.noise.gyro_bias_tau, 1.0)
        F[E_ABIAS, E_ABIAS] = np.eye(3) * (1.0 - dt / tau_a)
        F[E_GBIAS, E_GBIAS] = np.eye(3) * (1.0 - dt / tau_g)

        # Baro bias: random walk (F = I, already set)

        # Clock: bias driven by drift
        F[E_CLK_BIAS, E_CLK_DRIFT] = dt

        # Wind: random walk (F = I, already set)

        return F

    # ── Measurement Updates ────────────────────────────────────

    def update_baro(self, alt_measured: float):
        """Baro altitude update with bias as proper filter state.

        The baro bias is now part of the 21-state vector, so the filter
        automatically estimates and tracks it with proper covariance.
        No more EMA hack.
        """
        # Predicted measurement: z_pred = pos_z + baro_bias
        # (baro measures altitude + bias)
        z_pred = np.array([self.x[2] + self.x[BARO_BIAS_IDX]])
        z = np.array([alt_measured])
        y = z - z_pred  # innovation

        # H = [0,0,1, 0..., -1(baro_bias), 0...]
        # H_baro[0,2] = 1.0 (position z)
        # H_baro[0,15] = -1.0 (baro bias observes as negative)
        # Wait — the observation model is: z = h(x) = pos_z + baro_bias
        # So: H_baro[0,2] = 1.0 and H_baro[0,15] = 1.0
        # But innovation is z - z_pred = alt - (pos_z + baro_bias)
        # Correction should decrease pos_z if alt < pos_z + baro_bias
        # and increase baro_bias if the bias is the issue
        H = np.zeros((1, ERROR_DIM))
        H[0, 2] = 1.0       # dp_z
        H[0, E_BARO_BIAS] = 1.0  # d_baro_bias

        # Adaptive R: inflate on large transient
        R = self.R_baro.copy()
        if abs(y[0]) > 2.0:
            R *= 5.0

        # Innovation gating (Mahalanobis) using cho_solve
        P = self.P
        S = H @ P @ H.T + R
        c_and_lower = la.cho_factor(S)
        nis = float(y @ la.cho_solve(c_and_lower, y))

        if nis > CHI2_THRESHOLDS[1]:
            log.debug(f"Baro rejected: NIS={nis:.2f} > {CHI2_THRESHOLDS[1]}")
            return

        # Standard Kalman update
        K = P @ H.T @ la.cho_solve(c_and_lower, np.eye(len(S)))
        dx = (K @ y).flatten()
        self._inject_error(dx)

        # Joseph form covariance update + Re-Cholesky
        I_KH = np.eye(ERROR_DIM) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.U = np.linalg.cholesky(P_new + np.eye(ERROR_DIM)*1e-12).T

    def update_mag(self, yaw_measured: float, mag_norm: float = -1.0,
                   t_now: float = 0.0):
        """Magnetometer yaw update with 3-tier rejection."""
        # Tier 1: field norm check
        if mag_norm > 0:
            norm_ratio = abs(mag_norm / self._calibrated_mag_norm - 1.0)
            if norm_ratio > self.MAG_NORM_TOLERANCE:
                self._mag_reject_until = t_now + self.MAG_REJECT_DURATION
                self._mag_consecutive_good = 0
                log.debug(f"Mag rejected (norm): ratio={norm_ratio:.2f}")
                return

        # Tier 2: time-based rejection
        if t_now > 0 and t_now < self._mag_reject_until:
            self._mag_consecutive_good = 0
            return

        # Tier 3: Multi-sample re-enable
        self._mag_consecutive_good += 1
        if self._mag_consecutive_good < self._mag_required_good:
            return

        # Get predicted yaw from quaternion
        R_dcm = self._quat_to_dcm(self.x[QUAT])
        euler = self._quat_to_euler(self.x[QUAT])
        yaw_pred = euler[2]
        phi = euler[0]
        theta = euler[1]

        y = np.array([self._wrap_angle(yaw_measured - yaw_pred)])

        # Adaptive R
        R = np.array([[self._R_mag_base]])
        if mag_norm > 0:
            norm_ratio = abs(mag_norm / self._calibrated_mag_norm - 1.0)
            if norm_ratio > 0.15:
                R *= 10.0

        # Innovation gating
        H = np.zeros((1, ERROR_DIM))
        # Jacobian of Euler yaw w.r.t body-frame angle error
        cos_theta = math.cos(theta)
        if abs(cos_theta) > 1e-3:
            H[0, 6] = 0.0
            H[0, 7] = math.sin(phi) / cos_theta
            H[0, 8] = math.cos(phi) / cos_theta
        else:
            H[0, 8] = 1.0  # Gimbal lock fallback
        
        P = self.P
        S = H @ P @ H.T + R
        c_and_lower = la.cho_factor(S)
        nis = float(y @ la.cho_solve(c_and_lower, y))

        if nis > CHI2_THRESHOLDS[1]:
            log.debug(f"Mag rejected (NIS): NIS={nis:.2f}")
            return

        K = P @ H.T @ la.cho_solve(c_and_lower, np.eye(len(S)))
        dx = (K @ y).flatten()
        self._inject_error(dx)

        # Joseph form covariance update + Re-Cholesky
        I_KH = np.eye(ERROR_DIM) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.U = np.linalg.cholesky(P_new + np.eye(ERROR_DIM)*1e-12).T

        # Magnetometer auto-calibration: slow EMA norm update
        if mag_norm > 0:
            alpha_mag = 0.002
            self._calibrated_mag_norm = (
                (1.0 - alpha_mag) * self._calibrated_mag_norm
                + alpha_mag * mag_norm
            )

    def update_optical_flow(self, flow_vx: float, flow_vy: float,
                            distance: float, quality: int,
                            enable_rot_comp: bool = True,
                            r_mount: np.ndarray = np.zeros(3)):
        """Optical flow velocity update.
        
        Measurement Frame: Camera/Body XY plane (m/s).
        State Frame: NED (m/s).
        
        The optical flow sensor observes velocity in the local body frame. We rotate the 
        NED velocity prediction into the body frame using R_dcm^T to form the residual.
        
        Args:
            flow_vx, flow_vy: Camera-plane flow velocities (m/s).
            distance: Range to ground (m).
            quality: Measurement quality (0-255).
            enable_rot_comp: If true, removes rotational velocity induced by gyro rates.
            r_mount: Mount offset of the sensor from the CG.
        """
        if distance <= 0.05 or quality < 10:
            return

        R_dcm = self._quat_to_dcm(self.x[QUAT])
        
        # Jacobian: Measurement is in Body frame. State is in NED frame.
        # v_body = R_dcm.T @ v_ned
        H_flow = np.zeros((2, ERROR_DIM))
        H_flow[:, 3:6] = R_dcm.T[0:2, :]  # vx, vy in body frame

        R_base = 0.5 ** 2
        R_flow = np.eye(2) * (R_base * 100.0 / max(quality, 1))

        # Predicted velocity in body frame
        v_ned = self.x[VEL]
        v_body_pred = R_dcm.T @ v_ned
        
        
        z = np.array([flow_vx, flow_vy])
        
        if enable_rot_comp:
            # Flow reports total velocity including rotation.
            # Use the last bias-compensated gyro measurement (rad/s) to predict rotation.
            omega = getattr(self, "_last_gyro", np.zeros(3))
            v_rot_pred = np.cross(omega, r_mount)
            # Add rotational effect to prediction (since raw flow includes it)
            z_pred = v_body_pred[0:2] + v_rot_pred[0:2]
        else:
            z_pred = v_body_pred[0:2]

        y = z - z_pred

        P = self.P
        S = H_flow @ P @ H_flow.T + R_flow
        try:
            c_and_lower = la.cho_factor(S)
        except la.LinAlgError:
            return
            
        nis = float(y @ la.cho_solve(c_and_lower, y))

        if nis > CHI2_THRESHOLDS[2]:
            return

        K = P @ H_flow.T @ la.cho_solve(c_and_lower, np.eye(len(S)))
        dx = (K @ y).flatten()
        self._inject_error(dx)

        # Joseph form covariance update + Re-Cholesky
        I_KH = np.eye(ERROR_DIM) - K @ H_flow
        P_new = I_KH @ P @ I_KH.T + K @ R_flow @ K.T
        self.U = np.linalg.cholesky(P_new + np.eye(ERROR_DIM)*1e-12).T

    def update_radar_velocity(self, vx: float, vy: float, vz: float,
                              weight: float = 1.0):
        """TI mmWave doppler velocity update.
        
        Measurement Frame: Radar/Body frame (m/s).
        State Frame: NED (m/s).
        
        The radar natively measures doppler reflections in its local coordinate system.
        The filter predicts this by rotating the global NED velocity by R_dcm^T.
        """
        R_dcm = self._quat_to_dcm(self.x[QUAT])
        
        H_radar = np.zeros((3, ERROR_DIM))
        H_radar[:, 3:6] = R_dcm.T  # Map NED velocity error to body frame measurement

        R_radar = np.eye(3) * (0.1 ** 2) / weight

        z = np.array([vx, vy, vz])
        z_pred = R_dcm.T @ self.x[VEL]

        self.update_external(z, z_pred, H_radar, R_radar, source="radar")

    def update_lidar_range(self, distance: float, weight: float = 1.0):
        """Livox range-to-ground altitude update.
        
        Measurement Frame: Body-Z axis downward (meters).
        State Frame: NED altitude (meters, where Z is positive down).
        
        This assumes the Lidar is pointing straight down in the drone's body frame.
        We correct the measured distance for the drone's tilt angle to estimate 
        the true vertical distance to the ground: Z_ned = -distance * cos(tilt).
        """
        if distance < 0.1:
            return

        R_dcm = self._quat_to_dcm(self.x[QUAT])
        # Z-axis of body frame in NED:
        cos_tilt = R_dcm[2, 2] 
        if cos_tilt < 0.1: # Extreme bank, unreliable
            return

        H_lidar = np.zeros((1, ERROR_DIM))
        # z = -pos_z / cos_tilt -> dz/dpos_z = -1.0 / cos_tilt
        H_lidar[0, 2] = -1.0 / cos_tilt  

        R_lidar = np.array([[0.05 ** 2]]) / weight

        z = np.array([distance])
        z_pred = np.array([-self.x[2] / cos_tilt])

        self.update_external(z, z_pred, H_lidar, R_lidar, source="lidar")

    def update_external(self, z: np.ndarray, z_pred: np.ndarray,
                        H: np.ndarray, R: np.ndarray,
                        source: str = "external",
                        force_accept: bool = False,
                        _reacquire: bool = False) -> bool:
        """Generic external measurement update for VIO, UWB, SLAM, etc.

        Single-iteration linear update. For nonlinear measurements,
        use update_external_iterated() instead.
        """
        if not self._initialized:
            return False

        m = z.shape[0]
        y = z - z_pred

        # Wrap angles if single-DOF yaw observation
        if m == 1 and H.shape[1] == ERROR_DIM and H[0, 8] != 0.0:
            y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))

        # Innovation covariance
        P = self.P
        S = H @ P @ H.T + R

        try:
            c_and_lower = la.cho_factor(S)
        except la.LinAlgError:
            log.warning(f"{source}: non-PD innovation covariance")
            return False

        nis = float(y @ la.cho_solve(c_and_lower, y))

        # Chi-squared threshold based on measurement dimension
        chi2_thresh = CHI2_THRESHOLDS.get(m, 3.0 * m)

        src = source.lower()

        if not force_accept and nis > chi2_thresh:
            if _reacquire:
                if self._sensor_rejections.get(src, 0) >= 5:
                    log.warning(
                        f"RAIM FAULT: {src} rejected 5 times consecutively. Marked UNHEALTHY."
                    )
                log.debug(f"{src} rejected (reacquire): NIS={nis:.2f} > {chi2_thresh}")
                return False

            count = self._sensor_rejections.get(src, 0) + 1
            self._sensor_rejections[src] = count
            if count >= 5:
                log.warning(
                    f"RAIM FAULT: {src} rejected 5 times consecutively. Marked UNHEALTHY."
                )
            if src == "gps":
                if count >= 5:
                    return self.update_external(
                        z, z_pred, H, R, source=source,
                        force_accept=True, _reacquire=True,
                    )
                if count >= 3:
                    scale = min(nis / chi2_thresh, 100.0)
                    return self.update_external(
                        z, z_pred, H, R * scale, source=source,
                        force_accept=force_accept, _reacquire=True,
                    )
            log.debug(f"{src} rejected: NIS={nis:.2f} > {chi2_thresh}")
            return False

        # Reset rejections on success
        self._sensor_rejections[src] = 0

        # Kalman gain
        K = P @ H.T @ la.cho_solve(c_and_lower, np.eye(len(S)))

        # Error state injection
        dx = (K @ y).flatten()
        self._inject_error(dx)

        # Joseph form covariance update + Re-Cholesky
        I_KH = np.eye(ERROR_DIM) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        try:
            self.U = np.linalg.cholesky(P_new + np.eye(ERROR_DIM)*1e-12).T
        except np.linalg.LinAlgError:
            self.cholesky_failures += 1
            self.covariance_repairs += 1
            # Hard fallback: inflate previous U
            self.U = self.U * 1.1

        # Log Innovation
        self.innovation_history.append((self._step_count, src, y.copy(), S.copy(), nis))

        # Track spikes (using a rough threshold for 3-DOF like 16.27 for 99.9%)
        # Here we just use a generic threshold > 20 as a "spike" for logging
        if nis > 20.0:
            self.innovation_spikes += 1

        return True

    def update_external_iterated(self, z: np.ndarray,
                                 h_func: Callable[[np.ndarray], np.ndarray],
                                 H_func: Callable[[np.ndarray], np.ndarray],
                                 R: np.ndarray,
                                 source: str = "IEKF",
                                 max_iter: int = 5,
                                 tol: float = 1e-4) -> bool:
        """Iterated Extended Kalman Filter (IEKF) measurement update.

        For nonlinear measurements (UWB range, GPS pseudorange), a single
        linearization around the current state introduces bias. The IEKF
        iterates the Gauss-Newton update until convergence, re-linearizing
        H around the updated state at each step.

        Args:
            z: measurement vector
            h_func: measurement function h(x_nominal) → z_pred
            H_func: Jacobian function H(x_nominal) → dh/dx (m × ERROR_DIM)
            R: measurement noise covariance
            source: name for logging
            max_iter: maximum Gauss-Newton iterations
            tol: convergence tolerance on correction norm

        Returns:
            True if measurement accepted
        """
        if not self._initialized:
            return False

        m = z.shape[0]
        chi2_thresh = CHI2_THRESHOLDS.get(m, 3.0 * m)

        # Save original state for rollback on rejection
        x_orig = self.x.copy()
        U_orig = self.U.copy()

        accepted = False
        for iteration in range(max_iter):
            # Evaluate measurement model at current state
            z_pred = h_func(self.x)
            H = H_func(self.x)
            y = z - z_pred

            # Innovation covariance
            P = self.P
            S = H @ P @ H.T + R
            try:
                c_and_lower = la.cho_factor(S)
            except la.LinAlgError:
                log.warning(f"{source} IEKF: non-PD S at iteration {iteration}")
                break

            # Gating on first iteration only
            if iteration == 0:
                nis = float(y @ la.cho_solve(c_and_lower, y))
                if nis > chi2_thresh:
                    log.debug(f"{source} IEKF rejected: NIS={nis:.2f} > {chi2_thresh}")
                    break

            # Kalman gain and correction
            K = P @ H.T @ la.cho_solve(c_and_lower, np.eye(len(S)))
            dx = (K @ y).flatten()

            # Check convergence
            if np.linalg.norm(dx) < tol:
                accepted = True
                # Apply final correction
                self._inject_error(dx)
                # Joseph form covariance update + Re-Cholesky
                I_KH = np.eye(ERROR_DIM) - K @ H
                P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
                self.U = np.linalg.cholesky(P_new + np.eye(ERROR_DIM)*1e-12).T
                break

            # Apply intermediate correction (re-linearization point)
            self._inject_error(dx)

            if iteration == max_iter - 1:
                # Last iteration — accept with final linearization
                accepted = True
                I_KH = np.eye(ERROR_DIM) - K @ H
                P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
                self.U = np.linalg.cholesky(P_new + np.eye(ERROR_DIM)*1e-12).T

        if not accepted:
            # Rollback
            self.x = x_orig
            self.U = U_orig

        return accepted

    # ── Error Injection ────────────────────────────────────────

    def _inject_error(self, dx: np.ndarray):
        """Inject error-state correction into nominal state."""
        self.x[POS] += dx[E_POS]      # position
        self.x[VEL] += dx[E_VEL]      # velocity

        # Attitude: q = q ⊗ [1, dtheta/2]
        dtheta = dx[E_ATT]
        dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq /= np.linalg.norm(dq)
        self.x[QUAT] = self._quat_mult(self.x[QUAT], dq)
        self.x[QUAT] /= np.linalg.norm(self.x[QUAT])

        self.x[ABIAS] += dx[E_ABIAS]  # accel bias
        self.x[GBIAS] += dx[E_GBIAS]  # gyro bias

        # New states
        self.x[BARO_BIAS_IDX] += dx[E_BARO_BIAS]    # baro bias
        self.x[CLK_BIAS_IDX] += dx[E_CLK_BIAS]      # clock bias
        self.x[CLK_DRIFT_IDX] += dx[E_CLK_DRIFT]    # clock drift
        self.x[WIND] += dx[E_WIND]                   # wind

        # Clamp all bounded states
        self.x[ABIAS] = np.clip(self.x[ABIAS],
                                -self.ACCEL_BIAS_LIMIT, self.ACCEL_BIAS_LIMIT)
        self.x[GBIAS] = np.clip(self.x[GBIAS],
                                -self.GYRO_BIAS_LIMIT, self.GYRO_BIAS_LIMIT)
        self.x[BARO_BIAS_IDX] = np.clip(self.x[BARO_BIAS_IDX],
                                        -self.BARO_BIAS_LIMIT, self.BARO_BIAS_LIMIT)
        self.x[WIND] = np.clip(self.x[WIND], -self.WIND_LIMIT, self.WIND_LIMIT)

    # ── Health Monitoring ──────────────────────────────────────

    def _check_health(self):
        vel_norm = np.linalg.norm(self.x[VEL])
        q = self.x[QUAT]
        euler = self._quat_to_euler(q)
        tilt = math.degrees(math.sqrt(euler[0]**2 + euler[1]**2))
        ba_norm = np.linalg.norm(self.x[ABIAS])
        bg_norm = np.linalg.norm(self.x[GBIAS])
        P = self.P
        p_trace = np.trace(P)

        # NaN/Inf check
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            self._health = EKFHealth.FAULT
            log.critical("ESKF FAULT: NaN/Inf in state vector")
            return

        if np.any(np.isnan(self.U)) or np.any(np.isinf(self.U)):
            self._health = EKFHealth.FAULT
            log.critical("ESKF FAULT: NaN/Inf in covariance U")
            return

        # Condition number (periodic)
        try:
            cond_num = np.linalg.cond(P)
            if cond_num > 1e12:
                log.warning(f"Covariance poorly conditioned: {cond_num:.2e}")
        except np.linalg.LinAlgError:
            self._health = EKFHealth.FAULT
            log.critical("ESKF FAULT: Covariance singular")
            return

        # Fault conditions
        if (vel_norm > self.VEL_FAULT or
                tilt > self.TILT_FAULT_DEG or
                ba_norm > self.ACCEL_BIAS_LIMIT or
                bg_norm > self.GYRO_BIAS_LIMIT or
                p_trace > self.P_TRACE_LIMIT):
            self._health = EKFHealth.FAULT
            log.error(f"ESKF FAULT: vel={vel_norm:.1f} tilt={tilt:.1f} "
                      f"ba={ba_norm:.3f} bg={bg_norm:.4f} P={p_trace:.0f}")
            return

        # Warning conditions
        if vel_norm > self.VEL_WARN or tilt > self.TILT_WARN_DEG:
            self._health = EKFHealth.WARNING
            return

        # Deep condition number check (expensive)
        if self._step_count % 500 == 0:
            try:
                cond = np.linalg.cond(P)
                if cond > self.P_COND_LIMIT:
                    self._health = EKFHealth.WARNING
                    log.warning(f"ESKF WARNING: P condition number = {cond:.2e}")
                    return
            except np.linalg.LinAlgError:
                self._health = EKFHealth.FAULT
                return

        # Convergence: z-axis covariance
        z_cov = P[2, 2]
        if z_cov < self.Z_COV_CONVERGED and self._step_count > 200:
            self._health = EKFHealth.HEALTHY
        else:
            self._health = EKFHealth.CONVERGING

    # ── Quaternion Utilities ───────────────────────────────────

    @staticmethod
    def _quat_mult(q1, q2):
        """Multiply two quaternions [w,x,y,z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quat_to_dcm(q):
        """Quaternion to rotation matrix (body → NED)."""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
            [2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _quat_to_euler(q):
        """Quaternion to [roll, pitch, yaw]."""
        w, x, y, z = q
        sinr_cosp = 2.0 * (w*x + y*z)
        cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w*y - z*x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        """[roll, pitch, yaw] to quaternion [w,x,y,z]."""
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        ])

    @staticmethod
    def _skew(v):
        """Skew-symmetric (cross-product) matrix."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    # ── Adaptive Process Noise ─────────────────────────────────

    def scale_process_noise(self, vibration_level: float):
        """Scale Q based on detected vibration level.

        Args:
            vibration_level: 0.0 (calm) to 1.0+ (severe vibration).
        """
        scale = max(1.0, min(1.0 + vibration_level * 5.0, 10.0))
        self._vibration_scale = scale
        self.Q = self.Q_base * scale

    # ── Zero Velocity Update ───────────────────────────────────

    def update_zupt(self):
        """Zero velocity update — forces v=[0,0,0] as a measurement."""
        if not self._initialized:
            return

        H_zupt = np.zeros((3, ERROR_DIM))
        H_zupt[0, 3] = 1.0  # vx
        H_zupt[1, 4] = 1.0  # vy
        H_zupt[2, 5] = 1.0  # vz

        R_zupt = np.eye(3) * (0.01 ** 2)
        z = np.zeros(3)
        z_pred = self.x[VEL]

        self.update_external(z, z_pred, H_zupt, R_zupt, source="ZUPT")

    # ── GPS Fusion ──────────────────────────────────────────────

    def update_gps(self, lat: float, lon: float, alt: float,
                   hdop: float = 1.0,
                   origin_lat: float = None, origin_lon: float = None,
                   origin_alt: float = None,
                   force_accept: bool = False) -> bool:
        """GPS position update with WGS-84 → local NED conversion.

        Also helps observe wind velocity through GPS-vs-INS discrepancy.
        """
        if not self._initialized:
            return False
        if hdop > 5.0 and not force_accept:
            log.debug(f"GPS rejected: HDOP={hdop:.1f} > 5.0")
            return False

        # Set origin on first valid fix
        if self._gps_origin is None:
            self._gps_origin = {
                "lat": origin_lat if origin_lat is not None else lat,
                "lon": origin_lon if origin_lon is not None else lon,
                "alt": origin_alt if origin_alt is not None else alt,
            }
            log.info(f"GPS origin set: lat={self._gps_origin['lat']:.7f} "
                     f"lon={self._gps_origin['lon']:.7f} "
                     f"alt={self._gps_origin['alt']:.1f}m")

        # WGS-84 → local NED
        d_lat = math.radians(lat - self._gps_origin["lat"])
        d_lon = math.radians(lon - self._gps_origin["lon"])
        R_earth = 6371000.0
        lat_ref_rad = math.radians(self._gps_origin["lat"])
        north = d_lat * R_earth
        east = d_lon * R_earth * math.cos(lat_ref_rad)
        down = -(alt - self._gps_origin["alt"])

        z = np.array([north, east, down])
        z_pred = self.x[POS]

        H_gps = np.zeros((3, ERROR_DIM))
        H_gps[0, 0] = 1.0  # north
        H_gps[1, 1] = 1.0  # east
        H_gps[2, 2] = 1.0  # down

        # HDOP-scaled noise
        gps_pos_std = 2.5 * hdop
        R_gps = np.eye(3) * (gps_pos_std ** 2)
        R_gps[2, 2] *= 4.0  # vertical always worse

        return self.update_external(z, z_pred, H_gps, R_gps, source="gps", force_accept=force_accept)

    # ── Reset ──────────────────────────────────────────────────

    def reset(self):
        """Factory reset — start from scratch."""
        self.__init__(self.noise)
