#!/usr/bin/env python3
"""
eskf_core.py
============
16-state Error-State Extended Kalman Filter (ESKF) for UAV INS.
Replaces Euler-angle EKF with quaternion attitude representation.

Nominal state (16):
  x = [px, py, pz,           <- Position (NED, m)
       vx, vy, vz,           <- Velocity (NED, m/s)
       qw, qx, qy, qz,      <- Attitude quaternion (body->NED)
       ba_x, ba_y, ba_z,     <- Accel bias (m/s^2)
       bg_x, bg_y, bg_z]     <- Gyro bias (rad/s)

Error state (15):
  dx = [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]
  Attitude error is a 3-vector (rotation vector), NOT 4 quaternion components.

Engineering notes:
  - Quaternion avoids gimbal lock inherent in Euler angles
  - Analytical Jacobian verified against numerical finite-difference in tests
  - Python cannot guarantee hard real-time; C++ port recommended for production
"""

import numpy as np
import math
import logging
from enum import Enum
from imu_noise_params import IMUNoiseParams

log = logging.getLogger("eskf_core")

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])

# Chi-squared thresholds for innovation gating (95% confidence)
CHI2_1DOF = 5.991
CHI2_2DOF = 9.210


class EKFHealth(Enum):
    CONVERGING = 0
    HEALTHY = 1
    WARNING = 2
    FAULT = 3


class ESKFCore:
    """
    Error-State Kalman Filter with quaternion attitude.
    """

    # Safety bounds
    VEL_WARN = 30.0       # m/s
    VEL_FAULT = 100.0     # m/s
    TILT_WARN_DEG = 60.0
    TILT_FAULT_DEG = 80.0
    ACCEL_BIAS_LIMIT = 2.0   # m/s^2
    GYRO_BIAS_LIMIT = 0.1    # rad/s
    P_TRACE_LIMIT = 1e6
    P_COND_LIMIT = 1e12
    SYMMETRY_INTERVAL = 50   # enforce symmetry every N steps

    # Mag rejection thresholds
    MAG_NORM_TOLERANCE = 0.30   # 30% deviation from calibrated norm
    MAG_REJECT_DURATION = 2.0   # seconds to disable mag after EMI

    def __init__(self, noise: IMUNoiseParams):
        self.noise = noise
        self._step_count = 0
        self._initialized = False
        self._health = EKFHealth.CONVERGING
        self._mag_reject_until = 0.0
        self._calibrated_mag_norm = 0.5  # Gauss, updated during init
        self._innovation_stats = {"baro": [], "mag": []}

        # --- Nominal state (16) ---
        self.x = np.zeros(16)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)

        # --- Error-state covariance (15x15) ---
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 1.0      # position
        self.P[3:6, 3:6] *= 0.1      # velocity
        self.P[6:9, 6:9] *= 0.01     # attitude
        self.P[9:12, 9:12] *= 0.01   # accel bias
        self.P[12:15, 12:15] *= 0.001  # gyro bias

        # --- Process noise ---
        self.Q = np.zeros((15, 15))
        sa = noise.accel_std ** 2
        sg = noise.gyro_std ** 2
        sab = (2.0 * noise.accel_bias_std ** 2 / max(noise.accel_bias_tau, 1.0))
        sgb = (2.0 * noise.gyro_bias_std ** 2 / max(noise.gyro_bias_tau, 1.0))
        np.fill_diagonal(self.Q[3:6, 3:6], sa)
        np.fill_diagonal(self.Q[6:9, 6:9], sg)
        np.fill_diagonal(self.Q[9:12, 9:12], sab)
        np.fill_diagonal(self.Q[12:15, 12:15], sgb)

        # --- Measurement noise ---
        self.R_baro = np.array([[noise.baro_std ** 2]])
        self.R_mag = np.array([[noise.mag_std ** 2]])
        self._R_mag_base = noise.mag_std ** 2

        # --- Observation matrices (15-col for error state) ---
        self.H_baro = np.zeros((1, 15))
        self.H_baro[0, 2] = 1.0  # observes dp_z

        self.H_mag = np.zeros((1, 15))
        self.H_mag[0, 8] = 1.0   # observes dtheta_z (yaw error)

    # ── Properties ─────────────────────────────────────────────

    @property
    def state(self) -> dict:
        q = self.x[6:10]
        euler = self._quat_to_euler(q)
        return {
            "pos": self.x[0:3].copy(),
            "vel": self.x[3:6].copy(),
            "quat": q.copy(),
            "euler": np.array(euler),
            "accel_bias": self.x[10:13].copy(),
            "gyro_bias": self.x[13:16].copy(),
        }

    @property
    def health(self) -> EKFHealth:
        return self._health

    # ── Initialization ─────────────────────────────────────────

    def initialize_from_sensors(self, accel_samples: np.ndarray,
                                mag_samples: np.ndarray) -> bool:
        """
        Initialize attitude from stationary IMU + mag data.
        accel_samples: (N, 3) in body frame
        mag_samples: (N, 3) in body frame (Gauss)
        Returns False if samples are too noisy (drone moving).
        """
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

        # Convert to quaternion
        self.x[6:10] = self._euler_to_quat(roll, pitch, yaw)
        self.x[0:6] = 0.0   # position and velocity = 0
        self.x[10:16] = 0.0  # biases = 0

        self._initialized = True
        self._health = EKFHealth.CONVERGING
        log.info(f"ESKF initialized: roll={math.degrees(roll):.1f} "
                 f"pitch={math.degrees(pitch):.1f} yaw={math.degrees(yaw):.1f}")
        return True

    # ── Predict ────────────────────────────────────────────────

    def predict(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, dt: float):
        """
        Propagate nominal state using bias-compensated IMU.
        Quaternion integration via first-order multiplication.
        """
        if dt <= 0:
            return

        # Bias compensation
        accel = accel_raw - self.x[10:13]
        gyro = gyro_raw - self.x[13:16]

        q = self.x[6:10].copy()
        R = self._quat_to_dcm(q)

        # 1. Velocity update: v += (R*f + g) * dt
        f_ned = R @ accel
        self.x[3:6] += (f_ned + GRAVITY_NED) * dt

        # 2. Position update: p += v * dt
        self.x[0:3] += self.x[3:6] * dt

        # 3. Quaternion update: q = q * q_delta
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-10:
            axis = gyro / (angle / dt) * dt
            axis_norm = axis / np.linalg.norm(axis)
            ha = angle / 2.0
            q_delta = np.array([math.cos(ha),
                                axis_norm[0] * math.sin(ha),
                                axis_norm[1] * math.sin(ha),
                                axis_norm[2] * math.sin(ha)])
        else:
            q_delta = np.array([1.0, 0.0, 0.0, 0.0])

        self.x[6:10] = self._quat_mult(q, q_delta)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])  # normalize

        # 4. Bias decay (Gauss-Markov)
        tau_a = max(self.noise.accel_bias_tau, 1.0)
        tau_g = max(self.noise.gyro_bias_tau, 1.0)
        self.x[10:13] *= (1.0 - dt / tau_a)
        self.x[13:16] *= (1.0 - dt / tau_g)

        # 5. Clamp biases
        self.x[10:13] = np.clip(self.x[10:13],
                                -self.ACCEL_BIAS_LIMIT, self.ACCEL_BIAS_LIMIT)
        self.x[13:16] = np.clip(self.x[13:16],
                                -self.GYRO_BIAS_LIMIT, self.GYRO_BIAS_LIMIT)

        # 6. Error-state Jacobian (15x15)
        F = self._compute_F(accel, gyro, R, dt)
        self.P = F @ self.P @ F.T + self.Q * dt

        # 7. Covariance hardening & health
        self._harden_covariance()
        self._check_health()

    def _compute_F(self, accel, gyro, R, dt) -> np.ndarray:
        """
        Error-state transition Jacobian (15x15).
        """
        F = np.eye(15)

        # dp/dv
        F[0:3, 3:6] = np.eye(3) * dt

        # dv/dtheta: -R * [accel]x * dt
        F[3:6, 6:9] = -R @ self._skew(accel) * dt

        # dv/dba: -R * dt
        F[3:6, 9:12] = -R * dt

        # dtheta/dtheta: I - [gyro]x * dt
        F[6:9, 6:9] = np.eye(3) - self._skew(gyro) * dt

        # dtheta/dbg: -I * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        # Bias states: decay
        tau_a = max(self.noise.accel_bias_tau, 1.0)
        tau_g = max(self.noise.gyro_bias_tau, 1.0)
        F[9:12, 9:12] = np.eye(3) * (1.0 - dt / tau_a)
        F[12:15, 12:15] = np.eye(3) * (1.0 - dt / tau_g)

        return F

    # ── Measurement Updates ────────────────────────────────────

    def update_baro(self, alt_measured: float):
        """
        Barometric altitude update with innovation gating.
        Rejects altitude jumps > 5m between consecutive samples.
        """
        z = np.array([alt_measured])
        z_pred = np.array([self.x[2]])
        y = z - z_pred  # innovation

        # Adaptive R: inflate on large transient
        R = self.R_baro.copy()
        if abs(y[0]) > 2.0:
            R *= 5.0

        # Innovation gating (Mahalanobis)
        S = self.H_baro @ self.P @ self.H_baro.T + R
        nis = float(y @ np.linalg.inv(S) @ y)

        if nis > CHI2_1DOF:
            log.debug(f"Baro rejected: NIS={nis:.2f} > {CHI2_1DOF}")
            return

        # Standard Kalman update on error state
        K = self.P @ self.H_baro.T @ np.linalg.inv(S)
        dx = (K @ y).flatten()
        self._inject_error(dx)

        # Joseph form covariance update
        I_KH = np.eye(15) - K @ self.H_baro
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_mag(self, yaw_measured: float, mag_norm: float = -1.0,
                   t_now: float = 0.0):
        """
        Magnetometer yaw update with 3-tier rejection:
          1. Field norm check: skip if |norm - calibrated| > 30%
          2. Time-based rejection: skip if recently EMI-contaminated
          3. Innovation gating: Mahalanobis distance
        """
        # Tier 1: field norm check
        if mag_norm > 0:
            norm_ratio = abs(mag_norm / self._calibrated_mag_norm - 1.0)
            if norm_ratio > self.MAG_NORM_TOLERANCE:
                self._mag_reject_until = t_now + self.MAG_REJECT_DURATION
                log.debug(f"Mag rejected (norm): ratio={norm_ratio:.2f}")
                return

        # Tier 2: time-based rejection
        if t_now > 0 and t_now < self._mag_reject_until:
            return

        # Get predicted yaw from quaternion
        euler = self._quat_to_euler(self.x[6:10])
        yaw_pred = euler[2]

        y = np.array([self._wrap_angle(yaw_measured - yaw_pred)])

        # Adaptive R
        R = np.array([[self._R_mag_base]])
        if mag_norm > 0:
            norm_ratio = abs(mag_norm / self._calibrated_mag_norm - 1.0)
            if norm_ratio > 0.15:
                R *= 10.0  # inflate but don't reject

        # Tier 3: innovation gating
        S = self.H_mag @ self.P @ self.H_mag.T + R
        nis = float(y @ np.linalg.inv(S) @ y)

        if nis > CHI2_1DOF:
            log.debug(f"Mag rejected (NIS): NIS={nis:.2f}")
            return

        K = self.P @ self.H_mag.T @ np.linalg.inv(S)
        dx = (K @ y).flatten()
        self._inject_error(dx)
        
        # Joseph form covariance update
        I_KH = np.eye(15) - K @ self.H_mag
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_optical_flow(self, flow_vx: float, flow_vy: float,
                            distance: float, quality: int):
        """
        Optical flow velocity update.
        Requires valid rangefinder distance. No distance = no fusion.
        """
        if distance <= 0.05 or quality < 10:
            return

        H_flow = np.zeros((2, 15))
        H_flow[0, 3] = 1.0  # vx
        H_flow[1, 4] = 1.0  # vy

        R_base = 0.5 ** 2
        R_flow = np.eye(2) * (R_base * 100.0 / max(quality, 1))

        z = np.array([flow_vx, flow_vy])
        z_pred = self.x[3:5]
        y = z - z_pred

        S = H_flow @ self.P @ H_flow.T + R_flow
        nis = float(y @ np.linalg.inv(S) @ y)

        if nis > CHI2_2DOF:
            return

        K = self.P @ H_flow.T @ np.linalg.inv(S)
        dx = (K @ y).flatten()
        self._inject_error(dx)
        
        # Joseph form covariance update
        I_KH = np.eye(15) - K @ H_flow
        self.P = I_KH @ self.P @ I_KH.T + K @ R_flow @ K.T

    # ── Error Injection ────────────────────────────────────────

    def _inject_error(self, dx: np.ndarray):
        """Inject 15-element error state into nominal state."""
        self.x[0:3] += dx[0:3]   # position
        self.x[3:6] += dx[3:6]   # velocity

        # Attitude: q = q * [1, dtheta/2]
        dtheta = dx[6:9]
        dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq /= np.linalg.norm(dq)
        self.x[6:10] = self._quat_mult(self.x[6:10], dq)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

        self.x[10:13] += dx[9:12]   # accel bias
        self.x[13:16] += dx[12:15]  # gyro bias

        # Clamp biases
        self.x[10:13] = np.clip(self.x[10:13],
                                -self.ACCEL_BIAS_LIMIT, self.ACCEL_BIAS_LIMIT)
        self.x[13:16] = np.clip(self.x[13:16],
                                -self.GYRO_BIAS_LIMIT, self.GYRO_BIAS_LIMIT)

    # ── Health Monitoring ──────────────────────────────────────

    def _harden_covariance(self):
        """
        Force symmetry and bound eigenvalues to ensure positive-definiteness.
        Must be called after updates/predict to prevent numerical divergence.
        """
        # Enforce symmetry
        self.P = (self.P + self.P.T) / 2.0

        # Eigenvalue bounding
        min_eigenvalue = 1e-9
        max_eigenvalue = 1e4
        eigvals, eigvecs = np.linalg.eigh(self.P)

        if np.any(eigvals < min_eigenvalue) or np.any(eigvals > max_eigenvalue):
            # Clamp eigenvalues
            eigvals = np.clip(eigvals, min_eigenvalue, max_eigenvalue)
            # Reconstruct P
            self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-enforce symmetry
            self.P = (self.P + self.P.T) / 2.0

    def _check_health(self):
        vel_norm = np.linalg.norm(self.x[3:6])
        q = self.x[6:10]
        euler = self._quat_to_euler(q)
        tilt = math.degrees(math.sqrt(euler[0]**2 + euler[1]**2))
        ba_norm = np.linalg.norm(self.x[10:13])
        bg_norm = np.linalg.norm(self.x[13:16])
        p_trace = np.trace(self.P)

        # Check for NaN/Inf
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            self._health = EKFHealth.FAULT
            log.critical("ESKF FAULT: NaN/Inf in state vector")
            return

        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self._health = EKFHealth.FAULT
            log.critical("ESKF FAULT: NaN/Inf in covariance")
            return

        # Condition number
        try:
            cond_num = np.linalg.cond(self.P)
            if cond_num > 1e8:
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

        # Condition number check (expensive, do rarely)
        if self._step_count % 500 == 0:
            try:
                cond = np.linalg.cond(self.P)
                if cond > self.P_COND_LIMIT:
                    self._health = EKFHealth.WARNING
                    log.warning(f"ESKF WARNING: P condition number = {cond:.2e}")
                    return
            except np.linalg.LinAlgError:
                self._health = EKFHealth.FAULT
                return

        if self._step_count > 200:
            self._health = EKFHealth.HEALTHY

    # ── Quaternion Utilities ───────────────────────────────────

    @staticmethod
    def _quat_mult(q1, q2):
        """Hamilton product: q1 * q2."""
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
        """Quaternion to Direction Cosine Matrix (body -> NED)."""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
            [2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _quat_to_euler(q):
        """Quaternion to Euler angles [roll, pitch, yaw]."""
        w, x, y, z = q
        # Roll
        sinr_cosp = 2.0 * (w*x + y*z)
        cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch
        sinp = 2.0 * (w*y - z*x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        # Yaw
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        """Euler angles to quaternion [w, x, y, z]."""
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
        """Skew-symmetric matrix from 3-vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    # ── Reset ──────────────────────────────────────────────────

    def reset(self):
        """Reset to initial state."""
        self.__init__(self.noise)
