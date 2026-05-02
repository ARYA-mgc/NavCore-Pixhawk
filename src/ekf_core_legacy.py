"""
ekf_core_legacy.py
==================
DEPRECATED — DO NOT USE IN NEW CODE.

This module uses Euler angles and suffers from gimbal lock.
It is retained ONLY for migration verification.
Will be deleted after Phase 1 migration is complete.

Migration: Use eskf_core.ESKFCore instead.

15-state Extended Kalman Filter for UAV INS.
Enhanced version with IMU bias estimation.

State vector (15×1):
  x = [px, py, pz,          <- Position (NED, m)
       vx, vy, vz,          <- Velocity (NED, m/s)
       phi, theta, psi,     <- Euler Angles (rad)
       abx, aby, abz,       <- Accel Bias (m/s²)
       gbx, gby, gbz]       <- Gyro Bias (rad/s)
"""

import numpy as np
import math
from imu_noise_params import IMUNoiseParams

GRAVITY = np.array([0.0, 0.0, 9.80665])

class EKFCore:
    def __init__(self, noise: IMUNoiseParams):
        self.noise = noise
        
        # 1. Initialize State Vector (15 elements)
        self.x = np.zeros(15)
        
        # 2. Initialize Covariance Matrix (15x15)
        # We start with small uncertainties for position and velocity,
        # and very small ones for biases.
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 1.0     # Position (m²)
        self.P[3:6, 3:6] *= 0.1     # Velocity (m/s)²
        self.P[6:9, 6:9] *= 0.01    # Attitude (rad²)
        self.P[9:12, 9:12] *= 0.01  # Accel bias (m/s²)²
        self.P[12:15, 12:15] *= 0.001 # Gyro bias (rad/s)²

        # 3. Process Noise Q (15x15)
        # Models how much we "mistrust" our prediction over time.
        self.Q = np.zeros((15, 15))
        sa = noise.accel_std ** 2
        sg = noise.gyro_std  ** 2
        sab = 1e-6 # Accel bias stability
        sgb = 1e-8 # Gyro bias stability
        
        # Diagonal process noise elements
        np.fill_diagonal(self.Q[3:6, 3:6], sa)     # Velocity driven by accel noise
        np.fill_diagonal(self.Q[6:9, 6:9], sg)     # Attitude driven by gyro noise
        np.fill_diagonal(self.Q[9:12, 9:12], sab)  # Accel bias drift
        np.fill_diagonal(self.Q[12:15, 12:15], sgb)# Gyro bias drift

        # 4. Measurement Noise
        self.R_baro = np.array([[noise.baro_std ** 2]])
        self.R_mag  = np.array([[noise.mag_std ** 2]])

        # 5. Observation Matrices
        self.H_baro = np.zeros((1, 15))
        self.H_baro[0, 2] = 1.0  # Baro observes pz
        
        self.H_mag = np.zeros((1, 15))
        self.H_mag[0, 8] = 1.0   # Mag observes psi

    @property
    def state(self) -> dict:
        return {
            "pos":   self.x[0:3].copy(),
            "vel":   self.x[3:6].copy(),
            "euler": self.x[6:9].copy(),
            "accel_bias": self.x[9:12].copy(),
            "gyro_bias":  self.x[12:15].copy()
        }

    def predict(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, dt: float):
        """
        Predict step: Propagates the state forward using IMU mechanization.
        Now compensates for estimated biases.
        """
        # --- A. Compensate Biases ---
        accel = accel_raw - self.x[9:12]
        gyro  = gyro_raw  - self.x[12:15]
        
        phi, theta, psi = self.x[6:9]
        R = self._R_bn(phi, theta, psi)
        T = self._T_inv(phi, theta)

        # --- B. State Propagation ---
        # 1. Velocity Update (v_new = v + (R*f + g)*dt)
        f_ned = R @ accel
        acc_ned = f_ned + GRAVITY
        self.x[3:6] += acc_ned * dt

        # 2. Position Update (p_new = p + v*dt)
        self.x[0:3] += self.x[3:6] * dt

        # 3. Attitude Update (eta_new = eta + (T*omega)*dt)
        eta_dot = T @ gyro
        self.x[6:9] += eta_dot * dt
        self.x[6:9] = [self._wrap_angle(a) for a in self.x[6:9]]

        # --- C. Covariance Propagation (P = FPF' + Q) ---
        F = self._compute_F_15(accel, phi, theta, psi, dt)
        self.P = F @ self.P @ F.T + self.Q * dt

    def _compute_F_15(self, accel, phi, theta, psi, dt) -> np.ndarray:
        """Linearized state transition Jacobian (15x15)"""
        F = np.eye(15)
        
        # Pos derivative w.r.t Vel
        F[0:3, 3:6] += np.eye(3) * dt
        
        # Vel derivative w.r.t Euler (Numerical)
        R = self._R_bn(phi, theta, psi)
        f_ned = R @ accel
        eps = 1e-5
        for i in range(3):
            euler_p = np.array([phi, theta, psi])
            euler_p[i] += eps
            df = (self._R_bn(*euler_p) @ accel - f_ned) / eps
            F[3:6, 6+i] = df * dt
            
        # Vel derivative w.r.t Accel Bias
        F[3:6, 9:12] = -R * dt
        
        # Euler derivative w.r.t Gyro Bias
        T = self._T_inv(phi, theta)
        F[6:9, 12:15] = -T * dt
        
        return F

    def update_baro(self, alt_measured: float):
        self.x[6:9] = [self._wrap_angle(a) for a in self.x[6:9]]

    # ═══════════════════════════════════════════════════════════
    # UPDATE — magnetometer (yaw)
    # ═══════════════════════════════════════════════════════════
    def update_mag(self, yaw_measured: float):
        """
        EKF measurement update using magnetometer yaw.
        H = [0 0 0  0 0 0  0 0 1]  (observes psi)
        Innovation is wrapped to [-π, π] to avoid wrap-around jumps.
        """
        z_meas = np.array([yaw_measured])
        z_pred = np.array([self.x[8]])

        y = np.array([self._wrap_angle(z_meas[0] - z_pred[0])])

        S = self.H_mag @ self.P @ self.H_mag.T + self.R_mag
        K = self.P @ self.H_mag.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(9) - K @ self.H_mag) @ self.P

        self.x[6:9] = [self._wrap_angle(a) for a in self.x[6:9]]

    # ── util ───────────────────────────────────────────────────
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        import math
        return math.atan2(math.sin(a), math.cos(a))

    def reset(self):
        """Reset state and covariance to initial values."""
        self.__init__(self.noise)
