#!/usr/bin/env python3
# Taking our math guesses and ArduPilot's guesses and tossing them in a blender.

import logging
import math
import numpy as np
from enum import Enum, auto
from typing import Optional

log = logging.getLogger("ekf3_blender")


class BlendMode(Enum):
    ESKF_ONLY  = auto()
    EKF3_ONLY  = auto()
    WEIGHTED   = auto()
    CROSSCHECK = auto()


class EKF3Blender:
    # Cross-checking and covariance-intersection blending between

    # Divergence thresholds
    POS_DIVERGE_WARN = 2.0     # meters
    POS_DIVERGE_FAULT = 10.0
    VEL_DIVERGE_WARN = 3.0     # m/s
    VEL_DIVERGE_FAULT = 15.0
    ATT_DIVERGE_WARN = 10.0    # degrees
    ATT_DIVERGE_FAULT = 30.0

    # Hysteresis
    DIVERGE_WARN_COUNT = 5
    DIVERGE_FAULT_COUNT = 10

    def __init__(self, mode: BlendMode = BlendMode.ESKF_ONLY):
        self._mode = mode
        self._last_ekf3_state = None
        self._last_eskf_state = None

        # Divergence tracking
        self._diverge_warn_count = 0
        self._diverge_fault_count = 0
        self._total_diverge_events = 0

        # Statistics
        self._blend_count = 0

        log.info(f"EKF3 blender mode: {mode.name}")

    @property
    def mode(self) -> BlendMode:
        return self._mode

    @property
    def is_diverged(self) -> bool:
        return self._diverge_fault_count >= self.DIVERGE_FAULT_COUNT

    def set_mode(self, mode: BlendMode):
        old = self._mode
        self._mode = mode
        if old != mode:
            log.info(f"EKF3 blender mode: {old.name} -> {mode.name}")

    def update_eskf(self, pos: np.ndarray, vel: np.ndarray,
                    euler: np.ndarray, P: np.ndarray):
        # Update with latest ESKF state and covariance.
        self._last_eskf_state = {
            "pos": np.asarray(pos, dtype=float).copy(),
            "vel": np.asarray(vel, dtype=float).copy(),
            "euler": np.asarray(euler, dtype=float).copy(),
            "P_diag": np.diag(P).copy() if P.ndim == 2 else np.asarray(P).copy(),
        }

    def update_ekf3(self, pos: np.ndarray, vel: np.ndarray,
                    euler: np.ndarray,
                    pos_variance: float = 1.0,
                    vel_variance: float = 0.5,
                    att_variance: float = 0.01):
        # Update with latest ArduPilot EKF3 state.
        self._last_ekf3_state = {
            "pos": np.asarray(pos, dtype=float).copy(),
            "vel": np.asarray(vel, dtype=float).copy(),
            "euler": np.asarray(euler, dtype=float).copy(),
            "P_diag": np.array([
                pos_variance, pos_variance, pos_variance,
                vel_variance, vel_variance, vel_variance,
                att_variance, att_variance, att_variance,
                0.0, 0.0, 0.0,  # accel bias
                0.0, 0.0, 0.0,  # gyro bias
                0.0,            # baro bias
                0.0,            # clock bias
                0.0,            # clock drift
                0.0, 0.0        # wind
            ]),
        }

    def get_blended_output(self) -> Optional[dict]:
        # Compute the blended output based on current mode.
        eskf = self._last_eskf_state
        ekf3 = self._last_ekf3_state

        if self._mode == BlendMode.ESKF_ONLY:
            return eskf

        if self._mode == BlendMode.EKF3_ONLY:
            return ekf3

        if eskf is None or ekf3 is None:
            return eskf if eskf is not None else ekf3

        if self._mode == BlendMode.CROSSCHECK:
            return self._crosscheck(eskf, ekf3)

        if self._mode == BlendMode.WEIGHTED:
            return self._covariance_intersection(eskf, ekf3)

        return eskf

    def _crosscheck(self, eskf: dict, ekf3: dict) -> dict:
        # Use ESKF but track divergence with hysteresis.
        pos_diff = np.linalg.norm(eskf["pos"] - ekf3["pos"])
        vel_diff = np.linalg.norm(eskf["vel"] - ekf3["vel"])

        # Yaw difference with wrapping
        yaw_diff = abs(eskf["euler"][2] - ekf3["euler"][2])
        yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)
        att_diff_deg = math.degrees(yaw_diff)

        is_fault = (pos_diff > self.POS_DIVERGE_FAULT or
                    vel_diff > self.VEL_DIVERGE_FAULT or
                    att_diff_deg > self.ATT_DIVERGE_FAULT)

        is_warn = (pos_diff > self.POS_DIVERGE_WARN or
                   vel_diff > self.VEL_DIVERGE_WARN or
                   att_diff_deg > self.ATT_DIVERGE_WARN)

        if is_fault:
            self._diverge_fault_count += 1
            self._total_diverge_events += 1
            log.error(f"ESKF/EKF3 DIVERGENCE FAULT #{self._diverge_fault_count}: "
                      f"pos={pos_diff:.2f}m vel={vel_diff:.2f}m/s "
                      f"att={att_diff_deg:.1f}deg")
        elif is_warn:
            self._diverge_warn_count += 1
            if self._diverge_warn_count % 10 == 1:
                log.warning(f"ESKF/EKF3 divergence: "
                            f"pos={pos_diff:.2f}m vel={vel_diff:.2f}m/s "
                            f"att={att_diff_deg:.1f}deg")
        else:
            # Good agreement — decay counters
            self._diverge_warn_count = max(0, self._diverge_warn_count - 1)
            self._diverge_fault_count = max(0, self._diverge_fault_count - 1)

        result = eskf.copy()
        result["crosscheck"] = {
            "pos_diff": pos_diff,
            "vel_diff": vel_diff,
            "att_diff_deg": att_diff_deg,
            "is_diverged": self.is_diverged,
        }
        return result

    def _covariance_intersection(self, eskf: dict, ekf3: dict) -> dict:
        # Covariance Intersection (CI) blend.
        self._blend_count += 1

        pos_fused, P_pos = self._ci_blend_3dof(
            eskf["pos"], eskf["P_diag"][0:3],
            ekf3["pos"], ekf3["P_diag"][0:3])

        vel_fused, P_vel = self._ci_blend_3dof(
            eskf["vel"], eskf["P_diag"][3:6],
            ekf3["vel"], ekf3["P_diag"][3:6])

        # Euler blending with yaw wrapping
        euler_fused = self._blend_euler(
            eskf["euler"], eskf["P_diag"][6:9],
            ekf3["euler"], ekf3["P_diag"][6:9])

        return {
            "pos": pos_fused,
            "vel": vel_fused,
            "euler": euler_fused,
            "P_diag": np.concatenate([P_pos, P_vel, np.zeros(14)]),
            "blend_count": self._blend_count,
        }

    @staticmethod
    def _ci_blend_3dof(x1: np.ndarray, p1_diag: np.ndarray,
                       x2: np.ndarray, p2_diag: np.ndarray,
                       alpha_steps: int = 11):
        # 1D Covariance Intersection per axis, optimal alpha search.
        fused_x = np.zeros(3)
        fused_p = np.zeros(3)

        for i in range(3):
            v1 = max(p1_diag[i], 1e-10)
            v2 = max(p2_diag[i], 1e-10)

            # Search alpha that minimises fused variance
            best_alpha = 0.5
            best_var = float('inf')

            for k in range(alpha_steps):
                a = k / (alpha_steps - 1)
                var = 1.0 / (a / v1 + (1.0 - a) / v2)
                if var < best_var:
                    best_var = var
                    best_alpha = a

            # Fuse
            fused_p[i] = best_var
            fused_x[i] = fused_p[i] * (
                best_alpha * x1[i] / v1 +
                (1.0 - best_alpha) * x2[i] / v2
            )

        return fused_x, fused_p

    @staticmethod
    def _blend_euler(e1: np.ndarray, p1_diag: np.ndarray,
                     e2: np.ndarray, p2_diag: np.ndarray) -> np.ndarray:
        # Blend Euler angles with yaw wrapping.
        fused = np.zeros(3)
        for i in range(3):
            v1 = max(p1_diag[i], 1e-10)
            v2 = max(p2_diag[i], 1e-10)

            a1 = e1[i]
            a2 = e2[i]

            # Wrap difference for yaw (index 2)
            if i == 2:
                diff = a2 - a1
                diff = math.atan2(math.sin(diff), math.cos(diff))
                a2 = a1 + diff

            # Inverse-variance weighted average
            w1 = 1.0 / v1
            w2 = 1.0 / v2
            fused[i] = (w1 * a1 + w2 * a2) / (w1 + w2)

            # Wrap result
            if i == 2:
                fused[i] = math.atan2(math.sin(fused[i]),
                                       math.cos(fused[i]))

        return fused

    def get_status(self) -> dict:
        return {
            "mode": self._mode.name,
            "is_diverged": self.is_diverged,
            "warn_count": self._diverge_warn_count,
            "fault_count": self._diverge_fault_count,
            "total_events": self._total_diverge_events,
            "blend_count": self._blend_count,
        }
