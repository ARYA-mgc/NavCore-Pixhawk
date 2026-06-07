#!/usr/bin/env python3
# Visual Inertial Odometry.
# Camera + Math = Position. Keep the lens clean.

import logging
import math
import numpy as np
from typing import Optional
from collections import deque

log = logging.getLogger("vio_pipeline")

CHI2_3DOF = 7.815   # chi-squared 3-DOF, 95%
CHI2_1DOF = 5.991


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    # quat to rotation matrix
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def _quat_to_yaw(q: np.ndarray) -> float:
    # Extract yaw from quaternion [w,x,y,z].
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product of two quaternions [w,x,y,z].
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    # Inverse of a unit quaternion.
    return np.array([q[0], -q[1], -q[2], -q[3]])


class VIOPipeline:
    # Visual Inertial Odometry fusion for the ESKF.

    # Measurement noise (position and yaw)
    POS_STD = 0.05    # m — typical T265 / ORB-SLAM3
    YAW_STD = 0.03    # rad (~1.7 deg)
    MIN_CONFIDENCE = 0.3
    MIN_UPDATE_INTERVAL = 0.02  # 50 Hz max
    OUTLIER_WINDOW = 5

    def __init__(self, enable: bool = False,
                 pos_std: float = 0.05, yaw_std: float = 0.03):
        self._enabled = enable
        self._initialized = False
        self._last_pose_t = 0.0
        self._pose_count = 0
        self._rejected_count = 0

        self.POS_STD = pos_std
        self.YAW_STD = yaw_std

        # Frame alignment
        self._T_vio_to_ned = np.eye(4)
        self._R_vio_to_ned = np.eye(3)
        self._t_vio_to_ned = np.zeros(3)
        self._q_vio_to_ned = np.array([1.0, 0, 0, 0])

        # Outlier filter
        self._pos_history = deque(maxlen=self.OUTLIER_WINDOW)

        # Measurement matrices for ESKF (position: observe dx,dy,dz)
        self.H_pos = np.zeros((3, 20))
        self.H_pos[0, 0] = 1.0  # dp_x
        self.H_pos[1, 1] = 1.0  # dp_y
        self.H_pos[2, 2] = 1.0  # dp_z

        self.H_yaw = np.zeros((1, 20))
        self.H_yaw[0, 8] = 1.0  # dtheta_z

        self.R_pos = np.eye(3) * (pos_std ** 2)
        self.R_yaw = np.array([[yaw_std ** 2]])

        if enable:
            log.info(f"VIO pipeline enabled (pos_std={pos_std}m, yaw_std={yaw_std}rad)")

    @property
    def is_active(self) -> bool:
        return self._enabled and self._initialized

    @property
    def stats(self) -> dict:
        return {
            "accepted": self._pose_count,
            "rejected": self._rejected_count,
            "total": self._pose_count + self._rejected_count,
        }

    def initialize(self, eskf_pos: np.ndarray, eskf_quat: np.ndarray,
                   vio_pos: np.ndarray, vio_quat: np.ndarray):
        # Compute VIO→NED alignment from a matched pose pair.
        if not self._enabled:
            return

        # Rotation: R_ned = R_align * R_vio  =>  R_align = R_ned * R_vio^T
        R_ned = _quat_to_rotation(eskf_quat)
        R_vio = _quat_to_rotation(vio_quat)
        self._R_vio_to_ned = R_ned @ R_vio.T

        # Translation: t_ned = R_align * t_vio + offset
        self._t_vio_to_ned = eskf_pos - self._R_vio_to_ned @ vio_pos

        # Store as quaternion for yaw transform
        self._q_vio_to_ned = eskf_quat.copy()
        q_vio_inv = _quat_inverse(vio_quat)
        self._q_vio_to_ned = _quat_multiply(eskf_quat, q_vio_inv)

        # Build 4x4 homogeneous
        self._T_vio_to_ned[:3, :3] = self._R_vio_to_ned
        self._T_vio_to_ned[:3, 3] = self._t_vio_to_ned

        self._initialized = True
        log.info("VIO pipeline initialized with frame alignment")

    def process_vio_update(self, t: float,
                           position: np.ndarray,
                           orientation: np.ndarray,
                           confidence: float = 1.0) -> Optional[dict]:
        # Process an incoming VIO pose and produce ESKF measurements.
        if not self.is_active:
            return None

        # 1. Confidence check
        if confidence < self.MIN_CONFIDENCE:
            self._rejected_count += 1
            return None

        # 2. Rate limiting
        if (t - self._last_pose_t) < self.MIN_UPDATE_INTERVAL:
            return None

        # 3. Transform VIO → NED
        pos_ned = self._R_vio_to_ned @ position + self._t_vio_to_ned
        quat_ned = _quat_multiply(self._q_vio_to_ned, orientation)
        quat_ned /= np.linalg.norm(quat_ned)
        yaw_ned = _quat_to_yaw(quat_ned)

        # 4. Outlier filter (median on position)
        self._pos_history.append(pos_ned.copy())
        if len(self._pos_history) >= 3:
            median_pos = np.median(np.array(self._pos_history), axis=0)
            jump = np.linalg.norm(pos_ned - median_pos)
            if jump > 2.0:  # 2m position jump = outlier
                log.warning(f"VIO outlier rejected: jump={jump:.2f}m")
                self._rejected_count += 1
                return None

        # 5. Scale noise by inverse confidence
        noise_scale = 1.0 / max(confidence, 0.3)
        R_pos = self.R_pos * (noise_scale ** 2)
        R_yaw = self.R_yaw * (noise_scale ** 2)

        self._pose_count += 1
        self._last_pose_t = t

        return {
            "type": "VIO",
            "t": t,
            "pos_ned": pos_ned,
            "yaw_ned": yaw_ned,
            "quat_ned": quat_ned,
            "H_pos": self.H_pos,
            "H_yaw": self.H_yaw,
            "R_pos": R_pos,
            "R_yaw": R_yaw,
            "confidence": confidence,
        }
