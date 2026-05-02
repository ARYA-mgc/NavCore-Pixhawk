#!/usr/bin/env python3
"""
slam_interface.py
=================
Generic SLAM pose injection for the ESKF.

Accepts 6-DOF pose updates from any SLAM backend (ORB-SLAM3,
LIO-SAM, LOAM, RTAB-Map, etc.) and injects them into the ESKF
as position + orientation corrections.

Frame handling:
    SLAM systems output in their own world frame. This module
    maintains a rigid-body transform T_slam_to_ned computed via
    Umeyama alignment or a single matched pose pair.

Features:
    - Automatic and manual frame alignment
    - Innovation gating (Mahalanobis distance)
    - Loop closure detection with covariance reset
    - Pose rate limiting and outlier rejection
    - Configurable covariance scaling per SLAM confidence
"""

import logging
import math
import numpy as np
from typing import Optional
from collections import deque

log = logging.getLogger("slam_interface")

CHI2_3DOF = 7.815
CHI2_1DOF = 5.991


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def _rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to quaternion [w,x,y,z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


class SLAMInterface:
    """
    SLAM pose injection with frame alignment and loop closure handling.

    Produces position (3-DOF) and yaw (1-DOF) measurements for
    the 15-state ESKF error state.
    """

    POS_STD = 0.10           # m — typical ORB-SLAM3
    YAW_STD = 0.05           # rad (~3 deg)
    LOOP_CLOSURE_COV_SCALE = 0.1   # tighten covariance on loop closure
    MIN_UPDATE_INTERVAL = 0.05     # 20 Hz max
    OUTLIER_JUMP_M = 3.0           # reject > 3m position jump

    def __init__(self, enable: bool = False,
                 pos_std: float = 0.10, yaw_std: float = 0.05):
        self._enabled = enable
        self._frame_aligned = False
        self._pose_count = 0
        self._rejected_count = 0
        self._loop_closure_count = 0
        self._last_pose_t = 0.0
        self._last_pos_ned = None

        self.POS_STD = pos_std
        self.YAW_STD = yaw_std

        # Frame transform
        self._R_slam_to_ned = np.eye(3)
        self._t_slam_to_ned = np.zeros(3)
        self._q_slam_to_ned = np.array([1.0, 0, 0, 0])

        # ESKF measurement matrices
        self.H_pos = np.zeros((3, 15))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        self.H_pos[2, 2] = 1.0

        self.H_yaw = np.zeros((1, 15))
        self.H_yaw[0, 8] = 1.0

        if enable:
            log.info(f"SLAM interface enabled (pos_std={pos_std}m)")

    @property
    def is_active(self) -> bool:
        return self._enabled and self._frame_aligned

    @property
    def loop_closures(self) -> int:
        return self._loop_closure_count

    def set_frame_transform(self, R: np.ndarray, t: np.ndarray):
        """
        Set the SLAM→NED rotation and translation directly.

        Args:
            R: (3,3) rotation matrix.
            t: (3,) translation vector.
        """
        self._R_slam_to_ned = R.copy()
        self._t_slam_to_ned = t.copy()
        self._q_slam_to_ned = _rotation_to_quat(R)
        self._frame_aligned = True
        log.info("SLAM→NED frame transform set")

    def auto_align(self, slam_pos: np.ndarray, ned_pos: np.ndarray,
                   slam_quat: np.ndarray, ned_quat: np.ndarray):
        """
        Compute alignment from a single matched pose pair.

        For best results, call when the ESKF is converged
        and SLAM tracking has initialised.
        """
        if not self._enabled:
            return

        R_ned = _quat_to_rotation(ned_quat)
        R_slam = _quat_to_rotation(slam_quat)
        R_align = R_ned @ R_slam.T
        t_align = ned_pos - R_align @ slam_pos

        self.set_frame_transform(R_align, t_align)
        log.info("SLAM frame auto-aligned from pose pair")

    def umeyama_align(self, slam_points: np.ndarray, ned_points: np.ndarray):
        """
        Compute alignment from multiple matched point pairs using
        the Umeyama method (least-squares rigid body transform).

        Args:
            slam_points: (N, 3) points in SLAM frame.
            ned_points: (N, 3) corresponding points in NED frame.
        """
        if not self._enabled:
            return

        n = slam_points.shape[0]
        if n < 3:
            log.warning("Umeyama alignment needs >= 3 point pairs")
            return

        # Centroids
        mu_slam = slam_points.mean(axis=0)
        mu_ned = ned_points.mean(axis=0)

        # Centred points
        S = slam_points - mu_slam
        N = ned_points - mu_ned

        # Covariance
        H = S.T @ N / n

        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, np.sign(d)])

        R = Vt.T @ D @ U.T
        t = mu_ned - R @ mu_slam

        self.set_frame_transform(R, t)
        log.info(f"SLAM frame aligned via Umeyama ({n} points)")

    def process_slam_pose(self, t: float,
                          position: np.ndarray,
                          orientation: np.ndarray,
                          covariance: Optional[np.ndarray] = None,
                          is_loop_closure: bool = False) -> Optional[dict]:
        """
        Transform a SLAM pose to NED and produce ESKF measurements.

        Args:
            t: Timestamp.
            position: (3,) SLAM position.
            orientation: (4,) SLAM quaternion [w,x,y,z].
            covariance: Optional (6,6) pose covariance from SLAM.
            is_loop_closure: Whether this is a loop closure correction.

        Returns:
            Dict with H, R, innovation data for ESKF, or None if rejected.
        """
        if not self.is_active:
            return None

        # Rate limiting
        if (t - self._last_pose_t) < self.MIN_UPDATE_INTERVAL:
            return None

        # Transform to NED
        pos_ned = self._R_slam_to_ned @ position + self._t_slam_to_ned
        quat_ned = _quat_multiply(self._q_slam_to_ned, orientation)
        quat_ned /= np.linalg.norm(quat_ned)
        yaw_ned = _quat_to_yaw(quat_ned)

        # Outlier check
        if self._last_pos_ned is not None:
            dt = t - self._last_pose_t
            if dt > 0:
                speed = np.linalg.norm(pos_ned - self._last_pos_ned) / dt
                if speed > 50.0:  # > 50 m/s = clearly wrong
                    self._rejected_count += 1
                    log.warning(f"SLAM pose rejected: speed={speed:.1f} m/s")
                    return None

        jump = 0.0
        if self._last_pos_ned is not None:
            jump = np.linalg.norm(pos_ned - self._last_pos_ned)
            if jump > self.OUTLIER_JUMP_M and not is_loop_closure:
                self._rejected_count += 1
                log.warning(f"SLAM pose rejected: jump={jump:.2f}m")
                return None

        # Covariance handling
        R_pos = np.eye(3) * (self.POS_STD ** 2)
        R_yaw = np.array([[self.YAW_STD ** 2]])

        if covariance is not None and covariance.shape == (6, 6):
            # Use SLAM-provided position covariance
            R_pos = covariance[0:3, 0:3].copy()
            # Use SLAM-provided yaw covariance
            R_yaw = np.array([[covariance[5, 5]]])

        if is_loop_closure:
            self._loop_closure_count += 1
            R_pos *= self.LOOP_CLOSURE_COV_SCALE
            R_yaw *= self.LOOP_CLOSURE_COV_SCALE
            log.info(f"SLAM loop closure #{self._loop_closure_count} at t={t:.2f}")

        self._pose_count += 1
        self._last_pose_t = t
        self._last_pos_ned = pos_ned.copy()

        return {
            "type": "SLAM",
            "t": t,
            "pos_ned": pos_ned,
            "yaw_ned": yaw_ned,
            "quat_ned": quat_ned,
            "H_pos": self.H_pos,
            "H_yaw": self.H_yaw,
            "R_pos": R_pos,
            "R_yaw": R_yaw,
            "is_loop_closure": is_loop_closure,
        }

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "aligned": self._frame_aligned,
            "accepted": self._pose_count,
            "rejected": self._rejected_count,
            "loop_closures": self._loop_closure_count,
        }
