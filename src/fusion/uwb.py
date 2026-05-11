#!/usr/bin/env python3
# Radio beacons beep-booping to give us position. Kinda like GPS but indoors and way more annoying to set up.

import logging
import math
import numpy as np
from typing import Dict, Optional, List
from collections import deque

log = logging.getLogger("uwb_fusion")

CHI2_1DOF = 5.991


class UWBAnchor:
    """Known UWB anchor with position and measurement history."""
    def __init__(self, anchor_id: str, position: np.ndarray):
        self.id = anchor_id
        self.position = np.array(position, dtype=float)  # NED (3,)
        self.last_range = 0.0
        self.last_t = 0.0
        self.range_history = deque(maxlen=20)
        self.rejected_count = 0
        self.accepted_count = 0

    @property
    def is_stale(self) -> bool:
        """Consider stale if no update in 2 seconds."""
        return self.last_t == 0.0

    def staleness(self, t_now: float) -> float:
        if self.last_t == 0.0:
            return float('inf')
        return t_now - self.last_t


class UWBFusion:
    """
    Ultra-Wideband range fusion for the ESKF.

    Each range measurement from a known anchor produces a 1-DOF
    range-only update with a nonlinear observation model that is
    linearised around the current position estimate.
    """

    RANGE_STD = 0.15        # meters (typical DW1000)
    MAX_RANGE = 100.0       # meters
    MIN_RANGE = 0.3         # meters (too close = multipath)
    NLOS_THRESHOLD = 3.0    # residual > N*sigma = probable NLOS
    STALE_TIMEOUT = 2.0     # seconds
    MIN_ANCHORS_FOR_TRILATERATION = 3

    def __init__(self, enable: bool = False, range_std: float = 0.15):
        self._enabled = enable
        self._anchors: Dict[str, UWBAnchor] = {}
        self._measurement_count = 0
        self.RANGE_STD = range_std

        if enable:
            log.info(f"UWB fusion enabled (range_std={range_std}m)")

    @property
    def is_active(self) -> bool:
        return self._enabled and len(self._anchors) > 0

    @property
    def anchor_count(self) -> int:
        return len(self._anchors)

    def add_anchor(self, anchor_id: str, position: np.ndarray):
        """Register a UWB anchor at a known NED position."""
        self._anchors[anchor_id] = UWBAnchor(anchor_id, np.asarray(position))
        log.info(f"UWB anchor '{anchor_id}' registered at {position}")

    def remove_anchor(self, anchor_id: str):
        """Remove an anchor."""
        if anchor_id in self._anchors:
            del self._anchors[anchor_id]
            log.info(f"UWB anchor '{anchor_id}' removed")

    def process_range(self, anchor_id: str, range_m: float, t: float,
                      current_pos: np.ndarray) -> Optional[dict]:
        """
        Process a single range measurement and produce an ESKF update.

        Args:
            anchor_id: Ranging anchor ID.
            range_m: Measured range (meters).
            t: Timestamp (seconds).
            current_pos: (3,) current ESKF position estimate (NED).

        Returns:
            Dict with H, innovation, R for ESKF, or None if rejected.
        """
        if not self._enabled:
            return None

        if anchor_id not in self._anchors:
            log.debug(f"Unknown UWB anchor: {anchor_id}")
            return None

        # Range bounds check
        if range_m < self.MIN_RANGE or range_m > self.MAX_RANGE:
            return None

        anchor = self._anchors[anchor_id]

        # Predicted range from current position estimate
        diff = anchor.position - current_pos
        pred_range = np.linalg.norm(diff)

        if pred_range < 0.01:
            # Degenerate — anchor at drone position
            return None

        # Innovation (range residual)
        innovation = range_m - pred_range

        # Linearised observation matrix (1x15)
        # dz/dp = -(p_anchor - p_hat)^T / ||p_anchor - p_hat||
        unit_vec = diff / pred_range
        H = np.zeros((1, 15))
        H[0, 0:3] = -unit_vec

        # Measurement noise
        R = np.array([[self.RANGE_STD ** 2]])

        # Innovation covariance (scalar for 1-DOF)
        # S = H @ P @ H^T + R  -- caller (ESKF) will compute this

        # NLOS detection: if residual is too large, inflate noise
        nlos_detected = False
        if abs(innovation) > self.NLOS_THRESHOLD * self.RANGE_STD:
            log.debug(f"UWB anchor '{anchor_id}': possible NLOS "
                      f"(residual={innovation:.2f}m)")
            R *= 10.0  # heavily inflate
            nlos_detected = True

        # Update anchor state
        anchor.last_range = range_m
        anchor.last_t = t
        anchor.range_history.append(range_m)
        anchor.accepted_count += 1
        self._measurement_count += 1

        return {
            "type": "UWB_RANGE",
            "anchor_id": anchor_id,
            "anchor_pos": anchor.position.copy(),
            "range_m": range_m,
            "pred_range": pred_range,
            "innovation": np.array([innovation]),
            "H": H,
            "R": R,
            "nlos": nlos_detected,
        }

    def trilaterate(self, current_pos: np.ndarray,
                    t_now: float,
                    max_age: float = 1.0) -> Optional[np.ndarray]:
        """
        Least-squares position estimate from multiple recent ranges.

        Uses iterative Gauss-Newton starting from current_pos.

        Args:
            current_pos: (3,) initial position guess (NED).
            t_now: Current time for staleness filtering.
            max_age: Max age of range measurements (seconds).

        Returns:
            (3,) NED position estimate, or None if insufficient anchors.
        """
        if not self._enabled:
            return None

        # Collect recent ranges
        valid_anchors = []
        for anchor in self._anchors.values():
            if anchor.staleness(t_now) <= max_age and anchor.last_range > 0:
                valid_anchors.append(anchor)

        if len(valid_anchors) < self.MIN_ANCHORS_FOR_TRILATERATION:
            return None

        # Gauss-Newton iteration
        pos_est = current_pos.copy()
        for iteration in range(10):
            residuals = []
            jacobian_rows = []

            for anchor in valid_anchors:
                diff = anchor.position - pos_est
                pred_range = np.linalg.norm(diff)
                if pred_range < 0.01:
                    continue

                residuals.append(anchor.last_range - pred_range)
                jacobian_rows.append(-diff / pred_range)

            if len(residuals) < self.MIN_ANCHORS_FOR_TRILATERATION:
                return None

            J = np.array(jacobian_rows)         # (N, 3)
            r = np.array(residuals)              # (N,)

            # Normal equations: (J^T J) dx = J^T r
            JtJ = J.T @ J
            try:
                dx = np.linalg.solve(JtJ, J.T @ r)
            except np.linalg.LinAlgError:
                return None

            pos_est += dx

            if np.linalg.norm(dx) < 0.001:
                break

        log.debug(f"UWB trilateration: {len(valid_anchors)} anchors, "
                  f"result={pos_est}")
        return pos_est

    def get_status(self) -> dict:
        """Return status of all anchors."""
        return {
            "enabled": self._enabled,
            "anchor_count": len(self._anchors),
            "total_measurements": self._measurement_count,
            "anchors": {
                aid: {
                    "position": a.position.tolist(),
                    "last_range": a.last_range,
                    "accepted": a.accepted_count,
                    "rejected": a.rejected_count,
                }
                for aid, a in self._anchors.items()
            },
        }
