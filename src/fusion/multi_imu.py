#!/usr/bin/env python3
# Multi-IMU voting.
# When three sensors disagree, this finds the liar.

import numpy as np
import logging
from collections import deque

log = logging.getLogger("multi_imu")


class IMUChannel:
    """Tracks state and health of a single IMU channel."""

    def __init__(self, channel_id: int, window_size: int = 50):
        self.channel_id = channel_id
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.t_last = 0.0
        self.alive = False

        # Rolling variance for quality weighting
        self._accel_history = deque(maxlen=window_size)
        self._gyro_history = deque(maxlen=window_size)
        self.accel_var = 0.0
        self.gyro_var = 0.0

        # Health tracking
        self.fault_count = 0
        self.total_count = 0
        self._healthy = True

    def update(self, accel: np.ndarray, gyro: np.ndarray, t: float):
        self.accel = accel.copy()
        self.gyro = gyro.copy()
        self.t_last = t
        self.alive = True
        self.total_count += 1

        self._accel_history.append(accel.copy())
        self._gyro_history.append(gyro.copy())

        if len(self._accel_history) >= 10:
            self.accel_var = float(np.var(
                np.linalg.norm(np.array(self._accel_history), axis=1)))
            self.gyro_var = float(np.var(
                np.linalg.norm(np.array(self._gyro_history), axis=1)))

    @property
    def is_healthy(self) -> bool:
        return self._healthy and self.alive

    def mark_fault(self):
        self.fault_count += 1
        if self.fault_count > 20:
            self._healthy = False
            log.warning(f"IMU channel {self.channel_id} marked UNHEALTHY "
                        f"({self.fault_count} faults)")

    def mark_good(self):
        # Slow recovery — need many good readings to re-enable
        if self.fault_count > 0:
            self.fault_count -= 1
        if self.fault_count <= 0 and not self._healthy:
            self._healthy = True
            log.info(f"IMU channel {self.channel_id} recovered to HEALTHY")


class MultiIMUFusion:
    """Fuses up to 3 IMU sources with median voting + variance weighting.

    Cube Orange has 3 IMUs. This module:
    1. Stores latest reading from each channel
    2. Detects outlier IMUs via median voting
    3. Weights by inverse variance (lower noise = higher trust)
    4. Returns a single fused (accel, gyro) pair
    """

    # Maximum age before an IMU channel is considered stale
    MAX_AGE_S = 0.1   # 100ms

    # Outlier threshold: if an IMU disagrees by this much, flag it
    OUTLIER_THRESHOLD = 2.0  # m/s² for accel, rad/s for gyro

    def __init__(self, n_channels: int = 3):
        self.channels = [IMUChannel(i) for i in range(n_channels)]
        self._fused_accel = np.zeros(3)
        self._fused_gyro = np.zeros(3)
        self._confidence = 0.0
        self._n_active = 0

    def update_imu(self, channel: int, accel: np.ndarray,
                   gyro: np.ndarray, t: float):
        """Feed a raw IMU reading from a specific channel.

        Args:
            channel: IMU index (0=RAW_IMU, 1=SCALED_IMU2, 2=SCALED_IMU3)
            accel: accelerometer reading (m/s², body frame)
            gyro: gyroscope reading (rad/s, body frame)
            t: monotonic timestamp
        """
        if channel < 0 or channel >= len(self.channels):
            return
        self.channels[channel].update(accel, gyro, t)

    def get_fused(self, t_now: float) -> tuple:
        """Returns fused (accel, gyro, confidence).

        confidence: 0.0 (no IMU) to 1.0 (all IMUs agree)
        """
        # Collect alive, non-stale channels
        active = []
        for ch in self.channels:
            if ch.alive and ch.is_healthy and (t_now - ch.t_last) < self.MAX_AGE_S:
                active.append(ch)

        self._n_active = len(active)

        if len(active) == 0:
            # No IMU data — return last known values
            return self._fused_accel, self._fused_gyro, 0.0

        if len(active) == 1:
            # Single IMU — no fusion possible
            self._fused_accel = active[0].accel.copy()
            self._fused_gyro = active[0].gyro.copy()
            self._confidence = 0.5
            return self._fused_accel, self._fused_gyro, self._confidence

        # --- Multi-IMU fusion ---

        accels = np.array([ch.accel for ch in active])
        gyros = np.array([ch.gyro for ch in active])

        # Step 1: Median voting for outlier detection
        median_accel = np.median(accels, axis=0)
        median_gyro = np.median(gyros, axis=0)

        weights = []
        for ch_idx, ch in enumerate(active):
            accel_dev = np.linalg.norm(ch.accel - median_accel)
            gyro_dev = np.linalg.norm(ch.gyro - median_gyro)

            if accel_dev > self.OUTLIER_THRESHOLD or gyro_dev > self.OUTLIER_THRESHOLD:
                ch.mark_fault()
                weights.append(0.0)
                log.debug(f"IMU{ch.channel_id} outlier: "
                          f"accel_dev={accel_dev:.2f} gyro_dev={gyro_dev:.4f}")
            else:
                ch.mark_good()
                # Inverse variance weighting (lower variance = higher trust)
                var = max(ch.accel_var + ch.gyro_var, 1e-6)
                weights.append(1.0 / var)

        weights = np.array(weights)
        weight_sum = np.sum(weights)

        if weight_sum < 1e-10:
            # All outliers — fall back to median
            self._fused_accel = median_accel
            self._fused_gyro = median_gyro
            self._confidence = 0.2
        else:
            # Weighted average
            weights /= weight_sum
            self._fused_accel = np.zeros(3)
            self._fused_gyro = np.zeros(3)
            for i, ch in enumerate(active):
                self._fused_accel += weights[i] * ch.accel
                self._fused_gyro += weights[i] * ch.gyro

            # Confidence: higher when more IMUs agree
            n_good = np.sum(weights > 0)
            self._confidence = float(n_good / len(self.channels))

        return self._fused_accel, self._fused_gyro, self._confidence

    def get_imu_health(self) -> dict:
        """Returns per-channel health status."""
        return {
            f"imu{ch.channel_id}": {
                "alive": ch.alive,
                "healthy": ch.is_healthy,
                "accel_var": ch.accel_var,
                "gyro_var": ch.gyro_var,
                "faults": ch.fault_count,
                "total": ch.total_count,
            }
            for ch in self.channels
        }

    @property
    def n_active(self) -> int:
        return self._n_active

    @property
    def vibration_level(self) -> float:
        """Returns aggregate vibration level (0.0 = calm, 1.0+ = severe).

        Used to drive adaptive process noise scaling (Feature 6).
        """
        active = [ch for ch in self.channels if ch.alive and ch.is_healthy]
        if not active:
            return 0.0
        # Mean accel variance across active channels
        mean_var = np.mean([ch.accel_var for ch in active])
        # Normalize: typical vibration during flight is 0.01-0.5 m/s²²
        # Scale so that 0.1 → 0.5 vibration_level, 1.0 → 5.0
        return float(mean_var * 5.0)
