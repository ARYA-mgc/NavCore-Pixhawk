#!/usr/bin/env python3
# The panic button coordinator.

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional

log = logging.getLogger("fault_manager")


class FlightMode(Enum):
    """INS operating mode."""
    NOMINAL    = auto()   # All sensors healthy, full ESKF
    DEGRADED   = auto()   # Some sensors missing, reduced accuracy
    FAILSAFE   = auto()   # Critical sensor loss, dead-reckoning only
    EMERGENCY  = auto()   # Estimator diverged, request land/disarm


class SensorStatus(Enum):
    """Per-sensor health state."""
    ACTIVE    = auto()
    STALE     = auto()   # No data for > timeout
    REJECTED  = auto()   # Data arriving but failing quality checks
    OFFLINE   = auto()   # Explicitly disabled or hardware fault


@dataclass
class SensorHealth:
    """Tracks individual sensor liveness."""
    name: str
    status: SensorStatus = SensorStatus.OFFLINE
    last_update_t: float = 0.0
    timeout_s: float = 2.0
    dropout_count: int = 0
    total_samples: int = 0
    rejected_samples: int = 0

    def mark_active(self, t: float):
        """Record a successful sensor update."""
        self.status = SensorStatus.ACTIVE
        self.last_update_t = t
        self.total_samples += 1

    def mark_rejected(self, t: float):
        """Record a rejected sensor sample."""
        self.rejected_samples += 1
        self.last_update_t = t

    def check_staleness(self, t_now: float):
        """Check if sensor has gone stale."""
        if self.status == SensorStatus.OFFLINE:
            return
        if t_now - self.last_update_t > self.timeout_s:
            if self.status == SensorStatus.ACTIVE:
                self.dropout_count += 1
                log.warning(f"Sensor {self.name} dropout #{self.dropout_count}")
            self.status = SensorStatus.STALE

    @property
    def rejection_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.rejected_samples / self.total_samples


class FaultManager:
    """
    High-level fault supervisor.

    Monitors sensor health, decides operating mode, and manages
    EKF reset / recovery sequences.
    """

    # Hysteresis: require N consecutive good cycles before recovery
    RECOVERY_THRESHOLD = 50
    # Maximum dropouts before declaring sensor offline
    MAX_DROPOUT_BEFORE_OFFLINE = 5

    def __init__(self):
        self._mode = FlightMode.NOMINAL
        self._prev_mode = FlightMode.NOMINAL
        self._mode_entry_t = 0.0
        self._recovery_counter = 0
        self._ekf_reset_count = 0

        # Sensor health tracking
        self.sensors: Dict[str, SensorHealth] = {
            "imu":    SensorHealth("imu",    timeout_s=0.1),   # 100 Hz expected
            "baro":   SensorHealth("baro",   timeout_s=1.0),   # 1-10 Hz
            "mag":    SensorHealth("mag",    timeout_s=2.0),   # 1-5 Hz
            "flow":   SensorHealth("flow",   timeout_s=1.0),   # optional
            "gps":    SensorHealth("gps",    timeout_s=5.0),   # optional
        }

    @property
    def mode(self) -> FlightMode:
        return self._mode

    @property
    def mode_name(self) -> str:
        return self._mode.name

    def report_sensor_update(self, sensor_name: str, t: float,
                             rejected: bool = False):
        """Called by main loop when a sensor delivers data."""
        if sensor_name not in self.sensors:
            return

        sh = self.sensors[sensor_name]
        if rejected:
            sh.mark_rejected(t)
        else:
            sh.mark_active(t)

    def update(self, t_now: float, ekf_healthy: bool,
               safety_ok: bool) -> FlightMode:
        """
        Evaluate system health and determine operating mode.

        Args:
            t_now: Current timestamp.
            ekf_healthy: Whether the ESKF reports HEALTHY.
            safety_ok: Whether SafetyMonitor says no faults.

        Returns:
            Current FlightMode after evaluation.
        """
        # Check sensor staleness
        for sh in self.sensors.values():
            sh.check_staleness(t_now)

        # Mark sensors offline after too many dropouts
        for sh in self.sensors.values():
            if sh.dropout_count > self.MAX_DROPOUT_BEFORE_OFFLINE:
                if sh.status != SensorStatus.OFFLINE:
                    log.error(f"Sensor {sh.name} declared OFFLINE "
                              f"after {sh.dropout_count} dropouts")
                    sh.status = SensorStatus.OFFLINE

        # Count active critical sensors
        imu_ok  = self.sensors["imu"].status == SensorStatus.ACTIVE
        baro_ok = self.sensors["baro"].status in (SensorStatus.ACTIVE,
                                                   SensorStatus.STALE)
        mag_ok  = self.sensors["mag"].status in (SensorStatus.ACTIVE,
                                                  SensorStatus.STALE)

        # Determine target mode
        target_mode = self._evaluate_mode(
            imu_ok, baro_ok, mag_ok, ekf_healthy, safety_ok
        )

        # Apply mode transitions with hysteresis
        if target_mode.value > self._mode.value:
            # Escalation is immediate
            self._transition(target_mode, t_now)
        elif target_mode.value < self._mode.value:
            # Recovery requires sustained good state
            self._recovery_counter += 1
            if self._recovery_counter >= self.RECOVERY_THRESHOLD:
                self._transition(target_mode, t_now)
                self._recovery_counter = 0
        else:
            self._recovery_counter = 0

        return self._mode

    def _evaluate_mode(self, imu_ok: bool, baro_ok: bool, mag_ok: bool,
                       ekf_healthy: bool, safety_ok: bool) -> FlightMode:
        """Pure logic: given sensor states, determine target mode."""
        # EMERGENCY: no IMU or estimator diverged
        if not imu_ok:
            return FlightMode.EMERGENCY

        if not ekf_healthy and not safety_ok:
            return FlightMode.EMERGENCY

        # FAILSAFE: IMU only, no aiding sensors
        if not baro_ok and not mag_ok:
            return FlightMode.FAILSAFE

        # DEGRADED: missing one aiding sensor
        if not baro_ok or not mag_ok or not ekf_healthy:
            return FlightMode.DEGRADED

        # NOMINAL: everything good
        return FlightMode.NOMINAL

    def _transition(self, new_mode: FlightMode, t: float):
        """Execute a mode transition."""
        if new_mode == self._mode:
            return

        self._prev_mode = self._mode
        self._mode = new_mode
        self._mode_entry_t = t

        severity = {
            FlightMode.NOMINAL:   "INFO",
            FlightMode.DEGRADED:  "WARNING",
            FlightMode.FAILSAFE:  "ERROR",
            FlightMode.EMERGENCY: "CRITICAL",
        }

        msg = (f"FAULT MANAGER: {self._prev_mode.name} → {new_mode.name} "
               f"at t={t:.2f}s")

        level = getattr(logging, severity.get(new_mode, "INFO"))
        log.log(level, msg)

    def should_reset_ekf(self) -> bool:
        """
        Determine if the ESKF should be reset.

        Returns True once per EMERGENCY entry to attempt recovery.
        Subsequent calls return False until mode changes again.
        """
        if self._mode == FlightMode.EMERGENCY:
            if self._ekf_reset_count == 0 or self._prev_mode != FlightMode.EMERGENCY:
                self._ekf_reset_count += 1
                log.critical(f"EKF RESET requested (#{self._ekf_reset_count})")
                return True
        return False

    def get_status_summary(self) -> dict:
        """Return a snapshot of all sensor health and system mode."""
        return {
            "mode": self._mode.name,
            "sensors": {
                name: {
                    "status": sh.status.name,
                    "dropouts": sh.dropout_count,
                    "rejection_rate": f"{sh.rejection_rate:.1%}",
                }
                for name, sh in self.sensors.items()
            },
            "ekf_resets": self._ekf_reset_count,
        }
