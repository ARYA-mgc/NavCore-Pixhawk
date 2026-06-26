#!/usr/bin/env python3
# Safety Monitor.
# The designated adult of the system.

import math
import logging
import numpy as np
from enum import Enum

log = logging.getLogger("safety_monitor")


class SafetyAction(Enum):
    # Safety action types.
    NOMINAL = 0           # All clear — inject normally
    WARN = 1              # Elevated but acceptable — log warning
    LIMIT = 2             # Apply output limiting (clamp)
    DISABLE_INJECTION = 3 # Stop sending position to ArduPilot
    FORCE_DISARM = 4      # Critical — request disarm (emergency disarm)


class SafetyMonitor:
    # Monitors system state for safety violations.
    # Applies configured thresholds to ESKF output.

    def __init__(self,
                 max_horizontal_vel: float = 30.0,
                 max_vertical_vel: float = 15.0,
                 max_tilt_deg: float = 60.0,
                 max_position_jump: float = 5.0,
                 geofence_radius: float = 500.0,
                 geofence_height: float = 120.0):
        # Configurable limits
        self.max_horizontal_vel = max_horizontal_vel   # m/s
        self.max_vertical_vel = max_vertical_vel       # m/s
        self.max_tilt_deg = max_tilt_deg               # degrees
        self.max_position_jump = max_position_jump     # m (per step)
        self.geofence_radius = geofence_radius         # m
        self.geofence_height = geofence_height         # m AGL

        # Hard fault thresholds (non-configurable)
        self._vel_fault = 100.0    # m/s — physically impossible for multirotor
        self._tilt_fault = 80.0    # degrees — beyond recovery
        self._jump_fault = 20.0    # m — EKF has diverged

        # State tracking
        self._last_pos = None
        self._violation_count = 0
        self._consecutive_violations = 0
        self._max_consecutive_for_disable = 10
        self._last_action = SafetyAction.NOMINAL

        # Statistics
        self.stats = {
            "total_checks": 0,
            "warnings": 0,
            "limits": 0,
            "disables": 0,
            "disarms": 0,
        }

    def check(self, pos: np.ndarray, vel: np.ndarray,
              euler_rad: np.ndarray) -> SafetyAction:
        # Check state against limits.
        self.stats["total_checks"] += 1
        violations = []

        # --- Velocity check ---
        horiz_vel = math.sqrt(vel[0]**2 + vel[1]**2)
        vert_vel = abs(vel[2])

        if horiz_vel > self._vel_fault or vert_vel > self._vel_fault:
            violations.append(("FAULT_VEL", SafetyAction.FORCE_DISARM))
        elif horiz_vel > self.max_horizontal_vel:
            violations.append(("HORIZ_VEL", SafetyAction.LIMIT))
        elif vert_vel > self.max_vertical_vel:
            violations.append(("VERT_VEL", SafetyAction.LIMIT))

        # --- Tilt check ---
        roll_deg = abs(math.degrees(euler_rad[0]))
        pitch_deg = abs(math.degrees(euler_rad[1]))
        tilt = math.sqrt(roll_deg**2 + pitch_deg**2)

        if tilt > self._tilt_fault:
            violations.append(("FAULT_TILT", SafetyAction.FORCE_DISARM))
        elif tilt > self.max_tilt_deg:
            violations.append(("TILT", SafetyAction.LIMIT))

        # --- Position jump detection ---
        if self._last_pos is not None:
            jump = np.linalg.norm(pos - self._last_pos)
            if jump > self._jump_fault:
                violations.append(("FAULT_JUMP", SafetyAction.DISABLE_INJECTION))
            elif jump > self.max_position_jump:
                violations.append(("POS_JUMP", SafetyAction.WARN))
        self._last_pos = pos.copy()

        # --- Geofence ---
        horiz_dist = math.sqrt(pos[0]**2 + pos[1]**2)
        if horiz_dist > self.geofence_radius:
            violations.append(("GEOFENCE_H", SafetyAction.LIMIT))
        if abs(pos[2]) > self.geofence_height:
            violations.append(("GEOFENCE_V", SafetyAction.LIMIT))

        # --- NaN/Inf check ---
        if (np.any(np.isnan(pos)) or np.any(np.isnan(vel)) or
                np.any(np.isnan(euler_rad)) or
                np.any(np.isinf(pos)) or np.any(np.isinf(vel))):
            violations.append(("NAN_INF", SafetyAction.DISABLE_INJECTION))

        # --- Determine worst action ---
        if not violations:
            self._consecutive_violations = 0
            self._last_action = SafetyAction.NOMINAL
            return SafetyAction.NOMINAL

        # Take the most severe action
        worst = max(violations, key=lambda v: v[1].value)
        action = worst[1]

        # Track consecutive violations
        self._consecutive_violations += 1
        self._violation_count += 1

        # Escalate on sustained violations
        if (self._consecutive_violations >= self._max_consecutive_for_disable
                and action.value < SafetyAction.DISABLE_INJECTION.value):
            action = SafetyAction.DISABLE_INJECTION
            log.error(f"Safety: escalated to DISABLE after "
                      f"{self._consecutive_violations} consecutive violations")

        # Log
        violation_names = [v[0] for v in violations]
        if action == SafetyAction.FORCE_DISARM:
            log.critical(f"SAFETY FORCE_DISARM: {violation_names}")
            self.stats["disarms"] += 1
        elif action == SafetyAction.DISABLE_INJECTION:
            log.error(f"SAFETY DISABLE: {violation_names}")
            self.stats["disables"] += 1
        elif action == SafetyAction.LIMIT:
            log.warning(f"SAFETY LIMIT: {violation_names}")
            self.stats["limits"] += 1
        else:
            log.debug(f"SAFETY WARN: {violation_names}")
            self.stats["warnings"] += 1

        self._last_action = action
        return action

    @property
    def last_action(self) -> SafetyAction:
        return self._last_action

    @property
    def is_injection_safe(self) -> bool:
        # are we safe to keep using this data?
        return self._last_action.value < SafetyAction.DISABLE_INJECTION.value

    def reset(self):
        # Reset safety monitor.
        self._last_pos = None
        self._consecutive_violations = 0
        self._last_action = SafetyAction.NOMINAL
        log.info("Safety monitor reset")

    def summary(self) -> str:
        return (
            f"Safety Monitor Stats:\n"
            f"  Total checks : {self.stats['total_checks']}\n"
            f"  Warnings     : {self.stats['warnings']}\n"
            f"  Limits       : {self.stats['limits']}\n"
            f"  Disables     : {self.stats['disables']}\n"
            f"  Disarms      : {self.stats['disarms']}\n"
        )
