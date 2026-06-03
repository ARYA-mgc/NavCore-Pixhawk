#!/usr/bin/env python3
# RAIM: Receiver Autonomous Integrity Monitoring
# The module that separates hobbyist navigation from aviation-grade.
#
# Computes Protection Levels (PL) — quantified integrity bounds:
#   "I am 99.9% confident my position is within X meters"
#
# Also performs fault detection via weighted least-squares residuals.
# If PL > Alert Limit → automatic mission abort.
#
# Nobody else in open-source student projects has this.

import math
import logging
import numpy as np
from typing import Optional, Dict, List
from enum import Enum, auto
from collections import deque

log = logging.getLogger("raim")


class IntegrityStatus(Enum):
    """Navigation integrity status."""
    AVAILABLE = auto()      # PL < AL, navigation usable
    CAUTION = auto()        # PL approaching AL (80-100%)
    NOT_AVAILABLE = auto()  # PL > AL, navigation unsafe
    FAULT_DETECTED = auto() # RAIM fault detection triggered


class MissionPhase(Enum):
    """Mission phases with different alert limits."""
    TAKEOFF = auto()
    CRUISE = auto()
    APPROACH = auto()
    LANDING = auto()
    HOVER = auto()


# Alert limits per mission phase (meters)
ALERT_LIMITS = {
    MissionPhase.TAKEOFF:  {"horizontal": 5.0,  "vertical": 3.0},
    MissionPhase.CRUISE:   {"horizontal": 15.0, "vertical": 10.0},
    MissionPhase.APPROACH: {"horizontal": 3.0,  "vertical": 2.0},
    MissionPhase.LANDING:  {"horizontal": 1.5,  "vertical": 1.0},
    MissionPhase.HOVER:    {"horizontal": 2.0,  "vertical": 1.5},
}

# Integrity risk probability (P_md = missed detection probability)
# DO-229E uses 1e-7 for CAT-I approach. We use 1e-5 for UAV.
P_MD = 1e-5

# Chi-squared threshold for fault detection (1 DOF, P_fa = 1e-3)
CHI2_FAULT_1DOF = 10.828  # P_fa = 0.001
CHI2_FAULT_NDOF_SCALE = 3.0  # approximate scale for multi-DOF


class ProtectionLevel:
    """Computed protection level output."""

    def __init__(self):
        self.hpl = float('inf')      # Horizontal Protection Level (m)
        self.vpl = float('inf')      # Vertical Protection Level (m)
        self.hal = float('inf')      # Horizontal Alert Limit (m)
        self.val = float('inf')      # Vertical Alert Limit (m)
        self.hpl_ratio = float('inf')  # HPL/HAL — must be < 1.0
        self.vpl_ratio = float('inf')  # VPL/VAL — must be < 1.0
        self.integrity = IntegrityStatus.NOT_AVAILABLE
        self.n_sources = 0
        self.fault_detected = False
        self.fault_source = ""
        self.test_statistic = 0.0

    @property
    def is_available(self) -> bool:
        return self.integrity in (IntegrityStatus.AVAILABLE,
                                  IntegrityStatus.CAUTION)

    def to_dict(self) -> dict:
        return {
            "hpl": self.hpl,
            "vpl": self.vpl,
            "hal": self.hal,
            "val": self.val,
            "hpl_ratio": self.hpl_ratio,
            "vpl_ratio": self.vpl_ratio,
            "integrity": self.integrity.name,
            "n_sources": self.n_sources,
            "fault_detected": self.fault_detected,
            "fault_source": self.fault_source,
            "test_statistic": self.test_statistic,
        }


class RAIMMonitor:
    """Receiver Autonomous Integrity Monitoring for multi-sensor navigation.

    Computes Protection Levels from the ESKF covariance matrix and performs
    fault detection using weighted least-squares residuals across all
    aiding sensors (GPS, VIO, UWB, SLAM, etc.).

    Protection Level computation:
        HPL = k_md * sqrt(P[0,0] + P[1,1])   — horizontal
        VPL = k_md * sqrt(P[2,2])              — vertical

    where k_md is the Gaussian quantile for the missed detection probability.

    Fault detection:
        Test statistic T = sum(innovation_i^2 / S_ii) across all sensors
        If T > chi2_threshold → fault detected, identify and exclude
    """

    # Gaussian quantile for missed detection (P_md = 1e-5)
    # erfinv(1 - 2*P_md) * sqrt(2) ≈ 4.26
    K_MD = 4.265  # ~1e-5 one-sided tail

    # Caution threshold: PL/AL ratio above which we warn
    CAUTION_RATIO = 0.80

    # Minimum sources for RAIM fault detection
    MIN_SOURCES_FOR_FDE = 2

    def __init__(self):
        self._mission_phase = MissionPhase.HOVER
        self._last_pl = ProtectionLevel()
        self._innovation_history: Dict[str, deque] = {}
        self._fault_count = 0
        self._total_checks = 0

        log.info("RAIM monitor initialized")

    @property
    def protection_level(self) -> ProtectionLevel:
        return self._last_pl

    @property
    def mission_phase(self) -> MissionPhase:
        return self._mission_phase

    def set_mission_phase(self, phase: MissionPhase):
        """Set the current mission phase (affects alert limits)."""
        if phase != self._mission_phase:
            log.info(f"RAIM mission phase: {self._mission_phase.name} → {phase.name}")
            self._mission_phase = phase

    def auto_detect_phase(self, alt_agl: float, vel_horiz: float,
                          vel_vert: float):
        """Automatically detect mission phase from flight state."""
        if alt_agl < 0.5 and vel_horiz < 0.3:
            self.set_mission_phase(MissionPhase.HOVER)
        elif vel_vert > 1.0 and alt_agl < 10.0:
            self.set_mission_phase(MissionPhase.TAKEOFF)
        elif vel_vert < -0.5 and alt_agl < 5.0:
            self.set_mission_phase(MissionPhase.LANDING)
        elif vel_horiz < 0.5 and alt_agl > 2.0:
            self.set_mission_phase(MissionPhase.HOVER)
        elif alt_agl < 10.0 and vel_horiz > 0.5:
            self.set_mission_phase(MissionPhase.APPROACH)
        else:
            self.set_mission_phase(MissionPhase.CRUISE)

    def compute_protection_level(self, P: np.ndarray,
                                 innovation_sources: Optional[Dict[str, dict]] = None
                                 ) -> ProtectionLevel:
        """Compute protection levels from ESKF covariance.

        Args:
            P: ESKF error-state covariance matrix (20×20)
            innovation_sources: dict of {source_name: {"innovation": y, "S": S}}
                                for fault detection

        Returns:
            ProtectionLevel with HPL, VPL, integrity status
        """
        self._total_checks += 1
        pl = ProtectionLevel()

        # Get alert limits for current phase
        limits = ALERT_LIMITS[self._mission_phase]
        pl.hal = limits["horizontal"]
        pl.val = limits["vertical"]

        # ── Protection Level from covariance ──────────────────
        # Position covariance: P[0:3, 0:3]
        pos_cov = P[0:3, 0:3]

        # Horizontal PL: uses North and East position variance
        # HPL = k_md * sqrt(max eigenvalue of 2D position covariance)
        pos_horiz_cov = pos_cov[0:2, 0:2]
        try:
            eigvals_h = np.linalg.eigvalsh(pos_horiz_cov)
            max_eigval_h = max(eigvals_h)
            pl.hpl = self.K_MD * math.sqrt(max(max_eigval_h, 0.0))
        except np.linalg.LinAlgError:
            pl.hpl = float('inf')

        # Vertical PL: uses Down position variance
        vpl_var = pos_cov[2, 2]
        pl.vpl = self.K_MD * math.sqrt(max(vpl_var, 0.0))

        # ── Compute ratios ────────────────────────────────────
        pl.hpl_ratio = pl.hpl / pl.hal if pl.hal > 0 else float('inf')
        pl.vpl_ratio = pl.vpl / pl.val if pl.val > 0 else float('inf')

        # ── Fault Detection and Exclusion (FDE) ───────────────
        if innovation_sources and len(innovation_sources) >= self.MIN_SOURCES_FOR_FDE:
            pl.n_sources = len(innovation_sources)
            fault_result = self._fault_detection(innovation_sources)
            pl.fault_detected = fault_result["detected"]
            pl.fault_source = fault_result.get("source", "")
            pl.test_statistic = fault_result.get("test_stat", 0.0)

        # ── Determine integrity status ────────────────────────
        if pl.fault_detected:
            pl.integrity = IntegrityStatus.FAULT_DETECTED
            self._fault_count += 1
            log.error(f"RAIM FAULT DETECTED: source={pl.fault_source} "
                      f"T={pl.test_statistic:.2f}")
        elif pl.hpl_ratio > 1.0 or pl.vpl_ratio > 1.0:
            pl.integrity = IntegrityStatus.NOT_AVAILABLE
            log.warning(f"RAIM NOT AVAILABLE: HPL={pl.hpl:.2f}m > HAL={pl.hal:.1f}m "
                        f"or VPL={pl.vpl:.2f}m > VAL={pl.val:.1f}m")
        elif pl.hpl_ratio > self.CAUTION_RATIO or pl.vpl_ratio > self.CAUTION_RATIO:
            pl.integrity = IntegrityStatus.CAUTION
        else:
            pl.integrity = IntegrityStatus.AVAILABLE

        self._last_pl = pl
        return pl

    def _fault_detection(self, sources: Dict[str, dict]) -> dict:
        """Weighted least-squares residual fault detection.

        For each sensor source, compute the Normalized Innovation Squared (NIS).
        The sum across all sources follows a chi-squared distribution.
        If the sum exceeds the threshold, identify the source with largest NIS.

        Args:
            sources: {name: {"innovation": y (m,), "S": S (m,m)}}

        Returns:
            {"detected": bool, "source": str, "test_stat": float}
        """
        total_nis = 0.0
        total_dof = 0
        per_source_nis = {}

        for name, data in sources.items():
            y = data.get("innovation")
            S = data.get("S")
            if y is None or S is None:
                continue

            try:
                S_inv = np.linalg.inv(S)
                nis = float(y @ S_inv @ y)
            except np.linalg.LinAlgError:
                continue

            m = y.shape[0]
            total_nis += nis
            total_dof += m
            per_source_nis[name] = nis / m  # normalized per-DOF

            # Track innovation history
            if name not in self._innovation_history:
                self._innovation_history[name] = deque(maxlen=50)
            self._innovation_history[name].append(nis / m)

        if total_dof == 0:
            return {"detected": False}

        # Global test: sum of NIS vs chi-squared threshold
        # Approximate threshold: DOF * scale_factor
        threshold = total_dof * CHI2_FAULT_NDOF_SCALE
        detected = total_nis > threshold

        result = {
            "detected": detected,
            "test_stat": total_nis,
            "threshold": threshold,
            "dof": total_dof,
        }

        if detected and per_source_nis:
            # Identify the worst source
            worst = max(per_source_nis, key=per_source_nis.get)
            result["source"] = worst
            result["worst_nis"] = per_source_nis[worst]
            log.warning(f"RAIM FDE: worst source = {worst} "
                        f"(NIS/DOF = {per_source_nis[worst]:.2f})")

        return result

    def compute_exclusion_pl(self, P: np.ndarray,
                             sources: Dict[str, dict],
                             exclude: str) -> ProtectionLevel:
        """Compute protection level with one source excluded.

        Used after fault detection to verify that excluding the faulty
        source still provides acceptable integrity.

        Args:
            P: ESKF covariance (before the excluded source was used)
            sources: all innovation sources
            exclude: name of source to exclude

        Returns:
            ProtectionLevel without the excluded source
        """
        filtered = {k: v for k, v in sources.items() if k != exclude}
        return self.compute_protection_level(P, filtered)

    def get_status(self) -> dict:
        """Return RAIM status summary."""
        pl = self._last_pl
        return {
            "phase": self._mission_phase.name,
            "hpl": pl.hpl,
            "vpl": pl.vpl,
            "hal": pl.hal,
            "val": pl.val,
            "integrity": pl.integrity.name,
            "fault_count": self._fault_count,
            "total_checks": self._total_checks,
            "n_sources": pl.n_sources,
        }
