#!/usr/bin/env python3
"""
imu_noise_params.py
===================
Sensor noise configuration for Pixhawk Cube Orange.
Port of imu_noise_params.m

Sensors on Cube Orange:
  Primary IMU  : ICM-42688-P  (accel + gyro)
  Secondary IMU: ICM-20649    (accel + gyro, vibration-damped)
  Barometer    : MS5611-01BA03
  Magnetometer : RM3100  (external, typical)

All values from datasheets + ArduPilot calibration typical values.
Tune these in config/noise_params.yaml for your specific unit.
"""

import yaml
import os
import logging

log = logging.getLogger("imu_noise_params")

# Default config path
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "config", "noise_params.yaml"
)


class IMUNoiseParams:
    """
    Holds noise standard deviations for all sensors.
    Can be loaded from YAML or used with hardcoded defaults.
    Validates all parameters against physical bounds.
    """

    # Physical bounds for validation
    _BOUNDS = {
        "accel_std":      (0.001, 2.0),
        "accel_bias_std": (0.0001, 1.0),
        "accel_bias_tau": (1.0, 10000.0),
        "accel_bias_limit": (0.1, 10.0),
        "gyro_std":       (0.0001, 0.5),
        "gyro_bias_std":  (0.00001, 0.1),
        "gyro_bias_tau":  (1.0, 10000.0),
        "gyro_bias_limit": (0.01, 1.0),
        "baro_std":       (0.01, 5.0),
        "mag_std":        (0.001, 0.5),
    }

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        # ── IMU (ICM-42688-P on Cube Orange) ───────────────────
        # Accelerometer noise density: 70 µg/√Hz  → at 100 Hz BW:
        self.accel_std        = 0.05    # m/s²     (σ of white noise)
        self.accel_bias_std   = 0.02    # m/s²     (slowly-varying bias)
        self.accel_bias_tau   = 300.0   # s        (bias correlation time)
        self.accel_bias_limit = 2.0     # m/s²     (max bias magnitude)

        # Gyroscope noise density: 0.0028 °/s/√Hz
        self.gyro_std         = 0.005   # rad/s
        self.gyro_bias_std    = 0.001   # rad/s
        self.gyro_bias_tau    = 300.0   # s
        self.gyro_bias_limit  = 0.1     # rad/s

        # ── Barometer (MS5611) ─────────────────────────────────
        # Altitude noise ≈ 0.1-0.3 m RMS under vibration
        self.baro_std         = 0.30    # m

        # ── Magnetometer (RM3100) ──────────────────────────────
        # Yaw noise after hard/soft-iron calibration
        self.mag_std          = 0.02    # rad  (~1.1°)

        # ── GPS (when available, for reference) ────────────────
        self.gps_pos_std      = 2.5     # m   (CEP50 typical)
        self.gps_vel_std      = 0.1     # m/s

        # ── Load from YAML if it exists ────────────────────────
        self._load_yaml(config_path)
        self._validate()

    def _load_yaml(self, path: str):
        if not os.path.isfile(path):
            log.debug(f"No noise config at {path} — using defaults")
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)

            known_sections = {"imu", "baro", "mag", "gps"}
            unknown = set(cfg.keys()) - known_sections
            if unknown:
                log.warning(f"Unknown config sections ignored: {unknown}")

            imu  = cfg.get("imu",  {})
            baro = cfg.get("baro", {})
            mag  = cfg.get("mag",  {})

            self.accel_std      = imu.get("accel_std",      self.accel_std)
            self.accel_bias_std = imu.get("accel_bias_std", self.accel_bias_std)
            self.accel_bias_tau = imu.get("accel_bias_tau", self.accel_bias_tau)
            self.accel_bias_limit = imu.get("accel_bias_limit", self.accel_bias_limit)
            self.gyro_std       = imu.get("gyro_std",       self.gyro_std)
            self.gyro_bias_std  = imu.get("gyro_bias_std",  self.gyro_bias_std)
            self.gyro_bias_tau  = imu.get("gyro_bias_tau",  self.gyro_bias_tau)
            self.gyro_bias_limit = imu.get("gyro_bias_limit", self.gyro_bias_limit)
            self.baro_std       = baro.get("std",           self.baro_std)
            self.mag_std        = mag.get("std",            self.mag_std)

            log.info(f"Noise params loaded from {path}")
        except Exception as e:
            log.warning(f"Could not parse {path}: {e} — using defaults")

    def _validate(self):
        """Validate all parameters against physical bounds."""
        errors = []
        for param, (lo, hi) in self._BOUNDS.items():
            val = getattr(self, param, None)
            if val is None:
                continue
            if val <= 0:
                errors.append(f"{param}={val} must be > 0")
            elif val < lo or val > hi:
                log.warning(f"Noise param {param}={val} outside typical "
                            f"range [{lo}, {hi}]")
        if errors:
            raise ValueError("Invalid noise params: " + "; ".join(errors))

    def summary(self) -> str:
        return (
            f"IMU Noise Params (Cube Orange)\n"
            f"  Accel σ   : {self.accel_std:.4f} m/s²\n"
            f"  Accel bias: {self.accel_bias_std:.4f} m/s² "
            f"(tau={self.accel_bias_tau:.0f}s, limit={self.accel_bias_limit:.1f})\n"
            f"  Gyro  σ   : {self.gyro_std:.5f} rad/s\n"
            f"  Gyro  bias: {self.gyro_bias_std:.5f} rad/s "
            f"(tau={self.gyro_bias_tau:.0f}s, limit={self.gyro_bias_limit:.2f})\n"
            f"  Baro  σ   : {self.baro_std:.3f} m\n"
            f"  Mag   σ   : {self.mag_std:.4f} rad\n"
        )
