#!/usr/bin/env python3
"""
config_loader.py
================
Strict YAML configuration loader with schema validation, range
checking, and fail-fast on any anomaly.

Usage:
    from config_loader import load_config

    cfg = load_config("config/noise_params.yaml")
    # Raises ConfigError on any schema/range violation.

Design:
    - No silent defaults. If a required field is missing, we fail.
    - Every numeric parameter is bounds-checked against physical limits.
    - Unknown keys are rejected (prevents typo-silencing).
    - Produces a frozen, validated dict — not a mutable object.
"""

import os
import yaml
import logging
from dataclasses import dataclass, fields
from typing import Optional

log = logging.getLogger("config_loader")


class ConfigError(Exception):
    """Raised when configuration is invalid. Non-recoverable."""
    pass


# ── Schema Definition ──────────────────────────────────────────

@dataclass(frozen=True)
class _Bound:
    """Min/max bounds for a single parameter."""
    lo: float
    hi: float
    required: bool = True
    description: str = ""


# Full schema: section -> field -> bounds
_SCHEMA = {
    "imu": {
        "accel_std":        _Bound(0.001, 2.0,    True,  "Accelerometer white noise (m/s^2)"),
        "accel_bias_std":   _Bound(0.0001, 1.0,   True,  "Accelerometer bias instability (m/s^2)"),
        "accel_bias_tau":   _Bound(1.0, 10000.0,  False, "Accel bias correlation time (s)"),
        "accel_bias_limit": _Bound(0.1, 10.0,     False, "Max accel bias clamp (m/s^2)"),
        "gyro_std":         _Bound(0.0001, 0.5,   True,  "Gyroscope white noise (rad/s)"),
        "gyro_bias_std":    _Bound(0.00001, 0.1,  True,  "Gyroscope bias instability (rad/s)"),
        "gyro_bias_tau":    _Bound(1.0, 10000.0,  False, "Gyro bias correlation time (s)"),
        "gyro_bias_limit":  _Bound(0.01, 1.0,     False, "Max gyro bias clamp (rad/s)"),
    },
    "baro": {
        "std":              _Bound(0.01, 5.0,     True,  "Barometric altitude noise (m)"),
    },
    "mag": {
        "std":              _Bound(0.001, 0.5,    True,  "Magnetometer yaw noise (rad)"),
    },
    "gps": {
        "pos_std":          _Bound(0.1, 50.0,     False, "GPS position noise (m)"),
        "vel_std":          _Bound(0.01, 5.0,     False, "GPS velocity noise (m/s)"),
    },
}

_KNOWN_SECTIONS = set(_SCHEMA.keys())


def load_config(path: str) -> dict:
    """
    Load and strictly validate a YAML configuration file.

    Returns a nested dict matching the schema.
    Raises ConfigError on any violation.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}")

    if raw is None or not isinstance(raw, dict):
        raise ConfigError(f"Config file is empty or not a dict: {path}")

    errors = []

    # 1. Reject unknown top-level sections
    unknown_sections = set(raw.keys()) - _KNOWN_SECTIONS
    if unknown_sections:
        errors.append(f"Unknown config sections: {unknown_sections}")

    validated = {}

    for section, field_schema in _SCHEMA.items():
        section_data = raw.get(section, {})
        if section_data is None:
            section_data = {}

        if not isinstance(section_data, dict):
            errors.append(f"Section '{section}' must be a mapping, got {type(section_data).__name__}")
            continue

        # Reject unknown keys within each section
        unknown_keys = set(section_data.keys()) - set(field_schema.keys())
        if unknown_keys:
            errors.append(f"Unknown keys in '{section}': {unknown_keys}")

        validated_section = {}

        for field_name, bound in field_schema.items():
            value = section_data.get(field_name)

            if value is None:
                if bound.required:
                    errors.append(
                        f"Missing required field: {section}.{field_name} "
                        f"({bound.description})"
                    )
                continue

            # Type check
            if not isinstance(value, (int, float)):
                errors.append(
                    f"{section}.{field_name} must be numeric, "
                    f"got {type(value).__name__}: {value}"
                )
                continue

            value = float(value)

            # Positivity
            if value <= 0:
                errors.append(f"{section}.{field_name}={value} must be > 0")
                continue

            # Range check
            if value < bound.lo or value > bound.hi:
                errors.append(
                    f"{section}.{field_name}={value} outside valid range "
                    f"[{bound.lo}, {bound.hi}] — {bound.description}"
                )
                continue

            validated_section[field_name] = value

        validated[section] = validated_section

    if errors:
        msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        log.critical(msg)
        raise ConfigError(msg)

    log.info(f"Config loaded and validated: {path}")
    return validated


def validate_or_die(path: str) -> dict:
    """
    Convenience wrapper. Loads config or prints error and exits.
    Use this in main() entry points for fail-fast behavior.
    """
    try:
        return load_config(path)
    except ConfigError as e:
        log.critical(str(e))
        import sys
        sys.exit(1)
