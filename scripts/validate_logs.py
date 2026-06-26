#!/usr/bin/env python3
# Log validator.
# Checking if the flight data is garbage.

"""Log Validator — pre-analysis synchronization and integrity check.

Validates that all recorded sensor logs are synchronized, have no
critical gaps, and match in time span before running analysis.
Without this, misaligned logs produce incorrect RMSE numbers.

Usage:
    python scripts/validate_logs.py --data-dir flight_data/20260602_143000/
"""

import sys
import os
import csv
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


#  Validation Result 

class ValidationResult:
    """Result of a single validation check."""

    def __init__(self, name: str, passed: bool, message: str,
                 severity: str = "ERROR"):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # ERROR, WARNING, INFO


class LogValidationReport:
    """Aggregated validation report."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.streams: Dict[str, dict] = {}

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "ERROR")

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results
                   if not r.passed and r.severity == "WARNING")

    def add(self, result: ValidationResult):
        self.results.append(result)

    def print_report(self):
        print("\n" + "=" * 64)
        print("  LOG VALIDATION REPORT")
        print("=" * 64)

        # Stream summary table
        if self.streams:
            print(f"\n  {'Stream':<22} {'Samples':>8} {'Duration':>10} "
                  f"{'Rate':>8} {'Gaps':>6}")
            print(f"  {''*22} {''*8} {''*10} {''*8} {''*6}")
            for name, info in self.streams.items():
                print(f"  {name:<22} {info['count']:>8} "
                      f"{info['duration']:>9.1f}s "
                      f"{info['rate']:>7.1f}Hz "
                      f"{info['gaps']:>6}")

        # Validation results
        print(f"\n  {'Check':<40} {'Result':>8}")
        print(f"  {''*40} {''*8}")
        for r in self.results:
            icon = "✓ PASS" if r.passed else (
                "✗ FAIL" if r.severity == "ERROR" else "⚠ WARN")
            print(f"  {r.name:<40} {icon:>8}")
            if not r.passed:
                print(f"    → {r.message}")

        status = "PASSED" if self.passed else "FAILED"
        color = "" if self.passed else ""
        print(f"\n  Overall: {status}")
        if self.warnings > 0:
            print(f"  Warnings: {self.warnings}")
        print("=" * 64)


#  Core Validation Functions 

def load_timestamps(filepath: str) -> np.ndarray:
    """Load just the time_s column from a CSV file."""
    times = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                times.append(float(row["time_s"]))
            except (KeyError, ValueError):
                continue
    return np.array(times)


def analyze_stream(filepath: str, name: str) -> dict:
    """Analyze a single sensor stream for timing properties."""
    times = load_timestamps(filepath)
    if len(times) < 2:
        return {
            "count": len(times), "duration": 0, "rate": 0,
            "gaps": 0, "max_gap_s": 0, "mean_dt": 0,
            "t_start": times[0] if len(times) > 0 else 0,
            "t_end": times[-1] if len(times) > 0 else 0,
            "times": times,
        }

    dt = np.diff(times)
    duration = times[-1] - times[0]
    rate = len(times) / max(duration, 0.001)

    # Expected dt from observed rate
    expected_dt = 1.0 / rate if rate > 0 else 0.1
    # Gap = dt > 3× expected
    gap_threshold = max(3.0 * expected_dt, 0.5)
    gaps = int(np.sum(dt > gap_threshold))

    return {
        "count": len(times),
        "duration": duration,
        "rate": rate,
        "gaps": gaps,
        "max_gap_s": float(np.max(dt)) if len(dt) > 0 else 0,
        "mean_dt": float(np.mean(dt)) if len(dt) > 0 else 0,
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
        "dt_std": float(np.std(dt)) if len(dt) > 0 else 0,
        "times": times,
    }


def validate_logs(data_dir: str) -> LogValidationReport:
    """Run all validation checks on a flight data directory.

    Checks:
        1. Required files exist
        2. All streams have enough samples
        3. Time spans overlap sufficiently
        4. No critical gaps in IMU stream
        5. RTK ground truth has enough RTK_FIXED samples
        6. Timestamps are monotonically increasing
        7. Time alignment between streams is consistent
    """
    report = LogValidationReport()

    #  Check 1: Required files 
    required_files = {
        "imu_log.csv": "IMU",
        "baro_log.csv": "Barometer",
        "mag_log.csv": "Magnetometer",
        "rtk_ground_truth.csv": "RTK Ground Truth",
    }
    optional_files = {
        "gps_log.csv": "GPS",
        "eskf_state.csv": "ESKF State",
    }

    available_streams = {}

    for filename, label in {**required_files, **optional_files}.items():
        filepath = os.path.join(data_dir, filename)
        exists = os.path.exists(filepath)
        is_required = filename in required_files

        if exists:
            available_streams[filename] = filepath
            report.add(ValidationResult(
                f"File exists: {filename}",
                True, ""))
        else:
            report.add(ValidationResult(
                f"File exists: {filename}",
                not is_required,
                f"{'Required' if is_required else 'Optional'} "
                f"file missing: {filename}",
                severity="ERROR" if is_required else "WARNING"))

    if not report.passed:
        return report

    #  Check 2: Analyze each stream 
    stream_info = {}
    for filename, filepath in available_streams.items():
        name = filename.replace(".csv", "")
        info = analyze_stream(filepath, name)
        stream_info[name] = info
        report.streams[name] = info

        # Minimum samples
        min_samples = 10
        report.add(ValidationResult(
            f"Sufficient samples: {name}",
            info["count"] >= min_samples,
            f"Only {info['count']} samples (need ≥{min_samples})",
            severity="ERROR"))

    #  Check 3: Time span overlap 
    if "imu_log" in stream_info and "rtk_ground_truth" in stream_info:
        imu = stream_info["imu_log"]
        rtk = stream_info["rtk_ground_truth"]

        overlap_start = max(imu["t_start"], rtk["t_start"])
        overlap_end = min(imu["t_end"], rtk["t_end"])
        overlap = max(0, overlap_end - overlap_start)

        min_overlap = 5.0  # at least 5 seconds of overlap
        report.add(ValidationResult(
            "Time span overlap (IMU ∩ RTK)",
            overlap >= min_overlap,
            f"Only {overlap:.1f}s overlap (need ≥{min_overlap:.0f}s). "
            f"IMU: [{imu['t_start']:.1f}, {imu['t_end']:.1f}], "
            f"RTK: [{rtk['t_start']:.1f}, {rtk['t_end']:.1f}]",
            severity="ERROR"))

        # Time offset between streams
        time_offset = abs(imu["t_start"] - rtk["t_start"])
        report.add(ValidationResult(
            "Stream time alignment",
            time_offset < 2.0,
            f"Start time offset: {time_offset:.2f}s between IMU and RTK. "
            f"Possible clock drift or late start.",
            severity="WARNING" if time_offset < 10.0 else "ERROR"))

    #  Check 4: IMU gaps 
    if "imu_log" in stream_info:
        imu = stream_info["imu_log"]
        report.add(ValidationResult(
            "IMU stream continuity",
            imu["gaps"] == 0,
            f"{imu['gaps']} gaps detected (max gap: {imu['max_gap_s']:.3f}s). "
            f"EKF prediction accuracy degrades during gaps.",
            severity="WARNING"))

        # IMU rate check
        expected_imu_hz = 50.0  # minimum acceptable
        report.add(ValidationResult(
            f"IMU rate ≥ {expected_imu_hz:.0f} Hz",
            imu["rate"] >= expected_imu_hz,
            f"IMU rate is {imu['rate']:.1f} Hz (expected ≥{expected_imu_hz})",
            severity="WARNING"))

    #  Check 5: RTK quality 
    rtk_path = os.path.join(data_dir, "rtk_ground_truth.csv")
    if os.path.exists(rtk_path):
        rtk_fixed = 0
        rtk_total = 0
        with open(rtk_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rtk_total += 1
                if "fix_type" in row:
                    try:
                        if int(float(row["fix_type"])) >= 5:
                            rtk_fixed += 1
                    except ValueError:
                        pass

        if rtk_total > 0:
            pct = 100.0 * rtk_fixed / rtk_total
            report.add(ValidationResult(
                "RTK FIXED quality",
                pct > 50.0,
                f"Only {pct:.1f}% of RTK fixes are FIXED "
                f"({rtk_fixed}/{rtk_total}). "
                f"Ground truth accuracy may be degraded.",
                severity="WARNING" if pct > 20.0 else "ERROR"))

    #  Check 6: Monotonic timestamps 
    for name, info in stream_info.items():
        times = info["times"]
        if len(times) > 1:
            dt = np.diff(times)
            n_backwards = int(np.sum(dt < 0))
            report.add(ValidationResult(
                f"Monotonic timestamps: {name}",
                n_backwards == 0,
                f"{n_backwards} backwards jumps detected",
                severity="ERROR"))

    #  Check 7: Baro/Mag rate consistency 
    for name, expected_min in [("baro_log", 5.0), ("mag_log", 3.0)]:
        if name in stream_info:
            info = stream_info[name]
            report.add(ValidationResult(
                f"{name} rate ≥ {expected_min:.0f} Hz",
                info["rate"] >= expected_min,
                f"Rate is {info['rate']:.1f} Hz",
                severity="WARNING"))

    return report


#  CLI Entry Point 

def main():
    parser = argparse.ArgumentParser(
        description="Validate flight data logs before analysis")
    parser.add_argument("--data-dir", required=True,
                        help="Flight data directory")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Directory not found: {args.data_dir}")
        sys.exit(1)

    report = validate_logs(args.data_dir)
    report.print_report()

    if not report.passed:
        print("\n⚠ Log validation FAILED. Fix the above errors "
              "before running analysis.")
        sys.exit(1)
    else:
        print("\n✓ Logs are valid. Ready for analysis:")
        print(f"  python scripts/analyze_flight.py "
              f"--data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
