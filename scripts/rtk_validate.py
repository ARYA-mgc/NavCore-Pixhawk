#!/usr/bin/env python3
# RTK Validator.
# Making sure the filter matches reality.

"""RTK Ground Truth Flight Validation.

Replays simulated (or real) sensor data through the ESKF, compares
against RTK ground truth, and generates accuracy metrics.

Usage:
    # Generate sim data first:
    python scripts/generate_rtk_sim.py --trajectory circle --duration 60

    # Then validate:
    python scripts/rtk_validate.py --data-dir sim_data
"""

import sys
import os
import csv
import json
import math
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.eskf import ESKF, EKFHealth
from utils.noise import IMUNoiseParams


def load_csv(path: str) -> list:
    """Load CSV file as list of dicts with float values."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def merge_sensor_timeline(imu, gps, baro, mag):
    """Merge all sensor events into a single sorted timeline."""
    events = []
    for row in imu:
        events.append(("imu", row["time_s"], row))
    for row in gps:
        events.append(("gps", row["time_s"], row))
    for row in baro:
        events.append(("baro", row["time_s"], row))
    for row in mag:
        events.append(("mag", row["time_s"], row))

    events.sort(key=lambda e: e[1])
    return events


def run_eskf_replay(events, noise):
    """Replay sensor events through ESKF and collect state history."""
    eskf = ESKF(noise)

    # Initialize from first 50 IMU samples
    init_accel = []
    init_mag = []
    initialized = False
    last_imu_t = None

    state_log = []

    for etype, t, data in events:
        if not initialized:
            if etype == "imu":
                accel = np.array([data["ax"], data["ay"], data["az"]])
                init_accel.append(accel)
            elif etype == "mag":
                mag = np.array([data["mx"], data["my"], data["mz"]])
                init_mag.append(mag)

            if len(init_accel) >= 50 and len(init_mag) >= 10:
                success = eskf.initialize_from_sensors(
                    np.array(init_accel), np.array(init_mag))
                if success:
                    initialized = True
                    last_imu_t = t
            continue

        if etype == "imu":
            accel = np.array([data["ax"], data["ay"], data["az"]])
            gyro = np.array([data["gx"], data["gy"], data["gz"]])
            dt = t - last_imu_t if last_imu_t else 0.01
            dt = max(0.001, min(dt, 0.1))
            last_imu_t = t

            eskf.predict(accel, gyro, dt)

            # ZUPT detection
            if abs(np.linalg.norm(accel) - 9.80665) < 0.3 and np.linalg.norm(gyro) < 0.02:
                eskf.update_zupt()

            # Log state
            state = eskf.state
            state_log.append({
                "t": t,
                "px": state["pos"][0], "py": state["pos"][1],
                "pz": state["pos"][2],
                "vx": state["vel"][0], "vy": state["vel"][1],
                "vz": state["vel"][2],
                "health": eskf.health.name,
                "z_cov": eskf.P[2, 2],
                "p_trace": np.trace(eskf.P),
                "baro_bias": eskf._baro_bias,
            })

        elif etype == "baro":
            eskf.update_baro(data["alt_m"])

        elif etype == "mag":
            mag = np.array([data["mx"], data["my"], data["mz"]])
            mag_norm = np.linalg.norm(mag)
            yaw = math.atan2(-mag[1], mag[0])
            eskf.update_mag(yaw, mag_norm=mag_norm, t_now=t)

        elif etype == "gps":
            # Fuse GPS as NED position directly (sim data is already NED)
            z = np.array([data["north_m"], data["east_m"], data["down_m"]])
            z_pred = eskf.x[0:3]
            H = np.zeros((3, 15))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 2] = 1.0
            hdop = data.get("hdop", 1.0)
            gps_std = 2.5 * hdop
            R = np.eye(3) * (gps_std ** 2)
            R[2, 2] *= 4.0
            eskf.update_external(z, z_pred, H, R, source="gps")

    return state_log


def compute_errors(state_log, gt_data):
    """Compute position errors between ESKF output and ground truth."""
    gt_times = np.array([g["time_s"] for g in gt_data])
    gt_pos = np.array([[g["x_m"], g["y_m"], g["z_m"]] for g in gt_data])

    errors = []
    for s in state_log:
        idx = np.argmin(np.abs(gt_times - s["t"]))
        if abs(gt_times[idx] - s["t"]) > 0.05:
            continue

        est = np.array([s["px"], s["py"], s["pz"]])
        gt = gt_pos[idx]
        err = np.linalg.norm(est - gt)
        err_h = np.linalg.norm(est[:2] - gt[:2])
        err_v = abs(est[2] - gt[2])

        errors.append({
            "t": s["t"],
            "error_3d": err,
            "error_h": err_h,
            "error_v": err_v,
            "health": s["health"],
        })

    return errors


def print_report(errors, duration):
    """Print the validation report."""
    if not errors:
        print("ERROR: No matched poses found!")
        return

    e3d = np.array([e["error_3d"] for e in errors])
    eh = np.array([e["error_h"] for e in errors])
    ev = np.array([e["error_v"] for e in errors])

    # Find convergence time (first time health == HEALTHY)
    conv_t = None
    for e in errors:
        if e["health"] == "HEALTHY":
            conv_t = e["t"]
            break

    # After-convergence errors only
    if conv_t is not None:
        conv_mask = np.array([e["t"] > conv_t for e in errors])
        e3d_conv = e3d[conv_mask] if np.any(conv_mask) else e3d
    else:
        e3d_conv = e3d

    print()
    print("=" * 64)
    print("  RTK GROUND TRUTH VALIDATION RESULTS")
    print("=" * 64)
    print()
    print(f"  Duration          : {duration:.1f} s")
    print(f"  Matched poses     : {len(errors)}")
    print(f"  Convergence time  : {conv_t:.1f}s" if conv_t else
          "  Convergence time  : NEVER")
    print()
    print("   Overall (full flight) ")
    print(f"  3D RMSE           : {np.sqrt(np.mean(e3d**2)):.4f} m")
    print(f"  3D Mean           : {np.mean(e3d):.4f} m")
    print(f"  3D Max            : {np.max(e3d):.4f} m")
    print(f"  Horizontal RMSE   : {np.sqrt(np.mean(eh**2)):.4f} m")
    print(f"  Vertical RMSE     : {np.sqrt(np.mean(ev**2)):.4f} m")
    print()
    if conv_t is not None and len(e3d_conv) > 0:
        print("   After convergence ")
        print(f"  3D RMSE           : {np.sqrt(np.mean(e3d_conv**2)):.4f} m")
        print(f"  3D Mean           : {np.mean(e3d_conv):.4f} m")
        print(f"  3D Max            : {np.max(e3d_conv):.4f} m")
        print()

    # Drift rate (error growth per second in last 25%)
    n_last = max(1, len(errors) // 4)
    t_last = np.array([e["t"] for e in errors[-n_last:]])
    e_last = e3d[-n_last:]
    if len(t_last) > 2:
        dt_range = t_last[-1] - t_last[0]
        if dt_range > 0:
            drift = (e_last[-1] - e_last[0]) / dt_range
            print(f"  Drift rate (last 25%): {drift:.4f} m/s")

    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="RTK ground truth validation for ESKF")
    parser.add_argument("--data-dir", default="sim_data",
                        help="Directory with sensor CSV files")
    args = parser.parse_args()

    d = args.data_dir
    required = ["imu_log.csv", "baro_log.csv", "mag_log.csv",
                "rtk_ground_truth.csv"]
    for f in required:
        if not os.path.exists(os.path.join(d, f)):
            print(f"ERROR: Missing {f} in {d}/")
            print("Run generate_rtk_sim.py first.")
            sys.exit(1)

    print(f"Loading data from {d}/...")
    imu = load_csv(os.path.join(d, "imu_log.csv"))
    baro = load_csv(os.path.join(d, "baro_log.csv"))
    mag = load_csv(os.path.join(d, "mag_log.csv"))
    gt = load_csv(os.path.join(d, "rtk_ground_truth.csv"))

    # GPS is optional (may have been outage-only)
    gps_path = os.path.join(d, "gps_log.csv")
    gps = load_csv(gps_path) if os.path.exists(gps_path) else []

    print(f"  IMU: {len(imu)}, GPS: {len(gps)}, Baro: {len(baro)}, "
          f"Mag: {len(mag)}, GT: {len(gt)}")

    # Merge and replay
    events = merge_sensor_timeline(imu, gps, baro, mag)
    print(f"  Total events: {len(events)}")
    print("Replaying through ESKF...")

    noise = IMUNoiseParams()
    state_log = run_eskf_replay(events, noise)
    print(f"  ESKF produced {len(state_log)} state estimates")

    # Compute errors
    errors = compute_errors(state_log, gt)
    duration = gt[-1]["time_s"] - gt[0]["time_s"] if gt else 0

    print_report(errors, duration)

    # Save detailed error log
    error_path = os.path.join(d, "validation_errors.csv")
    if errors:
        with open(error_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=errors[0].keys())
            writer.writeheader()
            writer.writerows(errors)
        print(f"\nDetailed errors saved to {error_path}")


if __name__ == "__main__":
    main()
