#!/usr/bin/env python3
"""
ekf_comparison.py
=================
Compares ESKF (NavCore) state output against ArduPilot EKF3 state.

Reads two trajectory sources:
  1. NavCore ESKF: JSONL structured log.
  2. ArduPilot EKF3: CSV export from MAVLink telemetry
     (columns: time_s, x_m, y_m, z_m, vx, vy, vz, roll, pitch, yaw).

Computes per-axis divergence metrics over time.

Usage:
    python ekf_comparison.py \\
        --eskf logs/ins_structured_*.jsonl \\
        --ekf3 logs/ekf3_export.csv
"""

import csv
import json
import sys
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

log = logging.getLogger("ekf_comparison")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


@dataclass
class StateSnapshot:
    t: float
    pos: np.ndarray   # (3,)
    vel: np.ndarray   # (3,)
    euler: np.ndarray  # (3,) [roll, pitch, yaw] radians


# ── Loaders ──────────────────────────────────────────────────

def load_eskf_jsonl(path: str) -> List[StateSnapshot]:
    """Load NavCore ESKF states from JSONL structured log."""
    states = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") != "STATE":
                    continue
                s = rec["state"]
                states.append(StateSnapshot(
                    t=rec["t"],
                    pos=np.array(s["pos"]),
                    vel=np.array(s["vel"]),
                    euler=np.array(s["euler"])
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    log.info(f"Loaded {len(states)} ESKF snapshots from {path}")
    return states


def load_ekf3_csv(path: str) -> List[StateSnapshot]:
    """
    Load ArduPilot EKF3 state export CSV.
    Columns: time_s, x_m, y_m, z_m, vx, vy, vz, roll, pitch, yaw
    (angles in degrees).
    """
    states = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                states.append(StateSnapshot(
                    t=float(row["time_s"]),
                    pos=np.array([float(row["x_m"]),
                                  float(row["y_m"]),
                                  float(row["z_m"])]),
                    vel=np.array([float(row["vx"]),
                                  float(row["vy"]),
                                  float(row["vz"])]),
                    euler=np.radians([float(row["roll"]),
                                      float(row["pitch"]),
                                      float(row["yaw"])])
                ))
            except (KeyError, ValueError) as e:
                log.warning(f"Skipping malformed EKF3 row: {e}")
                continue

    log.info(f"Loaded {len(states)} EKF3 snapshots from {path}")
    return states


# ── Alignment ────────────────────────────────────────────────

def align_states(eskf: List[StateSnapshot],
                 ekf3: List[StateSnapshot],
                 max_dt: float = 0.05) -> List[Tuple[StateSnapshot, StateSnapshot]]:
    """Align ESKF and EKF3 states by nearest timestamp."""
    pairs = []
    ekf3_times = np.array([s.t for s in ekf3])

    for es in eskf:
        idx = np.argmin(np.abs(ekf3_times - es.t))
        if abs(ekf3_times[idx] - es.t) <= max_dt:
            pairs.append((es, ekf3[idx]))

    log.info(f"Aligned {len(pairs)} state pairs")
    return pairs


# ── Divergence Metrics ──────────────────────────────────────

def compute_divergence(pairs: List[Tuple[StateSnapshot, StateSnapshot]]) -> dict:
    """
    Compute per-axis divergence between ESKF and EKF3.

    Returns:
        Dict with position, velocity, and attitude divergence stats.
    """
    pos_errs = np.array([p[0].pos - p[1].pos for p in pairs])
    vel_errs = np.array([p[0].vel - p[1].vel for p in pairs])

    # Wrap yaw difference to [-pi, pi]
    att_errs = np.array([p[0].euler - p[1].euler for p in pairs])
    att_errs[:, 2] = np.arctan2(np.sin(att_errs[:, 2]),
                                np.cos(att_errs[:, 2]))

    pos_norms = np.linalg.norm(pos_errs, axis=1)
    vel_norms = np.linalg.norm(vel_errs, axis=1)
    att_norms = np.degrees(np.linalg.norm(att_errs, axis=1))

    results = {
        "position": {
            "rmse": float(np.sqrt(np.mean(pos_norms ** 2))),
            "mean": float(np.mean(pos_norms)),
            "max":  float(np.max(pos_norms)),
            "per_axis_rmse": {
                "x": float(np.sqrt(np.mean(pos_errs[:, 0] ** 2))),
                "y": float(np.sqrt(np.mean(pos_errs[:, 1] ** 2))),
                "z": float(np.sqrt(np.mean(pos_errs[:, 2] ** 2))),
            }
        },
        "velocity": {
            "rmse": float(np.sqrt(np.mean(vel_norms ** 2))),
            "mean": float(np.mean(vel_norms)),
            "max":  float(np.max(vel_norms)),
        },
        "attitude_deg": {
            "rmse": float(np.sqrt(np.mean(att_norms ** 2))),
            "mean": float(np.mean(att_norms)),
            "max":  float(np.max(att_norms)),
        },
        "n_pairs": len(pairs),
        "time_span": float(pairs[-1][0].t - pairs[0][0].t) if pairs else 0,
    }

    return results


def print_comparison(results: dict):
    """Pretty-print comparison results."""
    print(f"\n{'='*60}")
    print(f"ESKF vs EKF3 DIVERGENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Aligned pairs : {results['n_pairs']}")
    print(f"  Time span     : {results['time_span']:.1f} s")

    pos = results["position"]
    print(f"\nPosition Divergence:")
    print(f"  RMSE   : {pos['rmse']:.4f} m")
    print(f"  Mean   : {pos['mean']:.4f} m")
    print(f"  Max    : {pos['max']:.4f} m")
    print(f"  Per-axis RMSE: X={pos['per_axis_rmse']['x']:.4f} "
          f"Y={pos['per_axis_rmse']['y']:.4f} "
          f"Z={pos['per_axis_rmse']['z']:.4f} m")

    vel = results["velocity"]
    print(f"\nVelocity Divergence:")
    print(f"  RMSE   : {vel['rmse']:.4f} m/s")
    print(f"  Mean   : {vel['mean']:.4f} m/s")
    print(f"  Max    : {vel['max']:.4f} m/s")

    att = results["attitude_deg"]
    print(f"\nAttitude Divergence:")
    print(f"  RMSE   : {att['rmse']:.2f} deg")
    print(f"  Mean   : {att['mean']:.2f} deg")
    print(f"  Max    : {att['max']:.2f} deg")
    print(f"{'='*60}")


# ── CLI Entry Point ────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s: %(message)s")

    import argparse
    p = argparse.ArgumentParser(description="ESKF vs EKF3 divergence analysis")
    p.add_argument("--eskf", required=True, help="NavCore ESKF JSONL log")
    p.add_argument("--ekf3", required=True, help="ArduPilot EKF3 CSV export")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="Max timestamp difference for alignment (s)")
    args = p.parse_args()

    eskf_states = load_eskf_jsonl(args.eskf)
    ekf3_states = load_ekf3_csv(args.ekf3)

    if not eskf_states or not ekf3_states:
        log.error("No states loaded. Check file formats.")
        sys.exit(1)

    pairs = align_states(eskf_states, ekf3_states, max_dt=args.max_dt)
    if len(pairs) < 2:
        log.error(f"Only {len(pairs)} aligned pairs — need at least 2.")
        sys.exit(1)

    results = compute_divergence(pairs)
    print_comparison(results)
