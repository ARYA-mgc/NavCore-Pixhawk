#!/usr/bin/env python3
"""Post-Flight Analysis — RTK Ground Truth Validation Report.

Comprehensive analysis tool that unifies rtk_validate.py + gt_eval.py
to produce APE/RPE metrics, flight phase analysis, drift rate computation,
convergence time detection, trajectory overlay plots, and a markdown report.

Usage:
    # Analyze pre-recorded flight (uses eskf_state.csv)
    python scripts/analyze_flight.py --data-dir flight_data/20260602_143000/

    # Re-run ESKF offline and compare
    python scripts/analyze_flight.py --data-dir flight_data/20260602_143000/ --replay

    # Compare two flights
    python scripts/analyze_flight.py --data-dir flight1/ --compare flight2/
"""

import sys
import os
import csv
import json
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.gt_eval import (
    Pose, load_ground_truth_csv, align_timestamps,
    compute_ape, compute_rpe, umeyama_alignment
)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless RPi4
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── Flight Phase Detection ─────────────────────────────────────

class FlightPhase:
    GROUND    = "GROUND"
    TAKEOFF   = "TAKEOFF"
    CRUISE    = "CRUISE"
    HOVER     = "HOVER"
    MANEUVER  = "MANEUVER"
    LANDING   = "LANDING"


@dataclass
class PhaseSegment:
    phase: str
    t_start: float
    t_end: float
    indices: list


def detect_flight_phases(times: np.ndarray, pos: np.ndarray,
                         vel: np.ndarray) -> List[PhaseSegment]:
    """Detect flight phases from velocity and altitude profiles.

    Phases:
        GROUND:   speed < 0.3 m/s AND alt < 1m
        TAKEOFF:  alt increasing, vz < -0.3 m/s (NED: climbing)
        CRUISE:   stable alt, horiz speed > 1 m/s
        HOVER:    speed < 0.5 m/s, stable alt
        MANEUVER: high acceleration or angular rate
        LANDING:  descending toward ground
    """
    n = len(times)
    if n < 2:
        return [PhaseSegment(FlightPhase.GROUND, times[0], times[-1],
                             list(range(n)))]

    phases = []
    current_phase = FlightPhase.GROUND
    phase_start_idx = 0

    for i in range(n):
        speed_h = np.linalg.norm(vel[i, :2])  # horizontal speed
        speed_v = vel[i, 2]                     # vertical (NED: neg = up)
        alt_agl = -pos[i, 2]                    # NED → AGL

        # Determine phase at this timestep
        if alt_agl < 1.0 and speed_h < 0.3:
            phase = FlightPhase.GROUND
        elif speed_v < -0.3 and alt_agl < 5.0:
            phase = FlightPhase.TAKEOFF
        elif speed_v > 0.3 and alt_agl < 3.0:
            phase = FlightPhase.LANDING
        elif speed_h < 0.5 and abs(speed_v) < 0.3:
            phase = FlightPhase.HOVER
        elif speed_h > 1.0:
            phase = FlightPhase.CRUISE
        else:
            phase = FlightPhase.HOVER  # default to hover if unclear

        # Phase transition
        if phase != current_phase:
            if i > phase_start_idx:
                phases.append(PhaseSegment(
                    current_phase, times[phase_start_idx], times[i - 1],
                    list(range(phase_start_idx, i))))
            current_phase = phase
            phase_start_idx = i

    # Final segment
    if phase_start_idx < n:
        phases.append(PhaseSegment(
            current_phase, times[phase_start_idx], times[-1],
            list(range(phase_start_idx, n))))

    # Merge very short segments (< 1 second) into neighbors
    merged = []
    for seg in phases:
        if (merged and seg.t_end - seg.t_start < 1.0):
            merged[-1].t_end = seg.t_end
            merged[-1].indices.extend(seg.indices)
        else:
            merged.append(seg)

    return merged


# ── Data Loaders ────────────────────────────────────────────────

def load_csv_data(path: str) -> list:
    """Load CSV as list of dicts with float values (matching rtk_validate.py)."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_eskf_state_csv(path: str) -> List[Pose]:
    """Load pre-recorded ESKF state from eskf_state.csv."""
    poses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                poses.append(Pose(
                    t=float(row["time_s"]),
                    pos=np.array([float(row["px_m"]),
                                  float(row["py_m"]),
                                  float(row["pz_m"])]),
                    quat=np.array([float(row["qw"]),
                                   float(row["qx"]),
                                   float(row["qy"]),
                                   float(row["qz"])]),
                    vel=np.array([float(row["vx_ms"]),
                                  float(row["vy_ms"]),
                                  float(row["vz_ms"])]),
                ))
            except (KeyError, ValueError):
                continue
    return poses


def load_rtk_ground_truth(path: str) -> List[Pose]:
    """Load RTK ground truth CSV (format matching generate_rtk_sim.py)."""
    poses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                poses.append(Pose(
                    t=float(row["time_s"]),
                    pos=np.array([float(row["x_m"]),
                                  float(row["y_m"]),
                                  float(row["z_m"])]),
                    quat=np.array([
                        float(row.get("qw", 1.0)),
                        float(row.get("qx", 0.0)),
                        float(row.get("qy", 0.0)),
                        float(row.get("qz", 0.0)),
                    ]),
                    vel=np.array([
                        float(row.get("vx_mps", 0.0)),
                        float(row.get("vy_mps", 0.0)),
                        float(row.get("vz_mps", 0.0)),
                    ]),
                ))
            except (KeyError, ValueError):
                continue
    return poses


# ── Analysis Engine ─────────────────────────────────────────────

@dataclass
class FlightAnalysis:
    """Complete analysis results."""
    ape: dict
    rpe: dict
    convergence_time_s: Optional[float]
    drift_rate_mps: Optional[float]
    duration_s: float
    n_matched: int
    phases: List[PhaseSegment]
    per_phase_ape: Dict[str, dict]
    rtk_stats: dict
    est_poses: List[Pose]
    gt_poses: List[Pose]
    pairs: List[Tuple[Pose, Pose]]
    errors_3d: np.ndarray
    errors_h: np.ndarray
    errors_v: np.ndarray
    times: np.ndarray


def analyze_flight(data_dir: str, replay: bool = False) -> FlightAnalysis:
    """Run complete flight analysis.

    Args:
        data_dir: path to flight data directory
        replay: if True, replay ESKF from sensor data (like rtk_validate.py)
    """
    # ── Load data ─────────────────────────────────────────────
    rtk_path = os.path.join(data_dir, "rtk_ground_truth.csv")
    gt_poses = load_rtk_ground_truth(rtk_path)

    if replay:
        # Re-run ESKF from sensor data (reuse rtk_validate.py logic)
        est_poses = _run_eskf_replay(data_dir)
    else:
        # Use pre-recorded ESKF state
        eskf_path = os.path.join(data_dir, "eskf_state.csv")
        if os.path.exists(eskf_path):
            est_poses = load_eskf_state_csv(eskf_path)
        else:
            print("No eskf_state.csv found. Use --replay to run ESKF offline.")
            est_poses = _run_eskf_replay(data_dir)

    # ── Time alignment ────────────────────────────────────────
    pairs = align_timestamps(est_poses, gt_poses, max_dt=0.1)

    if len(pairs) < 3:
        raise ValueError(f"Only {len(pairs)} aligned pose pairs — "
                         f"need at least 3. Check time sync.")

    # ── APE / RPE ─────────────────────────────────────────────
    ape = compute_ape(pairs, align=True)
    rpe = compute_rpe(pairs, delta=10)

    # Per-axis errors
    est_pos = np.array([p[0].pos for p in pairs])
    gt_pos = np.array([p[1].pos for p in pairs])

    # Apply Umeyama alignment for consistent error computation
    if len(pairs) >= 3:
        R, t, s = umeyama_alignment(est_pos, gt_pos, with_scale=False)
        est_aligned = (R @ est_pos.T).T + t
    else:
        est_aligned = est_pos

    errors_3d = np.linalg.norm(est_aligned - gt_pos, axis=1)
    errors_h = np.linalg.norm(est_aligned[:, :2] - gt_pos[:, :2], axis=1)
    errors_v = np.abs(est_aligned[:, 2] - gt_pos[:, 2])
    times = np.array([p[0].t for p in pairs])

    # ── Convergence time ──────────────────────────────────────
    conv_time = _detect_convergence(errors_3d, times)

    # ── Drift rate ────────────────────────────────────────────
    drift_rate = _compute_drift_rate(errors_3d, times)

    # ── Flight phases ─────────────────────────────────────────
    gt_times = np.array([p.t for p in gt_poses])
    gt_positions = np.array([p.pos for p in gt_poses])
    gt_velocities = np.array([p.vel for p in gt_poses])
    phases = detect_flight_phases(gt_times, gt_positions, gt_velocities)

    # Per-phase APE
    per_phase_ape = {}
    for seg in phases:
        phase_mask = (times >= seg.t_start) & (times <= seg.t_end)
        if np.sum(phase_mask) >= 3:
            phase_errors = errors_3d[phase_mask]
            per_phase_ape[seg.phase] = {
                "rmse": float(np.sqrt(np.mean(phase_errors ** 2))),
                "mean": float(np.mean(phase_errors)),
                "max": float(np.max(phase_errors)),
                "n_samples": int(np.sum(phase_mask)),
                "duration_s": seg.t_end - seg.t_start,
            }

    # ── RTK quality stats ─────────────────────────────────────
    rtk_data = load_csv_data(rtk_path)
    rtk_stats = _compute_rtk_stats(rtk_data)

    duration = times[-1] - times[0] if len(times) > 0 else 0

    return FlightAnalysis(
        ape=ape, rpe=rpe,
        convergence_time_s=conv_time,
        drift_rate_mps=drift_rate,
        duration_s=duration,
        n_matched=len(pairs),
        phases=phases,
        per_phase_ape=per_phase_ape,
        rtk_stats=rtk_stats,
        est_poses=est_poses,
        gt_poses=gt_poses,
        pairs=pairs,
        errors_3d=errors_3d,
        errors_h=errors_h,
        errors_v=errors_v,
        times=times,
    )


def _run_eskf_replay(data_dir: str) -> List[Pose]:
    """Re-run ESKF from sensor data (imports rtk_validate.py logic)."""
    from core.eskf import ESKF
    from utils.noise import IMUNoiseParams

    # Import rtk_validate functions
    scripts_dir = os.path.dirname(__file__)
    sys.path.insert(0, scripts_dir)
    from rtk_validate import (
        load_csv as rtk_load_csv,
        merge_sensor_timeline,
        run_eskf_replay,
    )

    imu = rtk_load_csv(os.path.join(data_dir, "imu_log.csv"))
    baro = rtk_load_csv(os.path.join(data_dir, "baro_log.csv"))
    mag = rtk_load_csv(os.path.join(data_dir, "mag_log.csv"))

    gps_path = os.path.join(data_dir, "gps_log.csv")
    gps = rtk_load_csv(gps_path) if os.path.exists(gps_path) else []

    events = merge_sensor_timeline(imu, gps, baro, mag)
    noise = IMUNoiseParams()
    state_log = run_eskf_replay(events, noise)

    # Convert to Pose list
    poses = []
    for s in state_log:
        poses.append(Pose(
            t=s["t"],
            pos=np.array([s["px"], s["py"], s["pz"]]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),  # not available from replay
            vel=np.array([s["vx"], s["vy"], s["vz"]]),
        ))
    return poses


def _detect_convergence(errors: np.ndarray, times: np.ndarray,
                        threshold: float = 1.0,
                        window: int = 50) -> Optional[float]:
    """Detect convergence time: first time error stays below threshold.

    Uses a sliding window — error must stay below threshold for
    `window` consecutive samples.
    """
    if len(errors) < window:
        return None

    for i in range(len(errors) - window):
        if np.all(errors[i:i + window] < threshold):
            return float(times[i])

    return None


def _compute_drift_rate(errors: np.ndarray, times: np.ndarray) -> Optional[float]:
    """Compute position drift rate (m/s) from error growth in last 25%."""
    n = len(errors)
    if n < 10:
        return None

    # Use last 25% of data
    n_last = max(5, n // 4)
    t_last = times[-n_last:]
    e_last = errors[-n_last:]

    dt = t_last[-1] - t_last[0]
    if dt < 1.0:
        return None

    # Linear regression for drift rate
    coeffs = np.polyfit(t_last - t_last[0], e_last, 1)
    return float(coeffs[0])  # slope = m/s


def _compute_rtk_stats(rtk_data: list) -> dict:
    """Compute RTK fix quality statistics."""
    if not rtk_data:
        return {"total": 0}

    fix_types = []
    h_accs = []
    n_sats_list = []

    for row in rtk_data:
        if "fix_type" in row:
            try:
                fix_types.append(int(float(row["fix_type"])))
            except (ValueError, TypeError):
                pass
        if "h_acc_m" in row:
            try:
                h_accs.append(float(row["h_acc_m"]))
            except (ValueError, TypeError):
                pass
        if "n_sats" in row:
            try:
                n_sats_list.append(int(float(row["n_sats"])))
            except (ValueError, TypeError):
                pass

    stats = {
        "total": len(rtk_data),
        "rtk_fixed_pct": 100.0 * sum(1 for f in fix_types if f >= 5) / max(len(fix_types), 1),
        "rtk_float_pct": 100.0 * sum(1 for f in fix_types if f == 4) / max(len(fix_types), 1),
    }
    if h_accs:
        stats.update({
            "h_acc_mean_m": float(np.mean(h_accs)),
            "h_acc_p95_m": float(np.percentile(h_accs, 95)),
            "h_acc_max_m": float(np.max(h_accs)),
        })
    if n_sats_list:
        stats.update({
            "sats_mean": float(np.mean(n_sats_list)),
            "sats_min": int(np.min(n_sats_list)),
        })

    return stats


# ── Plotting ────────────────────────────────────────────────────

def generate_plots(analysis: FlightAnalysis, output_dir: str):
    """Generate all validation plots as PNG files."""
    if not HAS_PLOT:
        print("matplotlib not installed — skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    est_pos = np.array([p[0].pos for p in analysis.pairs])
    gt_pos = np.array([p[1].pos for p in analysis.pairs])

    # Apply alignment
    if len(analysis.pairs) >= 3:
        R, t, s = umeyama_alignment(est_pos, gt_pos, with_scale=False)
        est_aligned = (R @ est_pos.T).T + t
    else:
        est_aligned = est_pos

    times = analysis.times - analysis.times[0]  # relative time

    # ── Plot 1: 2D Trajectory Overlay (NE plane) ─────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(gt_pos[:, 1], gt_pos[:, 0], 'b-', linewidth=1.5,
            label='RTK Ground Truth', zorder=3)
    ax.plot(est_aligned[:, 1], est_aligned[:, 0], 'r--', linewidth=1.2,
            label='ESKF Estimate', alpha=0.8, zorder=2)
    ax.plot(gt_pos[0, 1], gt_pos[0, 0], 'g^', markersize=12,
            label='Start', zorder=4)
    ax.plot(gt_pos[-1, 1], gt_pos[-1, 0], 'rs', markersize=12,
            label='End', zorder=4)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Trajectory Overlay — ESKF vs RTK Ground Truth")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_2d.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Per-Axis Position Comparison ──────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ["North (m)", "East (m)", "Down (m)"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(times, gt_pos[:, i], 'b-', linewidth=1, label='RTK Truth')
        ax.plot(times, est_aligned[:, i], 'r--', linewidth=1,
                label='ESKF', alpha=0.8)
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position — ESKF vs RTK Ground Truth", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "position_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: APE Time Series ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, analysis.errors_3d, 'r-', linewidth=0.8, label='3D APE')
    ax.plot(times, analysis.errors_h, 'b-', linewidth=0.6,
            alpha=0.7, label='Horizontal')
    ax.plot(times, analysis.errors_v, 'g-', linewidth=0.6,
            alpha=0.7, label='Vertical')
    ax.axhline(analysis.ape["rmse"], color='k', linestyle='--', linewidth=1,
               label=f'RMSE = {analysis.ape["rmse"]:.3f} m')
    if analysis.convergence_time_s is not None:
        conv_rel = analysis.convergence_time_s - analysis.times[0]
        ax.axvline(conv_rel, color='purple', linestyle=':',
                   label=f'Converged @ {conv_rel:.1f}s')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Absolute Pose Error (APE)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ape_timeseries.png"), dpi=150)
    plt.close(fig)

    # ── Plot 4: Convergence Time Graph ────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Error convergence
    # Rolling average for cleaner view
    window = min(50, len(analysis.errors_3d) // 5)
    if window > 1:
        rolling = np.convolve(analysis.errors_3d,
                              np.ones(window) / window, mode='valid')
        t_roll = times[:len(rolling)]
    else:
        rolling = analysis.errors_3d
        t_roll = times

    ax1.plot(t_roll, rolling, 'r-', linewidth=1.2)
    ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5,
                label='1m threshold')
    if analysis.convergence_time_s is not None:
        conv_rel = analysis.convergence_time_s - analysis.times[0]
        ax1.axvline(conv_rel, color='purple', linestyle='-', linewidth=2,
                    label=f'Converged @ {conv_rel:.1f}s')
    ax1.set_ylabel("3D Error (m, smoothed)")
    ax1.set_title("Convergence Analysis")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Horizontal vs vertical error
    ax2.plot(times, analysis.errors_h, 'b-', linewidth=0.8,
             label='Horizontal error')
    ax2.plot(times, analysis.errors_v, 'g-', linewidth=0.8,
             label='Vertical error')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence.png"), dpi=150)
    plt.close(fig)

    # ── Plot 5: Per-Phase Error Bar Chart ─────────────────────
    if analysis.per_phase_ape:
        fig, ax = plt.subplots(figsize=(10, 5))
        phase_names = list(analysis.per_phase_ape.keys())
        rmses = [analysis.per_phase_ape[p]["rmse"] for p in phase_names]
        means = [analysis.per_phase_ape[p]["mean"] for p in phase_names]
        maxes = [analysis.per_phase_ape[p]["max"] for p in phase_names]

        x = np.arange(len(phase_names))
        width = 0.25

        ax.bar(x - width, rmses, width, label='RMSE', color='#e74c3c')
        ax.bar(x, means, width, label='Mean', color='#3498db')
        ax.bar(x + width, maxes, width, label='Max', color='#95a5a6')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names, rotation=45, ha='right')
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Per-Phase APE Breakdown")
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "phase_errors.png"), dpi=150)
        plt.close(fig)

    print(f"Plots saved to {output_dir}/")


# ── Markdown Report ─────────────────────────────────────────────

def generate_report(analysis: FlightAnalysis, output_dir: str,
                    data_dir: str):
    """Generate comprehensive markdown validation report."""
    report_path = os.path.join(output_dir, "flight_report.md")
    a = analysis

    lines = [
        "# NavCore-Pixhawk — Flight Validation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Data**: `{data_dir}`",
        f"**Duration**: {a.duration_s:.1f} s",
        f"**Matched poses**: {a.n_matched}",
        "",
        "---",
        "",
        "## RMSE Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **3D RMSE** | **{a.ape['rmse']:.4f} m** |",
        f"| 3D Mean | {a.ape['mean']:.4f} m |",
        f"| 3D Median | {a.ape['median']:.4f} m |",
        f"| 3D Max | {a.ape['max']:.4f} m |",
        f"| Horizontal RMSE | {float(np.sqrt(np.mean(a.errors_h**2))):.4f} m |",
        f"| Vertical RMSE | {float(np.sqrt(np.mean(a.errors_v**2))):.4f} m |",
        f"| RPE RMSE (δ=10) | {a.rpe['rmse']:.4f} m |",
        f"| Convergence time | {f'{a.convergence_time_s - a.times[0]:.1f} s' if a.convergence_time_s else 'N/A'} |",
        f"| Drift rate | {f'{a.drift_rate_mps:.4f} m/s' if a.drift_rate_mps else 'N/A'} |",
        "",
        "---",
        "",
        "## Trajectory Overlay",
        "",
        "![Trajectory 2D](trajectory_2d.png)",
        "",
        "---",
        "",
        "## Position Comparison",
        "",
        "![Position Comparison](position_comparison.png)",
        "",
        "---",
        "",
        "## Absolute Pose Error (APE)",
        "",
        "![APE Time Series](ape_timeseries.png)",
        "",
        "---",
        "",
        "## Convergence Analysis",
        "",
        "![Convergence](convergence.png)",
        "",
    ]

    # Per-phase table
    if a.per_phase_ape:
        lines.extend([
            "---",
            "",
            "## Per-Phase Error Breakdown",
            "",
            "| Phase | RMSE (m) | Mean (m) | Max (m) | Duration (s) | Samples |",
            "|-------|----------|----------|---------|--------------|---------|",
        ])
        for phase, metrics in a.per_phase_ape.items():
            lines.append(
                f"| {phase} | {metrics['rmse']:.4f} | "
                f"{metrics['mean']:.4f} | {metrics['max']:.4f} | "
                f"{metrics['duration_s']:.1f} | {metrics['n_samples']} |")
        lines.extend(["", "![Phase Errors](phase_errors.png)", ""])

    # RTK stats
    lines.extend([
        "---",
        "",
        "## RTK Ground Truth Quality",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Total fixes | {a.rtk_stats.get('total', 0)} |",
        f"| RTK FIXED | {a.rtk_stats.get('rtk_fixed_pct', 0):.1f}% |",
        f"| RTK FLOAT | {a.rtk_stats.get('rtk_float_pct', 0):.1f}% |",
    ])
    if "h_acc_mean_m" in a.rtk_stats:
        lines.extend([
            f"| H-accuracy mean | {a.rtk_stats['h_acc_mean_m']:.4f} m |",
            f"| H-accuracy P95 | {a.rtk_stats['h_acc_p95_m']:.4f} m |",
        ])
    if "sats_mean" in a.rtk_stats:
        lines.append(
            f"| Satellites (mean/min) | "
            f"{a.rtk_stats['sats_mean']:.0f} / {a.rtk_stats['sats_min']} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {report_path}")


# ── CLI Entry Point ─────────────────────────────────────────────

def print_console_report(a: FlightAnalysis):
    """Print analysis results to console."""
    print()
    print("=" * 68)
    print("  FLIGHT VALIDATION RESULTS — NavCore-Pixhawk")
    print("=" * 68)
    print()
    print(f"  Duration          : {a.duration_s:.1f} s")
    print(f"  Matched poses     : {a.n_matched}")
    conv = (f"{a.convergence_time_s - a.times[0]:.1f}s"
            if a.convergence_time_s else "NEVER")
    print(f"  Convergence time  : {conv}")
    drift = f"{a.drift_rate_mps:.4f} m/s" if a.drift_rate_mps else "N/A"
    print(f"  Drift rate        : {drift}")

    print()
    print("  ── Absolute Pose Error (APE) ─────────────")
    print(f"  3D RMSE           : {a.ape['rmse']:.4f} m")
    print(f"  3D Mean           : {a.ape['mean']:.4f} m")
    print(f"  3D Max            : {a.ape['max']:.4f} m")
    print(f"  Horizontal RMSE   : "
          f"{float(np.sqrt(np.mean(a.errors_h**2))):.4f} m")
    print(f"  Vertical RMSE     : "
          f"{float(np.sqrt(np.mean(a.errors_v**2))):.4f} m")

    print()
    print("  ── Relative Pose Error (RPE, δ=10) ───────")
    print(f"  RMSE              : {a.rpe['rmse']:.4f} m")
    print(f"  Mean              : {a.rpe['mean']:.4f} m")
    print(f"  Max               : {a.rpe['max']:.4f} m")

    if a.per_phase_ape:
        print()
        print("  ── Per-Phase APE ─────────────────────────")
        for phase, m in a.per_phase_ape.items():
            print(f"  {phase:<12} RMSE={m['rmse']:.4f}m  "
                  f"Mean={m['mean']:.4f}m  Max={m['max']:.4f}m  "
                  f"({m['n_samples']} pts, {m['duration_s']:.1f}s)")

    print()
    print("  ── RTK Quality ───────────────────────────")
    rtk = a.rtk_stats
    print(f"  RTK FIXED         : {rtk.get('rtk_fixed_pct', 0):.1f}%")
    if "h_acc_mean_m" in rtk:
        print(f"  H-accuracy        : mean={rtk['h_acc_mean_m']:.4f}m  "
              f"P95={rtk['h_acc_p95_m']:.4f}m")

    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Post-flight RTK validation analysis")
    parser.add_argument("--data-dir", required=True,
                        help="Flight data directory")
    parser.add_argument("--replay", action="store_true",
                        help="Re-run ESKF from sensor data (offline replay)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: data-dir/analysis)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Directory not found: {args.data_dir}")
        sys.exit(1)

    # Run log validation first
    from validate_logs import validate_logs
    report = validate_logs(args.data_dir)
    report.print_report()
    if not report.passed:
        print("\n⚠ Log validation failed. Fix errors before analysis.")
        sys.exit(1)

    # Run analysis
    print("\nRunning flight analysis...")
    analysis = analyze_flight(args.data_dir, replay=args.replay)

    # Console report
    print_console_report(analysis)

    # Plots and markdown report
    output_dir = args.output or os.path.join(args.data_dir, "analysis")
    if not args.no_plot:
        generate_plots(analysis, output_dir)
    generate_report(analysis, output_dir, args.data_dir)

    # Save error CSV
    error_path = os.path.join(output_dir, "errors.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(error_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "error_3d", "error_h", "error_v"])
        for i in range(len(analysis.times)):
            writer.writerow([
                f"{analysis.times[i]:.4f}",
                f"{analysis.errors_3d[i]:.6f}",
                f"{analysis.errors_h[i]:.6f}",
                f"{analysis.errors_v[i]:.6f}",
            ])
    print(f"Error CSV saved to {error_path}")


if __name__ == "__main__":
    main()
