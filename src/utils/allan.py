#!/usr/bin/env python3
# Allan Variance analysis.
# Staring at sensor noise for hours.

import json
import sys
import os
import math
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger("allan_variance")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def compute_allan_variance(data: np.ndarray, dt: float,
                           max_clusters: int = 100) -> tuple:
    # crunch the numbers on sensor noise characteristics
    N = len(data)
    if N < 10:
        raise ValueError(f"Need at least 10 samples, got {N}")

    # Generate logarithmically spaced cluster sizes
    max_m = N // 2
    ms = np.unique(
        np.logspace(0, np.log10(max_m), min(max_clusters, max_m)).astype(int)
    )
    ms = ms[ms >= 1]

    taus = ms * dt
    avars = np.zeros(len(ms))

    # Overlapping Allan Variance
    for i, m in enumerate(ms):
        # Compute cumulative sum for efficient averaging
        cumsum = np.cumsum(data)
        cumsum = np.insert(cumsum, 0, 0)

        # Overlapping differences
        n_overlaps = N - 2 * m
        if n_overlaps <= 0:
            avars[i] = np.nan
            continue

        # sigma^2(tau) = 1/(2*tau^2*(N-2m)) * sum((x[i+2m] - 2*x[i+m] + x[i])^2)
        diffs = cumsum[2*m:] - 2 * cumsum[m:N - m + 1][:n_overlaps] + cumsum[:n_overlaps]
        avars[i] = np.mean(diffs ** 2) / (2.0 * m * m)

    # Remove NaN entries
    valid = ~np.isnan(avars)
    return taus[valid], avars[valid]


def fit_noise_parameters(taus: np.ndarray, adevs: np.ndarray) -> dict:
    # extract the useful noise params from the allan curve
    log_taus = np.log10(taus)
    log_adevs = np.log10(adevs)

    # White noise: slope = -1/2 region (short tau)
    # ADEV = σ_w / sqrt(tau)  →  log(ADEV) = log(σ_w) - 0.5*log(tau)
    # At tau = 1, ADEV = σ_w
    short_mask = taus < np.median(taus)
    if np.sum(short_mask) >= 2:
        coeffs = np.polyfit(log_taus[short_mask], log_adevs[short_mask], 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        # White noise: extrapolate to tau=1
        white_noise = 10.0 ** (intercept)
    else:
        white_noise = adevs[0]

    # Bias instability: minimum of ADEV curve
    bias_instability = np.min(adevs) * (0.664)  # scaling factor

    return {
        "white_noise": float(white_noise),
        "bias_instability": float(bias_instability),
    }


def analyze_stationary_log(filepath: str,
                           imu_rate_hz: float = 100.0) -> dict:
    # sit the drone on a table and stare at its noise for hours
    dt = 1.0 / imu_rate_hz

    # Parse log
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "STATE":
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    if len(records) < 100:
        raise ValueError(f"Need at least 100 STATE records, got {len(records)}")

    log.info(f"Loaded {len(records)} STATE records from {filepath}")

    # Extract velocity (proxy for accel integration noise)
    vel_x = np.array([r["state"]["vel"][0] for r in records])
    vel_y = np.array([r["state"]["vel"][1] for r in records])
    vel_z = np.array([r["state"]["vel"][2] for r in records])

    # Extract euler (proxy for gyro integration noise)
    euler_r = np.array([r["state"]["euler"][0] for r in records])
    euler_p = np.array([r["state"]["euler"][1] for r in records])
    euler_y = np.array([r["state"]["euler"][2] for r in records])

    results = {}

    # Accel analysis (from velocity noise)
    for axis, data in [("x", vel_x), ("y", vel_y), ("z", vel_z)]:
        taus, avars = compute_allan_variance(data, dt)
        adevs = np.sqrt(avars)
        params = fit_noise_parameters(taus, adevs)
        results[f"accel_{axis}"] = params
        log.info(f"Accel {axis}: white_noise={params['white_noise']:.6f} "
                 f"bias_instability={params['bias_instability']:.6f}")

    # Gyro analysis (from attitude noise)
    for axis, data in [("roll", euler_r), ("pitch", euler_p), ("yaw", euler_y)]:
        taus, avars = compute_allan_variance(data, dt)
        adevs = np.sqrt(avars)
        params = fit_noise_parameters(taus, adevs)
        results[f"gyro_{axis}"] = params
        log.info(f"Gyro {axis}: white_noise={params['white_noise']:.6f} "
                 f"bias_instability={params['bias_instability']:.6f}")

    return results


def generate_tuned_yaml(results: dict, output_path: str = "config/noise_params_tuned.yaml"):
    # spit out a tuned config based on what we measured
    # Average across axes
    accel_wn = np.mean([results[f"accel_{a}"]["white_noise"] for a in "xyz"])
    accel_bi = np.mean([results[f"accel_{a}"]["bias_instability"] for a in "xyz"])
    gyro_wn  = np.mean([results[f"gyro_{a}"]["white_noise"] for a in ["roll", "pitch", "yaw"]])
    gyro_bi  = np.mean([results[f"gyro_{a}"]["bias_instability"] for a in ["roll", "pitch", "yaw"]])

    yaml_content = f"""# config/noise_params_tuned.yaml
# Auto-generated by allan_variance.py
# Source: Allan Variance analysis of stationary log
# 

imu:
  accel_std:      {accel_wn:.6f}     # m/s²  (from Allan Variance white noise fit)
  accel_bias_std: {accel_bi:.6f}     # m/s²  (from Allan Variance bias instability)
  accel_bias_tau: 300.0              # s     (default — tune with longer logs)
  gyro_std:       {gyro_wn:.6f}      # rad/s (from Allan Variance white noise fit)
  gyro_bias_std:  {gyro_bi:.6f}      # rad/s (from Allan Variance bias instability)
  gyro_bias_tau:  300.0              # s     (default — tune with longer logs)

baro:
  std:  0.30                         # m     (unchanged — tune separately)

mag:
  std:  0.02                         # rad   (unchanged — tune separately)
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_content)

    log.info(f"Tuned config written to {output_path}")
    print(f"\n{'='*60}")
    print(f"TUNED NOISE PARAMETERS (Allan Variance)")
    print(f"{'='*60}")
    print(f"  Accel white noise : {accel_wn:.6f} m/s²")
    print(f"  Accel bias inst.  : {accel_bi:.6f} m/s²")
    print(f"  Gyro white noise  : {gyro_wn:.6f} rad/s")
    print(f"  Gyro bias inst.   : {gyro_bi:.6f} rad/s")
    print(f"{'='*60}")
    print(f"Written to: {output_path}")

    return output_path


#  CLI Entry Point 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python allan_variance.py <structured_log.jsonl> [--rate HZ]")
        print("  Computes Allan Variance from a stationary vehicle log.")
        sys.exit(1)

    logfile = sys.argv[1]
    rate = 100.0
    if "--rate" in sys.argv:
        idx = sys.argv.index("--rate")
        rate = float(sys.argv[idx + 1])

    results = analyze_stationary_log(logfile, imu_rate_hz=rate)
    generate_tuned_yaml(results)
