#!/usr/bin/env python3
# Ground truth evaluation.
# Did we actually go where we think we went?

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import numpy as np
import math

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_csv(path, pos_cols=(1,2,3), time_col=0):
    # slurp up the CSV data
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    t = data[:, time_col]
    pos = data[:, list(pos_cols)]
    return t, pos


def time_align(t_ins, pos_ins, t_truth, pos_truth):
    # Align INS and truth by nearest-neighbor time interpolation.
    matched_ins = []
    matched_truth = []
    matched_t = []

    for i, t in enumerate(t_ins):
        idx = np.argmin(np.abs(t_truth - t))
        dt = abs(t_truth[idx] - t)
        if dt < 0.1:  # max 100ms mismatch
            matched_ins.append(pos_ins[i])
            matched_truth.append(pos_truth[idx])
            matched_t.append(t)

    return (np.array(matched_t), np.array(matched_ins),
            np.array(matched_truth))


def compute_metrics(pos_ins, pos_truth):
    # how badly did we miss the target?
    err = pos_ins - pos_truth

    rmse_total = math.sqrt(np.mean(np.sum(err**2, axis=1)))
    rmse_x = math.sqrt(np.mean(err[:, 0]**2))
    rmse_y = math.sqrt(np.mean(err[:, 1]**2))
    rmse_z = math.sqrt(np.mean(err[:, 2]**2))

    ape = np.sqrt(np.sum(err**2, axis=1))
    ape_mean = np.mean(ape)
    ape_median = np.median(ape)
    ape_max = np.max(ape)

    mean_err = np.mean(err, axis=0)
    max_err = np.max(np.abs(err), axis=0)

    return {
        "rmse_total": rmse_total,
        "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse_z": rmse_z,
        "ape_mean": ape_mean, "ape_median": ape_median, "ape_max": ape_max,
        "mean_error": mean_err, "max_error": max_err,
        "error": err, "ape": ape,
    }


def print_report(metrics):
    print("\n" + "="*50)
    print("  Ground Truth Evaluation Report")
    print("="*50)
    print(f"  Position RMSE (total) : {metrics['rmse_total']:.4f} m")
    print(f"  RMSE X={metrics['rmse_x']:.4f}  "
          f"Y={metrics['rmse_y']:.4f}  Z={metrics['rmse_z']:.4f} m")
    print(f"  APE  mean={metrics['ape_mean']:.4f}  "
          f"median={metrics['ape_median']:.4f}  "
          f"max={metrics['ape_max']:.4f} m")
    print(f"  Mean error: {metrics['mean_error']}")
    print(f"  Max  error: {metrics['max_error']}")
    print("="*50)


def plot_results(t, pos_ins, pos_truth, metrics, output_dir="logs"):
    if not HAS_PLOT:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    err = metrics["error"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X (North)", "Y (East)", "Z (Down)"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, pos_truth[:, i], 'b-', label='Truth', linewidth=1)
        ax.plot(t, pos_ins[:, i], 'r--', label='INS', linewidth=1)
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("INS vs Ground Truth -- Position")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eval_trajectory.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, metrics["ape"], 'r-', linewidth=0.8)
    ax2.axhline(metrics["ape_mean"], color='b', linestyle='--',
                label=f'Mean APE={metrics["ape_mean"]:.3f}m')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("APE (m)")
    ax2.set_title("Absolute Pose Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "eval_ape.png"), dpi=150)

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="INS Ground Truth Evaluation")
    p.add_argument("--ins", required=True, help="INS CSV log path")
    p.add_argument("--truth", required=True, help="Ground truth CSV path")
    p.add_argument("--output", default="logs", help="Output directory")
    args = p.parse_args()

    t_ins, pos_ins = load_csv(args.ins)
    t_truth, pos_truth = load_csv(args.truth)
    t, p_ins, p_truth = time_align(t_ins, pos_ins, t_truth, pos_truth)

    if len(t) < 10:
        print("ERROR: Too few matched points. Check time alignment.")
        sys.exit(1)

    metrics = compute_metrics(p_ins, p_truth)
    print_report(metrics)
    plot_results(t, p_ins, p_truth, metrics, args.output)
