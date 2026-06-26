#!/usr/bin/env python3
# Ground truth evaluation.
# Did we actually go where we think we went?

import csv
import json
import sys
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

log = logging.getLogger("ground_truth_eval")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


#  Data Structures 

@dataclass
class Pose:
    t: float
    pos: np.ndarray    # (3,) xyz NED meters
    quat: np.ndarray   # (4,) [qw, qx, qy, qz]
    vel: np.ndarray    # (3,) m/s


#  Loaders 

def load_ground_truth_csv(path: str) -> List[Pose]:
    # load the RTK data (the 'right' answer)
    poses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["time_s"])
                pos = np.array([float(row["x_m"]),
                                float(row["y_m"]),
                                float(row["z_m"])])
                quat = np.array([float(row["qw"]),
                                 float(row["qx"]),
                                 float(row["qy"]),
                                 float(row["qz"])])
                vel = np.array([float(row.get("vx_mps", 0)),
                                float(row.get("vy_mps", 0)),
                                float(row.get("vz_mps", 0))])
                poses.append(Pose(t=t, pos=pos, quat=quat, vel=vel))
            except (KeyError, ValueError) as e:
                log.warning(f"Skipping malformed ground-truth row: {e}")
                continue

    log.info(f"Loaded {len(poses)} ground-truth poses from {path}")
    return poses


def load_estimate_jsonl(path: str) -> List[Pose]:
    # load what our filter thought happened
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") != "STATE":
                    continue
                state = rec["state"]
                poses.append(Pose(
                    t=rec["t"],
                    pos=np.array(state["pos"]),
                    quat=np.array(state["quat"]),
                    vel=np.array(state["vel"])
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    log.info(f"Loaded {len(poses)} estimate poses from {path}")
    return poses


#  Time Alignment 

def align_timestamps(est: List[Pose], gt: List[Pose],
                     max_dt: float = 0.05) -> List[Tuple[Pose, Pose]]:
    # Associate estimate poses to ground-truth poses by nearest
    pairs = []
    gt_times = np.array([p.t for p in gt])

    for e in est:
        idx = np.argmin(np.abs(gt_times - e.t))
        dt = abs(gt_times[idx] - e.t)
        if dt <= max_dt:
            pairs.append((e, gt[idx]))

    log.info(f"Aligned {len(pairs)} pose pairs (max_dt={max_dt}s)")
    return pairs


#  Umeyama Alignment 

def umeyama_alignment(src: np.ndarray, dst: np.ndarray,
                      with_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    # Umeyama alignment: find R, t, s such that dst ≈ s*R*src + t.
    assert src.shape == dst.shape
    n, m = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_demean = src - mu_src
    dst_demean = dst - mu_dst

    sigma_src = np.sum(src_demean ** 2) / n
    cov = dst_demean.T @ src_demean / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_src if with_scale else 1.0
    t = mu_dst - s * R @ mu_src

    return R, t, s


#  Error Metrics 

def compute_ape(pairs: List[Tuple[Pose, Pose]],
                align: bool = True) -> dict:
    # Compute Absolute Pose Error (APE).
    est_pos = np.array([p[0].pos for p in pairs])
    gt_pos  = np.array([p[1].pos for p in pairs])

    if align and len(pairs) >= 3:
        R, t, s = umeyama_alignment(est_pos, gt_pos, with_scale=False)
        est_aligned = (R @ est_pos.T).T + t
    else:
        est_aligned = est_pos

    errors = np.linalg.norm(est_aligned - gt_pos, axis=1)

    return {
        "rmse":   float(np.sqrt(np.mean(errors ** 2))),
        "mean":   float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max":    float(np.max(errors)),
        "std":    float(np.std(errors)),
        "errors": errors,
    }


def compute_rpe(pairs: List[Tuple[Pose, Pose]],
                delta: int = 10) -> dict:
    # Compute Relative Pose Error (RPE) at fixed index intervals.
    if len(pairs) <= delta:
        log.warning(f"Not enough pairs for RPE with delta={delta}")
        return {"rmse": 0, "mean": 0, "median": 0, "max": 0, "std": 0}

    errors = []

    for i in range(len(pairs) - delta):
        # Estimate relative motion
        est_rel = pairs[i + delta][0].pos - pairs[i][0].pos
        # Ground truth relative motion
        gt_rel = pairs[i + delta][1].pos - pairs[i][1].pos

        err = np.linalg.norm(est_rel - gt_rel)
        errors.append(err)

    errors = np.array(errors)

    return {
        "rmse":   float(np.sqrt(np.mean(errors ** 2))),
        "mean":   float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max":    float(np.max(errors)),
        "std":    float(np.std(errors)),
    }


def print_results(ape: dict, rpe: dict):
    # print the report card
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nAbsolute Pose Error (APE):")
    print(f"  RMSE   : {ape['rmse']:.4f} m")
    print(f"  Mean   : {ape['mean']:.4f} m")
    print(f"  Median : {ape['median']:.4f} m")
    print(f"  Max    : {ape['max']:.4f} m")
    print(f"  Std    : {ape['std']:.4f} m")
    print(f"\nRelative Pose Error (RPE):")
    print(f"  RMSE   : {rpe['rmse']:.4f} m")
    print(f"  Mean   : {rpe['mean']:.4f} m")
    print(f"  Median : {rpe['median']:.4f} m")
    print(f"  Max    : {rpe['max']:.4f} m")
    print(f"  Std    : {rpe['std']:.4f} m")
    print(f"{'='*60}")


#  CLI Entry Point 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s: %(message)s")

    import argparse
    p = argparse.ArgumentParser(description="Ground truth trajectory evaluation")
    p.add_argument("--est", required=True, help="ESKF estimate JSONL log")
    p.add_argument("--gt",  required=True, help="Ground truth CSV file")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="Max timestamp difference for alignment (s)")
    p.add_argument("--rpe-delta", type=int, default=10,
                   help="Index interval for RPE computation")
    p.add_argument("--no-align", action="store_true",
                   help="Skip Umeyama alignment")
    args = p.parse_args()

    gt_poses  = load_ground_truth_csv(args.gt)
    est_poses = load_estimate_jsonl(args.est)

    if not gt_poses or not est_poses:
        log.error("No poses loaded. Check file formats.")
        sys.exit(1)

    pairs = align_timestamps(est_poses, gt_poses, max_dt=args.max_dt)
    if len(pairs) < 3:
        log.error(f"Only {len(pairs)} aligned pairs — need at least 3.")
        sys.exit(1)

    ape = compute_ape(pairs, align=not args.no_align)
    rpe = compute_rpe(pairs, delta=args.rpe_delta)
    print_results(ape, rpe)
