#!/usr/bin/env python3
# We simulate the drone flying 100 times because we don't want to crash 100 real drones.
# Spoiler: The math wizard usually wins.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import math
import argparse
import numpy as np
from core.eskf import ESKF
from core.dr import DeadReckon
from utils.noise import IMUNoiseParams

DURATION = 30.0
GRAVITY = 9.80665


def generate_trajectory(dt, duration):
    N = int(duration / dt)
    t = np.arange(N) * dt
    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    euler = np.zeros((N, 3))

    for i, ti in enumerate(t):
        if ti < 5:
            pos[i] = [0, 0, -ti * 4]
            vel[i] = [0, 0, -4]
        elif ti < 15:
            s = (ti - 5) / 10 * 2 * math.pi
            pos[i] = [10*math.sin(s), 5*math.sin(2*s), -20]
            euler[i] = [0, 0, s]
        elif ti < 25:
            s = (ti - 15) / 10 * 2 * math.pi
            pos[i] = [15*math.cos(s), 15*math.sin(s), -20 - 5*ti/25]
            euler[i] = [0.1, 0, s]
        else:
            pos[i] = [0, 0, -20 + (ti-25)*4]
            vel[i] = [0, 0, 4]
    return t, pos, vel, euler


def simulate_imu(true_pos, true_vel, true_euler, dt, noise, rng):
    N = len(true_euler)
    accel_arr = np.zeros((N, 3))
    gyro_arr = np.zeros((N, 3))
    g_ned = np.array([0.0, 0.0, GRAVITY])

    for i in range(N):
        phi, theta, psi = true_euler[i]
        a_ned = (true_vel[min(i+1,N-1)] - true_vel[i]) / dt if i < N-1 else np.zeros(3)

        cp, sp = math.cos(phi), math.sin(phi)
        ct, st = math.cos(theta), math.sin(theta)
        cy, sy = math.cos(psi), math.sin(psi)

        R_bn = np.array([
            [ct*cy, sp*st*cy-cp*sy, cp*st*cy+sp*sy],
            [ct*sy, sp*st*sy+cp*cy, cp*st*sy-sp*cy],
            [-st,   sp*ct,          cp*ct         ],
        ])
        f_body = R_bn.T @ (a_ned - g_ned)
        accel_arr[i] = f_body + rng.normal(0, noise.accel_std, 3)

        if i < N-1:
            d_euler = true_euler[i+1] - true_euler[i]
            d_euler = np.array([math.atan2(math.sin(a), math.cos(a)) for a in d_euler])
            eta_dot = d_euler / dt
            T = np.array([
                [1.0, 0.0, -st],
                [0.0, cp, sp*ct],
                [0.0, -sp, cp*ct],
            ])
            gyro_arr[i] = T @ eta_dot + rng.normal(0, noise.gyro_std, 3)

    return accel_arr, gyro_arr


def run_single(dt, seed):
    rng = np.random.default_rng(seed)
    noise = IMUNoiseParams()
    eskf = ESKF(noise)
    dr = DeadReckon(noise)

    # wake up the filter with known attitude
    eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
    eskf._initialized = True

    t_arr, true_pos, true_vel, true_euler = generate_trajectory(dt, DURATION)
    accel_arr, gyro_arr = simulate_imu(true_pos, true_vel, true_euler, dt, noise, rng)

    N = len(t_arr)
    eskf_pos = np.zeros((N, 3))
    dr_pos = np.zeros((N, 3))

    for i in range(N):
        eskf.predict(accel_arr[i], gyro_arr[i], dt)
        dr.update(accel_arr[i], gyro_arr[i], dt)

        if i % max(1, int(0.1/dt)) == 0:
            alt = -true_pos[i, 2] + rng.normal(0, noise.baro_std)
            eskf.update_baro(alt)

        if i % max(1, int(0.02/dt)) == 0:
            yaw = true_euler[i, 2] + rng.normal(0, noise.mag_std)
            eskf.update_mag(yaw)

        eskf_pos[i] = eskf.state["pos"]
        dr_pos[i] = dr.pos.copy()

    eskf_err = eskf_pos - true_pos
    dr_err = dr_pos - true_pos
    eskf_rmse = math.sqrt(np.mean(np.sum(eskf_err**2, axis=1)))
    dr_rmse = math.sqrt(np.mean(np.sum(dr_err**2, axis=1)))

    return eskf_rmse, dr_rmse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--hz", type=int, default=100)
    args = p.parse_args()

    dt = 1.0 / args.hz
    print(f"\n=== Monte Carlo Simulation: {args.runs} runs @ {args.hz} Hz ===\n")

    eskf_rmses = []
    dr_rmses = []
    t0 = time.monotonic()

    for i in range(args.runs):
        e_rmse, d_rmse = run_single(dt, seed=i)
        eskf_rmses.append(e_rmse)
        dr_rmses.append(d_rmse)
        if (i+1) % 10 == 0:
            print(f"  Run {i+1}/{args.runs} completed...")

    elapsed = time.monotonic() - t0
    eskf_rmses = np.array(eskf_rmses)
    dr_rmses = np.array(dr_rmses)

    print(f"\n{'='*55}")
    print(f"  Monte Carlo Results ({args.runs} runs, {elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  ESKF RMSE  mean={np.mean(eskf_rmses):.4f}  "
          f"std={np.std(eskf_rmses):.4f}  "
          f"95th={np.percentile(eskf_rmses, 95):.4f} m")
    print(f"  DR   RMSE  mean={np.mean(dr_rmses):.4f}  "
          f"std={np.std(dr_rmses):.4f}  "
          f"95th={np.percentile(dr_rmses, 95):.4f} m")
    print(f"  Improvement: {(1-np.mean(eskf_rmses)/np.mean(dr_rmses))*100:.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
