#!/usr/bin/env python3
# Let's see how fast this baby can run on a potato.
# Basically a drag race for the ESKF code.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import math
import numpy as np
from core.eskf        import ESKF
from core.dr import DeadReckon
from utils.noise      import IMUNoiseParams

# ── simulation parameters ────────────────────────────────────────
DT_LIST   = [0.02, 0.01]   # 50 Hz and 100 Hz
DURATION  = 30.0           # seconds
GRAVITY   = 9.80665

rng = np.random.default_rng(42)


def generate_true_trajectory(dt, duration):
    """
    Simulate a 30-second UAV flight:
      0-5 s   : vertical climb to 20 m
      5-15 s  : figure-8
      15-25 s : banked turn
      25-30 s : descent
    Returns arrays of true pos, vel, euler.
    """
    N    = int(duration / dt)
    t    = np.arange(N) * dt
    pos  = np.zeros((N, 3))
    vel  = np.zeros((N, 3))
    euler = np.zeros((N, 3))

    for i, ti in enumerate(t):
        if ti < 5:
            pos[i]   = [0, 0, -ti * 4]        # climb 4 m/s
            vel[i]   = [0, 0, -4]
            euler[i] = [0, 0, 0]
        elif ti < 15:
            s = (ti - 5) / 10 * 2 * math.pi
            pos[i]   = [10*math.sin(s), 5*math.sin(2*s), -20]
            vel[i]   = [0, 0, 0]
            euler[i] = [0, 0, s]
        elif ti < 25:
            s = (ti - 15) / 10 * 2 * math.pi
            pos[i]   = [15*math.cos(s), 15*math.sin(s), -20 - 5*ti/25]
            vel[i]   = [0, 0, 0]
            euler[i] = [0.1, 0, s]
        else:
            pos[i]   = [0, 0, -20 + (ti-25)*4]  # descend
            vel[i]   = [0, 0, 4]
            euler[i] = [0, 0, 0]
    return t, pos, vel, euler


def simulate_imu(true_pos, true_vel, true_euler, dt, noise: IMUNoiseParams):
    """
    Generate noisy IMU measurements from true trajectory.
    Specific force = R_nb * (a_ned - g_ned)  where g_ned = [0,0,+g]
    """
    N = len(true_euler)
    accel_arr = np.zeros((N, 3))
    gyro_arr  = np.zeros((N, 3))
    g_ned = np.array([0.0, 0.0, GRAVITY])

    for i in range(N):
        phi, theta, psi = true_euler[i]

        # True NED acceleration from velocity finite-diff
        if i < N-1:
            a_ned = (true_vel[i+1] - true_vel[i]) / dt
        else:
            a_ned = np.zeros(3)

        cp, sp = math.cos(phi),   math.sin(phi)
        ct, st = math.cos(theta), math.sin(theta)
        cy, sy = math.cos(psi),   math.sin(psi)

        # R_nb (NED → body) = R_bn^T
        R_bn = np.array([
            [ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy],
            [ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy],
            [-st,    sp*ct,             cp*ct            ],
        ])
        R_nb = R_bn.T

        # Specific force in body frame: f_body = R_nb*(a_ned - g_ned)
        f_body = R_nb @ (a_ned - g_ned)
        accel_arr[i] = f_body + rng.normal(0, noise.accel_std, 3)

        # Attitude rate → gyro
        if i < N-1:
            d_euler = true_euler[i+1] - true_euler[i]
            # wrap
            d_euler = np.array([math.atan2(math.sin(a), math.cos(a)) for a in d_euler])
            eta_dot = d_euler / dt
            # T matrix (euler rates → body rates)
            T = np.array([
                [1.0, 0.0,     -st       ],
                [0.0,  cp,      sp*ct    ],
                [0.0, -sp,      cp*ct    ],
            ])
            gyro_body = T @ eta_dot
        else:
            gyro_body = np.zeros(3)

        gyro_arr[i] = gyro_body + rng.normal(0, noise.gyro_std, 3)

    return accel_arr, gyro_arr


def run_benchmark(dt):
    noise = IMUNoiseParams()
    eskf  = ESKF(noise)
    dr    = DeadReckon(noise)

    # Initialize ESKF with known attitude (simulation starts level, north)
    eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
    eskf._initialized = True

    t_arr, true_pos, true_vel, true_euler = generate_true_trajectory(dt, DURATION)
    accel_arr, gyro_arr = simulate_imu(true_pos, true_vel, true_euler, dt, noise)

    N = len(t_arr)
    eskf_pos = np.zeros((N, 3))
    dr_pos   = np.zeros((N, 3))

    t_start = time.monotonic()

    for i in range(N):
        accel = accel_arr[i]
        gyro  = gyro_arr[i]

        eskf.predict(accel, gyro, dt)
        dr.update(accel, gyro, dt)

        # Baro update every 10 steps (~10 Hz)
        if i % max(1, int(0.1/dt)) == 0:
            alt = -true_pos[i, 2] + rng.normal(0, noise.baro_std)
            eskf.update_baro(alt)

        # Mag update every 2 steps (~50 Hz)
        if i % max(1, int(0.02/dt)) == 0:
            yaw = true_euler[i, 2] + rng.normal(0, noise.mag_std)
            eskf.update_mag(yaw)

        eskf_pos[i] = eskf.state["pos"]
        dr_pos[i]   = dr.pos.copy()

    elapsed = time.monotonic() - t_start
    eff_hz  = N / elapsed

    # ── error analysis ─────────────────────────────────────────
    eskf_err  = eskf_pos - true_pos
    dr_err    = dr_pos   - true_pos
    eskf_rmse = math.sqrt(np.mean(np.sum(eskf_err**2, axis=1)))
    dr_rmse   = math.sqrt(np.mean(np.sum(dr_err**2,  axis=1)))
    improve   = (dr_rmse - eskf_rmse) / dr_rmse * 100

    hz = int(1/dt)
    print(f"\n{'='*50}")
    print(f"  Benchmark @ {hz} Hz  (dt={dt*1000:.0f} ms, {N} steps)")
    print(f"{'='*50}")
    print(f"  Wall time          : {elapsed*1000:.1f} ms")
    print(f"  Effective rate     : {eff_hz:.0f} Hz")
    print(f"  ESKF Pos RMSE      : {eskf_rmse:.3f} m")
    print(f"  DR   Pos RMSE      : {dr_rmse:.3f} m")
    print(f"  Drift improvement  : {improve:.1f} %")
    x_rmse = math.sqrt(np.mean(eskf_err[:,0]**2))
    y_rmse = math.sqrt(np.mean(eskf_err[:,1]**2))
    z_rmse = math.sqrt(np.mean(eskf_err[:,2]**2))
    print(f"  Per-axis RMSE  X={x_rmse:.3f}  Y={y_rmse:.3f}  Z={z_rmse:.3f} m")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("\n=== INS Benchmark (ESKF — Quaternion) ===")
    for dt in DT_LIST:
        run_benchmark(dt)
