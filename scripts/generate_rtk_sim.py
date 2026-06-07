#!/usr/bin/env python3
# RTK Simulator.
# Making fake data to test real math.

"""Generate simulated flight data with RTK ground truth.

Outputs realistic sensor logs for ESKF validation:
- IMU at 100 Hz with configurable noise (from noise_params.yaml)
- GPS at 5 Hz with HDOP variation + configurable outages
- Baro at 25 Hz with temperature drift
- Mag at 10 Hz with soft-iron distortion
- RTK ground truth at 100 Hz

Trajectory options: circle, figure-8, waypoints, hover

Usage:
    python scripts/generate_rtk_sim.py --trajectory circle --duration 120
"""

import sys
import os
import csv
import json
import math
import argparse
import numpy as np

# Add src to path for noise params
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def rotation_matrix_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])


def generate_circle_trajectory(duration: float, dt: float,
                                radius: float = 10.0,
                                speed: float = 2.0,
                                altitude: float = 5.0):
    """Generate a circular trajectory in NED."""
    omega = speed / radius  # angular velocity
    t_vec = np.arange(0, duration, dt)
    n = len(t_vec)

    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    quat = np.zeros((n, 4))

    for i, t in enumerate(t_vec):
        theta = omega * t
        # Position (NED)
        pos[i] = [radius * math.cos(theta),
                  radius * math.sin(theta),
                  -altitude]
        # Velocity
        vel[i] = [-radius * omega * math.sin(theta),
                   radius * omega * math.cos(theta),
                   0.0]
        # Acceleration (centripetal)
        acc[i] = [-radius * omega**2 * math.cos(theta),
                  -radius * omega**2 * math.sin(theta),
                  0.0]
        # Yaw = tangent direction
        yaw = theta + math.pi / 2
        quat[i] = euler_to_quat(0.0, 0.0, yaw)

    return t_vec, pos, vel, acc, quat


def generate_figure8_trajectory(duration: float, dt: float,
                                 radius: float = 15.0,
                                 speed: float = 2.0,
                                 altitude: float = 5.0):
    """Generate a figure-8 (lemniscate) trajectory."""
    omega = speed / radius
    t_vec = np.arange(0, duration, dt)
    n = len(t_vec)

    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    quat = np.zeros((n, 4))

    for i, t in enumerate(t_vec):
        theta = omega * t
        # Lemniscate of Bernoulli
        denom = 1 + math.sin(theta)**2
        pos[i] = [radius * math.cos(theta) / denom,
                  radius * math.sin(theta) * math.cos(theta) / denom,
                  -altitude]
        # Numerical velocity
        if i > 0:
            vel[i] = (pos[i] - pos[i-1]) / dt
        if i > 1:
            acc[i] = (vel[i] - vel[i-1]) / dt
        yaw = math.atan2(vel[i, 1], vel[i, 0]) if np.linalg.norm(vel[i, :2]) > 0.1 else 0.0
        quat[i] = euler_to_quat(0.0, 0.0, yaw)

    return t_vec, pos, vel, acc, quat


def generate_hover_trajectory(duration: float, dt: float,
                               altitude: float = 3.0):
    """Generate a hover-in-place trajectory (for ZUPT testing)."""
    t_vec = np.arange(0, duration, dt)
    n = len(t_vec)

    pos = np.zeros((n, 3))
    pos[:, 2] = -altitude
    vel = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))

    return t_vec, pos, vel, acc, quat


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def quat_to_rotation(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


class FlightSimulator:
    """Generate realistic sensor data from a known trajectory."""

    GRAVITY_NED = np.array([0.0, 0.0, 9.80665])

    def __init__(self, accel_std=0.05, gyro_std=0.005,
                 baro_std=0.30, mag_std=0.02,
                 gps_std=2.5, baro_drift_rate=0.001):
        self.accel_std = accel_std
        self.gyro_std = gyro_std
        self.baro_std = baro_std
        self.mag_std = mag_std
        self.gps_std = gps_std
        self.baro_drift_rate = baro_drift_rate  # m/s drift
        self.rng = np.random.default_rng(42)

        # Simulated biases
        self.accel_bias = self.rng.normal(0, 0.01, 3)
        self.gyro_bias = self.rng.normal(0, 0.0005, 3)

        # Earth magnetic field (NED, typical mid-latitude)
        self.mag_field_ned = np.array([0.2, 0.0, 0.4])

    def generate_imu(self, t, acc_body, gyro_body, dt):
        """Generate noisy IMU measurement in body frame."""
        # True specific force: body frame = R^T * (a_ned - g)
        accel = acc_body + self.accel_bias + self.rng.normal(0, self.accel_std, 3)
        gyro = gyro_body + self.gyro_bias + self.rng.normal(0, self.gyro_std, 3)
        return accel, gyro

    def generate_baro(self, t, true_alt):
        """Generate noisy baro altitude with temperature drift."""
        drift = self.baro_drift_rate * t
        return true_alt + drift + self.rng.normal(0, self.baro_std)

    def generate_mag(self, R_body_to_ned, t):
        """Generate noisy mag reading in body frame."""
        mag_body = R_body_to_ned.T @ self.mag_field_ned
        # Add soft-iron distortion (slowly varying)
        distortion = 1.0 + 0.05 * math.sin(0.01 * t)
        mag_body *= distortion
        return mag_body + self.rng.normal(0, self.mag_std * 0.1, 3)

    def generate_gps(self, t, true_pos, hdop=1.0):
        """Generate noisy GPS position."""
        noise = self.rng.normal(0, self.gps_std * hdop, 3)
        noise[2] *= 2.0  # vertical is worse
        return true_pos + noise

    def run(self, t_vec, pos, vel, acc, quat,
            imu_hz=100, gps_hz=5, baro_hz=25, mag_hz=10,
            gps_outage_start=None, gps_outage_end=None):
        """Run full simulation, generating all sensor logs."""
        dt = t_vec[1] - t_vec[0]
        n = len(t_vec)

        imu_log = []
        gps_log = []
        baro_log = []
        mag_log = []
        gt_log = []

        imu_interval = max(1, int(1.0 / (imu_hz * dt)))
        gps_interval = max(1, int(1.0 / (gps_hz * dt)))
        baro_interval = max(1, int(1.0 / (baro_hz * dt)))
        mag_interval = max(1, int(1.0 / (mag_hz * dt)))

        for i in range(n):
            t = t_vec[i]
            R = quat_to_rotation(quat[i])

            # Ground truth (always at full rate)
            gt_log.append({
                "time_s": t,
                "x_m": pos[i, 0], "y_m": pos[i, 1], "z_m": pos[i, 2],
                "vx_mps": vel[i, 0], "vy_mps": vel[i, 1], "vz_mps": vel[i, 2],
                "qw": quat[i, 0], "qx": quat[i, 1],
                "qy": quat[i, 2], "qz": quat[i, 3],
            })

            # IMU (body frame)
            if i % imu_interval == 0:
                # True specific force in body: R^T * (a_ned - g)
                specific_force_ned = acc[i] - self.GRAVITY_NED
                acc_body = R.T @ specific_force_ned

                # Gyro from quaternion rate (numerical)
                if i > 0:
                    q0, q1 = quat[i-1], quat[i]
                    # Simple angular velocity extraction
                    dq = q1 - q0
                    gyro_body = 2.0 * np.array([dq[1], dq[2], dq[3]]) / dt
                else:
                    gyro_body = np.zeros(3)

                accel_meas, gyro_meas = self.generate_imu(t, acc_body, gyro_body, dt)
                imu_log.append({
                    "time_s": t,
                    "ax": accel_meas[0], "ay": accel_meas[1], "az": accel_meas[2],
                    "gx": gyro_meas[0], "gy": gyro_meas[1], "gz": gyro_meas[2],
                })

            # GPS
            if i % gps_interval == 0:
                in_outage = False
                if gps_outage_start and gps_outage_end:
                    in_outage = gps_outage_start <= t <= gps_outage_end

                if not in_outage:
                    hdop = 1.0 + 0.5 * math.sin(0.05 * t)
                    gps_pos = self.generate_gps(t, pos[i], hdop)
                    gps_log.append({
                        "time_s": t,
                        "north_m": gps_pos[0], "east_m": gps_pos[1],
                        "down_m": gps_pos[2],
                        "hdop": hdop,
                    })

            # Baro
            if i % baro_interval == 0:
                baro_alt = self.generate_baro(t, pos[i, 2])
                baro_log.append({
                    "time_s": t,
                    "alt_m": baro_alt,
                })

            # Mag
            if i % mag_interval == 0:
                mag_meas = self.generate_mag(R, t)
                mag_log.append({
                    "time_s": t,
                    "mx": mag_meas[0], "my": mag_meas[1], "mz": mag_meas[2],
                })

        return {
            "imu": imu_log,
            "gps": gps_log,
            "baro": baro_log,
            "mag": mag_log,
            "ground_truth": gt_log,
        }


def write_csv(data: list, path: str):
    if not data:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"  Wrote {len(data)} rows → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated flight data with RTK ground truth")
    parser.add_argument("--trajectory", choices=["circle", "figure8", "hover"],
                        default="circle", help="Trajectory type")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Flight duration (seconds)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Simulation timestep (seconds)")
    parser.add_argument("--radius", type=float, default=10.0,
                        help="Trajectory radius (meters)")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Flight speed (m/s)")
    parser.add_argument("--altitude", type=float, default=5.0,
                        help="Flight altitude AGL (meters)")
    parser.add_argument("--gps-outage-start", type=float, default=None,
                        help="GPS outage start time (seconds)")
    parser.add_argument("--gps-outage-end", type=float, default=None,
                        help="GPS outage end time (seconds)")
    parser.add_argument("--output-dir", default="sim_data",
                        help="Output directory")
    args = parser.parse_args()

    print(f"Generating {args.trajectory} trajectory ({args.duration}s)...")

    # Generate trajectory
    if args.trajectory == "circle":
        t, pos, vel, acc, quat = generate_circle_trajectory(
            args.duration, args.dt, args.radius, args.speed, args.altitude)
    elif args.trajectory == "figure8":
        t, pos, vel, acc, quat = generate_figure8_trajectory(
            args.duration, args.dt, args.radius, args.speed, args.altitude)
    else:
        t, pos, vel, acc, quat = generate_hover_trajectory(
            args.duration, args.dt, args.altitude)

    print(f"  Trajectory: {len(t)} samples at {1.0/args.dt:.0f} Hz")

    # Generate sensor data
    sim = FlightSimulator()
    data = sim.run(t, pos, vel, acc, quat,
                   gps_outage_start=args.gps_outage_start,
                   gps_outage_end=args.gps_outage_end)

    # Write outputs
    out = args.output_dir
    write_csv(data["ground_truth"], os.path.join(out, "rtk_ground_truth.csv"))
    write_csv(data["imu"], os.path.join(out, "imu_log.csv"))
    write_csv(data["gps"], os.path.join(out, "gps_log.csv"))
    write_csv(data["baro"], os.path.join(out, "baro_log.csv"))
    write_csv(data["mag"], os.path.join(out, "mag_log.csv"))

    print(f"\nDone! Files in {out}/")
    print(f"  Ground truth : {len(data['ground_truth'])} samples")
    print(f"  IMU          : {len(data['imu'])} samples")
    print(f"  GPS          : {len(data['gps'])} samples")
    print(f"  Baro         : {len(data['baro'])} samples")
    print(f"  Mag          : {len(data['mag'])} samples")

    if args.gps_outage_start:
        print(f"  GPS outage   : {args.gps_outage_start}s - {args.gps_outage_end}s")


if __name__ == "__main__":
    main()
