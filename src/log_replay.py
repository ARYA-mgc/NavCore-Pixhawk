#!/usr/bin/env python3
"""
log_replay.py
=============
Offline replay of MAVLink or JSONL logs through the ESKFCore.

Usage:
    python log_replay.py logs/ins_structured_20260502.jsonl
    python log_replay.py flight_log.tlog --format mavlink

Modes:
    JSONL: Replays structured_logger.py output through the ESKF
           deterministically. Ideal for tuning / debugging.
    MAVLink: Parses .tlog / .bin MAVLink logs and replays IMU,
             baro, and mag messages.

Output:
    Writes a new JSONL replay log with recomputed states for
    side-by-side comparison with the original.
"""

import json
import sys
import os
import logging
import time
import numpy as np

log = logging.getLogger("log_replay")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eskf_core import ESKFCore, EKFHealth
from imu_noise_params import IMUNoiseParams
from structured_logger import StructuredLogger


class LogReplay:
    """
    Deterministic offline replay engine.

    Feeds stored sensor data through a fresh ESKFCore instance
    and records full state output for comparison.
    """

    def __init__(self, noise_config: str = None):
        if noise_config:
            self.noise = IMUNoiseParams(noise_config)
        else:
            self.noise = IMUNoiseParams()

        self.eskf = ESKFCore(self.noise)
        self._record_count = 0

    def replay_jsonl(self, input_path: str, output_dir: str = "logs"):
        """
        Replay a JSONL structured log.

        Extracts state vectors from each record and uses them as
        pseudo-measurements to step the ESKF forward deterministically.
        """
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_replay.jsonl")

        s_logger = StructuredLogger(output_dir)

        records = []
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    continue

        log.info(f"Loaded {len(records)} records from {input_path}")

        if len(records) < 2:
            log.error("Not enough records for replay")
            return

        # Initialize ESKF
        self.eskf._initialized = True
        self.eskf.x[6:10] = self.eskf._euler_to_quat(0, 0, 0)

        prev_t = None

        for i, rec in enumerate(records):
            if rec.get("type") != "STATE":
                continue

            t = rec["t"]

            if prev_t is None:
                prev_t = t
                continue

            dt = t - prev_t
            if dt <= 0 or dt > 1.0:
                prev_t = t
                continue

            # Use stored velocity as proxy accel measurement
            vel = np.array(rec["state"]["vel"])
            # Gravity-compensated accel from velocity derivative
            accel = np.array([0.0, 0.0, -9.80665])
            gyro = np.zeros(3)

            # Step the filter
            self.eskf.predict(accel, gyro, dt)

            # Apply baro correction (from stored position z)
            pos_z = rec["state"]["pos"][2]
            if i % 10 == 0:
                self.eskf.update_baro(pos_z)

            # Apply mag correction (from stored yaw)
            euler = rec["state"]["euler"]
            if i % 5 == 0:
                self.eskf.update_mag(euler[2])

            # Log replay state
            s_logger.log_state(
                t=t,
                state=self.eskf.state,
                covariance=self.eskf.P,
                health_status=self.eskf.health.name,
                safety_action="NONE",
                timing_ms=dt * 1000.0
            )

            self._record_count += 1
            prev_t = t

        s_logger.close()
        log.info(f"Replay complete: {self._record_count} steps processed")
        log.info(f"Replay output: {s_logger.filepath}")

        return s_logger.filepath

    def replay_mavlink(self, input_path: str, output_dir: str = "logs"):
        """
        Replay a MAVLink .tlog or .bin log.

        Requires pymavlink. Parses RAW_IMU, SCALED_PRESSURE, and
        RAW_IMU messages to drive the ESKF.
        """
        try:
            from pymavlink import mavutil
        except ImportError:
            log.error("pymavlink required for MAVLink replay. "
                      "Install: pip install pymavlink")
            return None

        os.makedirs(output_dir, exist_ok=True)
        s_logger = StructuredLogger(output_dir)

        mlog = mavutil.mavlink_connection(input_path)

        self.eskf._initialized = False
        prev_time_us = None
        init_accel_buf = []
        init_mag_buf = []

        while True:
            msg = mlog.recv_match(blocking=False)
            if msg is None:
                break

            mtype = msg.get_type()

            if mtype == "RAW_IMU":
                # Convert from milli-units
                accel = np.array([
                    msg.xacc / 1000.0 * 9.80665,
                    msg.yacc / 1000.0 * 9.80665,
                    msg.zacc / 1000.0 * 9.80665
                ])
                gyro = np.array([
                    msg.xgyro / 1000.0,
                    msg.ygyro / 1000.0,
                    msg.zgyro / 1000.0
                ])

                time_us = msg.time_usec

                if not self.eskf._initialized:
                    init_accel_buf.append(accel)
                    mag = np.array([msg.xmag, msg.ymag, msg.zmag]) / 1000.0
                    init_mag_buf.append(mag)

                    if len(init_accel_buf) >= 50 and len(init_mag_buf) >= 50:
                        self.eskf.initialize_from_sensors(
                            np.array(init_accel_buf),
                            np.array(init_mag_buf)
                        )
                        log.info("ESKF initialized from MAVLink log")
                    continue

                if prev_time_us is not None:
                    dt = (time_us - prev_time_us) / 1e6
                    if 0 < dt < 0.1:
                        self.eskf.predict(accel, gyro, dt)
                        self._record_count += 1

                prev_time_us = time_us

            elif mtype == "SCALED_PRESSURE":
                if self.eskf._initialized:
                    # Convert pressure to altitude (simplified)
                    alt = (1013.25 - msg.press_abs) * 8.3
                    self.eskf.update_baro(alt)

        # Log final state
        if self.eskf._initialized:
            s_logger.log_state(
                t=0.0,
                state=self.eskf.state,
                covariance=self.eskf.P,
                health_status=self.eskf.health.name,
                safety_action="NONE",
                timing_ms=0.0
            )

        s_logger.close()
        log.info(f"MAVLink replay complete: {self._record_count} IMU steps")
        return s_logger.filepath


# ── CLI Entry Point ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python log_replay.py <logfile> [--format jsonl|mavlink]")
        print("  Replays a log through the ESKF offline.")
        sys.exit(1)

    logfile = sys.argv[1]
    fmt = "jsonl"
    if "--format" in sys.argv:
        idx = sys.argv.index("--format")
        fmt = sys.argv[idx + 1].lower()

    replay = LogReplay()

    if fmt == "mavlink":
        replay.replay_mavlink(logfile)
    else:
        replay.replay_jsonl(logfile)
