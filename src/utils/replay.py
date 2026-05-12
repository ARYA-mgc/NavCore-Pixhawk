#!/usr/bin/env python3
# Time travel. Replaying old flights to see why we crashed.

import json
import sys
import os
import logging
import time
import numpy as np

log = logging.getLogger("log_replay")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.eskf import ESKF, EKFHealth
from utils.noise import IMUNoiseParams
from logger.struct_log import StructuredLogger


class LogReplay:
    # the time machine — feed old flight data through a fresh filter

    def __init__(self, noise_config: str = None):
        if noise_config:
            self.noise = IMUNoiseParams(noise_config)
        else:
            self.noise = IMUNoiseParams()

        self.eskf = ESKF(self.noise)
        self._record_count = 0

    def replay_jsonl(self, input_path: str, output_dir: str = "logs"):
        # crack open a log file and relive the magic (or the crash)
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

        # wake up the filter
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

            # let the math do its thing
            self.eskf.predict(accel, gyro, dt)

            # baro says we're at this altitude
            pos_z = rec["state"]["pos"][2]
            if i % 10 == 0:
                self.eskf.update_baro(pos_z)

            # compass says we're pointing this way
            euler = rec["state"]["euler"]
            if i % 5 == 0:
                self.eskf.update_mag(euler[2])

            # write down what happened
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
        # same thing but for raw mavlink recordings from the pixhawk
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
                # pixhawk sends everything in milli-whatevers
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
                    # pressure to meters, the lazy way
                    alt = (1013.25 - msg.press_abs) * 8.3
                    self.eskf.update_baro(alt)

        # one last snapshot before we close
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
