#!/usr/bin/env python3
"""
=====================================================================
 Real-Time INS Navigation System — Pixhawk Cube Orange + RPi 4
 Protocol : MAVLink 2.0 via pymavlink
 Author   : ARYA MGC
 Hardware : Pixhawk Cube Orange  ←→  Raspberry Pi 4 (UART / USB)
=====================================================================
 Architecture
 ┌───────────────────────────────────────────────────────────────┐
 │  Pixhawk Cube Orange                                          │
 │   IMU (ICM-20689 / ICM-42688)  100 Hz                         │
 │   Barometer (MS5611)            10 Hz                         │
 │   Magnetometer (RM3100)         50 Hz                         │
 └────────────────────────┬──────────────────────────────────────┘
                          │ UART (/dev/ttyAMA0) or USB (/dev/ttyACM0)
                          │ MAVLink 2.0
 ┌────────────────────────▼──────────────────────────────────────┐
 │  Raspberry Pi 4                                               │
 │   mavlink_bridge.py  ── raw MAVLink parser                    │
 │   imu_noise_params.py ─ sensor noise config                   │
 │   ekf_core.py        ── 9-state Extended Kalman Filter        │
 │   dead_reckon.py     ── fallback dead-reckoning               │
 │   ins_logger.py      ── CSV / live telemetry log              │
 │   main_ins_navigation.py  ◄── YOU ARE HERE                    │
 └───────────────────────────────────────────────────────────────┘
"""

import time
import signal
import sys
import logging
import argparse
import threading
import numpy as np

from mavlink_bridge   import MAVLinkBridge
from ekf_core         import EKFCore
from imu_noise_params import IMUNoiseParams
from dead_reckon      import DeadReckon
from ins_logger       import INSLogger

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/ins_runtime.log"),
    ],
)
log = logging.getLogger("main_ins")

# ── Graceful shutdown ───────────────────────────────────────────
_running = True

def _signal_handler(sig, frame):
    global _running
    log.info("Shutdown signal received — stopping INS loop.")
    _running = False

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ═══════════════════════════════════════════════════════════════
class INSNavigationSystem:
    """
    Top-level coordinator.
    1. Opens MAVLink connection to Pixhawk
    2. Collects IMU / Baro / Mag messages
    3. Runs EKF predict (every IMU) + update (baro / mag)
    4. Logs and prints state at ~10 Hz
    """

    # ── constants ──────────────────────────────────────────────
    PRINT_INTERVAL_S = 0.10          # console print rate
    LOG_INTERVAL_S   = 0.02          # CSV log rate  (50 Hz)
    STATS_INTERVAL_S = 5.0           # stats summary rate

    def __init__(self, connection_string: str, baud: int, update_hz: int):
        self.connection_string = connection_string
        self.baud              = baud
        self.update_hz         = update_hz
        self.dt                = 1.0 / update_hz

        # sub-systems
        self.noise  = IMUNoiseParams()
        self.ekf    = EKFCore(self.noise)
        self.dr     = DeadReckon(self.noise)
        self.logger = INSLogger("logs/ins_data.csv")
        self.bridge = MAVLinkBridge(connection_string, baud)

        # bookkeeping
        self._imu_count  = 0
        self._baro_count = 0
        self._mag_count  = 0
        self._last_print  = 0.0
        self._last_log    = 0.0
        self._last_stats  = 0.0
        self._start_time  = None

        log.info("INS Navigation System initialised")
        log.info(f"  Connection : {connection_string}  baud={baud}")
        log.info(f"  EKF rate   : {update_hz} Hz  (dt={self.dt*1000:.1f} ms)")

    # ── entry point ────────────────────────────────────────────
    def run(self):
        log.info("Connecting to Pixhawk …")
        self.bridge.connect()
        log.info("Connected — waiting for heartbeat …")
        self.bridge.wait_heartbeat()
        log.info("Heartbeat received  ✓")

        # Request high-rate IMU stream
        self.bridge.request_data_streams(self.update_hz)
        log.info(f"Data streams requested at {self.update_hz} Hz  ✓")

        self._start_time = time.monotonic()
        log.info("=== INS main loop started ===")

        try:
            self._main_loop()
        finally:
            self.bridge.close()
            self.logger.close()
            self._print_final_stats()

    # ── main loop ──────────────────────────────────────────────
    def _main_loop(self):
        global _running
        while _running:
            msg = self.bridge.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue

            t_now = time.monotonic()
            mtype = msg.get_type()

            # ── IMU → EKF predict ──────────────────────────────
            if mtype == "RAW_IMU":
                accel, gyro = self.bridge.parse_raw_imu(msg)
                self.ekf.predict(accel, gyro, self.dt)
                self.dr.update(accel, gyro, self.dt)
                self._imu_count += 1

            elif mtype == "SCALED_IMU2":          # secondary IMU
                accel, gyro = self.bridge.parse_scaled_imu(msg)
                # optional: feed to a second EKF or average
                pass

            # ── Barometer → EKF altitude update ────────────────
            elif mtype in ("SCALED_PRESSURE", "SCALED_PRESSURE2"):
                alt_m = self.bridge.parse_baro(msg)
                self.ekf.update_baro(alt_m)
                self._baro_count += 1

            # ── Magnetometer → EKF yaw update ──────────────────
            elif mtype == "RAW_IMU":
                pass  # mag handled separately below

            elif mtype == "SCALED_IMU3":
                # Cube Orange exposes mag via SCALED_IMU3.xmag/ymag/zmag
                yaw_rad = self.bridge.parse_mag_yaw(msg)
                if yaw_rad is not None:
                    self.ekf.update_mag(yaw_rad)
                    self._mag_count += 1

            elif mtype == "ATTITUDE":
                # Pixhawk's own attitude estimate — used for cross-check only
                self.bridge.last_attitude = msg

            elif mtype == "GPS_RAW_INT":
                # Even in GPS-denied mode, log GPS fix quality
                self.bridge.last_gps = msg

            # ── periodic tasks ─────────────────────────────────
            if t_now - self._last_log >= self.LOG_INTERVAL_S:
                self._log_state(t_now)
                self._last_log = t_now

            if t_now - self._last_print >= self.PRINT_INTERVAL_S:
                self._print_state(t_now)
                self._last_print = t_now

            if t_now - self._last_stats >= self.STATS_INTERVAL_S:
                self._print_stats(t_now)
                self._last_stats = t_now

    # ── helpers ────────────────────────────────────────────────
    def _log_state(self, t: float):
        elapsed = t - self._start_time
        pos = self.ekf.state["pos"]
        vel = self.ekf.state["vel"]
        att = np.degrees(self.ekf.state["euler"])
        self.logger.write(elapsed, pos, vel, att, self.ekf.P)

    def _print_state(self, t: float):
        elapsed = t - self._start_time
        pos = self.ekf.state["pos"]
        vel = self.ekf.state["vel"]
        att = np.degrees(self.ekf.state["euler"])
        print(
            f"\r[{elapsed:7.2f}s] "
            f"Pos(m) X={pos[0]:+7.2f} Y={pos[1]:+7.2f} Z={pos[2]:+7.2f}  "
            f"Att(°) R={att[0]:+6.1f} P={att[1]:+6.1f} Y={att[2]:+6.1f}  "
            f"IMU={self._imu_count:6d}",
            end="", flush=True,
        )

    def _print_stats(self, t: float):
        elapsed = t - self._start_time
        eff_hz  = self._imu_count / max(elapsed, 0.001)
        log.info(
            f"Stats @ {elapsed:.1f}s — "
            f"IMU={self._imu_count} ({eff_hz:.0f} Hz)  "
            f"Baro={self._baro_count}  Mag={self._mag_count}"
        )

    def _print_final_stats(self):
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        log.info("=== INS Session Summary ===")
        log.info(f"  Total runtime : {elapsed:.1f} s")
        log.info(f"  IMU samples   : {self._imu_count}")
        log.info(f"  Baro samples  : {self._baro_count}")
        log.info(f"  Mag samples   : {self._mag_count}")
        log.info(f"  Avg IMU rate  : {self._imu_count/max(elapsed,0.001):.1f} Hz")
        pos = self.ekf.state["pos"]
        log.info(f"  Final pos (m) : X={pos[0]:.2f}  Y={pos[1]:.2f}  Z={pos[2]:.2f}")


# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="INS Navigation — Pixhawk Cube Orange + RPi4")
    p.add_argument(
        "--connection", "-c",
        default="/dev/ttyAMA0",
        help="MAVLink connection string  (default: /dev/ttyAMA0)\n"
             "  UART: /dev/ttyAMA0   USB: /dev/ttyACM0\n"
             "  TCP : tcp:127.0.0.1:5760  (SITL / Mission Planner)",
    )
    p.add_argument("--baud", "-b",  type=int, default=921600,
                   help="Serial baud rate (default 921600)")
    p.add_argument("--hz",         type=int, default=100,
                   help="EKF update rate in Hz (50 or 100, default 100)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ins  = INSNavigationSystem(args.connection, args.baud, args.hz)
    ins.run()
