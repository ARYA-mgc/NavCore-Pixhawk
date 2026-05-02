#!/usr/bin/env python3
"""
=====================================================================
 Real-Time INS Navigation System — Pixhawk Cube Orange + RPi 4
 Protocol : MAVLink 2.0 via pymavlink
 Author   : ARYA MGC
 Hardware : Pixhawk Cube Orange  <->  Raspberry Pi 4 (UART / USB)
=====================================================================
 Uses Error-State Extended Kalman Filter (ESKF) with quaternion
 attitude representation. Gimbal-lock-free, production-grade.
=====================================================================
"""

import time
import signal
import sys
import logging
import argparse
import threading
import numpy as np
import os
import traceback

from mavlink_bridge   import MAVLinkBridge
from eskf_core        import ESKFCore, EKFHealth
from imu_noise_params import IMUNoiseParams
from dead_reckon      import DeadReckon
from ins_logger       import INSLogger
from adaptive_pid     import AdaptivePID
from optical_flow_ins import OpticalFlowINS

# ── Logging setup ──────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
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
    Top-level coordinator with ESKF, exception safety, and health monitoring.
    """

    # ── constants ──────────────────────────────────────────────
    PRINT_INTERVAL_S = 0.10
    LOG_INTERVAL_S   = 0.02          # 50 Hz
    STATS_INTERVAL_S = 5.0
    IMU_WATCHDOG_S   = 0.5           # warn if no IMU for this long
    INIT_SAMPLES     = 50            # samples for sensor init

    def __init__(self, connection_string: str, baud: int, update_hz: int):
        self.connection_string = connection_string
        self.baud              = baud
        self.update_hz         = update_hz
        self.dt                = 1.0 / update_hz

        # sub-systems
        self.noise  = IMUNoiseParams()
        self.eskf   = ESKFCore(self.noise)
        self.dr     = DeadReckon(self.noise)
        self.logger = INSLogger("logs/ins_data.csv")
        self.bridge = MAVLinkBridge(connection_string, baud)
        self.adaptive_pid = AdaptivePID(kp_base=1.0, ki_base=0.1, kd_base=0.05)
        self.optical_flow = OpticalFlowINS()
        self.time_sync = TimeSynchronizer(default_dt=self.dt)

        # Vision injector (disabled by default, enabled after convergence)
        self._vision_enabled = False

        # Bookkeeping
        self._imu_count   = 0
        self._baro_count  = 0
        self._mag_count   = 0
        self._last_print  = 0.0
        self._last_log    = 0.0
        self._last_stats  = 0.0
        self._last_imu_t  = 0.0
        self._start_time  = None

        # Timing diagnostics
        self._loop_times   = []
        self._max_loop_ms  = 0.0
        self._overrun_count = 0

        # Initialization buffer
        self._init_accel_buf = []
        self._init_mag_buf   = []

        log.info("INS Navigation System initialised (ESKF)")
        log.info(f"  Connection : {connection_string}  baud={baud}")
        log.info(f"  EKF rate   : {update_hz} Hz  (dt={self.dt*1000:.1f} ms)")

    # ── entry point ────────────────────────────────────────────
    def run(self):
        log.info("Connecting to Pixhawk ...")
        try:
            self.bridge.connect()
        except Exception as e:
            log.critical(f"Connection failed: {e}")
            return

        log.info("Connected — waiting for heartbeat ...")
        self.bridge.wait_heartbeat()
        log.info("Heartbeat received")

        self.bridge.request_data_streams(self.update_hz)
        log.info(f"Data streams requested at {self.update_hz} Hz")

        self._start_time = time.monotonic()
        log.info("=== INS main loop started ===")

        try:
            self._main_loop()
        except Exception as e:
            log.critical(f"Fatal error in main loop: {e}")
            log.critical(traceback.format_exc())
        finally:
            self.bridge.close()
            self.logger.close()
            self._print_final_stats()

    # ── main loop ──────────────────────────────────────────────
    def _main_loop(self):
        global _running
        while _running:
            loop_start = time.monotonic()

            msg = self.bridge.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                # Watchdog: check for IMU timeout
                if (self._last_imu_t > 0 and
                        time.monotonic() - self._last_imu_t > self.IMU_WATCHDOG_S):
                    log.warning("IMU watchdog: no data for "
                                f"{time.monotonic()-self._last_imu_t:.2f}s")
                continue

            t_now = time.monotonic()

            try:
                mtype = msg.get_type()
                self._dispatch_message(mtype, msg, t_now)
            except Exception as e:
                log.error(f"Message processing error ({msg.get_type()}): {e}")

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

            # ── loop timing ────────────────────────────────────
            loop_ms = (time.monotonic() - loop_start) * 1000.0
            self._loop_times.append(loop_ms)
            if len(self._loop_times) > 1000:
                self._loop_times = self._loop_times[-500:]
            if loop_ms > self._max_loop_ms:
                self._max_loop_ms = loop_ms
            if loop_ms > self.dt * 2000:  # 2x expected period
                self._overrun_count += 1
                if self._overrun_count % 100 == 1:
                    log.warning(f"Loop overrun: {loop_ms:.1f} ms "
                                f"(expected <{self.dt*1000:.0f} ms)")

    # ── message dispatch ───────────────────────────────────────
    def _dispatch_message(self, mtype: str, msg, t_now: float):
        ekf = self.eskf

        # ── IMU → predict ──────────────────────────────────
        if mtype == "RAW_IMU":
            accel, gyro = self.bridge.parse_raw_imu(msg)

            # Use hardware-backed time synchronizer
            dt = self.time_sync.compute_dt(msg)
            self._last_imu_t = t_now

            # Initialization phase: collect samples
            if not self.eskf._initialized:
                self._init_accel_buf.append(accel.copy())
                if len(self._init_accel_buf) >= self.INIT_SAMPLES:
                    self._try_initialize()
                return

            ekf.predict(accel, gyro, dt)
            self.dr.update(accel, gyro, dt)
            self._imu_count += 1

        elif mtype == "SCALED_IMU2":
            pass  # secondary IMU, reserved

        # ── Barometer → altitude update ────────────────────
        elif mtype in ("SCALED_PRESSURE", "SCALED_PRESSURE2"):
            alt_m = self.bridge.parse_baro(msg)
            ekf.update_baro(alt_m)
            self._baro_count += 1

        # ── Magnetometer → yaw update ─────────────────────
        elif mtype == "SCALED_IMU3":
            yaw_rad = self.bridge.parse_mag_yaw(msg)
            if yaw_rad is not None:
                # Get mag norm for disturbance detection
                mag_norm = -1.0
                if hasattr(msg, 'xmag'):
                    mx = msg.xmag * 1e-3
                    my = msg.ymag * 1e-3
                    mz = msg.zmag * 1e-3
                    mag_norm = np.sqrt(mx**2 + my**2 + mz**2)

                ekf.update_mag(yaw_rad, mag_norm=mag_norm,
                               t_now=t_now)
                self._mag_count += 1

                # Collect mag for init
                if not self.eskf._initialized:
                    self._init_mag_buf.append(
                        np.array([msg.xmag, msg.ymag, msg.zmag]) * 1e-3)

        elif mtype == "ATTITUDE":
            self.bridge.last_attitude = msg

        elif mtype == "GPS_RAW_INT":
            self.bridge.last_gps = msg

        elif mtype == "OPTICAL_FLOW_RAD":
            if self.eskf._initialized:
                if hasattr(msg, 'distance') and msg.distance > 0:
                    flow_vx = (msg.integrated_x - msg.integrated_xgyro)
                    flow_vy = (msg.integrated_y - msg.integrated_ygyro)
                    dt_flow = msg.integration_time_us / 1e6
                    if dt_flow > 0:
                        vx = flow_vx * msg.distance / dt_flow
                        vy = flow_vy * msg.distance / dt_flow
                        self.eskf.update_optical_flow(
                            vx, vy, msg.distance, msg.quality)

        # ── Health-based vision injection control ──────────
        health = self.eskf.health
        if health == EKFHealth.FAULT and self._vision_enabled:
            self._vision_enabled = False
            log.error("ESKF FAULT: vision injection DISABLED")
            self.bridge.send_statustext("INS FAULT: vision disabled", 4)
        elif health == EKFHealth.HEALTHY and not self._vision_enabled:
            self._vision_enabled = True
            log.info("ESKF healthy: vision injection available")

    # ── sensor initialization ──────────────────────────────────
    def _try_initialize(self):
        accel_arr = np.array(self._init_accel_buf)
        mag_arr = np.array(self._init_mag_buf) if self._init_mag_buf else None

        if mag_arr is None or len(mag_arr) < 5:
            log.info("Waiting for magnetometer samples for initialization...")
            return

        success = self.eskf.initialize_from_sensors(accel_arr, mag_arr)
        if not success:
            log.warning("ESKF initialization failed, retrying...")
            self._init_accel_buf = self._init_accel_buf[-20:]
            self._init_mag_buf = self._init_mag_buf[-20:]

    # ── helpers ────────────────────────────────────────────────
    def _get_pi_temp(self) -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return 0.0

    def _log_state(self, t: float):
        elapsed = t - self._start_time
        pos = self.eskf.state["pos"]
        vel = self.eskf.state["vel"]
        att = np.degrees(self.eskf.state["euler"])

        z_error = 0.0 - pos[2]
        self.adaptive_pid.update(z_error, self.LOG_INTERVAL_S)
        pid_gains = self.adaptive_pid.get_gains()

        pi_temp = self._get_pi_temp()
        if pi_temp > 80.0:
            self.bridge.send_statustext(
                f"WARNING: Pi Temp High: {pi_temp:.1f}C", 4)

        flow_pos, flow_vel = self.optical_flow.get_state()
        self.logger.write(elapsed, pos, vel, att, self.eskf.P,
                          pi_temp, flow_vel, pid_gains)

    def _print_state(self, t: float):
        elapsed = t - self._start_time
        pos = self.eskf.state["pos"]
        att = np.degrees(self.eskf.state["euler"])
        health = self.eskf.health.name

        # NaN guard
        if np.any(np.isnan(pos)) or np.any(np.isnan(att)):
            health = "NAN_FAULT"

        print(
            f"\r[{elapsed:7.2f}s] "
            f"Health: {health:9s} | "
            f"Pos X={pos[0]:+6.2f} Y={pos[1]:+6.2f} Z={pos[2]:+6.2f} | "
            f"Att R={att[0]:+5.1f} P={att[1]:+5.1f} Y={att[2]:+5.1f} | "
            f"dt: {self.time_sync.latency_s*1000:3.0f}ms lat | "
            f"IMU={self._imu_count:6d}",
            end="", flush=True,
        )

    def _print_stats(self, t: float):
        elapsed = t - self._start_time
        eff_hz  = self._imu_count / max(elapsed, 0.001)
        avg_loop = np.mean(self._loop_times) if self._loop_times else 0

        log.info(
            f"Stats @ {elapsed:.1f}s — "
            f"IMU={self._imu_count} ({eff_hz:.0f} Hz)  "
            f"Baro={self._baro_count}  Mag={self._mag_count}  "
            f"Loop avg={avg_loop:.1f}ms max={self._max_loop_ms:.1f}ms  "
            f"Overruns={self._overrun_count}"
        )

    def _print_final_stats(self):
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        log.info("=== INS Session Summary ===")
        log.info(f"  Total runtime : {elapsed:.1f} s")
        log.info(f"  IMU samples   : {self._imu_count}")
        log.info(f"  Baro samples  : {self._baro_count}")
        log.info(f"  Mag samples   : {self._mag_count}")
        log.info(f"  Avg IMU rate  : {self._imu_count/max(elapsed,0.001):.1f} Hz")
        log.info(f"  Loop overruns : {self._overrun_count}")
        log.info(f"  Max loop time : {self._max_loop_ms:.1f} ms")

        pos = self.eskf.state["pos"]
        log.info(f"  Final pos (m) : X={pos[0]:.2f}  Y={pos[1]:.2f}  Z={pos[2]:.2f}")


# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="INS Navigation — Pixhawk Cube Orange + RPi4 (ESKF)")
    p.add_argument(
        "--connection", "-c",
        default="/dev/ttyAMA0",
        help="MAVLink connection string",
    )
    p.add_argument("--baud", "-b",  type=int, default=921600)
    p.add_argument("--hz",         type=int, default=100,
                   help="EKF update rate in Hz (50 or 100)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ins  = INSNavigationSystem(args.connection, args.baud, args.hz)
    ins.run()
