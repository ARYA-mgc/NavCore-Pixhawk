#!/usr/bin/env python3
# Our drone got lost once. Never again.
# This is the main brain that runs the ESKF math magic and talks to the Pixhawk.

import sys
from pathlib import Path
# Add 'src' to the python path so absolute imports work
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

import time
import signal
import sys
import logging
import argparse
import threading
import numpy as np
import math
import os
import traceback

from interfaces.mavlink import MAVLinkBridge
from core.eskf        import ESKF, EKFHealth
from utils.noise      import IMUNoiseParams
from core.dr import DeadReckon
from safety.safety import SafetyMonitor, SafetyAction
from safety.loop import LoopMonitor
from utils.sync import TimeSynchronizer
from logger.logger import INSLogger
from logger.struct_log import StructuredLogger
from utils.pid        import AdaptivePID
from fusion.opt_flow import OpticalFlowINS
from fusion.lr import LidarRadarFusion
from fusion.vio import VIOPipeline
from safety.mlp import MLAnomalyDetector
from safety.fault import FaultManager
from concurrent.futures import ThreadPoolExecutor

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
class INSNavSys:
    # The big boss class that keeps the drone from falling out of the sky

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
        self.eskf   = ESKF(self.noise)
        self.dr     = DeadReckon(self.noise)
        self.logger = INSLogger("logs/ins_data.csv")
        self.s_logger = StructuredLogger("logs")
        self.bridge = MAVLinkBridge(connection_string, baud)
        self.adaptive_pid = AdaptivePID(kp_base=1.0, ki_base=0.1, kd_base=0.05)
        self.optical_flow = OpticalFlowINS()
        self.vio = VIOPipeline(enable=False)  # enable when T265/ORB-SLAM3 connected
        self.time_sync = TimeSynchronizer(default_dt=self.dt)
        self.safety = SafetyMonitor()
        self.fault_mgr = FaultManager()

        # Advanced multi-threading for intensive ML/LiDAR math
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._ml_future = None
        self._lidar_future = None

        # Mission Planner tunable parameters
        self.params = {
            "ML_CONTAM": 0.01,
            "LR_VOX": 0.1,
            "LR_WEIGHT": 1.0,      # Fusion weight scaling
            "OBS_THRESH": 2.0,     # Obstacle detection threshold (m)
            "SENS_TIMEOUT": 0.5,   # Sensor timeout (s)
            "RDR_REJECT": 0.1,     # Radar static reject threshold
            "EKF_NOISE_SCL": 1.0,  # Noise scalar for tuning
            "RTH_MIN_ALT": 5.0,    # Minimum RTH altitude (m AGL)
            "RTH_CEIL_MARGIN": 2.0, # Minimum ceiling clearance required (m)
        }

        # Override defaults with params
        self.IMU_WATCHDOG_S = self.params["SENS_TIMEOUT"]

        self.lidar_radar = LidarRadarFusion(voxel_size=self.params["LR_VOX"], rdr_reject=self.params["RDR_REJECT"])
        self.ml_predictor = MLAnomalyDetector(contamination=self.params["ML_CONTAM"])

        # Vision injector (disabled by default, enabled after convergence)
        self._vision_enabled = False
        self._rth_active = False

        # Bookkeeping
        self._imu_count   = 0
        self._baro_count  = 0
        self._mag_count   = 0
        self._gps_count   = 0
        self._vio_count   = 0
        self._last_print  = 0.0
        self._last_log    = 0.0
        self._last_stats  = 0.0
        self._last_imu_t  = 0.0
        self._start_time  = None

        # Timing diagnostics
        self.loop_monitor = LoopMonitor(target_dt=self.dt)

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
            self.s_logger.close()
            self.loop_monitor.print_histogram()
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
            self.loop_monitor.record_loop(loop_ms)

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
            # Fuse GPS into ESKF when fix is 3D (fix_type >= 3)
            if (self.eskf._initialized and
                    hasattr(msg, 'fix_type') and msg.fix_type >= 3):
                lat = msg.lat / 1e7  # degE7 → degrees
                lon = msg.lon / 1e7
                alt = msg.alt / 1000.0  # mm → m
                hdop = msg.eph / 100.0 if hasattr(msg, 'eph') else 2.0
                self.eskf.update_gps(lat, lon, alt, hdop=hdop)
                self._gps_count += 1

        elif mtype == "OBSTACLE_DISTANCE":
            # Lidar data ingestion
            # TODO: Replace with real Livox Mid-360 ROS2 topic subscriber
            #       or serial parser. OBSTACLE_DISTANCE only has 1D ranges;
            #       for full 3D point cloud, use a ROS PointCloud2 subscriber.
            if hasattr(msg, 'distances'):
                # Extract valid distances from OBSTACLE_DISTANCE message
                # msg.distances is a 72-element array of distances in cm (0 = invalid)
                distances_cm = np.array(msg.distances, dtype=float)
                valid_mask = (distances_cm > 0) & (distances_cm < 65535)
                if not np.any(valid_mask):
                    return
                valid_dists = distances_cm[valid_mask] / 100.0  # cm → m
                n_pts = len(valid_dists)
                # Build pseudo-3D points from angular sectors
                angle_offset = getattr(msg, 'angle_offset', 0.0)
                increment = getattr(msg, 'increment', 5.0)  # degrees per bin
                indices = np.where(valid_mask)[0]
                angles_rad = np.radians(angle_offset + indices * increment)
                pts = np.column_stack([
                    valid_dists * np.cos(angles_rad),
                    valid_dists * np.sin(angles_rad),
                    np.zeros(n_pts),
                ])

                # Async execution to prevent blocking the 100Hz loop
                if self._lidar_future is None or self._lidar_future.done():
                    if self._lidar_future is not None:
                        safe_dist = self._lidar_future.result()
                        if safe_dist > 0:
                            self.eskf.update_lidar_range(safe_dist, weight=self.params["LR_WEIGHT"])

                            if safe_dist < self.params["OBS_THRESH"]:
                                self.bridge.send_statustext(f"OBSTACLE CLOSE: {safe_dist:.1f}m", 3)

                    # Spawn next processing task
                    self._lidar_future = self.executor.submit(
                        self.lidar_radar.process_livox_cloud, pts
                    )

        elif mtype == "RADAR_TARGET":
            # TI mmWave IWR6843AOP doppler target ingestion
            # TODO: Replace with real TI radar serial parser (UART at 921600).
            #       RADAR_TARGET is not a standard MAVLink msg — when using a
            #       custom parser, construct targets as [[x, y, z, doppler_v], ...]
            if not hasattr(msg, 'distance') or not hasattr(msg, 'velocity'):
                return
            # Build target array from MAVLink radar-like fields
            angle_rad = math.radians(getattr(msg, 'angle', 0.0))
            dist = float(msg.distance)
            vel = float(msg.velocity)
            target = np.array([[
                dist * math.cos(angle_rad),
                dist * math.sin(angle_rad),
                0.0,
                vel,
            ]])
            radar_vel = self.lidar_radar.process_ti_radar(target)
            if np.any(radar_vel):
                self.eskf.update_radar_velocity(*radar_vel, weight=self.params["LR_WEIGHT"])

        elif mtype == "PARAM_REQUEST_LIST":
            self._send_all_params()

        elif mtype == "PARAM_SET":
            # Mission Planner parameter update
            param_id = msg.param_id.strip("\x00")
            if param_id in self.params:
                self.params[param_id] = msg.param_value
                self._apply_param(param_id)
                self._send_param(param_id)
                log.info(f"Param updated via MAVLink: {param_id} = {msg.param_value}")

        elif mtype == "OPTICAL_FLOW_RAD":
            if self.eskf._initialized:
                # 1. Require valid rangefinder
                if not hasattr(msg, 'distance') or msg.distance <= 0.05:
                    return

                # 2. Reject during high angular rates (prevents smearing/aliasing)
                gyro_x_rate = abs(msg.integrated_xgyro / (msg.integration_time_us / 1e6))
                gyro_y_rate = abs(msg.integrated_ygyro / (msg.integration_time_us / 1e6))
                if gyro_x_rate > 1.5 or gyro_y_rate > 1.5:  # ~85 deg/s
                    return

                flow_vx = (msg.integrated_x - msg.integrated_xgyro)
                flow_vy = (msg.integrated_y - msg.integrated_ygyro)
                dt_flow = msg.integration_time_us / 1e6

                if dt_flow > 0:
                    # 3. Height scaling: angular_rate (rad/s) × height (m) = velocity (m/s)
                    vx = (flow_vx / dt_flow) * msg.distance
                    vy = (flow_vy / dt_flow) * msg.distance
                    self.eskf.update_optical_flow(
                        vx, vy, msg.distance, msg.quality)

        elif mtype == "VISION_POSITION_ESTIMATE":
            # VIO fusion (T265 / ORB-SLAM3)
            if self.vio.is_active:
                pos_vio = np.array([msg.x, msg.y, msg.z])
                # T265 provides quaternion; fallback to identity if unavailable
                quat_vio = np.array([1.0, 0.0, 0.0, 0.0])
                if hasattr(msg, 'q') and msg.q is not None:
                    quat_vio = np.array(msg.q)
                confidence = 0.8  # TODO: parse tracking confidence from T265 via STATUSTEXT
                result = self.vio.process_vio_update(
                    t_now, pos_vio, quat_vio, confidence)
                if result is not None:
                    self.eskf.update_external(
                        result["pos_ned"],
                        self.eskf.state["pos"],
                        result["H_pos"], result["R_pos"],
                        source="VIO_pos")
                    self.eskf.update_external(
                        np.array([result["yaw_ned"]]),
                        np.array([self.eskf.state["euler"][2]]),
                        result["H_yaw"], result["R_yaw"],
                        source="VIO_yaw")
                    self._vio_count += 1

        # ── Safety & Health ─────────────────────────────────────────
        pos = self.eskf.state["pos"]
        vel = self.eskf.state["vel"]
        att = self.eskf.state["euler"]
        
        # 1. ML Predictive Fault Analysis (Async)
        # Check vibration and covariance variance to predict failure before divergence
        accel_var = float(np.var(self._init_accel_buf[-20:])) if len(self._init_accel_buf) > 0 else 0.0
        gyro_proxy = float(np.linalg.norm(self.eskf.state["gyro_bias"]))
        p_trace = float(np.trace(self.eskf.P))
        
        # Async harvesting to prevent 100Hz loop blocking
        if self._ml_future is None or self._ml_future.done():
            if self._ml_future is not None and self._ml_future.result() == True:
                self.bridge.send_statustext("ML FAULT - ACTIVATING SMART RTH", 2)
                self._rth_active = True
                self.bridge.set_mode("GUIDED")
                self.fault_mgr.update(t_now, ekf_healthy=True, safety_ok=True, ml_fault=True)
            
            # Fire and forget the next prediction
            self._ml_future = self.executor.submit(
                self.ml_predictor.check_health, accel_var, gyro_proxy, p_trace
            )
            
        # 1.5 Smart Return to Home (RTH) Execution
        if self._rth_active:
            target_pos = np.array([0.0, 0.0])
            error = target_pos - pos[:2]
            dist = np.linalg.norm(error)
            
            if dist < 0.5:
                self.bridge.send_statustext("RTH COMPLETE - HOVERING", 6)
                self.bridge.send_velocity_target(0, 0, 0)
                self._rth_active = False
            else:
                p_gain = 0.5
                max_vel = 2.0
                desired_vel = error * p_gain
                
                speed = np.linalg.norm(desired_vel)
                if speed > max_vel:
                    desired_vel = (desired_vel / speed) * max_vel
                    
                # Obstacle Avoidance
                safe_dist = self.lidar_radar.safe_distance_m
                if safe_dist > 0 and safe_dist < self.params["OBS_THRESH"]:
                    scale = (safe_dist / self.params["OBS_THRESH"]) ** 2
                    desired_vel *= scale
                    if safe_dist < 0.5:
                        desired_vel = np.zeros(2)
                        self.bridge.send_statustext("RTH BLOCKED - HOLDING", 3)
                
                # Altitude Policy: climb to RTH_MIN_ALT if below, with ceiling check
                desired_vz = 0.0
                current_alt_agl = -pos[2]  # NED → AGL
                rth_min_alt = self.params["RTH_MIN_ALT"]
                ceil_margin = self.params["RTH_CEIL_MARGIN"]

                if current_alt_agl < rth_min_alt:
                    # Check vertical clearance via lidar before climbing
                    lidar_range = self.lidar_radar.safe_distance_m
                    if lidar_range > 0 and lidar_range < ceil_margin:
                        # Ceiling too close — hold altitude, do NOT climb
                        desired_vz = 0.0
                        self.bridge.send_statustext(
                            f"RTH CEIL BLOCKED: {lidar_range:.1f}m", 3)
                    elif lidar_range <= 0:
                        # No lidar data — refuse to climb blindly
                        desired_vz = 0.0
                        self.bridge.send_statustext(
                            "RTH NO LIDAR - HOLD ALT", 4)
                    else:
                        # Clear to climb
                        desired_vz = -0.5  # climb in NED

                self.bridge.send_velocity_target(desired_vel[0], desired_vel[1], desired_vz)
        
        # 2. Hard safety enforcement
        safety_action = self.safety.check(pos, vel, att)

        if safety_action == SafetyAction.FORCE_DISARM:
            # Tell Pixhawk to disarm! (MAVLink command)
            self.bridge.send_statustext("INS CRITICAL FAULT - DISARM", 2)
            # You could add actual MAV_CMD_COMPONENT_ARM_DISARM here

        health = self.eskf.health

        # Initialize VIO alignment once ESKF is healthy
        if (health == EKFHealth.HEALTHY and
                self.vio._enabled and not self.vio._initialized):
            self.vio.initialize(
                self.eskf.state["pos"], self.eskf.state["quat"],
                np.zeros(3), np.array([1.0, 0, 0, 0]))
            log.info("VIO pipeline aligned to ESKF frame")

        # Vision enabled ONLY if ESKF is healthy AND safety monitor says OK
        can_inject = (health == EKFHealth.HEALTHY and self.safety.is_injection_safe)

        if not can_inject and self._vision_enabled:
            self._vision_enabled = False
            log.error(f"Vision injection DISABLED! Health={health.name}, Safety={safety_action.name}")
            self.bridge.send_statustext("INS: Vision disabled", 4)
        elif can_inject and not self._vision_enabled:
            self._vision_enabled = True
            log.info("Vision injection ENABLED")
            self.bridge.send_statustext("INS: Vision enabled", 6)

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

    # ── MAVLink Parameter Server ───────────────────────────────
    def _send_all_params(self):
        for param_id in self.params:
            self._send_param(param_id)

    def _send_param(self, param_id: str):
        val = float(self.params[param_id])
        idx = list(self.params.keys()).index(param_id)
        count = len(self.params)
        
        try:
            from pymavlink import mavutil
            self.bridge._conn.mav.param_value_send(
                param_id.encode('utf-8'), val, 
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32, count, idx)
        except Exception as e:
            log.error(f"Failed to send param: {e}")

    def _apply_param(self, param_id: str):
        val = self.params[param_id]
        if param_id == "LR_VOX":
            self.lidar_radar.voxel_size = val
        elif param_id == "ML_CONTAM":
            self.ml_predictor = MLAnomalyDetector(contamination=val)
        elif param_id == "SENS_TIMEOUT":
            self.IMU_WATCHDOG_S = val
        elif param_id == "EKF_NOISE_SCL":
            self.eskf.Q *= val # Simple dynamic noise scalar

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

        # Structured log
        self.s_logger.log_state(
            t=elapsed,
            state=self.eskf.state,
            covariance=self.eskf.P,
            health_status=self.eskf.health.name,
            safety_action=self.safety.last_action.name,
            timing_ms=self.time_sync.latency_s * 1000.0
        )

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
        stats   = self.loop_monitor.get_stats()

        log.info(
            f"Stats @ {elapsed:.1f}s — "
            f"IMU={self._imu_count} ({eff_hz:.0f} Hz)  "
            f"Baro={self._baro_count}  Mag={self._mag_count}  "
            f"GPS={self._gps_count}  VIO={self._vio_count}  "
            f"Loop avg={stats['avg']:.1f}ms max={stats['max']:.1f}ms  "
            f"Overruns={stats['overruns']}"
        )

    def _print_final_stats(self):
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        log.info("=== INS Session Summary ===")
        log.info(f"  Total runtime : {elapsed:.1f} s")
        log.info(f"  IMU samples   : {self._imu_count}")
        log.info(f"  Baro samples  : {self._baro_count}")
        log.info(f"  Mag samples   : {self._mag_count}")
        log.info(f"  GPS samples   : {self._gps_count}")
        log.info(f"  VIO samples   : {self._vio_count}")
        log.info(f"  Avg IMU rate  : {self._imu_count/max(elapsed,0.001):.1f} Hz")
        stats = self.loop_monitor.get_stats()
        log.info(f"  Loop overruns : {stats['overruns']}")
        log.info(f"  Max loop time : {stats['max']:.1f} ms")

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
    ins  = INSNavSys(args.connection, args.baud, args.hz)
    ins.run()
