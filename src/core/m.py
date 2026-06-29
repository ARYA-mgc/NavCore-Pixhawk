#!/usr/bin/env python3
# Main execution loop. Ties the math to the hardware.
# Please don't edit this while airborne.

import sys
from pathlib import Path

import time
import signal
import logging
import argparse
import threading
import numpy as np
import math
import os
import traceback

from interfaces.mavlink import MAVLinkBridge
from core.eskf        import ESKF, EKFHealth
from core.mht         import MHTManager
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
from fusion.multi_imu import MultiIMUFusion
from fusion.gps_tight import TightGPSCoupling
from safety.mlp import MLAnomalyDetector
from safety.fault import FaultManager
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from fusion.mag_cal import MagAutoCalibrator
from fusion.trn import TerrainRelativeNavigation, DEMTile

# RTK ground truth — the "prove you're not lying" module (optional, --rtk flag)
try:
    from interfaces.rtk_collector import RTKCollector
    from interfaces.ntrip_client import NTRIPClient, generate_gga
    from logger.flight_recorder import FlightRecorder
    HAS_RTK = True
except ImportError:
    HAS_RTK = False

# ── Logging setup (because printf debugging is a war crime) ──
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

# ── Graceful shutdown (Ctrl+C is not a landing procedure) ──────
_running = True

def _signal_handler(sig, frame):
    global _running
    log.info("Shutdown signal received — stopping INS loop.")
    _running = False

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ═══════════════════════════════════════════════════════════════
class INSNavSys:
    # The big boss class. 980 lines of "please don't crash."
    # Started as 200 lines. Then GPS happened. Then baro drift.
    # Then someone said "what about optical flow?" and here we are.

    # ── constants (touch these and the drone WILL find you) ───
    PRINT_INTERVAL_S = 0.10
    LOG_INTERVAL_S   = 0.02          # 50 Hz — fast enough to catch the chaos
    STATS_INTERVAL_S = 5.0
    IMU_WATCHDOG_S   = 0.5           # if IMU goes quiet for this long, panic
    INIT_SAMPLES     = 50            # hold still for 50 samples or the math gods get angry

    def __init__(self, connection_string: str, baud: int, update_hz: int,
                 rtk_enabled: bool = False):
        self.connection_string = connection_string
        self.baud              = baud
        self.update_hz         = update_hz
        self.dt                = 1.0 / update_hz
        self._rtk_enabled      = rtk_enabled and HAS_RTK

        # sub-systems (the Avengers, but for navigation)
        self.noise  = IMUNoiseParams()
        _initial_eskf = ESKF(self.noise)
        self.mht    = MHTManager(_initial_eskf)
        self.dr     = DeadReckon(self.noise)
        self.logger = INSLogger("logs/ins_data.csv")
        self.s_logger = StructuredLogger("logs")
        self.mag_cal = MagAutoCalibrator()
        
        # TRN — terrain matching so we know WHERE the ground is before we meet it
        self.trn = TerrainRelativeNavigation(enable=True)
        # Flat fake DEM for testing. Real terrain has hills. We'll cross that bridge later.
        self.trn.load_dem(DEMTile(0.0, 0.0, np.zeros((100, 100)), resolution=30.0))

        self.bridge = MAVLinkBridge(connection_string, baud)
        self.adaptive_pid = AdaptivePID(kp_base=1.0, ki_base=0.1, kd_base=0.05)
        self.optical_flow = OpticalFlowINS()
        self.vio = VIOPipeline(enable=False)  # sleeping until T265/ORB-SLAM3 shows up
        self.imu_fusion = MultiIMUFusion(n_channels=3)  # 3 IMUs because 1 is never enough
        self.gps_tight = TightGPSCoupling(enable=False)  # needs u-blox F9P — it's on the wishlist
        self.time_sync = TimeSynchronizer(default_dt=self.dt)
        self.safety = SafetyMonitor()
        self.fault_mgr = FaultManager()

        # 2 threads for the heavy stuff. More threads = more problems. Ask me how I know.
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._ml_future = None
        self._lidar_future = None

        # Mission Planner tunable knobs (for when the field crew says "make it less wobbly")
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
            "OOSM_BUFFER_S": 2.0,  # OOSM buffer duration (s)
            "OOSM_ENABLE": 1.0,    # Enable OOSM replay
            "MAG_3D": 0.0          # 0=False, 1=True
        }

        # OOSM time-travel buffer — because GPS data arrives fashionably late
        self._oosm_buffer = deque(maxlen=int(self.params["OOSM_BUFFER_S"] * self.update_hz))

        # Let the config override our careful defaults (what could go wrong?)
        self.IMU_WATCHDOG_S = self.params["SENS_TIMEOUT"]
        self._emergency_land_triggered = False

        self.lidar_radar = LidarRadarFusion(voxel_size=self.params["LR_VOX"], rdr_reject=self.params["RDR_REJECT"])
        self.ml_predictor = MLAnomalyDetector(contamination=self.params["ML_CONTAM"])

        # Vision injection — stays OFF until ESKF proves it can walk before we let it run
        self._vision_enabled = False
        self._rth_active = False  # RTH: the "oh no" button for the filter

        # ── RTK ground truth (the expensive way to prove we're right) ──
        self.rtk_collector = None
        self.ntrip_client = None
        self.flight_recorder = None

        if self._rtk_enabled:
            self._init_rtk()

        # Bookkeeping — because "how many samples did we get?" is always the first question
        self._imu_count   = 0
        self._baro_count  = 0
        self._mag_count   = 0
        self._gps_count   = 0
        self._vio_count   = 0
        self._zupt_count  = 0   # consecutive stationary detections
        self._zupt_total  = 0   # total ZUPT updates applied
        self._last_print  = 0.0
        self._last_log    = 0.0
        self._last_stats  = 0.0
        self._last_imu_t  = 0.0
        self._start_time  = None

        # Timing diagnostics — proof that we actually hit 100Hz (sometimes)
        self.loop_monitor = LoopMonitor(target_dt=self.dt)

        # Init buffers — sit still and let us stare at gravity for a bit
        self._init_accel_buf = []
        self._init_mag_buf   = []

        log.info("INS Navigation System initialised (ESKF)")
        if self._rtk_enabled:
            log.info("  RTK ground truth collection ENABLED")
        log.info(f"  Connection : {connection_string}  baud={baud}")
        log.info(f"  EKF rate   : {update_hz} Hz  (dt={self.dt*1000:.1f} ms)")

    # ── entry point (where the magic / disaster begins) ────────
    def _init_rtk(self):
        """Initialize RTK collector, NTRIP client, and flight recorder."""
        import yaml
        rtk_config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "rtk_config.yaml")
        try:
            with open(rtk_config_path) as f:
                rtk_cfg = yaml.safe_load(f)
        except Exception as e:
            log.warning(f"Could not load rtk_config.yaml: {e} — using defaults")
            rtk_cfg = {}

        # RTK Collector
        self.rtk_collector = RTKCollector(rtk_cfg)
        if self.rtk_collector.connect():
            self.rtk_collector.start()
        else:
            log.warning("RTK F9P not connected — recording without ground truth")
            self.rtk_collector = None

        # NTRIP Client
        ntrip_cfg = rtk_cfg.get("ntrip", {})
        if ntrip_cfg.get("enabled", False) and self.rtk_collector:
            self.ntrip_client = NTRIPClient(
                ntrip_cfg,
                serial_write_fn=self.rtk_collector.write_to_serial)
            # Provide GGA source from RTK fixes
            def _gga_source():
                fix = self.rtk_collector.last_fix
                if fix and fix.fix_type >= 3:
                    return generate_gga(fix.lat_deg, fix.lon_deg, fix.alt_m)
                return None
            self.ntrip_client.set_gga_source(_gga_source)
            self.ntrip_client.start()

        # Flight Recorder
        rec_cfg = rtk_cfg.get("recording", {})
        self.flight_recorder = FlightRecorder(
            output_dir=rec_cfg.get("output_dir", "flight_data"),
            flush_interval_s=rec_cfg.get("flush_interval_s", 1.0))

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

        # Hit record — every flight is a potential YouTube video or incident report
        if self.flight_recorder:
            from datetime import datetime
            self.flight_recorder.start_session({
                "connection": self.connection_string,
                "baud": self.baud,
                "hz": self.update_hz,
                "start_time": datetime.now().isoformat(),
            })

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
            # Clean up RTK — pack up the ground station like a responsible adult
            if self.flight_recorder:
                self.flight_recorder.stop_session()
            if self.ntrip_client:
                self.ntrip_client.stop()
            if self.rtk_collector:
                log.info(self.rtk_collector.summary())
                self.rtk_collector.stop()

    # ── main loop (the heartbeat — if this stops, we're in trouble) ──
    def _main_loop(self):
        global _running
        while _running:
            loop_start = time.monotonic()

            msg = self.bridge.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                # Watchdog: if the IMU ghosts us, start worrying
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

            # ── periodic housework (logging, printing, pretending we have it together) ──
            if t_now - self._last_log >= self.LOG_INTERVAL_S:
                self._log_state(t_now)
                self._last_log = t_now

            if t_now - self._last_print >= self.PRINT_INTERVAL_S:
                self._print_state(t_now)
                self._last_print = t_now

            if t_now - self._last_stats >= self.STATS_INTERVAL_S:
                self._print_stats(t_now)
                self._last_stats = t_now

            # ── loop timing (are we actually hitting 100Hz? let's find out) ──
            loop_ms = (time.monotonic() - loop_start) * 1000.0
            self.loop_monitor.record_loop(loop_ms)

    # ── message dispatch (the giant switch statement we all knew was coming) ──
    def _dispatch_message(self, mtype: str, msg, t_now: float):
        # ── IMU → predict (the bread and butter — 100Hz of raw sensor chaos) ──
        if mtype == "RAW_IMU":
            accel, gyro = self.bridge.parse_raw_imu(msg)

            # Feed the primary IMU into the fusion blender as channel 0
            self.imu_fusion.update_imu(0, accel, gyro, t_now)

            # Get dt from hardware timestamps — because time.time() lies
            dt = self.time_sync.compute_dt(msg)
            self._last_imu_t = t_now

            # Still warming up? Keep collecting samples. Patience is a virtue.
            if not self.eskf._initialized:
                self._init_accel_buf.append(accel.copy())
                if len(self._init_accel_buf) >= self.INIT_SAMPLES:
                    self._try_initialize()
                return

            # Blend all IMU channels into one trustworthy reading (democracy of sensors)
            fused_accel, fused_gyro, imu_conf = self.imu_fusion.get_fused(t_now)

            # Vibration check — if the drone's shaking, widen the noise model
            vib_level = self.imu_fusion.vibration_level
            self.mht.scale_process_noise(vib_level)

            # Snapshot before prediction — in case GPS shows up late with Starbucks
            x_prev = self.eskf.x.copy()
            U_prev = self.eskf.U.copy()

            self.mht.predict(fused_accel, fused_gyro, dt)
            self.dr.update(fused_accel, fused_gyro, dt)
            self._imu_count += 1
            
            # Save to the time-travel buffer for late measurements
            self._oosm_buffer.append((t_now, x_prev, U_prev, fused_accel, fused_gyro, dt))

            # Black box recording — future crash investigators will thank us
            if self.flight_recorder and self.flight_recorder.is_recording:
                self.flight_recorder.record_imu(t_now, fused_accel, fused_gyro)

            # ZUPT — is this thing sitting still? If so, velocity MUST be zero. Science.
            accel_magnitude = np.linalg.norm(fused_accel)
            gyro_magnitude = np.linalg.norm(fused_gyro)
            horiz_accel = float(np.linalg.norm(fused_accel[0:2]))
            is_stationary = (
                abs(accel_magnitude - 9.80665) < 0.3 and
                gyro_magnitude < 0.02 and
                horiz_accel < 0.5
            )

            if is_stationary:
                self._zupt_count += 1
                if self._zupt_count > 20:  # 200ms of "yep, definitely not moving" → ZUPT
                    self.mht.update_zupt()
                    self._zupt_total += 1
            else:
                self._zupt_count = 0

        elif mtype == "SCALED_IMU2":
            # Backup IMU (the wingman) — Cube Orange ICM-20948 as channel 1
            accel2, gyro2 = self.bridge.parse_scaled_imu(msg)
            self.imu_fusion.update_imu(1, accel2, gyro2, t_now)

        # ── Barometer (the weather-dependent altitude guess) ────────────
        elif mtype in ("SCALED_PRESSURE", "SCALED_PRESSURE2"):
            alt_m = self.bridge.parse_baro(msg)
            self.mht.update_baro(alt_m)
            self._baro_count += 1
            if self.flight_recorder and self.flight_recorder.is_recording:
                self.flight_recorder.record_baro(t_now, alt_m)

        # ── Magnetometer (points north, unless you're near a motor) ───
        elif mtype == "SCALED_IMU3":
            if hasattr(msg, 'xmag'):
                mx = msg.xmag * 1e-3
                my = msg.ymag * 1e-3
                mz = msg.zmag * 1e-3
                
                # RLS calibrator — because magnetometers lie constantly and we auto-correct
                self.mag_cal.update(mx, my, mz)
                
                # Apply correction (returns raw garbage until it learns the truth)
                cal_m = self.mag_cal.apply(mx, my, mz)
                cmx, cmy, cmz = cal_m[0], cal_m[1], cal_m[2]
                
                # Tilt-compensated yaw: project mag into horizontal plane
                # using current roll/pitch from the ESKF state
                euler = self.eskf.state["euler"]
                roll_c, pitch_c = euler[0], euler[1]
                cr, sr = math.cos(roll_c), math.sin(roll_c)
                cp, sp = math.cos(pitch_c), math.sin(pitch_c)
                mag_x_h = cmx * cp + cmy * sr * sp + cmz * cr * sp
                mag_y_h = cmy * cr - cmz * sr
                yaw_rad = math.atan2(-mag_y_h, mag_x_h)
                    
                if self.params.get("MAG_3D", 0.0) > 0.5:
                    mag_norm = np.linalg.norm(cal_m)
                else:
                    mag_norm = np.sqrt(cmx**2 + cmy**2)

                self.mht.update_mag(yaw_rad, mag_norm=mag_norm,
                               t_now=t_now)
                self._mag_count += 1
                if self.flight_recorder and self.flight_recorder.is_recording:
                    self.flight_recorder.record_mag(
                        t_now, cmx, cmy, cmz)

                # Hoard mag samples for initialization — we need these to know which way is north
                if not self.mht._initialized:
                    self._init_mag_buf.append(
                        np.array([cmx, cmy, cmz]))

        elif mtype == "ATTITUDE":
            self.bridge.last_attitude = msg

        elif mtype == "GPS_RAW_INT":
            self.bridge.last_gps = msg
            # GPS: the most trusted liar. Only fuse if we have 3D fix.
            if (self.mht._initialized and
                    hasattr(msg, 'fix_type') and msg.fix_type >= 3):
                lat = msg.lat / 1e7  # degE7 → degrees
                lon = msg.lon / 1e7
                alt = msg.alt / 1000.0  # mm → m
                hdop = msg.eph / 100.0 if hasattr(msg, 'eph') else 2.0

                latency_s = self.params.get("GPS_LATENCY_S", 0.15)  # GPS: 150ms late. Every. Single. Time.
                t_meas = t_now - latency_s

                if self.params.get("OOSM_ENABLE", 1.0) > 0.5 and len(self._oosm_buffer) > 0:
                    # OOSM time-travel: rewind the filter, apply the late GPS, fast-forward back
                    # Step 1: find where we were when GPS SHOULD have arrived
                    replay_idx = -1
                    for i in range(len(self._oosm_buffer)-1, -1, -1):
                        if self._oosm_buffer[i][0] <= t_meas:
                            replay_idx = i
                            break
                    
                    if replay_idx != -1:
                        # Step 2: ctrl+Z the filter state back in time
                        _, x_rewind, U_rewind, _, _, _ = self._oosm_buffer[replay_idx]
                        self.eskf.x = x_rewind.copy()
                        self.eskf.U = U_rewind.copy()
                        
                        # Step 3: apply the late GPS fix (better late than never)
                        self.mht.update_gps(lat, lon, alt, hdop=hdop, t_now=t_meas)
                        
                        # Step 4: fast-forward back to present with corrected history
                        for i in range(replay_idx + 1, len(self._oosm_buffer)):
                            _, _, _, a, g, dt_hist = self._oosm_buffer[i]
                            # Overwrite history with the new, less-wrong version
                            self._oosm_buffer[i] = (self._oosm_buffer[i][0], self.eskf.x.copy(), self.eskf.U.copy(), a, g, dt_hist)
                            self.mht.predict(a, g, dt_hist)
                    else:
                        # Buffer ran out — we can't time-travel that far. Just wing it.
                        self.mht.update_gps(lat, lon, alt, hdop=hdop, t_now=t_now)
                else:
                    # Taylor extrapolation — the "good enough" compensation for GPS delay
                    v_ned = self.eskf.x[3:6]
                    lat_adj = lat + (v_ned[0] * latency_s) / 111320.0
                    lon_adj = lon + (v_ned[1] * latency_s) / (111320.0 * math.cos(math.radians(lat)))
                    alt_adj = alt - (v_ned[2] * latency_s)
                    self.mht.update_gps(lat_adj, lon_adj, alt_adj, hdop=hdop, t_now=t_now)
                self._gps_count += 1
                # Log GPS for the flight black box
                if self.flight_recorder and self.flight_recorder.is_recording:
                    # WGS84 → NED conversion (because pilots think in meters, not degrees)
                    if self.mht._gps_origin:
                        _o = self.mht._gps_origin
                        _dlat = math.radians(lat - _o['lat'])
                        _dlon = math.radians(lon - _o['lon'])
                        _n = _dlat * 6371000.0
                        _e = _dlon * 6371000.0 * math.cos(math.radians(_o['lat']))
                        _d = -(alt - _o['alt'])
                        self.flight_recorder.record_gps(
                            t_now, _n, _e, _d, hdop)

        elif mtype == "GPS_RTCM_DATA":
            # RTCM relay — shuttle correction bytes from Mission Planner to the F9P
            if self.rtk_collector and hasattr(msg, 'data') and hasattr(msg, 'len'):
                # raw bytes, raw vibes — just forward them to the RTK receiver
                rtcm_bytes = bytes(msg.data[:msg.len])
                self.rtk_collector.write_to_serial(rtcm_bytes)

        elif mtype == "OBSTACLE_DISTANCE":
            # Lidar goes brrr — ingesting distance data from obstacle sensors
            if hasattr(msg, 'distances'):
                distances_cm = np.array(msg.distances, dtype=float)
                valid_mask = (distances_cm > 0) & (distances_cm < 65535)
                if not np.any(valid_mask):
                    return
                valid_dists = distances_cm[valid_mask] / 100.0  # cm → m
                n_pts = len(valid_dists)
                # Turn flat angle+distance data into fake 3D points (MacGyver style)
                angle_offset = getattr(msg, 'angle_offset', 0.0)
                increment = getattr(msg, 'increment', 5.0)  # degrees per bin
                indices = np.where(valid_mask)[0]
                angles_rad = np.radians(angle_offset + indices * increment)
                pts = np.column_stack([
                    valid_dists * np.cos(angles_rad),
                    valid_dists * np.sin(angles_rad),
                    np.zeros(n_pts),
                ])

                # Offload to thread — point clouds are thicc and the main loop waits for no one
                if self._lidar_future is None or self._lidar_future.done():
                    if self._lidar_future is not None:
                        safe_dist = self._lidar_future.result()
                        if safe_dist > 0:
                            self.mht.update_lidar_range(safe_dist, weight=self.params["LR_WEIGHT"])

                            if safe_dist < self.params["OBS_THRESH"]:
                                self.bridge.send_statustext(f"OBSTACLE CLOSE: {safe_dist:.1f}m", 3)

                    # Yeet the next point cloud into the thread pool
                    self._lidar_future = self.executor.submit(
                        self.lidar_radar.process_livox_cloud, pts
                    )

        elif mtype == "RADAR_TARGET":
            if not hasattr(msg, 'distance') or not hasattr(msg, 'velocity'):
                return
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
                self.mht.update_radar_velocity(*radar_vel, weight=self.params["LR_WEIGHT"])

        elif mtype == "MOCK_LIVOX_CLOUD":
            # TRN mock injection — pretend lidar for terrain matching tests
            points = np.array(msg.points)
            # AGL = flip NED down to positive up (because NED is confusing and we all know it)
            res = self.trn.process_lidar_scan(points, self.eskf.state["pos"], -self.eskf.state["pos"][2], t_now)
            if res is not None:
                self.mht.update_external(res["z"], res["z_pred"], res["H"], res["R"], source="TRN")

        elif mtype == "PARAM_REQUEST_LIST":
            self._send_all_params()

        elif mtype == "PARAM_SET":
            # Ground station says "hey change this" — Mission Planner parameter update
            param_id = msg.param_id.strip("\x00")
            if param_id in self.params:
                self.params[param_id] = msg.param_value
                self._apply_param(param_id)
                self._send_param(param_id)
                log.info(f"Param updated via MAVLink: {param_id} = {msg.param_value}")

        elif mtype == "OPTICAL_FLOW_RAD":
            if self.mht._initialized:
                # Step 1: need a rangefinder reading or this is all meaningless
                if not hasattr(msg, 'distance') or msg.distance <= 0.05:
                    return

                # Step 2: spinning too fast? Flow sensor goes blurry. Reject it.
                gyro_x_rate = abs(msg.integrated_xgyro / (msg.integration_time_us / 1e6))
                gyro_y_rate = abs(msg.integrated_ygyro / (msg.integration_time_us / 1e6))
                if gyro_x_rate > 1.5 or gyro_y_rate > 1.5:  # ~85 deg/s
                    return

                # Raw flow or pre-compensated? Depends on who we trust more: us or ArduPilot.
                use_raw_flow = self.params.get("OPTFLOW_RAW", True)
                
                if use_raw_flow:
                    # Raw flow — we'll do our own gyro compensation, thanks
                    flow_vx = msg.integrated_x
                    flow_vy = msg.integrated_y
                else:
                    # ArduPilot already subtracted gyro — trust the FC on this one
                    flow_vx = (msg.integrated_x - msg.integrated_xgyro)
                    flow_vy = (msg.integrated_y - msg.integrated_ygyro)
                
                dt_flow = msg.integration_time_us / 1e6

                if dt_flow > 0:
                    # Step 3: angular_rate × height = ground velocity. High school physics saves the day.
                    vx = (flow_vx / dt_flow) * msg.distance
                    vy = (flow_vy / dt_flow) * msg.distance
                    self.mht.update_optical_flow(
                        vx, vy, msg.distance, msg.quality,
                        enable_rot_comp=use_raw_flow)

        elif mtype == "VISION_POSITION_ESTIMATE":
            # VIO: the camera thinks it knows where we are (T265 / ORB-SLAM3)
            if self.vio.is_active:
                pos_vio = np.array([msg.x, msg.y, msg.z])
                latency_s = self.params.get("VIO_LATENCY_S", 0.05)  # 50ms default delay
                t_meas = t_now - latency_s

                quat_vio = np.array([1.0, 0.0, 0.0, 0.0])
                if hasattr(msg, 'q') and msg.q is not None:
                    quat_vio = np.array(msg.q)
                confidence = 0.8

                if self.params.get("OOSM_ENABLE", 1.0) > 0.5 and len(self._oosm_buffer) > 0:
                    replay_idx = -1
                    for i in range(len(self._oosm_buffer)-1, -1, -1):
                        if self._oosm_buffer[i][0] <= t_meas:
                            replay_idx = i
                            break
                            
                    if replay_idx != -1:
                        # Rewind — same OOSM time-travel trick as GPS
                        _, x_rewind, U_rewind, _, _, _ = self._oosm_buffer[replay_idx]
                        self.eskf.x = x_rewind.copy()
                        self.eskf.U = U_rewind.copy()
                        
                        # Apply the VIO update at the correct historical moment
                        result = self.vio.process_vio_update(t_meas, pos_vio, quat_vio, confidence)
                        if result is not None:
                            self.mht.update_external(result["pos_ned"], self.eskf.state["pos"], result["H_pos"], result["R_pos"], source="VIO_pos")
                            self.mht.update_external(np.array([result["yaw_ned"]]), np.array([self.eskf.state["euler"][2]]), result["H_yaw"], result["R_yaw"], source="VIO_yaw")
                            self._vio_count += 1
                        
                        # Fast-forward back to now with corrected trajectory
                        for i in range(replay_idx + 1, len(self._oosm_buffer)):
                            _, _, _, a, g, dt_hist = self._oosm_buffer[i]
                            self._oosm_buffer[i] = (self._oosm_buffer[i][0], self.eskf.x.copy(), self.eskf.U.copy(), a, g, dt_hist)
                            self.mht.predict(a, g, dt_hist)
                else:
                    # No time-travel? Fine, just nudge VIO position by velocity × latency
                    v_ned = self.eskf.x[3:6]
                    pos_vio_adj = pos_vio + v_ned * latency_s
                    result = self.vio.process_vio_update(t_now, pos_vio_adj, quat_vio, confidence)
                    if result is not None:
                        self.mht.update_external(result["pos_ned"], self.eskf.state["pos"], result["H_pos"], result["R_pos"], source="VIO_pos")
                        self.mht.update_external(np.array([result["yaw_ned"]]), np.array([self.eskf.state["euler"][2]]), result["H_yaw"], result["R_yaw"], source="VIO_yaw")
                        self._vio_count += 1

        # ── Safety & Health (the parental controls for the filter) ─────────
        if not self.eskf._initialized:
            return

        pos = self.eskf.state["pos"]
        vel = self.eskf.state["vel"]
        att = self.eskf.state["euler"]
        
        # 1. ML Crystal Ball — predict if the filter is about to have a bad day
        # Sniff vibration + covariance traces for early signs of divergence doom
        accel_var = float(np.var(self._init_accel_buf[-20:])) if len(self._init_accel_buf) > 0 else 0.0
        gyro_proxy = float(np.linalg.norm(self.eskf.state["gyro_bias"]))
        p_trace = float(np.trace(self.eskf.P))
        
        # Harvest ML results async — can't let sklearn hog the main loop
        if self._ml_future is None or self._ml_future.done():
            if self._ml_future is not None and self._ml_future.result() == True:
                self.bridge.send_statustext("ML FAULT - ACTIVATING SMART RTH", 2)
                self._rth_active = True
                self.bridge.set_mode("GUIDED")
                self.fault_mgr.update(t_now, ekf_healthy=True, safety_ok=True, ml_fault=True)
            
            # Fire off the next health prediction. yolo.
            self._ml_future = self.executor.submit(
                self.ml_predictor.check_health, accel_var, gyro_proxy, p_trace
            )
            
        # 1.5 Smart RTH — the "something went wrong, let's go home" autopilot
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
                    
                # Don't fly into walls on the way home. That would be embarrassing.
                safe_dist = self.lidar_radar.safe_distance_m
                if safe_dist > 0 and safe_dist < self.params["OBS_THRESH"]:
                    scale = (safe_dist / self.params["OBS_THRESH"]) ** 2
                    desired_vel *= scale
                    if safe_dist < 0.5:
                        desired_vel = np.zeros(2)
                        self.bridge.send_statustext("RTH BLOCKED - HOLDING", 3)
                
                # Altitude policy: get high enough to clear stuff, but don't bonk the ceiling
                desired_vz = 0.0
                current_alt_agl = -pos[2]  # NED → AGL
                rth_min_alt = self.params["RTH_MIN_ALT"]
                ceil_margin = self.params["RTH_CEIL_MARGIN"]

                if current_alt_agl < rth_min_alt:
                    # Peek upward with lidar before climbing — is there room?
                    lidar_range = self.lidar_radar.safe_distance_m
                    if lidar_range > 0 and lidar_range < ceil_margin:
                        # Nope, ceiling too close. Stay put.
                        desired_vz = 0.0
                        self.bridge.send_statustext(
                            f"RTH CEIL BLOCKED: {lidar_range:.1f}m", 3)
                    elif lidar_range <= 0:
                        # No upward-looking lidar? We're NOT climbing blind.
                        desired_vz = 0.0
                        self.bridge.send_statustext(
                            "RTH NO LIDAR - HOLD ALT", 4)
                    else:
                        # All clear — going up! (negative in NED because NED is backwards)
                        desired_vz = -0.5

                self.bridge.send_velocity_target(desired_vel[0], desired_vel[1], desired_vz)
        
        # 2. Hard safety — the last line of defense before things get expensive
        safety_action = self.safety.check(pos, vel, att)

        if safety_action == SafetyAction.FORCE_DISARM:
            # PULL THE PLUG — tell Pixhawk to kill motors immediately
            self.bridge.send_statustext("INS CRITICAL FAULT - DISARM", 2)
            # TODO: actually send MAV_CMD_COMPONENT_ARM_DISARM (currently just yelling about it)

        # Phase 2: if we're REALLY lost, just land. Better on the ground than in a tree.
        self._check_emergency_landing(t_now)

    def _check_emergency_landing(self, t_now):
        # If the filter has no idea where we are (20m+ uncertainty), give up and land
        if self.eskf.health == EKFHealth.FAULT:
            pos_var = self.eskf.P[0,0] + self.eskf.P[1,1]
            if pos_var > 400.0:  # std > 20m
                if getattr(self, "_emergency_land_triggered", False) == False:
                    log.critical("TOTAL OBSERVABILITY COLLAPSE (Uncertainty > 20m). INITIATING EMERGENCY LANDING!")
                    self.bridge.send_statustext("NAV EMERGENCY: LANDING", 1)
                    # MAV_CMD_NAV_LAND = 21 (the command that means "put it on the ground NOW")
                    try:
                        self.bridge._conn.mav.command_long_send(
                            self.bridge._conn.target_system, self.bridge._conn.target_component,
                            21, 0,
                            0, 0, 0, 0, 0, 0, 0
                        )
                    except Exception as e:
                        log.error(f"Failed to send emergency land command: {e}")
                    self._emergency_land_triggered = True

        health = self.eskf.health

        # VIO alignment — only trust the camera once the ESKF knows what's up
        if (health == EKFHealth.HEALTHY and
                self.vio._enabled and not self.vio._initialized):
            self.vio.initialize(
                self.eskf.state["pos"], self.eskf.state["quat"],
                np.zeros(3), np.array([1.0, 0, 0, 0]))
            log.info("VIO pipeline aligned to ESKF frame")

        # Vision injection: only if ESKF is healthy AND the safety monitor isn't screaming
        can_inject = (health == EKFHealth.HEALTHY and self.safety.is_injection_safe)

        if not can_inject and self._vision_enabled:
            self._vision_enabled = False
            log.error(f"Vision injection DISABLED! Health={health.name}, Safety={self.safety.last_action.name}")
            self.bridge.send_statustext("INS: Vision disabled", 4)
        elif can_inject and not self._vision_enabled:
            self._vision_enabled = True
            log.info("Vision injection ENABLED")
            self.bridge.send_statustext("INS: Vision enabled", 6)

    # ── sensor init (sit still, stare at gravity, figure out which way is up) ──
    @property
    def eskf(self):
        """Access the primary ESKF instance from the MHT manager."""
        if not hasattr(self, 'mht'):
            raise AttributeError(
                "ESKF accessed before MHTManager initialization — "
                "check __init__ ordering")
        return self.mht.primary

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

    # ── MAVLink param server (so Mission Planner can boss us around) ──
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
            self.eskf.Q = self.eskf.Q_base * val  # Always scale from baseline. We learned this the hard way.

    # ── helpers (the unsung heroes) ────────────────────────────
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

        # Structured log — for the data scientists who'll analyze this later
        self.s_logger.log_state(
            t=elapsed,
            state=self.eskf.state,
            covariance=self.eskf.P,
            health_status=self.eskf.health.name,
            safety_action=self.safety.last_action.name,
            timing_ms=self.time_sync.latency_s * 1000.0
        )

        # Black box: ESKF state + RTK ground truth (for post-mortem, hopefully not literally)
        if self.flight_recorder and self.flight_recorder.is_recording:
            self.flight_recorder.record_eskf_state(
                t, self.eskf.state, self.eskf.P,
                self.eskf.health.name, self.eskf.baro_bias)
            # Grab the latest RTK ground truth — this is the "answer key" for our filter
            if self.rtk_collector and self.rtk_collector.last_fix:
                fix = self.rtk_collector.last_fix
                self.flight_recorder.record_rtk(t, fix)

    def _print_state(self, t: float):
        elapsed = t - self._start_time
        pos = self.eskf.state["pos"]
        att = np.degrees(self.eskf.state["euler"])
        health = self.eskf.health.name

        # NaN guard — if math broke, at least say so on screen
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

        rtk_info = ""
        if self.rtk_collector:
            rs = self.rtk_collector.stats
            rtk_info = (f"  RTK: {rs.rtk_fixed_count} FIXED / "
                        f"{rs.total_fixes} total")
            
            # Push RTK stats to Mission Planner so the GCS operator can feel smart
            if self.rtk_collector.last_fix:
                fix = self.rtk_collector.last_fix
                self.bridge.send_named_value_int("RTK_Fix", fix.fix_type)
                self.bridge.send_named_value_int("RTK_Sats", fix.n_sats)
                self.bridge.send_named_value_float("RTK_HAcc", fix.h_acc_m)

        print(f"| Updates     | IMU:{self._imu_count} Baro:{self._baro_count} Mag:{self._mag_count} GPS:{self._gps_count}")
        
        # Health dashboard — the filter's vital signs
        print(f"| Health      | Cholesky Fails: {self.eskf.cholesky_failures} | Cov Repairs: {self.eskf.covariance_repairs} | Spike count: {self.eskf.innovation_spikes}")
        print(f"| Condition U | {self.eskf.cond_num:.2e} (Min diag: {self.eskf.min_diag_U:.2e}, Max diag: {self.eskf.max_diag_U:.2e})")

        log.info(
            f"Stats @ {elapsed:.1f}s — "
            f"IMU={self._imu_count} ({eff_hz:.0f} Hz)  "
            f"Baro={self._baro_count}  Mag={self._mag_count}  "
            f"GPS={self._gps_count}  VIO={self._vio_count}  "
            f"Loop avg={stats['avg']:.1f}ms max={stats['max']:.1f}ms  "
            f"Overruns={stats['overruns']}{rtk_info}"
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

        if self.rtk_collector:
            log.info("  ── RTK Collection ──")
            log.info(self.rtk_collector.summary())
        if self.flight_recorder and self.flight_recorder.session_dir:
            log.info(f"  Flight data   : {self.flight_recorder.session_dir}")
            log.info(f"  Samples       : {self.flight_recorder.sample_counts}")


# ═══════════════════════════════════════════════════════════════
# CLI — because every good system starts from the command line
def parse_args():
    p = argparse.ArgumentParser(
        description="INS Navigation — Pixhawk Cube Orange + RPi4 (ESKF)")
    p.add_argument(
        "--connection", "-c",
        default="/dev/ttyAMA0",
        help="MAVLink connection string",
    )
    p.add_argument("--rtk", action="store_true",
                   help="Enable RTK ground truth collection")
    p.add_argument("--baud", "-b",  type=int, default=921600)
    p.add_argument("--hz",         type=int, default=100,
                   help="EKF update rate in Hz (50 or 100)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ins  = INSNavSys(args.connection, args.baud, args.hz,
                     rtk_enabled=args.rtk)
    ins.run()
