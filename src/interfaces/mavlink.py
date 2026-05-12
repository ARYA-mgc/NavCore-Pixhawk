#!/usr/bin/env python3
# mavlink_bridge.py

import math
import time
import logging
import numpy as np

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mav
except ImportError:
    raise ImportError(
        "pymavlink not installed.\n"
        "Run:  pip install pymavlink"
    )

log = logging.getLogger("mavlink_bridge")

# ─── physical scale factors for Cube Orange sensor raw values ───
# ICM-42688 accel: ±16g  → 2048 LSB/g
ACCEL_SCALE = 9.80665 / 2048.0      # m/s²  per LSB
# ICM-42688 gyro : ±2000 °/s → 16.384 LSB/(°/s)
GYRO_SCALE  = (math.pi / 180.0) / 16.384  # rad/s per LSB
# RM3100 mag (via SCALED_IMU3): already in milli-Gauss
MAG_SCALE   = 1e-3                   # Gauss per mGauss
# Barometer: pressure in hPa, temp in °C/100
SEA_LEVEL_PA  = 101325.0
BARO_EXPONENT = 1.0 / 5.257

# ────────────────────────────────────────────────────────────────
class MAVLinkBridge:
    # the phone line to the pixhawk — handles all the chatter

    def __init__(self, connection_string: str = "/dev/ttyAMA0",
                 baud: int = 921600):
        self.connection_string = connection_string
        self.baud              = baud
        self._conn             = None

        # latest parsed data (for cross-check, not EKF input)
        self.last_attitude = None
        self.last_gps      = None
        self.last_heartbeat_t = 0.0

    # ── connection ──────────────────────────────────────────────
    def connect(self):
        # Open MAVLink connection.  Supports:
        log.info(f"Opening MAVLink connection → {self.connection_string}")
        self._conn = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud,
            source_system=255,        # GCS system id
            source_component=0,
            autoreconnect=True,
            dialect="ardupilotmega",
        )
        log.info("MAVLink port opened")

    def wait_heartbeat(self, timeout: float = 30.0):
        # wait for the pixhawk to say hi
        log.info("Waiting for heartbeat …")
        self._conn.wait_heartbeat(timeout=timeout)
        self.last_heartbeat_t = time.monotonic()
        tgt = self._conn.target_system
        log.info(f"Heartbeat from sysid={tgt}  "
                 f"type={self._conn.flightmode}")

    def close(self):
        if self._conn:
            self._conn.close()
            log.info("MAVLink connection closed")

    # ── data stream requests ────────────────────────────────────
    def request_data_streams(self, hz: int = 100):
        # Ask Pixhawk to send sensor data at desired rate.
        conn = self._conn
        sysid = conn.target_system
        compid = conn.target_component

        # Legacy stream groups (ArduPilot < 4.1)
        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
        ]
        for stream in streams:
            conn.mav.request_data_stream_send(
                sysid, compid,
                stream,
                hz,
                1,   # start=1
            )

        # Per-message interval (ArduPilot 4.1+ / Copter 4.x)
        # interval_us = 1_000_000 / hz
        interval_us = int(1_000_000 / hz)
        msg_ids = [
            27,   # RAW_IMU
            29,   # SCALED_PRESSURE
            116,  # SCALED_IMU2
            129,  # SCALED_IMU3
            30,   # ATTITUDE
            24,   # GPS_RAW_INT
            106,  # OPTICAL_FLOW_RAD
        ]
        for mid in msg_ids:
            conn.mav.command_long_send(
                sysid, compid,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(mid),
                float(interval_us),
                0, 0, 0, 0, 0,
            )

        log.info(f"Data streams requested @ {hz} Hz")

    # ── receive ─────────────────────────────────────────────────
    def recv_match(self, blocking: bool = True, timeout: float = 0.05):
        # Return next MAVLink message or None.
        return self._conn.recv_match(
            blocking=blocking,
            timeout=timeout,
        )

    # ── parsers ─────────────────────────────────────────────────
    @staticmethod
    def parse_raw_imu(msg) -> tuple:
        # RAW_IMU → (accel_m_s2[3], gyro_rad_s[3])
        accel = np.array([
            msg.xacc * ACCEL_SCALE,
            msg.yacc * ACCEL_SCALE,
            msg.zacc * ACCEL_SCALE,
        ], dtype=float)

        gyro = np.array([
            msg.xgyro * GYRO_SCALE,
            msg.ygyro * GYRO_SCALE,
            msg.zgyro * GYRO_SCALE,
        ], dtype=float)

        return accel, gyro

    @staticmethod
    def parse_scaled_imu(msg) -> tuple:
        # SCALED_IMU2 → (accel m/s², gyro rad/s)
        accel = np.array([
            msg.xacc * 1e-3 * 9.80665,
            msg.yacc * 1e-3 * 9.80665,
            msg.zacc * 1e-3 * 9.80665,
        ], dtype=float)

        gyro = np.array([
            msg.xgyro * 1e-3,
            msg.ygyro * 1e-3,
            msg.zgyro * 1e-3,
        ], dtype=float)

        return accel, gyro

    @staticmethod
    def parse_baro(msg) -> float:
        # SCALED_PRESSURE / SCALED_PRESSURE2
        p_hpa = msg.press_abs          # hecto-Pascals
        p_pa  = p_hpa * 100.0
        # ISA altitude
        alt_m = 44330.0 * (1.0 - (p_pa / SEA_LEVEL_PA) ** BARO_EXPONENT)
        return alt_m

    @staticmethod
    def parse_mag_yaw(msg) -> float | None:
        # SCALED_IMU3 → yaw in radians.
        mx = msg.xmag * MAG_SCALE
        my = msg.ymag * MAG_SCALE
        mz = msg.zmag * MAG_SCALE

        mag_norm = math.sqrt(mx*mx + my*my + mz*mz)
        if mag_norm < 0.05:   # sanity: Earth field ~0.25-0.65 Gauss
            return None

        # Yaw from horizontal components (assumes level-ish flight)
        yaw_rad = math.atan2(-my, mx)
        return yaw_rad

    # ── command helpers ─────────────────────────────────────────
    def arm(self):
        # ARM — props WILL spin, don't lose a finger
        log.warning("Sending ARM command")
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def disarm(self, force: bool = False):
        # kill the motors, we're done
        log.info("Sending DISARM command")
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,                      # disarm
            21196.0 if force else 0,  # magic force-disarm value
            0, 0, 0, 0, 0,
        )

    def set_mode(self, mode_name: str):
        # switch flight mode on the pixhawk
        mode_id = self._conn.mode_mapping().get(mode_name.upper())
        if mode_id is None:
            log.error(f"Unknown mode: {mode_name}")
            return
        self._conn.mav.set_mode_send(
            self._conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        log.info(f"Mode set to {mode_name}")

    def send_statustext(self, text: str,
                        severity: int = mavutil.mavlink.MAV_SEVERITY_INFO):
        # put a message on the ground station screen
        encoded = text[:50].encode("utf-8").ljust(50, b"\x00")
        self._conn.mav.statustext_send(severity, encoded)

    def send_vision_position(self, pos: np.ndarray,
                              q: np.ndarray, t_us: int = 0):
        # tell the pixhawk where we think we are
        if t_us == 0:
            t_us = int(time.monotonic() * 1e6)
        self._conn.mav.vision_position_estimate_send(
            t_us,
            float(pos[0]), float(pos[1]), float(pos[2]),
            0.0, 0.0, 0.0,      # roll pitch yaw (optional)
        )
