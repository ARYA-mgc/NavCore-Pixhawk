#!/usr/bin/env python3
# MAVLink bridge.
# The phone line to the Pixhawk.

import math
import time
import logging
import numpy as np

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mav  # noqa: F401
except ImportError:
    raise ImportError("pymavlink not installed.\nRun:  pip install pymavlink")

log = logging.getLogger("mavlink_bridge")

# ─── physical scale factors for Cube Orange sensor raw values ───
# ICM-42688 accel: ±16g  → 2048 LSB/g
ACCEL_SCALE = 9.80665 / 2048.0  # m/s²  per LSB
# ICM-42688 gyro : ±2000 °/s → 16.384 LSB/(°/s)
GYRO_SCALE = (math.pi / 180.0) / 16.384  # rad/s per LSB
# RM3100 mag (via SCALED_IMU3): already in milli-Gauss
MAG_SCALE = 1e-3  # Gauss per mGauss
# Barometer: pressure in hPa, temp in °C/100
SEA_LEVEL_PA = 101325.0
BARO_EXPONENT = 1.0 / 5.257


# ────────────────────────────────────────────────────────────────
class MAVLinkBridge:
    # MAVLink communication bridge.

    def __init__(self, connection_string: str = "/dev/ttyAMA0", baud: int = 921600):
        self.connection_string = connection_string
        self.baud = baud
        self._conn = None

        # latest parsed data (for cross-check, not EKF input)
        self.last_attitude = None
        self.last_gps = None
        self.last_heartbeat_t = 0.0

    # ── connection ──────────────────────────────────────────────
    def connect(self):
        # Open MAVLink connection.  Supports:
        log.info(f"Opening MAVLink connection → {self.connection_string}")
        self._conn = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud,
            source_system=1,  # Vehicle system id
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,  # 191
            autoreconnect=True,
            dialect="ardupilotmega",
        )
        log.info("MAVLink port opened (SYSID=1, COMPID=191)")

    def wait_heartbeat(self, timeout: float = 30.0):
        # Wait for heartbeat.
        log.info("Waiting for heartbeat …")
        self._conn.wait_heartbeat(timeout=timeout)  # type: ignore
        self.last_heartbeat_t = time.monotonic()
        tgt = self._conn.target_system  # type: ignore
        log.info(f"Heartbeat from sysid={tgt}  type={self._conn.flightmode}")  # type: ignore

    def close(self):
        if self._conn:
            self._conn.close()
            log.info("MAVLink connection closed")

    # ── data stream requests ────────────────────────────────────
    def request_data_streams(self, hz: int = 100):
        # Ask Pixhawk to send sensor data at desired rate.
        conn = self._conn
        sysid = conn.target_system  # type: ignore
        compid = conn.target_component  # type: ignore

        # Legacy stream groups (ArduPilot < 4.1)
        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
        ]
        for stream in streams:
            conn.mav.request_data_stream_send(  # type: ignore
                sysid,
                compid,
                stream,
                hz,
                1,  # start=1
            )

        # Per-message interval (ArduPilot 4.1+ / Copter 4.x)
        # interval_us = 1_000_000 / hz
        interval_us = int(1_000_000 / hz)
        msg_ids = [
            27,  # RAW_IMU
            29,  # SCALED_PRESSURE
            116,  # SCALED_IMU2
            129,  # SCALED_IMU3
            30,  # ATTITUDE
            24,  # GPS_RAW_INT
            106,  # OPTICAL_FLOW_RAD
        ]
        for mid in msg_ids:
            conn.mav.command_long_send(  # type: ignore
                sysid,
                compid,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(mid),
                float(interval_us),
                0,
                0,
                0,
                0,
                0,
            )

        log.info(f"Data streams requested @ {hz} Hz")

    # ── receive ─────────────────────────────────────────────────
    def recv_match(self, blocking: bool = True, timeout: float = 0.05):
        # Return next MAVLink message or None.
        return self._conn.recv_match(  # type: ignore
            blocking=blocking,
            timeout=timeout,
        )

    # ── parsers ─────────────────────────────────────────────────
    @staticmethod
    def parse_raw_imu(msg) -> tuple:
        # RAW_IMU → (accel_m_s2[3], gyro_rad_s[3])
        accel = np.array(
            [
                msg.xacc * ACCEL_SCALE,
                msg.yacc * ACCEL_SCALE,
                msg.zacc * ACCEL_SCALE,
            ],
            dtype=float,
        )

        gyro = np.array(
            [
                msg.xgyro * GYRO_SCALE,
                msg.ygyro * GYRO_SCALE,
                msg.zgyro * GYRO_SCALE,
            ],
            dtype=float,
        )

        return accel, gyro

    @staticmethod
    def parse_scaled_imu(msg) -> tuple:
        # SCALED_IMU2 → (accel m/s², gyro rad/s)
        accel = np.array(
            [
                msg.xacc * 1e-3 * 9.80665,
                msg.yacc * 1e-3 * 9.80665,
                msg.zacc * 1e-3 * 9.80665,
            ],
            dtype=float,
        )

        gyro = np.array(
            [
                msg.xgyro * 1e-3,
                msg.ygyro * 1e-3,
                msg.zgyro * 1e-3,
            ],
            dtype=float,
        )

        return accel, gyro

    @staticmethod
    def parse_baro(msg) -> float:
        # SCALED_PRESSURE / SCALED_PRESSURE2
        p_hpa = msg.press_abs  # hecto-Pascals
        p_pa = p_hpa * 100.0
        # ISA altitude
        alt_m = 44330.0 * (1.0 - (p_pa / SEA_LEVEL_PA) ** BARO_EXPONENT)
        return alt_m

    @staticmethod
    def parse_mag_yaw(msg, roll: float = 0.0, pitch: float = 0.0) -> float | None:
        """SCALED_IMU3 → tilt-compensated yaw in radians.

        Args:
            msg: MAVLink SCALED_IMU3 message with xmag/ymag/zmag fields.
            roll: Current roll angle (rad) for tilt compensation.
            pitch: Current pitch angle (rad) for tilt compensation.

        If roll and pitch are both zero (default), this reduces to the
        level-flight approximation. For accurate yaw under tilt, pass
        the current ESKF roll/pitch estimates.
        """
        mx = msg.xmag * MAG_SCALE
        my = msg.ymag * MAG_SCALE
        mz = msg.zmag * MAG_SCALE

        mag_norm = math.sqrt(mx * mx + my * my + mz * mz)
        if mag_norm < 0.05:  # sanity: Earth field ~0.25-0.65 Gauss
            return None

        # Tilt-compensated yaw: project mag vector into horizontal plane
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        mag_x_h = mx * cp + my * sr * sp + mz * cr * sp
        mag_y_h = my * cr - mz * sr
        yaw_rad = math.atan2(-mag_y_h, mag_x_h)
        return yaw_rad

    # ── command helpers ─────────────────────────────────────────
    def arm(self):
        # ARM — Arm vehicle.
        log.warning("Sending ARM command")
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def disarm(self, force: bool = False):
        # Disarm vehicle.
        log.info("Sending DISARM command")
        self._conn.mav.command_long_send(  # type: ignore
            self._conn.target_system,  # type: ignore
            self._conn.target_component,  # type: ignore
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,  # disarm
            21196.0 if force else 0,  # Force disarm value
            0,
            0,
            0,
            0,
            0,
        )

    def set_mode(self, mode_name: str):
        # Set flight mode.
        if self._conn is None or not hasattr(self._conn, "mode_mapping"):
            log.debug(f"set_mode({mode_name}) skipped: no MAVLink link")
            return
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

    def send_statustext(
        self, text: str, severity: int = mavutil.mavlink.MAV_SEVERITY_INFO
    ):
        # Send status text.
        encoded = text[:50].encode("utf-8").ljust(50, b"\x00")
        self._conn.mav.statustext_send(severity, encoded)  # type: ignore

    def send_vision_position(self, pos: np.ndarray, q: np.ndarray, t_us: int = 0):
        # Send vision position estimate with attitude from quaternion.
        if t_us == 0:
            t_us = int(time.monotonic() * 1e6)

        # Convert quaternion [qw, qx, qy, qz] to Euler angles
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))  # clamp for asin safety
        pitch = math.asin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self._conn.mav.vision_position_estimate_send(  # type: ignore
            t_us,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            roll,
            pitch,
            yaw,
        )

    def send_velocity_target(self, vx: float, vy: float, vz: float):
        # Sends velocity targets to Pixhawk (requires GUIDED mode)
        # Type mask to use velocity only: 0b0000110111000111 (3527)
        try:
            self._conn.mav.set_position_target_local_ned_send(  # type: ignore
                0,  # time_boot_ms
                self._conn.target_system,  # type: ignore
                self._conn.target_component,  # type: ignore
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                3527,  # type_mask
                0,
                0,
                0,  # x, y, z
                vx,
                vy,
                vz,  # vx, vy, vz
                0,
                0,
                0,  # afx, afy, afz
                0,
                0,  # yaw, yaw_rate
            )
        except Exception as e:
            log.error(f"Failed to send velocity target: {e}")

    def send_named_value_float(self, name: str, value: float):
        # send a float to Mission Planner's status tab
        if not self._conn:
            return
        # name must be 10 characters max
        name_bytes = name[:10].encode("utf-8")
        t_ms = int(time.monotonic() * 1000) % 4294967296
        try:
            self._conn.mav.named_value_float_send(t_ms, name_bytes, float(value))
        except Exception:
            pass

    def send_named_value_int(self, name: str, value: int):
        # send an int to Mission Planner's status tab
        if not self._conn:
            return
        name_bytes = name[:10].encode("utf-8")
        t_ms = int(time.monotonic() * 1000) % 4294967296
        try:
            self._conn.mav.named_value_int_send(t_ms, name_bytes, int(value))
        except Exception:
            pass
