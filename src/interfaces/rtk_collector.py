#!/usr/bin/env python3
"""RTK Ground Truth Collector — u-blox F9P UBX NAV-PVT / NAV-HPPOSLLH.

Threaded serial reader that parses centimeter-level RTK fixes from the
u-blox F9P and converts them to local NED for ground truth comparison.

Supports UART, USB, and auto-detection of the F9P device.
"""

import math
import time
import struct
import logging
import threading
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque
from collections import deque

log = logging.getLogger("rtk_collector")

# ── Constants ────────────────────────────────────────────────
R_EARTH = 6371000.0       # mean Earth radius (m)

# UBX protocol constants
UBX_SYNC1 = 0xB5
UBX_SYNC2 = 0x62
UBX_CLASS_NAV = 0x01
UBX_CLASS_CFG = 0x06
UBX_ID_NAV_PVT = 0x07       # Position, Velocity, Time
UBX_ID_NAV_HPPOSLLH = 0x14  # High-precision geodetic position
UBX_ID_CFG_MSG = 0x01       # Set message rate


class RTKFixType(IntEnum):
    """u-blox NAV-PVT fixType + flags."""
    NO_FIX     = 0
    DEAD_RECK  = 1
    FIX_2D     = 2
    FIX_3D     = 3
    GNSS_DR    = 4   # GNSS + dead reckoning
    TIME_ONLY  = 5


class CarrierSolution(IntEnum):
    """u-blox NAV-PVT carrSoln field (bits 6-7 of flags)."""
    NO_CARRIER = 0
    RTK_FLOAT  = 1
    RTK_FIXED  = 2


@dataclass
class RTKFix:
    """A single RTK ground truth fix."""
    timestamp_s: float         # monotonic system time
    tow_ms: int                # GPS time of week (ms)
    lat_deg: float             # WGS-84 latitude (degrees)
    lon_deg: float             # WGS-84 longitude (degrees)
    alt_m: float               # height above ellipsoid (m)
    pos_ned: np.ndarray        # local NED position (m)
    vel_ned: np.ndarray        # NED velocity (m/s)
    fix_type: int              # RTKFixType value
    carrier_solution: int      # CarrierSolution value
    h_acc_m: float             # horizontal accuracy estimate (m)
    v_acc_m: float             # vertical accuracy estimate (m)
    n_sats: int                # number of satellites used
    pdop: float                # position DOP


@dataclass
class RTKStats:
    """Accumulated RTK collection statistics."""
    total_fixes: int = 0
    rtk_fixed_count: int = 0
    rtk_float_count: int = 0
    fix_3d_count: int = 0
    no_fix_count: int = 0
    rejected_accuracy: int = 0
    min_hacc_m: float = 999.0
    max_hacc_m: float = 0.0
    mean_hacc_m: float = 0.0
    _hacc_sum: float = 0.0
    _hacc_n: int = 0
    max_sats: int = 0
    min_sats: int = 99

    def update(self, fix: RTKFix, accepted: bool):
        self.total_fixes += 1
        if fix.carrier_solution == CarrierSolution.RTK_FIXED:
            self.rtk_fixed_count += 1
        elif fix.carrier_solution == CarrierSolution.RTK_FLOAT:
            self.rtk_float_count += 1
        elif fix.fix_type == RTKFixType.FIX_3D:
            self.fix_3d_count += 1
        else:
            self.no_fix_count += 1
        if not accepted:
            self.rejected_accuracy += 1
        self.min_hacc_m = min(self.min_hacc_m, fix.h_acc_m)
        self.max_hacc_m = max(self.max_hacc_m, fix.h_acc_m)
        self._hacc_sum += fix.h_acc_m
        self._hacc_n += 1
        self.mean_hacc_m = self._hacc_sum / self._hacc_n
        self.max_sats = max(self.max_sats, fix.n_sats)
        self.min_sats = min(self.min_sats, fix.n_sats)


class UBXNavParser:
    """Parses UBX NAV-PVT and NAV-HPPOSLLH binary frames.

    Reuses the same UBX frame-sync and Fletcher-8 checksum approach
    as gps_tight.py's UBXParser, extended for navigation messages.
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes) -> List[dict]:
        """Feed raw serial bytes. Returns list of parsed message dicts."""
        self._buffer.extend(data)
        messages = []

        while len(self._buffer) >= 8:
            # Find sync pattern
            idx = self._find_sync()
            if idx < 0:
                self._buffer.clear()
                break
            if idx > 0:
                self._buffer = self._buffer[idx:]

            if len(self._buffer) < 6:
                break

            # Parse header
            cls = self._buffer[2]
            msg_id = self._buffer[3]
            length = struct.unpack_from('<H', self._buffer, 4)[0]
            total_len = 6 + length + 2  # header + payload + checksum

            if len(self._buffer) < total_len:
                break

            # Verify Fletcher-8 checksum
            if not self._verify_checksum(self._buffer[:total_len]):
                self._buffer = self._buffer[2:]
                continue

            payload = bytes(self._buffer[6:6 + length])

            # Parse known message types
            if cls == UBX_CLASS_NAV and msg_id == UBX_ID_NAV_PVT:
                msg = self._parse_nav_pvt(payload)
                if msg:
                    messages.append(msg)

            elif cls == UBX_CLASS_NAV and msg_id == UBX_ID_NAV_HPPOSLLH:
                msg = self._parse_nav_hpposllh(payload)
                if msg:
                    messages.append(msg)

            self._buffer = self._buffer[total_len:]

        return messages

    def _find_sync(self) -> int:
        for i in range(len(self._buffer) - 1):
            if (self._buffer[i] == UBX_SYNC1 and
                    self._buffer[i + 1] == UBX_SYNC2):
                return i
        return -1

    def _verify_checksum(self, frame: bytearray) -> bool:
        """Fletcher-8 checksum (same as gps_tight.py)."""
        ck_a, ck_b = 0, 0
        for b in frame[2:-2]:
            ck_a = (ck_a + b) & 0xFF
            ck_b = (ck_b + ck_a) & 0xFF
        return ck_a == frame[-2] and ck_b == frame[-1]

    def _parse_nav_pvt(self, payload: bytes) -> Optional[dict]:
        """Parse UBX NAV-PVT (92 bytes).

        Key fields:
            iTOW(4) + year(2) + month(1) + day(1) + hour(1) + min(1) +
            sec(1) + valid(1) + tAcc(4) + nano(4) + fixType(1) + flags(1) +
            flags2(1) + numSV(1) + lon(4) + lat(4) + height(4) + hMSL(4) +
            hAcc(4) + vAcc(4) + velN(4) + velE(4) + velD(4) + gSpeed(4) +
            headMot(4) + sAcc(4) + headAcc(4) + pDOP(2) + ...
        """
        if len(payload) < 92:
            return None

        itow = struct.unpack_from('<I', payload, 0)[0]
        fix_type = payload[20]
        flags = payload[21]
        num_sv = payload[23]

        # Position: degE-7 → degrees
        lon_e7 = struct.unpack_from('<i', payload, 24)[0]
        lat_e7 = struct.unpack_from('<i', payload, 28)[0]
        height_mm = struct.unpack_from('<i', payload, 32)[0]   # ellipsoidal

        # Accuracy: mm → m
        h_acc_mm = struct.unpack_from('<I', payload, 40)[0]
        v_acc_mm = struct.unpack_from('<I', payload, 44)[0]

        # Velocity: mm/s → m/s
        vel_n = struct.unpack_from('<i', payload, 48)[0] / 1000.0
        vel_e = struct.unpack_from('<i', payload, 52)[0] / 1000.0
        vel_d = struct.unpack_from('<i', payload, 56)[0] / 1000.0

        # pDOP: 0.01
        pdop = struct.unpack_from('<H', payload, 76)[0] / 100.0

        # Carrier solution from flags bits 6-7
        carr_soln = (flags >> 6) & 0x03

        return {
            "type": "NAV_PVT",
            "tow_ms": itow,
            "lat_deg": lat_e7 / 1e7,
            "lon_deg": lon_e7 / 1e7,
            "alt_m": height_mm / 1000.0,
            "h_acc_m": h_acc_mm / 1000.0,
            "v_acc_m": v_acc_mm / 1000.0,
            "vel_n": vel_n,
            "vel_e": vel_e,
            "vel_d": vel_d,
            "fix_type": fix_type,
            "carrier_solution": carr_soln,
            "n_sats": num_sv,
            "pdop": pdop,
        }

    def _parse_nav_hpposllh(self, payload: bytes) -> Optional[dict]:
        """Parse UBX NAV-HPPOSLLH (36 bytes) — high-precision position.

        Adds sub-mm precision to NAV-PVT position via hp residuals.
        """
        if len(payload) < 36:
            return None

        version = payload[0]
        itow = struct.unpack_from('<I', payload, 4)[0]

        lon_e7 = struct.unpack_from('<i', payload, 8)[0]
        lat_e7 = struct.unpack_from('<i', payload, 12)[0]
        height_mm = struct.unpack_from('<i', payload, 16)[0]
        hmsl_mm = struct.unpack_from('<i', payload, 20)[0]

        # High-precision residuals (0.1 mm = 1e-4 m)
        lon_hp = struct.unpack_from('<b', payload, 24)[0]   # 1e-9 degrees
        lat_hp = struct.unpack_from('<b', payload, 25)[0]
        height_hp = struct.unpack_from('<b', payload, 26)[0]  # 0.1 mm
        hmsl_hp = struct.unpack_from('<b', payload, 27)[0]

        h_acc_mm = struct.unpack_from('<I', payload, 28)[0]  # 0.1 mm
        v_acc_mm = struct.unpack_from('<I', payload, 32)[0]

        # Combine standard + high-precision
        lat_deg = lat_e7 / 1e7 + lat_hp * 1e-9
        lon_deg = lon_e7 / 1e7 + lon_hp * 1e-9
        alt_m = height_mm / 1000.0 + height_hp * 0.0001

        return {
            "type": "NAV_HPPOSLLH",
            "tow_ms": itow,
            "lat_deg": lat_deg,
            "lon_deg": lon_deg,
            "alt_m": alt_m,
            "h_acc_m": h_acc_mm * 0.0001,   # 0.1mm → m
            "v_acc_m": v_acc_mm * 0.0001,
        }


# ── UBX Configuration Helpers ────────────────────────────────

def build_ubx_cfg_msg(cls: int, msg_id: int, rate: int) -> bytes:
    """Build UBX-CFG-MSG to set message output rate on current port."""
    payload = struct.pack('<BBB', cls, msg_id, rate)
    return _wrap_ubx_frame(UBX_CLASS_CFG, UBX_ID_CFG_MSG, payload)


def _wrap_ubx_frame(cls: int, msg_id: int, payload: bytes) -> bytes:
    """Wrap payload in a complete UBX frame with sync + checksum."""
    length = len(payload)
    header = struct.pack('<BBBBH', UBX_SYNC1, UBX_SYNC2,
                         cls, msg_id, length)
    frame = header + payload

    # Fletcher-8 checksum over class + id + length + payload
    ck_a, ck_b = 0, 0
    for b in frame[2:]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF

    return frame + bytes([ck_a, ck_b])


# ── RTK Collector ────────────────────────────────────────────

class RTKCollector:
    """Threaded RTK ground truth collector for u-blox F9P.

    Connects via UART or USB, parses UBX NAV-PVT + NAV-HPPOSLLH,
    and provides timestamped NED ground truth fixes.
    """

    def __init__(self, config: dict):
        self._config = config
        rtk_cfg = config.get("rtk", {})

        self._connection = rtk_cfg.get("connection", "auto")
        self._baud = rtk_cfg.get("uart_baud", 460800)
        self._output_rate = rtk_cfg.get("output_rate_hz", 5)
        self._min_fix = rtk_cfg.get("min_fix_type", 5)
        self._max_hacc = rtk_cfg.get("max_hacc_m", 0.05)
        self._max_vacc = rtk_cfg.get("max_vacc_m", 0.10)
        self._auto_devices = rtk_cfg.get("auto_detect_devices", [
            "/dev/ttyAMA1", "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0",
        ])

        self._serial = None
        self._parser = UBXNavParser()
        self._origin = None        # WGS-84 origin for NED conversion
        self._last_hp = None       # last HPPOSLLH for position refinement

        # Thread-safe fix buffer
        self._fixes: Deque[RTKFix] = deque(maxlen=50000)
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self.stats = RTKStats()
        self._last_fix: Optional[RTKFix] = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @property
    def last_fix(self) -> Optional[RTKFix]:
        return self._last_fix

    @property
    def fix_count(self) -> int:
        return len(self._fixes)

    def get_fixes(self) -> List[RTKFix]:
        """Get all collected fixes (thread-safe copy)."""
        with self._lock:
            return list(self._fixes)

    def get_recent_fixes(self, n: int = 100) -> List[RTKFix]:
        """Get the N most recent fixes."""
        with self._lock:
            fixes = list(self._fixes)
            return fixes[-n:] if len(fixes) > n else fixes

    # ── Connection ────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to u-blox F9P. Supports auto-detect, UART, and USB."""
        try:
            import serial
        except ImportError:
            log.error("pyserial not installed. Run: pip install pyserial")
            return False

        device = self._resolve_device()
        if device is None:
            log.error("Could not find u-blox F9P device")
            return False

        try:
            self._serial = serial.Serial(
                device, self._baud,
                timeout=0.1,
                write_timeout=1.0,
            )
            log.info(f"F9P connected: {device} @ {self._baud} baud")
        except Exception as e:
            log.error(f"F9P connection failed: {device}: {e}")
            return False

        # Configure message rates
        self._configure_messages()
        return True

    def _resolve_device(self) -> Optional[str]:
        """Resolve the serial device path."""
        import os

        if self._connection != "auto":
            return self._connection

        # Auto-detect: try each device in order
        for dev in self._auto_devices:
            if os.path.exists(dev):
                log.info(f"F9P auto-detected: {dev}")
                return dev

        return None

    def _configure_messages(self):
        """Send UBX-CFG-MSG to enable NAV-PVT and NAV-HPPOSLLH."""
        if not self._serial:
            return

        rate = max(1, 10 // self._output_rate)  # convert Hz to divider

        # Enable NAV-PVT
        cmd_pvt = build_ubx_cfg_msg(UBX_CLASS_NAV, UBX_ID_NAV_PVT, rate)
        self._serial.write(cmd_pvt)

        # Enable NAV-HPPOSLLH
        cmd_hp = build_ubx_cfg_msg(UBX_CLASS_NAV, UBX_ID_NAV_HPPOSLLH, rate)
        self._serial.write(cmd_hp)

        log.info(f"F9P configured: NAV-PVT + NAV-HPPOSLLH at ~{self._output_rate} Hz")

    # ── Threaded Operation ────────────────────────────────────

    def start(self):
        """Start background serial reader thread."""
        if not self.is_connected:
            log.error("Cannot start RTK collector — not connected")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="rtk_reader",
            daemon=True,
        )
        self._thread.start()
        log.info("RTK collector thread started")

    def stop(self):
        """Stop background thread and close serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
        log.info(f"RTK collector stopped. "
                 f"Total fixes: {self.stats.total_fixes}, "
                 f"RTK_FIXED: {self.stats.rtk_fixed_count}")

    def _reader_loop(self):
        """Background thread: read serial → parse UBX → buffer fixes."""
        while self._running:
            try:
                data = self._serial.read(512)
                if not data:
                    continue

                messages = self._parser.feed(data)
                for msg in messages:
                    self._handle_message(msg)

            except Exception as e:
                if self._running:
                    log.error(f"RTK reader error: {e}")
                    time.sleep(0.1)

    def write_to_serial(self, data: bytes):
        """Write data to F9P serial (used by NTRIP client for RTCM)."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(data)
            except Exception as e:
                log.debug(f"Serial write error: {e}")

    # ── Message Handling ──────────────────────────────────────

    def _handle_message(self, msg: dict):
        """Process parsed UBX message into RTKFix."""
        if msg["type"] == "NAV_HPPOSLLH":
            # Store high-precision position for refinement
            self._last_hp = msg
            return

        if msg["type"] != "NAV_PVT":
            return

        t_now = time.monotonic()

        # Use HPPOSLLH if available and matched by iTOW
        lat = msg["lat_deg"]
        lon = msg["lon_deg"]
        alt = msg["alt_m"]
        h_acc = msg["h_acc_m"]
        v_acc = msg["v_acc_m"]

        if (self._last_hp is not None and
                self._last_hp["tow_ms"] == msg["tow_ms"]):
            lat = self._last_hp["lat_deg"]
            lon = self._last_hp["lon_deg"]
            alt = self._last_hp["alt_m"]
            h_acc = min(h_acc, self._last_hp["h_acc_m"])
            v_acc = min(v_acc, self._last_hp["v_acc_m"])

        # Determine effective fix quality
        carrier = msg["carrier_solution"]
        effective_fix = msg["fix_type"]
        if carrier == CarrierSolution.RTK_FIXED:
            effective_fix = 5   # our convention
        elif carrier == CarrierSolution.RTK_FLOAT:
            effective_fix = 4

        # Set NED origin on first RTK_FIXED
        if self._origin is None and effective_fix >= self._min_fix:
            self._origin = {
                "lat": lat, "lon": lon, "alt": alt,
            }
            log.info(f"RTK NED origin set: lat={lat:.8f} "
                     f"lon={lon:.8f} alt={alt:.2f}m")

        # Convert to NED
        if self._origin is not None:
            pos_ned = self._wgs84_to_ned(lat, lon, alt)
        else:
            pos_ned = np.zeros(3)

        vel_ned = np.array([msg["vel_n"], msg["vel_e"], msg["vel_d"]])

        fix = RTKFix(
            timestamp_s=t_now,
            tow_ms=msg["tow_ms"],
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            pos_ned=pos_ned,
            vel_ned=vel_ned,
            fix_type=effective_fix,
            carrier_solution=carrier,
            h_acc_m=h_acc,
            v_acc_m=v_acc,
            n_sats=msg["n_sats"],
            pdop=msg["pdop"],
        )

        # Quality gate
        accepted = (effective_fix >= self._min_fix and
                    h_acc <= self._max_hacc and
                    v_acc <= self._max_vacc)

        self.stats.update(fix, accepted)
        self._last_fix = fix

        if accepted:
            with self._lock:
                self._fixes.append(fix)

    def _wgs84_to_ned(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """WGS-84 → local NED (same model as ESKF.update_gps)."""
        d_lat = math.radians(lat - self._origin["lat"])
        d_lon = math.radians(lon - self._origin["lon"])
        lat_ref_rad = math.radians(self._origin["lat"])
        north = d_lat * R_EARTH
        east = d_lon * R_EARTH * math.cos(lat_ref_rad)
        down = -(alt - self._origin["alt"])
        return np.array([north, east, down])

    def set_origin(self, lat: float, lon: float, alt: float):
        """Manually set NED origin (e.g., to match ESKF GPS origin)."""
        self._origin = {"lat": lat, "lon": lon, "alt": alt}
        log.info(f"RTK origin manually set: lat={lat:.8f} "
                 f"lon={lon:.8f} alt={alt:.2f}m")

    def summary(self) -> str:
        """Return a human-readable summary of collection stats."""
        s = self.stats
        lines = [
            "RTK Ground Truth Collection Summary",
            f"  Total fixes   : {s.total_fixes}",
            f"  RTK FIXED     : {s.rtk_fixed_count} "
            f"({100*s.rtk_fixed_count/max(s.total_fixes,1):.1f}%)",
            f"  RTK FLOAT     : {s.rtk_float_count}",
            f"  3D Fix        : {s.fix_3d_count}",
            f"  No fix        : {s.no_fix_count}",
            f"  Rejected (acc): {s.rejected_accuracy}",
            f"  H-accuracy    : min={s.min_hacc_m:.4f}m  "
            f"mean={s.mean_hacc_m:.4f}m  max={s.max_hacc_m:.4f}m",
            f"  Satellites    : {s.min_sats}–{s.max_sats}",
        ]
        return "\n".join(lines)
