#!/usr/bin/env python3
"""NTRIP Client — relay RTCM3 corrections to u-blox F9P.

Connects to an NTRIP caster (e.g., RTK2go.com), receives RTCM3
correction data, and writes it to the F9P serial port for
centimeter-level RTK positioning.

Features:
    - NTRIP v1 protocol with Basic auth
    - Automatic reconnection with exponential backoff (1s → 30s)
    - Periodic GGA sentence forwarding for VRS mountpoints
    - Thread-safe, daemon thread operation
"""

import time
import math
import socket
import base64
import logging
import threading
from typing import Optional, Callable

log = logging.getLogger("ntrip_client")


class NTRIPClient:
    """NTRIP v1 client that relays RTCM3 corrections to F9P.

    Args:
        config: dict from rtk_config.yaml 'ntrip' section
        serial_write_fn: callable(bytes) to write RTCM data to F9P serial
    """

    # Reconnect backoff parameters
    BACKOFF_INITIAL_S = 1.0
    BACKOFF_MULTIPLIER = 2.0
    BACKOFF_MAX_S = 30.0

    def __init__(self, config: dict,
                 serial_write_fn: Optional[Callable] = None):
        self._enabled = config.get("enabled", False)
        self._caster = config.get("caster", "rtk2go.com")
        self._port = config.get("port", 2101)
        self._mountpoint = config.get("mountpoint", "")
        self._username = config.get("username", "")
        self._password = config.get("password", "")
        self._gga_interval = config.get("gga_interval_s", 10.0)
        self._reconnect_max = config.get("reconnect_max_s", self.BACKOFF_MAX_S)

        self._serial_write = serial_write_fn
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # GGA source callback (set by RTKCollector to provide current position)
        self._gga_source: Optional[Callable] = None

        # Statistics
        self._bytes_received = 0
        self._reconnect_count = 0
        self._last_data_time = 0.0
        self._connected = False

    @property
    def is_active(self) -> bool:
        return self._enabled and self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "bytes_received": self._bytes_received,
            "reconnects": self._reconnect_count,
            "last_data_age_s": (time.monotonic() - self._last_data_time
                                if self._last_data_time > 0 else -1),
        }

    def set_gga_source(self, fn: Callable):
        """Set callback that returns current NMEA GGA string."""
        self._gga_source = fn

    # ── Connection ────────────────────────────────────────────

    def start(self):
        """Start NTRIP client in background thread."""
        if not self._enabled:
            log.info("NTRIP client disabled in config")
            return

        if not self._mountpoint or self._mountpoint == "YOUR_MOUNT":
            log.warning("NTRIP mountpoint not configured — skipping. "
                        "Edit config/rtk_config.yaml to set mountpoint.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._connection_loop,
            name="ntrip_client",
            daemon=True,
        )
        self._thread.start()
        log.info(f"NTRIP client started → {self._caster}:{self._port}"
                 f"/{self._mountpoint}")

    def stop(self):
        """Stop NTRIP client and close socket."""
        self._running = False
        self._close_socket()
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info(f"NTRIP client stopped. "
                 f"Total RTCM bytes: {self._bytes_received}, "
                 f"Reconnects: {self._reconnect_count}")

    # ── Main Loop with Auto-Reconnect ─────────────────────────

    def _connection_loop(self):
        """Main loop: connect → receive → reconnect on failure."""
        backoff = self.BACKOFF_INITIAL_S

        while self._running:
            try:
                if self._connect():
                    backoff = self.BACKOFF_INITIAL_S  # reset on success
                    self._receive_loop()
            except Exception as e:
                log.warning(f"NTRIP error: {e}")

            self._connected = False
            self._close_socket()

            if not self._running:
                break

            # Exponential backoff
            log.info(f"NTRIP reconnecting in {backoff:.0f}s...")
            time.sleep(backoff)
            backoff = min(backoff * self.BACKOFF_MULTIPLIER,
                          self._reconnect_max)
            self._reconnect_count += 1

    def _connect(self) -> bool:
        """Establish NTRIP v1 connection to caster."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.connect((self._caster, self._port))
        except (socket.error, OSError) as e:
            log.error(f"NTRIP socket connect failed: {e}")
            return False

        # Build NTRIP v1 request
        request = (
            f"GET /{self._mountpoint} HTTP/1.0\r\n"
            f"Host: {self._caster}\r\n"
            f"Ntrip-Version: Ntrip/1.0\r\n"
            f"User-Agent: NavCore-Pixhawk/1.0\r\n"
        )

        # Basic auth if credentials provided
        if self._username:
            credentials = f"{self._username}:{self._password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            request += f"Authorization: Basic {encoded}\r\n"

        request += "\r\n"

        try:
            self._socket.sendall(request.encode())
        except socket.error as e:
            log.error(f"NTRIP request send failed: {e}")
            return False

        # Read response
        try:
            response = self._socket.recv(4096).decode("ascii", errors="replace")
        except socket.error as e:
            log.error(f"NTRIP response read failed: {e}")
            return False

        if "ICY 200 OK" in response or "HTTP/1.1 200" in response:
            self._connected = True
            self._last_data_time = time.monotonic()
            log.info(f"NTRIP connected to {self._caster}/{self._mountpoint}")
            return True
        else:
            first_line = response.split("\n")[0].strip() if response else "empty"
            log.error(f"NTRIP rejected: {first_line}")
            return False

    def _receive_loop(self):
        """Receive RTCM3 data and relay to F9P serial."""
        self._socket.settimeout(30.0)  # longer timeout for data stream
        last_gga_t = 0.0

        while self._running and self._connected:
            # Receive RTCM data
            try:
                data = self._socket.recv(4096)
            except socket.timeout:
                log.warning("NTRIP data timeout (30s)")
                return  # will trigger reconnect
            except socket.error as e:
                log.warning(f"NTRIP recv error: {e}")
                return

            if not data:
                log.warning("NTRIP connection closed by caster")
                return

            self._bytes_received += len(data)
            self._last_data_time = time.monotonic()

            # Relay RTCM to F9P
            if self._serial_write:
                self._serial_write(data)

            # Periodic GGA forwarding (required by VRS mountpoints)
            now = time.monotonic()
            if now - last_gga_t >= self._gga_interval:
                self._send_gga()
                last_gga_t = now

    def _send_gga(self):
        """Send NMEA GGA sentence to caster."""
        if not self._socket or not self._gga_source:
            return

        try:
            gga = self._gga_source()
            if gga:
                self._socket.sendall((gga + "\r\n").encode())
        except Exception as e:
            log.debug(f"GGA send error: {e}")

    def _close_socket(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None


# ── GGA Generation Helper ────────────────────────────────────

def generate_gga(lat_deg: float, lon_deg: float, alt_m: float,
                 fix_quality: int = 4, n_sats: int = 12,
                 hdop: float = 1.0) -> str:
    """Generate NMEA GGA sentence from position.

    Args:
        lat_deg: latitude in degrees (positive = N)
        lon_deg: longitude in degrees (positive = E)
        alt_m: altitude above MSL in meters
        fix_quality: 1=GPS, 2=DGPS, 4=RTK, 5=Float
        n_sats: number of satellites
        hdop: horizontal DOP

    Returns:
        Complete NMEA GGA sentence with checksum
    """
    import datetime
    utc = datetime.datetime.now(datetime.timezone.utc)
    time_str = utc.strftime("%H%M%S.00")

    # Latitude: DDMM.MMMMM
    lat_abs = abs(lat_deg)
    lat_d = int(lat_abs)
    lat_m = (lat_abs - lat_d) * 60.0
    lat_str = f"{lat_d:02d}{lat_m:09.6f}"
    lat_ns = "N" if lat_deg >= 0 else "S"

    # Longitude: DDDMM.MMMMM
    lon_abs = abs(lon_deg)
    lon_d = int(lon_abs)
    lon_m = (lon_abs - lon_d) * 60.0
    lon_str = f"{lon_d:03d}{lon_m:09.6f}"
    lon_ew = "E" if lon_deg >= 0 else "W"

    body = (f"GPGGA,{time_str},{lat_str},{lat_ns},"
            f"{lon_str},{lon_ew},{fix_quality},{n_sats:02d},"
            f"{hdop:.1f},{alt_m:.2f},M,0.00,M,,")

    # NMEA checksum: XOR of all chars between $ and *
    checksum = 0
    for c in body:
        checksum ^= ord(c)

    return f"${body}*{checksum:02X}"
