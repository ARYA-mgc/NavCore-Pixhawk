#!/usr/bin/env python3
"""Flight Data Recorder — synchronized multi-stream CSV recording.

Records all sensor streams + RTK ground truth + ESKF state during flight.
Output format is directly compatible with rtk_validate.py and
generate_rtk_sim.py for seamless post-flight analysis.

Each recording session gets a timestamped directory:
    flight_data/YYYYMMDD_HHMMSS/
        ├── imu_log.csv          (100 Hz)
        ├── gps_log.csv          (5 Hz)
        ├── baro_log.csv         (25 Hz)
        ├── mag_log.csv          (10 Hz)
        ├── rtk_ground_truth.csv (5 Hz, RTK_FIXED only)
        ├── eskf_state.csv       (50 Hz)
        └── metadata.json
"""

import os
import csv
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

log = logging.getLogger("flight_recorder")


class FlightRecorder:
    """Synchronized multi-stream flight data recorder.

    CSV column names match generate_rtk_sim.py output exactly,
    so rtk_validate.py works unchanged on recorded data.
    """

    # CSV schemas (matching generate_rtk_sim.py output format)
    SCHEMAS = {
        "imu_log": ["time_s", "ax", "ay", "az", "gx", "gy", "gz"],
        "gps_log": ["time_s", "north_m", "east_m", "down_m", "hdop"],
        "baro_log": ["time_s", "alt_m"],
        "mag_log": ["time_s", "mx", "my", "mz"],
        "rtk_ground_truth": [
            "time_s", "x_m", "y_m", "z_m",
            "vx_mps", "vy_mps", "vz_mps",
            "qw", "qx", "qy", "qz",
            "fix_type", "h_acc_m", "v_acc_m", "n_sats",
        ],
        "eskf_state": [
            "time_s",
            "px_m", "py_m", "pz_m",
            "vx_ms", "vy_ms", "vz_ms",
            "qw", "qx", "qy", "qz",
            "ba_x", "ba_y", "ba_z",
            "bg_x", "bg_y", "bg_z",
            "health", "P_trace",
            "baro_bias",
        ],
    }

    def __init__(self, output_dir: str = "flight_data",
                 flush_interval_s: float = 1.0):
        self._base_dir = output_dir
        self._flush_interval = flush_interval_s
        self._session_dir: Optional[str] = None
        self._writers: Dict[str, csv.writer] = {}
        self._files: Dict[str, Any] = {}
        self._buffers: Dict[str, list] = {}
        self._counts: Dict[str, int] = {}
        self._recording = False
        self._start_time: Optional[float] = None
        self._last_flush = 0.0
        self._t0: Optional[float] = None  # reference time for relative timestamps

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_dir(self) -> Optional[str]:
        return self._session_dir

    @property
    def sample_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    # ── Session Management ────────────────────────────────────

    def start_session(self, metadata: Optional[dict] = None):
        """Start a new recording session."""
        if self._recording:
            log.warning("Recording already in progress")
            return

        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(self._base_dir, timestamp)
        os.makedirs(self._session_dir, exist_ok=True)

        # Open CSV files
        for name, schema in self.SCHEMAS.items():
            filepath = os.path.join(self._session_dir, f"{name}.csv")
            # Use line buffering for crash resilience
            f = open(filepath, "w", newline="", buffering=1)
            writer = csv.writer(f)
            writer.writerow(schema)
            self._files[name] = f
            self._writers[name] = writer
            self._buffers[name] = []
            self._counts[name] = 0

        self._recording = True
        self._start_time = time.monotonic()
        self._t0 = None
        self._last_flush = time.monotonic()

        # Write initial metadata
        if metadata:
            self._write_metadata(metadata)

        log.info(f"Flight recording started → {self._session_dir}")

    def stop_session(self, final_metadata: Optional[dict] = None):
        """Stop recording, flush all buffers, close files."""
        if not self._recording:
            return

        self._recording = False

        # Flush remaining buffers
        self._flush_all()

        # Close all files
        for name, f in self._files.items():
            try:
                f.flush()
                f.close()
            except Exception:
                pass

        self._files.clear()
        self._writers.clear()
        self._buffers.clear()

        # Write final metadata
        duration = time.monotonic() - self._start_time if self._start_time else 0
        meta = {
            "session_dir": self._session_dir,
            "duration_s": duration,
            "sample_counts": dict(self._counts),
            "stop_time": datetime.now().isoformat(),
        }
        if final_metadata:
            meta.update(final_metadata)
        self._write_metadata(meta)

        log.info(f"Flight recording stopped. Duration: {duration:.1f}s")
        log.info(f"  Samples: {self._counts}")
        log.info(f"  Data saved to: {self._session_dir}")

    # ── Record Methods ────────────────────────────────────────

    def record_imu(self, t: float, accel: np.ndarray, gyro: np.ndarray):
        """Record fused IMU measurement (100 Hz)."""
        if not self._recording:
            return
        t_rel = self._relative_time(t)
        self._buffer_row("imu_log", [
            f"{t_rel:.4f}",
            f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
            f"{gyro[0]:.6f}", f"{gyro[1]:.6f}", f"{gyro[2]:.6f}",
        ])

    def record_gps(self, t: float, north: float, east: float,
                   down: float, hdop: float):
        """Record GPS measurement in NED (5 Hz)."""
        if not self._recording:
            return
        t_rel = self._relative_time(t)
        self._buffer_row("gps_log", [
            f"{t_rel:.4f}",
            f"{north:.4f}", f"{east:.4f}", f"{down:.4f}",
            f"{hdop:.2f}",
        ])

    def record_baro(self, t: float, alt_m: float):
        """Record barometric altitude (25 Hz)."""
        if not self._recording:
            return
        t_rel = self._relative_time(t)
        self._buffer_row("baro_log", [
            f"{t_rel:.4f}",
            f"{alt_m:.4f}",
        ])

    def record_mag(self, t: float, mx: float, my: float, mz: float):
        """Record magnetometer measurement in body frame (10 Hz)."""
        if not self._recording:
            return
        t_rel = self._relative_time(t)
        self._buffer_row("mag_log", [
            f"{t_rel:.4f}",
            f"{mx:.6f}", f"{my:.6f}", f"{mz:.6f}",
        ])

    def record_rtk(self, t: float, fix):
        """Record RTK ground truth fix (5 Hz, only RTK_FIXED).

        Args:
            t: system monotonic time
            fix: RTKFix dataclass from rtk_collector
        """
        if not self._recording:
            return
        t_rel = self._relative_time(t)

        # Default quaternion (identity) — RTK doesn't provide attitude
        # In practice, use ESKF attitude for ground truth orientation
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        self._buffer_row("rtk_ground_truth", [
            f"{t_rel:.4f}",
            f"{fix.pos_ned[0]:.6f}", f"{fix.pos_ned[1]:.6f}",
            f"{fix.pos_ned[2]:.6f}",
            f"{fix.vel_ned[0]:.4f}", f"{fix.vel_ned[1]:.4f}",
            f"{fix.vel_ned[2]:.4f}",
            f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}",
            f"{fix.fix_type}",
            f"{fix.h_acc_m:.4f}", f"{fix.v_acc_m:.4f}",
            f"{fix.n_sats}",
        ])

    def record_eskf_state(self, t: float, state: dict,
                          P: np.ndarray, health_name: str,
                          baro_bias: float = 0.0):
        """Record ESKF state output (50 Hz)."""
        if not self._recording:
            return
        t_rel = self._relative_time(t)
        pos = state["pos"]
        vel = state["vel"]
        quat = state["quat"]
        ba = state["accel_bias"]
        bg = state["gyro_bias"]

        self._buffer_row("eskf_state", [
            f"{t_rel:.4f}",
            f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
            f"{vel[0]:.4f}", f"{vel[1]:.4f}", f"{vel[2]:.4f}",
            f"{quat[0]:.6f}", f"{quat[1]:.6f}",
            f"{quat[2]:.6f}", f"{quat[3]:.6f}",
            f"{ba[0]:.6f}", f"{ba[1]:.6f}", f"{ba[2]:.6f}",
            f"{bg[0]:.6f}", f"{bg[1]:.6f}", f"{bg[2]:.6f}",
            health_name,
            f"{np.trace(P):.4f}",
            f"{baro_bias:.6f}",
        ])

    # ── Buffering & Flush ─────────────────────────────────────

    def _buffer_row(self, stream: str, row: list):
        """Buffer a row and flush periodically."""
        self._buffers[stream].append(row)
        self._counts[stream] += 1

        # Periodic flush
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._flush_all()
            self._last_flush = now

    def _flush_all(self):
        """Flush all buffered rows to disk."""
        for name, buf in self._buffers.items():
            if buf and name in self._writers:
                self._writers[name].writerows(buf)
                if name in self._files:
                    self._files[name].flush()
                buf.clear()

    def _relative_time(self, t: float) -> float:
        """Convert monotonic time to session-relative time."""
        if self._t0 is None:
            self._t0 = t
        return t - self._t0

    def _write_metadata(self, meta: dict):
        """Write/update metadata.json in session directory."""
        if not self._session_dir:
            return
        path = os.path.join(self._session_dir, "metadata.json")
        # Merge with existing metadata if present
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update(meta)
        try:
            with open(path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Failed to write metadata: {e}")
