#!/usr/bin/env python3
# Writing numbers to CSV really, really fast.

import csv
import os
import time
import socket
import logging
import numpy as np

log = logging.getLogger("ins_logger")


class INSLogger:
    # dumps state to CSV — one row per timestep, simple and fast

    HEADER = [
        "time_s",
        "px_m", "py_m", "pz_m",
        "vx_ms", "vy_ms", "vz_ms",
        "roll_deg", "pitch_deg", "yaw_deg",
        "P_trace",
        "pi_temp_c", "flow_vx", "flow_vy",
        "pid_kp", "pid_ki", "pid_kd"
    ]

    def __init__(self, filepath: str = "logs/ins_data.csv",
                 udp_host: str = None, udp_port: int = 14550):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._f   = open(filepath, "w", newline="")
        self._csv = csv.writer(self._f)
        self._csv.writerow(self.HEADER)
        self._filepath = filepath

        # Optional UDP forward to GCS / QGroundControl
        self._udp_sock = None
        if udp_host:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_target = (udp_host, udp_port)
            log.info(f"UDP telemetry → {udp_host}:{udp_port}")

        log.info(f"Logging to {filepath}")

    def write(self, t: float, pos: np.ndarray,
              vel: np.ndarray, att_deg: np.ndarray,
              P: np.ndarray, pi_temp: float = 0.0,
              flow_vel: np.ndarray = np.zeros(2),
              pid_gains: tuple = (0.0, 0.0, 0.0)):
        row = [
            f"{t:.4f}",
            f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
            f"{vel[0]:.4f}", f"{vel[1]:.4f}", f"{vel[2]:.4f}",
            f"{att_deg[0]:.3f}", f"{att_deg[1]:.3f}", f"{att_deg[2]:.3f}",
            f"{np.trace(P):.6f}",
            f"{pi_temp:.1f}", f"{flow_vel[0]:.4f}", f"{flow_vel[1]:.4f}",
            f"{pid_gains[0]:.4f}", f"{pid_gains[1]:.4f}", f"{pid_gains[2]:.4f}"
        ]
        self._csv.writerow(row)

        # UDP telemetry (simple CSV line over UDP)
        if self._udp_sock:
            try:
                msg = ",".join(row).encode()
                self._udp_sock.sendto(msg, self._udp_target)
            except Exception:
                pass

    def close(self):
        self._f.flush()
        self._f.close()
        log.info(f"Log closed: {self._filepath}")
        if self._udp_sock:
            self._udp_sock.close()
