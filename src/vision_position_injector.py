#!/usr/bin/env python3
"""
vision_position_injector.py
============================
Feeds the INS-estimated position back into ArduPilot's EKF3
via VISION_POSITION_ESTIMATE MAVLink messages.

This enables GPS-denied flight using our INS as the primary
position source, replacing GPS in ArduPilot's EKF3.

ArduPilot parameter setup required (Mission Planner):
  EK3_SRC1_POSXY = 6   (ExternalNav)
  EK3_SRC1_POSZ  = 6
  EK3_SRC1_YAW   = 6
  VISO_TYPE      = 1   (enable vision odometry)
  SERIAL2_PROTOCOL = 2  (MAVLink — your TELEM2 port)
"""

import time
import threading
import logging
import numpy as np
from eskf_core import ESKFCore

log = logging.getLogger("vision_injector")


class VisionPositionInjector:
    """
    Background thread that periodically sends
    VISION_POSITION_ESTIMATE to Pixhawk based on INS state.
    """

    INJECT_HZ = 30    # 30 Hz is sufficient for ArduPilot EKF3

    def __init__(self, bridge, ekf: ESKFCore):
        self.bridge  = bridge
        self.ekf     = ekf
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vision_injector"
        )
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()
        log.info(f"Vision position injector started @ {self.INJECT_HZ} Hz")

    def stop(self):
        self._running = False
        log.info("Vision position injector stopped")

    def _loop(self):
        interval = 1.0 / self.INJECT_HZ
        while self._running:
            t0 = time.monotonic()
            try:
                pos = self.ekf.state["pos"]
                # Convert NED → ENU (ArduPilot vision uses NED internally,
                # but VISION_POSITION_ESTIMATE is in body frame convention)
                # Here we send NED directly (ArduPilot handles it).
                self.bridge.send_vision_position(pos, np.zeros(4))
            except Exception as e:
                log.warning(f"Vision inject error: {e}")
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)
