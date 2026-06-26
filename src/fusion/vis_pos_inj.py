# Vision Position Injection.
# Tricking the autopilot into thinking we have GPS.

import time
import threading
import logging
import numpy as np
from core.eskf import ESKF

log = logging.getLogger("vis_inj")

class VisPosInj:
    # Injection rate
    HZ = 30

    def __init__(self, bridge, ekf: ESKF):
        self.bridge = bridge
        self.ekf = ekf
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()
        log.info("VisPosInj: Vision position injection started")

    def stop(self):
        self._running = False
        log.info("VisPosInj: Vision position injection stopped")

    def _loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.monotonic()
            try:
                # Get ESKF position and send to flight controller
                pos = self.ekf.state["pos"]
                self.bridge.send_vision_position(pos, np.zeros(4))
            except Exception as e:
                log.warning(f"Failed to send vision position: {e}")
            
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))
