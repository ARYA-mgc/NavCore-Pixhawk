# Feeding fake GPS to ArduPilot so it stops panicking and crying about "NO GPS"
# Look at me, I'm the captain now.

import time
import threading
import logging
import numpy as np
from core.eskf import ESKF

log = logging.getLogger("vis_inj")

class VisPosInj:
    # 30 Hz is enough to keep the autopilot from having an existential crisis
    HZ = 30

    def __init__(self, bridge, ekf: ESKF):
        self.bridge = bridge
        self.ekf = ekf
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()
        log.info("VisPosInj: Fake GPS go brrr @ 30Hz")

    def stop(self):
        self._running = False
        log.info("VisPosInj: Nap time")

    def _loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.monotonic()
            try:
                # Grab position from our math wizard and send it
                pos = self.ekf.state["pos"]
                self.bridge.send_vision_position(pos, np.zeros(4))
            except Exception as e:
                log.warning(f"Oops, dropped the fake GPS signal: {e}")
            
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))
