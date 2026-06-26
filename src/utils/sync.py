# sync.py module.
# Does exactly what you think it does.

import time
import logging

log = logging.getLogger("time_sync")

class TimeSynchronizer:
    def __init__(self, default_dt=0.01):
        self.default_dt = default_dt
        self._last_msg_time_us = 0
        self._last_sys_time_s = 0.0
        self._initialized = False

        # Latency tracking
        self.latency_s = 0.0
        self._moving_avg_latency = 0.0
        self._alpha = 0.1  # smoothing factor

    def compute_dt(self, msg) -> float:
        # figure out exactly how much time passed since last reading
        now_s = time.monotonic()
        
        # Determine current message timestamp (usec)
        msg_time_us = 0
        if hasattr(msg, 'time_usec') and msg.time_usec > 0:
            msg_time_us = msg.time_usec
        elif hasattr(msg, 'time_boot_ms') and msg.time_boot_ms > 0:
            msg_time_us = msg.time_boot_ms * 1000

        # Initialization
        if not self._initialized:
            self._last_msg_time_us = msg_time_us
            self._last_sys_time_s = now_s
            self._initialized = True
            return self.default_dt

        dt = self.default_dt

        if msg_time_us > 0 and self._last_msg_time_us > 0:
            # Hardware timestamp available
            dt_us = msg_time_us - self._last_msg_time_us
            
            # Handle counter wraparound or weird jumps
            if 0 < dt_us < 1_000_000:  # < 1 second jump
                dt = dt_us / 1_000_000.0
            else:
                # Fallback to system time if jump is crazy
                sys_dt = now_s - self._last_sys_time_s
                if 0.001 < sys_dt < 1.0:
                    dt = sys_dt

            # Estimate latency: System dt minus Hardware dt (rough)
            # A more rigorous way is comparing local time to boot time
            sys_dt = now_s - self._last_sys_time_s
            inst_latency = max(0.0, sys_dt - dt)
            self._moving_avg_latency = (self._alpha * inst_latency + 
                                       (1 - self._alpha) * self._moving_avg_latency)
            self.latency_s = self._moving_avg_latency

        else:
            # Fallback to system time
            sys_dt = now_s - self._last_sys_time_s
            if 0.001 < sys_dt < 1.0:
                dt = sys_dt

        # Hard clamp dt to prevent math explosions in EKF
        dt = max(0.001, min(0.5, dt))

        self._last_msg_time_us = msg_time_us
        self._last_sys_time_s = now_s

        return dt

    def reset(self):
        self._initialized = False
        self.latency_s = 0.0
        self._moving_avg_latency = 0.0
