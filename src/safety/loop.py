# Loop timing monitor.
# Making sure the while loop doesn't take a nap.

import time
import logging
import os
import collections
import numpy as np

log = logging.getLogger("loop_monitor")

class LoopMonitor:
    def __init__(self, target_dt: float):
        self.target_dt = target_dt
        self.target_ms = target_dt * 1000.0
        self._loop_times_ms = collections.deque(maxlen=10000)
        self._overrun_count = 0
        self._max_loop_ms = 0.0
        self._start_time = time.monotonic()
        
        # Apply OS scheduling hints if possible
        self._apply_rt_hints()

    def _apply_rt_hints(self):
        # Apply real-time scheduling hints.
        try:
            if os.name == 'posix':
                # Nice value (requires root for negative values)
                os.nice(-10)
                log.info("Process priority increased (nice -10)")
            elif os.name == 'nt':
                import psutil
                p = psutil.Process(os.getpid())
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                log.info("Windows process priority set to HIGH")
        except Exception as e:
            log.debug(f"Could not set RT priority: {e}")

    def record_loop(self, loop_ms: float):
        # Record loop execution time.
        self._loop_times_ms.append(loop_ms)
        if loop_ms > self._max_loop_ms:
            self._max_loop_ms = loop_ms
            
        if loop_ms > self.target_ms * 1.5:
            self._overrun_count += 1
            if self._overrun_count % 50 == 1:
                log.warning(f"Loop overrun: {loop_ms:.1f}ms (target {self.target_ms:.0f}ms)")

    def get_stats(self):
        # Get loop timing statistics.
        if not self._loop_times_ms:
            return {"avg": 0.0, "max": 0.0, "p99": 0.0, "overruns": 0}
            
        arr = np.array(self._loop_times_ms)
        return {
            "avg": float(np.mean(arr)),
            "max": self._max_loop_ms,
            "p99": float(np.percentile(arr, 99)),
            "overruns": self._overrun_count
        }

    def print_histogram(self):
        # Print loop timing histogram.
        if not self._loop_times_ms:
            return
            
        arr = np.array(self._loop_times_ms)
        log.info("=== Loop Timing Histogram (ms) ===")
        hist, bins = np.histogram(arr, bins=10, range=(0, self.target_ms * 3))
        
        max_count = max(hist)
        for i in range(len(hist)):
            bar = "#" * int(20 * hist[i] / max(1, max_count))
            log.info(f"{bins[i]:4.1f} - {bins[i+1]:4.1f} | {hist[i]:5d} {bar}")
        log.info(f"Overruns: {self._overrun_count} | Max: {self._max_loop_ms:.1f}ms")
        log.info("==================================")
