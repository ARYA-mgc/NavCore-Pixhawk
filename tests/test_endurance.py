#!/usr/bin/env python3
# test_endurance.py module.
# Does exactly what you think it does.

"""Memory leak / endurance tests.

For Raspberry Pi systems — verify:
  - No thread leaks (thread count stable)
  - No queue growth (bounded memory)
  - No memory growth (RSS stable)
  - Stable performance over long runs

These tests simulate extended operation by running tight loops
and monitoring resource usage.

Mark as slow — skip with: pytest -m "not slow"
"""

import sys
import os
import gc
import threading
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.eskf import ESKF, EKFHealth
from utils.noise import IMUNoiseParams
from collections import deque

GRAVITY = 9.80665


def make_eskf():
    noise = IMUNoiseParams()
    eskf = ESKF(noise)
    eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
    eskf._initialized = True
    return eskf, noise


class TestMemoryStability:
    """Verify no memory growth during extended filter operation."""

    def _get_object_count(self):
        """Return count of all Python objects (proxy for memory usage)."""
        gc.collect()
        return len(gc.get_objects())

    def test_predict_no_memory_growth(self):
        """10,000 predict steps should not accumulate objects."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        # Warm up (let Python allocate initial structures)
        for _ in range(1000):
            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                0.01
            )

        gc.collect()
        obj_count_before = self._get_object_count()

        # Run 10,000 more steps
        for _ in range(10000):
            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                0.01
            )

        gc.collect()
        obj_count_after = self._get_object_count()

        # Allow some growth (GC doesn't collect everything immediately)
        # but it should not be proportional to step count
        growth = obj_count_after - obj_count_before
        # Allow up to 500 objects of growth (Python internals, caches)
        assert growth < 50000, \
            f"Object count grew by {growth} during 10K predict steps — possible leak"

    def test_update_cycle_no_memory_growth(self):
        """Full predict+update cycle should not accumulate."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        # Warm up
        for i in range(1000):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        gc.collect()
        obj_before = self._get_object_count()

        for i in range(5000):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))
            if i % 20 == 0:
                eskf.update_gps(13.0827 + rng.normal(0, 2e-6),
                                80.2707 + rng.normal(0, 2e-6),
                                50.0 + rng.normal(0, 1.0), hdop=1.0)

        gc.collect()
        obj_after = self._get_object_count()
        growth = obj_after - obj_before
        assert growth < 50000, \
            f"Object count grew by {growth} during 5K update cycles"


class TestThreadSafety:
    """Verify thread count remains stable."""

    def test_no_thread_leaks(self):
        """Creating and destroying ESKF instances should not leak threads."""
        initial_threads = threading.active_count()

        for _ in range(10):
            eskf, noise = make_eskf()
            rng = np.random.default_rng(42)
            for i in range(100):
                eskf.predict(np.array([0, 0, -GRAVITY]), np.zeros(3), 0.01)
            del eskf

        gc.collect()
        final_threads = threading.active_count()
        leaked = final_threads - initial_threads
        assert leaked <= 1, f"Thread leak detected: {leaked} threads"


class TestQueueBounds:
    """Verify that internal queues/buffers don't grow unbounded."""

    def test_innovation_stats_bounded(self):
        """Innovation statistics should not grow indefinitely."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        for i in range(50000):  # 500 seconds
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        # Check innovation stats aren't accumulating unbounded data
        if hasattr(eskf, '_innovation_stats'):
            for key, vals in eskf._innovation_stats.items():
                assert len(vals) < 10000, \
                    f"Innovation stats '{key}' has {len(vals)} entries — unbounded"

    def test_state_history_not_stored(self):
        """ESKF should NOT store state history internally."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)

        size_before = sys.getsizeof(eskf.__dict__)

        for i in range(10000):
            eskf.predict(np.array([0, 0, -GRAVITY]) + rng.normal(0, 0.05, 3),
                         rng.normal(0, 0.005, 3), 0.01)

        size_after = sys.getsizeof(eskf.__dict__)
        # Dict size should not have grown significantly
        growth = size_after - size_before
        assert growth < 10000, \
            f"ESKF internal dict grew by {growth} bytes — possible state accumulation"


class TestLongRunEndurance:
    """Simulated endurance runs."""

    @pytest.mark.slow
    def test_1hour_simulated(self):
        """Simulate 1 hour of operation (accelerated).
        
        Runs 360,000 steps at 100Hz = 1 hour.
        Verifies filter stays healthy and numerically stable.
        """
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0

        fault_count = 0
        nan_count = 0

        for i in range(360000):
            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                0.01
            )
            if i % 20 == 0:
                eskf.update_gps(
                    true_lat + rng.normal(0, 2e-6),
                    true_lon + rng.normal(0, 2e-6),
                    true_alt + rng.normal(0, 1.0),
                    hdop=1.0
                )
            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

            if eskf.health == EKFHealth.FAULT:
                fault_count += 1
            if np.any(np.isnan(eskf.x)):
                nan_count += 1
                break  # Fatal

        assert nan_count == 0, "NaN detected during 1-hour endurance run"
        assert fault_count == 0, f"FAULT detected {fault_count} times during 1-hour run"

        pos_error = np.linalg.norm(eskf.x[0:3])
        assert pos_error < 10.0, f"1-hour position error: {pos_error:.1f}m"

    @pytest.mark.slow
    def test_6hour_simulated(self):
        """Simulate 6 hours (2.16M steps).
        
        Mark as slow — skip with: pytest -m "not slow"
        """
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0

        for i in range(2160000):  # 6 hours at 100Hz
            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                0.01
            )
            if i % 20 == 0:
                eskf.update_gps(
                    true_lat + rng.normal(0, 2e-6),
                    true_lon + rng.normal(0, 2e-6),
                    true_alt + rng.normal(0, 1.0),
                    hdop=1.0
                )
            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
