#!/usr/bin/env python3
"""RTK state transitions and RTCM stream tests.

Tests:
  - No Fix → Float → Fixed transitions
  - Fixed → Float degradation
  - RTCM stream interruption handling
  - Convergence time estimation
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.eskf import ESKF, EKFHealth
from utils.noise import IMUNoiseParams

GRAVITY = 9.80665


def make_eskf():
    noise = IMUNoiseParams()
    eskf = ESKF(noise)
    eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
    eskf._initialized = True
    return eskf, noise


class TestRTKStateTransitions:
    """Simulate RTK fix quality transitions and verify ESKF adapts."""

    def _simulate_gps_quality(self, eskf, noise, rng, n_steps, hdop,
                              pos_std_m, dt=0.01):
        """Run GPS updates with specified quality level.
        
        Args:
            hdop: HDOP value (1.0 = RTK fixed, 2.0 = float, 5.0 = standalone)
            pos_std_m: position noise standard deviation
        """
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        for i in range(n_steps):
            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                dt
            )
            # GPS at 5 Hz
            if i % 20 == 0:
                lat_noise = rng.normal(0, pos_std_m / 111320.0)
                lon_noise = rng.normal(0, pos_std_m / (111320.0 * math.cos(math.radians(true_lat))))
                eskf.update_gps(
                    true_lat + lat_noise,
                    true_lon + lon_noise,
                    true_alt + rng.normal(0, pos_std_m * 2),
                    hdop=hdop
                )
            # Baro
            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            # Mag
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

    def test_no_fix_to_float_to_fixed(self):
        """Simulate: GPS warm-up → brief gap → Float RTK → Fixed RTK."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        # Warm-up with GPS (10s) — filter converges
        for i in range(1000):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 20 == 0:
                eskf.update_gps(true_lat + rng.normal(0, 2e-6),
                                true_lon + rng.normal(0, 2e-6),
                                true_alt + rng.normal(0, 1.0), hdop=1.5)
            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        # Brief GPS gap (5s) — covariance grows
        for i in range(500):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))

        pos_cov_no_fix = np.trace(eskf.P[0:2, 0:2])

        # Phase 2: Float RTK (20s, HDOP 2.0, ~1m accuracy)
        self._simulate_gps_quality(eskf, noise, rng, 2000, hdop=2.0, pos_std_m=1.0)
        pos_cov_float = np.trace(eskf.P[0:2, 0:2])

        # Phase 3: Fixed RTK (20s, HDOP 0.8, ~0.02m accuracy)
        self._simulate_gps_quality(eskf, noise, rng, 2000, hdop=0.8, pos_std_m=0.02)
        pos_cov_fixed = np.trace(eskf.P[0:2, 0:2])

        # Fixed should be better than float
        assert pos_cov_fixed < pos_cov_float, \
            f"Fixed RTK should reduce uncertainty (float={pos_cov_float:.4f}, fixed={pos_cov_fixed:.4f})"

        # Final position error should be bounded
        pos_error = np.linalg.norm(eskf.x[0:3])
        assert pos_error < 5.0, f"RTK Fixed position error: {pos_error:.3f}m"

    def test_fixed_to_float_degradation(self):
        """RTK Fixed → Float: filter should increase position uncertainty."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(77)

        # Start with RTK Fixed (30s)
        self._simulate_gps_quality(eskf, noise, rng, 3000, hdop=0.8, pos_std_m=0.02)
        pos_cov_fixed = np.trace(eskf.P[0:2, 0:2])

        # Degrade to Float (20s)
        self._simulate_gps_quality(eskf, noise, rng, 2000, hdop=2.0, pos_std_m=1.0)
        pos_cov_float = np.trace(eskf.P[0:2, 0:2])

        # Covariance should have grown
        assert pos_cov_float > pos_cov_fixed, \
            "Position uncertainty should increase when RTK degrades to Float"

    def test_rtcm_stream_interruption(self):
        """Simulate RTCM stream loss: RTK Fixed → slowly degrades → Float.
        
        When RTCM corrections stop, the F9P gradually loses its RTK fix.
        The ESKF should handle the increasing HDOP gracefully.
        """
        eskf, noise = make_eskf()
        rng = np.random.default_rng(55)

        # Start with RTK Fixed (20s)
        self._simulate_gps_quality(eskf, noise, rng, 2000, hdop=0.8, pos_std_m=0.02)

        # RTCM lost: HDOP gradually increases over 30s
        # This simulates the F9P losing its RTK fix
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        hdop_history = []
        for i in range(3000):
            t_degrade = i / 3000.0  # 0 → 1
            # HDOP ramps from 0.8 to 4.0 as RTK fix degrades
            current_hdop = 0.8 + 3.2 * t_degrade
            # Position noise also increases
            current_pos_std = 0.02 + 2.0 * t_degrade

            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                0.01
            )
            if i % 20 == 0:
                lat_noise = rng.normal(0, current_pos_std / 111320.0)
                lon_noise = rng.normal(0, current_pos_std / 111320.0)
                eskf.update_gps(
                    true_lat + lat_noise,
                    true_lon + lon_noise,
                    true_alt + rng.normal(0, current_pos_std * 2),
                    hdop=current_hdop
                )
                hdop_history.append(current_hdop)

            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))

        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT
        # Position error should have grown but not exploded
        pos_error = np.linalg.norm(eskf.x[0:3])
        assert pos_error < 20.0, \
            f"RTCM loss: position error={pos_error:.1f}m (should be <20m)"

    def test_rtk_reacquisition_after_rtcm_loss(self):
        """After RTCM stream resumes, RTK should re-converge."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(99)

        # Phase 1: RTK Fixed (15s)
        self._simulate_gps_quality(eskf, noise, rng, 1500, hdop=0.8, pos_std_m=0.02)

        # Phase 2: RTCM lost — standalone GPS (15s)
        self._simulate_gps_quality(eskf, noise, rng, 1500, hdop=3.0, pos_std_m=2.5)
        pos_error_standalone = np.linalg.norm(eskf.x[0:3])

        # Phase 3: RTCM back — RTK Fixed again (15s)
        self._simulate_gps_quality(eskf, noise, rng, 1500, hdop=0.8, pos_std_m=0.02)
        pos_error_reacquired = np.linalg.norm(eskf.x[0:3])

        # Should have recovered
        assert pos_error_reacquired < pos_error_standalone, \
            f"RTK re-acquisition should improve accuracy "  \
            f"(was {pos_error_standalone:.2f}m, now {pos_error_reacquired:.2f}m)"
        assert pos_error_reacquired < 1.0, \
            f"RTK re-acquisition error: {pos_error_reacquired:.3f}m"


class TestRTKConvergence:
    """Test RTK convergence timing characteristics."""

    def test_cold_start_convergence_time(self):
        """Measure how many GPS epochs to reach <1m position error."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        gps_epochs = 0
        converged = False

        for i in range(10000):  # 100 seconds
            eskf.predict(
                accel + rng.normal(0, noise.accel_std, 3),
                gyro + rng.normal(0, noise.gyro_std, 3),
                0.01
            )
            if i % 20 == 0:  # 5 Hz GPS
                gps_epochs += 1
                eskf.update_gps(
                    true_lat + rng.normal(0, 2e-6),
                    true_lon + rng.normal(0, 2e-6),
                    true_alt + rng.normal(0, 1.0),
                    hdop=1.0
                )
                pos_error = np.linalg.norm(eskf.x[0:3])
                if pos_error < 1.0 and not converged:
                    converged = True
                    break

            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        assert converged, f"Filter did not converge to <1m in {gps_epochs} GPS epochs"
        # Should converge within 50 epochs (10 seconds at 5 Hz)
        assert gps_epochs < 50, \
            f"Convergence took {gps_epochs} GPS epochs (expected <50)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
