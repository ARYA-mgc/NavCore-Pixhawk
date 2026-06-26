#!/usr/bin/env python3
# test_gps_outage.py module.
# Does exactly what you think it does.

"""GPS outage and recovery tests.

Default tests use 5s and 30s outages (complete in <10s each).
@pytest.mark.slow tests use 120s outages.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.eskf import ESKF, EKFHealth, ERROR_DIM
from utils.noise import IMUNoiseParams

GRAVITY = 9.80665


def make_eskf():
    noise = IMUNoiseParams()
    eskf = ESKF(noise)
    eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
    eskf._initialized = True
    return eskf, noise


def warm_up(eskf, noise, rng, n_steps=1000, dt=0.01):
    """10s warm-up with full aiding."""
    true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
    accel = np.array([0.0, 0.0, -GRAVITY])
    for i in range(n_steps):
        eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                     rng.normal(0, noise.gyro_std, 3), dt)
        if i % 20 == 0:
            eskf.update_gps(true_lat + rng.normal(0, 2e-6),
                            true_lon + rng.normal(0, 2e-6),
                            true_alt + rng.normal(0, 1.0), hdop=1.0)
        if i % 4 == 0:
            eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
        if i % 10 == 0:
            eskf.update_mag(rng.normal(0, noise.mag_std))


def run_outage(outage_s, dt=0.01):
    """Run: warm-up → GPS outage → GPS recovery."""
    eskf, noise = make_eskf()
    rng = np.random.default_rng(42)
    true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
    accel = np.array([0.0, 0.0, -GRAVITY])

    warm_up(eskf, noise, rng)
    pos_cov_before = np.trace(eskf.P[0:3, 0:3])

    # Outage: no GPS, keep baro+mag
    n_out = int(outage_s / dt)
    for i in range(n_out):
        eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                     rng.normal(0, noise.gyro_std, 3), dt)
        if i % 4 == 0:
            eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
        if i % 10 == 0:
            eskf.update_mag(rng.normal(0, noise.mag_std))

    pos_error_outage = np.linalg.norm(eskf.x[0:3])
    pos_cov_outage = np.trace(eskf.P[0:3, 0:3])

    # Pre-recovery snapshot
    pos_before_recovery = eskf.x[0:3].copy()
    cov_history = []
    innovations = []
    recovery_time = -1.0

    # Recovery: 5s with GPS
    for i in range(500):
        eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                     rng.normal(0, noise.gyro_std, 3), dt)
        
        if i % 20 == 0:
            # We want to catch the first innovation
            pre_update_innov = len(eskf.innovation_history)
            
            eskf.update_gps(true_lat + rng.normal(0, 2e-6),
                            true_lon + rng.normal(0, 2e-6),
                            true_alt + rng.normal(0, 1.0), hdop=1.0)
            
            if len(eskf.innovation_history) > pre_update_innov:
                _, source, y, S, nis = eskf.innovation_history[-1]
                if source == "gps":
                    innovations.append(np.linalg.norm(y))
                    
        if i % 4 == 0:
            eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            
        current_cov = np.trace(eskf.P[0:3, 0:3])
        cov_history.append(current_cov)
        
        if recovery_time < 0 and current_cov < pos_cov_before * 2.0:
            recovery_time = i * dt

    pos_after_first_updates = eskf.x[0:3].copy()
    position_jump = np.linalg.norm(pos_after_first_updates - pos_before_recovery)

    return {
        "eskf": eskf,
        "pos_error_outage": pos_error_outage,
        "pos_cov_before": pos_cov_before,
        "pos_cov_outage": pos_cov_outage,
        "final_error": np.linalg.norm(eskf.x[0:3]),
        "recovery_time": recovery_time,
        "position_jump": position_jump,
        "cov_history": cov_history,
        "innovations": innovations,
    }


class TestGPSOutage5s:

    def test_error_bounded(self):
        r = run_outage(5.0)
        assert r["pos_error_outage"] < 5.0

    def test_recovery(self):
        r = run_outage(5.0)
        assert r["final_error"] < 3.0

    def test_no_nan(self):
        r = run_outage(5.0)
        assert not np.any(np.isnan(r["eskf"].x))


class TestGPSOutage30s:

    def test_error_bounded(self):
        r = run_outage(30.0)
        assert r["pos_error_outage"] < 500.0

    def test_covariance_increases(self):
        r = run_outage(30.0)
        assert r["pos_cov_outage"] > r["pos_cov_before"] * 1.5

    def test_no_nan(self):
        r = run_outage(30.0)
        assert not np.any(np.isnan(r["eskf"].x))
        assert not np.any(np.isinf(r["eskf"].x))


class TestGPSOutage60s:

    def test_no_nan(self):
        r = run_outage(60.0)
        assert not np.any(np.isnan(r["eskf"].x))
        assert not np.any(np.isinf(r["eskf"].x))

    def test_covariance_grows(self):
        r = run_outage(60.0)
        assert r["pos_cov_outage"] > r["pos_cov_before"] * 5.0


    @pytest.mark.slow
    def test_metrics_logged(self):
        r = run_outage(30.0)
        # Verify new metrics are present
        assert r["recovery_time"] > 0
        assert r["position_jump"] > 0
        assert len(r["innovations"]) > 0
        assert r["cov_history"][-1] < r["pos_cov_outage"]

class TestGPSOutage120s:

    def test_no_nan(self):
        r = run_outage(120.0)
        assert not np.any(np.isnan(r["eskf"].x))
        assert not np.any(np.isinf(r["eskf"].x))

    def test_covariance_grows(self):
        r = run_outage(120.0)
        assert r["pos_cov_outage"] > r["pos_cov_before"] * 10.0


class TestRepeatedOutages:

    def test_3_cycles_stable(self):
        """3 cycles of 10s outage + 5s recovery."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(99)
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
        accel = np.array([0.0, 0.0, -GRAVITY])

        warm_up(eskf, noise, rng)

        for cycle in range(3):
            # 10s outage
            for i in range(1000):
                eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                             rng.normal(0, noise.gyro_std, 3), 0.01)
                if i % 4 == 0:
                    eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            # 5s recovery
            for i in range(500):
                eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                             rng.normal(0, noise.gyro_std, 3), 0.01)
                if i % 20 == 0:
                    eskf.update_gps(true_lat + rng.normal(0, 2e-6),
                                    true_lon + rng.normal(0, 2e-6),
                                    true_alt + rng.normal(0, 1.0), hdop=1.0)
                if i % 4 == 0:
                    eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))

        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
