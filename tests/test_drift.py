#!/usr/bin/env python3
"""Long-duration drift tests.

Default tests use 30-60s simulations (fast, ~5s each).
@pytest.mark.slow tests use 10-30 min simulations.

Verifies:
  - Position error remains bounded with aiding
  - Covariance increases monotonically without aiding
  - No NaN / Inf contamination
  - Filter remains numerically stable
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.eskf import ESKF, EKFHealth, NOMINAL_DIM, ERROR_DIM
from utils.noise import IMUNoiseParams

GRAVITY = 9.80665


class RealisticIMU:
    """MEMS IMU with bias random walk, temperature drift, scale factor error."""

    def __init__(self, noise: IMUNoiseParams, rng: np.random.Generator,
                 temp_drift: bool = True):
        self.noise = noise
        self.rng = rng
        self.temp_drift = temp_drift
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_scale = 1.0 + rng.uniform(-0.005, 0.005, 3)
        self.gyro_scale = 1.0 + rng.uniform(-0.005, 0.005, 3)
        self.temp_period = 3600.0
        self.accel_temp_coeff = rng.uniform(-0.002, 0.002, 3)
        self.gyro_temp_coeff = rng.uniform(-0.0001, 0.0001, 3)

    def sample(self, true_accel, true_gyro, dt, t):
        self.accel_bias += self.rng.normal(0, self.noise.accel_bias_std * math.sqrt(dt), 3)
        self.gyro_bias += self.rng.normal(0, self.noise.gyro_bias_std * math.sqrt(dt), 3)
        self.accel_bias = np.clip(self.accel_bias, -0.5, 0.5)
        self.gyro_bias = np.clip(self.gyro_bias, -0.02, 0.02)

        temp_bias_a = np.zeros(3)
        temp_bias_g = np.zeros(3)
        if self.temp_drift:
            td = 10.0 * math.sin(2 * math.pi * t / self.temp_period)
            temp_bias_a = self.accel_temp_coeff * td
            temp_bias_g = self.gyro_temp_coeff * td

        accel = true_accel * self.accel_scale + self.accel_bias + temp_bias_a
        accel += self.rng.normal(0, self.noise.accel_std, 3)
        gyro = true_gyro * self.gyro_scale + self.gyro_bias + temp_bias_g
        gyro += self.rng.normal(0, self.noise.gyro_std, 3)
        return accel, gyro


def make_eskf():
    noise = IMUNoiseParams()
    eskf = ESKF(noise)
    eskf.x[6:10] = eskf._euler_to_quat(0, 0, 0)
    eskf._initialized = True
    return eskf, noise


# ── Pure IMU Drift (no aiding) ────────────────────────────────

class TestPureIMUDrift:

    def _run_pure_imu(self, duration_s, dt=0.01):
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        imu = RealisticIMU(noise, rng)
        N = int(duration_s / dt)
        trace_history = []
        for i in range(N):
            a, g = imu.sample(np.array([0, 0, -GRAVITY]), np.zeros(3), dt, i*dt)
            eskf.predict(a, g, dt)
            if i % 100 == 0:
                trace_history.append(np.trace(eskf.P))
        return eskf, trace_history

    def test_60s_pure_imu_no_nan(self):
        """60s of pure IMU — no NaN."""
        eskf, _ = self._run_pure_imu(60.0)
        assert not np.any(np.isnan(eskf.x))
        assert not np.any(np.isinf(eskf.x))
        assert not np.any(np.isnan(eskf.P))

    def test_60s_covariance_grows(self):
        """Without aiding, covariance should grow."""
        _, traces = self._run_pure_imu(60.0)
        assert traces[-1] > traces[0] * 2.0, "Covariance should grow without aiding"

    @pytest.mark.slow
    def test_10min_pure_imu_no_nan(self):
        eskf, _ = self._run_pure_imu(600.0)
        assert not np.any(np.isnan(eskf.x))

    @pytest.mark.slow
    def test_30min_pure_imu_bounded(self):
        eskf, _ = self._run_pure_imu(1800.0)
        assert not np.any(np.isnan(eskf.x))
        assert np.linalg.norm(eskf.x[0:3]) < 1e8


# ── Aided Drift (baro + mag only) ────────────────────────────

class TestAidedDrift:

    def _run_aided(self, duration_s, dt=0.01):
        eskf, noise = make_eskf()
        rng = np.random.default_rng(77)
        imu = RealisticIMU(noise, rng)
        N = int(duration_s / dt)
        for i in range(N):
            a, g = imu.sample(np.array([0, 0, -GRAVITY]), np.zeros(3), dt, i*dt)
            eskf.predict(a, g, dt)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))
        return eskf

    def test_60s_aided_altitude_bounded(self):
        """60s baro+mag — altitude bounded."""
        eskf = self._run_aided(60.0)
        assert abs(eskf.x[2]) < 3.0

    def test_60s_aided_yaw_bounded(self):
        """60s with mag — yaw stays within 10°."""
        eskf = self._run_aided(60.0)
        yaw = eskf._quat_to_euler(eskf.x[6:10])[2]
        assert abs(yaw) < math.radians(10.0)

    def test_60s_aided_no_nan(self):
        eskf = self._run_aided(60.0)
        assert not np.any(np.isnan(eskf.x))

    @pytest.mark.slow
    def test_30min_aided_numerically_stable(self):
        """30 min baro+mag — numerically stable, altitude bounded."""
        eskf = self._run_aided(1800.0)
        assert not np.any(np.isnan(eskf.x))
        assert abs(eskf.x[2]) < 10.0


# ── Full Aiding (GPS + baro + mag) ────────────────────────────

class TestFullyAidedDrift:

    def _run_full_aided(self, duration_s, dt=0.01, gps_rate_hz=5.0):
        eskf, noise = make_eskf()
        rng = np.random.default_rng(123)
        imu = RealisticIMU(noise, rng, temp_drift=True)
        true_lat, true_lon, true_alt = 13.0827, 80.2707, 50.0
        N = int(duration_s / dt)
        gps_interval = int(1.0 / gps_rate_hz / dt)
        for i in range(N):
            a, g = imu.sample(np.array([0, 0, -GRAVITY]), np.zeros(3), dt, i*dt)
            eskf.predict(a, g, dt)
            if i % gps_interval == 0:
                eskf.update_gps(
                    true_lat + rng.normal(0, 2.5e-6),
                    true_lon + rng.normal(0, 2.5e-6),
                    true_alt + rng.normal(0, 1.0), hdop=1.0)
            if i % 4 == 0:
                eskf.update_baro(-true_alt + rng.normal(0, noise.baro_std))
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))
        return eskf

    def test_60s_full_aided_sub5m(self):
        """60s GPS+baro+mag — position error <5m."""
        eskf = self._run_full_aided(60.0)
        assert np.linalg.norm(eskf.x[0:3]) < 5.0

    def test_60s_full_aided_healthy(self):
        eskf = self._run_full_aided(60.0)
        assert eskf.health in (EKFHealth.HEALTHY, EKFHealth.CONVERGING)

    @pytest.mark.slow
    def test_10min_full_aided_sub15m(self):
        eskf = self._run_full_aided(600.0)
        assert np.linalg.norm(eskf.x[0:3]) < 15.0
        assert eskf.health in (EKFHealth.HEALTHY, EKFHealth.CONVERGING)

    @pytest.mark.slow
    def test_30min_full_aided_stable(self):
        eskf = self._run_full_aided(1800.0)
        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT
        assert np.linalg.norm(eskf.x[0:3]) < 10.0

    @pytest.mark.slow
    def test_1hour_full_aided_endurance(self):
        eskf = self._run_full_aided(3600.0)
        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT
        assert np.linalg.norm(eskf.x[0:3]) < 20.0


# ── Temperature Drift ─────────────────────────────────────────

class TestTemperatureDrift:

    def test_temp_drift_no_nan(self):
        """60s with temperature drift — numerically stable."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(55)
        imu = RealisticIMU(noise, rng, temp_drift=True)
        for i in range(6000):
            a, g = imu.sample(np.array([0, 0, -GRAVITY]), np.zeros(3), 0.01, i*0.01)
            eskf.predict(a, g, 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))
        assert not np.any(np.isnan(eskf.x))

    def test_no_temp_drift_stable(self):
        """60s without temp drift — stable."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(55)
        imu = RealisticIMU(noise, rng, temp_drift=False)
        for i in range(6000):
            a, g = imu.sample(np.array([0, 0, -GRAVITY]), np.zeros(3), 0.01, i*0.01)
            eskf.predict(a, g, 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))
        assert not np.any(np.isnan(eskf.x))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
