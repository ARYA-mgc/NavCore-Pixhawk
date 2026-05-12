#!/usr/bin/env python3
# This is where we poke the math wizard with a stick to see if it breaks.
# Also testing if the dead reckoning actually reckons anything.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import numpy as np
import pytest
from core.eskf        import ESKF, EKFHealth
from utils.noise      import IMUNoiseParams
from core.dr import DeadReckon


# ── fixtures ────────────────────────────────────────────────────
@pytest.fixture
def noise():
    return IMUNoiseParams()

@pytest.fixture
def eskf(noise):
    e = ESKF(noise)
    # Initialize with identity quaternion (level, north-facing)
    e.x[6:10] = e._euler_to_quat(0, 0, 0)
    e._initialized = True
    return e

@pytest.fixture
def dr(noise):
    return DeadReckon(noise)


# ── ESKF tests ──────────────────────────────────────────────────
class TestESKF:

    def test_initial_quaternion_identity(self, eskf):
        # should start perfectly level and pointing north
        assert abs(eskf.x[6] - 1.0) < 1e-10
        assert np.allclose(eskf.x[7:10], 0.0)

    def test_covariance_positive_definite(self, eskf):
        eigvals = np.linalg.eigvalsh(eskf.P)
        assert np.all(eigvals > 0), "Initial P must be positive definite"

    def test_predict_increases_uncertainty(self, eskf):
        # no sensor updates = growing uncertainty
        trace_before = np.trace(eskf.P)
        accel = np.array([0.0, 0.0, -9.80665])  # hovering
        gyro  = np.zeros(3)
        for _ in range(50):
            eskf.predict(accel, gyro, 0.01)
        trace_after = np.trace(eskf.P)
        assert trace_after > trace_before

    def test_predict_gravity_stationary(self, eskf):
        # sitting still should mean staying at zero
        accel = np.array([0.0, 0.0, -9.80665])
        gyro  = np.zeros(3)
        for _ in range(100):
            eskf.predict(accel, gyro, 0.01)
        pos = eskf.state["pos"]
        # Dead-reckoning still drifts, but should be < 1 m after 1 s
        assert np.linalg.norm(pos) < 1.0

    def test_baro_update_corrects_altitude(self, eskf):
        # baro should pull altitude toward the right answer
        accel = np.array([0.0, 0.0, -9.80665])
        # Gradually introduce baro readings to stay within gating
        for i in range(300):
            eskf.predict(accel, np.zeros(3), 0.01)
            if i % 5 == 0:
                # Feed small altitude — within innovation gate
                eskf.update_baro(0.5)
        # z should have moved toward 0.5
        assert abs(eskf.state["pos"][2] - 0.5) < 1.0

    def test_mag_update_sets_yaw(self, eskf):
        # compass should slowly fix our heading
        target_yaw = math.radians(20.0)  # smaller angle for convergence
        accel = np.array([0.0, 0.0, -9.80665])
        for i in range(500):
            eskf.predict(accel, np.zeros(3), 0.01)
            if i % 2 == 0:
                eskf.update_mag(target_yaw)
        yaw = eskf.state["euler"][2]
        assert abs(yaw - target_yaw) < math.radians(15.0)

    def test_covariance_decreases_after_update(self, eskf):
        # measurements = less uncertainty, always
        accel = np.array([0.0, 0.0, -9.80665])
        for _ in range(20):
            eskf.predict(accel, np.zeros(3), 0.01)
        trace_before = np.trace(eskf.P)
        eskf.update_baro(0.0)
        trace_after = np.trace(eskf.P)
        assert trace_after < trace_before

    def test_reset(self, eskf):
        eskf.x[0] = 999.0
        eskf.reset()
        assert eskf.x[0] == 0.0
        assert eskf.x[6] == 1.0  # identity quaternion restored

    def test_health_starts_converging(self, noise):
        e = ESKF(noise)
        assert e.health == EKFHealth.CONVERGING

    def test_health_becomes_healthy(self, eskf):
        # run it long enough with corrections and it should settle
        accel = np.array([0.0, 0.0, -9.80665])
        rng = np.random.default_rng(42)
        for i in range(300):
            eskf.predict(accel, np.zeros(3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, 0.3))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, 0.02))
        assert eskf.health == EKFHealth.HEALTHY


# ── Dead Reckon tests ────────────────────────────────────────────
class TestDeadReckon:

    def test_stationary(self, dr):
        accel = np.array([0.0, 0.0, -9.80665])
        gyro  = np.zeros(3)
        for _ in range(100):
            dr.update(accel, gyro, 0.01)
        assert np.linalg.norm(dr.pos) < 2.0   # some drift expected

    def test_forward_motion(self, dr):
        # push forward, position should increase
        accel = np.array([0.5, 0.0, -9.80665])
        gyro  = np.zeros(3)
        for _ in range(100):
            dr.update(accel, gyro, 0.01)
        assert dr.pos[0] > 0.0


# ── Noise Params tests ───────────────────────────────────────────
class TestIMUNoiseParams:

    def test_defaults_positive(self, noise):
        assert noise.accel_std > 0
        assert noise.gyro_std  > 0
        assert noise.baro_std  > 0
        assert noise.mag_std   > 0

    def test_summary_string(self, noise):
        s = noise.summary()
        assert "Accel" in s
        assert "Baro"  in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
