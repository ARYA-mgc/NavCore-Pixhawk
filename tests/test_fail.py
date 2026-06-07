#!/usr/bin/env python3
# test_fail.py module.
# Does exactly what you think it does.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import numpy as np
import pytest
from core.eskf import ESKF, EKFHealth
from utils.noise import IMUNoiseParams

rng = np.random.default_rng(99)


@pytest.fixture
def eskf():
    noise = IMUNoiseParams()
    e = ESKF(noise)
    e.x[6:10] = e._euler_to_quat(0, 0, 0)
    e._initialized = True
    return e


def run_predict_steps(eskf, n=100, dt=0.01):
    accel = np.array([0.0, 0.0, -9.80665])
    gyro = np.zeros(3)
    for _ in range(n):
        eskf.predict(accel, gyro, dt)


class TestSensorDropout:

    def test_baro_dropout_10s(self, eskf):
        # can it fly blind for 10 seconds without baro? let's see
        run_predict_steps(eskf, 1000, 0.01)  # 10s of IMU only
        assert eskf.health != EKFHealth.FAULT
        assert not np.any(np.isnan(eskf.x))

    def test_mag_dropout_30s(self, eskf):
        # no compass = growing uncertainty, filter should know
        yaw_var_before = eskf.P[8, 8]
        run_predict_steps(eskf, 3000, 0.01)
        yaw_var_after = eskf.P[8, 8]
        
        # Without any corrections, yaw uncertainty WILL grow significantly.
        # This is correct physical behavior.
        assert yaw_var_after > yaw_var_before * 10.0
        assert not np.any(np.isnan(eskf.x))

    def test_all_sensors_present(self, eskf):
        # everything connected = should report healthy
        accel = np.array([0.0, 0.0, -9.80665])
        gyro = np.zeros(3)
        for i in range(500):
            eskf.predict(accel, gyro, 0.01)
            if i % 10 == 0:
                eskf.update_baro(0.0 + rng.normal(0, 0.3))
            if i % 2 == 0:
                eskf.update_mag(0.0 + rng.normal(0, 0.02))
        assert eskf.health in (EKFHealth.HEALTHY, EKFHealth.CONVERGING)


class TestNoiseSikes:

    def test_accel_noise_spike(self, eskf):
        # blast it with noise, it should tank it
        accel_normal = np.array([0.0, 0.0, -9.80665])
        gyro = np.zeros(3)

        # Normal operation
        for _ in range(200):
            eskf.predict(accel_normal + rng.normal(0, 0.05, 3), gyro, 0.01)

        # 1s of 10x noise
        for _ in range(100):
            noise = rng.normal(0, 0.5, 3)
            eskf.predict(accel_normal + noise, gyro, 0.01)

        # Should still be alive
        assert not np.any(np.isnan(eskf.x))

    def test_baro_spike_rejected(self, eskf):
        # baro goes crazy? filter should ignore it
        run_predict_steps(eskf, 100)
        eskf.update_baro(0.0)  # normal
        eskf.update_baro(0.0)

        pos_before = eskf.state["pos"][2]
        eskf.update_baro(100.0)  # spike: 100m jump
        pos_after = eskf.state["pos"][2]

        # Position should NOT jump to 100m (gating should reject)
        assert abs(pos_after - pos_before) < 10.0


class TestMagDisturbance:

    def test_mag_field_doubled_rejected(self, eskf):
        # someone put a magnet nearby, filter should bail
        run_predict_steps(eskf, 100)

        # Normal mag updates
        for _ in range(10):
            eskf.update_mag(0.0, mag_norm=0.5, t_now=1.0)

        # Disturbance: field norm doubled
        yaw_before = eskf.state["euler"][2]
        eskf.update_mag(1.5, mag_norm=1.0, t_now=2.0)  # should reject
        yaw_after = eskf.state["euler"][2]

        # Yaw should not have changed significantly
        assert abs(yaw_after - yaw_before) < 0.1


class TestBiasExplosion:

    def test_large_bias_clamped(self, eskf):
        # force huge biases, they should get clamped
        eskf.x[10:13] = [5.0, 5.0, 5.0]  # way over limit
        run_predict_steps(eskf, 10)

        # Biases should have been clamped
        assert np.all(np.abs(eskf.x[10:13]) <= eskf.ACCEL_BIAS_LIMIT + 0.01)


class TestCovarianceHealth:

    def test_nan_detection(self, eskf):
        # inject NaN, should immediately go FAULT
        eskf.x[0] = float('nan')
        eskf._check_health()
        assert eskf.health == EKFHealth.FAULT

    def test_symmetry_enforcement(self, eskf):
        # covariance must stay symmetric or bad things happen
        eskf.P[0, 1] = 1.0
        eskf.P[1, 0] = 0.5  # asymmetric
        eskf._step_count = eskf.SYMMETRY_INTERVAL - 1
        run_predict_steps(eskf, 2)  # triggers enforcement at SYMMETRY_INTERVAL
        assert abs(eskf.P[0, 1] - eskf.P[1, 0]) < 1e-10


class TestQuaternionMath:

    def test_identity_quaternion(self, eskf):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = eskf._quat_to_dcm(q)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_euler_quat_roundtrip(self, eskf):
        for roll, pitch, yaw in [(0.1, 0.2, 0.3), (-0.5, 0.5, 1.0), (0, 0, 0)]:
            q = eskf._euler_to_quat(roll, pitch, yaw)
            euler = eskf._quat_to_euler(q)
            assert abs(euler[0] - roll) < 1e-10
            assert abs(euler[1] - pitch) < 1e-10
            assert abs(euler[2] - yaw) < 1e-10

    def test_quaternion_normalization(self, eskf):
        q = np.array([2.0, 1.0, 0.5, 0.3])
        q_norm = q / np.linalg.norm(q)
        R = eskf._quat_to_dcm(q_norm)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_90deg_rotations(self, eskf):
        # 90 deg around Z axis
        q = eskf._euler_to_quat(0, 0, math.pi/2)
        R = eskf._quat_to_dcm(q)
        v_body = np.array([1, 0, 0])
        v_ned = R @ v_body
        assert abs(v_ned[0]) < 1e-10
        assert abs(v_ned[1] - 1.0) < 1e-10

    def test_quat_mult_identity(self, eskf):
        q = eskf._euler_to_quat(0.3, 0.2, 0.1)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = eskf._quat_mult(q, identity)
        assert np.allclose(result, q, atol=1e-10)

    def test_skew_antisymmetric(self, eskf):
        v = np.array([1, 2, 3])
        S = eskf._skew(v)
        assert np.allclose(S, -S.T)


class TestJacobianValidation:

    def test_F_analytical_vs_numerical(self, eskf):
        # math check: does our jacobian match numerical differentiation?
        accel = np.array([0.1, -0.2, -9.8])
        gyro = np.array([0.01, -0.02, 0.03])
        dt = 0.01

        R = eskf._quat_to_dcm(eskf.x[6:10])
        F_analytical = eskf._compute_F(accel, gyro, R, dt)

        # Numerical Jacobian via finite differences
        eps = 1e-7
        F_numerical = np.zeros((20, 20))

        x0 = eskf.x.copy()
        P0 = eskf.P.copy()

        for j in range(20):
            # Perturb error state
            dx_plus = np.zeros(20)
            dx_plus[j] = eps

            # Forward
            eskf.x = x0.copy()
            eskf.P = P0.copy()
            eskf._inject_error(dx_plus)
            x_plus = eskf.x.copy()
            eskf.predict(accel + x0[10:13], gyro + x0[13:16], dt)
            f_plus = eskf.x[:6].copy()

            # Reset and do unperturbed
            eskf.x = x0.copy()
            eskf.P = P0.copy()
            eskf.predict(accel + x0[10:13], gyro + x0[13:16], dt)
            f_0 = eskf.x[:6].copy()

            F_numerical[:6, j] = (f_plus - f_0) / eps

            eskf.x = x0.copy()
            eskf.P = P0.copy()

        # Check position and velocity blocks (most critical)
        max_err = np.max(np.abs(
            F_analytical[:6, :6] - F_numerical[:6, :6]))
        # Relaxed tolerance due to numerical differentiation noise
        assert max_err < 0.1, f"Jacobian error too large: {max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
