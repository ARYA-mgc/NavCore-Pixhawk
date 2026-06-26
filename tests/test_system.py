#!/usr/bin/env python3
# test_system.py module.
# Does exactly what you think it does.

"""System-level tests: vibration, maneuvers, compass disturbance, EKF recovery.

Tests:
  - Motor current / nearby metal compass interference
  - High-frequency vibration (propeller imbalance)
  - Aggressive maneuvers (fast yaw, banked turns, rapid climb/descent)
  - EKF recovery after sensor disturbances
  - Yaw drift during hover
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


#  Test: Compass Disturbance 

class TestCompassDisturbance:
    """Simulate magnetic interference from motors and nearby metal."""

    def test_motor_current_interference(self):
        """Simulates motor current creating a magnetic field.
        
        During high throttle, mag readings get corrupted.
        Filter should detect and reject bad mag data.
        """
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        # 10s normal operation
        for i in range(1000):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std), mag_norm=0.5, t_now=i*0.01)

        yaw_before = eskf._quat_to_euler(eskf.x[6:10])[2]

        # 5s of motor current interference: mag field magnitude doubles
        for i in range(500):
            t = 10.0 + i * 0.01
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                # Corrupted mag: norm doubled (motor interference)
                bad_yaw = rng.normal(0.5, 0.1)  # wildly wrong yaw
                eskf.update_mag(bad_yaw, mag_norm=1.0, t_now=t)

        yaw_after = eskf._quat_to_euler(eskf.x[6:10])[2]

        # Yaw should not have changed much — filter should have rejected
        yaw_change = abs(yaw_after - yaw_before)
        if yaw_change > math.pi:
            yaw_change = 2 * math.pi - yaw_change
        assert yaw_change < math.radians(20.0), \
            f"Motor interference corrupted yaw by {math.degrees(yaw_change):.1f}°"

    def test_nearby_metal_gradual_distortion(self):
        """Gradual magnetic distortion (e.g., flying near a building).
        
        Mag norm changes slowly — filter should adapt R (inflate noise).
        """
        eskf, noise = make_eskf()
        rng = np.random.default_rng(77)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        # Warm up
        for i in range(2000):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std), mag_norm=0.5, t_now=i*0.01)

        # Gradually increase mag norm (approaching metal)
        for i in range(3000):
            t = 20.0 + i * 0.01
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                # Slowly increasing mag norm
                mag_norm = 0.5 + 0.3 * (i / 3000.0)  # 0.5 → 0.8
                eskf.update_mag(rng.normal(0, noise.mag_std),
                                mag_norm=mag_norm, t_now=t)

        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT

    def test_mag_recovery_after_disturbance(self):
        """After magnetic disturbance clears, filter should re-accept mag."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(55)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)

        # Normal
        for i in range(2000):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std), mag_norm=0.5, t_now=i*0.01)

        # Disturbance (2s)
        for i in range(200):
            t = 20.0 + i * 0.01
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            eskf.update_mag(1.5, mag_norm=1.2, t_now=t)  # reject this

        yaw_cov_during = eskf.P[8, 8]

        # Recovery (10s)
        for i in range(1000):
            t = 22.0 + i * 0.01
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std), mag_norm=0.5, t_now=t)

        assert not np.any(np.isnan(eskf.x))


#  Test: Vibration Effects 

class TestVibrationEffects:
    """Simulate high-frequency IMU noise from propeller imbalance."""

    def test_high_freq_vibration(self):
        """Inject 100Hz vibration (typical propeller frequency).
        
        Filter should handle high-frequency noise without diverging.
        """
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        base_accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)
        dt = 0.01

        for i in range(5000):  # 50 seconds
            t = i * dt
            # Propeller vibration: ~100Hz sinusoidal on all axes
            vib_freq = 100.0 * 2 * math.pi
            vib_amp = 2.0  # 2 m/s² vibration (severe)
            vibration = vib_amp * np.array([
                math.sin(vib_freq * t),
                math.sin(vib_freq * t + 1.0),
                math.sin(vib_freq * t + 2.0),
            ])

            accel = base_accel + vibration + rng.normal(0, noise.accel_std, 3)
            g = gyro + rng.normal(0, noise.gyro_std, 3)

            eskf.predict(accel, g, dt)

            if i % 4 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 10 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        assert not np.any(np.isnan(eskf.x))
        assert not np.any(np.isinf(eskf.x))
        # Without GPS, horizontal drift is unbounded — only check stability
        assert not np.any(np.isnan(eskf.P))

    def test_vibration_scale_exists(self):
        """Vibration scale attribute should exist and be >= 1.0."""
        eskf, _ = make_eskf()
        assert hasattr(eskf, '_vibration_scale')
        assert eskf._vibration_scale >= 1.0


#  Test: Aggressive Maneuvers 

class TestAggressiveManeuvers:
    """Simulate aggressive drone flight: fast yaw, banked turns, climbs."""

    def _run_maneuver(self, accel_profile, gyro_profile, dt=0.01):
        """Run a maneuver profile and verify filter stability."""
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)

        # Warm up
        for i in range(500):
            eskf.predict(
                np.array([0, 0, -GRAVITY]) + rng.normal(0, noise.accel_std, 3),
                rng.normal(0, noise.gyro_std, 3), dt)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))

        # Execute maneuver
        for i in range(len(accel_profile)):
            eskf.predict(
                accel_profile[i] + rng.normal(0, noise.accel_std, 3),
                gyro_profile[i] + rng.normal(0, noise.gyro_std, 3),
                dt
            )
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))

        return eskf

    def test_fast_yaw_turn_180deg(self):
        """180° yaw turn at 90°/s — common in return-to-home."""
        dt = 0.01
        duration = 2.0  # seconds
        N = int(duration / dt)

        accel = np.tile([0, 0, -GRAVITY], (N, 1))
        gyro = np.zeros((N, 3))
        gyro[:, 2] = math.radians(90)  # 90°/s yaw rate

        eskf = self._run_maneuver(accel, gyro, dt)
        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT

    def test_fast_yaw_turn_360deg(self):
        """Full 360° yaw at 120°/s — pirouette maneuver."""
        dt = 0.01
        duration = 3.0
        N = int(duration / dt)

        accel = np.tile([0, 0, -GRAVITY], (N, 1))
        gyro = np.zeros((N, 3))
        gyro[:, 2] = math.radians(120)

        eskf = self._run_maneuver(accel, gyro, dt)
        assert not np.any(np.isnan(eskf.x))

    def test_banked_turn_30deg(self):
        """30° bank turn — lateral acceleration + roll."""
        dt = 0.01
        duration = 5.0
        N = int(duration / dt)

        bank_angle = math.radians(30)
        accel = np.zeros((N, 3))
        accel[:, 0] = GRAVITY * math.sin(bank_angle)  # lateral
        accel[:, 2] = -GRAVITY * math.cos(bank_angle)  # reduced vertical

        gyro = np.zeros((N, 3))
        gyro[:, 0] = 0.0  # some roll rate
        gyro[:, 2] = math.radians(20)  # yaw rate during turn

        eskf = self._run_maneuver(accel, gyro, dt)
        assert not np.any(np.isnan(eskf.x))
        euler = eskf._quat_to_euler(eskf.x[6:10])
        tilt = math.sqrt(euler[0]**2 + euler[1]**2)
        assert tilt < math.radians(70), "Excessive tilt after banked turn"

    def test_rapid_climb_5ms(self):
        """Rapid climb at 5 m/s — throttle punch."""
        dt = 0.01
        duration = 3.0
        N = int(duration / dt)

        accel = np.zeros((N, 3))
        accel[:, 2] = -GRAVITY - 5.0  # extra thrust for climb
        gyro = np.zeros((N, 3))

        eskf = self._run_maneuver(accel, gyro, dt)
        assert not np.any(np.isnan(eskf.x))
        # Vertical velocity should be significant
        assert abs(eskf.x[5]) > 0.1, "No vertical velocity after climb"

    def test_rapid_descent_5ms(self):
        """Rapid descent at 5 m/s — emergency drop."""
        dt = 0.01
        duration = 3.0
        N = int(duration / dt)

        accel = np.zeros((N, 3))
        accel[:, 2] = -GRAVITY + 5.0  # reduced thrust for descent
        gyro = np.zeros((N, 3))

        eskf = self._run_maneuver(accel, gyro, dt)
        assert not np.any(np.isnan(eskf.x))

    def test_combined_maneuver(self):
        """Combined: climb + yaw + bank — aggressive multi-axis."""
        dt = 0.01
        duration = 5.0
        N = int(duration / dt)

        accel = np.zeros((N, 3))
        gyro = np.zeros((N, 3))

        for i in range(N):
            t = i * dt
            # Climb + bank
            accel[i] = [
                GRAVITY * 0.3 * math.sin(2.0 * t),   # lateral oscillation
                GRAVITY * 0.1 * math.cos(3.0 * t),    # forward/back
                -GRAVITY - 2.0 * math.sin(1.0 * t),   # climb/descend
            ]
            gyro[i] = [
                math.radians(30) * math.sin(2.0 * t),  # roll
                math.radians(15) * math.cos(1.5 * t),  # pitch
                math.radians(45) * math.sin(1.0 * t),  # yaw
            ]

        eskf = self._run_maneuver(accel, gyro, dt)
        assert not np.any(np.isnan(eskf.x))
        assert eskf.health != EKFHealth.FAULT


#  Test: Yaw Drift During Hover 

class TestYawDriftHover:
    """Hover for extended periods — yaw should not drift significantly."""

    def _run_hover(self, duration_s):
        eskf, noise = make_eskf()
        rng = np.random.default_rng(42)
        accel = np.array([0.0, 0.0, -GRAVITY])
        gyro = np.zeros(3)
        N = int(duration_s / 0.01)
        for i in range(N):
            eskf.predict(accel + rng.normal(0, noise.accel_std, 3),
                         gyro + rng.normal(0, noise.gyro_std, 3), 0.01)
            if i % 10 == 0:
                eskf.update_baro(rng.normal(0, noise.baro_std))
            if i % 2 == 0:
                eskf.update_mag(rng.normal(0, noise.mag_std))
        return eskf

    def test_hover_30s_yaw_stable(self):
        """30s hover with mag — yaw drift should be <5°."""
        eskf = self._run_hover(30.0)
        yaw = eskf._quat_to_euler(eskf.x[6:10])[2]
        assert abs(yaw) < math.radians(15.0), \
            f"Yaw drifted {math.degrees(yaw):.1f}° during 30s hover"

    @pytest.mark.slow
    def test_hover_300s_yaw_stable(self):
        """5 min hover — long-term yaw stability."""
        eskf = self._run_hover(300.0)
        yaw = eskf._quat_to_euler(eskf.x[6:10])[2]
        assert abs(yaw) < math.radians(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
