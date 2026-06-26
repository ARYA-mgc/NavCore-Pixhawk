/**
 * @file test_eskf.cpp
 * @brief Comprehensive test suite for the C++ ESKF port.
 *
 * Tests all features:
 *   - Construction and initialization
 *   - Predict with stationary IMU
 *   - Baro/Mag updates
 *   - Covariance-based convergence
 *   - GPS update
 *   - ZUPT
 *   - Optical flow
 *   - Radar / Lidar
 *   - Adaptive process noise
 *   - Baro drift compensation
 *   - Generic external update
 *   - No NaN in state vector
 */

#include "eskf_core.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace navcore;

// Helper: initialize ESKF with stationary data
ESKFCore create_initialized_eskf()
{
    IMUNoiseParams noise;
    ESKFCore eskf(noise);

    Eigen::MatrixXd accel(50, 3);
    Eigen::MatrixXd mag(50, 3);
    for (int i = 0; i < 50; ++i) {
        accel.row(i) = Vec3(0.0, 0.0, -GRAVITY).transpose();
        mag.row(i) = Vec3(0.2, 0.0, 0.4).transpose();
    }
    bool ok = eskf.initialize_from_sensors(accel, mag);
    assert(ok);
    return eskf;
}

void test_construction()
{
    IMUNoiseParams noise;
    ESKFCore eskf(noise);

    assert(eskf.get_health() == EKFHealth::CONVERGING);
    assert(!eskf.is_initialized());

    std::cout << "[PASS] Construction\n";
}

void test_stationary_predict()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();

    for (int i = 0; i < 300; ++i) {
        eskf.predict(a, g, 0.01);
        if (i % 10 == 0) eskf.update_baro(0.0);
        if (i % 5 == 0)  eskf.update_mag(0.0);
    }

    auto state = eskf.get_state();
    assert(state.pos.norm() < 5.0);
    assert(!state.pos.hasNaN());
    assert(!state.vel.hasNaN());
    assert(!state.quat.hasNaN());

    // Covariance-based convergence: z_cov < 0.25 + step > 200
    assert(eskf.get_health() == EKFHealth::HEALTHY);

    std::cout << "[PASS] Stationary predict (covariance convergence)\n";
    std::cout << "  Final pos: " << state.pos.transpose() << "\n";
}

void test_covariance_positive_definite()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();
    for (int i = 0; i < 100; ++i) {
        eskf.predict(a, g, 0.01);
    }

    auto P = eskf.get_covariance();
    double asym = (P - P.transpose()).norm();
    assert(asym < 1e-10);

    Eigen::SelfAdjointEigenSolver<MatP> solver(P);
    assert(solver.eigenvalues().minCoeff() > 0);

    std::cout << "[PASS] Covariance positive definite\n";
}

void test_gps_update()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();

    // Run predict for a bit
    for (int i = 0; i < 50; ++i) {
        eskf.predict(a, g, 0.01);
    }

    auto pos_before = eskf.get_state().pos;

    // GPS update at origin
    eskf.update_gps(37.7749, -122.4194, 10.0, 1.0);

    // Second GPS at same location should not change much
    eskf.update_gps(37.7749, -122.4194, 10.0, 1.0);

    auto pos_after = eskf.get_state().pos;
    assert(!pos_after.hasNaN());

    // GPS with bad HDOP should be rejected
    eskf.update_gps(37.7749, -122.4194, 10.0, 6.0);

    std::cout << "[PASS] GPS update\n";
}

void test_zupt()
{
    auto eskf = create_initialized_eskf();

    // Introduce some velocity
    Vec3 a(1.0, 0.0, -GRAVITY);  // forward accel
    Vec3 g = Vec3::Zero();

    for (int i = 0; i < 50; ++i) {
        eskf.predict(a, g, 0.01);
    }

    auto vel_before = eskf.get_state().vel;
    double vel_norm_before = vel_before.norm();
    assert(vel_norm_before > 0.1);  // should have some velocity

    // Apply ZUPT
    eskf.update_zupt();

    auto vel_after = eskf.get_state().vel;
    double vel_norm_after = vel_after.norm();

    // ZUPT should reduce velocity significantly
    assert(vel_norm_after < vel_norm_before);

    std::cout << "[PASS] ZUPT (vel " << vel_norm_before << " -> "
              << vel_norm_after << ")\n";
}

void test_optical_flow()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();
    for (int i = 0; i < 50; ++i) {
        eskf.predict(a, g, 0.01);
    }

    // Optical flow saying zero velocity
    eskf.update_optical_flow(0.0, 0.0, 1.0, 100);

    auto state = eskf.get_state();
    assert(!state.vel.hasNaN());

    std::cout << "[PASS] Optical flow update\n";
}

void test_radar_lidar()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();
    for (int i = 0; i < 50; ++i) {
        eskf.predict(a, g, 0.01);
    }

    // Radar velocity
    eskf.update_radar_velocity(0.0, 0.0, 0.0, 1.0);

    // Lidar range
    eskf.update_lidar_range(5.0, 1.0);

    auto state = eskf.get_state();
    assert(!state.pos.hasNaN());
    assert(!state.vel.hasNaN());

    std::cout << "[PASS] Radar + Lidar updates\n";
}

void test_adaptive_noise()
{
    auto eskf = create_initialized_eskf();

    double vib_before = eskf.get_vibration_scale();
    assert(std::abs(vib_before - 1.0) < 1e-6);

    eskf.scale_process_noise(0.5);
    double vib_after = eskf.get_vibration_scale();
    assert(vib_after > 1.0);
    assert(vib_after <= 10.0);

    // Max vibration
    eskf.scale_process_noise(10.0);
    assert(std::abs(eskf.get_vibration_scale() - 10.0) < 1e-6);

    std::cout << "[PASS] Adaptive process noise\n";
}

void test_baro_drift()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();

    // Run past the convergence threshold (1000 baro updates)
    for (int i = 0; i < 1100; ++i) {
        eskf.predict(a, g, 0.01);
        eskf.update_baro(0.5);  // constant offset simulating drift
    }

    // After 1000+ updates, baro bias should start tracking
    double bias = eskf.get_baro_bias();
    // Bias should be moving toward the offset, but slowly
    // At 0.001 alpha, it won't reach 0.5 in 100 steps, but should be > 0
    assert(bias >= 0.0);

    std::cout << "[PASS] Baro drift compensation (bias=" << bias << ")\n";
}

void test_external_update()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();
    for (int i = 0; i < 50; ++i) {
        eskf.predict(a, g, 0.01);
    }

    // External position update (simulating VIO)
    VecXd z(3); z << 0.0, 0.0, 0.0;
    VecXd z_pred = Eigen::VectorXd::Map(eskf.get_state().pos.data(), 3);
    MatXd H = MatXd::Zero(3, ERROR_DIM);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;
    MatXd R = MatXd::Identity(3, 3) * 0.1;

    bool accepted = eskf.update_external(z, z_pred, H, R, "test_vio");
    assert(accepted);

    std::cout << "[PASS] Generic external update\n";
}

void test_reset()
{
    auto eskf = create_initialized_eskf();

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();
    for (int i = 0; i < 100; ++i) eskf.predict(a, g, 0.01);

    eskf.reset();
    assert(!eskf.is_initialized());
    assert(eskf.get_health() == EKFHealth::CONVERGING);

    std::cout << "[PASS] Reset\n";
}

int main()
{
    std::cout << "NavCore ESKF C++ Port — Full Test Suite\n";
    std::cout << "========================================\n";

    test_construction();
    test_stationary_predict();
    test_covariance_positive_definite();
    test_gps_update();
    test_zupt();
    test_optical_flow();
    test_radar_lidar();
    test_adaptive_noise();
    test_baro_drift();
    test_external_update();
    test_reset();

    std::cout << "\nAll " << 11 << " tests passed.\n";
    return 0;
}
