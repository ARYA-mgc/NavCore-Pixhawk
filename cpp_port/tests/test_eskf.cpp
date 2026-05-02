/**
 * @file test_eskf.cpp
 * @brief Basic smoke test for the C++ ESKF port.
 *
 * Validates:
 *   - Construction and initialization
 *   - Predict step with stationary IMU
 *   - Baro/Mag updates
 *   - Health state machine convergence
 *   - No NaN in state vector
 */

#include "eskf_core.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace navcore;

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
    IMUNoiseParams noise;
    ESKFCore eskf(noise);

    // Manual initialization
    Eigen::MatrixXd accel(50, 3);
    Eigen::MatrixXd mag(50, 3);
    for (int i = 0; i < 50; ++i) {
        accel.row(i) = Vec3(0.0, 0.0, -GRAVITY).transpose();
        mag.row(i) = Vec3(0.2, 0.0, 0.4).transpose();
    }
    bool ok = eskf.initialize_from_sensors(accel, mag);
    assert(ok);
    assert(eskf.is_initialized());

    // Run 300 predict steps (stationary)
    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();

    for (int i = 0; i < 300; ++i) {
        eskf.predict(a, g, 0.01);
        if (i % 10 == 0) eskf.update_baro(0.0);
        if (i % 5 == 0)  eskf.update_mag(0.0);
    }

    auto state = eskf.get_state();

    // Position should stay near zero
    assert(state.pos.norm() < 5.0);

    // No NaN
    assert(!state.pos.hasNaN());
    assert(!state.vel.hasNaN());
    assert(!state.quat.hasNaN());

    // Should be healthy after 300 steps
    assert(eskf.get_health() == EKFHealth::HEALTHY);

    std::cout << "[PASS] Stationary predict\n";
    std::cout << "  Final pos: " << state.pos.transpose() << "\n";
    std::cout << "  Final vel: " << state.vel.transpose() << "\n";
}

void test_covariance_positive_definite()
{
    IMUNoiseParams noise;
    ESKFCore eskf(noise);

    Eigen::MatrixXd accel(50, 3);
    Eigen::MatrixXd mag(50, 3);
    for (int i = 0; i < 50; ++i) {
        accel.row(i) = Vec3(0.0, 0.0, -GRAVITY).transpose();
        mag.row(i) = Vec3(0.2, 0.0, 0.4).transpose();
    }
    eskf.initialize_from_sensors(accel, mag);

    Vec3 a(0.0, 0.0, -GRAVITY);
    Vec3 g = Vec3::Zero();
    for (int i = 0; i < 100; ++i) {
        eskf.predict(a, g, 0.01);
    }

    auto P = eskf.get_covariance();

    // Check symmetry
    double asym = (P - P.transpose()).norm();
    assert(asym < 1e-10);

    // Check positive definiteness via eigenvalues
    Eigen::SelfAdjointEigenSolver<MatP> solver(P);
    assert(solver.eigenvalues().minCoeff() > 0);

    std::cout << "[PASS] Covariance positive definite\n";
}

int main()
{
    std::cout << "NavCore ESKF C++ Port — Smoke Tests\n";
    std::cout << "====================================\n";

    test_construction();
    test_stationary_predict();
    test_covariance_positive_definite();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
