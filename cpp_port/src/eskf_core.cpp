/**
 * @file eskf_core.cpp
 * @brief 16-state ESKF implementation — C++17 + Eigen3.
 *
 * Stub implementations matching the Python ESKFCore API.
 * Core math to be ported from eskf_core.py.
 */

#include "eskf_core.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace navcore {

// ── Constructor ──────────────────────────────────────────────

ESKFCore::ESKFCore(const IMUNoiseParams& noise)
    : noise_(noise)
    , health_(EKFHealth::CONVERGING)
    , initialized_(false)
    , step_count_(0)
{
    // Initialize nominal state
    x_.setZero();
    x_(6) = 1.0;  // qw = 1 (identity quaternion)

    // Initialize error-state covariance
    P_.setIdentity();
    P_.block<3,3>(0,0) *= 1.0;     // position
    P_.block<3,3>(3,3) *= 0.1;     // velocity
    P_.block<3,3>(6,6) *= 0.01;    // attitude
    P_.block<3,3>(9,9) *= 0.01;    // accel bias
    P_.block<3,3>(12,12) *= 0.001; // gyro bias

    // Process noise
    Q_.setZero();
    double sa  = noise.accel_std * noise.accel_std;
    double sg  = noise.gyro_std * noise.gyro_std;
    double sab = 2.0 * noise.accel_bias_std * noise.accel_bias_std
                 / std::max(noise.accel_bias_tau, 1.0);
    double sgb = 2.0 * noise.gyro_bias_std * noise.gyro_bias_std
                 / std::max(noise.gyro_bias_tau, 1.0);

    Q_.block<3,3>(3,3) = Mat3::Identity() * sa;
    Q_.block<3,3>(6,6) = Mat3::Identity() * sg;
    Q_.block<3,3>(9,9) = Mat3::Identity() * sab;
    Q_.block<3,3>(12,12) = Mat3::Identity() * sgb;

    // Measurement matrices
    H_baro_.setZero();
    H_baro_(0, 2) = 1.0;  // observe pz

    H_mag_.setZero();
    H_mag_(0, 8) = 1.0;   // observe yaw (dtheta_z)
}

// ── Prediction ───────────────────────────────────────────────

void ESKFCore::predict(const Vec3& accel, const Vec3& gyro, double dt)
{
    if (!initialized_ || dt <= 0.0 || dt > 0.5) return;

    // Bias-corrected measurements
    Vec3 accel_corr = accel - x_.segment<3>(10);
    Vec3 gyro_corr  = gyro  - x_.segment<3>(13);

    // Current rotation
    Vec4 q = x_.segment<4>(6);
    Mat3 R = quat_to_rotation(q);

    // Gravity in NED
    Vec3 gravity_ned(0.0, 0.0, GRAVITY);

    // 1. Position update
    x_.segment<3>(0) += x_.segment<3>(3) * dt;

    // 2. Velocity update
    x_.segment<3>(3) += (R * accel_corr + gravity_ned) * dt;

    // 3. Quaternion update
    Vec3 dtheta = gyro_corr * dt;
    Vec4 dq;
    dq << 1.0, dtheta(0)/2.0, dtheta(1)/2.0, dtheta(2)/2.0;
    dq.normalize();
    x_.segment<4>(6) = quat_multiply(q, dq);
    x_.segment<4>(6).normalize();

    // 4. Build state transition matrix F (15x15)
    MatF F = MatF::Identity();

    // dp/dv
    F.block<3,3>(0,3) = Mat3::Identity() * dt;

    // dv/dtheta (skew-symmetric of R*accel_corr)
    Vec3 Ra = R * accel_corr;
    Mat3 skew;
    skew << 0, -Ra(2), Ra(1),
            Ra(2), 0, -Ra(0),
           -Ra(1), Ra(0), 0;
    F.block<3,3>(3,6) = -skew * dt;

    // dv/dba
    F.block<3,3>(3,9) = -R * dt;

    // dtheta/dbg
    F.block<3,3>(6,12) = -Mat3::Identity() * dt;

    // 5. Covariance propagation
    P_ = F * P_ * F.transpose() + Q_ * dt;

    // 6. Hardening
    step_count_++;
    harden_covariance();
    check_health();
}

// ── Baro Update ──────────────────────────────────────────────

void ESKFCore::update_baro(double alt_m)
{
    if (!initialized_) return;

    double z_pred = x_(2);
    double y = alt_m - z_pred;

    double R = noise_.baro_std * noise_.baro_std;
    double S = (H_baro_ * P_ * H_baro_.transpose())(0, 0) + R;

    // Innovation gating
    double nis = y * y / S;
    if (nis > CHI2_1DOF) return;

    // Kalman gain
    VecE K = P_ * H_baro_.transpose() / S;

    // Error state injection
    VecE dx = K * y;
    inject_error(dx);

    // Joseph form covariance update
    MatP I_KH = MatP::Identity() - K * H_baro_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
}

// ── Mag Update ───────────────────────────────────────────────

void ESKFCore::update_mag(double yaw_measured, double /*mag_norm*/,
                          double /*t_now*/)
{
    if (!initialized_) return;

    Vec4 q = x_.segment<4>(6);
    Vec3 euler = quat_to_euler(q);
    double yaw_pred = euler(2);

    double y = yaw_measured - yaw_pred;
    // Wrap to [-pi, pi]
    y = std::atan2(std::sin(y), std::cos(y));

    double R = noise_.mag_std * noise_.mag_std;
    double S = (H_mag_ * P_ * H_mag_.transpose())(0, 0) + R;

    double nis = y * y / S;
    if (nis > CHI2_1DOF) return;

    VecE K = P_ * H_mag_.transpose() / S;
    VecE dx = K * y;
    inject_error(dx);

    MatP I_KH = MatP::Identity() - K * H_mag_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
}

// ── Optical Flow Update ──────────────────────────────────────

void ESKFCore::update_optical_flow(double /*flow_vx*/, double /*flow_vy*/,
                                    double distance, int quality)
{
    if (!initialized_ || distance <= 0.05 || quality < 10) return;

    // TODO: Port full optical flow update from Python
}

// ── Initialization ───────────────────────────────────────────

bool ESKFCore::initialize_from_sensors(const Eigen::MatrixXd& accel_samples,
                                        const Eigen::MatrixXd& mag_samples)
{
    if (accel_samples.rows() < 10 || mag_samples.rows() < 10) {
        return false;
    }

    // Average accelerometer for gravity direction
    Vec3 accel_mean = accel_samples.colwise().mean().transpose();
    double g_norm = accel_mean.norm();
    if (g_norm < 5.0 || g_norm > 15.0) return false;

    // Roll and pitch from gravity
    double roll  = std::atan2(accel_mean(1), accel_mean(2));
    double pitch = std::atan2(-accel_mean(0),
                   std::sqrt(accel_mean(1)*accel_mean(1) +
                             accel_mean(2)*accel_mean(2)));

    // Yaw from magnetometer
    Vec3 mag_mean = mag_samples.colwise().mean().transpose();
    double cos_r = std::cos(roll), sin_r = std::sin(roll);
    double cos_p = std::cos(pitch), sin_p = std::sin(pitch);
    double mx = mag_mean(0)*cos_p + mag_mean(1)*sin_r*sin_p
                + mag_mean(2)*cos_r*sin_p;
    double my = mag_mean(1)*cos_r - mag_mean(2)*sin_r;
    double yaw = std::atan2(-my, mx);

    x_.segment<4>(6) = euler_to_quat(roll, pitch, yaw);
    initialized_ = true;

    return true;
}

// ── State Accessor ───────────────────────────────────────────

ESKFState ESKFCore::get_state() const
{
    ESKFState s;
    s.pos = x_.segment<3>(0);
    s.vel = x_.segment<3>(3);
    s.quat = x_.segment<4>(6);
    s.euler = quat_to_euler(s.quat);
    s.accel_bias = x_.segment<3>(10);
    s.gyro_bias = x_.segment<3>(13);
    return s;
}

// ── Error Injection ──────────────────────────────────────────

void ESKFCore::inject_error(const VecE& dx)
{
    x_.segment<3>(0) += dx.segment<3>(0);
    x_.segment<3>(3) += dx.segment<3>(3);

    Vec3 dtheta = dx.segment<3>(6);
    Vec4 dq;
    dq << 1.0, dtheta(0)/2.0, dtheta(1)/2.0, dtheta(2)/2.0;
    dq.normalize();
    x_.segment<4>(6) = quat_multiply(x_.segment<4>(6), dq);
    x_.segment<4>(6).normalize();

    x_.segment<3>(10) += dx.segment<3>(9);
    x_.segment<3>(13) += dx.segment<3>(12);

    // Clamp biases
    for (int i = 0; i < 3; ++i) {
        x_(10+i) = std::clamp(x_(10+i), -noise_.accel_bias_limit,
                               noise_.accel_bias_limit);
        x_(13+i) = std::clamp(x_(13+i), -noise_.gyro_bias_limit,
                               noise_.gyro_bias_limit);
    }
}

// ── Covariance Hardening ─────────────────────────────────────

void ESKFCore::harden_covariance()
{
    // Enforce symmetry
    P_ = (P_ + P_.transpose()) / 2.0;

    // Eigenvalue bounding
    Eigen::SelfAdjointEigenSolver<MatP> solver(P_);
    if (solver.info() != Eigen::Success) return;

    auto eigvals = solver.eigenvalues();
    bool needs_fix = false;

    constexpr double min_eig = 1e-9;
    constexpr double max_eig = 1e7;

    for (int i = 0; i < ERROR_DIM; ++i) {
        if (eigvals(i) < min_eig || eigvals(i) > max_eig) {
            needs_fix = true;
            break;
        }
    }

    if (needs_fix) {
        auto clamped = eigvals.cwiseMax(min_eig).cwiseMin(max_eig);
        P_ = solver.eigenvectors() * clamped.asDiagonal()
             * solver.eigenvectors().transpose();
        P_ = (P_ + P_.transpose()) / 2.0;
    }
}

// ── Health Check ─────────────────────────────────────────────

void ESKFCore::check_health()
{
    double vel_norm = x_.segment<3>(3).norm();
    Vec3 euler = quat_to_euler(x_.segment<4>(6));
    double tilt = std::sqrt(euler(0)*euler(0) + euler(1)*euler(1));
    tilt = tilt * 180.0 / M_PI;
    double p_trace = P_.trace();

    // NaN check
    if (x_.hasNaN() || P_.hasNaN()) {
        health_ = EKFHealth::FAULT;
        return;
    }

    // Fault conditions
    if (vel_norm > VEL_FAULT || tilt > TILT_FAULT_DEG || p_trace > P_TRACE_LIMIT) {
        health_ = EKFHealth::FAULT;
        return;
    }

    // Warning conditions
    if (vel_norm > VEL_WARN || tilt > TILT_WARN_DEG) {
        health_ = EKFHealth::WARNING;
        return;
    }

    // Convergence
    if (step_count_ > 200) {
        health_ = EKFHealth::HEALTHY;
    } else {
        health_ = EKFHealth::CONVERGING;
    }
}

// ── Quaternion Utilities ─────────────────────────────────────

Vec4 ESKFCore::quat_multiply(const Vec4& q1, const Vec4& q2)
{
    Vec4 result;
    result(0) = q1(0)*q2(0) - q1(1)*q2(1) - q1(2)*q2(2) - q1(3)*q2(3);
    result(1) = q1(0)*q2(1) + q1(1)*q2(0) + q1(2)*q2(3) - q1(3)*q2(2);
    result(2) = q1(0)*q2(2) - q1(1)*q2(3) + q1(2)*q2(0) + q1(3)*q2(1);
    result(3) = q1(0)*q2(3) + q1(1)*q2(2) - q1(2)*q2(1) + q1(3)*q2(0);
    return result;
}

Mat3 ESKFCore::quat_to_rotation(const Vec4& q)
{
    double w = q(0), x = q(1), y = q(2), z = q(3);
    Mat3 R;
    R << 1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
         2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
         2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y);
    return R;
}

Vec3 ESKFCore::quat_to_euler(const Vec4& q)
{
    double w = q(0), x = q(1), y = q(2), z = q(3);
    Vec3 euler;
    euler(0) = std::atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y));  // roll
    double sinp = 2*(w*y - z*x);
    euler(1) = (std::abs(sinp) >= 1)
                ? std::copysign(M_PI/2.0, sinp)
                : std::asin(sinp);  // pitch
    euler(2) = std::atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z));  // yaw
    return euler;
}

Vec4 ESKFCore::euler_to_quat(double roll, double pitch, double yaw)
{
    double cr = std::cos(roll/2),  sr = std::sin(roll/2);
    double cp = std::cos(pitch/2), sp = std::sin(pitch/2);
    double cy = std::cos(yaw/2),   sy = std::sin(yaw/2);

    Vec4 q;
    q(0) = cr*cp*cy + sr*sp*sy;
    q(1) = sr*cp*cy - cr*sp*sy;
    q(2) = cr*sp*cy + sr*cp*sy;
    q(3) = cr*cp*sy - sr*sp*cy;
    return q;
}

} // namespace navcore
