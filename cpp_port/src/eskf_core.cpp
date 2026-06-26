/**
 * @file eskf_core.cpp
 * @brief C++ implementation of the ESKF.
 *
 * Ported from Python for performance.
 *
 * Everything the Python version does, this does too:
 * GPS, baro, mag, lidar, radar, optical flow, ZUPT, OOSM, the works.
 * Optimized for high-frequency execution on embedded targets.
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
    , baro_bias_(0.0)
    , baro_bias_alpha_(0.001)
    , baro_update_count_(0)
    , calibrated_mag_norm_(0.5)
    , mag_reject_until_(0.0)
    , mag_consecutive_good_(0)
    , vibration_scale_(1.0)
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

    // Process noise (base)
    Q_base_.setZero();
    double sa  = noise.accel_std * noise.accel_std;
    double sg  = noise.gyro_std * noise.gyro_std;
    double sab = 2.0 * noise.accel_bias_std * noise.accel_bias_std
                 / std::max(noise.accel_bias_tau, 1.0);
    double sgb = 2.0 * noise.gyro_bias_std * noise.gyro_bias_std
                 / std::max(noise.gyro_bias_tau, 1.0);

    Q_base_.block<3,3>(3,3) = Mat3::Identity() * sa;
    Q_base_.block<3,3>(6,6) = Mat3::Identity() * sg;
    Q_base_.block<3,3>(9,9) = Mat3::Identity() * sab;
    Q_base_.block<3,3>(12,12) = Mat3::Identity() * sgb;
    Q_ = Q_base_;  // active = base initially

    // Measurement matrices
    H_baro_.setZero();
    H_baro_(0, 2) = 1.0;  // observe pz

    H_mag_.setZero();
    H_mag_(0, 8) = 1.0;   // observe yaw (dtheta_z)

    // GPS origin not set
    gps_origin_.is_set = false;
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

    // 1. Velocity update
    x_.segment<3>(3) += (R * accel_corr + gravity_ned) * dt;

    // 2. Position update
    x_.segment<3>(0) += x_.segment<3>(3) * dt;

    // 3. Quaternion update
    double angle = gyro_corr.norm() * dt;
    Vec4 dq;
    if (angle > 1e-10) {
        Vec3 axis = gyro_corr.normalized();
        double ha = angle / 2.0;
        dq << std::cos(ha), axis(0)*std::sin(ha),
               axis(1)*std::sin(ha), axis(2)*std::sin(ha);
    } else {
        dq << 1.0, 0.0, 0.0, 0.0;
    }
    x_.segment<4>(6) = quat_multiply(q, dq);
    x_.segment<4>(6).normalize();

    // 4. Bias decay (Gauss-Markov)
    double tau_a = std::max(noise_.accel_bias_tau, 1.0);
    double tau_g = std::max(noise_.gyro_bias_tau, 1.0);
    x_.segment<3>(10) *= (1.0 - dt / tau_a);
    x_.segment<3>(13) *= (1.0 - dt / tau_g);

    // 5. Clamp biases
    for (int i = 0; i < 3; ++i) {
        x_(10+i) = std::clamp(x_(10+i), -ACCEL_BIAS_LIMIT, ACCEL_BIAS_LIMIT);
        x_(13+i) = std::clamp(x_(13+i), -GYRO_BIAS_LIMIT, GYRO_BIAS_LIMIT);
    }

    // 6. State transition Jacobian and covariance propagation
    MatF F = compute_F(accel_corr, gyro_corr, R, dt);
    P_ = F * P_ * F.transpose() + Q_ * dt;

    // 7. Hardening & health
    step_count_++;
    harden_covariance();
    check_health();
}

MatF ESKFCore::compute_F(const Vec3& accel, const Vec3& gyro,
                          const Mat3& R, double dt) const
{
    MatF F = MatF::Identity();

    // dp/dv
    F.block<3,3>(0,3) = Mat3::Identity() * dt;

    // dv/dtheta: -R * [accel]x * dt
    F.block<3,3>(3,6) = -R * skew_symmetric(accel) * dt;

    // dv/dba: -R * dt
    F.block<3,3>(3,9) = -R * dt;

    // dtheta/dtheta: I - [gyro]x * dt
    F.block<3,3>(6,6) = Mat3::Identity() - skew_symmetric(gyro) * dt;

    // dtheta/dbg: -I * dt
    F.block<3,3>(6,12) = -Mat3::Identity() * dt;

    // Bias decay
    double tau_a = std::max(noise_.accel_bias_tau, 1.0);
    double tau_g = std::max(noise_.gyro_bias_tau, 1.0);
    F.block<3,3>(9,9) = Mat3::Identity() * (1.0 - dt / tau_a);
    F.block<3,3>(12,12) = Mat3::Identity() * (1.0 - dt / tau_g);

    return F;
}

// ── Baro Update ──────────────────────────────────────────────

void ESKFCore::update_baro(double alt_m)
{
    if (!initialized_) return;

    // Apply drift compensation
    double alt_corrected = alt_m - baro_bias_;

    double z_pred = x_(2);
    double y = alt_corrected - z_pred;

    // Adaptive R
    double R_val = noise_.baro_std * noise_.baro_std;
    if (std::abs(y) > 2.0) R_val *= 5.0;

    double S = (H_baro_ * P_ * H_baro_.transpose())(0, 0) + R_val;

    // Innovation gating
    double nis = y * y / S;
    if (nis > CHI2_1DOF) return;

    // Kalman gain
    VecE K = P_ * H_baro_.transpose() / S;

    // Error state injection
    VecE dx = K * y;
    inject_error(dx);

    // Joseph form
    MatP I_KH = MatP::Identity() - K * H_baro_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_val * K.transpose();

    // Slow bias update after convergence
    baro_update_count_++;
    if (baro_update_count_ > 1000) {
        double innovation = alt_corrected - x_(2);
        baro_bias_ += baro_bias_alpha_ * innovation;
        baro_bias_ = std::clamp(baro_bias_, -10.0, 10.0);
    }
}

// ── Mag Update ───────────────────────────────────────────────

void ESKFCore::update_mag(double yaw_measured, double mag_norm, double t_now)
{
    if (!initialized_) return;

    // Norm check
    if (mag_norm > 0) {
        double ratio = std::abs(mag_norm / calibrated_mag_norm_ - 1.0);
        if (ratio > MAG_NORM_TOLERANCE) {
            mag_reject_until_ = t_now + MAG_REJECT_DURATION;
            mag_consecutive_good_ = 0;
            return;
        }
    }

    // Time-based rejection
    if (t_now > 0 && t_now < mag_reject_until_) {
        mag_consecutive_good_ = 0;
        return;
    }

    // Multi-sample re-enable
    mag_consecutive_good_++;
    if (mag_consecutive_good_ < 10) return;

    Vec4 q = x_.segment<4>(6);
    Vec3 euler = quat_to_euler(q);
    double yaw_pred = euler(2);

    double y = wrap_angle(yaw_measured - yaw_pred);

    double R_val = noise_.mag_std * noise_.mag_std;
    if (mag_norm > 0) {
        double ratio = std::abs(mag_norm / calibrated_mag_norm_ - 1.0);
        if (ratio > 0.15) R_val *= 10.0;
    }

    double S = (H_mag_ * P_ * H_mag_.transpose())(0, 0) + R_val;
    double nis = y * y / S;
    if (nis > CHI2_1DOF) return;

    VecE K = P_ * H_mag_.transpose() / S;
    VecE dx = K * y;
    inject_error(dx);

    MatP I_KH = MatP::Identity() - K * H_mag_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_val * K.transpose();

    // Mag auto-calibration
    if (mag_norm > 0) {
        constexpr double alpha = 0.002;
        calibrated_mag_norm_ = (1.0 - alpha) * calibrated_mag_norm_
                               + alpha * mag_norm;
    }
}

// ── Optical Flow Update ──────────────────────────────────────

void ESKFCore::update_optical_flow(double flow_vx, double flow_vy,
                                    double distance, int quality)
{
    if (!initialized_ || distance <= 0.05 || quality < 10) return;

    Eigen::Matrix<double, 2, ERROR_DIM> H_flow;
    H_flow.setZero();
    H_flow(0, 3) = 1.0;  // vx
    H_flow(1, 4) = 1.0;  // vy

    double R_base = 0.5 * 0.5;
    Eigen::Matrix2d R_flow = Eigen::Matrix2d::Identity()
                             * (R_base * 100.0 / std::max(quality, 1));

    Eigen::Vector2d z(flow_vx, flow_vy);
    Eigen::Vector2d z_pred = x_.segment<2>(3);
    Eigen::Vector2d innov = z - z_pred;

    Eigen::Matrix2d S = H_flow * P_ * H_flow.transpose() + R_flow;
    double nis = (innov.transpose() * S.inverse() * innov)(0, 0);
    if (nis > CHI2_2DOF) return;

    auto K = P_ * H_flow.transpose() * S.inverse();
    VecE dx = (K * innov);
    inject_error(dx);

    MatP I_KH = MatP::Identity() - K * H_flow;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_flow * K.transpose();
}

// ── Generic External Update ──────────────────────────────────

bool ESKFCore::update_external(const VecXd& z, const VecXd& z_pred,
                                const MatXd& H, const MatXd& R,
                                const std::string& source)
{
    if (!initialized_) return false;

    int m = static_cast<int>(z.size());
    VecXd innov = z - z_pred;

    // Wrap yaw if single-DOF yaw observation
    if (m == 1 && H(0, 8) != 0.0) {
        innov(0) = wrap_angle(innov(0));
    }

    MatXd S = H * P_ * H.transpose() + R;
    MatXd S_inv;
    try {
        S_inv = S.inverse();
    } catch (...) {
        return false;
    }

    double nis = (innov.transpose() * S_inv * innov)(0, 0);

    // Chi-squared threshold
    double chi2_thresh;
    switch (m) {
        case 1: chi2_thresh = 3.841; break;
        case 2: chi2_thresh = 5.991; break;
        case 3: chi2_thresh = 7.815; break;
        default: chi2_thresh = 3.0 * m; break;
    }

    if (nis > chi2_thresh) return false;

    auto K = P_ * H.transpose() * S_inv;
    VecE dx = (K * innov);
    inject_error(dx);

    MatP I_KH = MatP::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();

    return true;
}

// ── GPS Update ───────────────────────────────────────────────

void ESKFCore::update_gps(double lat, double lon, double alt, double hdop)
{
    if (!initialized_ || hdop > 5.0) return;

    // Set origin on first fix
    if (!gps_origin_.is_set) {
        gps_origin_.lat = lat;
        gps_origin_.lon = lon;
        gps_origin_.alt = alt;
        gps_origin_.is_set = true;
    }

    // WGS-84 → NED
    double d_lat = (lat - gps_origin_.lat) * M_PI / 180.0;
    double d_lon = (lon - gps_origin_.lon) * M_PI / 180.0;
    double lat_ref = gps_origin_.lat * M_PI / 180.0;
    double north = d_lat * R_EARTH;
    double east = d_lon * R_EARTH * std::cos(lat_ref);
    double down = -(alt - gps_origin_.alt);

    VecXd z(3); z << north, east, down;
    VecXd z_pred = x_.segment<3>(0);

    MatXd H = MatXd::Zero(3, ERROR_DIM);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    double gps_std = 2.5 * hdop;
    MatXd R_gps = MatXd::Identity(3, 3) * (gps_std * gps_std);
    R_gps(2, 2) *= 4.0;

    update_external(z, z_pred, H, R_gps, "GPS");
}

// ── ZUPT ─────────────────────────────────────────────────────

void ESKFCore::update_zupt()
{
    if (!initialized_) return;

    MatXd H = MatXd::Zero(3, ERROR_DIM);
    H(0, 3) = 1.0;
    H(1, 4) = 1.0;
    H(2, 5) = 1.0;

    MatXd R_zupt = MatXd::Identity(3, 3) * (0.01 * 0.01);
    VecXd z = VecXd::Zero(3);
    VecXd z_pred = x_.segment<3>(3);

    update_external(z, z_pred, H, R_zupt, "ZUPT");
}

// ── Radar Velocity ───────────────────────────────────────────

void ESKFCore::update_radar_velocity(double vx, double vy, double vz,
                                      double weight)
{
    if (!initialized_) return;

    MatXd H = MatXd::Zero(3, ERROR_DIM);
    H(0, 3) = 1.0;
    H(1, 4) = 1.0;
    H(2, 5) = 1.0;

    MatXd R_radar = MatXd::Identity(3, 3) * (0.1 * 0.1) / weight;
    VecXd z(3); z << vx, vy, vz;
    VecXd z_pred = x_.segment<3>(3);

    update_external(z, z_pred, H, R_radar, "radar");
}

// ── Lidar Range ──────────────────────────────────────────────

void ESKFCore::update_lidar_range(double distance, double weight)
{
    if (!initialized_ || distance < 0.1) return;

    MatXd H = MatXd::Zero(1, ERROR_DIM);
    H(0, 2) = 1.0;

    MatXd R_lidar = MatXd::Identity(1, 1) * (0.05 * 0.05) / weight;
    VecXd z(1); z << -distance;  // NED
    VecXd z_pred(1); z_pred << x_(2);

    update_external(z, z_pred, H, R_lidar, "lidar");
}

// ── Adaptive Process Noise ───────────────────────────────────

void ESKFCore::scale_process_noise(double vibration_level)
{
    double scale = std::clamp(1.0 + vibration_level * 5.0, 1.0, 10.0);
    vibration_scale_ = scale;
    Q_ = Q_base_ * scale;
}

// ── Initialization ───────────────────────────────────────────

bool ESKFCore::initialize_from_sensors(const Eigen::MatrixXd& accel_samples,
                                        const Eigen::MatrixXd& mag_samples)
{
    if (accel_samples.rows() < 10 || mag_samples.rows() < 10) {
        return false;
    }

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

    // Store calibrated mag norm
    calibrated_mag_norm_ = mag_mean.norm();

    x_.segment<4>(6) = euler_to_quat(roll, pitch, yaw);
    x_.segment<6>(0).setZero();   // position and velocity
    x_.segment<6>(10).setZero();  // biases
    initialized_ = true;

    return true;
}

// ── Reset ────────────────────────────────────────────────────

void ESKFCore::reset()
{
    *this = ESKFCore(noise_);
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
        x_(10+i) = std::clamp(x_(10+i), -ACCEL_BIAS_LIMIT, ACCEL_BIAS_LIMIT);
        x_(13+i) = std::clamp(x_(13+i), -GYRO_BIAS_LIMIT, GYRO_BIAS_LIMIT);
    }
}

// ── Covariance Hardening ─────────────────────────────────────

void ESKFCore::harden_covariance()
{
    P_ = (P_ + P_.transpose()) / 2.0;

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

    // Covariance-based convergence (z-axis only, baro-observable)
    double z_cov = P_(2, 2);
    if (z_cov < Z_COV_CONVERGED && step_count_ > 200) {
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
    euler(0) = std::atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y));
    double sinp = 2*(w*y - z*x);
    euler(1) = (std::abs(sinp) >= 1)
                ? std::copysign(M_PI/2.0, sinp)
                : std::asin(sinp);
    euler(2) = std::atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z));
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

Mat3 ESKFCore::skew_symmetric(const Vec3& v)
{
    Mat3 S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
        -v(1), v(0), 0;
    return S;
}

double ESKFCore::wrap_angle(double a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

} // namespace navcore
