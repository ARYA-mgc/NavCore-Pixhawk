/**
 * @file eskf_core.hpp
 * @brief 16-state Error-State Kalman Filter (ESKF) — C++ port.
 *
 * Direct port of Python eskf_core.py to standalone C++17 + Eigen3.
 * No ROS2 dependency. Designed for Raspberry Pi / ARM deployment.
 *
 * Nominal state (16):
 *   x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
 *
 * Error state (15):
 *   dx = [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <array>

namespace navcore {

// ── Constants ────────────────────────────────────────────────

constexpr int STATE_DIM = 16;
constexpr int ERROR_DIM = 15;
constexpr double GRAVITY = 9.80665;
constexpr double CHI2_1DOF = 5.991;
constexpr double CHI2_2DOF = 9.210;

// ── Type Aliases ─────────────────────────────────────────────

using Vec3    = Eigen::Vector3d;
using Vec4    = Eigen::Vector4d;
using VecX    = Eigen::Matrix<double, STATE_DIM, 1>;
using VecE    = Eigen::Matrix<double, ERROR_DIM, 1>;
using Mat3    = Eigen::Matrix3d;
using Mat4    = Eigen::Matrix4d;
using MatP    = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>;
using MatQ    = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>;
using MatF    = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>;

// ── Noise Parameters ─────────────────────────────────────────

struct IMUNoiseParams {
    double accel_std       = 0.05;      // m/s²
    double accel_bias_std  = 0.02;      // m/s²
    double accel_bias_tau  = 300.0;     // s
    double accel_bias_limit = 2.0;      // m/s²
    double gyro_std        = 0.005;     // rad/s
    double gyro_bias_std   = 0.001;     // rad/s
    double gyro_bias_tau   = 300.0;     // s
    double gyro_bias_limit = 0.1;       // rad/s
    double baro_std        = 0.30;      // m
    double mag_std         = 0.02;      // rad
};

// ── Health Status ────────────────────────────────────────────

enum class EKFHealth {
    CONVERGING = 0,
    HEALTHY    = 1,
    WARNING    = 2,
    FAULT      = 3,
};

// ── State Output ─────────────────────────────────────────────

struct ESKFState {
    Vec3 pos;           // NED position (m)
    Vec3 vel;           // NED velocity (m/s)
    Vec4 quat;          // [w, x, y, z]
    Vec3 euler;         // [roll, pitch, yaw] (rad)
    Vec3 accel_bias;    // m/s²
    Vec3 gyro_bias;     // rad/s
};

// ── ESKF Core Class ──────────────────────────────────────────

class ESKFCore {
public:
    explicit ESKFCore(const IMUNoiseParams& noise);
    ~ESKFCore() = default;

    // Non-copyable, movable
    ESKFCore(const ESKFCore&) = delete;
    ESKFCore& operator=(const ESKFCore&) = delete;
    ESKFCore(ESKFCore&&) = default;
    ESKFCore& operator=(ESKFCore&&) = default;

    // ── Core Operations ─────────────────────────────────────

    /**
     * @brief IMU prediction step.
     * @param accel Raw accelerometer reading (m/s², body frame).
     * @param gyro  Raw gyroscope reading (rad/s, body frame).
     * @param dt    Time step (seconds).
     */
    void predict(const Vec3& accel, const Vec3& gyro, double dt);

    /**
     * @brief Barometric altitude update.
     * @param alt_m Measured altitude (m, NED down = positive).
     */
    void update_baro(double alt_m);

    /**
     * @brief Magnetometer yaw update.
     * @param yaw_measured Yaw angle (rad).
     * @param mag_norm     Measured field magnitude (Gauss), -1 to skip check.
     * @param t_now        Current time for EMI rejection.
     */
    void update_mag(double yaw_measured, double mag_norm = -1.0,
                    double t_now = 0.0);

    /**
     * @brief Optical flow velocity update.
     * @param flow_vx  Flow velocity X (m/s).
     * @param flow_vy  Flow velocity Y (m/s).
     * @param distance Rangefinder distance (m).
     * @param quality  Flow quality (0-255).
     */
    void update_optical_flow(double flow_vx, double flow_vy,
                             double distance, int quality);

    /**
     * @brief Initialize from stationary sensor data.
     * @param accel_samples (N, 3) accelerometer samples.
     * @param mag_samples   (N, 3) magnetometer samples.
     * @return True if initialization succeeded.
     */
    bool initialize_from_sensors(const Eigen::MatrixXd& accel_samples,
                                  const Eigen::MatrixXd& mag_samples);

    // ── Accessors ───────────────────────────────────────────

    ESKFState get_state() const;
    EKFHealth get_health() const { return health_; }
    const MatP& get_covariance() const { return P_; }
    bool is_initialized() const { return initialized_; }

private:
    // ── Internal Methods ────────────────────────────────────

    void inject_error(const VecE& dx);
    void harden_covariance();
    void check_health();

    // Quaternion utilities
    static Vec4 quat_multiply(const Vec4& q1, const Vec4& q2);
    static Mat3 quat_to_rotation(const Vec4& q);
    static Vec3 quat_to_euler(const Vec4& q);
    static Vec4 euler_to_quat(double roll, double pitch, double yaw);

    // ── State ───────────────────────────────────────────────

    VecX x_;                    // Nominal state (16)
    MatP P_;                    // Error-state covariance (15x15)
    MatQ Q_;                    // Process noise (15x15)
    IMUNoiseParams noise_;
    EKFHealth health_;
    bool initialized_;
    int step_count_;

    // Measurement matrices
    Eigen::Matrix<double, 1, ERROR_DIM> H_baro_;
    Eigen::Matrix<double, 1, ERROR_DIM> H_mag_;

    // Safety bounds
    static constexpr double VEL_WARN = 30.0;
    static constexpr double VEL_FAULT = 100.0;
    static constexpr double TILT_WARN_DEG = 60.0;
    static constexpr double TILT_FAULT_DEG = 80.0;
    static constexpr double P_TRACE_LIMIT = 1e6;
};

} // namespace navcore
