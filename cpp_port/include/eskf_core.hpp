/**
 * @file eskf_core.hpp
 * @brief 21-state ESKF C++ header.
 *
 * Full port of Python eskf.py to standalone C++17 + Eigen3.
 * Standalone dependency-free implementation.
 * Runs on Raspberry Pi / ARM. Supports SCHED_FIFO real-time scheduling
 * Designed for hard real-time execution.
 *
 * 
 *
 * State layout (same as Python version):
 *   Nominal (21): [pos(3), vel(3), quat(4), accel_bias(3), gyro_bias(3),
 *                  baro_bias, clk_bias, clk_drift, wind_n, wind_e]
 *   Error (20):   [dp(3), dv(3), dtheta(3), dba(3), dbg(3),
 *                  d_baro, d_clk, d_clk_drift, d_wind(2)]
 *
 * Features:
 *   - Covariance convergence detection (z_cov < 0.25 + step > 200)
 *   - Adaptive process noise (scales with vibration magnitude)
 *   - ZUPT (zero velocity update)
 *   - Baro drift compensation (bias estimation)
 *   - Mag auto-cal (magnetic distortion rejection)
 *   - GPS fusion (WGS-84 to NED local tangent plane)
 *   - Generic external measurements (modular update framework)
 *   - Lidar range + radar velocity (high-bandwidth relative updates)
 *   - Optical flow (GNSS-denied velocity tracking)
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <optional>
#include <string>

namespace navcore {

// ── Constants (Constants) ──────────────────────

constexpr int STATE_DIM = 21;
constexpr int ERROR_DIM = 20;
constexpr double GRAVITY = 9.80665; // Standard gravity
constexpr double CHI2_1DOF = 3.841;   // 1-DOF chi-squared threshold
constexpr double CHI2_2DOF = 5.991;   // 2-DOF gate
constexpr double CHI2_3DOF = 7.815;   // 3-DOF gate
constexpr double R_EARTH = 6371000.0; // Earth radius approximation
constexpr double Z_COV_CONVERGED = 1.5; // Convergence threshold

// ── Type Aliases (Type Aliases) ──

using Vec3    = Eigen::Vector3d;
using Vec4    = Eigen::Vector4d;
using VecX    = Eigen::Matrix<double, STATE_DIM, 1>;
using VecE    = Eigen::Matrix<double, ERROR_DIM, 1>;
using Mat3    = Eigen::Matrix3d;
using Mat4    = Eigen::Matrix4d;
using MatP    = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>;
using MatQ    = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>;
using MatF    = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>;

// Dynamic-size types for generic updates
using VecXd   = Eigen::VectorXd;
using MatXd   = Eigen::MatrixXd;

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

// ── GPS Origin ───────────────────────────────────────────────

struct GPSOrigin {
    double lat = 0.0;
    double lon = 0.0;
    double alt = 0.0;
    bool is_set = false;
};

// ── State Output ─────────────────────────────────────────────

struct ESKFState {
    Vec3 pos;           // NED position (m)
    Vec3 vel;           // NED velocity (m/s)
    Vec4 quat;          // [w, x, y, z]
    Vec3 euler;         // [roll, pitch, yaw] (rad)
    Vec3 accel_bias;    // m/s²
    Vec3 gyro_bias;     // rad/s
    double baro_bias;   // m
    double clock_bias;  // m
    double clock_drift; // m/s
    Eigen::Vector2d wind; // [north, east] m/s
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

    void predict(const Vec3& accel, const Vec3& gyro, double dt);

    void update_baro(double alt_m);

    void update_mag(double yaw_measured, double mag_norm = -1.0,
                    double t_now = 0.0);

    void update_optical_flow(double flow_vx, double flow_vy,
                             double distance, int quality);

    /**
     * @brief Generic external measurement update (VIO, UWB, SLAM, etc.)
     * @return true if measurement was accepted
     */
    bool update_external(const VecXd& z, const VecXd& z_pred,
                         const MatXd& H, const MatXd& R,
                         const std::string& source = "external");

    /**
     * @brief GPS position update with WGS-84 → NED conversion.
     */
    void update_gps(double lat, double lon, double alt, double hdop = 1.0);

    /**
     * @brief Zero velocity update — forces v=[0,0,0].
     */
    void update_zupt();

    /**
     * @brief Radar Doppler velocity update.
     */
    void update_radar_velocity(double vx, double vy, double vz,
                                double weight = 1.0);

    /**
     * @brief Lidar range-to-ground altitude update.
     */
    void update_lidar_range(double distance, double weight = 1.0);

    /**
     * @brief Scale process noise based on vibration level.
     */
    void scale_process_noise(double vibration_level);

    bool initialize_from_sensors(const Eigen::MatrixXd& accel_samples,
                                  const Eigen::MatrixXd& mag_samples);

    void reset();

    // ── Accessors ───────────────────────────────────────────

    ESKFState get_state() const;
    EKFHealth get_health() const { return health_; }
    const MatP& get_covariance() const { return P_; }
    bool is_initialized() const { return initialized_; }
    double get_baro_bias() const { return baro_bias_; }
    double get_vibration_scale() const { return vibration_scale_; }

private:
    // ── Internal Methods ────────────────────────────────────

    void inject_error(const VecE& dx);
    void harden_covariance();
    void check_health();
    MatF compute_F(const Vec3& accel, const Vec3& gyro,
                   const Mat3& R, double dt) const;

    // Quaternion utilities
    static Vec4 quat_multiply(const Vec4& q1, const Vec4& q2);
    static Mat3 quat_to_rotation(const Vec4& q);
    static Vec3 quat_to_euler(const Vec4& q);
    static Vec4 euler_to_quat(double roll, double pitch, double yaw);
    static Mat3 skew_symmetric(const Vec3& v);
    static double wrap_angle(double a);

    // ── State ───────────────────────────────────────────────

    VecX x_;                    // Nominal state (16)
    MatP P_;                    // Error-state covariance (15x15)
    MatQ Q_base_;               // Base process noise (15x15)
    MatQ Q_;                    // Active process noise (scaled)
    IMUNoiseParams noise_;
    EKFHealth health_;
    bool initialized_;
    int step_count_;

    // Measurement matrices
    Eigen::Matrix<double, 1, ERROR_DIM> H_baro_;
    Eigen::Matrix<double, 1, ERROR_DIM> H_mag_;

    // GPS origin
    GPSOrigin gps_origin_;

    // Baro drift compensation
    double baro_bias_;
    double baro_bias_alpha_;
    int baro_update_count_;

    // Mag auto-calibration
    double calibrated_mag_norm_;
    double mag_reject_until_;
    int mag_consecutive_good_;

    // Adaptive noise
    double vibration_scale_;

    // Safety bounds
    static constexpr double VEL_WARN = 30.0;
    static constexpr double VEL_FAULT = 100.0;
    static constexpr double TILT_WARN_DEG = 60.0;
    static constexpr double TILT_FAULT_DEG = 80.0;
    static constexpr double P_TRACE_LIMIT = 1e9;
    static constexpr double ACCEL_BIAS_LIMIT = 2.0;
    static constexpr double GYRO_BIAS_LIMIT = 0.1;
    static constexpr double MAG_NORM_TOLERANCE = 0.30;
    static constexpr double MAG_REJECT_DURATION = 2.0;
};

} // namespace navcore
