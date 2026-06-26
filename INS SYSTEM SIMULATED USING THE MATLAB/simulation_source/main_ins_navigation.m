%% =========================================================
%  Real-Time State Estimation using INS for UAV Navigation
%  GPS-Denied Environment | MATLAB Sensor Fusion
%  =========================================================
%  Author  : ARYA MGC
%  Date    : 2025
%  Version : 1.0
%
%  Description:
%    This script implements an Inertial Navigation System (INS)
%    for UAV navigation in GPS-denied environments. It uses IMU
%    data (accelerometer + gyroscope) with an Extended Kalman
%    Filter (EKF) for real-time state estimation at 50–100 Hz.
%
%  Simulink & Control Models:
%    Refer to 'controlmodel.png' and '800px-Drone_ctrl_simulink.png'
%    for the underlying system architecture and control logic.
% =========================================================

clc; clear; close all;

%% ── 0. Add src to path ──────────────────────────────────
addpath(genpath(fileparts(mfilename('fullpath'))));

%% ── 1. Simulation Parameters ────────────────────────────
dt          = 0.01;          % Time step (100 Hz)
T_total     = 30;            % Total simulation time (s)
t           = 0:dt:T_total;
N           = length(t);

fprintf('=== UAV INS Simulation ===\n');
fprintf('Duration : %.1f s  |  dt = %.4f s  |  Rate = %d Hz\n', ...
        T_total, dt, round(1/dt));

%% ── 2. IMU Noise Parameters ─────────────────────────────
imu = imu_noise_params();

%% ── 3. True Trajectory Generation ───────────────────────
[pos_true, vel_true, euler_true] = generate_trajectory(t, dt);

%% ── 4. IMU Data Simulation (with noise) ─────────────────
[accel_meas, gyro_meas] = simulate_imu(pos_true, vel_true, euler_true, ...
                                        dt, N, imu);

%% ── 5. EKF Initialisation ───────────────────────────────
[x_est, P_est] = ekf_init(pos_true(:,1), vel_true(:,1), euler_true(:,1));

%% ── 6. Storage Arrays ───────────────────────────────────
pos_ekf   = zeros(3, N);
vel_ekf   = zeros(3, N);
euler_ekf = zeros(3, N);
bias_acc  = zeros(3, N);   % Accel bias storage
bias_gyro = zeros(3, N);   % Gyro bias storage
pos_raw   = zeros(3, N);   % dead-reckoning only (no EKF)

pos_ekf(:,1)   = x_est(1:3);
vel_ekf(:,1)   = x_est(4:6);
euler_ekf(:,1) = x_est(7:9);
bias_acc(:,1)  = x_est(10:12);
bias_gyro(:,1) = x_est(13:15);
pos_raw(:,1)   = pos_true(:,1);

%% ── 7. Main Estimation Loop ─────────────────────────────
fprintf('\nRunning EKF estimation loop...\n');
tic;
vel_raw = vel_true(:,1);
pos_raw_cur = pos_true(:,1);

for k = 2:N
    %-- 7a. Dead-reckoning (raw IMU integration, no filter)
    [pos_raw_cur, vel_raw] = dead_reckon(pos_raw_cur, vel_raw, ...
                                          accel_meas(:,k), euler_true(:,k), dt);
    pos_raw(:,k) = pos_raw_cur;

    %-- 7b. EKF Predict
    [x_est, P_est] = ekf_predict(x_est, P_est, accel_meas(:,k), ...
                                  gyro_meas(:,k), dt, imu);

    %-- 7c. EKF Update (barometer at 10 Hz, magnetometer at 50 Hz)
    if mod(k, 10) == 0          % barometer @ 10 Hz
        z_baro = pos_true(3,k) + randn * imu.baro_std;
        [x_est, P_est] = ekf_update_baro(x_est, P_est, z_baro, imu);
    end
    if mod(k, 2) == 0           % magnetometer @ 50 Hz
        z_mag = euler_true(3,k) + randn * imu.mag_std;
        [x_est, P_est] = ekf_update_mag(x_est, P_est, z_mag, imu);
    end

    %-- 7d. Store results
    pos_ekf(:,k)   = x_est(1:3);
    vel_ekf(:,k)   = x_est(4:6);
    euler_ekf(:,k) = x_est(7:9);
    bias_acc(:,k)  = x_est(10:12);
    bias_gyro(:,k) = x_est(13:15);
end
elapsed = toc;
fprintf('Loop complete: %.3f s  (%.1f Hz effective)\n', elapsed, N/elapsed);

%% ── 8. Error Analysis ───────────────────────────────────
results = compute_errors(pos_true, vel_true, pos_ekf, vel_ekf, pos_raw);
print_error_summary(results);
fprintf('Final Accel Bias Estimates (m/s^2): X=%.4f Y=%.4f Z=%.4f\n', x_est(10), x_est(11), x_est(12));
fprintf('Final Gyro Bias Estimates (deg/s): X=%.4f Y=%.4f Z=%.4f\n', rad2deg(x_est(13)), rad2deg(x_est(14)), rad2deg(x_est(15)));

%% ── 9. Plots ────────────────────────────────────────────
plot_trajectory(t, pos_true, pos_ekf, pos_raw);
plot_errors(t, pos_true, pos_ekf, pos_raw, vel_true, vel_ekf);
plot_attitude(t, euler_true, euler_ekf);

fprintf('\nSimulation complete. Check figures for results.\n');
