function imu = imu_noise_params()
% IMU_NOISE_PARAMS  Returns IMU sensor noise and bias parameters.
%
%  Returns struct with fields:
%   accel_std   - accelerometer white noise std  [m/s^2]
%   accel_bias  - accelerometer bias             [m/s^2]
%   gyro_std    - gyroscope white noise std      [rad/s]
%   gyro_bias   - gyroscope bias                 [rad/s]
%   baro_std    - barometer altitude noise std   [m]
%   mag_std     - magnetometer heading noise std [rad]
%   Q_accel     - process noise covariance (accel)
%   Q_gyro      - process noise covariance (gyro)
%   R_baro      - measurement noise (barometer)
%   R_mag       - measurement noise (magnetometer)

% --- Consumer-grade MEMS IMU (e.g. MPU-6050 class) ---
imu.accel_std  = 0.05;          % m/s^2
imu.accel_bias = 0.02;          % m/s^2  (constant bias)
imu.gyro_std   = 0.005;         % rad/s
imu.gyro_bias  = 0.001;         % rad/s

% --- Aiding sensors ---
imu.baro_std   = 0.3;           % m   (altitude)
imu.mag_std    = 0.02;          % rad (yaw)

% --- EKF noise matrices ---
% Process noise Q  (9-state: pos, vel, euler)
q_pos   = (imu.accel_std * 0.01)^2;
q_vel   = imu.accel_std^2;
q_euler = imu.gyro_std^2;
imu.Q = diag([q_pos*ones(1,3), q_vel*ones(1,3), q_euler*ones(1,3)]);

% Measurement noise R
imu.R_baro = imu.baro_std^2;
imu.R_mag  = imu.mag_std^2;
end
