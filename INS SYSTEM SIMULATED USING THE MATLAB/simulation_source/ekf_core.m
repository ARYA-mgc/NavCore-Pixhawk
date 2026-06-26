%% =========================================================
%  Extended Kalman Filter — 15-State Core Functions
%  =========================================================
%  State vector x (15×1):
%    x(1:3)   = position        [m] (NED)
%    x(4:6)   = velocity        [m/s] (NED)
%    x(7:9)   = Euler angles    [rad] (roll, pitch, yaw)
%    x(10:12) = Accel bias      [m/s²]
%    x(13:15) = Gyro bias       [rad/s]
% =========================================================

function [x, P] = ekf_init(pos0, vel0, euler0)
    % Initialise EKF with 15 states
    x = zeros(15, 1);
    x(1:3) = pos0;
    x(4:6) = vel0;
    x(7:9) = euler0;
    
    % Initial covariance: trust position/velocity/euler, assume small bias
    P = diag([1,1,1, 0.5,0.5,0.5, 0.1,0.1,0.1, 0.01,0.01,0.01, 0.001,0.001,0.001].^2);
end

function [x_new, P_new] = ekf_predict(x, P, accel_raw, gyro_raw, dt, imu)
    % 1. Extract states
    pos   = x(1:3);
    vel   = x(4:6);
    euler = x(7:9);
    ba    = x(10:12); % Accel bias
    bg    = x(13:15); % Gyro bias
    
    % 2. Compensate for Biases
    accel = accel_raw - ba;
    gyro  = gyro_raw - bg;
    
    roll = euler(1); pitch = euler(2); yaw = euler(3);
    R_bn = rot_body2ned(roll, pitch, yaw);
    g_ned = [0; 0; 9.81];
    
    % 3. State Propagation (Kinematics)
    a_ned = R_bn * accel + g_ned;
    vel_new = vel + a_ned * dt;
    pos_new = pos + vel * dt + 0.5 * a_ned * dt^2;
    
    % Euler kinematics
    cr = cos(roll); sr = sin(roll); cp = cos(pitch); sp = sin(pitch);
    T_inv = [1, sr*sp/cp, cr*sp/cp;
             0, cr,       -sr;
             0, sr/cp,    cr/cp];
    euler_dot = T_inv * gyro;
    euler_new = angle_wrap(euler + euler_dot * dt);
    
    x_new = [pos_new; vel_new; euler_new; ba; bg]; % Biases assumed constant (random walk in Q)
    
    % 4. Jacobian F (15x15)
    F = eye(15);
    F(1:3, 4:6) = eye(3) * dt;
    
    % Velocity w.r.t Euler
    da_dr = R_bn * skew([1;0;0]) * accel * dt;
    da_dp = R_bn * skew([0;1;0]) * accel * dt;
    da_dy = R_bn * skew([0;0;1]) * accel * dt;
    F(4:6, 7) = da_dr;
    F(4:6, 8) = da_dp;
    F(4:6, 9) = da_dy;
    
    % Velocity w.r.t Accel Bias
    F(4:6, 10:12) = -R_bn * dt;
    
    % Euler w.r.t Gyro Bias
    F(7:9, 13:15) = -T_inv * dt;
    
    % 5. Covariance Propagation
    % imu.Q should be 15x15 (or padded)
    Q15 = eye(15) * 1e-6; 
    Q15(4:6, 4:6) = eye(3) * (imu.accel_std^2);
    Q15(7:9, 7:9) = eye(3) * (imu.gyro_std^2);
    
    P_new = F * P * F' + Q15 * dt;
end

function [x_new, P_new] = ekf_update_baro(x, P, z_baro, imu)
    H = zeros(1, 15);
    H(3) = 1; % Observe NED z-position
    
    S = H * P * H' + imu.R_baro;
    K = (P * H') / S;
    
    innov = z_baro - x(3);
    x_new = x + K * innov;
    x_new(7:9) = angle_wrap(x_new(7:9));
    P_new = (eye(15) - K*H) * P;
end

function [x_new, P_new] = ekf_update_mag(x, P, z_yaw, imu)
    H = zeros(1, 15);
    H(9) = 1; % Observe yaw
    
    S = H * P * H' + imu.R_mag;
    K = (P * H') / S;
    
    innov = angle_wrap(z_yaw - x(9));
    x_new = x + K * innov;
    x_new(7:9) = angle_wrap(x_new(7:9));
    P_new = (eye(15) - K*H) * P;
end

% ── Helpers ──────────────────────────────────────────────
function R = rot_body2ned(r,p,y)
    cr=cos(r);sr=sin(r);cp=cos(p);sp=sin(p);cy=cos(y);sy=sin(y);
    R=[cp*cy-sr*sp*sy, -cr*sy, sp*cy+sr*cp*sy;
       cp*sy+sr*sp*cy,  cr*cy, sp*sy-sr*cp*cy;
       -cr*sp,          sr,    cr*cp];
end

function S = skew(v)
    S=[0,-v(3),v(2); v(3),0,-v(1); -v(2),v(1),0];
end

function a = angle_wrap(a)
    a = mod(a+pi, 2*pi) - pi;
end
