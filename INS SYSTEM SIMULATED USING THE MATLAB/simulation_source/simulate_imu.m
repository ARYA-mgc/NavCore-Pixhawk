function [accel_meas, gyro_meas] = simulate_imu(pos, vel, euler, dt, N, imu)
% SIMULATE_IMU  Generates noisy IMU measurements from the true trajectory.
%
%  Accelerometer: specific force in body frame = R^T*(a_true - g)
%  Gyroscope    : angular rate in body frame from Euler angle rates
%
%  Inputs:
%   pos, vel, euler  - true state (3×N)
%   dt               - time step
%   N                - number of samples
%   imu              - noise parameter struct
%
%  Outputs:
%   accel_meas  - noisy specific force   (3×N)  [m/s^2]
%   gyro_meas   - noisy angular rate     (3×N)  [rad/s]

g = [0; 0; 9.81];    % gravity in NED (z down)

accel_meas = zeros(3, N);
gyro_meas  = zeros(3, N);

% Constant biases (random draw per run – mimics warm-start bias)
ab = imu.accel_bias * randn(3,1);
gb = imu.gyro_bias  * randn(3,1);

for k = 1:N
    roll  = euler(1,k);
    pitch = euler(2,k);
    yaw   = euler(3,k);

    %-- Rotation matrix: body ← NED
    R = rot_ned2body(roll, pitch, yaw);

    %-- True acceleration in NED
    if k < N
        a_ned = (vel(:,k+1) - vel(:,k)) / dt;
    else
        a_ned = (vel(:,k) - vel(:,k-1)) / dt;
    end

    %-- Specific force in body frame
    f_body = R * (a_ned - g);

    %-- Add noise + bias
    accel_meas(:,k) = f_body + ab + imu.accel_std * randn(3,1);

    %-- Angular rate from Euler angle derivatives
    if k < N
        dEuler = (euler(:,k+1) - euler(:,k)) / dt;
    else
        dEuler = (euler(:,k) - euler(:,k-1)) / dt;
    end
    omega_body = euler_rate2body(dEuler, roll, pitch);

    %-- Add noise + bias
    gyro_meas(:,k) = omega_body + gb + imu.gyro_std * randn(3,1);
end
end

% ── Helper: NED → Body rotation matrix (ZYX convention) ──
function R = rot_ned2body(r, p, y)
cr=cos(r); sr=sin(r); cp=cos(p); sp=sin(p); cy=cos(y); sy=sin(y);
R = [cp*cy,          cp*sy,         -sp;
     sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp;
     cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp];
end

% ── Helper: Euler rates → body angular rates ──
function w = euler_rate2body(dE, r, p)
cr=cos(r); sr=sin(r); cp=cos(p); sp=sin(p);
T = [1,  0,    -sp;
     0,  cr,    sr*cp;
     0, -sr,    cr*cp];
w = T * dE;
end
