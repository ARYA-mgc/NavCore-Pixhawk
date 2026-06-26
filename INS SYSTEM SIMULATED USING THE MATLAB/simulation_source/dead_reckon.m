function [pos_new, vel_new] = dead_reckon(pos, vel, accel_b, euler, dt)
% DEAD_RECKON  Pure IMU integration without any filter correction.
%   Used as baseline to show drift vs EKF.

roll=euler(1); pitch=euler(2); yaw=euler(3);
cr=cos(roll);sr=sin(roll);cp=cos(pitch);sp=sin(pitch);cy=cos(yaw);sy=sin(yaw);

R = [cp*cy-sr*sp*sy, -cr*sy, sp*cy+sr*cp*sy;
     cp*sy+sr*sp*cy,  cr*cy, sp*sy-sr*cp*cy;
     -cr*sp,          sr,    cr*cp];

g_ned  = [0;0;9.81];
a_ned  = R * accel_b + g_ned;
vel_new = vel + a_ned * dt;
pos_new = pos + vel * dt;
end

% =========================================================
function results = compute_errors(pos_true, vel_true, pos_ekf, vel_ekf, pos_raw)
% COMPUTE_ERRORS  Computes RMSE, max error, and drift metrics.

pos_err_ekf = pos_ekf  - pos_true;
pos_err_raw = pos_raw  - pos_true;
vel_err_ekf = vel_ekf  - vel_true;

% RMSE
results.pos_rmse_ekf  = sqrt(mean(sum(pos_err_ekf.^2, 1)));
results.pos_rmse_raw  = sqrt(mean(sum(pos_err_raw.^2, 1)));
results.vel_rmse_ekf  = sqrt(mean(sum(vel_err_ekf.^2, 1)));

% Max horizontal error
results.pos_max_ekf   = max(sqrt(sum(pos_err_ekf.^2, 1)));
results.pos_max_raw   = max(sqrt(sum(pos_err_raw.^2, 1)));

% Drift improvement (%)
results.improvement   = (1 - results.pos_rmse_ekf / results.pos_rmse_raw) * 100;

% Per-axis RMSE
results.x_rmse_ekf    = rms(pos_err_ekf(1,:));
results.y_rmse_ekf    = rms(pos_err_ekf(2,:));
results.z_rmse_ekf    = rms(pos_err_ekf(3,:));
end

% =========================================================
function print_error_summary(r)
fprintf('\n======= Error Analysis Summary =======\n');
fprintf('  EKF  Position RMSE : %6.3f m\n', r.pos_rmse_ekf);
fprintf('  Raw  Position RMSE : %6.3f m\n', r.pos_rmse_raw);
fprintf('  EKF  Velocity RMSE : %6.3f m/s\n', r.vel_rmse_ekf);
fprintf('  EKF  Max Pos Error : %6.3f m\n', r.pos_max_ekf);
fprintf('  Drift Improvement  : %6.1f %%\n', r.improvement);
fprintf('  Per-axis RMSE  X=%.3f  Y=%.3f  Z=%.3f m\n', ...
        r.x_rmse_ekf, r.y_rmse_ekf, r.z_rmse_ekf);
fprintf('======================================\n\n');
end
