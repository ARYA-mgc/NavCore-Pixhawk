function [pos, vel, euler] = generate_trajectory(t, dt)
% GENERATE_TRAJECTORY  Generates a smooth UAV reference trajectory.
%
%  The UAV performs:
%    0– 5 s  : takeoff (climb vertically)
%    5–15 s  : figure-8 horizontal manoeuvre at 20 m altitude
%   15–25 s  : banked turn with altitude change
%   25–30 s  : landing approach
%
%  Outputs (each 3×N):
%   pos   - [x; y; z]   positions in NED frame (m)
%   vel   - [vx;vy;vz]  velocities            (m/s)
%   euler - [roll;pitch;yaw]                  (rad)

N   = length(t);
pos = zeros(3, N);
vel = zeros(3, N);
euler = zeros(3, N);

v_cruise = 8;    % m/s
R_turn   = 15;   % m

for k = 1:N
    tk = t(k);

    if tk < 5
        %-- Takeoff
        pos(3,k) = -4 * tk;                     % climb (NED: z negative = up)
        vel(3,k) = -4;
        euler(1,k) = 0;   euler(2,k) = 0;   euler(3,k) = 0;

    elseif tk < 15
        %-- Figure-8 manoeuvre
        tau   = (tk - 5) / 10 * 2*pi;
        pos(1,k) = R_turn * sin(tau);
        pos(2,k) = R_turn * sin(tau) * cos(tau);
        pos(3,k) = -20;
        vel(1,k) = R_turn * cos(tau)       * (2*pi/10);
        vel(2,k) = R_turn * cos(2*tau)     * (2*pi/10);
        vel(3,k) = 0;
        euler(3,k) = atan2(vel(2,k), vel(1,k));
        euler(1,k) = 0.08 * sin(tau);       % mild roll in turns

    elseif tk < 25
        %-- Banked turn with altitude change
        tau   = (tk - 15) / 10 * pi;
        pos(1,k) = R_turn * cos(tau) + R_turn;
        pos(2,k) = R_turn * sin(tau) + 2*R_turn;
        pos(3,k) = -20 + 5*(tk-15)/10;      % gentle descent
        vel(1,k) = -R_turn * sin(tau) * (pi/10);
        vel(2,k) =  R_turn * cos(tau) * (pi/10);
        vel(3,k) = 0.5;
        euler(1,k) = 0.12;                   % banked
        euler(3,k) = atan2(vel(2,k), vel(1,k));

    else
        %-- Landing approach
        alpha = (tk - 25) / 5;
        pos(1,k) = R_turn * (1 - alpha);
        pos(2,k) = 2*R_turn;
        pos(3,k) = -15 * (1 - alpha);        % descend to ground
        vel(3,k) = 3;
        euler(2,k) = -0.05;                  % slight nose-down
    end
end

%-- Smooth with a short moving-average to remove step discontinuities
win = 5;
for ax = 1:3
    pos(ax,:)   = movmean(pos(ax,:),   win);
    vel(ax,:)   = movmean(vel(ax,:),   win);
    euler(ax,:) = movmean(euler(ax,:), win);
end
end
