%% =========================================================
%  Unit Tests – UAV INS Components
%  Run this script to verify all core functions.
% =========================================================
clc; clear;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..','src')));

passed = 0; failed = 0;

fprintf('=== Running UAV INS Unit Tests ===\n\n');

%% Test 1: IMU noise params
try
    imu = imu_noise_params();
    assert(imu.accel_std > 0,    'accel_std must be positive');
    assert(size(imu.Q,1) == 9,   'Q must be 9×9');
    assert(imu.R_baro > 0,       'R_baro must be positive');
    fprintf('[PASS] imu_noise_params\n');  passed = passed+1;
catch e
    fprintf('[FAIL] imu_noise_params: %s\n', e.message); failed = failed+1;
end

%% Test 2: Trajectory generation
try
    dt = 0.01; T = 5; t = 0:dt:T; N = length(t);
    [pos, vel, euler] = generate_trajectory(t, dt);
    assert(size(pos,1)==3 && size(pos,2)==N, 'pos wrong size');
    assert(size(vel,1)==3 && size(vel,2)==N, 'vel wrong size');
    assert(~any(isnan(pos(:))), 'NaN in pos');
    fprintf('[PASS] generate_trajectory\n'); passed = passed+1;
catch e
    fprintf('[FAIL] generate_trajectory: %s\n', e.message); failed = failed+1;
end

%% Test 3: EKF init
try
    [x0, P0] = ekf_init([0;0;0], [0;0;0], [0;0;0]);
    assert(length(x0)==9,           'x0 must be 9-element');
    assert(all(size(P0)==[9,9]),    'P0 must be 9×9');
    assert(all(diag(P0)>0),         'P0 must be positive definite');
    fprintf('[PASS] ekf_init\n'); passed = passed+1;
catch e
    fprintf('[FAIL] ekf_init: %s\n', e.message); failed = failed+1;
end

%% Test 4: EKF predict — state size preserved
try
    imu = imu_noise_params();
    [x0, P0] = ekf_init([0;0;-10], [5;0;0], [0;0.1;0.5]);
    a_b = [0;0;-9.6]; w_b = [0;0;0.05];
    [x1, P1] = ekf_predict(x0, P0, a_b, w_b, 0.01, imu);
    assert(length(x1)==9 && all(size(P1)==[9,9]), 'wrong output size');
    assert(~any(isnan(x1)), 'NaN in predicted state');
    fprintf('[PASS] ekf_predict\n'); passed = passed+1;
catch e
    fprintf('[FAIL] ekf_predict: %s\n', e.message); failed = failed+1;
end

%% Test 5: EKF baro update reduces altitude covariance
try
    imu = imu_noise_params();
    [x0, P0] = ekf_init([0;0;-10], [0;0;0], [0;0;0]);
    [x1, P1] = ekf_update_baro(x0, P0, -10.1, imu);
    assert(P1(3,3) < P0(3,3), 'Baro update should reduce P(3,3)');
    fprintf('[PASS] ekf_update_baro\n'); passed = passed+1;
catch e
    fprintf('[FAIL] ekf_update_baro: %s\n', e.message); failed = failed+1;
end

%% Test 6: EKF mag update reduces yaw covariance
try
    imu = imu_noise_params();
    [x0, P0] = ekf_init([0;0;0], [0;0;0], [0;0;0.3]);
    [x1, P1] = ekf_update_mag(x0, P0, 0.28, imu);
    assert(P1(9,9) < P0(9,9), 'Mag update should reduce P(9,9)');
    fprintf('[PASS] ekf_update_mag\n'); passed = passed+1;
catch e
    fprintf('[FAIL] ekf_update_mag: %s\n', e.message); failed = failed+1;
end

%% Test 7: Dead reckoning output size
try
    imu = imu_noise_params();
    [p2, v2] = dead_reckon([0;0;0],[5;0;0],[0;0;-9.6],[0;0.1;0.5],0.01);
    assert(length(p2)==3 && length(v2)==3, 'Wrong output size');
    fprintf('[PASS] dead_reckon\n'); passed = passed+1;
catch e
    fprintf('[FAIL] dead_reckon: %s\n', e.message); failed = failed+1;
end

%% Test 8: Error analysis
try
    N = 100;
    pt = rand(3,N); pe = pt + 0.1*rand(3,N); pr = pt + 0.5*rand(3,N);
    vt = rand(3,N); ve = vt + 0.05*rand(3,N);
    r = compute_errors(pt,vt,pe,ve,pr);
    assert(r.improvement > 0, 'EKF should outperform dead-reckoning');
    fprintf('[PASS] compute_errors\n'); passed = passed+1;
catch e
    fprintf('[FAIL] compute_errors: %s\n', e.message); failed = failed+1;
end

%% Summary
fprintf('\n=== Results: %d passed, %d failed ===\n', passed, failed);
if failed == 0
    fprintf('All tests PASSED ✓\n\n');
else
    fprintf('Some tests FAILED – review above output.\n\n');
end
