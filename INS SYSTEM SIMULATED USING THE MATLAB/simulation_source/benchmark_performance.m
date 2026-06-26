%% =========================================================
%  Performance Benchmark – Frequency vs Accuracy trade-off
%  Runs simulation at 50 Hz and 100 Hz, compares RMSE.
% =========================================================
clc; clear; close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')),'..','src')));

fprintf('=== INS Performance Benchmark ===\n\n');

rates_hz = [50, 100];
results_table = zeros(length(rates_hz), 4);

for ri = 1:length(rates_hz)
    hz  = rates_hz(ri);
    dt  = 1/hz;
    T   = 20;
    t   = 0:dt:T;
    N   = length(t);

    imu = imu_noise_params();
    [pos_t, vel_t, eul_t] = generate_trajectory(t, dt);
    [a_meas, g_meas] = simulate_imu(pos_t, vel_t, eul_t, dt, N, imu);

    [x, P] = ekf_init(pos_t(:,1), vel_t(:,1), eul_t(:,1));
    pos_ekf = zeros(3,N);
    pos_ekf(:,1) = x(1:3);

    tic;
    for k = 2:N
        [x, P] = ekf_predict(x, P, a_meas(:,k), g_meas(:,k), dt, imu);
        if mod(k,round(hz/10))==0
            z_b = pos_t(3,k) + randn*imu.baro_std;
            [x,P] = ekf_update_baro(x,P,z_b,imu);
        end
        if mod(k,round(hz/50))==0
            z_m = eul_t(3,k) + randn*imu.mag_std;
            [x,P] = ekf_update_mag(x,P,z_m,imu);
        end
        pos_ekf(:,k) = x(1:3);
    end
    elapsed = toc;

    err = sqrt(sum((pos_ekf - pos_t).^2, 1));
    results_table(ri,:) = [hz, rms(err), max(err), N/elapsed];
    fprintf('%3d Hz | RMSE=%.3f m | MaxErr=%.3f m | Eff.Rate=%.1f Hz\n', ...
            results_table(ri,:));
end

fprintf('\nBenchmark complete.\n');

%-- Bar chart
figure('Color','w','Position',[100 100 700 350]);
subplot(1,2,1);
bar(results_table(:,1), results_table(:,2));
xticklabels({'50 Hz','100 Hz'});
ylabel('Position RMSE (m)'); title('RMSE vs Update Rate'); grid on;

subplot(1,2,2);
bar(results_table(:,1), results_table(:,4));
xticklabels({'50 Hz','100 Hz'});
ylabel('Effective Rate (Hz)'); title('Computational Throughput'); grid on;

saveas(gcf, fullfile('..','results','benchmark_results.png'));
