function plot_trajectory(t, pos_true, pos_ekf, pos_raw)
% PLOT_TRAJECTORY  3-D and 2-D trajectory comparison.

figure('Name','Trajectory Comparison','Color','w','Position',[50 50 1200 500]);

%-- 3-D trajectory
subplot(1,2,1);
plot3(pos_true(1,:), pos_true(2,:), -pos_true(3,:), 'k-',  'LineWidth',2); hold on;
plot3(pos_ekf(1,:),  pos_ekf(2,:),  -pos_ekf(3,:),  'b--', 'LineWidth',1.5);
plot3(pos_raw(1,:),  pos_raw(2,:),  -pos_raw(3,:),   'r:',  'LineWidth',1.2);
legend('True','EKF Estimate','Dead Reckoning','Location','best');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Altitude (m)');
title('3-D UAV Trajectory'); grid on; view(35,25);

%-- Top-down (XY plane)
subplot(1,2,2);
plot(pos_true(1,:), pos_true(2,:), 'k-',  'LineWidth',2); hold on;
plot(pos_ekf(1,:),  pos_ekf(2,:),  'b--', 'LineWidth',1.5);
plot(pos_raw(1,:),  pos_raw(2,:),  'r:',  'LineWidth',1.2);
plot(pos_true(1,1), pos_true(2,1), 'go', 'MarkerSize',10,'MarkerFaceColor','g'); % start
plot(pos_true(1,end),pos_true(2,end),'rs','MarkerSize',10,'MarkerFaceColor','r'); % end
legend('True','EKF','Dead Reckoning','Start','End','Location','best');
xlabel('X (m)'); ylabel('Y (m)');
title('Top-Down View (XY Plane)'); grid on; axis equal;

saveas(gcf, fullfile('results','trajectory_comparison.png'));
end

% =========================================================
function plot_errors(t, pos_true, pos_ekf, pos_raw, vel_true, vel_ekf)
% PLOT_ERRORS  Position and velocity error time series.

pos_err_ekf = sqrt(sum((pos_ekf - pos_true).^2, 1));
pos_err_raw = sqrt(sum((pos_raw - pos_true).^2, 1));
vel_err_ekf = sqrt(sum((vel_ekf - vel_true).^2, 1));

figure('Name','Error Analysis','Color','w','Position',[50 600 1200 400]);

subplot(1,3,1);
plot(t, pos_err_ekf, 'b', 'LineWidth',1.5); hold on;
plot(t, pos_err_raw, 'r--', 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Position Error (m)');
title('3-D Position Error'); legend('EKF','Dead Reckoning'); grid on;

subplot(1,3,2);
labels = {'X','Y','Z'};
colors = {'b','g','r'};
for ax = 1:3
    plot(t, pos_ekf(ax,:)-pos_true(ax,:), colors{ax}, 'LineWidth',1.2); hold on;
end
xlabel('Time (s)'); ylabel('Error (m)');
title('Per-Axis Position Error (EKF)'); legend(labels,'Location','best'); grid on;

subplot(1,3,3);
plot(t, vel_err_ekf, 'm', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Velocity Error (m/s)');
title('3-D Velocity Error (EKF)'); grid on;

saveas(gcf, fullfile('results','error_analysis.png'));
end

% =========================================================
function plot_attitude(t, euler_true, euler_ekf)
% PLOT_ATTITUDE  Roll / Pitch / Yaw comparison.

labels = {'Roll (deg)','Pitch (deg)','Yaw (deg)'};
r2d = 180/pi;

figure('Name','Attitude Estimation','Color','w','Position',[50 200 1200 350]);
for ax = 1:3
    subplot(1,3,ax);
    plot(t, euler_true(ax,:)*r2d, 'k-',  'LineWidth',2); hold on;
    plot(t, euler_ekf(ax,:)*r2d,  'b--', 'LineWidth',1.5);
    xlabel('Time (s)'); ylabel(labels{ax});
    title(labels{ax}); legend('True','EKF'); grid on;
end
saveas(gcf, fullfile('results','attitude_estimation.png'));
end
