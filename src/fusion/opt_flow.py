#!/usr/bin/env python3
# Optical flow.
# Highly Advanced Pipeline with Terrain KF and Median Filtering

import math
import numpy as np


class TerrainKalmanFilter:
    """1D Kalman Filter to estimate terrain distance dynamically."""
    def __init__(self):
        self.d = 0.0      # Estimated distance
        self.P = 1.0      # Covariance
        self.Q = 0.5      # Process noise (how fast terrain changes)
        self.R = 0.5      # Measurement noise (rangefinder noise)
        self.initialized = False

    def predict(self, vz: float, dt: float):
        """Predict new distance based on vehicle vertical velocity."""
        if not self.initialized:
            return
        # If moving down (vz > 0 in NED), distance to ground decreases
        self.d = self.d - vz * dt
        self.P = self.P + self.Q * dt

    def update(self, z: float):
        """Update distance estimate from rangefinder."""
        if not self.initialized:
            self.d = z
            self.P = 1.0
            self.initialized = True
            return

        # Innovation
        y = z - self.d
        # Innovation covariance
        S = self.P + self.R
        # Kalman gain
        K = self.P / S
        
        self.d = self.d + K * y
        self.P = (1.0 - K) * self.P

    def get_distance(self):
        return self.d


class MedianFilter:
    """Sliding window median filter for outlier rejection."""
    def __init__(self, size=3):
        self.size = size
        self.buffer_x = []
        self.buffer_y = []

    def update(self, x: float, y: float):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_x) > self.size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)
        
        return np.median(self.buffer_x), np.median(self.buffer_y)


class OpticalFlowINS:
    def __init__(self):
        self.pos = np.zeros(3)  # x, y, z in local frame
        self.vel = np.zeros(3)  # vx, vy, vz in local frame
        self.last_time_us = 0
        self.quality_threshold = 10
        
        # Advanced filters
        self.terrain_kf = TerrainKalmanFilter()
        self.median_filter = MedianFilter(size=5)

        # Calibration parameters
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # Camera mounting offset (Lever Arm) relative to CG [x, y, z] in body frame
        self.r_mount = np.zeros(3)

    def update(self, flow_msg, current_yaw_rad: float, vz: float = 0.0, omega: np.ndarray = np.zeros(3)):
        # Standalone update for logging/dead-reckoning
        v_body_x, v_body_y = self.process_flow_for_eskf(flow_msg, use_raw_flow=True, vz=vz, omega=omega)
        
        if v_body_x is None or v_body_y is None:
            return

        dt_s = flow_msg.integration_time_us / 1e6
        
        c_yaw = math.cos(current_yaw_rad)
        s_yaw = math.sin(current_yaw_rad)

        v_ned_x = v_body_x * c_yaw - v_body_y * s_yaw
        v_ned_y = v_body_x * s_yaw + v_body_y * c_yaw

        self.vel[0] = v_ned_x
        self.vel[1] = v_ned_y

        self.pos[0] += v_ned_x * dt_s
        self.pos[1] += v_ned_y * dt_s

    def get_state(self):
        # Returns position and velocity estimates
        return self.pos.copy(), self.vel.copy()

    def process_flow_for_eskf(self, flow_msg, use_raw_flow=True, vz: float = 0.0, omega: np.ndarray = np.zeros(3)):
        """
        Pre-processes the optical flow message with advanced terrain KF, median filtering,
        and lever arm compensation to return robust body velocities.
        """
        if flow_msg.quality < self.quality_threshold or flow_msg.distance <= 0:
            return None, None
            
        dt_s = flow_msg.integration_time_us / 1e6
        if dt_s <= 0:
            return None, None
            
        # 1. Terrain Kalman Filter prediction and update
        self.terrain_kf.predict(vz, dt_s)
        self.terrain_kf.update(flow_msg.distance)
        distance = flow_msg.distance  # BYPASS TERRAIN KF to avoid vz feedback loop

        # 2. Extract flow rates
        if use_raw_flow:
            flow_x = flow_msg.integrated_x - flow_msg.integrated_xgyro
            flow_y = flow_msg.integrated_y - flow_msg.integrated_ygyro
        else:
            # Assume FC already compensated it
            flow_x = flow_msg.integrated_x
            flow_y = flow_msg.integrated_y
            
        flow_rate_x = flow_x / dt_s
        flow_rate_y = flow_y / dt_s

        # 3. Sliding Median Filter for outlier rejection
        filtered_rate_x, filtered_rate_y = self.median_filter.update(flow_rate_x, flow_rate_y)

        # 4. Body velocity mapping (correct axis mapping)
        v_cam_x = filtered_rate_y * distance * self.scale_x
        v_cam_y = -filtered_rate_x * distance * self.scale_y
        
        # 5. Lever Arm (Camera Offset) Compensation
        # v_cam = v_cg + omega x r_mount
        # v_cg = v_cam - omega x r_mount
        v_rot = np.cross(omega, self.r_mount)
        v_body_x = v_cam_x - v_rot[0]
        v_body_y = v_cam_y - v_rot[1]
        
        return v_body_x, v_body_y
