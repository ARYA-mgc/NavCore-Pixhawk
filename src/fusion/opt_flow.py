#!/usr/bin/env python3
# Looking at the ground to figure out how fast we're moving. Literally just staring at dirt.

import math
import numpy as np

class OpticalFlowINS:
    def __init__(self):
        self.pos = np.zeros(3)  # x, y, z in local frame
        self.vel = np.zeros(3)  # vx, vy, vz in local frame
        self.last_time_us = 0
        self.quality_threshold = 10  # Minimum quality to trust flow

    def update(self, flow_msg, current_yaw_rad: float):
        # Updates velocity and position from OPTICAL_FLOW_RAD message.
        if flow_msg.quality < self.quality_threshold or flow_msg.distance <= 0:
            return  # Cannot use this reading
            
        dt_s = flow_msg.integration_time_us / 1e6
        if dt_s <= 0:
            return

        # Body frame flow (compensated for rotation)
        # flow is in radians. integrated_x is flow around X axis (which is pitch, so Y movement)
        # integrated_y is flow around Y axis (roll, X movement)
        # However, MAVLink standard:
        # integrated_x: Flow in radians around X axis (Pitch)
        # integrated_y: Flow in radians around Y axis (Roll)
        # Body velocity:
        # v_x = (integrated_x - integrated_xgyro) * distance / dt
        # v_y = (integrated_y - integrated_ygyro) * distance / dt
        
        flow_x_rad = flow_msg.integrated_x
        flow_y_rad = flow_msg.integrated_y
        gyro_x_rad = flow_msg.integrated_xgyro
        gyro_y_rad = flow_msg.integrated_ygyro
        
        # Calculate velocity in body frame (m/s)
        v_body_x = (flow_x_rad - gyro_x_rad) * flow_msg.distance / dt_s
        v_body_y = (flow_y_rad - gyro_y_rad) * flow_msg.distance / dt_s
        
        # Rotate to local NED frame using yaw
        c_yaw = math.cos(current_yaw_rad)
        s_yaw = math.sin(current_yaw_rad)
        
        v_ned_x = v_body_x * c_yaw - v_body_y * s_yaw
        v_ned_y = v_body_x * s_yaw + v_body_y * c_yaw
        
        self.vel[0] = v_ned_x
        self.vel[1] = v_ned_y
        
        # Integrate for position
        self.pos[0] += v_ned_x * dt_s
        self.pos[1] += v_ned_y * dt_s

    def get_state(self):
        # Returns position and velocity estimates
        return self.pos.copy(), self.vel.copy()
