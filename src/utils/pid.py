#!/usr/bin/env python3
"""
adaptive_pid.py
================
Adaptive PID Controller for drone navigation and control.
Adjusts P, I, D gains dynamically based on error magnitude.
"""

import math

class AdaptivePID:
    def __init__(self, kp_base: float, ki_base: float, kd_base: float, 
                 k_adapt: float = 0.1, integral_limit: float = 1.0):
        self.kp_base = kp_base
        self.ki_base = ki_base
        self.kd_base = kd_base
        self.k_adapt = k_adapt
        
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0
        
        # Current adapted gains
        self.kp = kp_base
        self.ki = ki_base
        self.kd = kd_base

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        # Adaptive logic: increase P gain if error is large, decrease if small
        # This is a simplified gain scheduling approach
        abs_error = abs(error)
        self.kp = self.kp_base + self.k_adapt * abs_error
        self.ki = self.ki_base # Integral typically stays constant or adapts slowly
        self.kd = self.kd_base + (self.k_adapt * 0.1) * abs_error
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        # Anti-windup
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        self.last_error = error
        
        return p_term + i_term + d_term

    def get_gains(self) -> tuple:
        """Returns the current adapted gains (Kp, Ki, Kd)"""
        return self.kp, self.ki, self.kd

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
