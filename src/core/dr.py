#!/usr/bin/env python3
# Dead reckoning fallback.
# When sensors fail, we just integrate and pray.

import math
import numpy as np
from utils.noise import IMUNoiseParams

GRAVITY = np.array([0.0, 0.0, 9.80665])


class DeadReckon:
    # Performs pure inertial integration during sensor loss.

    def __init__(self, noise: IMUNoiseParams):
        self.pos   = np.zeros(3)
        self.vel   = np.zeros(3)
        self.euler = np.zeros(3)   # phi, theta, psi

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        phi, theta, psi = self.euler

        # Rotation matrix body → NED
        cp, sp = math.cos(phi),   math.sin(phi)
        ct, st = math.cos(theta), math.sin(theta)
        cy, sy = math.cos(psi),   math.sin(psi)

        R = np.array([
            [ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy],
            [ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy],
            [-st,    sp*ct,             cp*ct            ],
        ])

        # Velocity & position
        acc_ned   = R @ accel + GRAVITY
        self.vel += acc_ned * dt
        self.pos += self.vel * dt

        # Attitude
        if abs(ct) < 1e-6:
            ct = 1e-6
        T_inv = np.array([
            [1.0, sp*st/ct, cp*st/ct],
            [0.0, cp,      -sp      ],
            [0.0, sp/ct,   cp/ct   ],
        ])
        eta_dot    = T_inv @ gyro
        self.euler += eta_dot * dt
        self.euler  = np.array([math.atan2(math.sin(a), math.cos(a))
                                 for a in self.euler])

    def reset(self):
        self.pos   = np.zeros(3)
        self.vel   = np.zeros(3)
        self.euler = np.zeros(3)
