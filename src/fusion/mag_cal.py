#!/usr/bin/env python3
# mag_cal.py module.
# Does exactly what you think it does.

import numpy as np
import scipy.linalg as la
import logging

log = logging.getLogger("mag_cal")

class MagAutoCalibrator:
    """
    Online Magnetometer Calibration using Recursive Least Squares.
    Fits raw magnetometer readings to an ellipsoid:
    (m - b)^T A (m - b) = R^2
    """
    
    def __init__(self, forgetting_factor=0.999):
        # State vector theta: [a, b, c, d, e, f, g, h, i]^T
        # representing a x^2 + b y^2 + c z^2 + 2d xy + 2e xz + 2f yz + 2g x + 2h y + 2i z = 1
        self.theta = np.zeros(9)
        # Initialize with small spherical assumption
        self.theta[0] = 1.0
        self.theta[1] = 1.0
        self.theta[2] = 1.0
        
        # Covariance matrix P
        self.P = np.eye(9) * 1e4
        self.lam = forgetting_factor
        
        self.bias = np.zeros(3)
        self.W = np.eye(3)
        self.calibrated = False
        self.samples = 0
        
    def update(self, mx: float, my: float, mz: float):
        """Feed a raw 3D magnetometer vector to update the RLS estimator."""
        # Regressor
        phi = np.array([
            mx**2, my**2, mz**2, 
            2*mx*my, 2*mx*mz, 2*my*mz, 
            2*mx, 2*my, 2*mz
        ])
        
        # RLS update
        err = 1.0 - np.dot(phi, self.theta)
        
        # Kalman gain
        P_phi = self.P @ phi
        S = self.lam + np.dot(phi, P_phi)
        K = P_phi / S
        
        # Update theta
        self.theta += K * err
        
        # Update P (Joseph form-like or standard rank-1 update)
        self.P = (self.P - np.outer(K, P_phi)) / self.lam
        
        self.samples += 1
        
        # Periodically extract physical parameters to save CPU
        if self.samples > 100 and self.samples % 20 == 0:
            self._extract_parameters()
            
    def _extract_parameters(self):
        """Extract Bias and Soft-Iron matrix from algebraic parameters."""
        try:
            a, b, c, d, e, f, g, h, i = self.theta
            
            # Quadratic form matrix
            A = np.array([
                [a, d, e],
                [d, b, f],
                [e, f, c]
            ])
            
            # Linear form vector
            v = np.array([g, h, i])
            
            # Hard-iron Bias is -A^-1 v
            self.bias = -np.linalg.inv(A) @ v
            
            # Offset constant
            offset = np.dot(v, self.bias) + 1.0
            
            # Check for degenerate ellipsoid
            if offset <= 0:
                return
                
            # Normalized matrix
            A_norm = A / offset
            
            # Eigen decomposition to find matrix square root
            evals, evecs = np.linalg.eigh(A_norm)
            
            if np.any(evals <= 0):
                # Not a valid positive-definite ellipsoid yet
                return
                
            # Soft iron correction matrix W = sqrt(A_norm)
            D = np.diag(np.sqrt(evals))
            self.W = evecs @ D @ evecs.T
            
            self.calibrated = True
            
        except np.linalg.LinAlgError:
            pass
            
    def apply(self, mx: float, my: float, mz: float) -> np.ndarray:
        """Apply the live calibration to a raw reading. Returns calibrated 3D vector.

        Preserves the original field magnitude so that downstream norm-based
        rejection (ESKF Tier 1 mag check) continues to work correctly.
        The W matrix maps to a unit sphere; we rescale back to the raw norm.
        """
        raw = np.array([mx, my, mz])
        if not self.calibrated:
            return raw
        corrected = self.W @ (raw - self.bias)
        # Preserve original field magnitude: rescale corrected (unit-sphere)
        # vector to match the raw input's magnitude.
        corrected_norm = np.linalg.norm(corrected)
        raw_norm = np.linalg.norm(raw)
        if corrected_norm > 1e-10:
            corrected = corrected * (raw_norm / corrected_norm)
        return corrected

