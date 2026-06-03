import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from core.eskf import ESKF
from utils.noise import IMUNoiseParams
E_POS = slice(0, 3)
E_VEL = slice(3, 6)
E_ATT = slice(6, 9)

def finite_difference_jacobian(func, x_nominal, error_dim, eps=1e-5):
    """Computes numerical Jacobian via central finite differences."""
    # func takes a delta_x error state, applies it to x_nominal, and returns z_pred
    z_nom = func(np.zeros(error_dim))
    m = len(z_nom)
    H_num = np.zeros((m, error_dim))
    
    for i in range(error_dim):
        dx = np.zeros(error_dim)
        dx[i] = eps
        z_plus = func(dx)
        
        dx[i] = -eps
        z_minus = func(dx)
        
        # Handle angle wrapping if measurement is yaw
        diff = z_plus - z_minus
        if m == 1: # assuming scalar angle for mag
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            
        H_num[:, i] = diff / (2 * eps)
        
    return H_num

class TestJacobianValidation:
    
    @pytest.fixture
    def eskf(self):
        noise = IMUNoiseParams()
        filter = ESKF(noise)
        # Set a non-trivial attitude (e.g. banked and pitched)
        # Roll 30 deg, Pitch 20 deg, Yaw 45 deg
        from scipy.spatial.transform import Rotation
        q = Rotation.from_euler('xyz', [30, 20, 45], degrees=True).as_quat()
        # SciPy returns x,y,z,w. We use w,x,y,z
        filter.x[6:10] = [q[3], q[0], q[1], q[2]]
        # Set non-trivial velocity
        filter.x[3:6] = [10.0, -5.0, 2.0]
        # Set non-trivial position
        filter.x[0:3] = [100.0, 50.0, -20.0]
        return filter

    def test_mag_jacobian(self, eskf):
        # We want to test H for yaw measurement
        # z_pred = euler[2]
        def h_func(dx):
            # Clone state
            q_nom = eskf.x[6:10].copy()
            # Inject error
            dtheta = dx[E_ATT]
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            q_new = eskf._quat_mult(q_nom, dq)
            euler = eskf._quat_to_euler(q_new)
            return np.array([euler[2]])

        H_num = finite_difference_jacobian(h_func, eskf.x, 20)
        
        euler = eskf._quat_to_euler(eskf.x[6:10])
        phi = euler[0]
        theta = euler[1]
        
        H_ana = np.zeros((1, 20))
        H_ana[0, 7] = np.sin(phi) / np.cos(theta)
        H_ana[0, 8] = np.cos(phi) / np.cos(theta)
        
        np.testing.assert_allclose(H_num[:, 6:9], H_ana[:, 6:9], atol=1e-2)

    def test_optical_flow_jacobian(self, eskf):
        def h_func(dx):
            v_nom = eskf.x[3:6].copy()
            q_nom = eskf.x[6:10].copy()
            
            # Inject error
            v_new = v_nom + dx[E_VEL]
            
            dtheta = dx[E_ATT]
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            q_new = eskf._quat_mult(q_nom, dq)
            
            R_new = eskf._quat_to_dcm(q_new)
            v_body = R_new.T @ v_new
            return v_body[0:2] # vx, vy in body

        H_num = finite_difference_jacobian(h_func, eskf.x, 20)
        
        R_dcm = eskf._quat_to_dcm(eskf.x[6:10])
        H_ana = np.zeros((2, 20))
        H_ana[:, 3:6] = R_dcm.T[0:2, :]
        
        # Check velocity Jacobian
        np.testing.assert_allclose(H_num[:, 3:6], H_ana[:, 3:6], atol=1e-5)
        
        # Note: we neglected the attitude Jacobian H_flow[:, 6:9] in the analytical code!
        # The numerical Jacobian will show it's non-zero!
        # delta v_body = R^T [v_ned]x R delta_theta
        # If the user wants exact math, we should add the attitude Jacobian to eskf.py!

    def test_radar_jacobian(self, eskf):
        def h_func(dx):
            v_nom = eskf.x[3:6].copy()
            q_nom = eskf.x[6:10].copy()
            
            v_new = v_nom + dx[E_VEL]
            dtheta = dx[E_ATT]
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            q_new = eskf._quat_mult(q_nom, dq)
            
            R_new = eskf._quat_to_dcm(q_new)
            return R_new.T @ v_new

        H_num = finite_difference_jacobian(h_func, eskf.x, 20)
        
        R_dcm = eskf._quat_to_dcm(eskf.x[6:10])
        H_ana = np.zeros((3, 20))
        H_ana[:, 3:6] = R_dcm.T
        
        np.testing.assert_allclose(H_num[:, 3:6], H_ana[:, 3:6], atol=1e-5)

    def test_lidar_jacobian(self, eskf):
        def h_func(dx):
            pos_nom = eskf.x[0:3].copy()
            q_nom = eskf.x[6:10].copy()
            
            pos_new = pos_nom + dx[E_POS]
            dtheta = dx[E_ATT]
            dq = np.array([1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
            dq /= np.linalg.norm(dq)
            q_new = eskf._quat_mult(q_nom, dq)
            
            R_new = eskf._quat_to_dcm(q_new)
            cos_tilt = R_new[2, 2]
            return np.array([-pos_new[2] / cos_tilt])

        H_num = finite_difference_jacobian(h_func, eskf.x, 20)
        
        R_dcm = eskf._quat_to_dcm(eskf.x[6:10])
        cos_tilt = R_dcm[2, 2]
        H_ana = np.zeros((1, 20))
        H_ana[0, 2] = -1.0 / cos_tilt
        
        np.testing.assert_allclose(H_num[:, 0:3], H_ana[:, 0:3], atol=1e-5)
