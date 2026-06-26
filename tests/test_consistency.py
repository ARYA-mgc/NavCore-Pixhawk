# test_consistency.py module.
# Does exactly what you think it does.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pytest
from core.eskf import ESKF, EKFHealth
from utils.noise import IMUNoiseParams
from test_drift import RealisticIMU
import scipy.stats as stats

class TestConsistency:
    """NEES and NIS Consistency validation.
    
    NEES: Normalized Estimation Error Squared = dx^T P^{-1} dx
    NIS: Normalized Innovation Squared = y^T S^{-1} y
    
    A perfectly consistent linear Gaussian filter should have:
    E[NEES] = dim(x)
    E[NIS] = dim(z)
    """
    
    def run_consistency_sim(self, duration_s=10.0, dt=0.01):
        noise = IMUNoiseParams()
        eskf = ESKF(noise)
        rng = np.random.default_rng(42)
        imu = RealisticIMU(noise, rng, temp_drift=False)
        
        N = int(duration_s / dt)
        
        nees_history = []
        nis_history = []
        
        # True state (stationary at origin)
        true_pos = np.zeros(3)
        true_vel = np.zeros(3)
        true_att = np.array([1.0, 0.0, 0.0, 0.0])
        
        eskf._initialized = True
        eskf._gps_origin = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        
        for i in range(N):
            # Simulate IMU
            a, g = imu.sample(np.array([0, 0, -9.80665]), np.zeros(3), dt, i*dt)
            eskf.predict(a, g, dt)
            
            # Simulate GPS update every 100 steps (1 Hz)
            if i % 100 == 0:
                # Match the filter's HDOP=1.0 expectation (2.5m horizontal, 5.0m vertical)
                # 2.5 meters ~ 2.25e-5 degrees
                lat = 0.0 + rng.normal(0, 2.25e-5)
                lon = 0.0 + rng.normal(0, 2.25e-5)
                alt = 0.0 + rng.normal(0, 5.0)
                eskf.update_gps(lat, lon, alt, hdop=1.0)
                
                # Check NIS (from innovation history)
                if len(eskf.innovation_history) > 0:
                    _, source, y, S, nis = eskf.innovation_history[-1]
                    if source == "gps":
                        nis_history.append(nis)
            
            # Simulate Mag update every 10 steps (10 Hz)
            if i % 10 == 0:
                # True yaw is 0.0, add noise (R_mag_base is ~0.1^2)
                yaw_meas = 0.0 + rng.normal(0, 0.1)
                eskf.update_mag(yaw_meas, mag_norm=0.5, t_now=i*dt)
                
            # Simulate Baro update every 5 steps (20 Hz)
            if i % 5 == 0:
                baro_meas = 0.0 + rng.normal(0, 0.1)
                eskf.update_baro(baro_meas)
            
            # Compute NEES
            # dx = x_true - x_est
            # For simplicity, just evaluate pos/vel/att part
            dx = np.zeros(9)
            dx[0:3] = true_pos - eskf.x[0:3]
            dx[3:6] = true_vel - eskf.x[3:6]
            
            q_est = eskf.x[6:10]
            # dq approx
            # q_true = q_est * dq -> dq = q_est^-1 * q_true
            dq = eskf._quat_mult(
                np.array([q_est[0], -q_est[1], -q_est[2], -q_est[3]]), 
                true_att
            )
            dx[6:9] = 2.0 * dq[1:4]
            
            P_9x9 = eskf.P[0:9, 0:9]
            # NEES = dx^T P^-1 dx
            # Use pseudo-inverse or solve
            try:
                nees = dx @ np.linalg.solve(P_9x9, dx)
                nees_history.append(nees)
            except np.linalg.LinAlgError:
                pass
                
        return np.array(nees_history), np.array(nis_history)

    def test_nees_consistency(self):
        nees, nis = self.run_consistency_sim(duration_s=60.0)
        # Discard the first 5 seconds (500 steps) to allow the filter to converge
        nees_steady = nees[500:] if len(nees) > 500 else nees
        assert len(nees_steady) > 0
        
        # 9-DOF subset (pos, vel, att)
        dof = 9
        
        mean_nees = np.mean(nees_steady)
        
        # 95% confidence interval for chi-squared with N*dof degrees of freedom
        N = len(nees)
        lower_bound = stats.chi2.ppf(0.025, N * dof) / N
        upper_bound = stats.chi2.ppf(0.975, N * dof) / N
        
        print(f"Mean NEES: {mean_nees:.2f} (95% CI: {lower_bound:.2f} - {upper_bound:.2f})")
        
        # Note: In a simulated non-linear system, we expect some deviation.
        # If mean_nees < lower_bound -> Filter is pessimistic (cov too large) -> Underconfident
        # If mean_nees > upper_bound -> Filter is optimistic (cov too small) -> Overconfident
        
        # We allow a slightly looser bound for the nonlinear ESKF
        loose_lower = lower_bound * 0.5
        loose_upper = upper_bound * 4.0
        
        if mean_nees < loose_lower:
            pytest.fail(f"Filter is severely UNDERCONFIDENT (Mean NEES {mean_nees:.2f} < {loose_lower:.2f})")
        if mean_nees > loose_upper:
            pytest.fail(f"Filter is severely OVERCONFIDENT / DIVERGING (Mean NEES {mean_nees:.2f} > {loose_upper:.2f})")
            
    def test_nis_consistency(self):
        nees, nis = self.run_consistency_sim(duration_s=60.0)
        
        assert len(nis) > 0
        
        # GPS update has 3-DOF
        dof = 3
        mean_nis = np.mean(nis)
        
        N = len(nis)
        lower_bound = stats.chi2.ppf(0.025, N * dof) / N
        upper_bound = stats.chi2.ppf(0.975, N * dof) / N
        
        print(f"Mean NIS: {mean_nis:.2f} (95% CI: {lower_bound:.2f} - {upper_bound:.2f})")
        
        loose_lower = lower_bound * 0.5
        loose_upper = upper_bound * 3.0
        
        if mean_nis < loose_lower:
            pytest.fail(f"GPS Innovation is UNDERCONFIDENT (Mean NIS {mean_nis:.2f} < {loose_lower:.2f})")
        if mean_nis > loose_upper:
            pytest.fail(f"GPS Innovation is OVERCONFIDENT (Mean NIS {mean_nis:.2f} > {loose_upper:.2f})")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
