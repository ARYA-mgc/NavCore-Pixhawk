import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np

from fusion.mag_cal import MagAutoCalibrator
from fusion.trn import TerrainRelativeNavigation, DEMTile

class TestPhase3:
    
    def test_mag_auto_calibration_rls(self):
        """Test the Recursive Least Squares Magnetometer Hard/Soft Iron Calibrator."""
        cal = MagAutoCalibrator(forgetting_factor=1.0) # 1.0 for perfect static memory
        
        # True distortion parameters
        true_bias = np.array([0.5, -0.2, 0.1])
        # Soft iron scale (e.g., elliptical distortion)
        true_scale = np.array([1.2, 0.8, 1.0])
        
        # Generate 200 points on a unit sphere, then distort them
        np.random.seed(42)
        for _ in range(500):
            # Random unit vector
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            
            # Apply hard iron (bias) and soft iron (scale)
            # m_raw = W_inv * m_true + b
            # If true sphere is m_true^T m_true = 1
            # Then we can just generate a point, scale it, and offset it
            distorted = (v / true_scale) + true_bias
            
            cal.update(distorted[0], distorted[1], distorted[2])
            
        # Manually force extraction to be safe
        cal._extract_parameters()
        
        assert cal.calibrated == True
        
        # The estimated bias should match true_bias
        # (Within some tolerance since 500 samples might not perfectly span the sphere)
        assert np.linalg.norm(cal.bias - true_bias) < 0.1
        
        # Test a point application
        test_v = np.array([1.0, 0.0, 0.0])
        distorted_test = (test_v / true_scale) + true_bias
        
        calibrated_test = cal.apply(distorted_test[0], distorted_test[1], distorted_test[2])
        
        # The calibration should recover the original test_v roughly
        # Note: the norm might be arbitrarily scaled if the RLS fits a different overall magnitude,
        # but the direction and relative shape must be restored.
        # Let's check if the calibrated points form a sphere (constant norm)
        norms = []
        for _ in range(50):
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            distorted = (v / true_scale) + true_bias
            cal_pt = cal.apply(distorted[0], distorted[1], distorted[2])
            norms.append(np.linalg.norm(cal_pt))
            
        norms = np.array(norms)
        # Variance of norms should be very low (it is a sphere now)
        assert np.var(norms) < 0.01

    def test_terrain_relative_navigation(self):
        """Test Livox-to-DEM cross correlation matching."""
        
        # 1. Create a synthetic DEM (100x100, 2m resolution)
        # We put a "hill" in the middle
        dem_heights = np.zeros((100, 100))
        for r in range(100):
            for c in range(100):
                # Hill centered at (50, 50)
                dist_sq = (r - 50)**2 + (c - 50)**2
                dem_heights[r, c] = 50.0 * np.exp(-dist_sq / 100.0)
                
        dem = DEMTile(0.0, 0.0, dem_heights, resolution=2.0)
        
        trn = TerrainRelativeNavigation(enable=True)
        # Force map resolution to match DEM for simpler testing
        trn.MAP_RESOLUTION = 2.0
        trn.SEARCH_RADIUS = 20.0
        trn.load_dem(dem)
        
        # 2. Simulate a downward lidar scan
        # The drone is truly at DEM coordinates North=100m, East=100m (which is row 50, col 50)
        # But the ESKF *thinks* it is at North=110m, East=104m (offset by N:+10m, E:+4m)
        true_pos = np.array([100.0, 100.0, -100.0])
        eskf_pos = np.array([110.0, 104.0, -100.0])
        
        # Generate point cloud around the true position, but in vehicle frame, then add eskf_pos
        points = []
        for dr in range(-10, 11):
            for dc in range(-10, 11):
                dn = dr * 2.0
                de = dc * 2.0
                h = dem.get_height(true_pos[0] + dn, true_pos[1] + de)
                if h is not None:
                    # Point in relative vehicle frame
                    rel_n = dn
                    rel_e = de
                    # Added to eskf_pos to simulate what the drone thinks the absolute coords are
                    points.append([eskf_pos[0] + rel_n, eskf_pos[1] + rel_e, h])
                    
        points = np.array(points)
        
        # 3. Process the scan
        res = trn.process_lidar_scan(points, eskf_pos, current_alt_agl=100.0, t_now=5.0)
        
        # 4. Verify results
        assert res is not None
        assert res["type"] == "TRN"
        
        # The offset should be approx -10m North, -4m East to correct the ESKF position
        offset = res["offset_ne"]
        
        assert abs(offset[0] - (-10.0)) <= 2.0  # within one cell resolution
        assert abs(offset[1] - (-4.0)) <= 2.0
        assert res["correlation"] > 0.8
