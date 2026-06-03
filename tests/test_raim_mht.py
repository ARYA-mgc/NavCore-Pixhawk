import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from core.eskf import ESKF, EKFHealth
from core.mht import MHTManager
from utils.noise import IMUNoiseParams

class TestRAIMandMHT:
    
    def test_raim_rejection(self):
        """Test that a single massive GPS jump is rejected by RAIM."""
        noise = IMUNoiseParams()
        eskf = ESKF(noise)
        eskf._initialized = True
        eskf._gps_origin = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        
        mht = MHTManager(eskf)
        
        # Feed 10 good GPS updates to lower covariance
        t = 0.0
        for _ in range(10):
            # Perfect GPS at origin
            mht.update_gps(0.0, 0.0, 0.0, hdop=1.0, t_now=t)
            t += 0.1
            
        assert eskf._sensor_rejections.get("gps", 0) == 0
        
        # Inject massive multipath spike (111 meters)
        mht.update_gps(0.001, 0.0, 0.0, hdop=1.0, t_now=t)
        
        # Verify RAIM rejected it
        assert eskf._sensor_rejections.get("gps", 0) == 1
        
        # Verify state didn't jump
        assert np.linalg.norm(eskf.x[0:3]) < 5.0
        
    def test_mht_shadow_spawn_and_swap(self):
        """Test that a permanent GPS jump spawns a shadow filter and eventually swaps."""
        noise = IMUNoiseParams()
        eskf = ESKF(noise)
        eskf._initialized = True
        eskf._gps_origin = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        
        mht = MHTManager(eskf)
        
        # 1. Converge at origin
        t = 0.0
        for _ in range(20):
            mht.update_gps(0.0, 0.0, 0.0, hdop=1.0, t_now=t)
            t += 0.1
            
        assert mht.shadow is None
        
        # 2. Inject permanent GPS shift (e.g. 50 meters North)
        lat_jump = 50.0 / 111320.0
        
        # First 2 updates should just be rejected
        for _ in range(2):
            mht.update_gps(lat_jump, 0.0, 0.0, hdop=1.0, t_now=t)
            t += 0.1
            
        assert mht.shadow is None
        assert mht.primary._sensor_rejections["gps"] == 2
        
        # 3rd update should spawn shadow filter
        mht.update_gps(lat_jump, 0.0, 0.0, hdop=1.0, t_now=t)
        t += 0.1
        
        assert mht.shadow is not None
        assert mht.primary._sensor_rejections["gps"] == 3
        
        # 3. Keep feeding jumped GPS until evaluation window passes
        # MHT_EVALUATION_WINDOW is 3.0 seconds
        for _ in range(35):
            mht.predict(np.array([0,0,-9.8]), np.zeros(3), 0.1)
            mht.update_gps(lat_jump, 0.0, 0.0, hdop=1.0, t_now=t)
            t += 0.1
            
        # 4. Shadow should have swapped to primary!
        assert mht.shadow is None
        
        # The new primary should have absorbed the 50m jump
        assert abs(mht.primary.x[0] - 50.0) < 5.0
