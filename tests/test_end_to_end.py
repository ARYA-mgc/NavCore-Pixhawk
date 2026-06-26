# test_end_to_end.py module.
# Does exactly what you think it does.

import sys
import os
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock

# Mock pymavlink before importing m.py
import sys
sys.modules['pymavlink'] = MagicMock()
sys.modules['pymavlink.mavutil'] = MagicMock()
sys.modules['pymavlink.dialects'] = MagicMock()
sys.modules['pymavlink.dialects.v20'] = MagicMock()
sys.modules['pymavlink.dialects.v20.ardupilotmega'] = MagicMock()

from core.m import INSNavSys

class MockMsg:
    def __init__(self, mtype, **kwargs):
        self._type = mtype
        self.__dict__.update(kwargs)
    def get_type(self):
        return self._type

class MockConnection:
    def __init__(self):
        self.messages = []
        self.mav = MagicMock()
        self.target_system = 1
        self.target_component = 1

    def mode_mapping(self):
        return {"GUIDED": 4, "LOITER": 5, "RTL": 6}

    def wait_heartbeat(self):
        return MockMsg("HEARTBEAT", type=2)

def test_end_to_end_oosm():
    """Test the full INS system pipeline with synthetic MAVLink messages."""
    
    # Init system with a dummy connection string, but we won't run its thread
    nav = INSNavSys("udp:127.0.0.1:14550", baud=115200, update_hz=100)
    
    # Replace the MAVLink connection with our mock
    nav.mlog = MockConnection()
    nav.bridge._conn = MockConnection()
    nav.bridge._conn.mav = MagicMock()
    nav.bridge.send_statustext = MagicMock()
    
    # We will manually dispatch messages
    
    dt = 0.01
    t_now = 1000.0
    
    # 1. Provide INIT_SAMPLES of IMU and MAG to initialize the filter
    print("Initializing...")
    for i in range(150):
        # IMU perfectly flat, stationary
        msg_imu = MockMsg("RAW_IMU", time_usec=int(t_now*1e6),
                          xacc=0, yacc=0, zacc=-2048, # mg
                          xgyro=0, ygyro=0, zgyro=0)
        nav._dispatch_message("RAW_IMU", msg_imu, t_now)
        
        # Mag pointing North
        msg_mag = MockMsg("SCALED_IMU3", time_usec=int(t_now*1e6),
                          xmag=300, ymag=0, zmag=400)
        nav._dispatch_message("SCALED_IMU3", msg_mag, t_now)
        
        t_now += dt
        
    assert nav.eskf._initialized, "ESKF should be initialized after 100 samples"
    
    # 2. Add some motion and verify state updates
    print("Running motion...")
    
    for i in range(100):
        # ~1 m/s^2 forward (RAW_IMU LSB: 2048 ≈ 1 g → 209 LSB ≈ 1 m/s^2)
        msg_imu = MockMsg("RAW_IMU", time_usec=int(t_now*1e6),
                          xacc=209, yacc=0, zacc=-2048,
                          xgyro=0, ygyro=0, zgyro=0)
        nav._dispatch_message("RAW_IMU", msg_imu, t_now)
        if i % 4 == 0:
            nav.mht.update_baro(-50.0)
        t_now += dt
        
    # Velocity X should be roughly 1.0 m/s
    vel_x = nav.eskf.x[3]
    assert 0.9 < vel_x < 1.1, f"Velocity X {vel_x} did not integrate correctly"
    
    # 3. Test OOSM GPS Update
    print("Testing OOSM GPS...")

    # Current time is t_now (e.g. 1002.5). 
    # Let's send a GPS measurement from 200ms ago (latency)
    t_gps = t_now - 0.2
    
    # Set the GPS origin if it's not set
    nav.eskf._gps_origin = {"lat": 13.0, "lon": 80.0, "alt": 50.0}
    
    pre_update_pos = nav.eskf.x[0:3].copy()
    pre_update_cov = np.trace(nav.eskf.P[0:3, 0:3])
    
    # 1 degree lat is ~111km, so 1e-5 degrees is ~1.11 meters
    msg_gps = MockMsg("GPS_RAW_INT", time_usec=int(t_gps*1e6),
                      fix_type=3, lat=int((13.0 + 1e-5)*1e7), lon=int(80.0*1e7), alt=int(50.0*1000), eph=100)
    
    nav._dispatch_message("GPS_RAW_INT", msg_gps, t_now)
    
    post_update_pos = nav.eskf.x[0:3]
    post_update_cov = np.trace(nav.eskf.P[0:3, 0:3])
    
    # Covariance should shrink!
    assert post_update_cov < pre_update_cov, "Covariance did not shrink after GPS update"
    
    # Position should jump towards the GPS measurement
    jump = np.linalg.norm(post_update_pos - pre_update_pos)
    assert jump > 0.1, f"Position did not jump after GPS update, jump={jump}"
    
    print("End-to-End Test Passed!")

def test_end_to_end_stress():
    """Stress test the OOSM buffer with high frequency delayed updates to measure memory and computational cost."""
    import time
    
    # 400Hz update rate, 2-second buffer
    nav = INSNavSys("udp:127.0.0.1:14550", baud=115200, update_hz=400)
    nav.mlog = MockConnection()
    nav.bridge._conn = MockConnection()
    nav.bridge._conn.mav = MagicMock()
    nav.bridge.send_statustext = MagicMock()
    
    dt = 1.0 / 400.0
    t_now = 1000.0
    
    # Init
    for i in range(150):
        nav._dispatch_message("RAW_IMU", MockMsg("RAW_IMU", time_usec=int(t_now*1e6), xacc=0, yacc=0, zacc=-2048, xgyro=0, ygyro=0, zgyro=0), t_now)
        nav._dispatch_message("SCALED_IMU3", MockMsg("SCALED_IMU3", time_usec=int(t_now*1e6), xmag=300, ymag=0, zmag=400), t_now)
        t_now += dt
        
    nav.eskf._gps_origin = {"lat": 13.0, "lon": 80.0, "alt": 50.0}
    
    # Run 10 seconds of simulation
    # During this, inject a GPS measurement with a 500ms delay every 50ms (20Hz GPS!)
    # This will force the OOSM to rewind 200 IMU frames, 20 times a second!
    
    start_cpu_time = time.process_time()
    
    for i in range(4000): # 10 seconds at 400Hz
        nav._dispatch_message("RAW_IMU", MockMsg("RAW_IMU", time_usec=int(t_now*1e6), xacc=0, yacc=0, zacc=-2048, xgyro=0, ygyro=0, zgyro=0), t_now)
        if i % 4 == 0:
            nav.mht.update_baro(-50.0)

        # 20Hz GPS with 500ms latency
        if i % 20 == 0 and i > 400:
            t_gps = t_now - 0.5
            msg_gps = MockMsg("GPS_RAW_INT", time_usec=int(t_gps*1e6), fix_type=3, lat=int(13.0*1e7), lon=int(80.0*1e7), alt=int(50.0*1000), eph=100)
            nav._dispatch_message("GPS_RAW_INT", msg_gps, t_now)
            
        t_now += dt
        
    end_cpu_time = time.process_time()
    
    # Ensure memory buffer bounds are respected
    buffer_len = len(nav._oosm_buffer)
    assert buffer_len <= 800, f"OOSM buffer grew beyond maxlen! Size: {buffer_len}"
    
    # Profiling results
    compute_time = end_cpu_time - start_cpu_time
    print(f"\n--- OOSM Stress Test ---")
    print(f"Total CPU Time for 10s flight @ 400Hz with 20Hz 500ms-delayed GPS: {compute_time:.3f} s")
    
    # A Pi4 should easily do this in < 2.0 seconds
    # Our test environment might be different, but we check for catastrophic blowup
    assert compute_time < 5.0, f"OOSM replay is too slow! Took {compute_time:.3f}s for 10s of data."

def test_end_to_end_full_chain():
    """Test full pipeline: IMU fusion, delayed GPS, GPS dropout, and recovery."""
    nav = INSNavSys("udp:127.0.0.1:14550", baud=115200, update_hz=100)
    nav.mlog = MockConnection()
    nav.bridge._conn = MockConnection()
    nav.bridge._conn.mav = MagicMock()
    nav.bridge.send_statustext = MagicMock()
    
    dt = 0.01
    t_now = 1000.0
    
    # 1. Init
    for i in range(150):
        nav._dispatch_message("RAW_IMU", MockMsg("RAW_IMU", time_usec=int(t_now*1e6), xacc=0, yacc=0, zacc=-2048, xgyro=0, ygyro=0, zgyro=0), t_now)
        nav._dispatch_message("SCALED_IMU3", MockMsg("SCALED_IMU3", time_usec=int(t_now*1e6), xmag=300, ymag=0, zmag=400), t_now)
        t_now += dt
        
    nav.eskf._gps_origin = {"lat": 13.0, "lon": 80.0, "alt": 50.0}
    
    # 2. Normal Flight with delayed GPS (10 seconds)
    for i in range(1000):
        nav._dispatch_message("RAW_IMU", MockMsg("RAW_IMU", time_usec=int(t_now*1e6), xacc=0, yacc=0, zacc=-2048, xgyro=0, ygyro=0, zgyro=0), t_now)
        if i % 4 == 0:
            nav.mht.update_baro(-50.0)
        if i % 20 == 0:
            msg_gps = MockMsg("GPS_RAW_INT", time_usec=int((t_now - 0.2)*1e6), fix_type=3, lat=int(13.0*1e7), lon=int(80.0*1e7), alt=int(50.0*1000), eph=100)
            nav._dispatch_message("GPS_RAW_INT", msg_gps, t_now)
        t_now += dt
        
    cov_normal = np.trace(nav.eskf.P[0:3, 0:3])
    
    # 3. GPS Dropout (5 seconds)
    for i in range(500):
        nav._dispatch_message("RAW_IMU", MockMsg("RAW_IMU", time_usec=int(t_now*1e6), xacc=0, yacc=0, zacc=-2048, xgyro=0, ygyro=0, zgyro=0), t_now)
        if i % 4 == 0:
            nav.mht.update_baro(-50.0)
        t_now += dt
        
    cov_dropout = np.trace(nav.eskf.P[0:3, 0:3])
    assert cov_dropout > cov_normal, "Covariance should grow during GPS dropout"
    
    # 4. GPS Recovery
    for i in range(500):
        nav._dispatch_message("RAW_IMU", MockMsg("RAW_IMU", time_usec=int(t_now*1e6), xacc=0, yacc=0, zacc=-2048, xgyro=0, ygyro=0, zgyro=0), t_now)
        if i % 4 == 0:
            nav.mht.update_baro(-50.0)
        if i % 20 == 0:
            msg_gps = MockMsg("GPS_RAW_INT", time_usec=int((t_now - 0.2)*1e6), fix_type=3, lat=int(13.0*1e7), lon=int(80.0*1e7), alt=int(50.0*1000), eph=100)
            nav._dispatch_message("GPS_RAW_INT", msg_gps, t_now)
        t_now += dt
        
    cov_recovery = np.trace(nav.eskf.P[0:3, 0:3])
    assert cov_recovery < cov_dropout, "Covariance should shrink after GPS recovery"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
