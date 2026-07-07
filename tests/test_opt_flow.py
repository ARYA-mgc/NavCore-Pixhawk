import numpy as np
from fusion.opt_flow import TerrainKalmanFilter, MedianFilter, OpticalFlowINS

class MockFlowMsg:
    def __init__(self, integrated_x, integrated_y, integrated_xgyro, integrated_ygyro, distance, dt_us):
        self.integrated_x = integrated_x
        self.integrated_y = integrated_y
        self.integrated_xgyro = integrated_xgyro
        self.integrated_ygyro = integrated_ygyro
        self.distance = distance
        self.integration_time_us = dt_us
        self.quality = 255


def test_terrain_kalman_filter():
    tkf = TerrainKalmanFilter()
    assert not tkf.initialized
    
    # Initial update
    tkf.update(10.0)
    assert tkf.initialized
    assert np.isclose(tkf.get_distance(), 10.0)
    
    # Predict with moving down (vz = 2.0 m/s for 0.1s)
    # distance should decrease by 0.2m
    tkf.predict(vz=2.0, dt=0.1)
    assert np.isclose(tkf.get_distance(), 9.8)
    
    # Update with new reading slightly off (10.0). Since Q/R are tuned, it should blend.
    tkf.update(10.0)
    dist = tkf.get_distance()
    assert 9.8 < dist < 10.0


def test_median_filter():
    med = MedianFilter(size=3)
    
    # Add values
    med.update(1.0, 1.0)
    med.update(2.0, 2.0)
    mx, my = med.update(3.0, 3.0)
    # Median of [1, 2, 3] is 2
    assert np.isclose(mx, 2.0)
    assert np.isclose(my, 2.0)
    
    # Spike injected
    mx, my = med.update(100.0, -100.0)
    # buffer_x is now [2, 3, 100] -> Median is 3
    # buffer_y is now [2, 3, -100] -> Median is 2
    assert np.isclose(mx, 3.0)
    assert np.isclose(my, 2.0)


def test_optical_flow_ins_process():
    ins = OpticalFlowINS()
    # Set camera offset 1m forward
    ins.r_mount = np.array([1.0, 0.0, 0.0])
    
    dt_us = 100000  # 0.1s
    
    # Test case: purely translating forward at 2 m/s
    # altitude = 10m
    # Pitch rate = 0
    # True v_body_x = 2.0, v_body_y = 0.0
    # v_cam_x = 2.0, v_cam_y = 0.0
    # flow_y = v_cam_x * dt_s / distance = 2.0 * 0.1 / 10 = 0.02 rad
    # flow_x = -v_cam_y * dt_s / distance = 0.0 rad
    msg1 = MockFlowMsg(0.0, 0.02, 0.0, 0.0, 10.0, dt_us)
    
    # Need to prime the terrain filter and median filter
    ins.process_flow_for_eskf(msg1, use_raw_flow=True, vz=0.0, omega=np.zeros(3))
    ins.process_flow_for_eskf(msg1, use_raw_flow=True, vz=0.0, omega=np.zeros(3))
    vx, vy = ins.process_flow_for_eskf(msg1, use_raw_flow=True, vz=0.0, omega=np.zeros(3))
    
    assert vx is not None
    assert vy is not None
    # We should get exactly 2.0 back (modulo kalman blending, but distance should be steady)
    assert np.isclose(vx, 2.0, atol=0.1)
    assert np.isclose(vy, 0.0, atol=0.1)

    # Test case 2: Pure pitch rotation, no CG translation (CG velocity = 0)
    # pitch rate omega_y = 1.0 rad/s
    # Lever arm r_mount_x = 1.0m
    # v_cam_z = -omega_y * r_mount_x = -1.0 * 1.0 = -1.0 (downwards at camera)
    # Actually v_cam_x = v_cg_x + (omega x r_mount)_x
    # omega = [0, 1.0, 0]
    # r = [1.0, 0, 0]
    # omega x r = [0, 0, -1.0] -> v_cam_z = -1.0. v_cam_x = 0, v_cam_y = 0.
    # What if we have yaw rate? omega_z = 1.0
    # omega = [0, 0, 1.0], r = [1.0, 0, 0]
    # omega x r = [0, 1.0, 0] -> v_cam_y = 1.0
    # CG is still, so camera moves left at 1.0 m/s
    # v_cam_x = 0, v_cam_y = 1.0
    # flow_y = 0.0
    # flow_x = -1.0 * 0.1 / 10 = -0.01
    
    # Send messages with flow_x = -0.01, flow_y = 0.0 (and gyro_x = 0, gyro_y = 0, gyro_z not in flow)
    # However we pass omega explicitly to process_flow
    msg2 = MockFlowMsg(-0.01, 0.0, 0.0, 0.0, 10.0, dt_us)
    ins.process_flow_for_eskf(msg2, use_raw_flow=True, vz=0.0, omega=np.array([0.0, 0.0, 1.0]))
    ins.process_flow_for_eskf(msg2, use_raw_flow=True, vz=0.0, omega=np.array([0.0, 0.0, 1.0]))
    vx, vy = ins.process_flow_for_eskf(msg2, use_raw_flow=True, vz=0.0, omega=np.array([0.0, 0.0, 1.0]))
    
    # The output vx, vy should be the CG velocity, which is 0!
    assert np.isclose(vx, 0.0, atol=0.1)
    assert np.isclose(vy, 0.0, atol=0.1)

