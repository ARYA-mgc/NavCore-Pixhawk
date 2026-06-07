# Multi-Hypothesis Tracking.
# Trust issues with the GPS.

import copy
import logging
import numpy as np
from core.eskf import ESKF

log = logging.getLogger(__name__)

class MHTManager:
    """
    Multi-Hypothesis Tracking (MHT) for Aviation-Grade Safety.
    
    Instead of maintaining multiple filters continuously (which is expensive),
    we use Deferred-Decision MHT. If a critical sensor (like GPS) fails RAIM
    repeatedly, it might indicate a genuine coordinate shift rather than a glitch.
    
    We spawn a single parallel 'shadow' ESKF that accepts the rejected measurements.
    After an evaluation window, if the shadow filter stabilizes and shows better
    consistency than the primary, we swap them.
    """
    
    MHT_EVALUATION_WINDOW = 3.0  # seconds
    MAX_SHADOW_LIFETIME = 6.0    # seconds
    
    def __init__(self, primary_eskf: ESKF):
        self.primary = primary_eskf
        self.shadow = None
        self.shadow_spawn_time = 0.0
        self.shadow_last_update = 0.0

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        self.primary.predict(accel, gyro, dt)
        if self.shadow:
            self.shadow.predict(accel, gyro, dt)
            
    def _evaluate_and_merge(self, t_now: float):
        """Evaluate if the shadow filter successfully absorbed the jump."""
        if self.shadow is None:
            return
            
        prim_health = self.primary._health.value
        shad_health = self.shadow._health.value
        
        # We also check the condition numbers to ensure numerical stability
        prim_cond = np.linalg.cond(self.primary.P)
        shad_cond = np.linalg.cond(self.shadow.P)
        
        log.info(f"MHT Evaluation: Primary Cond={prim_cond:.1e}, Shadow Cond={shad_cond:.1e}")
        
        # If the shadow filter is healthy and primary is degraded, or shadow is numerically better
        swap = False
        if shad_health < prim_health:
            swap = True
        elif shad_health == prim_health and shad_health <= 1:  # Both HEALTHY
            # If both are healthy, we prefer the one with no active sensor rejections
            prim_rej = self.primary._sensor_rejections.get("gps", 0)
            shad_rej = self.shadow._sensor_rejections.get("gps", 0)
            if shad_rej == 0 and prim_rej > 0:
                swap = True
                
        if swap:
            log.warning("MHT: Shadow filter absorbed jump successfully! Swapping to Shadow.")
            self.primary = self.shadow
            self.shadow = None
        elif t_now - self.shadow_spawn_time > self.MAX_SHADOW_LIFETIME:
            log.info("MHT: Shadow filter failed to converge better than primary. Discarding shadow.")
            self.shadow = None

    # --- Sensor Updates ---
    
    def update_gps(self, lat: float, lon: float, alt: float, hdop: float, t_now: float):
        accepted = self.primary.update_gps(lat, lon, alt, hdop)
        rejections = self.primary._sensor_rejections.get("gps", 0)
        
        # Spawning logic: 3 consecutive rejections trigger shadow creation
        if not accepted and rejections >= 3 and self.shadow is None:
            log.warning("MHT: GPS rejected 3 times consecutively. Spawning shadow filter.")
            self.shadow = copy.deepcopy(self.primary)
            self.shadow_spawn_time = t_now
            
            # Inflate position and velocity covariance via Square-Root factor (U)
            self.shadow.U[0:3, 0:3] = np.eye(3) * 100.0  # 100m uncertainty (P = U^T U)
            self.shadow.U[3:6, 3:6] = np.eye(3) * 10.0   # 10m/s uncertainty
            
            # Force accept the jumped GPS in the shadow
            self.shadow.update_gps(lat, lon, alt, hdop, force_accept=True)
            self.shadow_last_update = t_now
            
        elif self.shadow is not None:
            # Shadow always accepts GPS if it's currently being tracked
            self.shadow.update_gps(lat, lon, alt, hdop, force_accept=True)
            self.shadow_last_update = t_now
            
            if t_now - self.shadow_spawn_time > self.MHT_EVALUATION_WINDOW:
                self._evaluate_and_merge(t_now)

    def update_mag(self, mag_yaw: float, mag_norm: float, t_now: float):
        self.primary.update_mag(mag_yaw, mag_norm, t_now)
        if self.shadow:
            self.shadow.update_mag(mag_yaw, mag_norm, t_now)

    def update_baro(self, baro_alt: float):
        self.primary.update_baro(baro_alt)
        if self.shadow:
            self.shadow.update_baro(baro_alt)
            
    def update_optical_flow(self, flow_vx: float, flow_vy: float, distance: float, quality: int, enable_rot_comp: bool = True):
        self.primary.update_optical_flow(flow_vx, flow_vy, distance, quality, enable_rot_comp=enable_rot_comp)
        if self.shadow:
            self.shadow.update_optical_flow(flow_vx, flow_vy, distance, quality, enable_rot_comp=enable_rot_comp)
            
    def update_external(self, z, z_pred, H, R, source="external"):
        # Not using force_accept here because this handles generic external inputs
        self.primary.update_external(z, z_pred, H, R, source=source)
        if self.shadow:
            self.shadow.update_external(z, z_pred, H, R, source=source)

    def scale_process_noise(self, vib_level: float):
        self.primary.scale_process_noise(vib_level)
        if self.shadow:
            self.shadow.scale_process_noise(vib_level)
            
    def update_zupt(self):
        self.primary.update_zupt()
        if self.shadow:
            self.shadow.update_zupt()
            
    def update_lidar_range(self, distance: float, weight: float = 1.0):
        self.primary.update_lidar_range(distance, weight=weight)
        if self.shadow:
            self.shadow.update_lidar_range(distance, weight=weight)
            
    def update_radar_velocity(self, vx: float, vy: float, vz: float, weight: float = 1.0):
        self.primary.update_radar_velocity(vx, vy, vz, weight=weight)
        if self.shadow:
            self.shadow.update_radar_velocity(vx, vy, vz, weight=weight)
            
    def initialize_from_sensors(self, accel_arr, mag_arr):
        return self.primary.initialize_from_sensors(accel_arr, mag_arr)
        
    @property
    def _initialized(self):
        return self.primary._initialized

    @property
    def _gps_origin(self):
        return self.primary._gps_origin
