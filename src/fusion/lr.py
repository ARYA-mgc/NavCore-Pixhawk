#!/usr/bin/env python3
# Lidar & Radar fusion.
# Actively avoiding trees since 2024.

import numpy as np
import logging

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False

log = logging.getLogger("lidar_radar")

class LidarRadarFusion:
    # Obstacle avoidance processor
    def __init__(self, voxel_size=0.1, rdr_reject=0.05):
        self.voxel_size = voxel_size
        self.rdr_reject = rdr_reject
        self.safe_distance_m = -1.0
        self.radar_vel = np.zeros(3)
        
        if not O3D_AVAILABLE:
            log.warning("Open3D missing! Lidar decimation will run in slow Python mode.")
            
    def process_livox_cloud(self, points: np.ndarray) -> float:
        # Decimate Livox Mid-360 non-repetitive scans fast.
        if points.shape[0] < 10:
            return -1.0
            
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            # Voxel downsample is critical for Livox's dense petal pattern
            downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            filtered_pts = np.asarray(downpcd.points)
        else:
            # Basic decimation
            filtered_pts = points[::10, :3]
            
        # Simplest collision check: finding the closest point
        distances = np.linalg.norm(filtered_pts, axis=1)
        self.safe_distance_m = float(np.min(distances))
        return self.safe_distance_m
        
    def process_ti_radar(self, targets: np.ndarray) -> np.ndarray:
        # Parses TI IWR6843AOP targets: expected format [x, y, z, doppler_velocity]
        if targets.shape[0] == 0:
            return np.zeros(3)
            
        # We want the dominant velocity vector, rejecting static clutter (v ~ 0)
        # Doppler is radial, so we project it back using the spatial coords
        spatial = targets[:, :3]
        doppler = targets[:, 3]
        
        norms = np.linalg.norm(spatial, axis=1)
        valid = (norms > 0.1) & (np.abs(doppler) > self.rdr_reject)  # ignore static noise
        
        if not np.any(valid):
            return np.zeros(3)
            
        t_spatial = spatial[valid]
        t_doppler = doppler[valid]
        n_valid = norms[valid]
        
        unit_vecs = t_spatial / n_valid[:, np.newaxis]
        
        # approximate 3D velocity vector = unit_vector * doppler_vel
        velocities = unit_vecs * t_doppler[:, np.newaxis]
        
        # Average the moving targets to get a gross velocity estimate
        self.radar_vel = np.mean(velocities, axis=0)
        return self.radar_vel
