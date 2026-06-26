#!/usr/bin/env python3
# Terrain Relative Navigation.
# Basically cruise missile tech.

import logging
import math
import numpy as np
from typing import Optional, Dict, Tuple
from collections import deque

log = logging.getLogger("trn")


class DEMTile:
    """A Digital Elevation Model tile — regular grid of terrain heights.

    DEM data can come from:
      - SRTM (30m resolution, free)
      - ALOS World 3D (5m resolution)
      - Custom photogrammetry
    """

    def __init__(self, origin_lat: float, origin_lon: float,
                 heights: np.ndarray, resolution: float = 30.0):
        """
        Args:
            origin_lat: latitude of SW corner (degrees)
            origin_lon: longitude of SW corner (degrees)
            heights: 2D array of terrain heights (m MSL), shape (rows, cols)
            resolution: grid spacing (meters)
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.heights = heights.astype(float)
        self.resolution = resolution
        self.n_rows, self.n_cols = heights.shape

        # Compute NED origin (for converting lat/lon to local)
        self._lat_rad = math.radians(origin_lat)

    def get_height(self, north: float, east: float) -> Optional[float]:
        """Get interpolated terrain height at a local NED position.

        Args:
            north: meters north of DEM origin
            east: meters east of DEM origin

        Returns:
            Terrain height (m MSL) or None if out of bounds
        """
        row = north / self.resolution
        col = east / self.resolution

        if row < 0 or row >= self.n_rows - 1 or col < 0 or col >= self.n_cols - 1:
            return None

        # Bilinear interpolation
        r0, c0 = int(row), int(col)
        r1, c1 = r0 + 1, c0 + 1
        fr, fc = row - r0, col - c0

        h00 = self.heights[r0, c0]
        h01 = self.heights[r0, c1]
        h10 = self.heights[r1, c0]
        h11 = self.heights[r1, c1]

        h = (h00 * (1-fr) * (1-fc) +
             h01 * (1-fr) * fc +
             h10 * fr * (1-fc) +
             h11 * fr * fc)

        return float(h)

    def get_patch(self, north: float, east: float,
                  radius_m: float) -> Optional[np.ndarray]:
        """Extract a square patch of DEM heights around a position.

        Args:
            north, east: center position in NED (m)
            radius_m: half-width of the patch (m)

        Returns:
            2D array of heights, or None if insufficient coverage
        """
        r_min = max(0, int((north - radius_m) / self.resolution))
        r_max = min(self.n_rows, int((north + radius_m) / self.resolution) + 1)
        c_min = max(0, int((east - radius_m) / self.resolution))
        c_max = min(self.n_cols, int((east + radius_m) / self.resolution) + 1)

        if r_max - r_min < 3 or c_max - c_min < 3:
            return None

        return self.heights[r_min:r_max, c_min:c_max].copy()


class TerrainRelativeNavigation:
    """Terrain-Relative Navigation using lidar-to-DEM matching.

    Provides absolute horizontal position without GPS by comparing
    the observed terrain height profile against a stored DEM.

    Matching algorithm:
      1. Build a local height map from Livox downward-looking points
      2. Extract the height map at the predicted position from the DEM
      3. Cross-correlate the two maps using normalized cross-correlation
      4. The correlation peak gives the position offset
      5. Feed the offset as an ESKF measurement update

    The key insight: terrain shape is a unique fingerprint that doesn't
    change over time (unlike visual features).
    """

    # Minimum altitude AGL for TRN to work
    MIN_ALT_AGL = 10.0  # meters — need enough swath for pattern matching

    # Height map resolution (meters per cell)
    MAP_RESOLUTION = 2.0

    # Search radius for correlation (meters)
    SEARCH_RADIUS = 30.0

    # Minimum correlation peak for acceptance
    MIN_CORRELATION = 0.6

    # Minimum points for valid height map
    MIN_POINTS = 50

    # Position measurement noise (meters) — depends on DEM resolution
    POS_STD = 5.0  # conservative for 30m SRTM

    # Update rate limiting
    MIN_UPDATE_INTERVAL = 2.0  # seconds

    def __init__(self, enable: bool = False):
        self._enabled = enable
        self._dem: Optional[DEMTile] = None
        self._dem_origin_ned = np.zeros(3)  # DEM origin in ESKF NED frame
        self._last_update_t = 0.0
        self._update_count = 0
        self._rejected_count = 0

        # Measurement matrices for ESKF
        self.H_pos = np.zeros((2, 20))
        self.H_pos[0, 0] = 1.0  # North
        self.H_pos[1, 1] = 1.0  # East

        self.R_pos = np.eye(2) * (self.POS_STD ** 2)

        if enable:
            log.info("Terrain Relative Navigation enabled")

    @property
    def is_active(self) -> bool:
        return self._enabled and self._dem is not None

    def load_dem(self, dem: DEMTile, origin_ned: np.ndarray = None):
        """Load a DEM tile for terrain matching.

        Args:
            dem: DEMTile object
            origin_ned: NED offset of DEM origin from ESKF origin
        """
        self._dem = dem
        if origin_ned is not None:
            self._dem_origin_ned = origin_ned
        log.info(f"DEM loaded: {dem.n_rows}x{dem.n_cols} cells @ "
                 f"{dem.resolution}m resolution")

    def process_lidar_scan(self, points: np.ndarray,
                           current_pos: np.ndarray,
                           current_alt_agl: float,
                           t_now: float) -> Optional[dict]:
        """Process a Livox lidar scan for terrain matching.

        Args:
            points: Nx3 point cloud in NED frame (meters)
            current_pos: ESKF-estimated NED position
            current_alt_agl: altitude above ground level (meters)
            t_now: current timestamp

        Returns:
            dict with position correction and H/R matrices for ESKF,
            or None if no valid match
        """
        if not self.is_active:
            return None

        # Rate limiting
        if t_now - self._last_update_t < self.MIN_UPDATE_INTERVAL:
            return None

        # Altitude check
        if current_alt_agl < self.MIN_ALT_AGL:
            return None

        if points.shape[0] < self.MIN_POINTS:
            return None

        # Step 1: Build local height map from lidar
        local_map = self._build_height_map(points, current_pos)
        if local_map is None:
            return None

        # Step 2: Get DEM patch at predicted position
        dem_north = current_pos[0] - self._dem_origin_ned[0]
        dem_east = current_pos[1] - self._dem_origin_ned[1]
        dem_patch = self._dem.get_patch(
            dem_north, dem_east,
            radius_m=self.SEARCH_RADIUS + local_map.shape[0] * self.MAP_RESOLUTION
        )
        if dem_patch is None:
            return None

        # Step 3: Cross-correlate
        offset, correlation = self._cross_correlate(local_map, dem_patch)
        if correlation < self.MIN_CORRELATION:
            self._rejected_count += 1
            log.debug(f"TRN rejected: correlation={correlation:.2f} < "
                      f"{self.MIN_CORRELATION}")
            return None

        # Step 4: Convert offset to position correction
        dn = offset[0] * self.MAP_RESOLUTION
        de = offset[1] * self.MAP_RESOLUTION

        # Measurement: true position ≈ predicted + offset
        z = current_pos[0:2] + np.array([dn, de])
        z_pred = current_pos[0:2]

        # Scale noise by inverse correlation (higher correlation = more trust)
        noise_scale = 1.0 / max(correlation, 0.3)
        R = self.R_pos * (noise_scale ** 2)

        self._update_count += 1
        self._last_update_t = t_now

        log.info(f"TRN match: offset=({dn:.1f}, {de:.1f})m "
                 f"correlation={correlation:.2f}")

        return {
            "type": "TRN",
            "z": z,
            "z_pred": z_pred,
            "H": self.H_pos,
            "R": R,
            "offset_ne": np.array([dn, de]),
            "correlation": correlation,
        }

    def _build_height_map(self, points: np.ndarray,
                          center: np.ndarray) -> Optional[np.ndarray]:
        """Build a 2D height map from lidar points.

        Grids the points and takes the mean height per cell.
        """
        # Relative to center
        rel = points[:, 0:2] - center[0:2]

        # Grid bounds
        half_size = self.SEARCH_RADIUS
        n_cells = int(2 * half_size / self.MAP_RESOLUTION)

        height_map = np.full((n_cells, n_cells), np.nan)
        count_map = np.zeros((n_cells, n_cells), dtype=int)

        for i in range(points.shape[0]):
            ci = int((rel[i, 0] + half_size) / self.MAP_RESOLUTION)
            cj = int((rel[i, 1] + half_size) / self.MAP_RESOLUTION)

            if 0 <= ci < n_cells and 0 <= cj < n_cells:
                if np.isnan(height_map[ci, cj]):
                    height_map[ci, cj] = points[i, 2]
                else:
                    height_map[ci, cj] += points[i, 2]
                count_map[ci, cj] += 1

        # Average heights
        valid = count_map > 0
        if np.sum(valid) < self.MIN_POINTS // 5:
            return None

        height_map[valid] /= count_map[valid]

        return height_map

    def _cross_correlate(self, local_map: np.ndarray,
                         dem_patch: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalized cross-correlation between local height map and DEM.

        Returns (offset_pixels, peak_correlation).
        """
        local_h, local_w = local_map.shape
        dem_h, dem_w = dem_patch.shape

        if local_h > dem_h or local_w > dem_w:
            return np.zeros(2), 0.0

        # Interpolate DEM to local map resolution
        scale = self._dem.resolution / self.MAP_RESOLUTION
        if abs(scale - 1.0) > 0.01:
            # Simple nearest-neighbor resampling
            new_h = int(dem_h * scale)
            new_w = int(dem_w * scale)
            if new_h < local_h or new_w < local_w:
                return np.zeros(2), 0.0
            # Resample
            row_idx = np.linspace(0, dem_h - 1, new_h).astype(int)
            col_idx = np.linspace(0, dem_w - 1, new_w).astype(int)
            dem_resampled = dem_patch[np.ix_(row_idx, col_idx)]
        else:
            dem_resampled = dem_patch

        # Replace NaN in local map with mean (for correlation)
        valid_mask = ~np.isnan(local_map)
        if np.sum(valid_mask) < 10:
            return np.zeros(2), 0.0

        local_filled = local_map.copy()
        local_mean = np.nanmean(local_map)
        local_filled[~valid_mask] = local_mean
        local_norm = local_filled - local_mean
        local_std = np.nanstd(local_map)
        if local_std < 0.1:
            return np.zeros(2), 0.0

        # Search window
        search_h = min(dem_resampled.shape[0] - local_h,
                       int(self.SEARCH_RADIUS / self.MAP_RESOLUTION))
        search_w = min(dem_resampled.shape[1] - local_w,
                       int(self.SEARCH_RADIUS / self.MAP_RESOLUTION))

        if search_h <= 0 or search_w <= 0:
            return np.zeros(2), 0.0

        best_corr = -1.0
        best_offset = np.zeros(2)

        # Center of search window
        center_r = (dem_resampled.shape[0] - local_h) // 2
        center_c = (dem_resampled.shape[1] - local_w) // 2

        for dr in range(-search_h // 2, search_h // 2 + 1):
            for dc in range(-search_w // 2, search_w // 2 + 1):
                r0 = center_r + dr
                c0 = center_c + dc
                if r0 < 0 or c0 < 0:
                    continue
                if r0 + local_h > dem_resampled.shape[0]:
                    continue
                if c0 + local_w > dem_resampled.shape[1]:
                    continue

                dem_sub = dem_resampled[r0:r0+local_h, c0:c0+local_w]
                dem_mean = np.mean(dem_sub)
                dem_std = np.std(dem_sub)
                if dem_std < 0.1:
                    continue

                # Normalized cross-correlation (only at valid positions)
                dem_norm = dem_sub - dem_mean
                ncc = np.sum(local_norm[valid_mask] * dem_norm[valid_mask]) / (
                    np.sum(valid_mask) * local_std * dem_std
                )

                if ncc > best_corr:
                    best_corr = ncc
                    best_offset = np.array([float(dr), float(dc)])

        return best_offset, max(best_corr, 0.0)

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "has_dem": self._dem is not None,
            "updates": self._update_count,
            "rejected": self._rejected_count,
        }
