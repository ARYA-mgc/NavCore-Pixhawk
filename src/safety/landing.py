#!/usr/bin/env python3
# Emergency Landing Site Detection
#
# When Return-To-Home is blocked or the drone needs to land NOW,
# this module analyzes Livox lidar point cloud data to find safe
# flat areas within reach.
#
# Algorithm:
#   1. Voxel-grid the incoming point cloud
#   2. Fit local planes to grid cells using SVD
#   3. Score cells by: surface roughness, area, slope, distance
#   4. Return ranked list of landing candidates
#
# Integrates with RTH logic in m.py — if the home path is blocked,
# divert to the nearest safe landing zone.

import logging
import math
import numpy as np
from typing import Optional, List
from dataclasses import dataclass

log = logging.getLogger("landing_detect")


@dataclass
class LandingCandidate:
    """A potential emergency landing site."""
    position: np.ndarray       # NED center position (m)
    roughness: float           # surface roughness RMS (m) — lower = flatter
    slope_deg: float           # surface slope (degrees) — lower = better
    area_m2: float             # estimated area (m²) — larger = safer
    distance: float            # distance from current position (m)
    score: float               # composite score (higher = better)
    n_points: int              # number of lidar points in this cell

    def to_dict(self) -> dict:
        return {
            "pos_n": float(self.position[0]),
            "pos_e": float(self.position[1]),
            "pos_d": float(self.position[2]),
            "roughness": self.roughness,
            "slope_deg": self.slope_deg,
            "area_m2": self.area_m2,
            "distance": self.distance,
            "score": self.score,
            "n_points": self.n_points,
        }


class EmergencyLandingDetector:
    """Detects safe landing zones from Livox lidar point cloud data.

    Uses a grid-based approach to evaluate terrain suitability:
      - Divides the ground plane into cells
      - Fits a plane to points in each cell (SVD)
      - Scores by roughness (plane fit residual), slope, area, distance
      - Returns ranked candidates

    Designed to run asynchronously in the ThreadPoolExecutor to avoid
    blocking the 100Hz navigation loop.
    """

    # Grid cell size for terrain analysis
    CELL_SIZE = 2.0  # meters — each cell represents a 2×2m landing zone

    # Minimum points per cell for valid analysis
    MIN_POINTS_PER_CELL = 10

    # Maximum acceptable slope for landing (degrees)
    MAX_SLOPE_DEG = 10.0

    # Maximum acceptable surface roughness (m RMS)
    MAX_ROUGHNESS = 0.15

    # Minimum area for safe landing (m²)
    MIN_AREA = 2.0  # 2 m² minimum for a multi-rotor

    # Maximum search radius (m)
    MAX_SEARCH_RADIUS = 50.0

    # Scoring weights
    W_ROUGHNESS = 0.30   # lower roughness = higher score
    W_SLOPE = 0.25       # lower slope = higher score
    W_AREA = 0.15        # larger area = higher score
    W_DISTANCE = 0.30    # closer = higher score

    def __init__(self):
        self._last_candidates: List[LandingCandidate] = []
        self._scan_count = 0
        self._best_site: Optional[LandingCandidate] = None

        log.info("Emergency landing detector initialized")

    @property
    def best_site(self) -> Optional[LandingCandidate]:
        return self._best_site

    @property
    def has_safe_site(self) -> bool:
        return self._best_site is not None and self._best_site.score > 0.5

    def analyze_point_cloud(self, points: np.ndarray,
                            current_pos: np.ndarray,
                            current_alt_agl: float) -> List[LandingCandidate]:
        """Analyze a lidar point cloud for safe landing zones.

        Args:
            points: Nx3 point cloud in body frame (meters)
            current_pos: current drone NED position (meters)
            current_alt_agl: current altitude above ground (meters)

        Returns:
            List of LandingCandidate sorted by score (best first)
        """
        self._scan_count += 1

        if points.shape[0] < 20:
            return []

        # Transform points to NED (approximate: assume level flight)
        # In a full implementation, use the current attitude quaternion
        pts_ned = points.copy()
        pts_ned[:, 2] += current_alt_agl  # shift to ground level

        # Add current position offset
        pts_ned[:, 0] += current_pos[0]
        pts_ned[:, 1] += current_pos[1]

        # Filter by search radius
        horiz_dist = np.sqrt(
            (pts_ned[:, 0] - current_pos[0]) ** 2 +
            (pts_ned[:, 1] - current_pos[1]) ** 2
        )
        in_range = horiz_dist < self.MAX_SEARCH_RADIUS
        pts_ned = pts_ned[in_range]

        if pts_ned.shape[0] < self.MIN_POINTS_PER_CELL:
            return []

        # Grid the points
        candidates = self._grid_analysis(pts_ned, current_pos)

        # Sort by score (best first)
        candidates.sort(key=lambda c: c.score, reverse=True)

        self._last_candidates = candidates
        if candidates:
            self._best_site = candidates[0]
            if self._best_site.score > 0.5:
                log.debug(f"Landing site found: score={self._best_site.score:.2f} "
                          f"dist={self._best_site.distance:.1f}m "
                          f"rough={self._best_site.roughness:.3f}m "
                          f"slope={self._best_site.slope_deg:.1f}°")
        else:
            self._best_site = None

        return candidates

    def _grid_analysis(self, pts_ned: np.ndarray,
                       current_pos: np.ndarray) -> List[LandingCandidate]:
        """Divide point cloud into grid cells and analyze each."""
        candidates = []

        # Compute grid boundaries
        min_n = pts_ned[:, 0].min()
        max_n = pts_ned[:, 0].max()
        min_e = pts_ned[:, 1].min()
        max_e = pts_ned[:, 1].max()

        n_cells_n = max(1, int((max_n - min_n) / self.CELL_SIZE) + 1)
        n_cells_e = max(1, int((max_e - min_e) / self.CELL_SIZE) + 1)

        for ci in range(n_cells_n):
            for cj in range(n_cells_e):
                # Cell boundaries
                cell_n_min = min_n + ci * self.CELL_SIZE
                cell_n_max = cell_n_min + self.CELL_SIZE
                cell_e_min = min_e + cj * self.CELL_SIZE
                cell_e_max = cell_e_min + self.CELL_SIZE

                # Points in this cell
                mask = (
                    (pts_ned[:, 0] >= cell_n_min) &
                    (pts_ned[:, 0] < cell_n_max) &
                    (pts_ned[:, 1] >= cell_e_min) &
                    (pts_ned[:, 1] < cell_e_max)
                )
                cell_pts = pts_ned[mask]

                if cell_pts.shape[0] < self.MIN_POINTS_PER_CELL:
                    continue

                # Fit plane using SVD
                roughness, slope_deg, normal = self._fit_plane(cell_pts)

                if roughness is None:
                    continue

                # Reject obviously bad surfaces
                if slope_deg > self.MAX_SLOPE_DEG * 2.0:
                    continue

                # Cell center
                center = np.array([
                    (cell_n_min + cell_n_max) / 2.0,
                    (cell_e_min + cell_e_max) / 2.0,
                    np.mean(cell_pts[:, 2]),
                ])

                # Distance from current position
                dist = np.linalg.norm(center[0:2] - current_pos[0:2])

                # Estimated area (cell coverage based on point density)
                coverage = min(cell_pts.shape[0] / 50.0, 1.0)
                area = self.CELL_SIZE ** 2 * coverage

                # Score the candidate
                score = self._compute_score(roughness, slope_deg, area, dist)

                candidates.append(LandingCandidate(
                    position=center,
                    roughness=roughness,
                    slope_deg=slope_deg,
                    area_m2=area,
                    distance=dist,
                    score=score,
                    n_points=cell_pts.shape[0],
                ))

        return candidates

    def _fit_plane(self, points: np.ndarray):
        """Fit a plane to 3D points using SVD.

        Returns (roughness_rms, slope_degrees, normal_vector) or (None, None, None).
        """
        if points.shape[0] < 3:
            return None, None, None

        # Center the points
        centroid = points.mean(axis=0)
        centered = points - centroid

        # SVD — the normal is the singular vector with smallest singular value
        try:
            _, s, Vt = np.linalg.svd(centered)
        except np.linalg.LinAlgError:
            return None, None, None

        normal = Vt[-1]  # last row = smallest singular value direction

        # Ensure normal points upward (negative D in NED)
        if normal[2] > 0:
            normal = -normal

        # Roughness: RMS of distances from the fitted plane
        distances = centered @ normal
        roughness = float(np.sqrt(np.mean(distances ** 2)))

        # Slope: angle between surface normal and vertical (NED down = [0,0,-1])
        vertical = np.array([0.0, 0.0, -1.0])
        cos_angle = abs(np.dot(normal, vertical))
        cos_angle = min(1.0, max(0.0, cos_angle))
        slope_deg = math.degrees(math.acos(cos_angle))

        return roughness, slope_deg, normal

    def _compute_score(self, roughness: float, slope_deg: float,
                       area: float, distance: float) -> float:
        """Compute a composite landing site score (0.0 = worst, 1.0 = best).

        Score components:
          - Roughness: 1.0 at 0m, 0.0 at MAX_ROUGHNESS
          - Slope: 1.0 at 0°, 0.0 at MAX_SLOPE_DEG
          - Area: 0.0 at 0 m², 1.0 at 4×MIN_AREA
          - Distance: 1.0 at 0m, 0.0 at MAX_SEARCH_RADIUS
        """
        # Roughness score (lower = better)
        s_rough = max(0.0, 1.0 - roughness / self.MAX_ROUGHNESS)

        # Slope score (lower = better)
        s_slope = max(0.0, 1.0 - slope_deg / self.MAX_SLOPE_DEG)

        # Area score (larger = better)
        s_area = min(1.0, area / (4.0 * self.MIN_AREA))

        # Distance score (closer = better)
        s_dist = max(0.0, 1.0 - distance / self.MAX_SEARCH_RADIUS)

        # Weighted composite
        score = (self.W_ROUGHNESS * s_rough +
                 self.W_SLOPE * s_slope +
                 self.W_AREA * s_area +
                 self.W_DISTANCE * s_dist)

        return float(score)

    def get_status(self) -> dict:
        """Return landing detector status."""
        return {
            "scan_count": self._scan_count,
            "has_safe_site": self.has_safe_site,
            "n_candidates": len(self._last_candidates),
            "best_site": self._best_site.to_dict() if self._best_site else None,
        }
