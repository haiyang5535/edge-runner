#!/usr/bin/env python3
"""
GroundPlane - Homography Calibration and BEV Projection
========================================================

Ground-plane homography for BEV (Bird's Eye View) distance calculation.
Enables accurate real-world distance measurement in meters.

KEY DESIGN: H maps image pixels → meters directly.
No scale_px_per_m conversion needed — simpler, fewer bugs.

Reference: OpenCV getPerspectiveTransform
- The function computes a 3×3 matrix that maps src points to dst points.
- By making dst points in meters, the transform output is in meters.

Usage:
    gp = GroundPlane("configs/camera_calibration.json")
    distance_m = gp.distance_meters(point1_px, point2_px)
    bev_pos = gp.project_to_bev(point_px)
"""

import json
import numpy as np
import cv2
from typing import Tuple, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CalibrationData:
    """Calibration data structure."""
    camera_id: str
    calibration_date: str
    frame_size: Tuple[int, int]
    source_points_px: List[List[int]]
    destination_points_m: List[List[float]]
    homography_matrix: np.ndarray
    valid_floor_region: Optional[List[List[float]]] = None
    notes: Optional[str] = None


class GroundPlane:
    """
    Ground-plane homography for BEV distance calculation.
    
    Key design: H maps image pixels → meters directly.
    No scale_px_per_m needed — simpler, fewer bugs.
    """
    
    def __init__(self, calibration_path: Optional[str] = None):
        """
        Initialize GroundPlane.
        
        Args:
            calibration_path: Path to calibration JSON file.
                             If None, use identity (no transformation).
        """
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.calibration: Optional[CalibrationData] = None
        self.is_calibrated = False
        
        if calibration_path:
            self.load_calibration(calibration_path)
    
    def load_calibration(self, path: str) -> bool:
        """
        Load homography calibration from JSON file.
        
        Args:
            path: Path to calibration JSON
            
        Returns:
            True if loaded successfully
        """
        path = Path(path)
        if not path.exists():
            print(f"⚠️ Calibration file not found: {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.H = np.array(data['homography_matrix'], dtype=np.float64)
            self.H_inv = np.linalg.inv(self.H)
            
            self.calibration = CalibrationData(
                camera_id=data.get('camera_id', 'unknown'),
                calibration_date=data.get('calibration_date', ''),
                frame_size=tuple(data.get('frame_size', [1920, 1080])),
                source_points_px=data.get('source_points_px', []),
                destination_points_m=data.get('destination_points_m', []),
                homography_matrix=self.H,
                valid_floor_region=data.get('valid_floor_region_normalized'),
                notes=data.get('notes')
            )
            
            self.is_calibrated = True
            print(f"✅ GroundPlane calibration loaded: {self.calibration.camera_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False
    
    @staticmethod
    def compute_homography(
        src_points_px: List[List[int]],
        dst_points_m: List[List[float]]
    ) -> np.ndarray:
        """
        Compute homography matrix from point correspondences.
        
        Args:
            src_points_px: Image coordinates in pixels
                          e.g., [[500,800], [1400,800], [1600,1000], [300,1000]]
            dst_points_m: Real-world coordinates in METERS
                         e.g., [[0,0], [3,0], [3,2], [0,2]]
        
        Returns:
            3x3 homography matrix
        """
        src = np.array(src_points_px, dtype=np.float32)
        dst = np.array(dst_points_m, dtype=np.float32)
        
        # Use findHomography for >4 points + RANSAC option
        H, _ = cv2.findHomography(src, dst)
        return H
    
    def project_to_bev(self, point_px: Tuple[int, int]) -> Tuple[float, float]:
        """
        Project image point to BEV coordinates (meters).
        
        Args:
            point_px: (x, y) in image pixel coordinates
            
        Returns:
            (x_m, y_m) in real-world meters
            
        Raises:
            ValueError: If not calibrated
        """
        if not self.is_calibrated or self.H is None:
            raise ValueError("GroundPlane not calibrated. Load calibration first.")
        
        pt = np.array([[point_px]], dtype=np.float32)
        bev_pt = cv2.perspectiveTransform(pt, self.H)
        return float(bev_pt[0][0][0]), float(bev_pt[0][0][1])
    
    def project_to_image(self, point_m: Tuple[float, float]) -> Tuple[int, int]:
        """
        Project BEV point (meters) back to image coordinates.
        
        Args:
            point_m: (x, y) in real-world meters
            
        Returns:
            (x, y) in image pixel coordinates
            
        Raises:
            ValueError: If not calibrated
        """
        if not self.is_calibrated or self.H_inv is None:
            raise ValueError("GroundPlane not calibrated. Load calibration first.")
        
        pt = np.array([[point_m]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pt, self.H_inv)
        return int(img_pt[0][0][0]), int(img_pt[0][0][1])
    
    def distance_meters(
        self,
        point1_px: Tuple[int, int],
        point2_px: Tuple[int, int]
    ) -> float:
        """
        Calculate real-world distance (meters) between two image points.
        
        Both points are projected to BEV, then Euclidean distance is computed.
        Output is DIRECTLY in meters — no conversion needed.
        
        Args:
            point1_px: First point in image pixel coordinates
            point2_px: Second point in image pixel coordinates
            
        Returns:
            Distance in meters
            
        Raises:
            ValueError: If not calibrated
        """
        bev1 = self.project_to_bev(point1_px)
        bev2 = self.project_to_bev(point2_px)
        # Already in meters — just compute Euclidean
        return float(np.linalg.norm(np.array(bev1) - np.array(bev2)))
    
    def is_point_valid(
        self,
        point_px: Tuple[int, int],
        max_distance_m: float = 50.0
    ) -> bool:
        """
        Sanity check: reject points that project to unreasonable distances.
        
        Catches points outside the calibrated floor region.
        
        Args:
            point_px: Point in image pixel coordinates
            max_distance_m: Maximum reasonable distance from origin
            
        Returns:
            True if point projects to reasonable BEV coordinates
        """
        if not self.is_calibrated:
            return True  # Allow all if not calibrated
        
        try:
            bev = self.project_to_bev(point_px)
            dist_from_origin = np.linalg.norm(bev)
            return 0 < dist_from_origin < max_distance_m
        except Exception:
            return False
    
    def save_calibration(
        self,
        path: str,
        camera_id: str,
        src_points_px: List[List[int]],
        dst_points_m: List[List[float]],
        frame_size: Tuple[int, int],
        valid_floor_region: Optional[List[List[float]]] = None,
        notes: Optional[str] = None
    ):
        """
        Save calibration to JSON file.
        
        Args:
            path: Output file path
            camera_id: Identifier for this camera
            src_points_px: Source points in pixels
            dst_points_m: Destination points in meters
            frame_size: Frame dimensions (width, height)
            valid_floor_region: Optional normalized polygon of valid floor area
            notes: Optional calibration notes
        """
        H = self.compute_homography(src_points_px, dst_points_m)
        
        data = {
            "camera_id": camera_id,
            "calibration_date": datetime.now().isoformat(),
            "frame_size": list(frame_size),
            "source_points_px": src_points_px,
            "destination_points_m": dst_points_m,
            "homography_matrix": H.tolist(),
        }
        
        if valid_floor_region:
            data["valid_floor_region_normalized"] = valid_floor_region
        if notes:
            data["notes"] = notes
        
        # Calculate real-world rectangle dimensions for reference
        if len(dst_points_m) >= 4:
            width_m = abs(dst_points_m[1][0] - dst_points_m[0][0])
            height_m = abs(dst_points_m[2][1] - dst_points_m[1][1])
            data["real_world_rect"] = {
                "width_m": width_m,
                "height_m": height_m,
                "description": f"Calibration rectangle: {width_m}m wide × {height_m}m deep on floor"
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Calibration saved to {path}")
        
        # Update internal state
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.is_calibrated = True
    
    def reset(self):
        """Reset calibration state (for LoopResetManager interface)."""
        pass  # GroundPlane calibration persists across loops


# ============================================================
# Module-level singleton
# ============================================================

_ground_plane: Optional[GroundPlane] = None


def get_ground_plane(calibration_path: Optional[str] = None) -> GroundPlane:
    """Get or create singleton GroundPlane instance."""
    global _ground_plane
    if _ground_plane is None:
        _ground_plane = GroundPlane(calibration_path=calibration_path)
    return _ground_plane


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    # Create test calibration
    # Scenario: 4 points on floor forming 3m x 2m rectangle
    src_points = [
        [500, 800],    # top-left in image
        [1400, 800],   # top-right in image
        [1600, 1000],  # bottom-right in image
        [300, 1000]    # bottom-left in image
    ]
    dst_points = [
        [0.0, 0.0],    # 0m, 0m
        [3.0, 0.0],    # 3m, 0m
        [3.0, 2.0],    # 3m, 2m
        [0.0, 2.0]     # 0m, 2m
    ]
    
    # Create temporary calibration file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        H = GroundPlane.compute_homography(src_points, dst_points)
        data = {
            "camera_id": "test_cam",
            "calibration_date": "2026-01-28T10:00:00Z",
            "frame_size": [1920, 1080],
            "source_points_px": src_points,
            "destination_points_m": dst_points,
            "homography_matrix": H.tolist()
        }
        json.dump(data, f)
    
    try:
        # Load and test
        gp = GroundPlane(temp_path)
        
        print("GroundPlane Test")
        print("=" * 40)
        
        # Test projection
        test_points_px = [
            (500, 800),    # Should be near (0, 0)
            (1400, 800),   # Should be near (3, 0)
            (950, 900),    # Should be somewhere in middle
        ]
        
        print("\nProjection test:")
        for px in test_points_px:
            bev = gp.project_to_bev(px)
            print(f"  {px} -> ({bev[0]:.2f}m, {bev[1]:.2f}m)")
        
        # Test distance
        print("\nDistance test:")
        dist = gp.distance_meters((500, 800), (1400, 800))
        print(f"  (500,800) to (1400,800): {dist:.2f}m (expected: ~3.0m)")
        
        dist2 = gp.distance_meters((500, 800), (300, 1000))
        print(f"  (500,800) to (300,1000): {dist2:.2f}m (expected: ~2.0m)")
        
        # Test validity check
        print("\nValidity test:")
        print(f"  (950, 900) valid: {gp.is_point_valid((950, 900))}")
        print(f"  (0, 0) valid: {gp.is_point_valid((0, 0))}")
        
        print("\nGroundPlane test complete!")
        
    finally:
        os.unlink(temp_path)
