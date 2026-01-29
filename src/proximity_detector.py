#!/usr/bin/env python3
"""
ProximityDetector - Person-Forklift Proximity Alert System
===========================================================

Safety-critical module for detecting dangerous proximity between
pedestrians and forklifts using Bottom-Center ground projection.

Architecture:
- Uses bbox bottom-center (foot/wheel position) for accurate ground distance
- Supports BEV (meters) with GroundPlane calibration (production)
- Falls back to y-scaled pixel distance (interim)
- Configurable distance thresholds

Usage:
    detector = ProximityDetector()
    violations = detector.check_proximity(detections, frame_size)
    
    # With BEV calibration (recommended):
    detector = ProximityDetector(ground_plane=gp)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING
from datetime import datetime

from .zone_manager import TrackedObject

if TYPE_CHECKING:
    from .ground_plane import GroundPlane


@dataclass
class ProximityViolation:
    """Detected proximity violation between person and forklift."""
    person_track_id: int
    forklift_track_id: int
    distance_px: float
    distance_m: Optional[float]  # BEV distance in meters (if calibrated)
    person_bbox: List[int]
    forklift_bbox: List[int]
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "person_track_id": self.person_track_id,
            "forklift_track_id": self.forklift_track_id,
            "distance_px": self.distance_px,
            "distance_m": self.distance_m,
            "person_bbox": self.person_bbox,
            "forklift_bbox": self.forklift_bbox,
            "timestamp": self.timestamp
        }


class ProximityDetector:
    """
    Detects dangerous proximity between pedestrians and forklifts.
    
    Uses Bottom-Center point (foot/wheel ground position) for distance
    calculation. Supports three distance modes:
    
    1. BEV (meters) - Production: Uses GroundPlane homography for accurate
       real-world distance. Requires calibration.
    
    2. Y-Scaled (pixels) - Interim: Accounts for perspective by scaling
       threshold based on vertical position. No calibration needed.
    
    3. Fixed (pixels) - Fallback: Simple Euclidean distance with fixed
       threshold. Only for testing.
    """
    
    # Distance thresholds in pixels (calibrated for 1080p)
    DANGER_DISTANCE_PX = 200      # Immediate danger
    WARNING_DISTANCE_PX = 350     # Caution zone
    
    # BEV distance thresholds in meters
    DANGER_DISTANCE_M = 2.0       # Immediate danger: <2 meters
    WARNING_DISTANCE_M = 4.0      # Caution zone: 2-4 meters
    
    # Y-Scaled distance parameters
    Y_SCALE_BASE_THRESHOLD = 150  # Pixels at y_ratio=0.5
    Y_SCALE_MIN = 0.5             # Scale factor at top of frame
    Y_SCALE_MAX = 1.5             # Scale factor at bottom of frame
    
    def __init__(
        self,
        danger_distance: int = None,
        warning_distance: int = None,
        danger_distance_m: float = None,
        ground_plane: Optional['GroundPlane'] = None,
        use_y_scaling: bool = True
    ):
        """
        Initialize ProximityDetector.
        
        Args:
            danger_distance: Override DANGER_DISTANCE_PX
            warning_distance: Override WARNING_DISTANCE_PX
            danger_distance_m: Override DANGER_DISTANCE_M (BEV mode)
            ground_plane: GroundPlane instance for BEV distance (recommended)
            use_y_scaling: Enable y-scaled distance (when BEV not available)
        """
        if danger_distance:
            self.DANGER_DISTANCE_PX = danger_distance
        if warning_distance:
            self.WARNING_DISTANCE_PX = warning_distance
        if danger_distance_m:
            self.DANGER_DISTANCE_M = danger_distance_m
        
        self.ground_plane = ground_plane
        self.use_y_scaling = use_y_scaling
        self.use_bev = ground_plane is not None and ground_plane.is_calibrated
    
    def _get_ground_point(self, bbox: List[int]) -> Tuple[float, float]:
        """
        Extract Bottom-Center point from bounding box.
        
        This represents the object's ground contact point (feet for person,
        wheels for forklift), providing more accurate distance estimation.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (x, y) bottom-center coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        return (cx, float(y2))  # Use y2 (bottom edge)
    
    def _calculate_distance(
        self, 
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def _get_y_scaled_threshold(
        self,
        bbox_bottom_y: float,
        frame_height: int
    ) -> float:
        """
        Get y-scaled danger threshold accounting for perspective.
        
        Objects lower in frame (higher y) are closer to camera and appear
        larger, so we use a larger threshold. Objects higher in frame
        (lower y) are farther and use a smaller threshold.
        
        Args:
            bbox_bottom_y: Bottom y-coordinate of bbox
            frame_height: Height of frame
            
        Returns:
            Adjusted distance threshold in pixels
        """
        y_ratio = bbox_bottom_y / frame_height
        # Scale from Y_SCALE_MIN at top to Y_SCALE_MAX at bottom
        scale = self.Y_SCALE_MIN + y_ratio * (self.Y_SCALE_MAX - self.Y_SCALE_MIN)
        return self.Y_SCALE_BASE_THRESHOLD * scale
    
    def _check_proximity_bev(
        self,
        person_ground: Tuple[float, float],
        forklift_ground: Tuple[float, float]
    ) -> Tuple[bool, float]:
        """
        Check proximity using BEV (meters) distance.
        
        Args:
            person_ground: Person footpoint in pixels
            forklift_ground: Forklift footpoint in pixels
            
        Returns:
            (is_danger, distance_meters)
        """
        if not self.ground_plane:
            return False, 0.0
        
        try:
            distance_m = self.ground_plane.distance_meters(
                (int(person_ground[0]), int(person_ground[1])),
                (int(forklift_ground[0]), int(forklift_ground[1]))
            )
            return distance_m < self.DANGER_DISTANCE_M, distance_m
        except Exception:
            return False, 0.0
    
    def check_proximity(
        self, 
        detections: List[TrackedObject],
        frame_size: Tuple[int, int] = (1920, 1080)
    ) -> List[ProximityViolation]:
        """
        Check for dangerous proximity between persons and forklifts.
        
        Args:
            detections: List of TrackedObject from YOLO detection
            frame_size: (width, height) for scaling
            
        Returns:
            List of ProximityViolation for detected dangers
        """
        # Separate persons and forklifts
        persons = [d for d in detections if d.class_name == "person"]
        forklifts = [d for d in detections if d.class_name == "forklift"]
        
        if not persons or not forklifts:
            return []
        
        violations = []
        now = datetime.now().isoformat()
        frame_height = frame_size[1]
        
        for person in persons:
            person_ground = self._get_ground_point(person.bbox)
            
            for forklift in forklifts:
                forklift_ground = self._get_ground_point(forklift.bbox)
                
                # Calculate pixel distance (always needed for logging)
                distance_px = self._calculate_distance(person_ground, forklift_ground)
                distance_m = None
                is_danger = False
                
                if self.use_bev and self.ground_plane:
                    # BEV mode: use real-world meters
                    is_danger, distance_m = self._check_proximity_bev(
                        person_ground, forklift_ground
                    )
                elif self.use_y_scaling:
                    # Y-scaled mode: adjust threshold based on position
                    # Use average y position of both objects
                    avg_y = (person_ground[1] + forklift_ground[1]) / 2
                    threshold = self._get_y_scaled_threshold(avg_y, frame_height)
                    is_danger = distance_px < threshold
                else:
                    # Fixed threshold mode
                    is_danger = distance_px < self.DANGER_DISTANCE_PX
                
                if is_danger:
                    violations.append(ProximityViolation(
                        person_track_id=person.track_id,
                        forklift_track_id=forklift.track_id,
                        distance_px=distance_px,
                        distance_m=distance_m,
                        person_bbox=person.bbox,
                        forklift_bbox=forklift.bbox,
                        timestamp=now
                    ))
        
        return violations
    
    def set_ground_plane(self, ground_plane: 'GroundPlane'):
        """
        Set or update ground plane for BEV mode.
        
        Args:
            ground_plane: Calibrated GroundPlane instance
        """
        self.ground_plane = ground_plane
        self.use_bev = ground_plane is not None and ground_plane.is_calibrated
        
        return violations
    
    def get_all_proximities(
        self,
        detections: List[TrackedObject]
    ) -> List[Tuple[int, int, float]]:
        """
        Get all person-forklift distances (for visualization/debugging).
        
        Returns:
            List of (person_id, forklift_id, distance) tuples
        """
        persons = [d for d in detections if d.class_name == "person"]
        forklifts = [d for d in detections if d.class_name == "forklift"]
        
        proximities = []
        for person in persons:
            person_ground = self._get_ground_point(person.bbox)
            for forklift in forklifts:
                forklift_ground = self._get_ground_point(forklift.bbox)
                distance = self._calculate_distance(person_ground, forklift_ground)
                proximities.append((person.track_id, forklift.track_id, distance))
        
        return proximities


# ============================================================
# Module-level singleton
# ============================================================

_proximity_detector: Optional[ProximityDetector] = None


def get_proximity_detector(**kwargs) -> ProximityDetector:
    """Get or create singleton ProximityDetector instance."""
    global _proximity_detector
    if _proximity_detector is None:
        _proximity_detector = ProximityDetector(**kwargs)
    return _proximity_detector


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Create test detector
    detector = ProximityDetector()
    
    # Create test detections
    person = TrackedObject(
        track_id=1,
        class_name="person",
        bbox=[500, 300, 600, 500],  # Bottom at y=500
        center=(550, 400),
        confidence=0.9
    )
    
    forklift_close = TrackedObject(
        track_id=10,
        class_name="forklift",
        bbox=[550, 400, 700, 550],  # Bottom at y=550, close to person
        center=(625, 475),
        confidence=0.85
    )
    
    forklift_far = TrackedObject(
        track_id=11,
        class_name="forklift",
        bbox=[1200, 300, 1400, 500],  # Far from person
        center=(1300, 400),
        confidence=0.8
    )
    
    # Test proximity detection
    detections = [person, forklift_close, forklift_far]
    violations = detector.check_proximity(detections)
    
    print(f"Found {len(violations)} proximity violations:")
    for v in violations:
        print(f"  Person #{v.person_track_id} <-> Forklift #{v.forklift_track_id}: {v.distance_px:.1f}px")
    
    # Test all proximities
    all_prox = detector.get_all_proximities(detections)
    print(f"\nAll proximities:")
    for pid, fid, dist in all_prox:
        status = "DANGER" if dist < detector.DANGER_DISTANCE_PX else "OK"
        print(f"  Person #{pid} <-> Forklift #{fid}: {dist:.1f}px [{status}]")
    
    print("\nProximityDetector test complete!")
