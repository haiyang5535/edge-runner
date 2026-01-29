#!/usr/bin/env python3
"""
ROIFilter - Polygon-Based Detection Exclusion Filter
=====================================================

Rejects detections in static "no-forklift" polygons to eliminate
false positives from cargo, shelving, or other static objects.

Uses BOTTOM_CENTER (footpoint) anchor for geometric consistency with:
- Supervision PolygonZone (Position.BOTTOM_CENTER)
- Ground-plane BEV projection
- Proximity distance calculation

Usage:
    filter = ROIFilter(exclusion_zones=[...])
    valid_detections = filter.filter(detections, frame_size)
"""

import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from .zone_manager import TrackedObject


@dataclass
class ExclusionZone:
    """Definition of an exclusion zone where forklift detections are rejected."""
    id: str
    polygon: List[List[float]]  # Normalized coords [[x1,y1], [x2,y2], ...]
    target_classes: List[str]  # Classes to filter (e.g., ["forklift"])
    description: Optional[str] = None


class ROIFilter:
    """
    Polygon-based ROI exclusion filter.
    
    Rejects detections in exclusion zones where specific classes
    (e.g., forklift) should not be detected (static cargo areas, corners).
    
    Uses BOTTOM_CENTER anchor (footpoint) for consistency with:
    - Supervision PolygonZone (Position.BOTTOM_CENTER)
    - Ground-plane BEV projection
    - Proximity distance calculation
    """
    
    def __init__(
        self,
        exclusion_zones: Optional[List[ExclusionZone]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize ROIFilter.
        
        Args:
            exclusion_zones: List of ExclusionZone objects
            config_path: Path to YAML config file (alternative to direct zones)
        """
        self.exclusion_zones: List[ExclusionZone] = []
        
        if config_path:
            self.load_config(config_path)
        elif exclusion_zones:
            self.exclusion_zones = exclusion_zones
    
    def load_config(self, config_path: str):
        """Load exclusion zones from YAML config file."""
        path = Path(config_path)
        if not path.exists():
            print(f"⚠️ ROI exclusion config not found: {config_path}")
            return
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.exclusion_zones.clear()
        
        for zone_data in config.get('exclusion_zones', []):
            self.exclusion_zones.append(ExclusionZone(
                id=zone_data.get('id', 'unknown'),
                polygon=zone_data['polygon'],
                target_classes=zone_data.get('target_classes', ['forklift']),
                description=zone_data.get('description')
            ))
        
        print(f"✅ ROIFilter loaded {len(self.exclusion_zones)} exclusion zones")
    
    @staticmethod
    def get_footpoint(bbox: List[int]) -> Tuple[int, int]:
        """
        Get bottom-center of bbox (ground contact point).
        
        This represents where the object contacts the floor - feet for
        a person, wheels for a forklift.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (x, y) bottom-center coordinates
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)
    
    @staticmethod
    def point_in_polygon(
        point: Tuple[float, float],
        polygon: List[List[float]]
    ) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        
        Args:
            point: (x, y) normalized coordinates (0-1)
            polygon: List of [x, y] normalized vertices
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def filter(
        self,
        detections: List[TrackedObject],
        frame_size: Tuple[int, int]
    ) -> List[TrackedObject]:
        """
        Filter detections, rejecting those in exclusion zones.
        
        Args:
            detections: List of TrackedObject from detection pipeline
            frame_size: (width, height) of frame
            
        Returns:
            Filtered list of TrackedObject (exclusions removed)
        """
        if not self.exclusion_zones:
            return detections
        
        valid = []
        width, height = frame_size
        
        for det in detections:
            # Check if this detection's class should be filtered
            should_check = any(
                det.class_name in zone.target_classes
                for zone in self.exclusion_zones
            )
            
            if not should_check:
                # Class not in any exclusion target, keep it
                valid.append(det)
                continue
            
            # Use BOTTOM_CENTER (footpoint), not bbox center
            foot_x, foot_y = self.get_footpoint(det.bbox)
            fx_norm = foot_x / width
            fy_norm = foot_y / height
            
            # Check if footpoint is in any exclusion zone for this class
            in_exclusion = False
            for zone in self.exclusion_zones:
                if det.class_name not in zone.target_classes:
                    continue
                if self.point_in_polygon((fx_norm, fy_norm), zone.polygon):
                    in_exclusion = True
                    break
            
            if not in_exclusion:
                valid.append(det)
        
        return valid
    
    def add_exclusion_zone(self, zone: ExclusionZone):
        """Add an exclusion zone."""
        self.exclusion_zones.append(zone)
    
    def remove_exclusion_zone(self, zone_id: str) -> bool:
        """Remove an exclusion zone by ID."""
        for i, zone in enumerate(self.exclusion_zones):
            if zone.id == zone_id:
                del self.exclusion_zones[i]
                return True
        return False
    
    def get_zones_for_overlay(
        self,
        frame_size: Tuple[int, int]
    ) -> List[dict]:
        """
        Get exclusion zones scaled for visualization overlay.
        
        Args:
            frame_size: (width, height) of frame
            
        Returns:
            List of dicts with polygon info for rendering
        """
        width, height = frame_size
        overlays = []
        
        for zone in self.exclusion_zones:
            scaled_poly = [
                [int(p[0] * width), int(p[1] * height)]
                for p in zone.polygon
            ]
            overlays.append({
                'id': zone.id,
                'polygon': scaled_poly,
                'color': (128, 128, 128),  # Gray for exclusion
                'description': zone.description or f"Exclusion: {zone.id}"
            })
        
        return overlays
    
    def reset(self):
        """Reset filter state (no-op, but required by LoopResetManager interface)."""
        pass  # ROIFilter is stateless


# ============================================================
# Module-level singleton
# ============================================================

_roi_filter: Optional[ROIFilter] = None


def get_roi_filter(config_path: Optional[str] = None) -> ROIFilter:
    """Get or create singleton ROIFilter instance."""
    global _roi_filter
    if _roi_filter is None:
        _roi_filter = ROIFilter(config_path=config_path)
    return _roi_filter


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Create test filter with corner exclusion zones
    zones = [
        ExclusionZone(
            id="corner_left",
            polygon=[[0.0, 0.7], [0.15, 0.7], [0.15, 1.0], [0.0, 1.0]],
            target_classes=["forklift"],
            description="Left corner - static cargo"
        ),
        ExclusionZone(
            id="corner_right",
            polygon=[[0.85, 0.7], [1.0, 0.7], [1.0, 1.0], [0.85, 1.0]],
            target_classes=["forklift"],
            description="Right corner - static cargo"
        )
    ]
    
    filter = ROIFilter(exclusion_zones=zones)
    
    # Create test detections
    person = TrackedObject(
        track_id=1,
        class_name="person",
        bbox=[500, 300, 600, 500],
        center=(550, 400),
        confidence=0.9
    )
    
    forklift_valid = TrackedObject(
        track_id=10,
        class_name="forklift",
        bbox=[800, 400, 1000, 600],  # Center of frame
        center=(900, 500),
        confidence=0.8
    )
    
    forklift_excluded = TrackedObject(
        track_id=11,
        class_name="forklift",
        bbox=[50, 800, 250, 1000],  # Left corner (should be excluded)
        center=(150, 900),
        confidence=0.75
    )
    
    detections = [person, forklift_valid, forklift_excluded]
    frame_size = (1920, 1080)
    
    # Test filtering
    valid = filter.filter(detections, frame_size)
    
    print(f"Input: {len(detections)} detections")
    print(f"Output: {len(valid)} detections")
    print("\nValid detections:")
    for det in valid:
        print(f"  - {det.class_name} #{det.track_id}")
    
    print("\nROIFilter test complete!")
