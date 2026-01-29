#!/usr/bin/env python3
"""
ZoneManager - Polygon Zone Definition and Violation Detection
=============================================================

Safety MVP Component for rule-based zone monitoring.

Features:
- Normalized coordinates (0-1) for resolution independence
- Polygon-based zone definitions
- Per-zone rule configuration
- Dwell time tracking per track_id per zone
- Fast violation checking (~1ms)

Usage:
    manager = ZoneManager("configs/zones.yaml")
    violations = manager.check_violations(detections, (640, 480))
"""

import yaml
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path
import threading
import numpy as np


# ============================================================
# Enums
# ============================================================

class ZoneType(str, Enum):
    """Zone classification types"""
    RESTRICTED = "RESTRICTED"
    VEHICLE_LANE = "VEHICLE_LANE"
    FLOW = "FLOW"
    SAFE = "SAFE"


class RuleType(str, Enum):
    """Rule types for zone violations"""
    PERSON_PRESENCE = "PERSON_PRESENCE"
    OCCUPANCY = "OCCUPANCY"
    SPEED_LIMIT = "SPEED_LIMIT"


class Severity(str, Enum):
    """Violation severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ZoneRule:
    """Rule definition for a zone"""
    rule_type: RuleType
    max_dwell_seconds: float
    severity: Severity
    target_classes: List[str] = field(default_factory=lambda: ["person"])


@dataclass
class Zone:
    """Zone definition with polygon and rules"""
    id: str
    zone_type: ZoneType
    polygon: List[List[float]]  # Normalized coords [[x1,y1], [x2,y2], ...]
    rules: List[ZoneRule]
    enabled: bool = True
    display_name: Optional[str] = None
    color: Tuple[int, int, int] = (0, 255, 255)  # BGR yellow default
    
    def get_scaled_polygon(self, frame_size: Tuple[int, int]) -> np.ndarray:
        """Scale normalized polygon to frame size."""
        width, height = frame_size
        scaled = []
        for point in self.polygon:
            scaled.append([int(point[0] * width), int(point[1] * height)])
        return np.array(scaled, dtype=np.int32)


@dataclass 
class ZoneViolation:
    """Detected zone violation"""
    zone_id: str
    zone_type: ZoneType
    rule_type: RuleType
    severity: Severity
    track_id: int
    dwell_seconds: float
    bbox: List[int]
    timestamp: str
    
    def should_trigger(self, max_dwell: float) -> bool:
        """Check if dwell time exceeds threshold."""
        return self.dwell_seconds >= max_dwell


@dataclass
class TrackedObject:
    """Object being tracked in zones"""
    track_id: int
    class_name: str
    bbox: List[int]
    center: Tuple[int, int]
    confidence: float


# ============================================================
# ZoneManager Implementation
# ============================================================

class ZoneManager:
    """
    Manages zone definitions and violation detection.
    
    All coordinates in zones.yaml are normalized (0-1) for
    resolution independence. Scaling happens at runtime.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ZoneManager.
        
        Args:
            config_path: Path to zones.yaml, or None for empty zones
        """
        self._lock = threading.Lock()  # Thread-safe lock for hot-swap
        self.zones: Dict[str, Zone] = {}
        self.dwell_tracker: Dict[str, Dict[int, float]] = {}  # zone_id -> {track_id: first_seen_time}
        self.last_seen: Dict[str, Dict[int, float]] = {}  # zone_id -> {track_id: last_seen_time}
        
        if config_path:
            self.load_config(config_path)
    
    def update_zones(self, new_zones: List[dict]):
        """
        Thread-safe hot swap of zones from VLM calibration.
        
        Args:
            new_zones: List of zone dicts with id, type, polygon, etc.
        """
        with self._lock:
            old_count = len(self.zones)
            self.zones = {z.get('id', f'zone_{i}'): self._parse_zone(z) 
                          for i, z in enumerate(new_zones)}
            # Reset dwell tracking for new zones
            self.dwell_tracker = {zid: {} for zid in self.zones}
            self.last_seen = {zid: {} for zid in self.zones}
            print(f"🔄 Zones Hot-Swapped: {old_count} -> {len(self.zones)} zones")
    
    def load_config(self, config_path: str):
        """Load zones from YAML config file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Zone config not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.zones.clear()
        self.dwell_tracker.clear()
        self.last_seen.clear()
        
        for zone_data in config.get('zones', []):
            zone = self._parse_zone(zone_data)
            self.zones[zone.id] = zone
            self.dwell_tracker[zone.id] = {}
            self.last_seen[zone.id] = {}
    
    def _parse_zone(self, data: dict) -> Zone:
        """Parse zone from config dict."""
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(ZoneRule(
                rule_type=RuleType(rule_data['type']),
                max_dwell_seconds=rule_data.get('max_dwell_seconds', 5.0),
                severity=Severity(rule_data.get('severity', 'MEDIUM')),
                target_classes=rule_data.get('target_classes', ['person'])
            ))
        
        color = data.get('color', [0, 255, 255])
        if isinstance(color, list):
            color = tuple(color)
        
        return Zone(
            id=data['id'],
            zone_type=ZoneType(data.get('type', 'RESTRICTED')),
            polygon=data['polygon'],
            rules=rules,
            enabled=data.get('enabled', True),
            display_name=data.get('display_name'),
            color=color
        )
    
    def save_config(self, config_path: str):
        """Save current zones to YAML config file."""
        config = {'zones': []}
        
        for zone in self.zones.values():
            zone_data = {
                'id': zone.id,
                'type': zone.zone_type.value,
                'polygon': zone.polygon,
                'enabled': zone.enabled,
                'rules': []
            }
            
            if zone.display_name:
                zone_data['display_name'] = zone.display_name
            if zone.color != (0, 255, 255):
                zone_data['color'] = list(zone.color)
            
            for rule in zone.rules:
                zone_data['rules'].append({
                    'type': rule.rule_type.value,
                    'max_dwell_seconds': rule.max_dwell_seconds,
                    'severity': rule.severity.value,
                    'target_classes': rule.target_classes
                })
            
            config['zones'].append(zone_data)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def check_violations(
        self,
        detections: List[TrackedObject],
        frame_size: Tuple[int, int],
        current_time: Optional[float] = None
    ) -> List[ZoneViolation]:
        """
        Check all detections against all zones.
        
        Args:
            detections: List of TrackedObject from YOLO+tracker
            frame_size: (width, height) of current frame
            current_time: Optional timestamp, defaults to time.time()
            
        Returns:
            List of ZoneViolation for violations that exceed thresholds
        """
        if current_time is None:
            current_time = time.time()
        
        violations = []
        
        with self._lock:  # Read lock for thread-safe access during hot-swap
            active_tracks: Dict[str, set] = {zone_id: set() for zone_id in self.zones}
        
        for det in detections:
            for zone_id, zone in self.zones.items():
                if not zone.enabled:
                    continue
                
                # Check if detection center is inside zone polygon
                scaled_poly = zone.get_scaled_polygon(frame_size)
                if self._point_in_polygon(det.center, scaled_poly):
                    active_tracks[zone_id].add(det.track_id)
                    
                    # Update dwell tracking
                    if det.track_id not in self.dwell_tracker[zone_id]:
                        self.dwell_tracker[zone_id][det.track_id] = current_time
                    
                    self.last_seen[zone_id][det.track_id] = current_time
                    
                    # Calculate dwell time
                    first_seen = self.dwell_tracker[zone_id][det.track_id]
                    dwell_seconds = current_time - first_seen
                    
                    # Check each rule
                    for rule in zone.rules:
                        if det.class_name not in rule.target_classes:
                            continue
                        
                        # P0 FIX: HIGH severity RESTRICTED zones trigger INSTANTLY
                        # No dwell time required - immediate safety violation
                        should_trigger = False
                        if (rule.severity == Severity.HIGH and 
                            zone.zone_type == ZoneType.RESTRICTED):
                            # Instant alert for high-risk zones
                            should_trigger = True
                        elif dwell_seconds >= rule.max_dwell_seconds:
                            # Standard dwell-time based trigger
                            should_trigger = True
                        
                        if should_trigger:
                            from datetime import datetime
                            violations.append(ZoneViolation(
                                zone_id=zone_id,
                                zone_type=zone.zone_type,
                                rule_type=rule.rule_type,
                                severity=rule.severity,
                                track_id=det.track_id,
                                dwell_seconds=dwell_seconds,
                                bbox=det.bbox,
                                timestamp=datetime.now().isoformat()
                            ))
        
        # Cleanup stale tracks (not seen for >2 seconds)
        self._cleanup_stale_tracks(active_tracks, current_time, stale_threshold=2.0)
        
        return violations
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        
        Args:
            point: (x, y) coordinates
            polygon: numpy array of polygon vertices
            
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
    
    def _cleanup_stale_tracks(
        self,
        active_tracks: Dict[str, set],
        current_time: float,
        stale_threshold: float
    ):
        """Remove tracks not seen recently."""
        for zone_id in self.zones:
            stale_ids = []
            for track_id, last_time in self.last_seen.get(zone_id, {}).items():
                if track_id not in active_tracks[zone_id]:
                    if current_time - last_time > stale_threshold:
                        stale_ids.append(track_id)
            
            for track_id in stale_ids:
                self.dwell_tracker[zone_id].pop(track_id, None)
                self.last_seen[zone_id].pop(track_id, None)
    
    def get_dwell_time(self, zone_id: str, track_id: int) -> float:
        """Get current dwell time for a track in a zone."""
        if zone_id not in self.dwell_tracker:
            return 0.0
        if track_id not in self.dwell_tracker[zone_id]:
            return 0.0
        return time.time() - self.dwell_tracker[zone_id][track_id]
    
    def reset_dwell(self, zone_id: str, track_id: Optional[int] = None):
        """Reset dwell tracking for a zone or specific track."""
        if zone_id not in self.dwell_tracker:
            return
        
        if track_id is None:
            self.dwell_tracker[zone_id].clear()
            self.last_seen[zone_id].clear()
        else:
            self.dwell_tracker[zone_id].pop(track_id, None)
            self.last_seen[zone_id].pop(track_id, None)
    
    def reset_dwell_all(self):
        """
        Reset dwell tracking for all zones.
        
        Called by LoopResetManager when video loops to prevent
        stale dwell times from persisting across loop iterations.
        """
        for zone_id in self.zones:
            if zone_id in self.dwell_tracker:
                self.dwell_tracker[zone_id].clear()
            if zone_id in self.last_seen:
                self.last_seen[zone_id].clear()
    
    def reset(self):
        """
        Reset all state (LoopResetManager interface).
        
        Alias for reset_dwell_all() to match Resettable protocol.
        """
        self.reset_dwell_all()
    
    def get_overlay_polygons(self, frame_size: Tuple[int, int]) -> List[dict]:
        """
        Get scaled polygons for dashboard overlay.
        
        Args:
            frame_size: (width, height) of frame
            
        Returns:
            List of dicts with polygon info for rendering
        """
        overlays = []
        for zone in self.zones.values():
            if not zone.enabled:
                continue
            
            scaled_poly = zone.get_scaled_polygon(frame_size)
            overlays.append({
                'id': zone.id,
                'type': zone.zone_type.value,
                'polygon': scaled_poly.tolist(),
                'color': zone.color,
                'display_name': zone.display_name or zone.id
            })
        
        return overlays
    
    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get zone by ID."""
        return self.zones.get(zone_id)
    
    def get_all_zones(self) -> List[Zone]:
        """Get all zones."""
        return list(self.zones.values())
    
    def add_zone(self, zone: Zone):
        """Add or update a zone."""
        self.zones[zone.id] = zone
        self.dwell_tracker[zone.id] = {}
        self.last_seen[zone.id] = {}
    
    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.dwell_tracker.pop(zone_id, None)
            self.last_seen.pop(zone_id, None)
            return True
        return False
    
    def enable_zone(self, zone_id: str, enabled: bool = True):
        """Enable or disable a zone."""
        if zone_id in self.zones:
            self.zones[zone_id].enabled = enabled


# ============================================================
# Helper Functions
# ============================================================

def detections_from_yolo(results, target_classes: List[str] = None) -> List[TrackedObject]:
    """
    Convert YOLO results to TrackedObject list.
    
    Args:
        results: YOLO model output
        target_classes: Optional list of class names to include
        
    Returns:
        List of TrackedObject
    """
    if target_classes is None:
        target_classes = ['person']
    
    # COCO class names
    COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                  'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                  'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                  'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                  'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    objects = []
    
    if results is None or len(results) == 0:
        return objects
    
    boxes = results[0].boxes
    if boxes is None:
        return objects
    
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id >= len(COCO_NAMES):
            continue
        
        class_name = COCO_NAMES[cls_id]
        if class_name not in target_classes:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        conf = float(box.conf[0])
        
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        objects.append(TrackedObject(
            track_id=track_id,
            class_name=class_name,
            bbox=[x1, y1, x2, y2],
            center=center,
            confidence=conf
        ))
    
    return objects


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Create test zone manager
    manager = ZoneManager()
    
    # Add test zones manually
    test_zone = Zone(
        id="test_restricted",
        zone_type=ZoneType.RESTRICTED,
        polygon=[[0.2, 0.6], [0.8, 0.6], [0.8, 1.0], [0.2, 1.0]],
        rules=[ZoneRule(
            rule_type=RuleType.PERSON_PRESENCE,
            max_dwell_seconds=3.0,
            severity=Severity.HIGH
        )]
    )
    manager.add_zone(test_zone)
    
    # Test with fake detection
    det = TrackedObject(
        track_id=1,
        class_name="person",
        bbox=[200, 350, 300, 450],
        center=(250, 400),
        confidence=0.9
    )
    
    frame_size = (640, 480)
    
    # First check - should not trigger (just entered)
    violations = manager.check_violations([det], frame_size)
    print(f"Check 1: {len(violations)} violations")
    
    # Simulate time passing
    import time
    time.sleep(0.1)
    
    # Force dwell time for testing
    manager.dwell_tracker["test_restricted"][1] = time.time() - 5.0
    
    # Check again - should trigger
    violations = manager.check_violations([det], frame_size)
    print(f"Check 2: {len(violations)} violations")
    for v in violations:
        print(f"  - {v.zone_id}: {v.severity.value} ({v.dwell_seconds:.1f}s)")
    
    # Test overlay generation
    overlays = manager.get_overlay_polygons(frame_size)
    print(f"\nOverlays: {len(overlays)}")
    for o in overlays:
        print(f"  - {o['id']}: {o['type']}")
    
    print("\nZoneManager test complete!")
