#!/usr/bin/env python3
"""
Tests for ZoneManager - Safety MVP Component
"""

import pytest
import os
import tempfile
import shutil
import time

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.zone_manager import (
    ZoneManager, Zone, ZoneRule, ZoneType, RuleType, Severity,
    TrackedObject, ZoneViolation, detections_from_yolo
)


class TestZoneManager:
    """Test cases for ZoneManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample zones.yaml config file."""
        config_path = os.path.join(temp_dir, "zones.yaml")
        config_content = """
zones:
  - id: test_restricted
    type: RESTRICTED
    display_name: "Test Restricted Zone"
    color: [0, 0, 255]
    polygon:
      - [0.2, 0.6]
      - [0.8, 0.6]
      - [0.8, 1.0]
      - [0.2, 1.0]
    rules:
      - type: PERSON_PRESENCE
        max_dwell_seconds: 3
        severity: HIGH
        target_classes: ["person"]
    enabled: true

  - id: test_flow
    type: FLOW
    polygon:
      - [0.3, 0.1]
      - [0.7, 0.1]
      - [0.7, 0.5]
      - [0.3, 0.5]
    rules:
      - type: OCCUPANCY
        max_dwell_seconds: 30
        severity: MEDIUM
    enabled: true
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path
    
    @pytest.fixture
    def manager(self, sample_config):
        """Create a ZoneManager with sample config."""
        return ZoneManager(sample_config)
    
    def test_load_config(self, manager):
        """Test loading zones from config."""
        assert len(manager.zones) == 2
        assert "test_restricted" in manager.zones
        assert "test_flow" in manager.zones
    
    def test_zone_properties(self, manager):
        """Test zone properties are loaded correctly."""
        zone = manager.get_zone("test_restricted")
        
        assert zone is not None
        assert zone.zone_type == ZoneType.RESTRICTED
        assert zone.display_name == "Test Restricted Zone"
        assert zone.color == (0, 0, 255)
        assert len(zone.rules) == 1
        assert zone.rules[0].max_dwell_seconds == 3
        assert zone.rules[0].severity == Severity.HIGH
    
    def test_scaled_polygon(self, manager):
        """Test polygon scaling."""
        zone = manager.get_zone("test_restricted")
        frame_size = (640, 480)
        
        scaled = zone.get_scaled_polygon(frame_size)
        
        # Check first point: [0.2, 0.6] * [640, 480] = [128, 288]
        assert scaled[0][0] == 128
        assert scaled[0][1] == 288
    
    def test_point_in_polygon(self, manager):
        """Test point-in-polygon detection."""
        zone = manager.get_zone("test_restricted")
        frame_size = (640, 480)
        scaled = zone.get_scaled_polygon(frame_size)
        
        # Point inside zone (center of lower portion)
        inside = manager._point_in_polygon((320, 400), scaled)
        assert inside is True
        
        # Point outside zone (top of frame)
        outside = manager._point_in_polygon((320, 100), scaled)
        assert outside is False
    
    def test_check_violations_no_dwell(self, manager):
        """Test that violations don't trigger immediately."""
        det = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),  # Inside test_restricted zone
            confidence=0.9
        )
        
        frame_size = (640, 480)
        violations = manager.check_violations([det], frame_size)
        
        # Should not trigger immediately (dwell time = 0)
        assert len(violations) == 0
    
    def test_check_violations_with_dwell(self, manager):
        """Test that violations trigger after dwell time."""
        det = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),  # Inside test_restricted zone
            confidence=0.9
        )
        
        frame_size = (640, 480)
        
        # First detection
        manager.check_violations([det], frame_size)
        
        # Simulate time passing (>3 seconds for test_restricted)
        manager.dwell_tracker["test_restricted"][1] = time.time() - 5.0
        
        # Check again
        violations = manager.check_violations([det], frame_size)
        
        assert len(violations) >= 1
        assert violations[0].zone_id == "test_restricted"
        assert violations[0].severity == Severity.HIGH
        assert violations[0].dwell_seconds >= 3.0
    
    def test_instant_alert_high_severity_restricted(self, manager):
        """Test that HIGH severity RESTRICTED zones trigger immediately (P0 fix)."""
        det = TrackedObject(
            track_id=99,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),  # Inside test_restricted zone
            confidence=0.9
        )
        
        frame_size = (640, 480)
        
        # First detection should trigger IMMEDIATELY for HIGH severity RESTRICTED
        # No dwell time required!
        violations = manager.check_violations([det], frame_size)
        
        # P0 FIX: Should trigger on first frame
        assert len(violations) >= 1, "HIGH severity RESTRICTED zone should trigger immediately"
        assert violations[0].zone_id == "test_restricted"
        assert violations[0].severity == Severity.HIGH
        # Dwell time should be very small (first frame)
        assert violations[0].dwell_seconds < 1.0
    
    def test_multiple_detections(self, manager):
        """Test handling multiple detections."""
        detections = [
            TrackedObject(
                track_id=1,
                class_name="person",
                bbox=[250, 350, 350, 450],
                center=(300, 400),
                confidence=0.9
            ),
            TrackedObject(
                track_id=2,
                class_name="person",
                bbox=[350, 150, 450, 250],
                center=(400, 200),  # Inside test_flow zone
                confidence=0.85
            ),
        ]
        
        frame_size = (640, 480)
        manager.check_violations(detections, frame_size)
        
        # Verify both tracks are being tracked
        assert 1 in manager.dwell_tracker["test_restricted"]
        assert 2 in manager.dwell_tracker["test_flow"]
    
    def test_stale_track_cleanup(self, manager):
        """Test that stale tracks are cleaned up."""
        det = TrackedObject(
            track_id=99,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),
            confidence=0.9
        )
        
        frame_size = (640, 480)
        
        # First detection
        manager.check_violations([det], frame_size)
        assert 99 in manager.dwell_tracker["test_restricted"]
        
        # Set last seen to past (stale)
        manager.last_seen["test_restricted"][99] = time.time() - 10.0
        
        # Check with empty detections
        manager.check_violations([], frame_size)
        
        # Track should be cleaned up
        assert 99 not in manager.dwell_tracker["test_restricted"]
    
    def test_get_dwell_time(self, manager):
        """Test getting dwell time for a track."""
        det = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),
            confidence=0.9
        )
        
        frame_size = (640, 480)
        manager.check_violations([det], frame_size)
        
        # Small delay
        time.sleep(0.1)
        
        dwell = manager.get_dwell_time("test_restricted", 1)
        assert dwell >= 0.1
    
    def test_reset_dwell(self, manager):
        """Test resetting dwell time."""
        det = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),
            confidence=0.9
        )
        
        frame_size = (640, 480)
        manager.check_violations([det], frame_size)
        
        # Reset
        manager.reset_dwell("test_restricted", 1)
        
        dwell = manager.get_dwell_time("test_restricted", 1)
        assert dwell == 0.0
    
    def test_get_overlay_polygons(self, manager):
        """Test overlay polygon generation."""
        frame_size = (640, 480)
        overlays = manager.get_overlay_polygons(frame_size)
        
        assert len(overlays) == 2
        
        # Find test_restricted overlay
        restricted = next(o for o in overlays if o['id'] == 'test_restricted')
        assert restricted['type'] == 'RESTRICTED'
        assert restricted['color'] == (0, 0, 255)
        assert len(restricted['polygon']) == 4
    
    def test_add_zone(self, manager):
        """Test adding a new zone."""
        new_zone = Zone(
            id="new_zone",
            zone_type=ZoneType.SAFE,
            polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
            rules=[]
        )
        
        manager.add_zone(new_zone)
        
        assert "new_zone" in manager.zones
        assert manager.get_zone("new_zone").zone_type == ZoneType.SAFE
    
    def test_remove_zone(self, manager):
        """Test removing a zone."""
        result = manager.remove_zone("test_flow")
        
        assert result is True
        assert "test_flow" not in manager.zones
        
        # Try removing non-existent zone
        result = manager.remove_zone("non_existent")
        assert result is False
    
    def test_enable_disable_zone(self, manager):
        """Test enabling/disabling zones."""
        manager.enable_zone("test_restricted", enabled=False)
        
        assert manager.get_zone("test_restricted").enabled is False
        
        # Disabled zone should not trigger violations
        det = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=[250, 350, 350, 450],
            center=(300, 400),
            confidence=0.9
        )
        
        frame_size = (640, 480)
        manager.dwell_tracker["test_restricted"][1] = time.time() - 10.0
        violations = manager.check_violations([det], frame_size)
        
        # Should have no violations from disabled zone
        restricted_violations = [v for v in violations if v.zone_id == "test_restricted"]
        assert len(restricted_violations) == 0
    
    def test_save_config(self, manager, temp_dir):
        """Test saving config to file."""
        output_path = os.path.join(temp_dir, "output_zones.yaml")
        manager.save_config(output_path)
        
        assert os.path.exists(output_path)
        
        # Load and verify
        new_manager = ZoneManager(output_path)
        assert len(new_manager.zones) == 2
        assert "test_restricted" in new_manager.zones


class TestTrackedObject:
    """Test TrackedObject class."""
    
    def test_creation(self):
        """Test TrackedObject creation."""
        obj = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=[100, 200, 300, 400],
            center=(200, 300),
            confidence=0.95
        )
        
        assert obj.track_id == 1
        assert obj.class_name == "person"
        assert obj.center == (200, 300)


class TestZoneViolation:
    """Test ZoneViolation class."""
    
    def test_should_trigger(self):
        """Test violation trigger threshold."""
        from datetime import datetime
        
        violation = ZoneViolation(
            zone_id="test",
            zone_type=ZoneType.RESTRICTED,
            rule_type=RuleType.PERSON_PRESENCE,
            severity=Severity.HIGH,
            track_id=1,
            dwell_seconds=5.0,
            bbox=[100, 200, 300, 400],
            timestamp=datetime.now().isoformat()
        )
        
        assert violation.should_trigger(3.0) is True
        assert violation.should_trigger(10.0) is False


class TestZoneTypes:
    """Test zone type enums."""
    
    def test_zone_types(self):
        """Verify zone types."""
        assert ZoneType.RESTRICTED.value == "RESTRICTED"
        assert ZoneType.VEHICLE_LANE.value == "VEHICLE_LANE"
        assert ZoneType.FLOW.value == "FLOW"
        assert ZoneType.SAFE.value == "SAFE"
    
    def test_rule_types(self):
        """Verify rule types."""
        assert RuleType.PERSON_PRESENCE.value == "PERSON_PRESENCE"
        assert RuleType.OCCUPANCY.value == "OCCUPANCY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
