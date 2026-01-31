#!/usr/bin/env python3
"""
Tests for EventStore - Safety MVP Component
"""

import pytest
import os
import time
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_store import (
    EventStore, SafetyEvent, EventType, Severity, get_event_store
)


class TestEventStore:
    """Test cases for EventStore class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def store(self, temp_dir):
        """Create a test EventStore instance."""
        db_path = os.path.join(temp_dir, "test_events.db")
        snapshot_dir = os.path.join(temp_dir, "snapshots")
        store = EventStore(
            db_path=db_path,
            snapshot_dir=snapshot_dir,
            flush_interval=0.1  # Fast flush for testing
        )
        yield store
        store.shutdown()
    
    def test_create_event(self):
        """Test SafetyEvent creation."""
        event = SafetyEvent(
            event_type=EventType.RESTRICTED_ZONE_PRESENCE,
            zone_id="forklift_lane",
            severity=Severity.HIGH,
            track_id=42,
            duration_seconds=6.5
        )
        
        assert event.event_type == EventType.RESTRICTED_ZONE_PRESENCE
        assert event.zone_id == "forklift_lane"
        assert event.severity == Severity.HIGH
        assert event.track_id == 42
        assert event.duration_seconds == 6.5
        assert event.acknowledged is False
        assert event.timestamp is not None
    
    def test_event_to_dict(self):
        """Test SafetyEvent serialization."""
        event = SafetyEvent(
            event_type=EventType.PEDESTRIAN_IN_VEHICLE_LANE,
            zone_id="vehicle_lane",
            severity=Severity.HIGH,
            track_id=1,
            duration_seconds=3.0,
            metadata={"extra": "data"}
        )
        
        data = event.to_dict()
        
        assert data["event_type"] == "PEDESTRIAN_IN_VEHICLE_LANE"
        assert data["severity"] == "HIGH"
        assert data["zone_id"] == "vehicle_lane"
        assert data["metadata"]["extra"] == "data"
    
    def test_log_event_non_blocking(self, store):
        """Test that log_event returns immediately."""
        event = SafetyEvent(
            event_type=EventType.AISLE_OCCUPANCY_TOO_LONG,
            zone_id="main_aisle",
            severity=Severity.MEDIUM,
            track_id=5,
            duration_seconds=35.0
        )
        
        start = time.time()
        store.log_event(event)
        elapsed = time.time() - start
        
        # Should return almost immediately
        assert elapsed < 0.1
    
    def test_log_and_query_event(self, store):
        """Test logging and querying an event."""
        event = SafetyEvent(
            event_type=EventType.RESTRICTED_ZONE_PRESENCE,
            zone_id="loading_dock",
            severity=Severity.MEDIUM,
            track_id=10,
            duration_seconds=12.0
        )
        
        store.log_event(event)
        
        # Wait for background writer
        time.sleep(0.5)
        
        # Query events
        events = store.query_events(limit=10)
        
        assert len(events) >= 1
        found = any(e.zone_id == "loading_dock" for e in events)
        assert found
    
    def test_log_with_snapshot(self, store):
        """Test logging event with frame snapshot."""
        event = SafetyEvent(
            event_type=EventType.RESTRICTED_ZONE_PRESENCE,
            zone_id="test_zone",
            severity=Severity.HIGH,
            track_id=1,
            duration_seconds=5.0
        )
        
        # Create fake frame
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_frame[100:200, 100:200] = [255, 0, 0]  # Blue square
        
        store.log_event(event, fake_frame)
        
        # Wait for background writer
        time.sleep(0.5)
        
        # Query and check snapshot path
        events = store.query_events(zone_id="test_zone")
        assert len(events) >= 1
        
        latest = events[0]
        assert latest.snapshot_path is not None
        assert os.path.exists(latest.snapshot_path)
    
    def test_query_filters(self, store):
        """Test query filters."""
        # Create events with different severities
        events = [
            SafetyEvent(
                event_type=EventType.RESTRICTED_ZONE_PRESENCE,
                zone_id="zone_a",
                severity=Severity.HIGH,
                track_id=1,
                duration_seconds=5.0
            ),
            SafetyEvent(
                event_type=EventType.AISLE_OCCUPANCY_TOO_LONG,
                zone_id="zone_b",
                severity=Severity.LOW,
                track_id=2,
                duration_seconds=40.0
            ),
        ]
        
        for event in events:
            store.log_event(event)
        
        time.sleep(0.5)
        
        # Filter by severity
        high_events = store.query_events(severity="HIGH")
        assert all(e.severity == Severity.HIGH for e in high_events)
        
        # Filter by zone
        zone_a_events = store.query_events(zone_id="zone_a")
        assert all(e.zone_id == "zone_a" for e in zone_a_events)
    
    def test_acknowledge_event(self, store):
        """Test acknowledging an event."""
        event = SafetyEvent(
            event_type=EventType.RESTRICTED_ZONE_PRESENCE,
            zone_id="ack_test",
            severity=Severity.HIGH,
            track_id=1,
            duration_seconds=5.0
        )
        
        store.log_event(event)
        time.sleep(0.5)
        
        # Get event ID
        events = store.query_events(zone_id="ack_test")
        assert len(events) >= 1
        event_id = events[0].id
        
        # Acknowledge
        result = store.acknowledge(event_id)
        assert result is True
        
        # Verify
        updated = store.get_event(event_id)
        assert updated.acknowledged is True
        assert updated.acknowledge_time is not None
    
    def test_daily_summary(self, store):
        """Test daily summary aggregation."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Log multiple events
        for i in range(3):
            event = SafetyEvent(
                event_type=EventType.RESTRICTED_ZONE_PRESENCE,
                zone_id=f"zone_{i}",
                severity=Severity.HIGH if i == 0 else Severity.MEDIUM,
                track_id=i,
                duration_seconds=5.0
            )
            store.log_event(event)
        
        time.sleep(0.5)
        
        # Get summary
        summary = store.get_daily_summary(today)
        
        assert summary["total_events"] >= 3
        assert summary["high_severity_count"] >= 1
    
    def test_export_csv(self, store, temp_dir):
        """Test CSV export."""
        event = SafetyEvent(
            event_type=EventType.RESTRICTED_ZONE_PRESENCE,
            zone_id="export_test",
            severity=Severity.HIGH,
            track_id=1,
            duration_seconds=5.0
        )
        store.log_event(event)
        time.sleep(0.5)
        
        # Export
        csv_path = os.path.join(temp_dir, "export.csv")
        start = (datetime.now() - timedelta(hours=1)).isoformat()
        end = (datetime.now() + timedelta(hours=1)).isoformat()
        
        store.export_csv(start, end, csv_path)
        
        assert os.path.exists(csv_path)
        
        # Verify content
        with open(csv_path, 'r') as f:
            content = f.read()
        
        assert "export_test" in content
        assert "RESTRICTED_ZONE_PRESENCE" in content
    
    def test_unacknowledged_count(self, store):
        """Test getting unacknowledged count."""
        # Log some events
        for i in range(5):
            event = SafetyEvent(
                event_type=EventType.RESTRICTED_ZONE_PRESENCE,
                zone_id=f"count_zone_{i}",
                severity=Severity.MEDIUM,
                track_id=i,
                duration_seconds=5.0
            )
            store.log_event(event)
        
        time.sleep(0.5)
        
        count = store.get_unacknowledged_count()
        assert count >= 5


class TestEventTypes:
    """Test event type enum values."""
    
    def test_event_types(self):
        """Verify all event types exist."""
        assert EventType.RESTRICTED_ZONE_PRESENCE.value == "RESTRICTED_ZONE_PRESENCE"
        assert EventType.PEDESTRIAN_IN_VEHICLE_LANE.value == "PEDESTRIAN_IN_VEHICLE_LANE"
        assert EventType.AISLE_OCCUPANCY_TOO_LONG.value == "AISLE_OCCUPANCY_TOO_LONG"
    
    def test_severity_levels(self):
        """Verify all severity levels exist."""
        assert Severity.LOW.value == "LOW"
        assert Severity.MEDIUM.value == "MEDIUM"
        assert Severity.HIGH.value == "HIGH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
