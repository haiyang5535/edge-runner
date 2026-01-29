#!/usr/bin/env python3
"""
AlertManager - Alert Triggering and Management
==============================================

Safety MVP Component for visual/audio alerts with acknowledge/silence.

Features:
- Visual flash indicators (overlay on frame)
- Optional audio alerts via system speaker
- Acknowledge individual events
- Silence zones temporarily
- Alert state tracking for dashboard

Usage:
    alert_mgr = AlertManager(event_store)
    alert_mgr.trigger_alert(event)  # Returns False if silenced
    alert_mgr.acknowledge(event_id)
    alert_mgr.silence_zone("forklift_lane", minutes=5)
"""

import threading
import time
import os
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

from .event_store import EventStore, SafetyEvent, Severity


# ============================================================
# Data Classes
# ============================================================

@dataclass
class AlertState:
    """Current alert state for dashboard display"""
    active: bool = False
    severity: Optional[Severity] = None
    zone_id: Optional[str] = None
    event_id: Optional[int] = None
    message: str = ""
    timestamp: Optional[str] = None
    flash_until: Optional[float] = None  # time.time() when flash should stop


@dataclass
class ActiveAlert:
    """Represents an active unacknowledged alert"""
    event: SafetyEvent
    triggered_at: float
    acknowledged: bool = False


# ============================================================
# AlertManager Implementation
# ============================================================

class AlertManager:
    """
    Manages alert triggering, acknowledgement, and zone silencing.
    
    Alert Modes by Severity:
    - HIGH: Red flash + banner + 3x beep (optional)
    - MEDIUM: Orange pulse + 1x beep (optional)
    - LOW: Yellow highlight only
    """
    
    def __init__(
        self,
        event_store: EventStore,
        audio_enabled: bool = False,
        flash_duration: float = 3.0,
        cooldown_seconds: float = 10.0
    ):
        """
        Initialize AlertManager.
        
        Args:
            event_store: EventStore instance for persistence
            audio_enabled: Whether to play audio alerts
            flash_duration: How long visual flash lasts (seconds)
            cooldown_seconds: Minimum time between alerts for same zone+track
        """
        self.event_store = event_store
        self.audio_enabled = audio_enabled
        self.flash_duration = flash_duration
        self.cooldown_seconds = cooldown_seconds
        
        # Zone silencing: zone_id -> silence_end_time
        self.silenced_until: Dict[str, datetime] = {}
        self.silence_lock = threading.Lock()
        
        # Active alerts for dashboard
        self.active_alerts: Dict[int, ActiveAlert] = {}  # event_id -> ActiveAlert
        self.alerts_lock = threading.Lock()
        
        # Current visual state
        self.current_state = AlertState()
        self.state_lock = threading.Lock()
        
        # Cooldown tracking: (zone_id, track_id) -> last_alert_time
        self.cooldown_tracker: Dict[tuple, float] = {}
        
        # Alert callbacks for external integrations
        self.on_alert_callbacks: List[Callable[[SafetyEvent], None]] = []
    
    def trigger_alert(self, event: SafetyEvent, frame=None) -> bool:
        """
        Trigger an alert for a safety event.
        
        Args:
            event: SafetyEvent to alert on
            frame: Optional frame for snapshot
            
        Returns:
            True if alert was triggered, False if silenced/cooldown
        """
        # Check if zone is silenced
        if self._is_silenced(event.zone_id):
            return False
        
        # Check cooldown
        cooldown_key = (event.zone_id, event.track_id)
        current_time = time.time()
        
        if cooldown_key in self.cooldown_tracker:
            last_alert = self.cooldown_tracker[cooldown_key]
            if current_time - last_alert < self.cooldown_seconds:
                return False
        
        self.cooldown_tracker[cooldown_key] = current_time
        
        # Log event to store
        self.event_store.log_event(event, frame)
        
        # Update visual state
        self._set_alert_state(event)
        
        # Play audio if enabled
        if self.audio_enabled:
            self._play_audio(event.severity)
        
        # Track active alert
        with self.alerts_lock:
            if event.id:
                self.active_alerts[event.id] = ActiveAlert(
                    event=event,
                    triggered_at=current_time
                )
        
        # Notify callbacks
        for callback in self.on_alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Alert callback error: {e}")
        
        return True
    
    def _is_silenced(self, zone_id: str) -> bool:
        """Check if zone is currently silenced."""
        with self.silence_lock:
            if zone_id not in self.silenced_until:
                return False
            
            if datetime.now() >= self.silenced_until[zone_id]:
                # Silence expired
                del self.silenced_until[zone_id]
                return False
            
            return True
    
    def _set_alert_state(self, event: SafetyEvent):
        """Update current alert state for dashboard."""
        with self.state_lock:
            self.current_state = AlertState(
                active=True,
                severity=event.severity,
                zone_id=event.zone_id,
                event_id=event.id,
                message=self._get_alert_message(event),
                timestamp=event.timestamp,
                flash_until=time.time() + self.flash_duration
            )
    
    def _get_alert_message(self, event: SafetyEvent) -> str:
        """Generate human-readable alert message."""
        # Use ASCII-safe markers (emojis break cv2.putText rendering -> shows as ??????)
        severity_marker = {
            Severity.HIGH: "[!]",
            Severity.MEDIUM: "[*]",
            Severity.LOW: "[-]"
        }
        marker = severity_marker.get(event.severity, "")
        
        messages = {
            "RESTRICTED_ZONE_PRESENCE": f"{marker} Person in restricted zone: {event.zone_id}",
            "PEDESTRIAN_IN_VEHICLE_LANE": f"{marker} Pedestrian in vehicle lane!",
            "AISLE_OCCUPANCY_TOO_LONG": f"{marker} Aisle blocked: {event.zone_id}"
        }
        
        event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
        return messages.get(event_type, f"{marker} Safety event in {event.zone_id}")
    
    def _play_audio(self, severity: Severity):
        """Play audio alert based on severity (non-blocking)."""
        def play():
            try:
                beeps = {
                    Severity.HIGH: 3,
                    Severity.MEDIUM: 1,
                    Severity.LOW: 0
                }
                num_beeps = beeps.get(severity, 0)
                
                for _ in range(num_beeps):
                    # Try multiple audio methods
                    if os.path.exists("/usr/bin/aplay"):
                        # Linux with ALSA
                        subprocess.run(
                            ["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                            timeout=1,
                            capture_output=True
                        )
                    elif os.path.exists("/usr/bin/paplay"):
                        # PulseAudio
                        subprocess.run(
                            ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                            timeout=1,
                            capture_output=True
                        )
                    else:
                        # Terminal bell as fallback
                        print("\a", end="", flush=True)
                    
                    if num_beeps > 1:
                        time.sleep(0.3)
            except Exception:
                pass  # Audio is optional, don't fail on errors
        
        # Run in background thread
        threading.Thread(target=play, daemon=True).start()
    
    def acknowledge(self, event_id: int) -> bool:
        """
        Acknowledge an event.
        
        Args:
            event_id: ID of event to acknowledge
            
        Returns:
            True if event was acknowledged
        """
        # Update in store
        result = self.event_store.acknowledge(event_id)
        
        # Update active alerts
        with self.alerts_lock:
            if event_id in self.active_alerts:
                self.active_alerts[event_id].acknowledged = True
        
        # Clear current state if this was the active alert
        with self.state_lock:
            if self.current_state.event_id == event_id:
                self.current_state = AlertState()
        
        return result
    
    def silence_zone(self, zone_id: str, minutes: int = 5) -> datetime:
        """
        Silence alerts for a zone temporarily.
        
        Args:
            zone_id: Zone to silence
            minutes: Duration of silence (default 5)
            
        Returns:
            Datetime when silence expires
        """
        silence_end = datetime.now() + timedelta(minutes=minutes)
        
        with self.silence_lock:
            self.silenced_until[zone_id] = silence_end
        
        return silence_end
    
    def unsilence_zone(self, zone_id: str):
        """Remove silence from a zone."""
        with self.silence_lock:
            self.silenced_until.pop(zone_id, None)
    
    def get_silenced_zones(self) -> Dict[str, str]:
        """
        Get currently silenced zones.
        
        Returns:
            Dict of zone_id -> silence_end_time (ISO format)
        """
        with self.silence_lock:
            now = datetime.now()
            # Clean up expired silences
            expired = [z for z, t in self.silenced_until.items() if t <= now]
            for z in expired:
                del self.silenced_until[z]
            
            return {
                zone_id: end_time.isoformat()
                for zone_id, end_time in self.silenced_until.items()
            }
    
    def get_current_state(self) -> AlertState:
        """Get current alert state for dashboard."""
        with self.state_lock:
            # Check if flash has expired
            if self.current_state.flash_until:
                if time.time() > self.current_state.flash_until:
                    self.current_state.active = False
            
            return self.current_state
    
    def get_active_alerts(self) -> List[dict]:
        """Get list of active unacknowledged alerts."""
        with self.alerts_lock:
            alerts = []
            for event_id, alert in self.active_alerts.items():
                if not alert.acknowledged:
                    alerts.append({
                        "event_id": event_id,
                        "event": alert.event.to_dict(),
                        "triggered_at": alert.triggered_at,
                        "age_seconds": time.time() - alert.triggered_at
                    })
            return alerts
    
    def clear_old_alerts(self, max_age_seconds: float = 3600):
        """Remove old acknowledged alerts from memory."""
        with self.alerts_lock:
            current_time = time.time()
            to_remove = []
            
            for event_id, alert in self.active_alerts.items():
                if alert.acknowledged:
                    if current_time - alert.triggered_at > max_age_seconds:
                        to_remove.append(event_id)
            
            for event_id in to_remove:
                del self.active_alerts[event_id]
    
    def register_callback(self, callback: Callable[[SafetyEvent], None]):
        """Register a callback to be called on each alert."""
        self.on_alert_callbacks.append(callback)
    
    def get_alert_overlay(self, frame_size: tuple) -> Optional[dict]:
        """
        Get overlay info for drawing alert on frame.
        
        Args:
            frame_size: (width, height) of frame
            
        Returns:
            Dict with overlay info or None if no active alert
        """
        state = self.get_current_state()
        
        if not state.active:
            return None
        
        colors = {
            Severity.HIGH: (0, 0, 255),      # BGR Red
            Severity.MEDIUM: (0, 165, 255),  # BGR Orange
            Severity.LOW: (0, 255, 255)      # BGR Yellow
        }
        
        return {
            "color": colors.get(state.severity, (255, 255, 255)),
            "message": state.message,
            "zone_id": state.zone_id,
            "severity": state.severity.value if state.severity else "UNKNOWN",
            "flash": state.flash_until and time.time() < state.flash_until
        }
    
    def trigger_ttc_alert(
        self, 
        person_id: int, 
        forklift_id: int, 
        ttc_seconds: float, 
        distance_m: float,
        frame=None
    ) -> bool:
        """
        Trigger a collision risk alert based on TTC calculation.
        
        This integrates TTCCalculator results with the AlertManager
        to produce collision risk alerts when:
        - TTC < 3 seconds AND
        - Objects are approaching (not diverging)
        
        Args:
            person_id: Track ID of person
            forklift_id: Track ID of forklift
            ttc_seconds: Time-to-collision in seconds
            distance_m: Current distance in meters
            frame: Optional frame for snapshot
            
        Returns:
            True if alert was triggered, False if silenced/cooldown
        """
        from .event_store import EventType
        
        # Create TTC-specific event
        event = SafetyEvent(
            event_type=EventType.PEDESTRIAN_FORKLIFT_PROXIMITY,
            zone_id=f"collision_risk_{person_id}_{forklift_id}",
            severity=Severity.HIGH if ttc_seconds < 1.5 else Severity.MEDIUM,
            track_id=person_id,
            duration_seconds=ttc_seconds,
            metadata={
                "person_id": person_id,
                "forklift_id": forklift_id,
                "ttc_seconds": round(ttc_seconds, 2),
                "distance_m": round(distance_m, 2),
                "alert_type": "COLLISION_RISK"
            }
        )
        
        # Override alert message
        self._ttc_alert_message = (
            f"[!] COLLISION RISK: Person #{person_id} - Forklift #{forklift_id} "
            f"| TTC: {ttc_seconds:.1f}s | Dist: {distance_m:.1f}m"
        )
        
        return self.trigger_alert(event, frame)
    
    def check_and_trigger_ttc_alerts(self, ttc_pair, frame=None) -> bool:
        """
        Check TTC pair and trigger alert if within threshold.
        
        This is the main integration point for cv_loop to call.
        
        Args:
            ttc_pair: TTCPair from TTCCalculator.get_top_threat()
            frame: Optional frame for snapshot
            
        Returns:
            True if alert was triggered
        """
        if ttc_pair is None:
            return False
        
        # Only alert if approaching and TTC < danger threshold
        if not ttc_pair.approaching:
            return False
        
        # TTC threshold already checked in get_top_threat (3.0s)
        # But we double-check here for safety
        if ttc_pair.ttc_seconds >= 3.0:
            return False
        
        return self.trigger_ttc_alert(
            person_id=ttc_pair.person_id,
            forklift_id=ttc_pair.forklift_id,
            ttc_seconds=ttc_pair.ttc_seconds,
            distance_m=ttc_pair.distance_m,
            frame=frame
        )


# ============================================================
# Module-level singleton
# ============================================================

_alert_manager: Optional[AlertManager] = None


def get_alert_manager(event_store: EventStore = None, **kwargs) -> AlertManager:
    """Get or create singleton AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        if event_store is None:
            from .event_store import get_event_store
            event_store = get_event_store()
        _alert_manager = AlertManager(event_store, **kwargs)
    return _alert_manager


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    from .event_store import EventStore, SafetyEvent, EventType
    
    # Create test instances
    store = EventStore(db_path="test_alerts.db")
    manager = AlertManager(store, audio_enabled=False)
    
    # Create test event
    event = SafetyEvent(
        event_type=EventType.RESTRICTED_ZONE_PRESENCE,
        zone_id="forklift_lane",
        severity=Severity.HIGH,
        track_id=42,
        duration_seconds=6.5
    )
    
    # Test alert triggering
    print("Testing alert trigger...")
    triggered = manager.trigger_alert(event)
    print(f"  Triggered: {triggered}")
    
    # Check state
    state = manager.get_current_state()
    print(f"  State: active={state.active}, severity={state.severity}")
    print(f"  Message: {state.message}")
    
    # Test silence
    print("\nTesting zone silence...")
    manager.silence_zone("forklift_lane", minutes=1)
    silenced = manager.get_silenced_zones()
    print(f"  Silenced zones: {silenced}")
    
    # Try to trigger again (should fail)
    event2 = SafetyEvent(
        event_type=EventType.RESTRICTED_ZONE_PRESENCE,
        zone_id="forklift_lane",
        severity=Severity.HIGH,
        track_id=43,
        duration_seconds=7.0
    )
    triggered2 = manager.trigger_alert(event2)
    print(f"  Second trigger (silenced): {triggered2}")
    
    # Unsilence and try again
    manager.unsilence_zone("forklift_lane")
    
    # Wait for cooldown
    time.sleep(0.1)
    manager.cooldown_tracker.clear()  # Clear cooldown for test
    
    triggered3 = manager.trigger_alert(event2)
    print(f"  Third trigger (unsilenced): {triggered3}")
    
    # Cleanup
    store.shutdown()
    print("\nAlertManager test complete!")
