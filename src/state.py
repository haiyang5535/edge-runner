import threading
import cv2
from typing import Optional, List

from .machina_schema import MachinaDecision
from .state_machine import MachinaStateMachine, ControlState
from .metrics import MachinaMetrics
from .event_store import EventStore
from .zone_manager import ZoneManager
from .alert_manager import AlertManager
from .settings import settings

class SharedState:
    """
    Thread-safe shared state with Machina metrics integration.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self.frame = None  # Raw camera frame (for VLM)
        self.annotated_frame = None # Annotated frame (for Web, lazy encoded)
        self.jpeg_buffer = None  # Pre-encoded JPEG (for Web)
        self.state_machine = MachinaStateMachine()
        # Use larger window (150 frames = 5 seconds @ 30 FPS) for better FPS smoothing
        self.metrics = MachinaMetrics(fast_window=150)
        
        # VLM state
        self.vlm_processing = False
        self.last_decision: Optional[MachinaDecision] = None
        
        # CV state (for VLM context injection)
        self.current_track_id = -1
        self.current_bbox = [0, 0, 0, 0]
        self.current_track_status = "SEARCHING"
        
        # Running state
        self.running = True
        self.frame_id = 0
        
        # Control state for HUD
        self.current_state = "SEARCHING"
        self.current_cmd = "ROTATE"
        self.current_error = 0
        
        # Safety MVP components (initialized later in main())
        self.event_store: Optional[EventStore] = None
        self.zone_manager: Optional[ZoneManager] = None
        self.alert_manager: Optional[AlertManager] = None
        self.safety_enabled = settings.SAFETY_CHECK_ENABLED
        
        # Current YOLO detections for safety checking
        self.current_detections: list = []
        self.current_frame_dims: tuple = (640, 480)
        
        # TTC Threat State
        self.ttc_threat: Optional[dict] = None
    
    def update_ttc_threat(self, threat):
        """Update current TTC threat state."""
        with self.lock:
            if threat:
                self.ttc_threat = {
                    "person_id": threat.person_id,
                    "forklift_id": threat.forklift_id,
                    "distance_m": threat.distance_m,
                    "ttc_seconds": threat.ttc_seconds,
                    "approaching": threat.approaching
                }
            else:
                self.ttc_threat = None

    def init_safety_components(self):
        """Initialize Safety MVP components."""
        try:
            self.event_store = EventStore(
                db_path=settings.EVENTS_DB,
                snapshot_dir=settings.SNAPSHOTS_DIR
            )
            self.zone_manager = ZoneManager(settings.ZONES_CONFIG)
            self.alert_manager = AlertManager(
                self.event_store,
                audio_enabled=False  # Set True for audio alerts
            )
            print("✅ Safety MVP components initialized")
            print(f"   Zones loaded: {len(self.zone_manager.zones)}")
        except Exception as e:
            print(f"⚠️ Safety MVP init failed: {e}")
            self.safety_enabled = False
    
    def update_detections(self, detections: list, frame_dims: tuple = (640, 480)):
        """Update current detections for safety checking."""
        with self.lock:
            self.current_detections = detections
            self.current_frame_dims = frame_dims
    
    def get_detections(self) -> tuple:
        """Get current detections and frame dims."""
        with self.lock:
            return self.current_detections.copy(), self.current_frame_dims
    
    def get_state(self) -> ControlState:
        with self.lock:
            return self.state_machine.get_state()
    
    def get_command(self) -> str:
        with self.lock:
            return self.state_machine.get_command().value
    
    def update_cv_context(self, track_id: int, bbox: list, track_status: str):
        """Update CV context for VLM injection."""
        with self.lock:
            self.current_track_id = track_id
            self.current_bbox = bbox
            self.current_track_status = track_status
    
    def get_cv_context(self) -> tuple:
        """Get CV context for VLM call."""
        with self.lock:
            return (self.current_track_id, 
                    self.current_bbox.copy(), 
                    self.current_track_status)
    
    def update_from_vlm(self, decision: MachinaDecision):
        """Update state machine and metrics from VLM decision."""
        with self.lock:
            self.state_machine.update(decision)
            self.metrics.record_slow_loop(decision)
            self.last_decision = decision
    
    def set_frame(self, frame):
        """Set raw frame for VLM thread."""
        with self.lock:
            self.frame = frame.copy()
    
    def get_frame(self):
        """Get raw frame for VLM."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def set_annotated_frame(self, frame):
        """Store annotated frame for lazy encoding."""
        with self.lock:
            self.annotated_frame = frame
            self.jpeg_buffer = None  # Invalidate cache
            self.frame_id += 1
    
    def get_jpeg_buffer(self):
        """Get pre-encoded JPEG buffer for streaming (Lazy Encoding)."""
        with self.lock:
            # Return cached buffer if valid
            if self.jpeg_buffer is not None:
                return self.jpeg_buffer, self.frame_id
            
            # No frame to encode
            if not hasattr(self, 'annotated_frame') or self.annotated_frame is None:
                return None, 0
            
            # Grab frame for encoding
            frame_to_encode = self.annotated_frame
            current_id = self.frame_id

        # Encode OUTSIDE the lock
        _, jpeg = cv2.imencode('.jpg', frame_to_encode, 
            [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY])
        jpeg_bytes = jpeg.tobytes()
        
        with self.lock:
            # Cache if frame hasn't changed
            if self.frame_id == current_id:
                self.jpeg_buffer = jpeg_bytes
            return jpeg_bytes, current_id
    
    def update_control_state(self, state, cmd, track_id, error):
        """Update control state for HUD display."""
        with self.lock:
            self.current_state = state
            self.current_cmd = cmd
            self.current_track_id = track_id
            self.current_error = error
    
    def get_api_status(self) -> dict:
        """Get complete status for API endpoint."""
        with self.lock:
            metrics_summary = self.metrics.get_summary()
            
            # Get recent decisions for log
            decision_log = []
            for entry in self.metrics.get_decision_log(10):
                decision_log.append({
                    "decision_id": entry.decision_id,
                    "decision": entry.decision,
                    "confidence": round(entry.confidence, 2),
                    "action": entry.action,
                    "track_id": entry.track_id,
                    "latency_ms": entry.vlm_latency_ms,
                    "fallback": entry.fallback_used,
                    "notes": entry.notes
                })
            
            # Safety MVP status
            safety_status = {
                "enabled": self.safety_enabled,
                "zones_loaded": 0,
                "active_alerts": [],
                "silenced_zones": {},
                "unacknowledged_count": 0
            }
            
            if self.safety_enabled and self.zone_manager:
                safety_status["zones_loaded"] = len(self.zone_manager.zones)
            
            if self.safety_enabled and self.alert_manager:
                safety_status["active_alerts"] = self.alert_manager.get_active_alerts()
                safety_status["silenced_zones"] = self.alert_manager.get_silenced_zones()
                alert_state = self.alert_manager.get_current_state()
                safety_status["current_alert"] = {
                    "active": alert_state.active,
                    "message": alert_state.message,
                    "severity": alert_state.severity.value if alert_state.severity else None
                }
            
            if self.safety_enabled and self.event_store:
                safety_status["unacknowledged_count"] = self.event_store.get_unacknowledged_count()
            
            return {
                "metrics": metrics_summary,
                # Use current UI state (LOCKED/DETECTED/LOST) instead of VLM state machine
                "state": {"state": self.current_state},
                "vlm_processing": self.vlm_processing,
                "decision_log": decision_log,
                "safety": safety_status,
                "ttc": self.ttc_threat
            }
    
    def stop(self):
        with self.lock:
            self.running = False
    
    def is_running(self):
        with self.lock:
            return self.running
