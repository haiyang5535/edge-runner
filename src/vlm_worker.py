import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from enum import Enum

from .settings import settings
from .state import SharedState
from .vlm_client import query_vlm, check_vlm_health
from .machina_schema import SAFE_FALLBACK

class TriggerType(Enum):
    """Type of VLM trigger event."""
    REFLEX = "reflex"        # YOLO detected violation → immediate analysis
    HEARTBEAT = "heartbeat"  # Periodic safety scan (every 30s)

@dataclass
class VLMTrigger:
    """Event payload for VLM trigger queue."""
    trigger_type: TriggerType
    frame: object            # np.ndarray - captured frame
    prompt_context: str      # Context for VLM prompt
    track_id: int = -1
    bbox: list = field(default_factory=list)
    zone_id: str = ""

# Global trigger queue
vlm_trigger_queue: Queue = Queue(maxsize=5)

def vlm_worker(shared: SharedState):
    """
    Event-Driven VLM Thread with Machina decision contracts.
    """
    print(f"🧠 VLM Thread started (Event-Driven, Heartbeat: {settings.VLM_HEARTBEAT_INTERVAL:.0f}s)")
    
    last_vlm_call = time.time()
    
    while shared.is_running():
        # Check if heartbeat trigger is needed
        now = time.time()
        time_since_last = now - last_vlm_call
        
        if time_since_last >= settings.VLM_HEARTBEAT_INTERVAL:
            snapshot = shared.get_frame()
            if snapshot is not None and not vlm_trigger_queue.full():
                try:
                    vlm_trigger_queue.put_nowait(VLMTrigger(
                        trigger_type=TriggerType.HEARTBEAT,
                        frame=snapshot,
                        prompt_context="Perform general safety inspection. Check for spills, obstacles, or unsafe stacking."
                    ))
                    print(f"\n💓 Heartbeat trigger queued (idle for {time_since_last:.0f}s)")
                except:
                    pass
        
        try:
            trigger = vlm_trigger_queue.get(timeout=2.0)
        except Empty:
            continue
        
        if not shared.is_running():
            break
        
        trigger_icon = "⚡" if trigger.trigger_type == TriggerType.REFLEX else "💓"
        print(f"\n{trigger_icon} VLM Trigger: {trigger.trigger_type.value} | {trigger.prompt_context[:50]}...")
        
        try:
            shared.vlm_processing = True
            
            track_id = trigger.track_id
            bbox = trigger.bbox if trigger.bbox else [0, 0, 0, 0]
            track_status = "VIOLATION" if trigger.trigger_type == TriggerType.REFLEX else "INSPECTION"
            
            decision = query_vlm(
                trigger.frame,
                track_id=track_id,
                bbox=bbox,
                track_status=track_status
            )
            
            shared.vlm_processing = False
            last_vlm_call = time.time()
            
            shared.update_from_vlm(decision)
            
            d = decision.slow_path
            
            # P1 FIX: Filter VLM output - only show safety-relevant decisions
            # SAFE decisions are silent to reduce log noise
            if d.decision.value in ["SUSPICIOUS", "BREACH"]:
                print(f"🚨 VLM ALERT: {d.decision.value} | "
                      f"conf={d.confidence:.2f} | "
                      f"action={decision.action.type.value} | "
                      f"reason={d.evidence.notes[:50]}...")
            elif trigger.trigger_type == TriggerType.REFLEX:
                # Reflex triggers should always log (they came from zone violations)
                print(f"🧠 VLM Analyzed zone violation: {d.decision.value} | "
                      f"conf={d.confidence:.2f} | "
                      f"latency={decision.runtime.vlm_latency_ms}ms")
            
        except Exception as e:
            shared.vlm_processing = False
            print(f"\n🧠 VLM Error: {e}")
            shared.update_from_vlm(SAFE_FALLBACK)
    
    print("\n🧠 VLM Thread stopped")
