#!/usr/bin/env python3
"""
EdgeRunner VLM Client - Machina with Flat JSON
=================================================

Strategy: Flat JSON structure + internal conversion to MachinaDecision

Key finding: Qwen3-VL cannot reliably output deeply nested JSON structures.
Using a flat structure achieves 100% parse rate.

VLM outputs: {"decision": "SAFE", "confidence": 0.85, "action": "FOLLOW", "objects": ["person"], "notes": "..."}
We convert to: MachinaDecision with full slow_path/action structure
"""

import requests
import base64
import json
import cv2
import time
import os

from .machina_schema import (
    MachinaDecision,
    SlowPath,
    Action,
    Evidence,
    FastPath,
    Runtime,
    Decision,
    ActionType,
    TrackStatus,
    ReasonCode,
    Priority,
    SAFE_FALLBACK,
)

# ============================================================
# Configuration
# ============================================================

VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")
VLM_TIMEOUT = 30  # seconds

# System Prompt - FLAT JSON structure for reliable Qwen3-VL output
MACHINA_SYSTEM_PROMPT = """You are Machina. Analyze the image and output JSON with these exact keys:

{"decision": "SAFE", "confidence": 0.85, "action": "FOLLOW", "objects": ["person"], "notes": "brief description"}

VALUES:
- decision: SAFE, SUSPICIOUS, BREACH, or UNKNOWN
- confidence: 0.0 to 1.0
- action: FOLLOW, STOP, ALERT, SEARCH, or LOG_ONLY
- objects: list of items seen
- notes: under 15 words

RULES:
- Normal person: decision=SAFE, action=FOLLOW
- Suspicious object: decision=SUSPICIOUS, action=ALERT
- Unclear: decision=UNKNOWN, action=LOG_ONLY

Output only JSON."""


def encode_image_to_data_uri(frame, quality: int = 85, max_width: int = 640) -> str:
    """
    Convert OpenCV frame to base64 data URI.
    
    Resizes image if larger than max_width to prevent exceeding VLM context size.
    A 1920x1080 image would use ~2200 tokens; 640x360 uses ~400 tokens.
    
    Args:
        frame: OpenCV BGR image
        quality: JPEG quality (0-100)
        max_width: Maximum width in pixels (height scaled proportionally)
    
    Returns:
        data:image/jpeg;base64,... URI string
    """
    h, w = frame.shape[:2]
    
    # Resize if needed
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"


def _flat_to_machina(
    flat_output: dict,
    track_id: int = -1,
    bbox: list = None,
    track_status: str = "SEARCHING",
    vlm_latency_ms: int = 0
) -> MachinaDecision:
    """
    Convert flat VLM output to full MachinaDecision.
    
    Input:  {"decision": "SAFE", "confidence": 0.85, "action": "FOLLOW", "objects": ["person"], "notes": "..."}
    Output: Full MachinaDecision with proper structure
    """
    if bbox is None:
        bbox = [0, 0, 0, 0]
    
    try:
        # Extract flat fields with defaults
        decision_str = flat_output.get("decision", "UNKNOWN").upper()
        confidence = float(flat_output.get("confidence", 0.0))
        action_str = flat_output.get("action", "LOG_ONLY").upper()
        objects = flat_output.get("objects", ["unknown"])
        notes = str(flat_output.get("notes", ""))[:200]
        
        # Validate and map decision
        try:
            decision = Decision(decision_str)
        except ValueError:
            decision = Decision.UNKNOWN
        
        # Validate and map action
        try:
            action_type = ActionType(action_str)
        except ValueError:
            action_type = ActionType.LOG_ONLY
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Map decision to reason code
        reason_map = {
            Decision.SAFE: ReasonCode.AUTHORIZED,
            Decision.SUSPICIOUS: ReasonCode.UNAUTHORIZED_OBJECT,
            Decision.BREACH: ReasonCode.UNAUTHORIZED_OBJECT,
            Decision.UNKNOWN: ReasonCode.INSUFFICIENT_EVIDENCE,
        }
        reason_code = reason_map.get(decision, ReasonCode.INSUFFICIENT_EVIDENCE)
        
        # Map action to priority
        priority_map = {
            ActionType.STOP: Priority.HIGH,
            ActionType.ALERT: Priority.HIGH,
            ActionType.FOLLOW: Priority.MEDIUM,
            ActionType.SEARCH: Priority.MEDIUM,
            ActionType.LOG_ONLY: Priority.LOW,
        }
        priority = priority_map.get(action_type, Priority.LOW)
        
        # Build full MachinaDecision
        return MachinaDecision(
            fast_path=FastPath(
                track_id=track_id,
                bbox_xyxy=bbox,
                track_status=TrackStatus(track_status) if track_status in ["TRACKING", "LOST", "SEARCHING"] else TrackStatus.SEARCHING
            ),
            slow_path=SlowPath(
                decision=decision,
                reason_codes=[reason_code],
                confidence=confidence,
                evidence=Evidence(
                    objects_seen=objects if isinstance(objects, list) else [str(objects)],
                    notes=notes
                )
            ),
            action=Action(
                type=action_type,
                priority=priority
            ),
            runtime=Runtime(
                vlm_latency_ms=vlm_latency_ms,
                json_parse_ok=True,
                fallback_used=False
            )
        )
    except Exception as e:
        # Build fallback with error info
        fallback = SAFE_FALLBACK.model_copy(deep=True)
        fallback.fast_path.track_id = track_id
        fallback.fast_path.bbox_xyxy = bbox
        fallback.runtime.vlm_latency_ms = vlm_latency_ms
        fallback.slow_path.evidence.notes = f"Conversion error: {str(e)[:50]}"
        return fallback


def query_vlm(
    frame,
    track_id: int = -1,
    bbox: list = None,
    track_status: str = "SEARCHING",
    timeout: int = VLM_TIMEOUT
) -> MachinaDecision:
    """
    Query VLM with flat JSON structure and convert to MachinaDecision.
    
    Uses flat JSON format for 100% reliability with Qwen3-VL:
    {"decision": "SAFE", "confidence": 0.85, "action": "FOLLOW", "objects": [...], "notes": "..."}
    
    Args:
        frame: OpenCV BGR image
        track_id: Current ByteTrack ID (-1 if no target)
        bbox: Current bounding box [x1,y1,x2,y2] or None
        track_status: Current tracking status
        timeout: Request timeout in seconds
    
    Returns:
        MachinaDecision: Complete decision contract (or SAFE_FALLBACK on error)
    """
    if bbox is None:
        bbox = [0, 0, 0, 0]
    
    start_time = time.monotonic()
    
    try:
        # 1. Encode image
        data_uri = encode_image_to_data_uri(frame)
        
        # 2. Build payload with flat JSON structure
        payload = {
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "system",
                    "content": MACHINA_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        },
                        {
                            "type": "text",
                            "text": f"Track ID: {track_id}, Status: {track_status}. Analyze the image."
                        }
                    ]
                }
            ],
            "max_tokens": 128,  # Flat JSON needs fewer tokens
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
        
        # 3. Make request
        resp = requests.post(VLM_ENDPOINT, json=payload, timeout=timeout)
        resp.raise_for_status()
        
        # 4. Extract content
        result = resp.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # Calculate latency
        latency_ms = int((time.monotonic() - start_time) * 1000)
        
        # 5. Parse flat JSON
        try:
            flat_output = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed: {e}")
            print(f"   Raw content: {content[:100]}")
            fallback = SAFE_FALLBACK.model_copy(deep=True)
            fallback.fast_path.track_id = track_id
            fallback.fast_path.bbox_xyxy = bbox
            fallback.runtime.vlm_latency_ms = latency_ms
            fallback.slow_path.evidence.notes = f"JSON parse error: {str(e)[:50]}"
            return fallback
        
        # 6. Convert flat output to MachinaDecision
        decision = _flat_to_machina(
            flat_output=flat_output,
            track_id=track_id,
            bbox=bbox,
            track_status=track_status,
            vlm_latency_ms=latency_ms
        )
        
        return decision
    
    except requests.exceptions.Timeout:
        latency_ms = int((time.monotonic() - start_time) * 1000)
        print(f"⚠️ VLM timeout after {latency_ms}ms")
        fallback = SAFE_FALLBACK.model_copy(deep=True)
        fallback.fast_path.track_id = track_id
        fallback.fast_path.bbox_xyxy = bbox
        fallback.runtime.vlm_latency_ms = latency_ms
        fallback.slow_path.evidence.notes = "VLM request timeout"
        return fallback
    
    except Exception as e:
        latency_ms = int((time.monotonic() - start_time) * 1000)
        print(f"⚠️ VLM error: {e}")
        fallback = SAFE_FALLBACK.model_copy(deep=True)
        fallback.fast_path.track_id = track_id
        fallback.fast_path.bbox_xyxy = bbox
        fallback.runtime.vlm_latency_ms = latency_ms
        fallback.slow_path.evidence.notes = f"VLM error: {str(e)[:50]}"
        return fallback


def query_vlm_legacy(frame, prompt=None, timeout=15):
    """
    Legacy VLM query for backward compatibility.
    
    Returns old-style dict with {detected, confidence, reason}.
    """
    decision = query_vlm(frame, timeout=timeout)
    
    detected = decision.slow_path.decision in [Decision.SAFE, Decision.SUSPICIOUS, Decision.BREACH]
    
    return {
        "detected": detected,
        "confidence": decision.slow_path.confidence,
        "reason": decision.slow_path.evidence.notes,
        "action": decision.action.type.value.lower(),
        "_machina_decision": decision.model_dump()
    }


def check_vlm_health() -> bool:
    """Check if VLM service is healthy."""
    try:
        health_url = VLM_ENDPOINT.replace("/v1/chat/completions", "/health")
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except:
        return False


def get_vlm_info() -> dict:
    """Get VLM service info."""
    try:
        props_url = VLM_ENDPOINT.replace("/v1/chat/completions", "/props")
        resp = requests.get(props_url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {"status": "unknown"}


# ============================================================
# Test Functions
# ============================================================

def test_vlm_with_image(image_path: str) -> MachinaDecision:
    """Test VLM with a specific image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image {image_path}")
        return SAFE_FALLBACK
    
    h, w = frame.shape[:2]
    if h > 480 or w > 640:
        frame = cv2.resize(frame, (640, 480))
        print(f"Resized image from {w}x{h} to 640x480")
    
    print(f"Testing VLM with image: {image_path}")
    print("-" * 60)
    
    decision = query_vlm(
        frame,
        track_id=1,
        bbox=[100, 100, 300, 400],
        track_status="TRACKING"
    )
    
    print(f"\n� Result:")
    print(f"  Decision: {decision.slow_path.decision.value}")
    print(f"  Confidence: {decision.slow_path.confidence:.2f}")
    print(f"  Action: {decision.action.type.value}")
    print(f"  Latency: {decision.runtime.vlm_latency_ms}ms")
    print(f"  Fallback: {decision.runtime.fallback_used}")
    print(f"  Notes: {decision.slow_path.evidence.notes}")
    
    return decision


def test_vlm_with_camera():
    """Test VLM with live camera feed."""
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Camera opened. Running VLM test (3 iterations)...")
    print("-" * 60)
    
    success = 0
    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame capture failed")
            continue
        
        decision = query_vlm(frame, track_id=i+1, track_status="TRACKING")
        
        d = decision.slow_path.decision.value
        c = decision.slow_path.confidence
        a = decision.action.type.value
        fb = decision.runtime.fallback_used
        lat = decision.runtime.vlm_latency_ms
        
        status = "✅" if not fb else "⚠️"
        print(f"[{i+1}] {status} {lat}ms | {d} | conf={c:.2f} | {a}")
        
        if not fb:
            success += 1
        
        time.sleep(0.5)
    
    cap.release()
    print(f"\nResults: {success}/3 success")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("EdgeRunner Machina VLM Client")
    print("Strategy: Flat JSON + Internal Conversion")
    print("=" * 60)
    
    if not check_vlm_health():
        print("❌ VLM service is not healthy!")
        print("   Run: sudo systemctl start llama-server")
        sys.exit(1)
    
    print("✅ VLM service is healthy")
    print()
    
    if len(sys.argv) > 1:
        test_vlm_with_image(sys.argv[1])
    else:
        test_vlm_with_camera()
