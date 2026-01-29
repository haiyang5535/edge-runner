#!/usr/bin/env python3
"""
VLM-Powered Zone Calibrator
============================

Uses local Qwen3-VL to automatically identify floor regions in industrial scenes.
Outputs normalized polygon coordinates for zones.yaml.

Usage:
    python -m src.zone_calibrator --video test_videos/test.mp4
    python -m src.zone_calibrator --camera 0
    python -m src.zone_calibrator --image snapshot.jpg --output configs/zones_auto.yaml
"""

import cv2
import json
import yaml
import argparse
import requests
import base64
import os
from pathlib import Path
from typing import List, Dict, Optional

from .settings import settings

# ============================================================
# VLM Zone Detection Prompt
# ============================================================

ZONE_CALIBRATION_PROMPT = """Look at this warehouse image. Find the floor area where forklifts operate.

Return a simple JSON with 4 coordinates (values 0-1):
{"left":0.1,"top":0.5,"right":0.9,"bottom":0.95}

- left: x-coordinate of left edge (0=image left, 1=image right)
- top: y-coordinate of top edge (0=image top, 1=image bottom)  
- right: x-coordinate of right edge
- bottom: y-coordinate of bottom edge

Look at the floor area and estimate coordinates. Return only JSON."""


def encode_frame_to_data_uri(frame, quality: int = 85) -> str:
    """Convert OpenCV frame to base64 data URI."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"


def query_vlm_for_zones(frame, endpoint: str = None, timeout: int = 60) -> Dict:
    """
    Query VLM to detect floor zones in the image.
    
    Args:
        frame: OpenCV BGR image
        endpoint: VLM endpoint URL
        timeout: Request timeout
        
    Returns:
        Dict with 'zones' list or empty dict on failure
    """
    if endpoint is None:
        endpoint = os.environ.get("VLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")
    
    # Resize for VLM (save tokens)
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640 / w
        frame = cv2.resize(frame, (640, int(h * scale)))
    
    data_uri = encode_frame_to_data_uri(frame)
    
    payload = {
        "model": "qwen3-vl",
        "messages": [
            {
                "role": "system",
                "content": ZONE_CALIBRATION_PROMPT
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
                        "text": "Analyze this image and identify floor zones for safety monitoring."
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
        "response_format": {"type": "json_object"}
    }
    
    try:
        print("🧠 Querying VLM for zone detection...")
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        
        result = resp.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # Debug: show raw VLM response (truncated)
        print(f"📝 VLM raw response ({len(content)} chars): {content[:500]}...")
        
        # Robust JSON extraction (handles Chain-of-Thought output)
        zones_data = _clean_json_response(content)
        print(f"✅ VLM returned {len(zones_data.get('zones', []))} zones")
        return zones_data
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ VLM request error: {e}")
        return {"zones": []}
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {"zones": []}


def _clean_json_response(content: str) -> Dict:
    """
    Robust JSON extraction from VLM response with Chain-of-Thought cleanup.
    
    Handles:
    - <think>...</think> tags from reasoning models
    - ```json code blocks
    - Raw JSON content
    - Comma-separated 4-number format (x1,y1,x2,y2)
    """
    import re
    import logging
    
    try:
        # 1. Remove <think> tags (Qwen3-VL Chain-of-Thought)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.strip()
        
        # TRY: Parse as 4 comma-separated numbers first
        nums = re.findall(r'(\d+\.?\d*)', content)
        if len(nums) >= 4:
            x1, y1, x2, y2 = [float(n) for n in nums[:4]]
            # Validate range
            if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                print(f"📐 Parsed coordinates: left={x1}, top={y1}, right={x2}, bottom={y2}")
                return {
                    "zones": [{
                        "id": "vlm_floor",
                        "type": "VEHICLE_LANE",
                        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }]
                }
        
        # 2. Extract JSON from code blocks
        match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: find first { and last }
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                logging.warning("No JSON found in VLM response")
                return {"zones": []}
        
        # 3. Parse JSON
        data = json.loads(json_str)
        
        # 4. Handle flat left/top/right/bottom format (preferred for Qwen3-VL)
        if "left" in data and "top" in data and "right" in data and "bottom" in data:
            left = float(data["left"])
            top = float(data["top"])
            right = float(data["right"])
            bottom = float(data["bottom"])
            print(f"📐 Parsed flat JSON: left={left}, top={top}, right={right}, bottom={bottom}")
            return {
                "zones": [{
                    "id": data.get("id", "vlm_floor"),
                    "type": data.get("type", "VEHICLE_LANE"),
                    "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]]
                }]
            }
        
        # 5. Handle single-zone response (VLM returned zone without "zones" wrapper)
        if "zones" not in data and "polygon" in data:
            return {"zones": [data]}
        
        # 6. Handle x1,y1,x2,y2 format in JSON
        if "zones" not in data and "x1" in data:
            x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
            return {
                "zones": [{
                    "id": data.get("id", "vlm_floor"),
                    "type": data.get("type", "VEHICLE_LANE"),
                    "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                }]
            }
        
        return data
        
    except json.JSONDecodeError as e:
        logging.error(f"VLM JSON Parse Failed: {e}. Raw: {content[:200]}...")
        return {"zones": []}
        
        return data
        
    except json.JSONDecodeError as e:
        logging.error(f"VLM JSON Parse Failed: {e}. Raw: {content[:200]}...")
        return {"zones": []}


def zones_to_yaml(zones_data: Dict, output_path: str = None) -> str:
    """
    Convert VLM zone output to zones.yaml format.
    
    Args:
        zones_data: Dict from VLM with 'zones' list
        output_path: Optional path to save YAML
        
    Returns:
        YAML string
    """
    zones_list = zones_data.get('zones', [])
    
    # Fallback: If VLM returned 0 zones, add a sensible default zone
    if not zones_list:
        print("📍 Adding default fallback zone (bottom vehicle lane)")
        zones_list = [
            {
                "id": "vehicle_lane",
                "type": "VEHICLE_LANE",
                "display_name": "Vehicle Operating Area",
                "polygon": [[0.1, 0.6], [0.9, 0.6], [0.9, 0.95], [0.1, 0.95]],
                "reason": "Default zone - bottom portion of frame where vehicles typically operate"
            }
        ]
    
    # Default color palette (BGR)
    colors = {
        "RESTRICTED": [80, 100, 200],   # Muted coral
        "VEHICLE_LANE": [100, 140, 180], # Muted tan
        "FLOW": [60, 180, 220],          # Soft amber
    }
    
    yaml_zones = []
    for zone in zones_list:
        zone_type = zone.get('type', 'RESTRICTED')
        yaml_zone = {
            'id': zone.get('id', 'zone_auto'),
            'type': zone_type,
            'display_name': zone.get('display_name', zone.get('id', 'Auto Zone')),
            'color': colors.get(zone_type, [0, 255, 255]),
            'polygon': zone.get('polygon', [[0.2, 0.6], [0.8, 0.6], [0.8, 0.9], [0.2, 0.9]]),
            'rules': [
                {
                    'type': 'PERSON_PRESENCE',
                    'max_dwell_seconds': 5 if zone_type == 'RESTRICTED' else 30,
                    'severity': 'HIGH' if zone_type == 'RESTRICTED' else 'MEDIUM',
                    'target_classes': ['person']
                }
            ],
            'enabled': True
        }
        yaml_zones.append(yaml_zone)
    
    config = {
        'zones': yaml_zones,
        'settings': {
            'default_frame_size': [1920, 1080],
            'stale_track_timeout': 2.0,
            'min_confidence': 0.5
        }
    }
    
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f"# Auto-generated by VLM Zone Calibrator\n")
            f.write(f"# Source: VLM analysis of camera frame\n\n")
            f.write(yaml_str)
        print(f"✅ Saved zones to: {output_path}")
    
    return yaml_str


def capture_calibration_frame(source) -> Optional:
    """Capture a single frame for calibration."""
    if isinstance(source, str) and os.path.isfile(source):
        # Image file
        frame = cv2.imread(source)
        if frame is not None:
            print(f"📷 Loaded image: {source}")
            return frame
        # Video file
        cap = cv2.VideoCapture(source)
    else:
        # Camera index
        cap = cv2.VideoCapture(int(source))
    
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        h, w = frame.shape[:2]
        print(f"📷 Captured frame: {w}x{h}")
        return frame
    
    return None


def calibrate_zones(source, output_path: str = None, dry_run: bool = False) -> Dict:
    """
    Main calibration function.
    
    Args:
        source: Video path, image path, or camera index
        output_path: Path to save zones.yaml
        dry_run: If True, just print results without saving
        
    Returns:
        Dict with detected zones
    """
    print("=" * 60)
    print("VLM Zone Calibrator")
    print("=" * 60)
    
    frame = capture_calibration_frame(source)
    if frame is None:
        return {"zones": []}
    
    zones_data = query_vlm_for_zones(frame)
    
    if zones_data.get('zones'):
        print("\n📋 Detected Zones:")
        for zone in zones_data['zones']:
            print(f"   - {zone.get('id')}: {zone.get('type')} ({zone.get('reason', 'no reason')})")
            print(f"     Polygon: {zone.get('polygon')}")
    else:
        print("\n⚠️ No zones detected by VLM")
        print("   This may mean:")
        print("   - The scene doesn't have clear floor markings")
        print("   - VLM couldn't identify industrial zones")
        print("   - Consider manual configuration in zones.yaml")
    
    if not dry_run and output_path:
        zones_to_yaml(zones_data, output_path)
    elif dry_run:
        print("\n📄 Generated YAML (dry-run):")
        print("-" * 40)
        print(zones_to_yaml(zones_data))
    
    return zones_data


def main():
    parser = argparse.ArgumentParser(description='VLM-Powered Zone Calibrator')
    parser.add_argument('--video', '-v', type=str, help='Video file path')
    parser.add_argument('--image', '-i', type=str, help='Image file path')
    parser.add_argument('--camera', '-c', type=int, default=None, help='Camera index')
    parser.add_argument('--output', '-o', type=str, default='configs/zones_auto.yaml', help='Output zones.yaml path')
    parser.add_argument('--dry-run', action='store_true', help='Print results without saving')
    args = parser.parse_args()
    
    source = args.video or args.image or args.camera
    if source is None:
        source = 0  # Default to camera 0
    
    calibrate_zones(source, args.output, args.dry_run)


if __name__ == "__main__":
    main()
