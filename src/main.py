#!/usr/bin/env python3
"""
EdgeRunner Visual Mission Control - Machina Runtime

Implements the vNext pipeline architecture with:
- ROIFilter: Reject detections in static exclusion zones
- MotionGate: Require displacement before confirming forklift
- GroundPlane: BEV projection for accurate distance
- LoopResetManager: Clean state reset for video loops
- ZoneConfigManager: Single source of truth for zones
"""

import cv2
import time
import sys
import argparse
import threading
import os
import subprocess
import numpy as np
from ultralytics import YOLO

# Import refactored modules
from .settings import settings
from .state import SharedState
from .vlm_worker import vlm_worker, vlm_trigger_queue, VLMTrigger, TriggerType, check_vlm_health
from .server import run_web_server
from .box_predictor import BoxPredictor
from .zone_manager import TrackedObject
from .event_store import SafetyEvent, EventType, Severity
from .proximity_detector import ProximityDetector, get_proximity_detector
from .bbox_smoother import BboxSmoother, get_forklift_smoother
from .zone_calibrator import query_vlm_for_zones, zones_to_yaml
from . import vis

# New pipeline modules (vNext)
from .roi_filter import ROIFilter, get_roi_filter
from .motion_gate import MotionGate, get_motion_gate
from .ground_plane import GroundPlane, get_ground_plane
from .loop_reset import LoopResetManager, ZoneDwellWrapper, get_loop_reset_manager
from .zone_config_manager import ZoneConfigManager, get_zone_config_manager
from .runtime_mode import RuntimeMode, get_mode_config, determine_runtime_mode
from .ttc import get_ttc_calculator, TTCCalculator


# ============================================================ 
# Background Zone Calibration (Async Hot-Swap)
# ============================================================ 
def background_zone_calibration(frame, zone_config_manager):
    """
    Background task: Call VLM to detect zones and hot-swap via ZoneConfigManager.
    Runs in daemon thread, does not block main CV loop.
    """
    try:
        print("🚀 [Background] VLM Zone Calibration starting...")
        zones_data = query_vlm_for_zones(frame)
        
        if zones_data.get('zones'):
            # Use ZoneConfigManager for hot-swap (validates and prefixes VLM zones)
            zone_config_manager.apply_vlm_zones(zones_data['zones'], merge=True)
            print("✅ [Background] Zones calibrated via ZoneConfigManager")
        else:
            print("⚠️ [Background] VLM returned no zones, keeping cached config")
    except Exception as e:
        print(f"⚠️ [Background] Zone calibration failed: {e}")
        # Failure is OK - main loop keeps using cached zones

# ============================================================ 
# Helper: Service Management
# ============================================================ 
def manage_llama_service(action: str):
    """
    Manage the llama-server systemd service to free up GPU memory.
    action: 'start', 'stop', 'restart', or 'is-active'
    """
    service_name = "llama-server.service"
    
    if action == "is-active":
        # Returns 0 if active, non-zero otherwise
        res = subprocess.run(["systemctl", "is-active", service_name], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
        
    print(f"🔄 Attempting to {action} {service_name}...")
    try:
        # Try with sudo non-interactively
        subprocess.run(["sudo", "systemctl", action, service_name], check=True)
        print(f"✅ Successfully {action}ed {service_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed to {action} {service_name}. You might need to run:")
        print(f"   sudo systemctl {action} {service_name}")
        return False
    except Exception as e:
        print(f"⚠️ Error managing service: {e}")
        return False

# ============================================================ 
# SOTA Model Loading with TensorRT Auto-Export
# ============================================================ 
def load_sota_model(model_path: str, 
                    engine_path: str = None,
                    target_classes: list = None):
    """SOTA Model Loading with TensorRT Auto-Export for Jetson."""
    if target_classes is None:
        target_classes = ["person", "forklift"]
    
    if engine_path is None:
        engine_path = model_path.replace('.pt', '.engine')
    
    if os.path.exists(engine_path):
        print(f"🚀 Loading TensorRT Engine: {engine_path}")
        model = YOLO(engine_path, task='detect')
        print(f"✅ Engine loaded - FP16 optimized for Jetson")
        return model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"⚠️ Engine not found. Optimizing {model_path} for Jetson...")
    print(f"   This will take ~5-10 minutes on Orin Nano (one-time)")
    
    # Check and stop llama-server to free GPU memory
    service_was_active = manage_llama_service("is-active")
    if service_was_active:
        print("🛑 Stopping llama-server to free up GPU memory for export...")
        if not manage_llama_service("stop"):
            print("⚠️ WARNING: Could not stop llama-server. Export might fail due to OOM.")
            time.sleep(3)
    
    try:
        model = YOLO(model_path)
        if 'world' in model_path.lower():
            print(f"🔧 Baking vocabulary: {target_classes}")
            model.set_classes(target_classes)
            
            # Save intermediate model to bake vocabulary
            # This is CRITICAL for YOLO World to work in TensorRT
            custom_model_path = model_path.replace('.pt', '_custom.pt')
            model.save(custom_model_path)
            print(f"✅ Baked vocabulary to intermediate model: {custom_model_path}")
            
            # Reload the custom model
            model = YOLO(custom_model_path)
        
        print("⚙️ Exporting to TensorRT FP16...")
        # export() returns the path to the exported file
        exported_path = model.export(format='engine', half=True, device=0)
        
        # If we used a custom model, the engine will be named accordingly (e.g., _custom.engine)
        # We need to rename it to the expected engine_path
        if exported_path and str(exported_path) != engine_path:
             print(f"🔄 Renaming {exported_path} -> {engine_path}")
             import shutil
             shutil.move(str(exported_path), engine_path)

        print(f"✅ Engine created: {engine_path}")
        
        # Cleanup
        if 'world' in model_path.lower() and os.path.exists(custom_model_path):
            try:
                os.remove(custom_model_path)
            except OSError:
                pass
                
    except Exception as e:
        print(f"❌ TensorRT Export Failed: {e}")
        # Attempt to restore service on failure
        if service_was_active:
            print("🔄 Restoring llama-server...")
            manage_llama_service("start")
        raise e

    # Restart service if it was running
    if service_was_active:
        print("🚀 Restarting llama-server...")
        manage_llama_service("start")

    return YOLO(engine_path, task='detect')

# ============================================================ 
# Control Logic  
# ============================================================ 
def compute_control(bbox, frame_center, threshold=settings.CONTROL_THRESHOLD * 2.5):
    """Compute control command from bbox (using 1080p coords)."""
    # Scale threshold for 1080p if settings is for 480p
    # Default settings.CONTROL_THRESHOLD is likely ~50. 
    # 50 * 2.5 = 125 pixels deviation for 1920 width is reasonable.
    
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    error_x = int(bbox_center_x - frame_center[0])
    
    if error_x < -threshold:
        cmd = "LEFT"
    elif error_x > threshold:
        cmd = "RIGHT"
    else:
        cmd = "FWD"
    
    return error_x, cmd

# ============================================================ 
# Thread 1: Main CV Loop
# ============================================================ 
def main():
    parser = argparse.ArgumentParser(description='EdgeRunner Machina Visual Mission Control')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress console FPS output')
    parser.add_argument('--video', '-v', type=str, metavar='PATH', help='Use video file instead of camera')
    parser.add_argument('--loop', '-l', action='store_true', help='Loop video file')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index')
    parser.add_argument('--demo', '-d', action='store_true', help='Force DEMO_MODE (uncalibrated allowed)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EdgeRunner Machina Visual Mission Control")
    print("=" * 60)
    
    if not check_vlm_health():
        print("❌ VLM service not available!")
        return
    print("✅ VLM service is healthy")
    
    # Load Model
    print("📦 Loading YOLO model...")
    if settings.USE_WORLD_MODEL:
        if os.path.exists(settings.YOLO_WORLD_MODEL_PATH.replace('.pt', '.engine')):
            model = YOLO(settings.YOLO_WORLD_MODEL_PATH.replace('.pt', '.engine'), task='detect')
        elif os.path.exists(settings.YOLO_WORLD_MODEL_PATH):
            model = load_sota_model(settings.YOLO_WORLD_MODEL_PATH, target_classes=settings.WORLD_CLASSES)
        else:
            print("Fallback to standard YOLO")
            model = YOLO(settings.YOLO_ENGINE_PATH, task='detect')
    else:
        if os.path.exists(settings.YOLO_ENGINE_PATH):
            model = YOLO(settings.YOLO_ENGINE_PATH, task='detect')
        elif os.path.exists(settings.YOLO_ENGINE_PATH.replace('.engine', '.pt')):
            model = load_sota_model(settings.YOLO_ENGINE_PATH.replace('.engine', '.pt'), engine_path=settings.YOLO_ENGINE_PATH)
        else:
            print(f"❌ Model not found: {settings.YOLO_ENGINE_PATH}")
            return

    # Open Source
    HIGH_RES_W, HIGH_RES_H = 1920, 1080
    INFER_W, INFER_H = 640, 360  # 16:9 Aspect Ratio to match 1080p
    SCALE_FACTOR = HIGH_RES_W / INFER_W # 3.0

    if args.video:
        # Expand user path (e.g. ~) just in case
        video_path = os.path.expanduser(args.video)
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source: {video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / video_fps
    else:
        print(f"📷 Opening Camera {args.camera} at {HIGH_RES_W}x{HIGH_RES_H} MJPG...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_RES_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_RES_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        frame_delay = 0

    # Verify resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera Resolution: {actual_w}x{actual_h}")

    shared_state = SharedState()
    if settings.SAFETY_CHECK_ENABLED:
        shared_state.init_safety_components()

    # ============================================================
    # Initialize New Pipeline Components (vNext)
    # ============================================================
    
    # Initialize GroundPlane for BEV distance (if calibration exists)
    ground_plane = None
    if settings.ENABLE_BEV_DISTANCE and os.path.exists(settings.CAMERA_CALIBRATION):
        ground_plane = get_ground_plane(settings.CAMERA_CALIBRATION)
        if ground_plane.is_calibrated:
            print(f"✅ BEV calibration loaded: {ground_plane.calibration.camera_id}")
    
    # Determine runtime mode based on calibration and --demo flag
    runtime_mode = determine_runtime_mode(
        force_demo=args.demo,
        calibration_path=settings.CAMERA_CALIBRATION,
        strict=False  # Don't block startup, just warn
    )
    mode_config = get_mode_config(runtime_mode)
    print(f"\n🎯 Runtime Mode: {runtime_mode.value}")
    if mode_config.show_uncalibrated_banner:
        print(f"⚠️  {mode_config.banner_text}")
    
    # Initialize ROI Filter (exclusion zones for static cargo)
    roi_filter = None
    if settings.ENABLE_ROI_FILTER:
        roi_filter = get_roi_filter(settings.ROI_EXCLUSION_CONFIG)
        print(f"✅ ROI Filter: {len(roi_filter.exclusion_zones)} exclusion zones")
    
    # Initialize Motion Gate (mode-aware)
    motion_gate = None
    if settings.ENABLE_MOTION_GATE and mode_config.motion_gate_enabled:
        if mode_config.motion_gate_require_bev and not (ground_plane and ground_plane.is_calibrated):
            print(f"⚠️ Motion Gate: DISABLED (SAFETY_MODE requires BEV calibration)")
        else:
            motion_gate = get_motion_gate(
                ground_plane=ground_plane,
                require_bev=mode_config.motion_gate_require_bev,
                use_adaptive_threshold=True
            )
            print(f"✅ Motion Gate: {'BEV mode' if motion_gate.use_bev else 'adaptive pixel mode'}")
    elif not mode_config.motion_gate_enabled:
        print(f"⚠️ Motion Gate: DISABLED (DEMO_MODE)")
    
    # Initialize ZoneConfigManager (single source of truth)
    zone_config_manager = get_zone_config_manager(
        settings.ZONES_CONFIG,
        settings.ZONES_VLM_CACHE
    )
    # Note: SharedState already loaded zones, so we just wrap it
    # In a future refactor, ZoneConfigManager should be the primary loader
    
    # Initialize LoopResetManager
    loop_reset_mgr = get_loop_reset_manager()
    
    # Initialize TTC Calculator
    ttc_calculator = get_ttc_calculator(ground_plane)
    
    # Register components for loop reset
    forklift_smoother = get_forklift_smoother()
    predictor = BoxPredictor()
    
    loop_reset_mgr.register(ttc_calculator)
    loop_reset_mgr.register(forklift_smoother)
    loop_reset_mgr.register(predictor)
    if motion_gate:
        loop_reset_mgr.register(motion_gate)
    if shared_state.zone_manager:
        loop_reset_mgr.register(ZoneDwellWrapper(shared_state.zone_manager))
    
    print(f"✅ Loop Reset Manager: {loop_reset_mgr.component_count} components registered")
    
    # Update proximity detector with ground plane
    if ground_plane and ground_plane.is_calibrated:
        proximity_detector = get_proximity_detector()
        proximity_detector.set_ground_plane(ground_plane)
        print(f"✅ Proximity Detector: BEV mode enabled")

    # Start Threads
    threading.Thread(target=vlm_worker, args=(shared_state,), daemon=True).start()
    threading.Thread(target=run_web_server, args=(shared_state,), daemon=True).start()
    
    # Capture first frame for VLM zone calibration
    ret, first_frame = cap.read()
    if ret and shared_state.zone_manager:
        # Start background VLM zone calibration (non-blocking async hot-swap)
        # Use zone_config_manager for proper validation and prefixing
        threading.Thread(
            target=background_zone_calibration,
            args=(first_frame.copy(), zone_config_manager),
            daemon=True
        ).start()
        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"🚀 Running! Dashboard: http://localhost:{settings.WEB_PORT}/dashboard")

    frame_count = 0
    target_id = None
    last_loop_time = time.monotonic()
    
    try:
        while True:
            loop_start = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                if args.video and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    # Post-seek flush: discard 3 frames to clear VideoCapture buffer
                    # This prevents ghost boxes from stale frames in the buffer
                    for _ in range(3):
                        cap.grab()
                    
                    # Reset tracker to prevent bbox persistence after loop
                    if hasattr(model, 'predictor') and model.predictor and hasattr(model.predictor, 'trackers'):
                        for tracker in model.predictor.trackers:
                            tracker.reset()
                    
                    # Reset all pipeline components via LoopResetManager
                    loop_reset_mgr.reset_all()
                    print(f"\n🔄 Video looped - all state reset, flushed 3 frames (loop #{loop_reset_mgr.reset_count})")
                    continue
                break
            
            # frame is 1920x1080 (High Res)
            shared_state.set_frame(frame)
            vis_frame = frame.copy() # Use High Res frame for drawing
            frame_center = (HIGH_RES_W // 2, HIGH_RES_H // 2)
            
            # Crosshair removed for cleaner demo
            # vis_frame = vis.draw_crosshair(vis_frame, frame_center)
            now = time.monotonic()
            
            run_detector = (frame_count % settings.DETECTOR_INTERVAL == 0)
            shared_state.metrics.record_detector_run(run_detector)
            
            best_person = None
            all_detections = []
            active_violations = []
            
            if run_detector:
                # Resize for Inference
                infer_frame = cv2.resize(frame, (INFER_W, INFER_H))
                
                results = model.track(
                    infer_frame, 
                    classes=settings.SAFETY_CLASSES,
                    persist=True,
                    tracker="configs/bytetrack_stable.yaml",
                    verbose=False,
                    conf=0.35,  # Lowered for safety-first (avoid false negatives)
                )
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get coords in INFER_W space
                        x1_s, y1_s, x2_s, y2_s = map(float, box.xyxy[0])
                        
                        # Scale up to HIGH_RES space
                        x1 = int(x1_s * SCALE_FACTOR)
                        y1 = int(y1_s * SCALE_FACTOR)
                        x2 = int(x2_s * SCALE_FACTOR)
                        y2 = int(y2_s * SCALE_FACTOR)
                        
                        tid = int(box.id[0]) if box.id is not None else frame_count
                        
                        class_info = settings.CLASS_MAP.get(cls_id, {"name": "unknown", "is_person": False})
                        class_name = class_info["name"]
                        is_person = class_info.get("is_person", False)
                        
                        detection = TrackedObject(
                            track_id=tid,
                            class_name=class_name,
                            bbox=[x1, y1, x2, y2],
                            center=((x1 + x2) // 2, (y1 + y2) // 2),
                            confidence=conf
                        )
                        all_detections.append(detection)
                        
                        if is_person and conf > 0.5:
                             best_person = (tid, [x1, y1, x2, y2], conf)
                
                # ============================================================
                # vNext Pipeline: Apply ROI Filter + Motion Gate
                # ============================================================
                
                # Step 1: ROI Filter - Remove detections in exclusion zones
                if roi_filter:
                    all_detections = roi_filter.filter(all_detections, (HIGH_RES_W, HIGH_RES_H))
                
                # Step 2: Motion Gate - Require displacement for forklift confirmation
                # Note: Motion gate only filters forklifts, persons pass through
                if motion_gate:
                    gated_detections = []
                    for det in all_detections:
                        if motion_gate.check(det):
                            gated_detections.append(det)
                    # Update all_detections with gated results
                    # Keep all persons, only keep confirmed forklifts
                    all_detections = gated_detections
                
                # Step 3: Apply bbox smoothing to forklift detections (reduces flickering)
                forklift_detections = [d for d in all_detections if d.class_name != "person"]
                stable_forklifts = forklift_smoother.smooth(forklift_detections, now)
                
                # Draw stable forklift detections only
                for det in stable_forklifts:
                    vis_frame = vis.draw_detection(
                        vis_frame, det.bbox, det.track_id, 
                        det.class_name, det.confidence, frame_center
                    )
                
                # Safety Checks (using High Res Coords)
                if shared_state.safety_enabled and shared_state.zone_manager and all_detections:
                    violations = shared_state.zone_manager.check_violations(all_detections, (HIGH_RES_W, HIGH_RES_H))
                    for violation in violations:
                        event = SafetyEvent(
                            event_type=EventType.RESTRICTED_ZONE_PRESENCE, # Simplified mapping
                            zone_id=violation.zone_id,
                            severity=violation.severity,
                            track_id=violation.track_id,
                            duration_seconds=violation.dwell_seconds,
                            metadata={"bbox": violation.bbox}
                        )
                        if shared_state.alert_manager and shared_state.alert_manager.trigger_alert(event, frame):
                            print(f"\n🚨 ALERT: {violation.zone_id} (track {violation.track_id})")
                            if not vlm_trigger_queue.full():
                                # VLM gets the High Res frame, but downscaled by worker if needed
                                vlm_trigger_queue.put_nowait(VLMTrigger(
                                    trigger_type=TriggerType.REFLEX,
                                    frame=frame.copy(),
                                    prompt_context=f"ALERT: Person in {violation.zone_id}",
                                    track_id=violation.track_id, bbox=violation.bbox, zone_id=violation.zone_id
                                ))
                        active_violations.append(violation)
                
                # Proximity Checks (P0 Safety Feature - now with BEV support)
                if shared_state.safety_enabled and all_detections:
                    proximity_detector = get_proximity_detector()
                    prox_violations = proximity_detector.check_proximity(all_detections, (HIGH_RES_W, HIGH_RES_H))
                    for pv in prox_violations:
                        # Include distance_m in metadata if available (BEV mode)
                        metadata = {
                            "person_bbox": pv.person_bbox,
                            "forklift_bbox": pv.forklift_bbox,
                            "forklift_track_id": pv.forklift_track_id,
                            "distance_px": pv.distance_px
                        }
                        if pv.distance_m is not None:
                            metadata["distance_m"] = pv.distance_m
                        
                        event = SafetyEvent(
                            event_type=EventType.PEDESTRIAN_FORKLIFT_PROXIMITY,
                            zone_id="proximity",
                            severity=Severity.HIGH,
                            track_id=pv.person_track_id,
                            duration_seconds=0.0,
                            metadata=metadata
                        )
                        if shared_state.alert_manager and shared_state.alert_manager.trigger_alert(event, frame):
                            # Log with meters if available, otherwise pixels
                            if pv.distance_m is not None:
                                print(f"\n⚠️ PROXIMITY ALERT: Person #{pv.person_track_id} <-> Forklift #{pv.forklift_track_id} ({pv.distance_m:.1f}m)")
                                prompt_context = f"PROXIMITY DANGER: Person near forklift ({pv.distance_m:.1f}m)"
                            else:
                                print(f"\n⚠️ PROXIMITY ALERT: Person #{pv.person_track_id} <-> Forklift #{pv.forklift_track_id} ({pv.distance_px:.0f}px)")
                                prompt_context = f"PROXIMITY DANGER: Person near forklift ({pv.distance_px:.0f}px)"
                            
                            if not vlm_trigger_queue.full():
                                vlm_trigger_queue.put_nowait(VLMTrigger(
                                    trigger_type=TriggerType.REFLEX,
                                    frame=frame.copy(),
                                    prompt_context=prompt_context,
                                    track_id=pv.person_track_id,
                                    bbox=pv.person_bbox,
                                    zone_id="proximity"
                                ))
                
                # ============================================================
                # P1: TTC Calculation & Alerts
                # ============================================================
                if ttc_calculator and all_detections:
                    # Separate persons and forklifts
                    persons = [d for d in all_detections if d.class_name == "person"]
                    forklifts = [d for d in all_detections if d.class_name != "person"]
                    
                    # Update TTC calculator with all track positions
                    dt = 1.0 / max(shared_state.metrics.get_fast_fps(), 10.0)
                    for det in all_detections:
                        ttc_calculator.update(det.track_id, det.bbox, dt)
                    
                    # Get top threat (closest TTC pair)
                    ttc_threat = ttc_calculator.get_top_threat(persons, forklifts)
                    shared_state.update_ttc_threat(ttc_threat)
                    
                    if ttc_threat:
                        # Build bbox dicts for visualization
                        persons_bboxes = {d.track_id: d.bbox for d in persons}
                        forklifts_bboxes = {d.track_id: d.bbox for d in forklifts}
                        
                        # Draw TTC overlay (threat line, countdown, velocity arrows)
                        vis.draw_ttc_overlay(vis_frame, ttc_threat, persons_bboxes, forklifts_bboxes)
                        
                        # Trigger TTC alert via AlertManager
                        if shared_state.alert_manager:
                            shared_state.alert_manager.check_and_trigger_ttc_alerts(ttc_threat, frame)
                
                if best_person:
                    predictor.update(best_person[1], best_person[0], best_person[2], now)
            else:
                prediction = predictor.predict(now)
                if prediction:
                    best_person = prediction

            # Draw Zones (Pass High Res Size)
            if shared_state.safety_enabled and shared_state.zone_manager:
                vis_frame = vis.draw_zone_overlays(vis_frame, shared_state.zone_manager, (HIGH_RES_W, HIGH_RES_H), shared_state.alert_manager, active_violations)

            # Control Logic & UI
            vlm_state = shared_state.get_state().value
            
            # P1 FIX: UI State Machine with safety-first priority
            # Check for active HIGH severity alerts FIRST
            ui_status = "LOST"  # Default
            has_danger_alert = False
            
            if shared_state.alert_manager:
                alert = shared_state.alert_manager.get_current_state()
                if alert.active and alert.severity and alert.severity.value == "HIGH":
                    ui_status = "DANGER"
                    has_danger_alert = True
            
            if best_person:
                tid, bbox, conf = best_person
                if target_id is None: target_id = tid
                error_x, cmd = compute_control(bbox, frame_center)
                shared_state.update_cv_context(tid, bbox, "TRACKING")
                vis_frame = vis.draw_target(vis_frame, bbox, tid, frame_center)
                
                # Use DANGER if alert active, otherwise MONITORING (not LOCKED)
                if not has_danger_alert:
                    ui_status = "MONITORING"
                shared_state.update_control_state(ui_status, cmd, tid, error_x)
            else:
                if vlm_state == "SEARCHING": target_id = None
                if not has_danger_alert:
                    ui_status = "LOST"
                shared_state.update_control_state(ui_status, "ROTATE", -1, 0)
                cmd = "ROTATE"
                tid = -1
                error_x = 0

            # Metrics & Display
            loop_elapsed_ms = (time.monotonic() - last_loop_time) * 1000
            shared_state.metrics.record_fast_loop(loop_elapsed_ms)
            last_loop_time = time.monotonic()
            
            # HUD
            vis_frame = vis.draw_hud(vis_frame, ui_status, cmd, tid, error_x, shared_state.vlm_processing, shared_state.metrics)
            
            # P0: Visual UNCALIBRATED banner on video frame (not just console)
            if mode_config.show_uncalibrated_banner:
                vis_frame = vis.draw_uncalibrated_banner(vis_frame, mode="DEMO")
            
            shared_state.set_annotated_frame(vis_frame)
            frame_count += 1
            
            if not args.quiet:
                fps = shared_state.metrics.get_fast_fps()
                sys.stdout.write(f"\rFPS:{fps:5.1f} | {ui_status:10} | ID:{tid:3} | {cmd:8}")
                sys.stdout.flush()
            
            if args.video and frame_delay > 0:
                elapsed = time.monotonic() - loop_start
                if frame_delay - elapsed > 0:
                    time.sleep(frame_delay - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        shared_state.stop()
        cap.release()
        if shared_state.event_store:
            shared_state.event_store.shutdown()

if __name__ == "__main__":
    main()