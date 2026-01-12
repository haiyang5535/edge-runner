#!/usr/bin/env python3
"""
EdgeRunner CV Loop - TensorRT Accelerated
Step 2: CV Pipeline with Person Tracking and Control Logic
"""
from ultralytics import YOLO
import cv2
import time


def compute_control(bbox, frame_center=(320, 240), threshold=50):
    """
    Calculate control commands from bounding box
    
    Args:
        bbox: [x1, y1, x2, y2]
        frame_center: Frame center point
        threshold: Dead zone threshold (pixels)
    
    Returns:
        dict: Control commands
    """
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2
    
    error_x = bbox_center_x - frame_center[0]
    error_y = bbox_center_y - frame_center[1]
    
    if abs(error_x) > threshold:
        direction = "left" if error_x < 0 else "right"
        return {"action": "turn", "direction": direction, "error": abs(error_x)}
    else:
        return {"action": "forward", "error": 0}


def main():
    # Load TensorRT engine directly (fast, low VRAM)
    # Note: first load may be slow (~30s), fast afterwards
    print("Loading YOLOv8n TensorRT engine...")
    model = YOLO('yolov8n.engine', task='detect')
    print("Model loaded successfully.")

    # V4L2 direct mode (more stable than GStreamer)
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit(1)

    print("Camera opened successfully. Press Ctrl+C to exit.")
    print("-" * 50)

    frame_count = 0
    start_time = time.time()
    target_id = None  # Tracking target ID

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Frame drop")
                continue
            
            # Inference (ByteTrack tracking)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # Extract person detections
            best_person = None
            best_conf = 0
            
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:  # class 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    # If target ID exists, prioritize tracking it
                    if target_id is not None and track_id == target_id:
                        best_person = (track_id, [x1, y1, x2, y2], conf)
                        break
                    # Otherwise select highest confidence
                    elif conf > best_conf:
                        best_conf = conf
                        best_person = (track_id, [x1, y1, x2, y2], conf)
            
            # Process detection results
            if best_person:
                track_id, bbox, conf = best_person
                
                # Lock onto first detected target
                if target_id is None and track_id != -1:
                    target_id = track_id
                    print(f"🎯 Locked target: Person ID {target_id}")
                
                # Calculate control commands
                control = compute_control(bbox)
                
                # Print status every 30 frames
                if frame_count % 30 == 0:
                    print(f"Person ID:{track_id} conf:{conf:.2f} bbox:{bbox} → {control}")
            else:
                if frame_count % 30 == 0:
                    print("No person detected")
            
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 FPS: {fps:.1f} | Frames: {frame_count}")

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Stopping...")
        elapsed = time.time() - start_time
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {frame_count / elapsed:.1f}")
        print(f"Runtime: {elapsed:.1f}s")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
