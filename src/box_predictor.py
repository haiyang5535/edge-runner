"""
CPU-based Box Predictor for Machina QoS
==========================================

Linear extrapolation predictor for bbox positions (<0.1ms per call).
Used on skip frames when YOLO detector is not running.

This is SOTA for drone/robot control - avoids control jitter from stale
bbox reuse (Zero-Order Hold) by predicting current position.
"""

import time


class BoxPredictor:
    """
    Lightweight linear predictor for bounding box positions.
    
    On detection frames: call update() with new bbox
    On skip frames: call predict() to get extrapolated position
    
    Uses center-point + size representation for smoother interpolation.
    """
    
    def __init__(self):
        self.last_box = None  # (cx, cy, w, h) center format
        self.velocity = [0.0, 0.0, 0.0, 0.0]  # pixels per second
        self.last_time = 0.0
        self.track_id = -1
        self.confidence = 0.0
    
    def xyxy_to_cxcywh(self, xyxy: list) -> tuple:
        """Convert [x1, y1, x2, y2] to (cx, cy, w, h)."""
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)
    
    def cxcywh_to_xyxy(self, cxcywh: tuple) -> list:
        """Convert (cx, cy, w, h) to [x1, y1, x2, y2]."""
        cx, cy, w, h = cxcywh
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def update(self, bbox_xyxy: list, track_id: int = -1, 
               confidence: float = 0.0, now: float = None) -> list:
        """
        Update with new detection. Call on detection frames.
        
        Args:
            bbox_xyxy: [x1, y1, x2, y2] bounding box
            track_id: ByteTrack ID
            confidence: Detection confidence
            now: Current timestamp (monotonic)
        
        Returns:
            Same bbox_xyxy (passthrough for convenience)
        """
        now = now or time.monotonic()
        box = self.xyxy_to_cxcywh(bbox_xyxy)
        
        if self.last_box and self.last_time:
            dt = now - self.last_time
            if dt > 0:
                # Calculate velocity (pixels per second)
                self.velocity = [(b - l) / dt for b, l in zip(box, self.last_box)]
        
        self.last_box = box
        self.last_time = now
        self.track_id = track_id
        self.confidence = confidence
        
        return bbox_xyxy
    
    def predict(self, now: float = None) -> tuple:
        """
        Predict current position. Call on skip frames.
        
        Args:
            now: Current timestamp (monotonic)
        
        Returns:
            Tuple of (track_id, bbox_xyxy, confidence) or None if no history
        """
        if not self.last_box:
            return None
        
        now = now or time.monotonic()
        dt = now - self.last_time
        
        # Linear extrapolation: x_new = x_old + v * dt
        pred_box = tuple(l + v * dt for l, v in zip(self.last_box, self.velocity))
        
        # Convert back to xyxy format
        bbox_xyxy = self.cxcywh_to_xyxy(pred_box)
        
        return (self.track_id, bbox_xyxy, self.confidence)
    
    def reset(self):
        """Reset when target lost."""
        self.last_box = None
        self.velocity = [0.0, 0.0, 0.0, 0.0]
        self.last_time = 0.0
        self.track_id = -1
        self.confidence = 0.0
    
    def has_target(self) -> bool:
        """Check if we have a valid target."""
        return self.last_box is not None
