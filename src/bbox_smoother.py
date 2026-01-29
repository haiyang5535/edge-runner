#!/usr/bin/env python3
"""
BboxSmoother - Multi-Object Bounding Box Stabilization
=======================================================

Reduces bbox flickering through:
1. Hysteresis filter: Only show detection after N consecutive frames
2. EMA smoothing: Exponential moving average for bbox coordinates
3. Track persistence: Keep displaying bbox briefly after detection lost

Usage:
    smoother = BboxSmoother()
    stable_detections = smoother.smooth(raw_detections)
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .zone_manager import TrackedObject


@dataclass
class SmoothedTrack:
    """Smoothed track data."""
    track_id: int
    class_name: str
    bbox: List[int]  # Smoothed [x1, y1, x2, y2]
    confidence: float
    consecutive_frames: int  # Frames detected in a row
    last_seen: float  # timestamp
    is_stable: bool  # Passed hysteresis threshold
    is_stale: bool = False  # True when in persistence-only state (not in current frame)


class BboxSmoother:
    """
    Multi-object bbox smoother with hysteresis and EMA.
    
    - Hysteresis: Requires N consecutive detections before showing
    - EMA: Smooths bbox coordinates to reduce jitter
    - Persistence: Keeps bbox visible briefly after detection lost
    """
    
    # Tunable parameters (tuned for forklift stability)
    HYSTERESIS_FRAMES = 2     # Require 2 consecutive frames to show (faster response)
    EMA_ALPHA = 0.35          # EMA weight (smoother transitions)
    PERSISTENCE_SEC = 1.5     # Keep bbox for 1.5s after lost (forklift doesn't vanish)
    
    def __init__(
        self,
        hysteresis_frames: int = None,
        ema_alpha: float = None,
        persistence_sec: float = None
    ):
        if hysteresis_frames:
            self.HYSTERESIS_FRAMES = hysteresis_frames
        if ema_alpha:
            self.EMA_ALPHA = ema_alpha
        if persistence_sec:
            self.PERSISTENCE_SEC = persistence_sec
        
        # Track storage: track_id -> SmoothedTrack
        self.tracks: Dict[int, SmoothedTrack] = {}
    
    def _ema_bbox(self, old_bbox: List[int], new_bbox: List[int]) -> List[int]:
        """Apply EMA smoothing to bbox coordinates."""
        alpha = self.EMA_ALPHA
        return [
            int(alpha * n + (1 - alpha) * o) 
            for o, n in zip(old_bbox, new_bbox)
        ]
    
    def smooth(
        self, 
        detections: List[TrackedObject],
        now: float = None
    ) -> List[TrackedObject]:
        """
        Apply smoothing to raw detections.
        
        Args:
            detections: Raw detections from YOLO tracker
            now: Current timestamp (optional, uses time.monotonic())
            
        Returns:
            Smoothed detections (only stable tracks)
        """
        now = now or time.monotonic()
        current_ids = set()
        
        # Update tracks with current detections
        for det in detections:
            tid = det.track_id
            current_ids.add(tid)
            
            if tid in self.tracks:
                # Existing track: update with EMA smoothing
                track = self.tracks[tid]
                track.bbox = self._ema_bbox(track.bbox, det.bbox)
                track.confidence = det.confidence
                track.consecutive_frames += 1
                track.last_seen = now
                track.is_stable = track.consecutive_frames >= self.HYSTERESIS_FRAMES
            else:
                # New track: initialize
                self.tracks[tid] = SmoothedTrack(
                    track_id=tid,
                    class_name=det.class_name,
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    consecutive_frames=1,
                    last_seen=now,
                    is_stable=False  # Need more frames
                )
        
        # Handle missing tracks (track persistence)
        stable_outputs = []
        tracks_to_remove = []
        
        for tid, track in self.tracks.items():
            if tid not in current_ids:
                # Track not detected this frame - mark as stale
                track.is_stale = True
                time_since_seen = now - track.last_seen
                if time_since_seen > self.PERSISTENCE_SEC:
                    # Track is too stale, remove it
                    tracks_to_remove.append(tid)
                else:
                    # Still in persistence window, keep showing if was stable
                    if track.is_stable:
                        stable_outputs.append(self._to_tracked_object(track))
            else:
                # Track detected this frame - not stale
                track.is_stale = False
                if track.is_stable:
                    stable_outputs.append(self._to_tracked_object(track))
        
        # Remove stale tracks
        for tid in tracks_to_remove:
            del self.tracks[tid]
        
        return stable_outputs
    
    def _to_tracked_object(self, track: SmoothedTrack) -> TrackedObject:
        """Convert SmoothedTrack to TrackedObject for compatibility."""
        return TrackedObject(
            track_id=track.track_id,
            class_name=track.class_name,
            bbox=track.bbox,
            center=((track.bbox[0] + track.bbox[2]) // 2, 
                    (track.bbox[1] + track.bbox[3]) // 2),
            confidence=track.confidence
        )
    
    def get_track_info(self, track_id: int) -> Optional[SmoothedTrack]:
        """Get info for a specific track."""
        return self.tracks.get(track_id)
    
    def reset(self):
        """Clear all tracked data."""
        self.tracks.clear()


# Module-level singleton
_forklift_smoother: Optional[BboxSmoother] = None


def get_forklift_smoother(**kwargs) -> BboxSmoother:
    """Get or create singleton BboxSmoother for forklifts."""
    global _forklift_smoother
    if _forklift_smoother is None:
        _forklift_smoother = BboxSmoother(**kwargs)
    return _forklift_smoother


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    from .zone_manager import TrackedObject
    
    smoother = BboxSmoother(hysteresis_frames=2)
    
    # Simulate detections over multiple frames
    for frame_idx in range(5):
        # Forklift appears in frames 0-3, disappears in frame 4
        if frame_idx < 4:
            det = TrackedObject(
                track_id=10,
                class_name="forklift",
                bbox=[500 + frame_idx*5, 300, 700 + frame_idx*5, 500],
                center=(600, 400),
                confidence=0.8
            )
            stable = smoother.smooth([det])
        else:
            stable = smoother.smooth([])  # Forklift gone
        
        print(f"Frame {frame_idx}: {len(stable)} stable detections")
        for s in stable:
            print(f"  - {s.class_name} #{s.track_id} bbox={s.bbox}")
    
    print("\nBboxSmoother test complete!")
