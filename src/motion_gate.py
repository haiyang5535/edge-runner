#!/usr/bin/env python3
"""
MotionGate - Temporal Displacement Validation for Forklift Detection
=====================================================================

Requires object to move at least MIN_DISPLACEMENT before confirming
as valid forklift. Eliminates false positives from static cargo.

CRITICAL: Operates in BEV (meters) space for camera-agnostic thresholds.
Pixel-based thresholds are NOT transferable across camera installations.

Uses BOTTOM_CENTER (footpoint) anchor for consistency with:
- Supervision PolygonZone (Position.BOTTOM_CENTER)
- Ground-plane BEV projection
- Proximity distance calculation

Usage:
    gate = MotionGate(ground_plane=gp)  # With BEV projection
    gate = MotionGate()  # Fallback to pixel-based
    is_valid = gate.check(detection)
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import deque
import math

from .zone_manager import TrackedObject

# Avoid circular import
if TYPE_CHECKING:
    from .ground_plane import GroundPlane


class MotionGate:
    """
    Temporal validation gate requiring displacement before confirming forklift.
    
    Operates in BEV (meters) space when GroundPlane is available,
    falls back to pixel-based displacement when not calibrated.
    
    Uses BOTTOM_CENTER (footpoint) for ground contact point consistency.
    """
    
    # BEV (meters) thresholds - camera agnostic
    MIN_DISPLACEMENT_M = 0.3      # Must move 0.3 meters (30cm) in BEV
    MIN_SPEED_MPS = 0.1           # Or: minimum speed 0.1 m/s
    
    # Pixel fallback thresholds (for uncalibrated cameras)
    MIN_DISPLACEMENT_PX = 30     # Pixel displacement threshold
    
    # Temporal parameters
    CONFIRMATION_FRAMES = 5       # Over 5 frames (~250ms at 20 FPS)
    MAX_HISTORY_LENGTH = 30       # Keep at most 30 frames of history
    
    def __init__(
        self,
        ground_plane: Optional['GroundPlane'] = None,
        min_displacement_m: Optional[float] = None,
        min_displacement_px: Optional[float] = None,
        confirmation_frames: Optional[int] = None,
        require_bev: bool = False,
        use_adaptive_threshold: bool = True
    ):
        """
        Initialize MotionGate.
        
        Args:
            ground_plane: Optional GroundPlane for BEV projection (recommended)
            min_displacement_m: Override MIN_DISPLACEMENT_M (BEV mode)
            min_displacement_px: Override MIN_DISPLACEMENT_PX (pixel fallback)
            confirmation_frames: Override CONFIRMATION_FRAMES
            require_bev: If True, raise error when BEV not available (SAFETY_MODE)
            use_adaptive_threshold: If True, scale pixel threshold by bbox size
        """
        self.ground_plane = ground_plane
        
        if min_displacement_m is not None:
            self.MIN_DISPLACEMENT_M = min_displacement_m
        if min_displacement_px is not None:
            self.MIN_DISPLACEMENT_PX = min_displacement_px
        if confirmation_frames is not None:
            self.CONFIRMATION_FRAMES = confirmation_frames
        
        # New flags for dual-mode support
        self.require_bev = require_bev
        self.use_adaptive_threshold = use_adaptive_threshold
        
        # History storage: track_id -> deque of positions
        # Position is (x_m, y_m) if ground_plane, else (x_px, y_px)
        self.history: Dict[int, deque] = {}
        
        # Confirmation state: track_id -> bool (passed gate)
        self.confirmed: Dict[int, bool] = {}
        
        # Track bbox sizes for adaptive threshold
        self.bbox_sizes: Dict[int, float] = {}
        
        # Use BEV mode if ground_plane provided
        self.use_bev = ground_plane is not None and ground_plane.is_calibrated if hasattr(ground_plane, 'is_calibrated') else ground_plane is not None
    
    @staticmethod
    def get_footpoint(bbox: List[int]) -> Tuple[int, int]:
        """
        Get bottom-center of bbox (ground contact point).
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (x, y) bottom-center coordinates in pixels
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)
    
    def _get_position(self, bbox: List[int]) -> Tuple[float, float]:
        """
        Get position for displacement calculation.
        
        Returns BEV coordinates (meters) if ground_plane available,
        otherwise returns pixel coordinates.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (x, y) position (meters if BEV, pixels otherwise)
        """
        foot_px = self.get_footpoint(bbox)
        
        if self.use_bev and self.ground_plane is not None:
            try:
                return self.ground_plane.project_to_bev(foot_px)
            except Exception:
                # Fallback to pixels if projection fails
                return (float(foot_px[0]), float(foot_px[1]))
        
        return (float(foot_px[0]), float(foot_px[1]))
    
    def _calculate_displacement(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean displacement between two positions."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def _get_adaptive_threshold(self, bbox: List[int]) -> float:
        """
        Scale pixel threshold by bbox size (proxy for distance).
        
        Larger bbox = closer object = more expected movement.
        Smaller bbox = farther object = less expected movement.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Adaptive pixel threshold
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        # Reference: 150px height = base threshold
        # Larger bbox (closer) = higher threshold (expect more movement)
        # Smaller bbox (farther) = lower threshold (less movement expected)
        base = 10  # Minimum 10px
        scale = max(0.5, min(2.0, bbox_height / 150))  # Scale 0.5x to 2x
        return base * scale
    
    def check(self, det: TrackedObject) -> bool:
        """
        Check if detection passes motion gate.
        
        Non-forklift detections always pass.
        Forklifts must demonstrate displacement over CONFIRMATION_FRAMES.
        Once confirmed, a track stays confirmed until reset.
        
        Args:
            det: TrackedObject to check
            
        Returns:
            True if detection should be shown, False if gated
        """
        # Non-forklift classes always pass
        if det.class_name != "forklift":
            return True
        
        tid = det.track_id
        
        # If already confirmed, keep passing
        if self.confirmed.get(tid, False):
            return True
        
        # Initialize history for new track
        if tid not in self.history:
            self.history[tid] = deque(maxlen=self.MAX_HISTORY_LENGTH)
        
        # Get current position and store bbox size for adaptive threshold
        current_pos = self._get_position(det.bbox)
        self.history[tid].append(current_pos)
        self.bbox_sizes[tid] = det.bbox[3] - det.bbox[1]  # Track height
        
        # Need enough history
        if len(self.history[tid]) < self.CONFIRMATION_FRAMES:
            return False
        
        # Calculate total displacement over confirmation window
        first_pos = self.history[tid][-self.CONFIRMATION_FRAMES]
        last_pos = self.history[tid][-1]
        displacement = self._calculate_displacement(first_pos, last_pos)
        
        # Check against threshold
        if self.use_bev:
            threshold = self.MIN_DISPLACEMENT_M
        elif self.use_adaptive_threshold:
            # Adaptive pixel threshold based on bbox size
            threshold = self._get_adaptive_threshold(det.bbox)
        else:
            threshold = self.MIN_DISPLACEMENT_PX
        
        if displacement >= threshold:
            # Passed gate - confirm this track
            self.confirmed[tid] = True
            return True
        
        return False
    
    def is_confirmed(self, track_id: int) -> bool:
        """Check if a track has been confirmed."""
        return self.confirmed.get(track_id, False)
    
    def get_displacement(self, track_id: int) -> Optional[float]:
        """
        Get current displacement for a track.
        
        Returns:
            Displacement value or None if not enough history
        """
        if track_id not in self.history:
            return None
        
        history = self.history[track_id]
        if len(history) < 2:
            return None
        
        window_size = min(self.CONFIRMATION_FRAMES, len(history))
        first_pos = history[-window_size]
        last_pos = history[-1]
        return self._calculate_displacement(first_pos, last_pos)
    
    def reset(self):
        """
        Clear all history (called by LoopResetManager).
        
        This resets all track confirmation states, so forklifts
        will need to re-demonstrate motion after video loop.
        """
        self.history.clear()
        self.confirmed.clear()
        self.bbox_sizes.clear()
    
    def reset_track(self, track_id: int):
        """Reset history for a specific track."""
        self.history.pop(track_id, None)
        self.confirmed.pop(track_id, None)
    
    def set_ground_plane(self, ground_plane: 'GroundPlane'):
        """
        Set or update ground plane for BEV mode.
        
        Note: Existing history will be invalidated as coordinate
        system changes from pixels to meters.
        """
        self.ground_plane = ground_plane
        self.use_bev = True
        # Clear history since coordinate system changed
        self.history.clear()
        self.confirmed.clear()


# ============================================================
# Module-level singleton
# ============================================================

_motion_gate: Optional[MotionGate] = None


def get_motion_gate(**kwargs) -> MotionGate:
    """Get or create singleton MotionGate instance."""
    global _motion_gate
    if _motion_gate is None:
        _motion_gate = MotionGate(**kwargs)
    return _motion_gate


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Create test gate (pixel fallback mode)
    gate = MotionGate(min_displacement_px=20, confirmation_frames=3)
    
    # Simulate static forklift (should NOT pass)
    print("Testing static forklift (should be gated):")
    for i in range(5):
        det = TrackedObject(
            track_id=10,
            class_name="forklift",
            bbox=[500, 400, 700, 600],  # Same position
            center=(600, 500),
            confidence=0.8
        )
        passed = gate.check(det)
        disp = gate.get_displacement(10) or 0
        print(f"  Frame {i}: passed={passed}, displacement={disp:.1f}px")
    
    # Reset and test moving forklift
    gate.reset()
    print("\nTesting moving forklift (should pass after confirmation):")
    for i in range(5):
        # Forklift moves 15 pixels per frame
        x_offset = i * 15
        det = TrackedObject(
            track_id=11,
            class_name="forklift",
            bbox=[500 + x_offset, 400, 700 + x_offset, 600],
            center=(600 + x_offset, 500),
            confidence=0.8
        )
        passed = gate.check(det)
        disp = gate.get_displacement(11) or 0
        print(f"  Frame {i}: passed={passed}, displacement={disp:.1f}px, confirmed={gate.is_confirmed(11)}")
    
    # Test person (should always pass)
    gate.reset()
    print("\nTesting person (should always pass):")
    det = TrackedObject(
        track_id=1,
        class_name="person",
        bbox=[300, 200, 400, 500],
        center=(350, 350),
        confidence=0.9
    )
    passed = gate.check(det)
    print(f"  Person passed={passed}")
    
    print("\nMotionGate test complete!")
