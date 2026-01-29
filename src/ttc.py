#!/usr/bin/env python3
"""
TTC Calculator — Time-to-Collision Predictive Logic
====================================================

Computes Time-to-Collision from track velocities for predictive alerting.
Requires BEV calibration for accurate distance/velocity estimation.

Usage:
    ttc_calc = TTCCalculator(ground_plane)
    ttc_calc.update(track_id, bbox, dt)
    threat = ttc_calc.get_top_threat(persons, forklifts)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

# Forward reference for type hints
try:
    from .ground_plane import GroundPlane
except ImportError:
    GroundPlane = None


@dataclass
class TTCPair:
    """Threat pair with TTC calculation."""
    person_id: int
    forklift_id: int
    distance_m: float
    ttc_seconds: float
    approaching: bool
    person_velocity: Tuple[float, float]  # (vx, vy) in m/s
    forklift_velocity: Tuple[float, float]


class TTCCalculator:
    """
    Time-to-Collision calculator from track velocities.
    
    Estimates velocity using position history and EMA smoothing,
    then computes TTC based on relative velocity and position.
    """
    
    # Alert thresholds
    DANGER_TTC_SEC = 3.0      # Alert if collision in <3 seconds
    WARNING_TTC_SEC = 5.0     # Warn if collision in <5 seconds
    MIN_VELOCITY_MPS = 0.1    # Ignore stationary objects
    
    # Velocity estimation
    HISTORY_SIZE = 10         # Frames of velocity history
    EMA_ALPHA = 0.3           # Velocity smoothing factor
    
    def __init__(self, ground_plane: Optional['GroundPlane'] = None):
        """
        Initialize TTC calculator.
        
        Args:
            ground_plane: GroundPlane for BEV projection (required for accuracy)
        """
        self.ground_plane = ground_plane
        self.use_bev = ground_plane is not None and getattr(ground_plane, 'is_calibrated', False)
        
        # Position history: track_id -> deque of (x, y, timestamp)
        self.positions: Dict[int, deque] = {}
        
        # Smoothed velocities: track_id -> (vx, vy) in m/s or pixels/s
        self.velocities: Dict[int, Tuple[float, float]] = {}
        
        # Last known positions for TTC calculation
        self.last_positions: Dict[int, Tuple[float, float]] = {}
    
    def _get_footpoint(self, bbox: List[int]) -> Tuple[int, int]:
        """Get bottom-center of bbox."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)
    
    def _project_to_bev(self, foot_px: Tuple[int, int]) -> Tuple[float, float]:
        """Project pixel footpoint to BEV coordinates (meters)."""
        if self.use_bev and self.ground_plane is not None:
            try:
                return self.ground_plane.project_to_bev(foot_px)
            except Exception:
                pass
        # Fallback to pixels (less accurate)
        return (float(foot_px[0]), float(foot_px[1]))
    
    def update(self, track_id: int, bbox: List[int], dt: float):
        """
        Update velocity estimate for a track.
        
        Args:
            track_id: Track identifier
            bbox: [x1, y1, x2, y2] bounding box
            dt: Time delta since last update (seconds)
        """
        if dt <= 0:
            return
        
        foot_px = self._get_footpoint(bbox)
        pos = self._project_to_bev(foot_px)
        
        # Initialize position history if needed
        if track_id not in self.positions:
            self.positions[track_id] = deque(maxlen=self.HISTORY_SIZE)
        
        # Calculate velocity from position change
        if track_id in self.last_positions:
            last_pos = self.last_positions[track_id]
            vx = (pos[0] - last_pos[0]) / dt
            vy = (pos[1] - last_pos[1]) / dt
            
            # EMA smoothing
            if track_id in self.velocities:
                old_vx, old_vy = self.velocities[track_id]
                vx = self.EMA_ALPHA * vx + (1 - self.EMA_ALPHA) * old_vx
                vy = self.EMA_ALPHA * vy + (1 - self.EMA_ALPHA) * old_vy
            
            self.velocities[track_id] = (vx, vy)
        
        self.last_positions[track_id] = pos
    
    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Get smoothed velocity for a track."""
        return self.velocities.get(track_id)
    
    def get_speed(self, track_id: int) -> float:
        """Get speed magnitude for a track."""
        vel = self.velocities.get(track_id)
        if vel is None:
            return 0.0
        return math.sqrt(vel[0]**2 + vel[1]**2)
    
    def compute_ttc(self, person_id: int, forklift_id: int) -> Optional[float]:
        """
        Compute Time-to-Collision between person and forklift.
        
        Uses relative position and velocity to estimate collision time.
        Returns None if tracks not found, infinity if not approaching.
        
        Args:
            person_id: Person track ID
            forklift_id: Forklift track ID
            
        Returns:
            TTC in seconds, or None if cannot calculate
        """
        if person_id not in self.last_positions or forklift_id not in self.last_positions:
            return None
        
        p_pos = np.array(self.last_positions[person_id])
        f_pos = np.array(self.last_positions[forklift_id])
        
        p_vel = np.array(self.velocities.get(person_id, (0, 0)))
        f_vel = np.array(self.velocities.get(forklift_id, (0, 0)))
        
        # Relative position and velocity
        rel_pos = p_pos - f_pos
        rel_vel = p_vel - f_vel
        
        # Check if approaching (dot product < 0 means getting closer)
        closing_rate = -np.dot(rel_pos, rel_vel)
        if closing_rate <= 0:
            return float('inf')  # Not approaching
        
        # Distance
        distance = np.linalg.norm(rel_pos)
        if distance < 0.1:  # Already very close
            return 0.0
        
        # TTC = distance / closing_rate (simplified linear model)
        vel_sq = np.dot(rel_vel, rel_vel)
        if vel_sq < 1e-6:
            return float('inf')  # Stationary
        
        ttc = (distance * distance) / closing_rate
        return max(0, ttc)
    
    def get_distance(self, id1: int, id2: int) -> Optional[float]:
        """Get distance between two tracks."""
        if id1 not in self.last_positions or id2 not in self.last_positions:
            return None
        
        pos1 = np.array(self.last_positions[id1])
        pos2 = np.array(self.last_positions[id2])
        return float(np.linalg.norm(pos1 - pos2))
    
    def get_top_threat(self, persons: List, forklifts: List) -> Optional[TTCPair]:
        """
        Find the closest TTC pair among all person-forklift combinations.
        
        Args:
            persons: List of TrackedObject for persons
            forklifts: List of TrackedObject for forklifts
            
        Returns:
            TTCPair with minimum TTC, or None if no threats
        """
        if not persons or not forklifts:
            return None
        
        min_ttc = float('inf')
        threat = None
        
        for p in persons:
            p_id = p.track_id
            if p_id not in self.last_positions:
                continue
            
            for f in forklifts:
                f_id = f.track_id
                if f_id not in self.last_positions:
                    continue
                
                ttc = self.compute_ttc(p_id, f_id)
                if ttc is None:
                    continue
                
                if ttc < min_ttc:
                    min_ttc = ttc
                    distance = self.get_distance(p_id, f_id)
                    
                    threat = TTCPair(
                        person_id=p_id,
                        forklift_id=f_id,
                        distance_m=distance if distance else 0.0,
                        ttc_seconds=ttc,
                        approaching=ttc < float('inf'),
                        person_velocity=self.velocities.get(p_id, (0, 0)),
                        forklift_velocity=self.velocities.get(f_id, (0, 0))
                    )
        
        # Only return if within danger threshold
        if threat and threat.ttc_seconds < self.DANGER_TTC_SEC:
            return threat
        
        return None
    
    def reset(self):
        """Clear all velocity/position history (for LoopResetManager)."""
        self.positions.clear()
        self.velocities.clear()
        self.last_positions.clear()
    
    def reset_track(self, track_id: int):
        """Reset history for a specific track."""
        self.positions.pop(track_id, None)
        self.velocities.pop(track_id, None)
        self.last_positions.pop(track_id, None)


# ============================================================
# Module-level singleton
# ============================================================

_ttc_calculator: Optional[TTCCalculator] = None


def get_ttc_calculator(ground_plane: Optional['GroundPlane'] = None) -> TTCCalculator:
    """Get or create singleton TTCCalculator instance."""
    global _ttc_calculator
    if _ttc_calculator is None:
        _ttc_calculator = TTCCalculator(ground_plane=ground_plane)
    return _ttc_calculator


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    from dataclasses import dataclass as dc
    
    @dc
    class MockTrackedObject:
        track_id: int
        bbox: List[int]
        class_name: str
    
    print("TTC Calculator Tests")
    print("=" * 40)
    
    # Create calculator (pixel mode, no BEV)
    calc = TTCCalculator()
    
    # Simulate person standing still
    person_bbox = [100, 200, 150, 300]
    
    # Simulate forklift approaching
    for i in range(10):
        # Forklift moves 20 pixels per frame toward person
        forklift_bbox = [500 - i*20, 200, 600 - i*20, 350]
        
        calc.update(1, person_bbox, dt=0.033)  # Person (stationary)
        calc.update(2, forklift_bbox, dt=0.033)  # Forklift (moving)
        
        ttc = calc.compute_ttc(1, 2)
        dist = calc.get_distance(1, 2)
        
        if i > 1:  # Need at least 2 updates for velocity
            print(f"Frame {i}: distance={dist:.0f}px, TTC={ttc:.2f}s")
    
    # Test get_top_threat
    person = MockTrackedObject(track_id=1, bbox=person_bbox, class_name="person")
    forklift = MockTrackedObject(track_id=2, bbox=[300, 200, 400, 350], class_name="forklift")
    
    threat = calc.get_top_threat([person], [forklift])
    if threat:
        print(f"\nTop threat: person #{threat.person_id} vs forklift #{threat.forklift_id}")
        print(f"  Distance: {threat.distance_m:.1f}px, TTC: {threat.ttc_seconds:.2f}s")
    else:
        print("\nNo imminent threat detected")
    
    print("\nTTC Calculator tests complete!")
