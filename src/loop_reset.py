#!/usr/bin/env python3
"""
LoopResetManager - Centralized State Reset for Video Loop Mode
==============================================================

Manages clean state reset when video loops back to frame 0.
Prevents ghost boxes and stale tracks from persisting across loops.

Usage:
    loop_reset_mgr = LoopResetManager()
    loop_reset_mgr.register(bbox_smoother)
    loop_reset_mgr.register(motion_gate)
    
    # On video loop detected:
    loop_reset_mgr.reset_all()
"""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Resettable(Protocol):
    """Protocol for components that can be reset."""
    def reset(self) -> None:
        """Reset component state."""
        ...


class LoopResetManager:
    """
    Centralized state reset for video loop mode.
    
    When video loops, call reset_all() to clear:
    - Tracker state (handled separately via model.predictor.trackers)
    - BboxSmoother tracks
    - MotionGate history
    - BoxPredictor state
    - Zone dwell tracking
    
    Components must implement the Resettable protocol (have a reset() method).
    """
    
    def __init__(self):
        """Initialize LoopResetManager."""
        self.components: List[Resettable] = []
        self._reset_count = 0
    
    def register(self, component: Resettable) -> bool:
        """
        Register a component for reset management.
        
        Args:
            component: Object with a reset() method
            
        Returns:
            True if registered successfully
        """
        if not hasattr(component, 'reset') or not callable(getattr(component, 'reset')):
            print(f"⚠️ LoopResetManager: {type(component).__name__} has no reset() method")
            return False
        
        if component not in self.components:
            self.components.append(component)
            return True
        return False
    
    def unregister(self, component: Resettable) -> bool:
        """
        Unregister a component.
        
        Args:
            component: Previously registered component
            
        Returns:
            True if unregistered successfully
        """
        if component in self.components:
            self.components.remove(component)
            return True
        return False
    
    def reset_all(self):
        """
        Reset all registered components.
        
        Call this when video loops back to frame 0.
        """
        self._reset_count += 1
        errors = []
        
        for component in self.components:
            try:
                component.reset()
            except Exception as e:
                errors.append(f"{type(component).__name__}: {e}")
        
        if errors:
            print(f"⚠️ LoopResetManager: Some resets failed: {errors}")
        else:
            component_names = [type(c).__name__ for c in self.components]
            print(f"🔄 LoopResetManager: Reset {len(self.components)} components: {component_names}")
    
    @property
    def reset_count(self) -> int:
        """Number of times reset_all() has been called."""
        return self._reset_count
    
    @property
    def component_count(self) -> int:
        """Number of registered components."""
        return len(self.components)
    
    def get_registered_names(self) -> List[str]:
        """Get names of registered components."""
        return [type(c).__name__ for c in self.components]


class ZoneDwellWrapper:
    """
    Wrapper to make ZoneManager compatible with LoopResetManager.
    
    ZoneManager has reset_dwell_all() instead of reset(), so we
    wrap it to provide the expected interface.
    """
    
    def __init__(self, zone_manager):
        """
        Initialize wrapper.
        
        Args:
            zone_manager: ZoneManager instance with reset_dwell_all() method
        """
        self.zone_manager = zone_manager
    
    def reset(self):
        """Reset dwell tracking for all zones."""
        if hasattr(self.zone_manager, 'reset_dwell_all'):
            self.zone_manager.reset_dwell_all()
        elif hasattr(self.zone_manager, 'dwell_tracker'):
            # Fallback: directly clear dwell trackers
            for zone_id in list(self.zone_manager.dwell_tracker.keys()):
                self.zone_manager.dwell_tracker[zone_id].clear()
                if hasattr(self.zone_manager, 'last_seen'):
                    self.zone_manager.last_seen.get(zone_id, {}).clear()


# ============================================================
# Module-level singleton
# ============================================================

_loop_reset_manager: LoopResetManager = None


def get_loop_reset_manager() -> LoopResetManager:
    """Get or create singleton LoopResetManager instance."""
    global _loop_reset_manager
    if _loop_reset_manager is None:
        _loop_reset_manager = LoopResetManager()
    return _loop_reset_manager


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Create test components
    class MockSmoother:
        def __init__(self):
            self.tracks = {"track1": [1, 2, 3], "track2": [4, 5, 6]}
        
        def reset(self):
            self.tracks.clear()
            print("  MockSmoother reset")
    
    class MockMotionGate:
        def __init__(self):
            self.history = {10: [(1, 2), (3, 4)]}
            self.confirmed = {10: True}
        
        def reset(self):
            self.history.clear()
            self.confirmed.clear()
            print("  MockMotionGate reset")
    
    class MockPredictor:
        def __init__(self):
            self.last_box = (100, 200, 300, 400)
        
        def reset(self):
            self.last_box = None
            print("  MockPredictor reset")
    
    # Create manager and register components
    manager = LoopResetManager()
    
    smoother = MockSmoother()
    gate = MockMotionGate()
    predictor = MockPredictor()
    
    manager.register(smoother)
    manager.register(gate)
    manager.register(predictor)
    
    print(f"Registered {manager.component_count} components:")
    print(f"  {manager.get_registered_names()}")
    
    # Verify initial state
    print(f"\nBefore reset:")
    print(f"  Smoother tracks: {smoother.tracks}")
    print(f"  MotionGate confirmed: {gate.confirmed}")
    print(f"  Predictor last_box: {predictor.last_box}")
    
    # Reset all
    print(f"\nCalling reset_all():")
    manager.reset_all()
    
    # Verify reset state
    print(f"\nAfter reset:")
    print(f"  Smoother tracks: {smoother.tracks}")
    print(f"  MotionGate confirmed: {gate.confirmed}")
    print(f"  Predictor last_box: {predictor.last_box}")
    
    print(f"\nReset count: {manager.reset_count}")
    print("\nLoopResetManager test complete!")
