#!/usr/bin/env python3
"""
Machina State Machine
=======================

Control state management based on MachinaDecision contracts.
Maps VLM decisions to control states and actions.

Supports both:
- New MachinaDecision format (recommended)
- Legacy {detected, confidence} format (backward compat)
"""

from enum import Enum
from typing import Optional, Union
from dataclasses import dataclass

# Import will work when running as module
try:
    from .machina_schema import MachinaDecision, Decision, ActionType, SAFE_FALLBACK
except ImportError:
    # Fallback for direct script execution
    from machina_schema import MachinaDecision, Decision, ActionType, SAFE_FALLBACK


class ControlState(str, Enum):
    """High-level control states."""
    TRACKING = "TRACKING"
    LOST = "LOST"
    SEARCHING = "SEARCHING"
    ALERT = "ALERT"
    STOPPED = "STOPPED"


class ControlCommand(str, Enum):
    """Control commands for motor/actuator."""
    FOLLOW = "FOLLOW"       # Follow the target
    SLOW_TURN = "SLOW_TURN" # Lost target, slow search
    ROTATE = "ROTATE"       # Searching, rotate to find
    STOP = "STOP"           # Emergency stop
    HOLD = "HOLD"           # Hold position


@dataclass
class StateTransition:
    """Record of a state transition for audit."""
    from_state: ControlState
    to_state: ControlState
    trigger: str  # What caused the transition
    decision_id: str


class MachinaStateMachine:
    """
    Machina-aware state machine for target tracking control.
    
    Maps MachinaDecision outputs to control states and commands.
    
    State Mapping:
        VLM Decision  -> Control State
        SAFE          -> TRACKING
        SUSPICIOUS    -> ALERT (but keep tracking)
        BREACH        -> STOPPED
        UNKNOWN       -> LOST or SEARCHING
    
    Action Mapping:
        VLM Action    -> Control Command
        FOLLOW        -> FOLLOW
        STOP          -> STOP
        ALERT         -> FOLLOW (but with alert flag)
        SEARCH        -> ROTATE
        LOG_ONLY      -> HOLD
    
    Fallback Behavior:
        - If fallback_used=True, don't change state (hold previous)
        - Count consecutive fallbacks
        - If too many, switch to SEARCHING
    """
    
    def __init__(self, max_consecutive_fallbacks: int = 5):
        self.state = ControlState.SEARCHING
        self.command = ControlCommand.ROTATE
        self.target_id: Optional[int] = None
        
        # Decision tracking
        self.last_decision: Optional[MachinaDecision] = None
        self.consecutive_fallbacks = 0
        self.consecutive_unknown = 0
        self.max_consecutive_fallbacks = max_consecutive_fallbacks
        
        # Transition history for audit
        self.transitions: list[StateTransition] = []
        self.max_history = 50
    
    def update(self, decision: Union[MachinaDecision, dict]) -> ControlState:
        """
        Update state machine from VLM decision.
        
        Args:
            decision: MachinaDecision or legacy dict format
        
        Returns:
            Current control state
        """
        # Handle legacy format
        if isinstance(decision, dict):
            return self._update_legacy(decision)
        
        # Store decision
        self.last_decision = decision
        
        # Handle fallback (don't change state, just count)
        if decision.runtime.fallback_used:
            self.consecutive_fallbacks += 1
            if self.consecutive_fallbacks >= self.max_consecutive_fallbacks:
                self._transition_to(ControlState.SEARCHING, 
                                   f"Too many fallbacks ({self.consecutive_fallbacks})",
                                   decision.decision_id)
            return self.state
        
        # Reset fallback counter on successful decision
        self.consecutive_fallbacks = 0
        
        # Update target ID if we have one
        if decision.fast_path.track_id >= 0:
            self.target_id = decision.fast_path.track_id
        
        # Map VLM decision to state
        prev_state = self.state
        vlm_decision = decision.slow_path.decision
        vlm_action = decision.action.type
        
        if vlm_decision == Decision.SAFE:
            self.consecutive_unknown = 0
            new_state = ControlState.TRACKING
            new_command = self._map_action_to_command(vlm_action)
        
        elif vlm_decision == Decision.SUSPICIOUS:
            self.consecutive_unknown = 0
            new_state = ControlState.ALERT
            new_command = ControlCommand.FOLLOW  # Keep tracking but alert
        
        elif vlm_decision == Decision.BREACH:
            self.consecutive_unknown = 0
            new_state = ControlState.STOPPED
            new_command = ControlCommand.STOP
        
        elif vlm_decision == Decision.UNKNOWN:
            self.consecutive_unknown += 1
            if self.consecutive_unknown >= 3:
                new_state = ControlState.SEARCHING
                new_command = ControlCommand.ROTATE
            else:
                new_state = ControlState.LOST
                new_command = ControlCommand.SLOW_TURN
        
        else:
            # Shouldn't happen with constrained decoding
            new_state = ControlState.LOST
            new_command = ControlCommand.HOLD
        
        # Apply transition
        if new_state != prev_state:
            self._transition_to(new_state, 
                               f"VLM: {vlm_decision.value}",
                               decision.decision_id)
        
        self.state = new_state
        self.command = new_command
        return self.state
    
    def _update_legacy(self, result: dict) -> ControlState:
        """Handle legacy {detected, confidence} format."""
        detected = result.get('detected', False)
        
        if detected:
            self.consecutive_unknown = 0
            self.state = ControlState.TRACKING
            self.command = ControlCommand.FOLLOW
        else:
            self.consecutive_unknown += 1
            if self.consecutive_unknown > 10:
                self.state = ControlState.SEARCHING
                self.command = ControlCommand.ROTATE
            elif self.consecutive_unknown > 3:
                self.state = ControlState.LOST
                self.command = ControlCommand.SLOW_TURN
        
        return self.state
    
    def _map_action_to_command(self, action: ActionType) -> ControlCommand:
        """Map VLM action to control command."""
        mapping = {
            ActionType.FOLLOW: ControlCommand.FOLLOW,
            ActionType.STOP: ControlCommand.STOP,
            ActionType.ALERT: ControlCommand.FOLLOW,  # Keep tracking
            ActionType.SEARCH: ControlCommand.ROTATE,
            ActionType.LOG_ONLY: ControlCommand.HOLD,
        }
        return mapping.get(action, ControlCommand.HOLD)
    
    def _transition_to(self, new_state: ControlState, trigger: str, decision_id: str):
        """Record a state transition."""
        transition = StateTransition(
            from_state=self.state,
            to_state=new_state,
            trigger=trigger,
            decision_id=decision_id
        )
        self.transitions.append(transition)
        if len(self.transitions) > self.max_history:
            self.transitions.pop(0)
    
    def get_state(self) -> ControlState:
        """Get current control state."""
        return self.state
    
    def get_command(self) -> ControlCommand:
        """Get current control command."""
        return self.command
    
    def get_action(self) -> str:
        """Legacy compatibility: get action as string."""
        return self.command.value.lower()
    
    def set_target(self, target_id: int):
        """Set/lock target ID."""
        self.target_id = target_id
        self.state = ControlState.TRACKING
        self.consecutive_unknown = 0
        self.consecutive_fallbacks = 0
        print(f"🎯 Target locked: ID {target_id}")
    
    def reset(self):
        """Reset state machine to initial state."""
        self.state = ControlState.SEARCHING
        self.command = ControlCommand.ROTATE
        self.target_id = None
        self.last_decision = None
        self.consecutive_fallbacks = 0
        self.consecutive_unknown = 0
        self.transitions.clear()
        print("🔄 State machine reset")
    
    def get_status(self) -> dict:
        """Get current status for display/API."""
        return {
            "state": self.state.value,
            "command": self.command.value,
            "target_id": self.target_id,
            "consecutive_unknown": self.consecutive_unknown,
            "consecutive_fallbacks": self.consecutive_fallbacks,
            "last_decision_id": self.last_decision.decision_id if self.last_decision else None
        }
    
    def __str__(self):
        return (f"State: {self.state.value} | Command: {self.command.value} | "
                f"Target: {self.target_id} | Unknown: {self.consecutive_unknown}")


# ============================================================
# Legacy Compatibility Alias
# ============================================================

# Keep old name for backward compatibility
FollowStateMachine = MachinaStateMachine


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    from .machina_schema import build_full_decision, SAFE_FALLBACK
    
    print("=" * 60)
    print("MachinaStateMachine Test")
    print("=" * 60)
    
    sm = MachinaStateMachine()
    print(f"\nInitial: {sm}")
    
    # Test with SAFE decision
    print("\n--- Test: SAFE decision ---")
    safe_output = {
        "slow_path": {
            "decision": "SAFE",
            "confidence": 0.85,
            "evidence": {"objects_seen": ["person"], "notes": "Person standing"}
        },
        "action": {"type": "FOLLOW"}
    }
    decision = build_full_decision(safe_output, track_id=42, vlm_latency_ms=3000)
    sm.update(decision)
    print(f"After SAFE: {sm}")
    
    # Test with SUSPICIOUS decision
    print("\n--- Test: SUSPICIOUS decision ---")
    suspicious_output = {
        "slow_path": {
            "decision": "SUSPICIOUS",
            "confidence": 0.7,
            "evidence": {"objects_seen": ["person", "phone"], "notes": "Phone visible"}
        },
        "action": {"type": "ALERT"}
    }
    decision = build_full_decision(suspicious_output, track_id=42, vlm_latency_ms=3200)
    sm.update(decision)
    print(f"After SUSPICIOUS: {sm}")
    
    # Test with UNKNOWN decisions
    print("\n--- Test: Multiple UNKNOWN decisions ---")
    unknown_output = {
        "slow_path": {
            "decision": "UNKNOWN",
            "confidence": 0.3,
            "evidence": {"objects_seen": ["unknown"], "notes": "Target obscured"}
        },
        "action": {"type": "SEARCH"}
    }
    for i in range(4):
        decision = build_full_decision(unknown_output, track_id=-1, vlm_latency_ms=3000)
        sm.update(decision)
        print(f"  After UNKNOWN {i+1}: {sm}")
    
    # Test fallback handling
    print("\n--- Test: Fallback decisions ---")
    sm.reset()
    for i in range(6):
        fallback = SAFE_FALLBACK.model_copy(deep=True)
        sm.update(fallback)
        print(f"  After fallback {i+1}: {sm}")
    
    # Print status
    print("\n📊 Final Status:")
    print(sm.get_status())
    
    print("\n" + "=" * 60)
    print("State machine test complete!")
