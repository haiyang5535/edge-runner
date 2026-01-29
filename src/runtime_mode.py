#!/usr/bin/env python3
"""
RuntimeMode - Dual Runtime Mode Configuration
==============================================

Separates demo reliability from production safety:
- DEMO_MODE: Uncalibrated allowed, relaxed gating for LOI demos
- SAFETY_MODE: Calibration required, strict BEV enforcement

Usage:
    from .runtime_mode import RuntimeMode, get_mode_config, determine_runtime_mode
    
    mode = determine_runtime_mode(args, calibration_path)
    config = get_mode_config(mode)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any
import os


class RuntimeMode(Enum):
    """Runtime mode for edge-runner Safety System."""
    DEMO_MODE = "DEMO"       # Uncalibrated allowed, relaxed gating
    SAFETY_MODE = "SAFETY"   # Calibration required, strict enforcement


@dataclass
class ModeConfig:
    """Configuration for a specific runtime mode."""
    motion_gate_enabled: bool
    motion_gate_require_bev: bool
    proximity_use_bev: bool
    zones_all_enabled: bool  # Override zone enabled flags
    show_uncalibrated_banner: bool
    allow_ttc_alerts: bool
    banner_text: str
    banner_color: tuple  # BGR


# Mode-specific configurations
MODE_CONFIGS: Dict[RuntimeMode, ModeConfig] = {
    RuntimeMode.DEMO_MODE: ModeConfig(
        motion_gate_enabled=False,        # Disable to avoid flicker
        motion_gate_require_bev=False,
        proximity_use_bev=False,          # Fallback to Y-scaled
        zones_all_enabled=True,           # Show all zones
        show_uncalibrated_banner=True,
        allow_ttc_alerts=False,           # TTC requires calibration
        banner_text="⚠️ UNCALIBRATED — DEMO ONLY",
        banner_color=(0, 165, 255),       # Orange
    ),
    RuntimeMode.SAFETY_MODE: ModeConfig(
        motion_gate_enabled=True,
        motion_gate_require_bev=True,     # BEV required
        proximity_use_bev=True,
        zones_all_enabled=False,          # Respect config
        show_uncalibrated_banner=False,
        allow_ttc_alerts=True,
        banner_text="✅ CALIBRATED — SAFETY ACTIVE",
        banner_color=(136, 230, 98),      # Green
    ),
}


def get_mode_config(mode: RuntimeMode) -> ModeConfig:
    """Get configuration for specified runtime mode."""
    return MODE_CONFIGS[mode]


def calibration_exists(calibration_path: str) -> bool:
    """Check if calibration file exists and is valid."""
    if not os.path.exists(calibration_path):
        return False
    
    # Basic validation: file is non-empty JSON
    try:
        import json
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        return 'H' in data or 'homography' in data or 'points' in data
    except (json.JSONDecodeError, IOError):
        return False


class StartupError(Exception):
    """Raised when edge-runner cannot start due to configuration issues."""
    pass


def determine_runtime_mode(
    force_demo: bool = False,
    calibration_path: str = None,
    strict: bool = False
) -> RuntimeMode:
    """
    Determine runtime mode based on flags and calibration availability.
    
    Args:
        force_demo: If True, force DEMO_MODE regardless of calibration
        calibration_path: Path to camera_calibration.json
        strict: If True, require calibration or raise StartupError
        
    Returns:
        RuntimeMode enum value
        
    Raises:
        StartupError: If strict=True and no calibration available
    """
    # Force demo mode
    if force_demo:
        return RuntimeMode.DEMO_MODE
    
    # Check calibration
    has_calibration = calibration_path and calibration_exists(calibration_path)
    
    if has_calibration:
        return RuntimeMode.SAFETY_MODE
    
    # No calibration
    if strict:
        raise StartupError(
            "🚨 Calibration required for SAFETY_MODE.\n"
            "   Run: python -m tools.calibrate_floor\n"
            "   Or use: --demo to run in DEMO_MODE"
        )
    
    # Default to DEMO_MODE with warning
    return RuntimeMode.DEMO_MODE


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import tempfile
    import json
    
    print("RuntimeMode Tests")
    print("=" * 40)
    
    # Test DEMO_MODE
    mode = determine_runtime_mode(force_demo=True)
    config = get_mode_config(mode)
    print(f"\nDEMO_MODE config:")
    print(f"  motion_gate_enabled: {config.motion_gate_enabled}")
    print(f"  zones_all_enabled: {config.zones_all_enabled}")
    print(f"  banner: {config.banner_text}")
    
    # Test SAFETY_MODE with valid calibration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, f)
        temp_path = f.name
    
    mode = determine_runtime_mode(calibration_path=temp_path)
    config = get_mode_config(mode)
    print(f"\nSAFETY_MODE config:")
    print(f"  motion_gate_enabled: {config.motion_gate_enabled}")
    print(f"  motion_gate_require_bev: {config.motion_gate_require_bev}")
    print(f"  banner: {config.banner_text}")
    
    os.unlink(temp_path)
    
    # Test strict mode without calibration
    print("\nTesting strict mode without calibration:")
    try:
        determine_runtime_mode(strict=True, calibration_path="/nonexistent.json")
    except StartupError as e:
        print(f"  StartupError raised: ✅")
    
    print("\nRuntimeMode tests complete!")
