#!/usr/bin/env python3
"""
ZoneConfigManager - Single Source of Truth for Zone Configuration
=================================================================

Manages zone configuration with support for:
- Loading from primary config (zones.yaml)
- Merging VLM-detected zones
- Hot-swap to running ZoneManager
- Thread-safe operations

Usage:
    zcm = ZoneConfigManager("configs/zones.yaml", "configs/zones_vlm.yaml")
    zone_manager = zcm.load()
    zcm.apply_vlm_zones(vlm_zones, merge=True)
"""

import yaml
import threading
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from .zone_manager import ZoneManager, Zone


class ZoneConfigManager:
    """
    Single source of truth for zone configuration.
    
    Handles:
    - Loading from primary config (zones.yaml)
    - Merging VLM-detected zones
    - Hot-swap to running ZoneManager
    - Thread-safe operations
    """
    
    def __init__(
        self,
        primary_config: str,
        vlm_cache: Optional[str] = None
    ):
        """
        Initialize ZoneConfigManager.
        
        Args:
            primary_config: Path to primary zones config (zones.yaml)
            vlm_cache: Path to VLM zones cache file (zones_vlm.yaml)
        """
        self.primary_config = primary_config
        self.vlm_cache = vlm_cache or primary_config.replace('.yaml', '_vlm.yaml')
        self.zone_manager: Optional[ZoneManager] = None
        self._lock = threading.Lock()
        self._vlm_zones: List[dict] = []
    
    def load(self) -> ZoneManager:
        """
        Load zones from primary config.
        
        Returns:
            Initialized ZoneManager with zones from config
        """
        with self._lock:
            self.zone_manager = ZoneManager(self.primary_config)
            print(f"✅ ZoneConfigManager: Loaded {len(self.zone_manager.zones)} zones from {self.primary_config}")
            return self.zone_manager
    
    def get_zone_manager(self) -> Optional[ZoneManager]:
        """Get the current ZoneManager instance."""
        return self.zone_manager
    
    def apply_vlm_zones(
        self,
        vlm_zones: List[dict],
        merge: bool = True
    ):
        """
        Apply VLM-detected zones.
        
        Args:
            vlm_zones: List of zone dicts from VLM
            merge: If True, add to existing zones; if False, replace all
        """
        with self._lock:
            if self.zone_manager is None:
                print("⚠️ ZoneConfigManager: No ZoneManager loaded, call load() first")
                return
            
            # Validate and prefix VLM zones
            validated_zones = []
            for zone in vlm_zones:
                validated = self._validate_vlm_zone(zone)
                if validated:
                    validated_zones.append(validated)
            
            if not validated_zones:
                print("⚠️ ZoneConfigManager: No valid VLM zones to apply")
                return
            
            if merge:
                # Add VLM zones to existing zones
                for zone_data in validated_zones:
                    zone = self.zone_manager._parse_zone(zone_data)
                    self.zone_manager.add_zone(zone)
                print(f"✅ ZoneConfigManager: Merged {len(validated_zones)} VLM zones")
            else:
                # Replace all zones with VLM zones
                self.zone_manager.update_zones(validated_zones)
                print(f"✅ ZoneConfigManager: Replaced zones with {len(validated_zones)} VLM zones")
            
            # Cache to disk
            self._save_vlm_cache(validated_zones)
            self._vlm_zones = validated_zones
    
    def _validate_vlm_zone(self, zone: dict) -> Optional[dict]:
        """
        Validate VLM zone data.
        
        Args:
            zone: Zone dict from VLM
            
        Returns:
            Validated zone dict with vlm_ prefix, or None if invalid
        """
        try:
            # Required fields
            if 'polygon' not in zone:
                return None
            
            polygon = zone['polygon']
            if not isinstance(polygon, list) or len(polygon) < 3:
                return None
            
            # Validate polygon points are normalized (0-1)
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    return None
                x, y = point
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    # Auto-correct if slightly out of bounds
                    point[0] = max(0, min(1, x))
                    point[1] = max(0, min(1, y))
            
            # Generate or prefix zone ID
            zone_id = zone.get('id', f'zone_{len(self._vlm_zones)}')
            if not zone_id.startswith('vlm_'):
                zone_id = f'vlm_{zone_id}'
            
            # Build validated zone
            validated = {
                'id': zone_id,
                'type': zone.get('type', 'RESTRICTED'),
                'polygon': polygon,
                'display_name': zone.get('display_name') or self._get_semantic_zone_name(polygon),
                'rules': zone.get('rules', [{
                    'type': 'PERSON_PRESENCE',
                    'max_dwell_seconds': 5.0,
                    'severity': 'MEDIUM',
                    'target_classes': ['person']
                }]),
                'enabled': zone.get('enabled', True),
                'color': zone.get('color', [100, 200, 200])  # Teal for VLM zones
            }
            
            return validated
            
        except Exception as e:
            print(f"⚠️ ZoneConfigManager: Invalid VLM zone: {e}")
            return None
    
    def _get_semantic_zone_name(self, polygon: List[List[float]]) -> str:
        """
        Generate meaningful zone name based on position.
        
        Args:
            polygon: Zone polygon (normalized coordinates)
            
        Returns:
            Descriptive zone name
        """
        if not polygon or len(polygon) < 3:
            return "vlm_zone"
        
        # Calculate centroid
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        
        # Vertical position
        if cy > 0.75:
            base = "floor"
        elif cy > 0.5:
            base = "mid"
        elif cy > 0.25:
            base = "upper"
        else:
            base = "ceiling"
        
        # Horizontal position
        if cx < 0.33:
            position = "left"
        elif cx > 0.66:
            position = "right"
        else:
            position = "center"
        
        return f"{base}_{position}_zone"
    
    def _save_vlm_cache(self, zones: List[dict]):
        """
        Save VLM zones to cache file.
        
        Args:
            zones: List of zone dicts to cache
        """
        try:
            cache_data = {
                'generated_at': datetime.now().isoformat(),
                'source': 'vlm_calibration',
                'zones': zones
            }
            
            with open(self.vlm_cache, 'w') as f:
                yaml.dump(cache_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ ZoneConfigManager: Cached VLM zones to {self.vlm_cache}")
            
        except Exception as e:
            print(f"⚠️ ZoneConfigManager: Failed to cache VLM zones: {e}")
    
    def load_vlm_cache(self) -> bool:
        """
        Load previously cached VLM zones.
        
        Returns:
            True if cache loaded successfully
        """
        cache_path = Path(self.vlm_cache)
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r') as f:
                data = yaml.safe_load(f)
            
            zones = data.get('zones', [])
            if zones:
                self.apply_vlm_zones(zones, merge=True)
                return True
            return False
            
        except Exception as e:
            print(f"⚠️ ZoneConfigManager: Failed to load VLM cache: {e}")
            return False
    
    def get_all_zones(self) -> List[Zone]:
        """Get all zones (manual + VLM)."""
        if self.zone_manager:
            return self.zone_manager.get_all_zones()
        return []
    
    def get_vlm_zones(self) -> List[dict]:
        """Get currently applied VLM zones."""
        return self._vlm_zones.copy()
    
    def reload_config(self):
        """
        Reload configuration from disk.
        
        Useful after manual config file changes.
        """
        with self._lock:
            if self.zone_manager:
                self.zone_manager.load_config(self.primary_config)
                print(f"✅ ZoneConfigManager: Reloaded config from {self.primary_config}")


# ============================================================
# Module-level singleton
# ============================================================

_zone_config_manager: Optional[ZoneConfigManager] = None


def get_zone_config_manager(
    primary_config: Optional[str] = None,
    vlm_cache: Optional[str] = None
) -> ZoneConfigManager:
    """Get or create singleton ZoneConfigManager instance."""
    global _zone_config_manager
    if _zone_config_manager is None:
        if primary_config is None:
            raise ValueError("primary_config required for first initialization")
        _zone_config_manager = ZoneConfigManager(primary_config, vlm_cache)
    return _zone_config_manager


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    # Create temporary config files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        primary_path = f.name
        yaml.dump({
            'zones': [{
                'id': 'manual_zone_1',
                'type': 'RESTRICTED',
                'polygon': [[0.2, 0.6], [0.8, 0.6], [0.8, 1.0], [0.2, 1.0]],
                'rules': [{'type': 'PERSON_PRESENCE', 'max_dwell_seconds': 5, 'severity': 'HIGH'}]
            }]
        }, f)
    
    vlm_path = primary_path.replace('.yaml', '_vlm.yaml')
    
    try:
        # Create manager
        manager = ZoneConfigManager(primary_path, vlm_path)
        
        # Load primary config
        zm = manager.load()
        print(f"Loaded zones: {[z.id for z in manager.get_all_zones()]}")
        
        # Apply VLM zones
        vlm_zones = [
            {
                'id': 'detected_zone',
                'type': 'RESTRICTED',
                'polygon': [[0.1, 0.3], [0.4, 0.3], [0.4, 0.5], [0.1, 0.5]]
            },
            {
                'polygon': [[0.6, 0.7], [0.9, 0.7], [0.9, 0.95], [0.6, 0.95]]  # No ID
            }
        ]
        
        manager.apply_vlm_zones(vlm_zones, merge=True)
        
        print(f"After VLM merge: {[z.id for z in manager.get_all_zones()]}")
        print(f"VLM zones: {manager.get_vlm_zones()}")
        
        # Test semantic naming
        test_polygons = [
            [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],  # upper left
            [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6]],  # mid center
            [[0.8, 0.9], [0.9, 0.9], [0.9, 1.0], [0.8, 1.0]],  # floor right
        ]
        print("\nSemantic zone naming:")
        for poly in test_polygons:
            name = manager._get_semantic_zone_name(poly)
            print(f"  {poly[0]} -> {name}")
        
        print("\nZoneConfigManager test complete!")
        
    finally:
        os.unlink(primary_path)
        if os.path.exists(vlm_path):
            os.unlink(vlm_path)
