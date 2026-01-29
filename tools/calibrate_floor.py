#!/usr/bin/env python3
"""
Floor Calibration Tool for Ground-Plane Homography
===================================================

Interactive CLI tool for calibrating ground-plane homography.
Creates camera_calibration.json for BEV distance calculation.

Usage:
    python -m tools.calibrate_floor --video path/to/video.mp4
    python -m tools.calibrate_floor --camera 0
    python -m tools.calibrate_floor --video path/to/video.mp4 --output configs/cam1_calibration.json
    
Advanced:
    python -m tools.calibrate_floor --validate configs/camera_calibration.json
    python -m tools.calibrate_floor --batch configs/cameras.yaml
    python -m tools.calibrate_floor --video video.mp4 --reference-distance 2.5 --reference-points 2

Instructions:
    1. Click 4 points on the floor forming a rectangle
    2. Enter real-world dimensions (width x height in meters)
    3. Review BEV preview
    4. Confirm to save calibration

Keyboard shortcuts:
    - 'r': Reset points
    - 'u': Undo last point
    - 'q': Quit without saving
    - 'v': Toggle validation overlay
    - 'g': Show grid overlay
    - Enter: Confirm and proceed
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ground_plane import GroundPlane


# ============================================================
# Calibration Validation
# ============================================================

class CalibrationValidator:
    """Validate calibration accuracy and quality."""
    
    # Quality thresholds
    MIN_COVERAGE_PCT = 15.0      # Calibration rect should cover >15% of frame
    MAX_COVERAGE_PCT = 90.0      # Should not cover >90% (probably wrong)
    MAX_REPROJECTION_ERROR = 5.0  # pixels
    MIN_RECT_ASPECT = 0.2       # width/height ratio limits
    MAX_RECT_ASPECT = 5.0
    
    @staticmethod
    def validate_calibration(calib_path: str) -> Dict:
        """
        Validate an existing calibration file.
        
        Returns:
            Dict with 'valid', 'score', 'issues', 'warnings'
        """
        result = {
            'valid': True,
            'score': 100,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Load calibration
        try:
            with open(calib_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"Failed to load calibration: {e}")
            result['score'] = 0
            return result
        
        # Check required fields
        required = ['camera_id', 'frame_size', 'source_points_px', 
                   'destination_points_m', 'homography_matrix']
        for field in required:
            if field not in data:
                result['valid'] = False
                result['issues'].append(f"Missing required field: {field}")
                result['score'] -= 20
        
        if not result['valid']:
            return result
        
        # Validate geometry
        src_pts = np.array(data['source_points_px'], dtype=np.float32)
        dst_pts = np.array(data['destination_points_m'], dtype=np.float32)
        frame_w, frame_h = data['frame_size']
        
        # Check coverage
        rect_area = cv2.contourArea(src_pts)
        frame_area = frame_w * frame_h
        coverage_pct = (rect_area / frame_area) * 100
        result['metrics']['coverage_pct'] = round(coverage_pct, 1)
        
        if coverage_pct < CalibrationValidator.MIN_COVERAGE_PCT:
            result['warnings'].append(
                f"Low coverage ({coverage_pct:.1f}%) - calibration rect may be too small"
            )
            result['score'] -= 10
        elif coverage_pct > CalibrationValidator.MAX_COVERAGE_PCT:
            result['warnings'].append(
                f"High coverage ({coverage_pct:.1f}%) - verify points are correct"
            )
            result['score'] -= 10
        
        # Check aspect ratio
        if 'real_world_rect' in data:
            width = data['real_world_rect']['width_m']
            height = data['real_world_rect']['height_m']
            aspect = width / height if height > 0 else 0
            result['metrics']['aspect_ratio'] = round(aspect, 2)
            
            if aspect < CalibrationValidator.MIN_RECT_ASPECT:
                result['warnings'].append(
                    f"Extreme aspect ratio ({aspect:.2f}) - very narrow rectangle"
                )
                result['score'] -= 10
            elif aspect > CalibrationValidator.MAX_RECT_ASPECT:
                result['warnings'].append(
                    f"Extreme aspect ratio ({aspect:.2f}) - very wide rectangle"
                )
                result['score'] -= 10
        
        # Verify homography matrix
        H = np.array(data['homography_matrix'], dtype=np.float64)
        result['metrics']['homography_det'] = round(np.linalg.det(H), 6)
        
        if abs(np.linalg.det(H)) < 1e-10:
            result['valid'] = False
            result['issues'].append("Degenerate homography matrix (det ≈ 0)")
            result['score'] = 0
            return result
        
        # Compute reprojection error
        try:
            projected = cv2.perspectiveTransform(
                src_pts.reshape(1, -1, 2), H
            ).reshape(-1, 2)
            
            errors = np.linalg.norm(projected - dst_pts, axis=1)
            max_error = float(np.max(errors))
            mean_error = float(np.mean(errors))
            
            result['metrics']['reprojection_max_m'] = round(max_error, 4)
            result['metrics']['reprojection_mean_m'] = round(mean_error, 4)
            
            if max_error > 0.1:  # 10cm error in meters
                result['warnings'].append(
                    f"High reprojection error ({max_error:.3f}m) - verify calibration"
                )
                result['score'] -= 15
        except Exception as e:
            result['warnings'].append(f"Could not compute reprojection error: {e}")
        
        # Check for convex quadrilateral (points should be ordered correctly)
        if not CalibrationValidator._is_convex_quad(src_pts):
            result['valid'] = False
            result['issues'].append(
                "Source points do not form a convex quadrilateral - check point order"
            )
            result['score'] = 0
        
        # Check calibration age
        if 'calibration_date' in data:
            try:
                calib_date = datetime.fromisoformat(data['calibration_date'].replace('Z', '+00:00'))
                age_days = (datetime.now(calib_date.tzinfo or None) - calib_date).days if calib_date.tzinfo else (datetime.now() - calib_date).days
                result['metrics']['age_days'] = age_days
                
                if age_days > 90:
                    result['warnings'].append(
                        f"Calibration is {age_days} days old - consider recalibrating"
                    )
                    result['score'] -= 5
            except Exception:
                pass
        
        result['score'] = max(0, result['score'])
        return result
    
    @staticmethod
    def _is_convex_quad(pts: np.ndarray) -> bool:
        """Check if 4 points form a convex quadrilateral."""
        if len(pts) != 4:
            return False
        
        # Check cross products have same sign
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        signs = []
        for i in range(4):
            o = pts[i]
            a = pts[(i + 1) % 4]
            b = pts[(i + 2) % 4]
            signs.append(cross(o, a, b) > 0)
        
        return all(signs) or not any(signs)
    
    @staticmethod
    def measure_reference_distance(
        gp: GroundPlane, 
        point1_px: Tuple[int, int], 
        point2_px: Tuple[int, int]
    ) -> float:
        """Measure distance between two pixel points in meters."""
        return gp.distance_meters(point1_px, point2_px)


class CalibrationTool:
    """Interactive floor calibration tool with enhanced UX."""
    
    WINDOW_NAME = "Floor Calibration - Click 4 corners"
    BEV_WINDOW_NAME = "BEV Preview"
    
    # Colors (BGR)
    COLOR_POINT = (0, 255, 0)       # Green
    COLOR_POINT_ACTIVE = (0, 255, 255)  # Yellow
    COLOR_LINE = (255, 200, 0)      # Cyan
    COLOR_TEXT = (255, 255, 255)    # White
    COLOR_GRID = (100, 100, 100)    # Gray
    COLOR_VALIDATION = (0, 200, 255)  # Orange
    
    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.frame: Optional[np.ndarray] = None
        self.frame_display: Optional[np.ndarray] = None
        self.frame_size: Tuple[int, int] = (0, 0)
        self.real_width_m: float = 0.0
        self.real_height_m: float = 0.0
        
        # Enhanced UX state
        self.show_grid = False
        self.show_validation = False
        self.reference_points: List[Tuple[int, int]] = []
        self.validation_mode = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.validation_mode:
                # Validation mode: collect reference points
                if len(self.reference_points) < 2:
                    self.reference_points.append((x, y))
                    self._update_display()
                    if len(self.reference_points) == 2:
                        print(f"   Reference points: {self.reference_points}")
            else:
                # Normal mode: collect calibration points
                if len(self.points) < 4:
                    self.points.append((x, y))
                    self._update_display()
    
    def _update_display(self):
        """Update display with current points and overlays."""
        if self.frame is None:
            return
        
        self.frame_display = self.frame.copy()
        
        # Draw grid overlay if enabled
        if self.show_grid:
            self._draw_grid()
        
        # Draw instructions
        instructions = [
            "Click 4 corners of a floor rectangle (clockwise from top-left)",
            f"Points: {len(self.points)}/4",
            "Keys: 'r'=reset, 'u'=undo, 'g'=grid, 'v'=validate, 'q'=quit, Enter=confirm"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self.frame_display, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2)
        
        # Draw points
        for i, pt in enumerate(self.points):
            color = self.COLOR_POINT_ACTIVE if i == len(self.points) - 1 else self.COLOR_POINT
            cv2.circle(self.frame_display, pt, 8, color, -1)
            cv2.circle(self.frame_display, pt, 12, color, 2)  # Outer ring
            cv2.putText(self.frame_display, str(i + 1), (pt[0] + 15, pt[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw lines connecting points
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.frame_display, self.points[i], self.points[i + 1],
                        self.COLOR_LINE, 2)
            if len(self.points) == 4:
                cv2.line(self.frame_display, self.points[3], self.points[0],
                        self.COLOR_LINE, 2)
                
                # Draw crosshairs to help verify alignment
                center_x = sum(p[0] for p in self.points) // 4
                center_y = sum(p[1] for p in self.points) // 4
                cv2.circle(self.frame_display, (center_x, center_y), 5, 
                          self.COLOR_VALIDATION, -1)
        
        # Draw reference measurement points
        for i, pt in enumerate(self.reference_points):
            cv2.circle(self.frame_display, pt, 6, self.COLOR_VALIDATION, -1)
            cv2.putText(self.frame_display, f"R{i+1}", (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_VALIDATION, 2)
        
        if len(self.reference_points) == 2:
            cv2.line(self.frame_display, self.reference_points[0], 
                    self.reference_points[1], self.COLOR_VALIDATION, 2)
        
        cv2.imshow(self.WINDOW_NAME, self.frame_display)
    
    def _draw_grid(self):
        """Draw helper grid on frame."""
        h, w = self.frame_display.shape[:2]
        
        # Draw vertical lines
        for x in range(0, w, 50):
            alpha = 0.3 if x % 100 == 0 else 0.15
            cv2.line(self.frame_display, (x, 0), (x, h), self.COLOR_GRID, 1)
        
        # Draw horizontal lines
        for y in range(0, h, 50):
            cv2.line(self.frame_display, (0, y), (w, y), self.COLOR_GRID, 1)
    
    def get_frame(self, source) -> bool:
        """Get frame from video or camera."""
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.frame = frame
            self.frame_size = (frame.shape[1], frame.shape[0])
            return True
        return False
    
    def collect_points(self) -> bool:
        """Collect 4 calibration points from user."""
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_callback)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('r'):
                self.points.clear()
                self.reference_points.clear()
                self._update_display()
            
            elif key == ord('u') and self.points:
                self.points.pop()
                self._update_display()
            
            elif key == ord('g'):
                self.show_grid = not self.show_grid
                self._update_display()
            
            elif key == ord('v'):
                self.validation_mode = not self.validation_mode
                if self.validation_mode:
                    print("\n📏 Validation mode: Click 2 points to measure distance")
                    self.reference_points.clear()
                else:
                    self.reference_points.clear()
                self._update_display()
            
            elif key == 13 and len(self.points) == 4:  # Enter
                break
        
        cv2.destroyAllWindows()
        return True
    
    def get_dimensions(self) -> bool:
        """Get real-world dimensions from user."""
        print("\n" + "=" * 50)
        print("Enter real-world dimensions of the calibration rectangle")
        print("(Measure the floor area marked by your 4 points)")
        print("=" * 50)
        
        try:
            width_str = input("Width (meters, e.g., 3.0): ").strip()
            self.real_width_m = float(width_str)
            
            height_str = input("Height/Depth (meters, e.g., 2.0): ").strip()
            self.real_height_m = float(height_str)
            
            if self.real_width_m <= 0 or self.real_height_m <= 0:
                print("Error: Dimensions must be positive")
                return False
            
            print(f"\nCalibration rectangle: {self.real_width_m}m x {self.real_height_m}m")
            return True
            
        except ValueError:
            print("Error: Invalid number format")
            return False
    
    def compute_and_preview(self) -> Optional[np.ndarray]:
        """Compute homography and show BEV preview."""
        if len(self.points) != 4:
            return None
        
        # Destination points in meters (rectangle)
        dst_points_m = [
            [0.0, 0.0],
            [self.real_width_m, 0.0],
            [self.real_width_m, self.real_height_m],
            [0.0, self.real_height_m]
        ]
        
        # Compute homography
        H = GroundPlane.compute_homography(self.points, dst_points_m)
        
        # Create BEV preview (scale meters to pixels for visualization)
        scale_factor = 100  # 100 pixels per meter
        bev_width = int(self.real_width_m * scale_factor) + 100
        bev_height = int(self.real_height_m * scale_factor) + 100
        
        # Create preview showing calibration rectangle
        bev_preview = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        bev_preview[:] = (40, 40, 40)  # Dark gray background
        
        # Draw calibration rectangle
        rect_pts = np.array([
            [50, 50],
            [50 + int(self.real_width_m * scale_factor), 50],
            [50 + int(self.real_width_m * scale_factor), 50 + int(self.real_height_m * scale_factor)],
            [50, 50 + int(self.real_height_m * scale_factor)]
        ], dtype=np.int32)
        cv2.polylines(bev_preview, [rect_pts], True, (0, 255, 0), 2)
        
        # Add scale labels
        cv2.putText(bev_preview, f"{self.real_width_m}m", 
                   (50 + int(self.real_width_m * scale_factor // 2) - 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bev_preview, f"{self.real_height_m}m",
                   (20, 50 + int(self.real_height_m * scale_factor // 2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Test projection of calibration points
        gp = GroundPlane()
        gp.H = H
        gp.is_calibrated = True
        
        for i, pt in enumerate(self.points):
            try:
                bev_pt = gp.project_to_bev(pt)
                bev_x = int(50 + bev_pt[0] * scale_factor)
                bev_y = int(50 + bev_pt[1] * scale_factor)
                cv2.circle(bev_preview, (bev_x, bev_y), 5, (0, 255, 255), -1)
                cv2.putText(bev_preview, str(i + 1), (bev_x + 8, bev_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            except Exception:
                pass
        
        # Add title
        cv2.putText(bev_preview, "BEV Preview (Bird's Eye View)", (10, bev_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(self.BEV_WINDOW_NAME, bev_preview)
        
        return H
    
    def save_calibration(self, output_path: str, camera_id: str):
        """Save calibration to JSON file with validation metadata."""
        dst_points_m = [
            [0.0, 0.0],
            [self.real_width_m, 0.0],
            [self.real_width_m, self.real_height_m],
            [0.0, self.real_height_m]
        ]
        
        H = GroundPlane.compute_homography(self.points, dst_points_m)
        
        # Compute calibration quality metrics
        rect_area = cv2.contourArea(np.array(self.points, dtype=np.float32))
        frame_area = self.frame_size[0] * self.frame_size[1]
        coverage_pct = (rect_area / frame_area) * 100 if frame_area > 0 else 0
        
        data = {
            "camera_id": camera_id,
            "calibration_date": datetime.now().isoformat(),
            "calibration_tool_version": "2.0.0",
            "frame_size": list(self.frame_size),
            "source_points_px": [list(p) for p in self.points],
            "destination_points_m": dst_points_m,
            "real_world_rect": {
                "width_m": self.real_width_m,
                "height_m": self.real_height_m,
                "description": f"Calibration rectangle: {self.real_width_m}m wide × {self.real_height_m}m deep on floor"
            },
            "homography_matrix": H.tolist(),
            "valid_floor_region_normalized": [
                [p[0] / self.frame_size[0], p[1] / self.frame_size[1]]
                for p in self.points
            ],
            "quality_metrics": {
                "coverage_pct": round(coverage_pct, 1),
                "aspect_ratio": round(self.real_width_m / self.real_height_m, 2) if self.real_height_m > 0 else 0,
                "homography_det": round(float(np.linalg.det(H)), 6)
            },
            "notes": "Generated by tools/calibrate_floor.py v2.0"
        }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Calibration saved to: {output_path}")
        
        # Run validation
        print("\n📊 Validating calibration...")
        result = CalibrationValidator.validate_calibration(output_path)
        print(f"   Score: {result['score']}/100")
        
        if result['warnings']:
            print("   ⚠️  Warnings:")
            for w in result['warnings']:
                print(f"      - {w}")
        
        if result['issues']:
            print("   ❌ Issues:")
            for i in result['issues']:
                print(f"      - {i}")
        
        return True


def validate_calibration_cmd(calib_path: str) -> int:
    """Validate an existing calibration file."""
    print("=" * 60)
    print("Calibration Validation")
    print("=" * 60)
    
    if not Path(calib_path).exists():
        print(f"❌ File not found: {calib_path}")
        return 1
    
    print(f"\n📂 Validating: {calib_path}")
    result = CalibrationValidator.validate_calibration(calib_path)
    
    print(f"\n{'='*40}")
    print(f"Validation Result")
    print(f"{'='*40}")
    print(f"   Valid: {'✅ Yes' if result['valid'] else '❌ No'}")
    print(f"   Score: {result['score']}/100")
    
    if result['metrics']:
        print(f"\n📊 Metrics:")
        for key, value in result['metrics'].items():
            print(f"   {key}: {value}")
    
    if result['warnings']:
        print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for w in result['warnings']:
            print(f"   - {w}")
    
    if result['issues']:
        print(f"\n❌ Issues ({len(result['issues'])}):")
        for i in result['issues']:
            print(f"   - {i}")
    
    if result['valid'] and result['score'] >= 80:
        print(f"\n🎉 Calibration is good to use!")
        return 0
    elif result['valid']:
        print(f"\n⚠️  Calibration is valid but has quality concerns")
        return 0
    else:
        print(f"\n❌ Calibration is invalid - please recalibrate")
        return 1


def batch_calibrate_cmd(batch_config: str) -> int:
    """Batch calibration from YAML config."""
    print("=" * 60)
    print("Batch Calibration")
    print("=" * 60)
    
    if not Path(batch_config).exists():
        print(f"❌ Config not found: {batch_config}")
        return 1
    
    with open(batch_config, 'r') as f:
        config = yaml.safe_load(f)
    
    cameras = config.get('cameras', [])
    if not cameras:
        print("❌ No cameras defined in config")
        return 1
    
    print(f"\n📷 Found {len(cameras)} camera(s) to calibrate")
    
    results = []
    for cam in cameras:
        cam_id = cam.get('id', 'unknown')
        source = cam.get('source')  # video path or camera index
        output = cam.get('output', f'configs/{cam_id}_calibration.json')
        
        print(f"\n{'='*40}")
        print(f"Camera: {cam_id}")
        print(f"Source: {source}")
        print(f"Output: {output}")
        
        # Check if calibration exists and is valid
        if Path(output).exists():
            result = CalibrationValidator.validate_calibration(output)
            if result['valid'] and result['score'] >= 80:
                print(f"✅ Existing calibration is valid (score: {result['score']})")
                confirm = input("   Recalibrate anyway? [y/N]: ").strip().lower()
                if confirm != 'y':
                    results.append({'camera': cam_id, 'status': 'skipped', 'score': result['score']})
                    continue
        
        # Run calibration
        tool = CalibrationTool()
        
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            src = int(source) if isinstance(source, str) else source
        else:
            src = source
            if not Path(src).exists():
                print(f"❌ Source not found: {src}")
                results.append({'camera': cam_id, 'status': 'error', 'reason': 'source not found'})
                continue
        
        if not tool.get_frame(src):
            print(f"❌ Failed to get frame")
            results.append({'camera': cam_id, 'status': 'error', 'reason': 'frame capture failed'})
            continue
        
        print(f"✅ Frame captured: {tool.frame_size[0]}x{tool.frame_size[1]}")
        
        if not tool.collect_points():
            print("Calibration cancelled")
            results.append({'camera': cam_id, 'status': 'cancelled'})
            continue
        
        if not tool.get_dimensions():
            results.append({'camera': cam_id, 'status': 'error', 'reason': 'invalid dimensions'})
            continue
        
        H = tool.compute_and_preview()
        if H is None:
            results.append({'camera': cam_id, 'status': 'error', 'reason': 'homography failed'})
            continue
        
        print("\nPress 'y' to save, any other key to skip")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('y'):
            tool.save_calibration(output, cam_id)
            result = CalibrationValidator.validate_calibration(output)
            results.append({'camera': cam_id, 'status': 'success', 'score': result['score']})
        else:
            results.append({'camera': cam_id, 'status': 'skipped'})
    
    # Summary
    print(f"\n{'='*60}")
    print("Batch Calibration Summary")
    print(f"{'='*60}")
    for r in results:
        status_icon = {'success': '✅', 'skipped': '⏭️', 'cancelled': '🚫', 'error': '❌'}.get(r['status'], '?')
        score = f" (score: {r.get('score', 'N/A')})" if 'score' in r else ''
        reason = f" - {r.get('reason', '')}" if 'reason' in r else ''
        print(f"   {status_icon} {r['camera']}: {r['status']}{score}{reason}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Floor Calibration Tool for Ground-Plane Homography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive calibration
    python -m tools.calibrate_floor --video test_videos/warehouse.mp4
    python -m tools.calibrate_floor --camera 0 --output configs/cam0_calibration.json
    python -m tools.calibrate_floor --video video.mp4 --camera-id warehouse_cam_01
    
    # Validate existing calibration
    python -m tools.calibrate_floor --validate configs/camera_calibration.json
    
    # Batch calibration from config
    python -m tools.calibrate_floor --batch configs/cameras.yaml
        """
    )
    
    parser.add_argument('--video', '-v', type=str, 
                       help='Video file for calibration frame')
    parser.add_argument('--camera', '-c', type=int, default=None,
                       help='Camera index (default: None)')
    parser.add_argument('--output', '-o', type=str, 
                       default='configs/camera_calibration.json',
                       help='Output calibration file')
    parser.add_argument('--camera-id', type=str, default='camera_01',
                       help='Camera identifier for the calibration file')
    parser.add_argument('--validate', type=str, metavar='FILE',
                       help='Validate an existing calibration file')
    parser.add_argument('--batch', type=str, metavar='CONFIG',
                       help='Batch calibration from YAML config file')
    
    args = parser.parse_args()
    
    # Handle validation mode
    if args.validate:
        return validate_calibration_cmd(args.validate)
    
    # Handle batch mode
    if args.batch:
        return batch_calibrate_cmd(args.batch)
    
    # Determine source
    if args.video:
        source = args.video
        if not Path(source).exists():
            print(f"❌ Video file not found: {source}")
            return 1
    elif args.camera is not None:
        source = args.camera
    else:
        print("❌ Please specify --video, --camera, --validate, or --batch")
        parser.print_help()
        return 1
    
    print("=" * 60)
    print("Floor Calibration Tool")
    print("=" * 60)
    
    tool = CalibrationTool()
    
    # Get frame
    print(f"\n📷 Loading frame from: {source}")
    if not tool.get_frame(source):
        print("❌ Failed to get frame from source")
        return 1
    
    print(f"✅ Frame size: {tool.frame_size[0]}x{tool.frame_size[1]}")
    
    # Collect points
    print("\n📍 Click 4 corners of a known rectangle on the floor")
    print("   (clockwise starting from top-left)")
    if not tool.collect_points():
        print("Calibration cancelled")
        return 0
    
    print(f"✅ Points collected: {tool.points}")
    
    # Get dimensions
    if not tool.get_dimensions():
        return 1
    
    # Compute and preview
    print("\n🔄 Computing homography...")
    H = tool.compute_and_preview()
    
    if H is None:
        print("❌ Failed to compute homography")
        return 1
    
    print("\n✅ Homography computed successfully")
    print("   Review the BEV preview window")
    print("   Press 'y' to save, any other key to cancel")
    
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    
    if key == ord('y'):
        tool.save_calibration(args.output, args.camera_id)
        print("\n🎉 Calibration complete!")
        return 0
    else:
        print("\nCalibration cancelled")
        return 0


if __name__ == "__main__":
    exit(main())
