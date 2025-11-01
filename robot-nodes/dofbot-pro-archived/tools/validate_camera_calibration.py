#!/usr/bin/env python3
"""
DOFBOT Pro Camera Calibration Validation Tool

Standalone utility for validating camera calibration quality and accuracy.
This tool analyzes calibration parameters and tests them with live camera data.

Usage:
    python3 validate_camera_calibration.py [options]

Requirements:
    - Existing camera calibration file (camera_calibration.json)
    - pyorbbecsdk installed
    - Checkerboard pattern for validation (optional)
"""

import cv2
import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# Try to import Orbbec SDK
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, FrameSet, OBError
    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    print("❌ Error: pyorbbecsdk not available. Please install the Orbbec SDK.")
    ORBBEC_SDK_AVAILABLE = False
    exit(1)


class CalibrationValidator:
    """Validates camera calibration quality and accuracy."""
    
    def __init__(self, calibration_file: str):
        """Initialize the validator.
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_file = Path(calibration_file)
        self.calibration_data = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Camera setup
        self.pipeline = None
        self.config = None
        
    def load_calibration(self) -> bool:
        """Load calibration data from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.calibration_file.exists():
                print(f"❌ Calibration file not found: {self.calibration_file}")
                return False
            
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            # Extract calibration parameters
            self.camera_matrix = np.array(self.calibration_data["camera_matrix"])
            self.distortion_coeffs = np.array(self.calibration_data["distortion_coefficients"])
            
            print(f"✅ Loaded calibration from: {self.calibration_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def validate_calibration_parameters(self) -> Dict[str, Any]:
        """Validate calibration parameters for reasonableness.
        
        Returns:
            Dictionary with validation results
        """
        print("\n🔍 Validating calibration parameters...")
        
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "info": {}
        }
        
        if self.calibration_data is None:
            results["valid"] = False
            results["errors"].append("No calibration data loaded")
            return results
        
        # Check basic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        results["info"]["focal_length_x"] = fx
        results["info"]["focal_length_y"] = fy
        results["info"]["principal_point_x"] = cx
        results["info"]["principal_point_y"] = cy
        results["info"]["reprojection_error"] = self.calibration_data.get("reprojection_error", "unknown")
        
        # Validate focal lengths
        if fx < 200 or fx > 1000:
            results["warnings"].append(f"Unusual focal length X: {fx:.2f} (expected 200-1000)")
        
        if fy < 200 or fy > 1000:
            results["warnings"].append(f"Unusual focal length Y: {fy:.2f} (expected 200-1000)")
        
        # Check focal length ratio
        focal_ratio = fx / fy if fy > 0 else 0
        if focal_ratio < 0.9 or focal_ratio > 1.1:
            results["warnings"].append(f"Focal length ratio unusual: {focal_ratio:.3f} (expected ~1.0)")
        
        # Validate principal point
        expected_cx = self.calibration_data.get("image_width", 640) / 2
        expected_cy = self.calibration_data.get("image_height", 480) / 2
        
        if abs(cx - expected_cx) > 50:
            results["warnings"].append(f"Principal point X offset: {cx:.1f} vs expected {expected_cx:.1f}")
        
        if abs(cy - expected_cy) > 50:
            results["warnings"].append(f"Principal point Y offset: {cy:.1f} vs expected {expected_cy:.1f}")
        
        # Check reprojection error
        reproj_error = self.calibration_data.get("reprojection_error", 999)
        if reproj_error > 1.0:
            results["warnings"].append(f"High reprojection error: {reproj_error:.3f} pixels (should be < 1.0)")
        elif reproj_error > 0.5:
            results["warnings"].append(f"Moderate reprojection error: {reproj_error:.3f} pixels (ideally < 0.5)")
        
        # Check distortion coefficients
        if len(self.distortion_coeffs) >= 5:
            k1, k2, p1, p2, k3 = self.distortion_coeffs[:5]
            
            if abs(k1) > 0.5:
                results["warnings"].append(f"High radial distortion k1: {k1:.4f}")
            if abs(k2) > 0.5:
                results["warnings"].append(f"High radial distortion k2: {k2:.4f}")
            if abs(p1) > 0.01:
                results["warnings"].append(f"High tangential distortion p1: {p1:.4f}")
            if abs(p2) > 0.01:
                results["warnings"].append(f"High tangential distortion p2: {p2:.4f}")
        
        # Check calibration age
        calib_date = self.calibration_data.get("calibration_date")
        if calib_date:
            try:
                calib_datetime = datetime.fromisoformat(calib_date.replace('Z', '+00:00'))
                age_days = (datetime.now() - calib_datetime.replace(tzinfo=None)).days
                results["info"]["calibration_age_days"] = age_days
                
                if age_days > 90:
                    results["warnings"].append(f"Calibration is {age_days} days old (consider recalibrating)")
            except:
                results["warnings"].append("Could not parse calibration date")
        
        # Summary
        if results["errors"]:
            results["valid"] = False
            print("❌ Calibration validation failed")
        elif results["warnings"]:
            print("⚠️  Calibration has warnings but is usable")
        else:
            print("✅ Calibration parameters look good")
        
        return results
    
    def setup_camera(self) -> bool:
        """Setup camera for live validation.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.pipeline = Pipeline()
            self.config = Config()
            
            # Setup color stream
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if not color_profiles or color_profiles.get_count() == 0:
                print("❌ No color profiles available")
                return False
            
            # Try to get matching resolution from calibration
            calib_width = self.calibration_data.get("image_width", 640)
            calib_height = self.calibration_data.get("image_height", 480)
            
            try:
                color_profile = color_profiles.get_video_stream_profile(calib_width, calib_height, OBFormat.RGB, 15)
            except OBError:
                try:
                    color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 15)
                except OBError:
                    color_profile = color_profiles.get_default_video_stream_profile()
            
            if not color_profile:
                print("❌ Could not get suitable color profile")
                return False
            
            self.config.enable_stream(color_profile)
            self.pipeline.start(self.config)
            
            print(f"✅ Camera setup for validation: {color_profile.get_width()}x{color_profile.get_height()}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup camera: {e}")
            return False
    
    def test_undistortion(self, duration: float = 10.0) -> bool:
        """Test calibration by showing undistorted camera feed.
        
        Args:
            duration: How long to run the test (seconds)
            
        Returns:
            True if test completed successfully
        """
        print(f"\n📹 Testing undistortion for {duration} seconds...")
        print("Press 'q' to quit early, 's' to save test image")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Get frame
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert to BGR
                bgr_image = self._frame_to_bgr(color_frame)
                if bgr_image is None:
                    continue
                
                # Apply undistortion
                undistorted = cv2.undistort(bgr_image, self.camera_matrix, self.distortion_coeffs)
                
                # Create side-by-side comparison
                h, w = bgr_image.shape[:2]
                comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison[:, :w] = bgr_image
                comparison[:, w:] = undistorted
                
                # Add labels
                cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(comparison, "Undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add grid lines to help see distortion correction
                self._draw_grid(comparison[:, :w], (255, 0, 0))  # Blue grid on original
                self._draw_grid(comparison[:, w:], (0, 255, 0))  # Green grid on undistorted
                
                cv2.imshow('Calibration Validation - Original vs Undistorted', comparison)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save test image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"calibration_test_{timestamp}.jpg"
                    cv2.imwrite(filename, comparison)
                    print(f"💾 Saved test image: {filename}")
            
            cv2.destroyAllWindows()
            print("✅ Undistortion test completed")
            return True
            
        except Exception as e:
            print(f"❌ Error during undistortion test: {e}")
            cv2.destroyAllWindows()
            return False
    
    def _draw_grid(self, image: np.ndarray, color: Tuple[int, int, int]):
        """Draw grid lines on image to visualize distortion."""
        h, w = image.shape[:2]
        
        # Vertical lines
        for x in range(0, w, w // 8):
            cv2.line(image, (x, 0), (x, h), color, 1)
        
        # Horizontal lines  
        for y in range(0, h, h // 6):
            cv2.line(image, (0, y), (w, y), color, 1)
    
    def validate_with_checkerboard(self, checkerboard_size: Tuple[int, int] = (10, 7)) -> bool:
        """Validate calibration using live checkerboard detection.
        
        Args:
            checkerboard_size: Size of checkerboard pattern
            
        Returns:
            True if validation successful
        """
        print(f"\n🎯 Live checkerboard validation...")
        print("Hold a checkerboard pattern in front of the camera")
        print("Press SPACE to test detection, 'q' to quit")
        
        detection_count = 0
        error_sum = 0
        
        try:
            while True:
                # Get frame
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                bgr_image = self._frame_to_bgr(color_frame)
                if bgr_image is None:
                    continue
                
                # Undistort image
                undistorted = cv2.undistort(bgr_image, self.camera_matrix, self.distortion_coeffs)
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                display_image = undistorted.copy()
                
                if ret:
                    # Refine corners
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    
                    # Draw corners
                    cv2.drawChessboardCorners(display_image, checkerboard_size, corners_refined, ret)
                    cv2.putText(display_image, "PATTERN DETECTED! Press SPACE to validate", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_image, "Position checkerboard pattern", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(display_image, f"Validations: {detection_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Checkerboard Validation', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and ret:
                    # Calculate reprojection error for this detection
                    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
                    
                    # Project points back
                    rvec, tvec = cv2.solvePnPRansac(objp, corners_refined, self.camera_matrix, self.distortion_coeffs)[:2]
                    projected_points, _ = cv2.projectPoints(objp, rvec, tvec, self.camera_matrix, self.distortion_coeffs)
                    
                    error = cv2.norm(corners_refined, projected_points, cv2.NORM_L2) / len(projected_points)
                    error_sum += error
                    detection_count += 1
                    
                    print(f"✅ Validation {detection_count}: reprojection error = {error:.3f} pixels")
                
                elif key == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            
            if detection_count > 0:
                avg_error = error_sum / detection_count
                print(f"📊 Average reprojection error: {avg_error:.3f} pixels ({detection_count} tests)")
                
                if avg_error < 0.5:
                    print("✅ Excellent calibration quality!")
                elif avg_error < 1.0:
                    print("✅ Good calibration quality")
                else:
                    print("⚠️  Calibration quality could be improved")
                
                return True
            else:
                print("❌ No successful validations performed")
                return False
                
        except Exception as e:
            print(f"❌ Error during checkerboard validation: {e}")
            cv2.destroyAllWindows()
            return False
    
    def _frame_to_bgr(self, frame) -> Optional[np.ndarray]:
        """Convert Orbbec frame to BGR image."""
        try:
            width = frame.get_width()
            height = frame.get_height()
            color_format = frame.get_format()
            data = np.asanyarray(frame.get_data())
            
            if color_format == OBFormat.RGB:
                image = np.resize(data, (height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif color_format == OBFormat.BGR:
                image = np.resize(data, (height, width, 3))
            elif color_format == OBFormat.YUYV:
                image = np.resize(data, (height, width, 2))
                image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
            elif color_format == OBFormat.MJPG:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            else:
                return None
            
            return image
            
        except Exception:
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="DOFBOT Pro Camera Calibration Validation Tool")
    parser.add_argument("--calibration-file", type=str, default="./calibration_data/camera_calibration.json",
                       help="Path to calibration file")
    parser.add_argument("--test-undistortion", action="store_true", help="Test undistortion with live feed")
    parser.add_argument("--test-checkerboard", action="store_true", help="Test with checkerboard pattern")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    print("🔍 DOFBOT Pro Camera Calibration Validation Tool")
    print("=" * 55)
    
    if not ORBBEC_SDK_AVAILABLE:
        print("❌ Orbbec SDK not available. Please install pyorbbecsdk.")
        return 1
    
    # Create validator
    validator = CalibrationValidator(args.calibration_file)
    
    try:
        # Load calibration
        if not validator.load_calibration():
            return 1
        
        # Validate parameters
        validation_results = validator.validate_calibration_parameters()
        
        # Print results
        print(f"\n📊 Calibration Information:")
        for key, value in validation_results["info"].items():
            print(f"  {key}: {value}")
        
        if validation_results["warnings"]:
            print(f"\n⚠️  Warnings:")
            for warning in validation_results["warnings"]:
                print(f"  - {warning}")
        
        if validation_results["errors"]:
            print(f"\n❌ Errors:")
            for error in validation_results["errors"]:
                print(f"  - {error}")
            return 1
        
        # Live tests if requested
        if args.test_undistortion or args.test_checkerboard:
            if not validator.setup_camera():
                print("❌ Cannot run live tests - camera setup failed")
                return 1
            
            if args.test_undistortion:
                validator.test_undistortion(args.duration)
            
            if args.test_checkerboard:
                validator.validate_with_checkerboard()
        
        print("\n✅ Calibration validation completed!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
        
    finally:
        validator.cleanup()


if __name__ == "__main__":
    exit(main())
