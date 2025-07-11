#!/usr/bin/env python3
"""
DOFBOT Pro Depth Camera Calibration Tool

Standalone utility for calibrating the Orbbec depth camera using checkerboard patterns.
This tool should be run during robot setup or maintenance, not as part of normal operations.

Usage:
    python3 calibrate_depth_camera.py [options]

Requirements:
    - Checkerboard calibration pattern (printed)
    - Good lighting conditions
    - Stable camera mount
    - pyorbbecsdk installed
"""

import cv2
import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Try to import Orbbec SDK
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, FrameSet, OBError
    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    print("❌ Error: pyorbbecsdk not available. Please install the Orbbec SDK.")
    ORBBEC_SDK_AVAILABLE = False
    exit(1)


class DepthCameraCalibrator:
    """Handles depth camera calibration using checkerboard patterns."""
    
    def __init__(self, output_dir: str = "./calibration_data"):
        """Initialize the calibrator.
        
        Args:
            output_dir: Directory to save calibration results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Camera setup
        self.pipeline = None
        self.config = None
        
        # Calibration data
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.calibration_images = []
        
        # Results
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_error = None
        
    def setup_camera(self) -> bool:
        """Setup Orbbec camera for calibration.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.pipeline = Pipeline()
            self.config = Config()
            
            # Setup color stream for calibration
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if not color_profiles or color_profiles.get_count() == 0:
                print("❌ No color profiles available")
                return False
            
            # Try to get 640x480 RGB at 15fps
            try:
                color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 15)
            except OBError:
                try:
                    color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 10)
                except OBError:
                    color_profile = color_profiles.get_default_video_stream_profile()
            
            if not color_profile:
                print("❌ Could not get suitable color profile")
                return False
            
            self.config.enable_stream(color_profile)
            self.pipeline.start(self.config)
            
            print(f"✅ Camera initialized: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup camera: {e}")
            return False
    
    def capture_calibration_images(self, 
                                 checkerboard_size: Tuple[int, int],
                                 square_size_mm: float,
                                 num_images: int = 15,
                                 capture_delay: float = 2.0) -> bool:
        """Capture images for calibration.
        
        Args:
            checkerboard_size: (width, height) of checkerboard corners
            square_size_mm: Size of checkerboard squares in millimeters
            num_images: Number of calibration images to capture
            capture_delay: Delay between captures in seconds
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n📸 Starting calibration image capture...")
        print(f"Target: {num_images} images with {checkerboard_size[0]}x{checkerboard_size[1]} checkerboard")
        print(f"Square size: {square_size_mm}mm")
        print(f"\nInstructions:")
        print("- Hold checkerboard pattern in front of camera")
        print("- Move pattern to different positions and angles")
        print("- Press SPACE when pattern is detected to capture")
        print("- Press ESC to finish early")
        print("- Press 'q' to quit\n")
        
        # Prepare object points (3D coordinates of checkerboard corners)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm
        
        captured_count = 0
        last_capture_time = 0
        
        print("🎯 Ready to capture! Position your checkerboard pattern...")
        
        while captured_count < num_images:
            try:
                # Get frame from camera
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert to BGR for OpenCV
                bgr_image = self._frame_to_bgr(color_frame)
                if bgr_image is None:
                    continue
                
                # Convert to grayscale for corner detection
                gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                # Draw corners and status
                display_image = bgr_image.copy()
                
                if ret:
                    # Refine corner positions
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    
                    # Draw corners
                    cv2.drawChessboardCorners(display_image, checkerboard_size, corners_refined, ret)
                    
                    # Add status text
                    cv2.putText(display_image, f"PATTERN DETECTED! Press SPACE to capture ({captured_count}/{num_images})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if enough time has passed since last capture
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_delay:
                        cv2.putText(display_image, "Ready to capture!", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        remaining = capture_delay - (current_time - last_capture_time)
                        cv2.putText(display_image, f"Wait {remaining:.1f}s before next capture", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display_image, f"Position checkerboard pattern ({captured_count}/{num_images})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show image
                cv2.imshow('Camera Calibration', display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' ') and ret:  # Space to capture
                    current_time = time.time()
                    if current_time - last_capture_time >= capture_delay:
                        # Save calibration data
                        self.object_points.append(objp)
                        self.image_points.append(corners_refined)
                        self.calibration_images.append(bgr_image.copy())
                        
                        captured_count += 1
                        last_capture_time = current_time
                        
                        print(f"✅ Captured image {captured_count}/{num_images}")
                        
                        # Save image for review
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = self.output_dir / f"calibration_image_{captured_count:02d}_{timestamp}.jpg"
                        cv2.imwrite(str(image_path), bgr_image)
                    else:
                        print(f"⏳ Please wait {capture_delay:.1f}s between captures")
                
                elif key == 27:  # ESC to finish early
                    print(f"\n⏹️  Capture stopped early. Got {captured_count} images.")
                    break
                    
                elif key == ord('q'):  # Q to quit
                    print("\n❌ Calibration cancelled by user")
                    return False
                
            except Exception as e:
                print(f"❌ Error during capture: {e}")
                continue
        
        cv2.destroyAllWindows()
        
        if captured_count < 5:
            print(f"❌ Not enough images captured ({captured_count}). Need at least 5 for calibration.")
            return False
        
        print(f"✅ Captured {captured_count} calibration images")
        return True
    
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
            
        except Exception as e:
            print(f"Error converting frame: {e}")
            return None
    
    def calculate_calibration(self, image_size: Tuple[int, int]) -> bool:
        """Calculate camera calibration from captured images.
        
        Args:
            image_size: (width, height) of calibration images
            
        Returns:
            True if calibration successful, False otherwise
        """
        if len(self.object_points) < 5:
            print("❌ Not enough calibration data. Need at least 5 image pairs.")
            return False
        
        print(f"\n🔧 Calculating calibration from {len(self.object_points)} image pairs...")
        
        try:
            # Perform camera calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, image_size, None, None
            )
            
            if not ret:
                print("❌ Camera calibration failed")
                return False
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.object_points)):
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error
            
            mean_error = total_error / len(self.object_points)
            
            # Store results
            self.camera_matrix = camera_matrix
            self.distortion_coeffs = dist_coeffs
            self.calibration_error = mean_error
            
            print(f"✅ Calibration successful!")
            print(f"📊 Mean reprojection error: {mean_error:.3f} pixels")
            print(f"📐 Camera matrix:")
            print(f"    fx: {camera_matrix[0,0]:.2f}")
            print(f"    fy: {camera_matrix[1,1]:.2f}")
            print(f"    cx: {camera_matrix[0,2]:.2f}")
            print(f"    cy: {camera_matrix[1,2]:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during calibration calculation: {e}")
            return False
    
    def save_calibration(self) -> bool:
        """Save calibration results to file.
        
        Returns:
            True if save successful, False otherwise
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            print("❌ No calibration data to save")
            return False
        
        try:
            # Prepare calibration data
            calibration_data = {
                "calibration_date": datetime.now().isoformat(),
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coeffs.tolist(),
                "reprojection_error": float(self.calibration_error),
                "num_images_used": len(self.object_points),
                "image_width": 640,  # Standard resolution
                "image_height": 480,
                "calibration_method": "checkerboard",
                "notes": "DOFBOT Pro depth camera calibration"
            }
            
            # Save to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calibration_file = self.output_dir / f"camera_calibration_{timestamp}.json"
            
            with open(calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            # Also save as the standard filename for the robot to use
            standard_file = self.output_dir / "camera_calibration.json"
            with open(standard_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"✅ Calibration saved to:")
            print(f"    {calibration_file}")
            print(f"    {standard_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False
    
    def cleanup(self):
        """Cleanup camera resources."""
        try:
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


def main():
    """Main calibration function."""
    parser = argparse.ArgumentParser(description="DOFBOT Pro Depth Camera Calibration Tool")
    parser.add_argument("--width", type=int, default=10, help="Checkerboard width (corners)")
    parser.add_argument("--height", type=int, default=7, help="Checkerboard height (corners)")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square size in mm")
    parser.add_argument("--num-images", type=int, default=15, help="Number of calibration images")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between captures (seconds)")
    parser.add_argument("--output-dir", type=str, default="./calibration_data", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎯 DOFBOT Pro Depth Camera Calibration Tool")
    print("=" * 50)
    
    if not ORBBEC_SDK_AVAILABLE:
        print("❌ Orbbec SDK not available. Please install pyorbbecsdk.")
        return 1
    
    # Create calibrator
    calibrator = DepthCameraCalibrator(args.output_dir)
    
    try:
        # Setup camera
        if not calibrator.setup_camera():
            print("❌ Failed to setup camera")
            return 1
        
        # Capture calibration images
        if not calibrator.capture_calibration_images(
            (args.width, args.height), args.square_size, args.num_images, args.delay
        ):
            print("❌ Failed to capture calibration images")
            return 1
        
        # Calculate calibration
        if not calibrator.calculate_calibration((640, 480)):
            print("❌ Failed to calculate calibration")
            return 1
        
        # Save results
        if not calibrator.save_calibration():
            print("❌ Failed to save calibration")
            return 1
        
        print("\n🎉 Camera calibration completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print("\n💡 Next steps:")
        print("1. Copy camera_calibration.json to your robot's data directory")
        print("2. Run validate_camera_calibration.py to verify the results")
        print("3. Test vision-based robot actions")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Calibration interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
        
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    exit(main())
