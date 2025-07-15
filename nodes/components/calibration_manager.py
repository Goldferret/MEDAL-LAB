"""
Calibration Manager Component for DOFBOT Pro

Handles all calibration operations including:
- Camera calibration using checkerboard patterns
- Depth camera calibration and validation
- Parameter loading and saving
- Calibration validation and testing
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class CalibrationManager:
    """Manages calibration operations for DOFBOT Pro robot."""
    
    def __init__(self, logger, camera_manager, data_path: str = "./captures", experiment_logger=None):
        """Initialize calibration manager.
        
        Args:
            logger: Logger instance for logging
            camera_manager: CameraManager instance for frame capture
            data_path: Path for saving calibration data
            experiment_logger: ExperimentLogger instance for consistent data saving
        """
        self.logger = logger
        self.camera_manager = camera_manager
        self.experiment_logger = experiment_logger
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # Default calibration parameters
        self.default_calibration = {
            "fx": 525.0,
            "fy": 525.0,
            "cx": 320.0,
            "cy": 240.0,
            "depth_scale": 1000.0,
            "width": 640,
            "height": 480,
            "camera_matrix": [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
    
    def perform_camera_calibration(self, 
                                 checkerboard_width: int = 10,
                                 checkerboard_height: int = 7,
                                 square_size_mm: float = 25.0,
                                 num_images: int = 15,
                                 capture_delay: float = 2.0) -> Dict[str, Any]:
        """Perform camera calibration using checkerboard pattern.
        
        Args:
            checkerboard_width: Number of internal corners horizontally (default: 10 for 11x8 board)
            checkerboard_height: Number of internal corners vertically (default: 7 for 11x8 board)
            square_size_mm: Physical size of checkerboard squares in millimeters (default: 25.0)
            num_images: Number of calibration images to capture (default: 15)
            capture_delay: Delay between image captures in seconds (default: 2.0)
            
        Returns:
            Dictionary containing calibration results
        """
        if not self.camera_manager.is_available():
            return {"success": False, "error": "Camera not available"}
        
        try:
            checkerboard_size = (checkerboard_width, checkerboard_height)
            
            # Prepare object points (3D points in real world space)
            objp = np.zeros((checkerboard_width * checkerboard_height, 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_width, 0:checkerboard_height].T.reshape(-1, 2)
            objp *= square_size_mm
            
            # Arrays to store object points and image points from all images
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane
            
            captured_images = 0
            failed_attempts = 0
            max_failed_attempts = num_images * 3  # Allow some failed attempts
            
            self.logger.log_info(f"Starting camera calibration...")
            self.logger.log_info(f"Checkerboard: {checkerboard_width}x{checkerboard_height} internal corners")
            self.logger.log_info(f"Square size: {square_size_mm}mm")
            self.logger.log_info(f"Target images: {num_images}")
            self.logger.log_info("Move the checkerboard to different positions and angles between captures")
            
            while captured_images < num_images and failed_attempts < max_failed_attempts:
                # Capture image using optimized synchronized method
                rgb_img, _, _, _ = self.camera_manager.capture_synchronized_data()
                if rgb_img is None:
                    failed_attempts += 1
                    self.logger.log_warning(f"Failed to capture image (attempt {failed_attempts})")
                    time.sleep(0.5)
                    continue
                
                # Note: capture_synchronized_data() already returns BGR format
                img = rgb_img  # Actually BGR format despite variable name
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                if ret:
                    # Refine corner positions
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Store object points and image points
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    
                    captured_images += 1
                    self.logger.log_info(f"Captured calibration image {captured_images}/{num_images}")
                    
                    # Save calibration image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = self.data_path / f"calibration_img_{captured_images}_{timestamp}.jpg"
                    
                    # Draw and display the corners
                    img_with_corners = img.copy()
                    cv2.drawChessboardCorners(img_with_corners, checkerboard_size, corners2, ret)
                    
                    # Use experiment logger if available, otherwise fallback to direct save
                    if self.experiment_logger:
                        self.experiment_logger.save_frame(img_with_corners, str(img_filename), "calibration")
                    else:
                        cv2.imwrite(str(img_filename), img_with_corners)
                    
                    # Wait before next capture
                    if captured_images < num_images:
                        self.logger.log_info(f"Waiting {capture_delay}s before next capture...")
                        time.sleep(capture_delay)
                else:
                    failed_attempts += 1
                    if failed_attempts % 5 == 0:
                        self.logger.log_warning(f"No checkerboard found in {failed_attempts} attempts. Ensure checkerboard is visible and well-lit.")
                    time.sleep(0.1)
            
            if captured_images < num_images:
                return {
                    "success": False,
                    "error": f"Only captured {captured_images}/{num_images} images",
                    "captured_images": captured_images
                }
            
            # Perform camera calibration
            self.logger.log_info("Performing camera calibration...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if not ret:
                return {"success": False, "error": "Camera calibration failed"}
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            mean_error /= len(objpoints)
            
            # Extract individual parameters
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            
            # Prepare calibration data
            calibration_data = {
                "success": True,
                "calibration_date": datetime.now().isoformat(),
                "mean_reprojection_error": float(mean_error),
                "num_images_used": captured_images,
                
                # Camera matrix
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.flatten().tolist(),
                
                # Image dimensions
                "image_width": gray.shape[1],
                "image_height": gray.shape[0],
                
                # Individual parameters for point cloud generation
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "depth_scale": 1000.0,  # Default depth scale
                
                # Calibration settings used
                "checkerboard_size": [checkerboard_width, checkerboard_height],
                "square_size_mm": square_size_mm
            }
            
            # Save calibration data
            calibration_file = self.data_path / "camera_calibration.json"
            
            # Use experiment logger if available, otherwise fallback to direct save
            if self.experiment_logger:
                self.experiment_logger.save_json_data(calibration_data, str(calibration_file), "calibration")
            else:
                with open(calibration_file, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
            
            self.logger.log_info(f"Camera calibration completed successfully!")
            self.logger.log_info(f"Mean reprojection error: {mean_error:.3f} pixels")
            self.logger.log_info(f"Calibration saved to: {calibration_file}")
            
            return calibration_data
            
        except Exception as e:
            self.logger.log_error(f"Error during camera calibration: {e}")
            return {"success": False, "error": str(e)}
    
    def load_camera_calibration(self) -> Dict[str, Any]:
        """Load camera calibration parameters from file.
        
        Returns:
            Dictionary containing calibration parameters
        """
        try:
            calibration_file = self.data_path / "camera_calibration.json"
            if calibration_file.exists():
                with open(calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                
                # Extract camera matrix if available
                if "camera_matrix" in calibration_data:
                    matrix = calibration_data["camera_matrix"]
                    result = self.default_calibration.copy()
                    result.update({
                        "fx": matrix[0][0],
                        "fy": matrix[1][1],
                        "cx": matrix[0][2],
                        "cy": matrix[1][2],
                        "depth_scale": calibration_data.get("depth_scale", 1000.0),
                        "width": calibration_data.get("image_width", 640),
                        "height": calibration_data.get("image_height", 480)
                    })
                    return result
                else:
                    return calibration_data
            else:
                self.logger.log_info("No calibration file found, using defaults")
                return self.default_calibration.copy()
                
        except Exception as e:
            self.logger.log_error(f"Error loading camera calibration: {e}")
            return self.default_calibration.copy()
    
    def validate_camera_calibration(self) -> Dict[str, Any]:
        """Validate the current camera calibration.
        
        Returns:
            Dictionary containing validation results
        """
        try:
            calibration = self.load_camera_calibration()
            
            validation_results = {
                "calibration_loaded": True,
                "calibration_date": calibration.get("calibration_date", "Unknown"),
                "image_dimensions": f"{calibration.get('width', 'Unknown')}x{calibration.get('height', 'Unknown')}",
                "focal_lengths": f"fx={calibration['fx']:.1f}, fy={calibration['fy']:.1f}",
                "principal_point": f"cx={calibration['cx']:.1f}, cy={calibration['cy']:.1f}",
                "reprojection_error": calibration.get("mean_reprojection_error", "Unknown"),
                "warnings": [],
                "errors": []
            }
            
            # Check for reasonable parameter values
            fx, fy = calibration["fx"], calibration["fy"]
            cx, cy = calibration["cx"], calibration["cy"]
            width, height = calibration.get("width", 640), calibration.get("height", 480)
            
            # Validate focal lengths (should be positive and reasonable for the sensor)
            if fx <= 0 or fy <= 0:
                validation_results["errors"].append("Invalid focal lengths (must be positive)")
            elif fx < 200 or fx > 1000 or fy < 200 or fy > 1000:
                validation_results["warnings"].append("Focal lengths seem unusual for this camera")
            
            # Validate principal point (should be near image center)
            if abs(cx - width/2) > width/4 or abs(cy - height/2) > height/4:
                validation_results["warnings"].append("Principal point far from image center")
            
            # Check reprojection error if available
            if "mean_reprojection_error" in calibration:
                error = calibration["mean_reprojection_error"]
                if error > 1.0:
                    validation_results["warnings"].append(f"High reprojection error: {error:.3f} pixels")
                elif error > 2.0:
                    validation_results["errors"].append(f"Very high reprojection error: {error:.3f} pixels")
            
            validation_results["valid"] = len(validation_results["errors"]) == 0
            
            return validation_results
            
        except Exception as e:
            return {
                "calibration_loaded": False,
                "valid": False,
                "errors": [f"Error validating calibration: {str(e)}"]
            }
    
    def test_calibration_accuracy(self, num_test_images: int = 5) -> Dict[str, Any]:
        """Test calibration accuracy by capturing new images.
        
        Args:
            num_test_images: Number of test images to capture
            
        Returns:
            Dictionary containing test results
        """
        if not self.camera_manager.is_available():
            return {"success": False, "error": "Camera not available"}
        
        try:
            calibration = self.load_camera_calibration()
            camera_matrix = np.array(calibration["camera_matrix"])
            dist_coeffs = np.array(calibration["distortion_coefficients"])
            
            test_results = {
                "success": True,
                "num_test_images": 0,
                "reprojection_errors": [],
                "mean_test_error": 0.0,
                "calibration_error": calibration.get("mean_reprojection_error", 0.0)
            }
            
            checkerboard_size = (10, 7)  # Default checkerboard size
            
            # Prepare object points
            objp = np.zeros((10 * 7, 3), np.float32)
            objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)
            objp *= 25.0  # 25mm squares
            
            self.logger.log_info(f"Testing calibration accuracy with {num_test_images} images...")
            
            for i in range(num_test_images):
                # Capture test image using optimized synchronized method
                rgb_img, _, _, _ = self.camera_manager.capture_synchronized_data()
                if rgb_img is None:
                    continue
                
                # Note: capture_synchronized_data() already returns BGR format
                img = rgb_img  # Actually BGR format despite variable name
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                if ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Calculate pose
                    ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
                    
                    if ret_pnp:
                        # Project points back to image
                        projected_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
                        
                        # Calculate reprojection error
                        error = cv2.norm(corners2, projected_points, cv2.NORM_L2) / len(projected_points)
                        test_results["reprojection_errors"].append(float(error))
                        test_results["num_test_images"] += 1
                        
                        self.logger.log_info(f"Test image {i+1}: reprojection error = {error:.3f} pixels")
                
                time.sleep(1.0)  # Delay between captures
            
            if test_results["num_test_images"] > 0:
                test_results["mean_test_error"] = np.mean(test_results["reprojection_errors"])
                self.logger.log_info(f"Calibration test completed: mean error = {test_results['mean_test_error']:.3f} pixels")
            else:
                test_results["success"] = False
                test_results["error"] = "No valid test images captured"
            
            return test_results
            
        except Exception as e:
            self.logger.log_error(f"Error testing calibration accuracy: {e}")
            return {"success": False, "error": str(e)}
    
    def save_calibration_backup(self) -> bool:
        """Save a backup of the current calibration.
        
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            calibration_file = self.data_path / "camera_calibration.json"
            if not calibration_file.exists():
                self.logger.log_warning("No calibration file to backup")
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.data_path / f"camera_calibration_backup_{timestamp}.json"
            
            # Copy calibration file
            with open(calibration_file, 'r') as src:
                calibration_data = json.load(src)
            
            # Use experiment logger if available, otherwise fallback to direct save
            if self.experiment_logger:
                self.experiment_logger.save_json_data(calibration_data, str(backup_file), "calibration_backup")
            else:
                with open(backup_file, 'w') as dst:
                    json.dump(calibration_data, dst, indent=2)
            
            self.logger.log_info(f"Calibration backup saved to: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Error saving calibration backup: {e}")
            return False
    
    def restore_calibration_backup(self, backup_filename: str) -> bool:
        """Restore calibration from a backup file.
        
        Args:
            backup_filename: Name of the backup file to restore
            
        Returns:
            True if restore was successful, False otherwise
        """
        try:
            backup_file = self.data_path / backup_filename
            if not backup_file.exists():
                self.logger.log_error(f"Backup file not found: {backup_filename}")
                return False
            
            # Save current calibration as backup before restoring
            self.save_calibration_backup()
            
            # Restore from backup
            calibration_file = self.data_path / "camera_calibration.json"
            with open(backup_file, 'r') as src:
                calibration_data = json.load(src)
            
            # Use experiment logger if available, otherwise fallback to direct save
            if self.experiment_logger:
                self.experiment_logger.save_json_data(calibration_data, str(calibration_file), "calibration_restore")
            else:
                with open(calibration_file, 'w') as dst:
                    json.dump(calibration_data, dst, indent=2)
            
            self.logger.log_info(f"Calibration restored from: {backup_filename}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Error restoring calibration backup: {e}")
            return False
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current calibration status.
        
        Returns:
            Dictionary containing calibration summary
        """
        try:
            calibration = self.load_camera_calibration()
            validation = self.validate_camera_calibration()
            
            return {
                "calibration_available": validation["calibration_loaded"],
                "calibration_valid": validation["valid"],
                "calibration_date": calibration.get("calibration_date", "Unknown"),
                "parameters": {
                    "fx": calibration["fx"],
                    "fy": calibration["fy"],
                    "cx": calibration["cx"],
                    "cy": calibration["cy"],
                    "image_size": f"{calibration['width']}x{calibration['height']}"
                },
                "quality": {
                    "reprojection_error": calibration.get("mean_reprojection_error", "Unknown"),
                    "num_images": calibration.get("num_images_used", "Unknown")
                },
                "warnings": validation.get("warnings", []),
                "errors": validation.get("errors", [])
            }
            
        except Exception as e:
            return {
                "calibration_available": False,
                "error": str(e)
            }
