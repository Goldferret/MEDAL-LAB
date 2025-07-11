"""
Modular Robot Arm Interface for DOFBOT Pro

Coordinates all robot components and provides high-level functionality:
- Camera operations via CameraManager
- Vision processing via VisionDetector  
- Movement control via MovementController
- Calibration via CalibrationManager
"""

import cv2
import time
import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from functools import wraps

from components import CameraManager, VisionDetector, MovementController, CalibrationManager


def handle_component_errors(component_name: str):
    """Decorator for consistent component error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                self.logger.log_error(f"Error in {component_name}: {e}")
                return None
        return wrapper
    return decorator


class RobotArmInterface:
    """Modular interface for DOFBOT Pro robot arm with component-based architecture."""
    
    def __init__(self, logger, config, device_number: int = 0, data_path: str = "./captures"):
        """Initialize the modular robot arm interface.
        
        Args:
            logger: Logger instance for logging
            config: Configuration object
            device_number: Device number for the robot arm
            data_path: Path for saving data and captures
        """
        self.logger = logger
        self.config = config
        self.device_number = device_number
        self.data_path = data_path
        
        # Initialize components
        self.logger.log_info("Initializing robot arm components...")
        
        # Camera Manager - handles all camera operations
        self.camera_manager = CameraManager(logger, data_path)
        
        # Vision Detector - handles computer vision
        self.vision_detector = VisionDetector(logger, config)
        
        # Movement Controller - handles robot movement
        self.movement_controller = MovementController(logger, config)
        
        # Calibration Manager - handles calibration operations
        self.calibration_manager = CalibrationManager(logger, self.camera_manager, data_path)
        
        self.logger.log_info("Robot arm interface initialized successfully")
    
    def __getattr__(self, name):
        """Automatically delegate to appropriate component."""
        # Try movement controller first (most commonly used)
        if hasattr(self.movement_controller, name):
            return getattr(self.movement_controller, name)
        # Then camera manager
        elif hasattr(self.camera_manager, name):
            return getattr(self.camera_manager, name)
        # Then calibration manager
        elif hasattr(self.calibration_manager, name):
            return getattr(self.calibration_manager, name)
        # Then vision detector
        elif hasattr(self.vision_detector, name):
            return getattr(self.vision_detector, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    # ========== Component Access Properties ==========
    
    @property
    def joint_angles(self) -> List[int]:
        """Get current joint angles from movement controller."""
        return self.movement_controller.joint_angles
    
    @property
    def is_moving(self) -> bool:
        """Check if robot is currently moving."""
        return self.movement_controller.is_moving
    
    @property
    def gripper_closed(self) -> bool:
        """Check if gripper is closed."""
        return self.movement_controller.gripper_closed
    
    @property
    def recording(self) -> bool:
        """Check if trajectory recording is active."""
        return self.movement_controller.recording
    
    # ========== High-Level Operations ==========
    
    def scan_working_area(self, object_type: str, color: str) -> Dict[str, Any]:
        """Scan the working area by rotating servo 1 to find specified target object.
        
        Args:
            object_type: Type of object to find ('cube' or 'rectangular_prism')
            color: Color of object to find ('red', 'green', 'blue', 'yellow')
            
        Returns:
            Dictionary with scanning results
        """
        scan_positions = self.config.scan_positions
        scan_delay = self.config.scan_delay
        
        self.logger.log_info(f"Starting working area scan for {color} {object_type}")
        self.logger.log_info(f"Scan positions: {scan_positions}")
        
        start_time = time.time()
        positions_checked = []
        
        try:
            for position in scan_positions:
                self.logger.log_info(f"Scanning at servo1={position}°...")
                
                # Move servo 1 to scanning position
                self.movement_controller.move_single_joint(1, position, record_action=False)
                positions_checked.append(position)
                
                # Wait for camera to stabilize
                time.sleep(scan_delay)
                
                # Check for target object at this position
                detection_result = self._scan_at_position(position, object_type, color)
                
                if detection_result["found"]:
                    # Target found - return success
                    scan_time = time.time() - start_time
                    return self._build_scan_result(
                        success=True,
                        target_found=True,
                        position=position,
                        detection_data=detection_result,
                        positions_checked=positions_checked,
                        scan_time=scan_time,
                        object_type=object_type,
                        color=color
                    )
            
            # Target not found after scanning all positions
            scan_time = time.time() - start_time
            self.logger.log_info(f"Scan completed - no {color} {object_type} found")
            
            return self._build_scan_result(
                success=False,
                target_found=False,
                positions_checked=positions_checked,
                scan_time=scan_time,
                object_type=object_type,
                color=color
            )
            
        except Exception as e:
            self.logger.log_error(f"Error during scan_working_area: {e}")
            return self._build_scan_result(
                success=False,
                target_found=False,
                error_message=f"Scanning failed: {str(e)}",
                positions_checked=positions_checked,
                scan_time=time.time() - start_time,
                object_type=object_type,
                color=color
            )
    
    def _scan_at_position(self, position: int, object_type: str, color: str) -> Dict[str, Any]:
        """Scan for target object at a specific position.
        
        Args:
            position: Servo position to scan at
            object_type: Type of object to find
            color: Color of object to find
            
        Returns:
            Dictionary with detection results
        """
        detection_attempts = 3
        
        for attempt in range(detection_attempts):
            self.logger.log_info(f"Detection attempt {attempt + 1}/{detection_attempts} at position {position}°")
            
            try:
                # Capture frame using simple method (more reliable)
                bgr_image = self.camera_manager.capture_frame_simple()
                if bgr_image is None:
                    self.logger.log_warning(f"Failed to capture frame (attempt {attempt + 1})")
                    continue
                
                self.logger.log_info(f"Successfully captured frame at position {position}°, attempt {attempt + 1}")
                
                # Save debug frame
                self._save_debug_frame(bgr_image, position, attempt + 1)
                
                # Detect target object
                detection_result = self.vision_detector.detect_target_object(bgr_image, object_type, color)
                
                if detection_result is not None:
                    centroid, confidence = detection_result
                    self.logger.log_info(f"🎯 TARGET DETECTED at position {position}°: {color} {object_type}")
                    self.logger.log_info(f"   Centroid: ({centroid[0]}, {centroid[1]}), Confidence: {confidence:.2f}")
                    
                    if confidence > self.config.confidence_threshold:
                        self.logger.log_info(f"✅ Target accepted (confidence {confidence:.2f} > {self.config.confidence_threshold})")
                        return {
                            "found": True,
                            "centroid": centroid,
                            "confidence": confidence,
                            "attempt": attempt + 1
                        }
                    else:
                        self.logger.log_info(f"❌ Target rejected (confidence {confidence:.2f} < {self.config.confidence_threshold})")
                else:
                    self.logger.log_info(f"❌ No {color} {object_type} detected at position {position}°, attempt {attempt + 1}")
                
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.log_error(f"Detection attempt {attempt + 1} failed at position {position}°: {e}")
                continue
        
        return {"found": False}
    
    def _save_debug_frame(self, bgr_image: np.ndarray, position: int, attempt: int):
        """Save debug frame for analysis."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_filename = f"scan_debug_pos{position}_attempt{attempt}_{timestamp}.jpg"
            frame_path = Path(self.data_path) / frame_filename
            frame_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(frame_path), bgr_image)
            self.logger.log_info(f"Saved debug frame: {frame_filename}")
        except Exception as e:
            self.logger.log_warning(f"Failed to save debug frame: {e}")
    
    def _build_scan_result(self, success: bool, target_found: bool, **kwargs) -> Dict[str, Any]:
        """Build standardized scan result dictionary."""
        result = {
            "success": success,
            "target_found": target_found,
            "current_joint_angles": self.joint_angles.copy(),
            "found_at_servo1_position": kwargs.get("position"),
            "object_centroid": None,
            "object_type": kwargs.get("object_type"),
            "object_color": kwargs.get("color"),
            "detection_confidence": 0.0,
            "scan_positions_checked": kwargs.get("positions_checked", []),
            "total_scan_time": kwargs.get("scan_time", 0.0)
        }
        
        # Add detection-specific data if found
        if target_found and "detection_data" in kwargs:
            detection_data = kwargs["detection_data"]
            if detection_data.get("found"):
                centroid = detection_data["centroid"]
                result["object_centroid"] = {"x": int(centroid[0]), "y": int(centroid[1])}
                result["detection_confidence"] = detection_data["confidence"]
        
        # Add error message if provided
        if "error_message" in kwargs:
            result["error_message"] = kwargs["error_message"]
        
        return result
    
    @handle_component_errors("camera centering")
    def center_camera_on_object(self, object_color: str, timeout: float) -> Dict[str, Any]:
        """Center the camera on a colored object using PID control.
        
        Args:
            object_color: Color of object to track ('red', 'green', 'blue', 'yellow')
            timeout: Maximum time to attempt centering (seconds)
            
        Returns:
            Dictionary with centering results
        """
        # Move to starting position first
        self.logger.log_info("Moving to starting position for object tracking...")
        try:
            self.movement_controller.move_all_joints(self.config.starting_scan_position, record_action=False)
            time.sleep(2)
        except Exception as e:
            self.logger.log_warning(f"Failed to move to starting position: {e}")
        
        # Get configuration parameters
        target_x, target_y = self.config.target_center_point
        pid_x = self.config.pid_params["x"]
        pid_y = self.config.pid_params["y"]
        position_threshold = self.config.position_threshold
        
        # PID state variables
        prev_error_x = prev_error_y = 0
        integral_x = integral_y = 0
        
        # Convergence parameters
        max_iterations = int(timeout * 5)  # 5Hz operation
        stable_count_required = 5
        min_object_detections = 3
        
        self.logger.log_info(f"Target center: ({target_x}, {target_y}), Threshold: ±{position_threshold}px")
        
        start_time = time.time()
        iteration = 0
        stable_count = 0
        last_centroid = None
        consecutive_detections = 0
        detection_history = []
        
        while iteration < max_iterations:
            current_time = time.time()
            if current_time - start_time > timeout:
                break
            
            # Capture frame using simple method
            bgr_image = self.camera_manager.capture_frame_simple()
            if bgr_image is None:
                continue
            
            # Detect colored object
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            detected_centroid = self.vision_detector.detect_colored_object_yahboom(
                bgr_image, None, None, object_color
            )
            
            if detected_centroid is None:
                consecutive_detections = 0
                stable_count = 0
                iteration += 1
                time.sleep(0.2)
                continue
            
            # Add to detection history for stability
            detection_history.append(detected_centroid)
            if len(detection_history) > 5:
                detection_history.pop(0)
            
            consecutive_detections += 1
            
            # Only proceed if we have stable detections
            if consecutive_detections < min_object_detections:
                iteration += 1
                time.sleep(0.2)
                continue
            
            # Use averaged centroid for stability
            if len(detection_history) >= 3:
                avg_x = sum(d[0] for d in detection_history[-3:]) / 3
                avg_y = sum(d[1] for d in detection_history[-3:]) / 3
                centroid_x, centroid_y = int(avg_x), int(avg_y)
            else:
                centroid_x, centroid_y = detected_centroid
            
            last_centroid = (centroid_x, centroid_y)
            
            # Calculate errors
            error_x = target_x - centroid_x
            error_y = target_y - centroid_y
            
            # Check if within threshold
            if abs(error_x) <= position_threshold and abs(error_y) <= position_threshold:
                stable_count += 1
                if stable_count >= stable_count_required:
                    return {
                        "success": True,
                        "final_centroid": {"x": int(centroid_x), "y": int(centroid_y)},
                        "final_error": {"x": int(error_x), "y": int(error_y)},
                        "iterations": iteration,
                        "time_elapsed": current_time - start_time,
                        "object_color": object_color
                    }
            else:
                stable_count = 0
            
            # PID Control (only if error is significant)
            if abs(error_x) > 2 or abs(error_y) > 2:
                dt = 0.2
                
                # X-axis PID
                integral_x += error_x * dt
                integral_x = max(-50, min(50, integral_x))  # Limit windup
                derivative_x = (error_x - prev_error_x) / dt
                pid_output_x = pid_x["kp"] * error_x + pid_x["ki"] * integral_x + pid_x["kd"] * derivative_x
                prev_error_x = error_x
                
                # Y-axis PID
                integral_y += error_y * dt
                integral_y = max(-50, min(50, integral_y))  # Limit windup
                derivative_y = (error_y - prev_error_y) / dt
                pid_output_y = pid_y["kp"] * error_y + pid_y["ki"] * integral_y + pid_y["kd"] * derivative_y
                prev_error_y = error_y
                
                # Convert to servo adjustments
                servo1_adjustment = pid_output_x * 0.05
                servo4_adjustment = pid_output_y * 0.05
                
                # Get current positions and calculate new positions
                current_servo1 = self.joint_angles[0] if len(self.joint_angles) > 0 else 90
                current_servo4 = self.joint_angles[3] if len(self.joint_angles) > 3 else 0
                
                new_servo1 = max(60, min(120, current_servo1 + servo1_adjustment))
                new_servo4 = max(-10, min(30, current_servo4 + servo4_adjustment))
                
                # Apply movements if change is significant
                if abs(new_servo1 - current_servo1) > 0.5 or abs(new_servo4 - current_servo4) > 0.5:
                    try:
                        self.movement_controller.arm.Arm_serial_servo_write(1, int(new_servo1), 800)
                        time.sleep(0.1)
                        self.movement_controller.arm.Arm_serial_servo_write(4, int(new_servo4), 800)
                        
                        # Update tracking
                        self.movement_controller.joint_angles[0] = int(new_servo1)
                        self.movement_controller.joint_angles[3] = int(new_servo4)
                        
                    except Exception as e:
                        self.logger.log_warning(f"Servo movement failed: {e}")
            
            iteration += 1
            time.sleep(0.2)
        
        # Timeout reached
        if last_centroid:
            final_error_x = target_x - last_centroid[0]
            final_error_y = target_y - last_centroid[1]
            
            return {
                "success": False,
                "final_centroid": {"x": int(last_centroid[0]), "y": int(last_centroid[1])},
                "final_error": {"x": int(final_error_x), "y": int(final_error_y)},
                "iterations": iteration,
                "time_elapsed": current_time - start_time,
                "object_color": object_color,
                "reason": "timeout",
                "error_message": f"Failed to center within {timeout}s"
            }
        else:
            return {
                "success": False,
                "iterations": iteration,
                "time_elapsed": current_time - start_time,
                "object_color": object_color,
                "reason": "no_object_detected",
                "error_message": f"No {object_color} object detected"
            }
    
    # ========== Utility Methods ==========
    
    def _get_component_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "camera_available": self.camera_manager.is_available(),
            "movement_available": self.movement_controller.is_available(),
            "camera_queue_size": self.camera_manager.get_queue_size(),
        }
    
    def get_device_number(self) -> int:
        """Get device number."""
        return self.device_number
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        component_status = self._get_component_status()
        
        return {
            "device_number": self.device_number,
            "joint_angles": self.joint_angles.copy(),
            "is_moving": self.is_moving,
            "gripper_closed": self.gripper_closed,
            "recording": self.recording,
            **component_status,
            "components": {
                "camera_manager": "initialized",
                "vision_detector": "initialized", 
                "movement_controller": "initialized",
                "calibration_manager": "initialized"
            }
        }
    
    def cleanup(self):
        """Cleanup all components."""
        self.logger.log_info("Cleaning up robot arm interface...")
        
        try:
            self.camera_manager.cleanup()
            self.movement_controller.cleanup()
            self.logger.log_info("Robot arm interface cleanup completed")
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")
