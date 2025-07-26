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
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from functools import wraps

from components import CameraManager, VisionDetector, MovementController, CalibrationManager, ExperimentLogger


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
        
        # Experiment Logger - handles all data logging and persistence
        self.experiment_logger = ExperimentLogger(logger, data_path)
        
        # Calibration Manager - handles calibration operations
        self.calibration_manager = CalibrationManager(logger, self.camera_manager, data_path, self.experiment_logger)
        
        # Connect vision detector to camera manager for debug image coordination
        self.vision_detector.camera_manager = self.camera_manager
        
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
        """Check if trajectory recording is active (delegated to experiment logger)."""
        return self.experiment_logger.is_recording()
    
    @recording.setter
    def recording(self, value: bool):
        """Set recording state (delegated to experiment logger)."""
        # This setter is kept for backward compatibility but logs a warning
        self.logger.log_warning("Direct recording property setting is deprecated. Use start_recording()/stop_recording() methods.")
    
    # ========== Camera Interface Methods ==========
    
    def capture_rgb_image(self) -> Optional[np.ndarray]:
        """Capture RGB image using optimized synchronized method.
        
        Returns:
            RGB image as numpy array or None if capture fails
        """
        rgb_image, _, _, _ = self.camera_manager.capture_synchronized_data()
        return rgb_image
    
    def capture_synchronized_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized RGB, depth, and point cloud data.
        
        Returns:
            Tuple of (rgb_image, depth_image, depth_colormap, point_cloud_data)
        """
        return self.camera_manager.capture_synchronized_data()
    
    # ========== High-Level Operations ==========
    
    def scan_for_target(self, object_type: str, color: str) -> Dict[str, Any]:
        """Scan using action-level pipeline switching for guaranteed fresh frames."""
        
        scan_positions = self.config.scan_positions
        self.logger.log_info(f"Starting action-level scan for {color} {object_type}")
        
        start_time = time.time()
        positions_checked = []
        
        # Switch to scanning mode once for the entire action
        if not self.camera_manager.switch_to_scanning_mode({"action": "scan_for_target", "start_time": start_time}):
            return self._build_scan_result(
                success=False,
                target_found=False,
                error_message="Failed to switch to scanning mode",
                positions_checked=[],
                scan_time=0,
                object_type=object_type,
                color=color
            )
        
        try:
            for position in scan_positions:
                self.logger.log_info(f"Action-level scanning at servo1={position}°...")
                
                # STEP 0: Move robot to position
                self.movement_controller.move_single_joint(1, position, record_action=False)
                time.sleep(self.config.scan_delay)  # Wait for movement to complete
                
                try:
                    # STEP 1: Get Frame from Camera Manager (memory optimized)
                    rgb_image, depth_image, depth_colormap, _ = self.camera_manager.grab_fresh_scanning_frame(
                        timeout_ms=2000, save_for_recording=True
                    )
                    
                    if rgb_image is None:
                        self.logger.log_warning(f"No frame received at position {position} - continuing to next position")
                        continue
                    
                    # STEP 2: Pass Frame to Vision Detector (enhanced with debug images)
                    detection_result, debug_images = self.vision_detector.detect_target_object(
                        rgb_image, object_type, color
                    )
                    
                    # STEP 3: Collect Return Values and Debug Data
                    scan_data = {
                        "position": position,
                        "detection_result": detection_result,
                        "debug_images": debug_images,
                        "frame_metadata": {
                            "timestamp": time.time(),
                            "object_type": object_type,
                            "color": color,
                            "rgb_shape": rgb_image.shape
                        }
                    }
                    
                    # STEP 4: Pass to Experiment Logger (including depth image)
                    if not self.experiment_logger.log_scanning_event(
                        original_frame=rgb_image,
                        depth_frame=depth_image,
                        depth_colormap=depth_colormap,
                        scan_data=scan_data
                    ):
                        self.logger.log_warning(f"Failed to log scanning event at position {position}")
                    
                    # STEP 5: Function Conclusion - Evaluate Results and Handle Early Return
                    if detection_result and detection_result[1] > self.config.confidence_threshold:
                        scan_time = time.time() - start_time
                        return self._build_scan_result(
                            success=True,
                            target_found=True,
                            position=position,
                            detection_data={"found": True, "centroid": detection_result[0], "confidence": detection_result[1]},
                            positions_checked=positions_checked + [position],
                            scan_time=scan_time,
                            object_type=object_type,
                            color=color
                        )
                
                except Exception as e:
                    self.logger.log_error(f"Error scanning at position {position}: {e}")
                    # Continue to next position on error
                
                positions_checked.append(position)
            
            # Target not found after scanning all positions
            scan_time = time.time() - start_time
            return self._build_scan_result(
                success=False,
                target_found=False,
                positions_checked=positions_checked,
                scan_time=scan_time,
                object_type=object_type,
                color=color
            )
            
        except Exception as e:
            self.logger.log_error(f"Error during action-level scan: {e}")
            return self._build_scan_result(
                success=False,
                target_found=False,
                error_message=f"Action-level scanning failed: {str(e)}",
                positions_checked=positions_checked,
                scan_time=time.time() - start_time,
                object_type=object_type,
                color=color
            )
        
        finally:
            # Always switch back to recording mode
            if not self.camera_manager.switch_to_recording_mode():
                self.logger.log_error("Failed to switch back to recording mode")
    
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
            
            # Capture frame using optimized synchronized method
            rgb_image, _, _, _ = self.camera_manager.capture_synchronized_data()
            if rgb_image is None:
                continue
            
            # Note: capture_synchronized_data() already returns BGR format
            bgr_image = rgb_image  # Actually BGR format despite variable name
            
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
    
    # ========== Recording Methods (Interface-Level Coordination) ==========
    
    def start_recording(self, annotation: str = "", task_description: str = "") -> str:
        """Start coordinated recording across movement and camera components.
        
        Args:
            annotation: Text description of the trajectory being recorded
            task_description: Detailed description of the task
            
        Returns:
            ID of the trajectory being recorded
        """
        if self.recording:
            self.logger.log_warning("Already recording a trajectory. Stop the current recording first.")
            current_data = self.experiment_logger.get_current_trajectory_data()
            return current_data.get("trajectory_id", "") if current_data else ""
        
        # Generate unique experiment ID
        experiment_id = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start recording through experiment logger
        actual_experiment_id = self.experiment_logger.start_recording(experiment_id)
        
        # Add annotation and task description to trajectory data
        current_data = self.experiment_logger.get_current_trajectory_data()
        if current_data:
            current_data["metadata"]["annotation"] = annotation
            current_data["metadata"]["task_description"] = task_description
        
        self.logger.log_info(f"Started recording experiment {actual_experiment_id}")
        return actual_experiment_id
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Stop coordinated recording and save trajectory data.
        
        Returns:
            Dictionary with trajectory data or None if no recording was active
        """
        if not self.recording:
            self.logger.log_warning("No active recording to stop.")
            return None
        
        # Stop recording through experiment logger
        trajectory_data = self.experiment_logger.stop_recording()
        
        if trajectory_data:
            experiment_id = trajectory_data["trajectory_id"]
            self.logger.log_info(f"Stopped and saved experiment {experiment_id}")
        
        return trajectory_data
    
    def record_data_point(self, action_name: str = "", action_params: Dict = None):
        """Record coordinated data point with movement and camera data.
        
        Args:
            action_name: Name of the action being performed
            action_params: Parameters of the action being performed
        """
        if not self.recording:
            return
        
        # Delegate to experiment logger with movement controller data
        self.experiment_logger.record_data_point(
            action_name=action_name,
            action_params=action_params,
            joint_angles=self.movement_controller.joint_angles,
            gripper_closed=self.movement_controller.gripper_closed,
            is_moving=self.movement_controller.is_moving,
            camera_manager=self.camera_manager
        )
    
    # ========== Movement Methods with Recording Integration ==========
    
    def move_single_joint(self, servo_id: int, angle: int, record_action: bool = True) -> None:
        """Move single joint with optional continuous recording."""
        # Create recording callback if recording is active
        recording_callback = None
        if self.recording and record_action:
            recording_callback = lambda: self.record_data_point("move_single_joint", {"servo_id": servo_id, "angle": angle})
        
        # Call movement controller with callback
        self.movement_controller.move_single_joint(servo_id, angle, record_action=False, recording_callback=recording_callback)
    
    def move_all_joints(self, angles: List[int], record_action: bool = True) -> None:
        """Move all joints with optional continuous recording."""
        # Create recording callback if recording is active
        recording_callback = None
        if self.recording and record_action:
            recording_callback = lambda: self.record_data_point("move_all_joints", {"angles": angles})
        
        # Call movement controller with callback
        self.movement_controller.move_all_joints(angles, record_action=False, recording_callback=recording_callback)
    
    def close_gripper(self, record_action: bool = True) -> None:
        """Close gripper with optional recording."""
        # Call movement controller (without recording since we handle it here)
        self.movement_controller.close_gripper(record_action=False)
        
        # Record data point if recording is active
        if self.recording and record_action:
            self.record_data_point("close_gripper", {})
    
    def open_gripper(self, record_action: bool = True) -> None:
        """Open gripper with optional recording."""
        # Call movement controller (without recording since we handle it here)
        self.movement_controller.open_gripper(record_action=False)
        
        # Record data point if recording is active
        if self.recording and record_action:
            self.record_data_point("open_gripper", {})
    
    def pick_from_source(self, source_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True) -> None:
        """Pick from source with optional continuous recording."""
        # Create recording callback if recording is active
        recording_callback = None
        if self.recording and record_action:
            recording_callback = lambda: self.record_data_point("pick_from_source", {
                "source_angle": source_angle,
                "approach_key": approach_key,
                "return_key": return_key,
                "grab_key": grab_key
            })
        
        # Call movement controller with callback
        self.movement_controller.pick_from_source(source_angle, approach_key, return_key, grab_key, record_action=False, recording_callback=recording_callback)
    
    def place_at_target(self, target_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True) -> None:
        """Place at target with optional continuous recording."""
        # Create recording callback if recording is active
        recording_callback = None
        if self.recording and record_action:
            recording_callback = lambda: self.record_data_point("place_at_target", {
                "target_angle": target_angle,
                "approach_key": approach_key,
                "return_key": return_key,
                "grab_key": grab_key
            })
        
        # Call movement controller with callback
        self.movement_controller.place_at_target(target_angle, approach_key, return_key, grab_key, record_action=False, recording_callback=recording_callback)
    
    # ========== Cleanup ==========
    
    def cleanup(self):
        """Cleanup all components."""
        self.logger.log_info("Cleaning up robot arm interface...")
        
        try:
            self.camera_manager.cleanup()
            self.movement_controller.cleanup()
            self.logger.log_info("Robot arm interface cleanup completed")
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")
