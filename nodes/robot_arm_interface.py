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
        
        # Recording state (interface-level coordination)
        self._recording = False
        self.current_trajectory_data = {}
        
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
        """Check if trajectory recording is active at interface level."""
        return getattr(self, '_recording', False)
    
    @recording.setter
    def recording(self, value: bool):
        """Set recording state at interface level."""
        self._recording = value
    
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
                
                # Wait for robot to settle
                time.sleep(scan_delay)
                
                # CRITICAL: Flush old frames from queue to ensure fresh frames from current position
                self.logger.log_info(f"Flushing frame queue to get fresh frames from position {position}°")
                flushed_count = 0
                while not self.camera_manager._frameset_queue.empty() and flushed_count < 35:
                    try:
                        self.camera_manager._frameset_queue.get_nowait()
                        flushed_count += 1
                    except:
                        break
                self.logger.log_info(f"Flushed {flushed_count} old frames from queue")
                
                # Wait for adequate frames to populate queue from current position
                # At 10Hz, need ~1.5s to get 15+ frames for reliable capture
                self.logger.log_info("Waiting for queue to refill with fresh frames...")
                time.sleep(1.5)  # Allow 15+ fresh frames to accumulate
                
                queue_size = self.camera_manager._frameset_queue.qsize()
                self.logger.log_info(f"Queue refilled: {queue_size} frames available")
                
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
                # Capture frame using optimized synchronized method (same as recording workflow)
                rgb_image, depth_image, depth_colormap, point_cloud = self.camera_manager.capture_synchronized_data()
                if rgb_image is None:
                    self.logger.log_warning(f"Failed to capture frame (attempt {attempt + 1})")
                    continue
                
                # Note: capture_synchronized_data() already returns BGR format, no conversion needed
                bgr_image = rgb_image  # Actually BGR format despite variable name
                
                self.logger.log_info(f"Successfully captured frame at position {position}°, attempt {attempt + 1}")
                
                # Save debug frame and color mask using camera manager
                debug_prefix = f"scan_debug_pos{position}_attempt{attempt + 1}"
                self.camera_manager.save_debug_frame(bgr_image, debug_prefix, self.data_path)
                
                mask_prefix = f"color_mask_{color}_pos{position}_attempt{attempt + 1}"
                self.camera_manager.save_color_mask_debug(bgr_image, color, mask_prefix, 
                                                        self.vision_detector, self.data_path)
                
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
            return self.current_trajectory_data.get("trajectory_id", "")
        
        # Generate unique experiment ID
        experiment_id = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = Path(self.data_path) / experiment_id
        
        # Create directory structure
        experiment_dir.mkdir(exist_ok=True)
        (experiment_dir / "rgb_images").mkdir(exist_ok=True)
        (experiment_dir / "depth_images").mkdir(exist_ok=True)
        (experiment_dir / "point_clouds").mkdir(exist_ok=True)
        
        # Initialize recording state
        self.current_trajectory_data = {
            "trajectory_id": experiment_id,
            "annotation": annotation,
            "task_description": task_description,
            "start_time": time.time(),
            "joint_states": [],
            "actions": [],
            "images": []
        }
        
        self.recording = True
        self.logger.log_info(f"Started recording experiment {experiment_id}")
        return experiment_id
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Stop coordinated recording and save trajectory data.
        
        Returns:
            Dictionary with trajectory data or None if no recording was active
        """
        if not self.recording:
            self.logger.log_warning("No active recording to stop.")
            return None
        
        experiment_id = self.current_trajectory_data["trajectory_id"]
        experiment_dir = Path(self.data_path) / experiment_id
        
        # Add end time and duration
        self.current_trajectory_data["end_time"] = time.time()
        self.current_trajectory_data["duration"] = (
            self.current_trajectory_data["end_time"] - self.current_trajectory_data["start_time"]
        )
        
        # Save trajectory data
        with open(experiment_dir / "trajectory_data.json", "w") as f:
            json.dump(self.current_trajectory_data, f, indent=2)
        
        self.recording = False
        self.logger.log_info(f"Stopped and saved experiment {experiment_id}")
        
        # Return copy of trajectory data for the caller
        return self.current_trajectory_data.copy()
    
    def record_data_point(self, action_name: str = "", action_params: Dict = None):
        """Record coordinated data point with movement and camera data.
        
        Args:
            action_name: Name of the action being performed
            action_params: Parameters of the action being performed
        """
        if not self.recording:
            return
        
        try:
            timestamp = time.time()
            experiment_dir = Path(self.data_path) / self.current_trajectory_data["trajectory_id"]
            frame_id = f"frame_{len(self.current_trajectory_data['joint_states']):06d}"
            
            # Get movement data from movement controller
            joint_state = {
                "timestamp": timestamp,
                "joint_angles": self.movement_controller.joint_angles.copy(),
                "gripper_closed": self.movement_controller.gripper_closed,
                "is_moving": self.movement_controller.is_moving,
                "action_name": action_name,
                "action_params": action_params or {}
            }
            
            # Capture images from camera manager
            image_data = {"frame_id": frame_id, "timestamp": timestamp}
            
            try:
                # Get synchronized camera data
                rgb_image, depth_image, depth_colormap, point_cloud = self.camera_manager.capture_synchronized_data()
                
                # Save RGB image
                if rgb_image is not None:
                    rgb_path = experiment_dir / "rgb_images" / f"{frame_id}.jpg"
                    cv2.imwrite(str(rgb_path), rgb_image)
                    image_data["rgb_path"] = f"rgb_images/{frame_id}.jpg"
                
                # Save depth data
                if depth_image is not None:
                    depth_path = experiment_dir / "depth_images" / f"{frame_id}_depth.png"
                    cv2.imwrite(str(depth_path), depth_image)
                    image_data["depth_path"] = f"depth_images/{frame_id}_depth.png"
                    
                    # Save colorized depth map
                    if depth_colormap is not None:
                        colormap_path = experiment_dir / "depth_images" / f"{frame_id}_depth_colormap.jpg"
                        cv2.imwrite(str(colormap_path), depth_colormap)
                        image_data["depth_colormap_path"] = f"depth_images/{frame_id}_depth_colormap.jpg"
                
                # Save point cloud
                if point_cloud is not None:
                    pc_path = experiment_dir / "point_clouds" / f"{frame_id}_pointcloud.npy"
                    np.save(str(pc_path), point_cloud)
                    image_data["point_cloud_path"] = f"point_clouds/{frame_id}_pointcloud.npy"
                    image_data["point_cloud_size"] = len(point_cloud)
                    
            except Exception as e:
                self.logger.log_error(f"Error capturing images during recording: {e}")
            
            # Store coordinated data
            self.current_trajectory_data["joint_states"].append(joint_state)
            self.current_trajectory_data["images"].append(image_data)
            
        except Exception as e:
            self.logger.log_error(f"Error recording data point: {e}")
    
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
