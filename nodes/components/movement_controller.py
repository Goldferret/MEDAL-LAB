"""
Movement Controller Component for DOFBOT Pro

Handles all robot movement and servo control operations including:
- Joint movement and servo control
- Trajectory recording and playback
- Pick and place operations
- Gripper control
- Movement validation and safety
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

# Import robot arm library
try:
    from Arm_Lib import Arm_Device
    ARM_LIB_AVAILABLE = True
except ImportError:
    ARM_LIB_AVAILABLE = False
    Arm_Device = None


class MovementController:
    """Handles robot movement and servo control for DOFBOT Pro robot."""
    
    def __init__(self, logger, config):
        """Initialize movement controller.
        
        Args:
            logger: Logger instance for logging
            config: Configuration object with movement settings
        """
        self.logger = logger
        self.config = config
        
        # Initialize robot arm
        if ARM_LIB_AVAILABLE:
            self.arm = Arm_Device()
            self.logger.log_info("Robot arm initialized successfully")
        else:
            self.arm = None
            self.logger.log_error("Arm_Lib not available - robot movement disabled")
        
        # Robot state
        self.joint_angles = [90, 180, 0, 0, 90]  # Default joint positions
        self.last_angles = [90, 180, 0, 0, 90]
        self.last_timestamp = time.time()
        self.is_moving = False
        self.gripper_closed = False
        self.torque_state = False
        
        # Recording state
        self.recording = False
        self.current_trajectory_data = {
            "joint_states": [],
            "images": [],
            "timestamps": []
        }
        
        # Movement presets from config
        self.approach_method = {
            "directly_above": [40, 60, 0],  # Basic lean-on-top approach (servos 2, 3, 4)
            "angled_above": [67, 0, 45],    # lean-forward with gripper at 45 degree angle
        }
        
        self.return_method = {
            "directly_above": [180, 0, 0],  # Basic return to upright position
            "angled_above": [165, 0, 62],   # lean-back with arm parallel to table
        }
        
        self.grabbing_method = {
            "horizontal": 90,        # Pinch from left-to-right of item (x axis)
            "vertical": 0,           # Pinch from top-to-bottom of item (y axis)
            "directly_above": 0,     # Gripper angle for direct approach
            "angled_above": 45,      # Gripper angle for angled approach
        }
    
    @contextmanager
    def record_action(self, action_name: str, action_params: Dict = None, record_action: bool = True):
        """Context manager for automatic recording start/end."""
        # Start recording if enabled
        if record_action and self.recording:
            self._record_data_point(action_name=action_name, action_params=action_params or {})
        
        try:
            yield  # Execute the action
        finally:
            # End recording if enabled
            if record_action and self.recording:
                self._record_data_point()
    
    def _execute_servo_movement(self, angles: List[int], duration_ms: int = 2000, movement_time_s: float = 2.0):
        """Execute servo movement with gripper angle and recording support.
        
        Args:
            angles: Joint angles (without gripper)
            duration_ms: Movement duration in milliseconds
            movement_time_s: Time to wait/record during movement
        """
        # Add gripper angle automatically
        angles_with_gripper = angles.copy()
        angles_with_gripper.append(145 if self.gripper_closed else 90)
        
        # Execute movement
        self.arm.Arm_serial_servo_write6_array(angles_with_gripper, duration_ms)
        
        # Handle recording during movement
        if self.recording:
            start_time = time.time()
            interval = 1.0 / 10.0  # 10Hz recording
            while time.time() - start_time < movement_time_s:
                self._record_data_point()
                time.sleep(interval)
        else:
            time.sleep(movement_time_s)
    
    def _move_gripper(self, closed: bool, record_action: bool = True):
        """Unified gripper movement method.
        
        Args:
            closed: True to close gripper, False to open
            record_action: Whether to record this action
        """
        action_name = "close_gripper" if closed else "open_gripper"
        angle = 145 if closed else 90
        
        with self.record_action(action_name, {}, record_action):
            self.arm.Arm_serial_servo_write(6, angle, 1000)
            time.sleep(1)
            self.gripper_closed = closed
    
    # ========== Core Movement Methods ==========
    
    def get_device_number(self) -> int:
        """Get the device number for the robot arm."""
        return 0
    
    def change_torque_state(self) -> None:
        """Toggle the robot's torque state."""
        if not self.arm:
            self.logger.log_warning("Cannot change torque - arm not initialized")
            return
        
        try:
            self.torque_state = not self.torque_state
            self.arm.Arm_serial_set_torque(1 if self.torque_state else 0)
            self.logger.log_debug(f"Torque {'enabled' if self.torque_state else 'disabled'}")
        except Exception as e:
            self.logger.log_error(f"Error changing torque state: {e}")
    
    def move_single_joint(self, servo_id: int, angle: int, record_action: bool = True) -> None:
        """Move a single joint to the specified angle."""
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        if servo_id < 1 or servo_id > 6:
            raise ValueError("Invalid servo id - must be 1-6")
        
        with self.record_action("move_single_joint", {"servo_id": servo_id, "angle": angle}, record_action):
            # Update joint angles array and execute movement
            angles = self.joint_angles.copy()
            angles[servo_id - 1] = angle
            
            self._execute_servo_movement(angles)
            
            # Update joint state
            self.joint_angles[servo_id - 1] = angle
    
    def move_all_joints(self, angles: List[int], record_action: bool = True) -> None:
        """Move all joints to the specified angles simultaneously."""
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        if len(angles) != 5:
            raise ValueError("Expected 5 joint angles")
        
        with self.record_action("move_all_joints", {"angles": angles}, record_action):
            self._execute_servo_movement(angles)
            
            # Update joint state
            self.joint_angles = angles.copy()
    
    def close_gripper(self, record_action: bool = True) -> None:
        """Close the robot gripper."""
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        self._move_gripper(True, record_action)
    
    def open_gripper(self, record_action: bool = True) -> None:
        """Open the robot gripper."""
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        self._move_gripper(False, record_action)
    
    # ========== High-Level Operations ==========
    
    def pick_from_source(self, source_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True) -> None:
        """Pick an item from the specified source location."""
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        with self.record_action("pick_from_source", {
            "source_angle": source_angle,
            "approach_key": approach_key,
            "return_key": return_key,
            "grab_key": grab_key
        }, record_action):
            
            # Build movement sequences
            pick_method = [source_angle] + self.approach_method[approach_key] + [self.grabbing_method[grab_key]]
            return_method = [source_angle] + self.return_method[return_key] + [self.grabbing_method[grab_key]]
            
            # Execute pick sequence
            self.move_all_joints([90, 180, 0, 0, 90], record_action=False)  # Move to grabber position
            self.move_single_joint(1, source_angle, record_action=False)     # Rotate to face source location
            self.move_all_joints(pick_method, record_action=False)           # Lean forward to grab
            self.close_gripper(record_action=False)                          # Close gripper
            self.move_all_joints(return_method, record_action=False)         # Move back up
    
    def place_at_target(self, target_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True) -> None:
        """Place an item at the specified target location."""
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        with self.record_action("place_at_target", {
            "target_angle": target_angle,
            "approach_key": approach_key,
            "return_key": return_key,
            "grab_key": grab_key
        }, record_action):
            
            # Build movement sequences
            place_method = [target_angle] + self.approach_method[approach_key] + [self.grabbing_method[grab_key]]
            return_method = [target_angle] + self.return_method[return_key] + [self.grabbing_method[grab_key]]
            
            # Execute place sequence
            self.move_single_joint(1, target_angle, record_action=False)     # Rotate to face target location
            self.move_all_joints(place_method, record_action=False)          # Lean forward to place
            self.open_gripper(record_action=False)                           # Open gripper
            self.move_all_joints(return_method, record_action=False)         # Move back up
            self.move_all_joints([90, 180, 0, 0, 90], record_action=False)  # Move back to grabber position
    
    # ========== State Management ==========
    
    def update_joint_velocities(self):
        """Update joint velocity calculations based on current angles."""
        try:
            current_time = time.time()
            dt = current_time - self.last_timestamp
            
            if dt > 0 and len(self.joint_angles) == len(self.last_angles):
                # Calculate velocities (degrees per second)
                velocities = []
                for current, last in zip(self.joint_angles, self.last_angles):
                    if current is not None and last is not None:
                        velocity = (current - last) / dt
                        velocities.append(velocity)
                    else:
                        velocities.append(0.0)
                
                # Update last_angles with valid values only
                self.last_angles = [angle if angle is not None else last 
                                  for angle, last in zip(self.joint_angles, self.last_angles)]
            
            self.last_timestamp = current_time
            
        except Exception as e:
            self.logger.log_error(f"Error updating joint velocities: {e}")
    
    def get_all_angles(self) -> List[Optional[int]]:
        """Get all current joint angles."""
        if not self.arm:
            return self.joint_angles
        
        try:
            current_angles = []
            for i in range(1, 6):  # Servos 1-5
                angle = self.arm.Arm_serial_servo_read(i)
                current_angles.append(angle)
            
            # Update stored angles with valid readings
            for i, angle in enumerate(current_angles):
                if angle is not None:
                    self.joint_angles[i] = angle
            
            return current_angles
            
        except Exception as e:
            self.logger.log_error(f"Error reading joint angles: {e}")
            return self.joint_angles
    
    # ========== Recording Methods ==========
    
    def start_recording(self, annotation: str = "", task_description: str = ""):
        """Start recording trajectory data."""
        self.recording = True
        self.current_trajectory_data = {
            "joint_states": [],
            "images": [],
            "timestamps": [],
            "annotation": annotation,
            "task_description": task_description,
            "start_time": time.time()
        }
        self.logger.log_info("Started trajectory recording")
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return trajectory data."""
        self.recording = False
        self.current_trajectory_data["end_time"] = time.time()
        self.current_trajectory_data["duration"] = (
            self.current_trajectory_data["end_time"] - 
            self.current_trajectory_data["start_time"]
        )
        
        self.logger.log_info(f"Stopped trajectory recording - {len(self.current_trajectory_data['joint_states'])} data points")
        return self.current_trajectory_data.copy()
    
    def _record_data_point(self, action_name: str = "", action_params: Dict = None):
        """Record a single data point during trajectory recording."""
        if not self.recording:
            return
        
        try:
            # Record joint state
            joint_state = {
                "timestamp": time.time(),
                "joint_angles": self.joint_angles.copy(),
                "gripper_closed": self.gripper_closed,
                "is_moving": self.is_moving,
                "action_name": action_name,
                "action_params": action_params or {}
            }
            
            # Add to trajectory data
            self.current_trajectory_data["joint_states"].append(joint_state)
            self.current_trajectory_data["timestamps"].append(time.time())
            
        except Exception as e:
            self.logger.log_error(f"Error recording data point: {e}")
    
    # ========== Status and Utility ==========
    
    def is_available(self) -> bool:
        """Check if movement controller is available."""
        return self.arm is not None and ARM_LIB_AVAILABLE
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current robot state."""
        return {
            "joint_angles": self.joint_angles.copy(),
            "is_moving": self.is_moving,
            "gripper_closed": self.gripper_closed,
            "torque_state": self.torque_state,
            "recording": self.recording,
            "available": self.is_available()
        }
    
    def emergency_stop(self):
        """Emergency stop - disable torque and stop all movement."""
        try:
            if self.arm and self.torque_state:
                self.arm.Arm_serial_set_torque(0)
                self.torque_state = False
            
            self.is_moving = False
            self.logger.log_warning("Emergency stop activated")
            
        except Exception as e:
            self.logger.log_error(f"Error during emergency stop: {e}")
    
    def cleanup(self):
        """Cleanup movement controller resources."""
        try:
            # Stop any recording
            if self.recording:
                self.stop_recording()
            
            # Disable torque
            if self.arm and self.torque_state:
                self.arm.Arm_serial_set_torque(0)
                self.torque_state = False
            
            self.logger.log_info("Movement controller cleanup completed")
            
        except Exception as e:
            self.logger.log_error(f"Error during movement controller cleanup: {e}")
