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
    
    def _execute_servo_movement(self, angles: List[int], duration_ms: int = 2000, movement_time_s: float = 2.0, recording_callback=None):
        """Execute servo movement with optional continuous recording via callback.
        
        Args:
            angles: Joint angles (without gripper)
            duration_ms: Movement duration in milliseconds
            movement_time_s: Time to wait/record during movement
            recording_callback: Optional function to call continuously during movement for recording
        """
        # Add gripper angle automatically
        angles_with_gripper = angles.copy()
        angles_with_gripper.append(145 if self.gripper_closed else 90)
        
        # Execute movement
        self.arm.Arm_serial_servo_write6_array(angles_with_gripper, duration_ms)
        
        # Handle continuous recording via callback if provided
        if recording_callback:
            start_time = time.time()
            interval = 1.0 / 10.0  # 10Hz recording
            while time.time() - start_time < movement_time_s:
                try:
                    recording_callback()  # Interface provides the recording function
                except Exception as e:
                    # Don't let recording errors break movement - just log and continue
                    pass
                time.sleep(interval)
        else:
            # Just wait for movement to complete
            time.sleep(movement_time_s)
    
    def _move_gripper(self, closed: bool, record_action: bool = True):
        """Unified gripper movement method.
        
        Args:
            closed: True to close gripper, False to open
            record_action: Whether to record this action
        """
        angle = 145 if closed else 90
        
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
    
    def move_single_joint(self, servo_id: int, angle: int, record_action: bool = True, recording_callback=None) -> None:
        """Move a single joint to the specified angle.
        
        Args:
            servo_id: Servo ID (1-6)
            angle: Target angle
            record_action: Whether to record this action (for backward compatibility)
            recording_callback: Optional callback function for continuous recording during movement
        """
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        if servo_id < 1 or servo_id > 6:
            raise ValueError("Invalid servo id - must be 1-6")
        
        # Update joint angles array and execute movement
        angles = self.joint_angles.copy()
        angles[servo_id - 1] = angle
        
        self._execute_servo_movement(angles, recording_callback=recording_callback)
        
        # Update joint state
        self.joint_angles[servo_id - 1] = angle
    
    def move_all_joints(self, angles: List[int], record_action: bool = True, recording_callback=None) -> None:
        """Move all joints to the specified angles simultaneously.
        
        Args:
            angles: List of 5 joint angles
            record_action: Whether to record this action (for backward compatibility)
            recording_callback: Optional callback function for continuous recording during movement
        """
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        if len(angles) != 5:
            raise ValueError("Expected 5 joint angles")
        
        self._execute_servo_movement(angles, recording_callback=recording_callback)
        
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
    
    def pick_from_source(self, source_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True, recording_callback=None) -> None:
        """Pick an item from the specified source location.
        
        Args:
            source_angle: Angle to rotate to for source location
            approach_key: Key for approach method
            return_key: Key for return method  
            grab_key: Key for grabbing method
            record_action: Whether to record this action (for backward compatibility)
            recording_callback: Optional callback function for continuous recording during movements
        """
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        # Build movement sequences
        pick_method = [source_angle] + self.approach_method[approach_key] + [self.grabbing_method[grab_key]]
        return_method = [source_angle] + self.return_method[return_key] + [self.grabbing_method[grab_key]]
        
        # Execute pick sequence - pass callback to movements that need continuous recording
        self.move_all_joints([90, 180, 0, 0, 90], record_action=False, recording_callback=recording_callback)  # Move to grabber position
        self.move_single_joint(1, source_angle, record_action=False, recording_callback=recording_callback)     # Rotate to face source location
        self.move_all_joints(pick_method, record_action=False, recording_callback=recording_callback)           # Lean forward to grab
        self.close_gripper(record_action=False)                          # Close gripper (no continuous recording needed)
        self.move_all_joints(return_method, record_action=False, recording_callback=recording_callback)         # Move back up
    
    def place_at_target(self, target_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True, recording_callback=None) -> None:
        """Place an item at the specified target location.
        
        Args:
            target_angle: Angle to rotate to for target location
            approach_key: Key for approach method
            return_key: Key for return method
            grab_key: Key for grabbing method  
            record_action: Whether to record this action (for backward compatibility)
            recording_callback: Optional callback function for continuous recording during movements
        """
        if not self.arm:
            raise RuntimeError("Robot arm not initialized")
        
        # Build movement sequences
        place_method = [target_angle] + self.approach_method[approach_key] + [self.grabbing_method[grab_key]]
        return_method = [target_angle] + self.return_method[return_key] + [self.grabbing_method[grab_key]]
        
        # Execute place sequence - pass callback to movements that need continuous recording
        self.move_single_joint(1, target_angle, record_action=False, recording_callback=recording_callback)     # Rotate to face target location
        self.move_all_joints(place_method, record_action=False, recording_callback=recording_callback)          # Lean forward to place
        self.open_gripper(record_action=False)                           # Open gripper (no continuous recording needed)
        self.move_all_joints(return_method, record_action=False, recording_callback=recording_callback)         # Move back up
        self.move_all_joints([90, 180, 0, 0, 90], record_action=False, recording_callback=recording_callback)  # Move back to grabber position
    
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
            "available": self.is_available()
        }
    
    def cleanup(self):
        """Cleanup movement controller resources."""
        try:
            # Recording is now handled at interface level, no cleanup needed here
            
            # Disable torque
            if self.arm and self.torque_state:
                self.arm.Arm_serial_set_torque(0)
                self.torque_state = False
            
            self.logger.log_info("Movement controller cleanup completed")
            
        except Exception as e:
            self.logger.log_error(f"Error during movement controller cleanup: {e}")
