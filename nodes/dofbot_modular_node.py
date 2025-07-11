#!/usr/bin/env python3
"""
DOFBOT Pro Modular Node

A modular, component-based implementation of the DOFBOT Pro robot node using MADSci framework.
This version uses a clean component architecture for better maintainability and reliability.

Components:
- CameraManager: Camera operations and frame capture
- VisionDetector: Computer vision and object detection  
- MovementController: Robot movement and servo control
- CalibrationManager: Camera and system calibration
- RobotArmInterface: High-level coordination

Author: MADSci Framework
"""

import sys
from pathlib import Path
from typing import List, Any
from functools import wraps

# Add the current directory to Python path for component imports
sys.path.insert(0, str(Path(__file__).parent))

from madsci.common.types.action_types import ActionFailed, ActionResult, ActionSucceeded
from madsci.node_module.helpers import action
from madsci.node_module.rest_node_module import RestNode

from robot_arm_config import RobotArmConfig
from robot_arm_interface import RobotArmInterface


# Decorators for common validation
def require_robot_interface(func):
    """Decorator to ensure robot interface is initialized."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.robot_interface is None:
            return ActionFailed(errors="Robot interface not initialized")
        return func(self, *args, **kwargs)
    return wrapper


def require_not_moving(func):
    """Decorator to ensure robot is not currently moving."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.robot_interface and self.robot_interface.is_moving:
            return ActionFailed(errors="Robot is already moving")
        return func(self, *args, **kwargs)
    return wrapper


def require_robot_ready(func):
    """Decorator to ensure robot is ready and manage movement state at node level."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check robot interface exists
        if self.robot_interface is None:
            return ActionFailed(errors="Robot interface not initialized")
        
        # Check movement controller exists and is available
        if not hasattr(self.robot_interface, 'movement_controller') or not self.robot_interface.movement_controller:
            return ActionFailed(errors="Robot movement controller not found")
        
        # Check if movement controller is available (has arm hardware)
        if not self.robot_interface.movement_controller.is_available():
            return ActionFailed(errors="Robot movement controller not available")
        
        # Check if robot is already moving (like old expert node)
        if getattr(self.robot_interface.movement_controller, 'is_moving', False):
            return ActionFailed(errors="Robot is already moving")
        
        # Set moving state at movement controller level (like old expert node)
        self.robot_interface.movement_controller.is_moving = True
        try:
            return func(self, *args, **kwargs)
        finally:
            # Always reset moving state, even if exception occurs
            self.robot_interface.movement_controller.is_moving = False
    return wrapper


class DofbotModularNode(RestNode):
    """Modular DOFBOT Pro Robot Node with component-based architecture."""
    
    robot_interface: RobotArmInterface = None
    config_model = RobotArmConfig
    
    def startup_handler(self) -> None:
        """Handle the startup of the node."""
        # Get config values with defaults if attributes are missing
        device_number = getattr(self.config, 'device_number', 0)
        data_path = getattr(self.config, 'data_collection_path', "./captures")
        
        # Initialize robot interface with all components
        try:
            self.robot_interface = RobotArmInterface(
                logger=self.logger,
                config=self.config,
                device_number=device_number,
                data_path=data_path
            )
            self.logger.log_info("DOFBOT Pro modular node initialized successfully")
        except Exception as e:
            self.logger.log_error(f"Failed to initialize robot interface: {e}")
    
    def state_handler(self) -> None:
        """Update the node's public-facing state information.
        
        This method updates self.node_state which is returned by the /state endpoint.
        """
        if not self.robot_interface:
            self.node_state = {
                "status": "initializing",
                "device_ready": False,
                "error": "Robot interface not initialized"
            }
            return
        
        try:
            self.node_state = {
                # Core robot state
                "joint_angles": self.robot_interface.joint_angles,
                "is_moving": self.robot_interface.is_moving,
                "gripper_closed": self.robot_interface.gripper_closed,
                
                # System availability
                "camera_active": self.robot_interface.camera_manager.is_available(),
                "device_ready": self.robot_interface.movement_controller.is_available(),
                
                # Recording state
                "recording_active": self.robot_interface.recording,
                
                # Queue status for monitoring
                "camera_queue_size": self.robot_interface.camera_manager.get_queue_size(),
                
                # Overall status
                "status": "ready" if self.robot_interface.movement_controller.is_available() else "error",
                "node_type": "dofbot_pro_robot_arm"
            }
            
        except Exception as e:
            self.logger.log_error(f"Error updating robot state: {e}")
            self.node_state = {
                "status": "error",
                "device_ready": False,
                "error": str(e),
                "node_type": "dofbot_pro_robot_arm"
            }
    
    # Validation helpers
    def validate_color(self, color: str) -> ActionResult:
        """Validate color parameter."""
        supported_colors = ["red", "green", "blue", "yellow"]
        if color.lower() not in supported_colors:
            return ActionFailed(errors=f"Unsupported color: {color}. Supported: {supported_colors}")
        return None
    
    def validate_object_type(self, object_type: str) -> ActionResult:
        """Validate object type parameter."""
        supported_objects = ["cube", "rectangular_prism"]
        if object_type.lower() not in supported_objects:
            return ActionFailed(errors=f"Unsupported object type: {object_type}. Supported: {supported_objects}")
        return None
    
    # ========== Movement Actions ==========
    
    @action(name="move_joint", description="Move the robot's joint by the specified joint angle")
    @require_robot_ready
    def move_joints(self, servo_id: int, joint_angle: int) -> ActionResult:
        """Move a single joint to the specified angle."""
        try:
            self.robot_interface.change_torque_state()  # Turn torque on
            self.robot_interface.move_single_joint(servo_id, joint_angle)
            self.robot_interface.change_torque_state()  # Turn torque off
            
            self.logger.log_info(f"Moved servo {servo_id} to {joint_angle} degrees")
            return ActionSucceeded()
            
        except Exception as e:
            self.logger.log_error(f"Error moving joint: {e}")
            return ActionFailed(errors=str(e))
    
    @action(name="move_all_joints", description="Move all the robot's joints by the specified angles")
    @require_robot_ready
    def move_all_joints(self, joint_angles: List[int]) -> ActionResult:
        """Move all joints to the specified angles simultaneously."""
        if len(joint_angles) != 5:
            return ActionFailed(errors="Invalid number of joint angles. Expected 5.")
        
        try:
            self.robot_interface.change_torque_state()  # Turn torque on
            self.robot_interface.move_all_joints(joint_angles)
            self.robot_interface.change_torque_state()  # Turn torque off
            
            self.logger.log_info(f"Moved robot to joint angles: {joint_angles}")
            return ActionSucceeded()
            
        except Exception as e:
            self.logger.log_error(f"Error moving all joints: {e}")
            return ActionFailed(errors=str(e))
    
    @action(name="grabber_position", description="Move the robot to the grabber position")
    @require_robot_ready
    def grabber_position(self) -> ActionResult:
        """Move the robot to the standard grabber position."""
        try:
            self.robot_interface.change_torque_state()  # Turn torque on
            self.robot_interface.move_all_joints([90, 180, 0, 0, 90])
            self.robot_interface.change_torque_state()  # Turn torque off
            
            self.logger.log_info("Moved robot to grabber position")
            return ActionSucceeded()
            
        except Exception as e:
            self.logger.log_error(f"Error moving to grabber position: {e}")
            return ActionFailed(errors=str(e))
    
    @action(name="transfer", description="Transfer item from source to target location")
    @require_robot_ready
    def transfer(self, locations: List[int], movement_keys: List[str]) -> ActionResult:
        """Transfer an item from source to target location."""
        if len(locations) != 2:
            return ActionFailed(errors="Invalid number of location arguments. Expected 2.")
        
        if len(movement_keys) != 3:
            return ActionFailed(errors="Invalid number of movement keys. Expected 3.")
        
        try:
            self.robot_interface.change_torque_state()  # Turn torque on
            
            # Perform pick and place sequence
            self.robot_interface.pick_from_source(locations[0], *movement_keys)
            self.robot_interface.place_at_target(locations[1], *movement_keys)
            
            self.robot_interface.change_torque_state()  # Turn torque off
            
            self.logger.log_info(f"Transferred item from {locations[0]}° to {locations[1]}°")
            return ActionSucceeded()
            
        except Exception as e:
            self.logger.log_error(f"Error during transfer: {e}")
            return ActionFailed(errors=str(e))
    
    # ========== Vision and Detection Actions ==========
    
    @action(name="scan_for_target", description="Scan working area to find and face specified colored object")
    @require_robot_ready
    def scan_for_target(self, 
                       object_type: str = "cube",
                       color: str = "red") -> ActionResult:
        """Scan the working area to find and face a specified colored object."""
        # Validate parameters using helpers
        color_validation = self.validate_color(color)
        if color_validation:
            return color_validation
        
        object_validation = self.validate_object_type(object_type)
        if object_validation:
            return object_validation
        
        try:
            self.logger.log_info(f"Starting scan for {color} {object_type}")
            
            # Turn on torque for servo movements
            self.robot_interface.change_torque_state()
            
            # Move to starting position (leftmost scanning position)
            self.logger.log_info("Moving to starting scan position...")
            self.robot_interface.move_all_joints(self.config.starting_scan_position, record_action=False)
            
            # Perform the scanning
            result = self.robot_interface.scan_working_area(object_type.lower(), color.lower())
            
            # Turn off torque after scanning
            self.robot_interface.change_torque_state()
            
            if result["success"]:
                self.logger.log_info(f"Successfully found {color} {object_type} at servo1={result['found_at_servo1_position']}°")
                return ActionSucceeded(data=result)
            else:
                self.logger.log_info(f"No {color} {object_type} found during scan")
                return ActionFailed(errors=f"Target not found: {color} {object_type}", data=result)
                
        except Exception as e:
            # Make sure to turn off torque even if there's an exception
            try:
                self.robot_interface.change_torque_state()
            except:
                pass
            self.logger.log_error(f"Error during scan_for_target: {e}")
            return ActionFailed(errors=f"Scanning failed due to error: {str(e)}")
    
    @action(name="center_on_target", description="Center camera on colored object using PID control")
    @require_robot_ready
    def center_on_target(self, 
                        object_color: str = "red",
                        timeout: float = 30.0) -> ActionResult:
        """Center the camera on a colored object using PID control."""
        # Validate parameters using helper
        color_validation = self.validate_color(object_color)
        if color_validation:
            return color_validation
        
        if timeout <= 0 or timeout > 300:  # Max 5 minutes
            return ActionFailed(errors="Timeout must be between 0 and 300 seconds")
        
        try:
            self.logger.log_info(f"Starting center_on_target for {object_color} object (timeout: {timeout}s)")
            
            # Turn on torque for servo movements
            self.robot_interface.change_torque_state()
            
            # Call robot interface method to perform the centering
            result = self.robot_interface.center_camera_on_object(object_color.lower(), timeout)
            
            # Turn off torque after centering
            self.robot_interface.change_torque_state()
            
            if result["success"]:
                self.logger.log_info(f"Successfully centered on {object_color} object")
                return ActionSucceeded(data=result)
            else:
                self.logger.log_warning(f"Failed to center on {object_color} object: {result.get('reason', 'Unknown')}")
                return ActionFailed(errors=result.get("error_message", "Centering failed"), data=result)
                
        except Exception as e:
            # Make sure to turn off torque even if there's an exception
            try:
                self.robot_interface.change_torque_state()
            except:
                pass
            self.logger.log_error(f"Error during center_on_target: {e}")
            return ActionFailed(errors=f"Centering failed due to error: {str(e)}")
    
    # ========== Camera and Capture Actions ==========
    
    @action(name="capture_single_image", description="Capture a single image from the camera")
    @require_robot_interface
    def capture_single_image(self) -> ActionResult:
        """Capture a single RGB image from the camera."""
        try:
            self.logger.log_info("Capturing image from camera")
            rgb_image = self.robot_interface.capture_rgb_image()
            
            if rgb_image is None:
                return ActionFailed(errors="Failed to capture image - no frames available")
            
            # Save image with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = Path(self.config.data_collection_path) / filename
            filepath.parent.mkdir(exist_ok=True)
            
            import cv2
            cv2.imwrite(str(filepath), rgb_image)
            
            self.logger.log_info(f"Image captured and saved: {filename}")
            return ActionSucceeded(data={
                "filename": filename,
                "filepath": str(filepath),
                "image_shape": rgb_image.shape,
                "timestamp": timestamp
            })
            
        except Exception as e:
            self.logger.log_error(f"Error capturing image: {e}")
            return ActionFailed(errors=str(e))
    
    @action(name="test_camera_capture", description="Test camera capture functionality")
    @require_robot_interface
    def test_camera_capture(self) -> ActionResult:
        """Test the camera capture functionality and return detailed status."""
        try:
            # Test synchronized capture
            rgb_image, depth_image, depth_colormap, point_cloud_data = self.robot_interface.capture_synchronized_data()
            
            result = {
                "rgb_available": rgb_image is not None,
                "depth_available": depth_image is not None,
                "depth_colormap_available": depth_colormap is not None,
                "point_cloud_available": point_cloud_data is not None,
                "camera_status": self.robot_interface.camera_manager.is_available(),
                "queue_size": self.robot_interface.camera_manager.get_queue_size()
            }
            
            if rgb_image is not None:
                result["rgb_shape"] = rgb_image.shape
            if depth_image is not None:
                result["depth_shape"] = depth_image.shape
            if point_cloud_data is not None:
                result["point_cloud_points"] = len(point_cloud_data)
            
            self.logger.log_info("Camera test completed successfully")
            return ActionSucceeded(data=result)
            
        except Exception as e:
            self.logger.log_error(f"Camera test failed: {e}")
            return ActionFailed(errors=str(e))
    
    # ========== Recording Actions ==========
    
    @action(name="start_recording", description="Begin recording expert trajectory data")
    @require_robot_interface
    def start_recording(self, annotation: str = "", task_description: str = "") -> ActionResult:
        """Begin recording expert trajectory data with camera integration."""
        try:
            self.robot_interface.start_recording(annotation, task_description)
            
            self.logger.log_info(f"Started recording trajectory data")
            if annotation:
                self.logger.log_info(f"Annotation: {annotation}")
            if task_description:
                self.logger.log_info(f"Task: {task_description}")
            
            return ActionSucceeded(data={
                "recording": True,
                "annotation": annotation,
                "task_description": task_description
            })
            
        except Exception as e:
            self.logger.log_error(f"Error starting recording: {e}")
            return ActionFailed(errors=str(e))
    
    @action(name="stop_recording", description="Stop recording and save trajectory data")
    @require_robot_interface
    def stop_recording(self) -> ActionResult:
        """Stop recording and save trajectory data."""
        if not self.robot_interface.recording:
            return ActionFailed(errors="No recording in progress")
        
        try:
            trajectory_data = self.robot_interface.stop_recording()
            
            # Save trajectory data
            import json
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"
            filepath = Path(self.config.data_collection_path) / filename
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(trajectory_data, f, indent=2, default=str)
            
            self.logger.log_info(f"Recording stopped and saved: {filename}")
            self.logger.log_info(f"Recorded {len(trajectory_data['joint_states'])} data points")
            
            return ActionSucceeded(data={
                "recording": False,
                "filename": filename,
                "filepath": str(filepath),
                "data_points": len(trajectory_data['joint_states']),
                "duration": trajectory_data.get('duration', 0),
                "annotation": trajectory_data.get('annotation', ''),
                "task_description": trajectory_data.get('task_description', '')
            })
            
        except Exception as e:
            self.logger.log_error(f"Error stopping recording: {e}")
            return ActionFailed(errors=str(e))
    
    # ========== Status and Information Actions ==========
    
    @action(name="get_robot_status", description="Get comprehensive robot status")
    @require_robot_interface
    def get_robot_status(self) -> ActionResult:
        """Get comprehensive status of all robot components."""
        try:
            status = self.robot_interface.get_status()
            return ActionSucceeded(data=status)
            
        except Exception as e:
            self.logger.log_error(f"Error getting robot status: {e}")
            return ActionFailed(errors=str(e))
    
    @action(name="reset_movement_state", description="Reset movement state if stuck")
    @require_robot_interface
    def reset_movement_state(self) -> ActionResult:
        """Emergency reset of movement state if robot gets stuck."""
        try:
            # Force reset the movement state
            self.robot_interface.movement_controller.is_moving = False
            self.logger.log_info("Movement state reset successfully")
            return ActionSucceeded(data={"is_moving": False, "reset": True})
            
        except Exception as e:
            self.logger.log_error(f"Error resetting movement state: {e}")
            return ActionFailed(errors=str(e))
    
    # ========== Cleanup ==========
    
    def cleanup(self):
        """Cleanup all robot resources."""
        if self.robot_interface:
            try:
                self.robot_interface.cleanup()
            except Exception as e:
                self.logger.log_error(f"Error during cleanup: {e}")
        
        super().cleanup()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    from madsci.common.types.node_types import NodeDefinition
    
    # Load environment variables from root directory
    root_dir = Path(__file__).parent.parent
    load_dotenv(root_dir / '.env')
    
    # Get configuration from environment
    definition_path = Path(os.getenv("NODE_DEFINITION", "nodes/default.node.yaml"))
    node_url = os.getenv("ROBOT_NODE_URL", "http://localhost:2000")
    
    # Load node definition
    node_definition = None
    if definition_path.exists():
        node_definition = NodeDefinition.from_yaml(definition_path)
    
    # Initialize and start node
    robot_arm_node = DofbotModularNode(
        node_definition=node_definition,
        node_config=RobotArmConfig(node_url=node_url)
    )
    
    print(f"Starting DOFBOT Pro Modular Node (definition: {definition_path}, url: {node_url})")
    robot_arm_node.start_node()
