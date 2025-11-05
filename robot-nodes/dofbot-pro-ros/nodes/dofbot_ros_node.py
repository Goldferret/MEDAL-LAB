"""
MADSci Robot Node for ROS-Based DOFBOT Pro

Provides MADSci-compliant interface for controlling DOFBOT Pro via ROS and MoveIT.
"""

from typing import Any
from madsci.node_module.rest_node_module import RestNode
from madsci.node_module.helpers import action
from madsci.common.types.resource_types import Slot
from dofbot_ros_config import DofbotRosConfig
from dofbot_ros_interface import DofbotRosInterface


class DofbotRosNode(RestNode):
    """MADSci-compliant robot node for ROS-based DOFBOT Pro."""
    
    robot_interface: DofbotRosInterface = None
    config_model = DofbotRosConfig
    
    def startup_handler(self) -> None:
        """Initialize robot interface and resources."""
        # Initialize ROS/MoveIT interface
        self.robot_interface = DofbotRosInterface(self.config, self.logger)
        self.logger.log("Robot interface initialized")
        
        # Initialize Location Manager client
        from madsci.client import LocationClient
        self.location_client = LocationClient()
        self.logger.log("Location client initialized")
        
        # Create gripper resource for tracking what robot is holding
        self.gripper = self.resource_client.add_resource(
            Slot(
                resource_name=f"dofbot_gripper_{self.node_definition.node_name}",
                resource_class="robot_gripper",
                capacity=1,
                attributes={
                    "gripper_type": "dofbot_pro",
                    "description": "DOFBOT Pro gripper for holding objects"
                }
            )
        )
        
        self.logger.log(f"Gripper resource initialized: {self.gripper.resource_id}")
        self.logger.log("DOFBOT Pro robot node startup complete")
    
    def shutdown_handler(self) -> None:
        """Clean up robot interface."""
        self.logger.log("Shutting down robot node")
        if self.robot_interface:
            del self.robot_interface
    
    def state_handler(self) -> None:
        """Report current robot state."""
        if self.robot_interface:
            current_joints = self.robot_interface.get_current_joints()
            self.node_state = {
                "joint_positions": current_joints,
                "gripper_resource_id": self.gripper.resource_id if self.gripper else None,
                "num_joints": len(current_joints)
            }
    
    @action
    def move_to_position(self, joint_positions: list[float]) -> dict:
        """
        Move robot to specified joint positions.
        
        Args:
            joint_positions: List of 5 joint angles in radians
            
        Returns:
            Dictionary with status and message
        """
        # Validate input
        if len(joint_positions) != 5:
            return {
                "status": "error",
                "message": f"Expected 5 joint positions, got {len(joint_positions)}"
            }
        
        # Execute movement via interface
        self.logger.log(f"Moving to position: {joint_positions}")
        success = self.robot_interface.move_to_joints(joint_positions)
        
        if success:
            return {
                "status": "success",
                "message": "Movement completed",
                "final_position": joint_positions
            }
        else:
            return {
                "status": "error",
                "message": "Movement failed - check logs for details"
            }
    
    @action
    def get_current_position(self) -> dict:
        """
        Get current robot joint positions.
        
        Returns:
            Dictionary with current joint positions
        """
        current_joints = self.robot_interface.get_current_joints()
        
        if current_joints:
            return {
                "status": "success",
                "joint_positions": current_joints,
                "num_joints": len(current_joints)
            }
        else:
            return {
                "status": "error",
                "message": "Failed to get current position"
            }
    
    @action
    def home_robot(self) -> dict:
        """
        Move robot to home position.
        
        Returns:
            Dictionary with status and message
        """
        home_position = self.config.home_position
        self.logger.log(f"Moving to home position: {home_position}")
        
        success = self.robot_interface.move_to_joints(home_position)
        
        if success:
            return {
                "status": "success",
                "message": "Robot homed successfully",
                "home_position": home_position
            }
        else:
            return {
                "status": "error",
                "message": "Failed to home robot"
            }
    
    @action
    def open_gripper(self) -> dict:
        """
        Open the gripper to release objects.
        
        Returns:
            Dictionary with status and message
        """
        open_position = [-1.25, -1.25, -1.25]
        self.logger.log("Opening gripper")
        
        success = self.robot_interface.move_gripper(open_position)
        
        if success:
            return {
                "status": "success",
                "message": "Gripper opened successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to open gripper"
            }
    
    @action
    def close_gripper(self) -> dict:
        """
        Close the gripper to grasp objects.
        
        Returns:
            Dictionary with status and message
        """
        closed_position = [-0.5, -0.5, -0.5]
        self.logger.log("Closing gripper")
        
        success = self.robot_interface.move_gripper(closed_position)
        
        if success:
            return {
                "status": "success",
                "message": "Gripper closed successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to close gripper"
            }
    
    @action
    def capture_camera_image(self) -> dict:
        """
        Capture image from camera and return as datapoint.
        
        Returns:
            Dictionary with datapoint_id of captured image
        """
        import tempfile
        import os
        from pathlib import Path
        import cv2
        from madsci.common.types.datapoint_types import FileDataPoint
        
        try:
            self.logger.log("Capturing camera image")
            
            # Get image from ROS (interface layer)
            cv_image = self.robot_interface.capture_camera_image()
            
            if cv_image is None:
                return {
                    "status": "error",
                    "message": "Failed to capture image from camera"
                }
            
            # Write to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = Path(temp_file.name)
            temp_file.close()
            cv2.imwrite(str(temp_path), cv_image)
            
            # Submit to Data Manager
            datapoint = FileDataPoint(
                label="camera_capture",
                path=str(temp_path)
            )
            submitted = self.data_client.submit_datapoint(datapoint)
            
            # Delete temp file (Data Manager has copied it)
            os.remove(temp_path)
            
            self.logger.log(f"Image captured and stored: {submitted.datapoint_id}")
            
            # Return the datapoint ID - MADSci will handle it
            return submitted.datapoint_id
            
        except Exception as e:
            self.logger.log_error(f"Failed to capture image: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to capture image: {str(e)}"
            }
    
    def pick_from_location(self, location_id: str) -> dict:
        """
        Pick object from specified location.
        
        Args:
            location_id: ID of location to pick from
            
        Returns:
            Dictionary with status, message, and resource_id
        """
        try:
            self.logger.log(f"Picking from location: {location_id}")
            
            # Get location from Location Manager
            location = self.location_client.get_location(location_id)
            
            # Extract representations for this node
            node_name = self.node_definition.node_name
            if node_name not in location.representations:
                return {
                    "status": "error",
                    "message": f"No representation found for node {node_name}"
                }
            
            representations = location.representations[node_name]
            
            # Move to raised position
            
            # Open gripper before approaching
            if not self.robot_interface.move_gripper([-1.25, -1.25, -1.25]):
                return {"status": "error", "message": "Failed to open gripper"}
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to move to raised position"}
            
            # Move to lowered position
            if not self.robot_interface.move_to_joints(representations["lowered"]):
                return {"status": "error", "message": "Failed to move to lowered position"}
            
            # Close gripper
            if not self.robot_interface.move_gripper([-0.5, -0.5, -0.5]):
                return {"status": "error", "message": "Failed to close gripper"}
            
            # Get resource at location
            resource_id = location.resource_id
            if not resource_id:
                self.logger.log_warning("No resource at location, continuing anyway")
            else:
                # Push resource to gripper
                self.resource_client.push(self.gripper.resource_id, resource_id)
                self.logger.log(f"Pushed resource {resource_id} to gripper")
            
            # Move back to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to return to raised position"}
            
            return {
                "status": "success",
                "message": "Pick completed successfully",
                "resource_id": resource_id
            }
            
        except Exception as e:
            self.logger.log_error(f"Pick failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Pick failed: {str(e)}"
            }
    
    def place_at_location(self, location_id: str) -> dict:
        """
        Place held object at specified location.
        
        Args:
            location_id: ID of location to place at
            
        Returns:
            Dictionary with status, message, and resource_id
        """
        try:
            self.logger.log(f"Placing at location: {location_id}")
            
            # Pop resource from gripper
            popped_resource = None
            try:
                popped, updated_gripper = self.resource_client.pop(self.gripper.resource_id)
                popped_resource = popped
                self.logger.log(f"Popped resource {popped.resource_id} from gripper")
            except Exception as e:
                self.logger.log_warning(f"No resource in gripper: {str(e)}")
            
            # Get location from Location Manager
            location = self.location_client.get_location(location_id)
            
            # Extract representations for this node
            node_name = self.node_definition.node_name
            if node_name not in location.representations:
                return {
                    "status": "error",
                    "message": f"No representation found for node {node_name}"
                }
            
            representations = location.representations[node_name]
            
            # Move to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to move to raised position"}
            
            # Move to lowered position
            if not self.robot_interface.move_to_joints(representations["lowered"]):
                return {"status": "error", "message": "Failed to move to lowered position"}
            
            # Open gripper
            if not self.robot_interface.move_gripper([-1.25, -1.25, -1.25]):
                return {"status": "error", "message": "Failed to open gripper"}
            
            # Attach resource to location
            if popped_resource:
                self.location_client.attach_resource(
                    location_id=location_id,
                    resource_id=popped_resource.resource_id
                )
                self.logger.log(f"Attached resource {popped_resource.resource_id} to location")
            
            # Move back to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to return to raised position"}
            
            return {
                "status": "success",
                "message": "Place completed successfully",
                "resource_id": popped_resource.resource_id if popped_resource else None
            }
            
        except Exception as e:
            self.logger.log_error(f"Place failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Place failed: {str(e)}"
            }
    
    @action
    def swap_blocks(
        self, 
        location_a_id: str, 
        location_b_id: str,
        temp_location_id: str
    ) -> dict:
        """
        Swap blocks between two locations using temporary location.
        
        Args:
            location_a_id: First location ID
            location_b_id: Second location ID
            temp_location_id: Temporary holding location ID
            
        Returns:
            Dictionary with status and message
        """
        try:
            self.logger.log(f"Swapping blocks: {location_a_id} <-> {location_b_id}")
            
            # Step 1: Pick from A, place at Temp
            result = self.pick_from_location(location_a_id)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to pick from A: {result['message']}"}
            
            result = self.place_at_location(temp_location_id)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to place at Temp: {result['message']}"}
            
            # Step 2: Pick from B, place at A
            result = self.pick_from_location(location_b_id)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to pick from B: {result['message']}"}
            
            result = self.place_at_location(location_a_id)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to place at A: {result['message']}"}
            
            # Step 3: Pick from Temp, place at B
            result = self.pick_from_location(temp_location_id)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to pick from Temp: {result['message']}"}
            
            result = self.place_at_location(location_b_id)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to place at B: {result['message']}"}
            
            self.logger.log("Swap completed successfully")
            return {
                "status": "success",
                "message": "Swap completed successfully"
            }
            
        except Exception as e:
            self.logger.log_error(f"Swap failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Swap failed: {str(e)}"
            }


if __name__ == "__main__":
    import os
    
    # Get node URL from environment (uses actual robot IP from .env.global)
    # With host networking, RestNode can bind to the robot's specific IP
    # Default uses .200 to verify env substitution is working (actual is .77)
    node_url = os.getenv("DOFBOT_PRO_1_URL", "http://192.168.1.200:2000/")
    
    node = DofbotRosNode(node_config=DofbotRosConfig(node_url=node_url))
    node.start_node()
