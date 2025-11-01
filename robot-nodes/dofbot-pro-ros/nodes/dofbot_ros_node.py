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


if __name__ == "__main__":
    import os
    
    # Get node URL from environment (uses actual robot IP from .env.global)
    # With host networking, RestNode can bind to the robot's specific IP
    # Default uses .200 to verify env substitution is working (actual is .77)
    node_url = os.getenv("DOFBOT_PRO_1_URL", "http://192.168.1.200:2000/")
    
    node = DofbotRosNode(node_config=DofbotRosConfig(node_url=node_url))
    node.start_node()
