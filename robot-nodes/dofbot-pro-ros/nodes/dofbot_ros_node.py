"""
MADSci Robot Node for ROS-Based DOFBOT Pro

Provides MADSci-compliant interface for controlling DOFBOT Pro via ROS and MoveIT.
"""

import tempfile
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from madsci.node_module.rest_node_module import RestNode
from madsci.node_module.helpers import action
from madsci.common.types.resource_types import Slot
from madsci.common.types.datapoint_types import FileDataPoint
from dofbot_ros_config import DofbotRosConfig
from dofbot_ros_interface import DofbotRosInterface


class DofbotRosNode(RestNode):
    """MADSci-compliant robot node for ROS-based DOFBOT Pro."""
    
    robot_interface: DofbotRosInterface = None
    config_model = DofbotRosConfig
    
    def startup_handler(self) -> None:
        """Initialize robot interface and resources."""
        self.robot_interface = DofbotRosInterface(self.config, self.logger)
        self.logger.log("Robot interface initialized")
        
        from madsci.client import LocationClient
        self.location_client = LocationClient()
        self.logger.log("Location client initialized")
        
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
        self.logger.log("Shutting down robot node")
        if self.robot_interface:
            del self.robot_interface
    
    def state_handler(self) -> None:
        if self.robot_interface:
            current_joints = self.robot_interface.get_current_joints()
            self.node_state = {
                "joint_positions": current_joints,
                "gripper_resource_id": self.gripper.resource_id if self.gripper else None,
                "num_joints": len(current_joints)
            }
    
    # ========================================================================
    # BASIC ACTIONS
    # ========================================================================
    
    @action
    def move_to_position(self, joint_positions: list[float]) -> dict:
        """Move robot to specified joint positions."""
        if len(joint_positions) != 5:
            return {"status": "error", "message": f"Expected 5 joints, got {len(joint_positions)}"}
        
        success = self.robot_interface.move_to_joints(joint_positions)
        if success:
            return {"status": "success", "message": "Movement completed", "final_position": joint_positions}
        return {"status": "error", "message": "Movement failed"}
    
    @action
    def get_current_position(self) -> dict:
        """Get current robot joint positions."""
        joints = self.robot_interface.get_current_joints()
        if joints:
            return {"status": "success", "joint_positions": joints}
        return {"status": "error", "message": "Failed to get position"}
    
    @action
    def home_robot(self) -> dict:
        """Move robot to home/init position."""
        success = self.robot_interface.move_to_joints(self.config.home_position)
        if success:
            return {"status": "success", "message": "Robot homed"}
        return {"status": "error", "message": "Failed to home robot"}
    
    @action
    def open_gripper(self) -> dict:
        """Open the gripper."""
        if self.robot_interface.open_gripper():
            return {"status": "success", "message": "Gripper opened"}
        return {"status": "error", "message": "Failed to open gripper"}
    
    @action
    def close_gripper(self) -> dict:
        """Close the gripper."""
        if self.robot_interface.move_gripper(-0.5):
            return {"status": "success", "message": "Gripper closed"}
        return {"status": "error", "message": "Failed to close gripper"}
    
    # ========================================================================
    # CAMERA
    # ========================================================================
    
    def _save_as_datapoint(self, data, filename: str, label: str):
        """Save data to a temp file and submit as FileDataPoint. Returns datapoint_id."""
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename}')
        temp_path = Path(temp.name)
        temp.close()
        
        if filename.endswith('.npy'):
            np.save(str(temp_path), data)
        else:
            cv2.imwrite(str(temp_path), data)
        
        datapoint = FileDataPoint(label=label, path=str(temp_path))
        submitted = self.data_client.submit_datapoint(datapoint)
        os.remove(temp_path)
        return submitted.datapoint_id
    
    @action
    def capture_camera_image(self) -> dict:
        """
        Capture color and depth images from camera.
        
        Returns:
            Dict with color_datapoint_id and depth_datapoint_id.
        """
        try:
            color = self.robot_interface.capture_camera_image()
            if color is None:
                return {"status": "error", "message": "Failed to capture color image"}
            
            depth = self.robot_interface.capture_depth_image()
            if depth is None:
                return {"status": "error", "message": "Failed to capture depth image"}
            
            color_id = self._save_as_datapoint(color, "color.jpg", "camera_color")
            depth_id = self._save_as_datapoint(depth, "depth.npy", "camera_depth")
            
            self.logger.log(f"Captured color={color_id}, depth={depth_id}")
            return {
                "status": "success",
                "color_datapoint_id": color_id,
                "depth_datapoint_id": depth_id,
            }
            
        except Exception as e:
            self.logger.log_error(f"Capture failed: {e}")
            return {"status": "error", "message": str(e)}
    
    # ========================================================================
    # VISION-GUIDED PICK AND PLACE
    # ========================================================================
    
    @action
    def pick_at_pixel_coords(
        self,
        pixel_x: int,
        pixel_y: int,
        depth_mm: int,
        width_px: float,
    ) -> dict:
        """
        Pick an object at pixel coordinates.
        
        Client detects object in color image, reads depth, sends pixel coords.
        Node deprojects, TF transforms, and executes pick sequence.
        
        Args:
            pixel_x, pixel_y: Object centroid in color image pixels
            depth_mm: Depth at centroid in millimeters
            width_px: Object width in pixels (from minAreaRect)
        """
        cfg = self.config
        iface = self.robot_interface
        
        # Convert pixel width to meters
        width_m = iface.pixel_width_to_meters(width_px, depth_mm)
        width_cm = width_m * 100.0
        if width_cm < cfg.min_object_width_cm:
            return {"status": "error", "message": f"Object too small ({width_cm:.1f}cm < {cfg.min_object_width_cm}cm)"}
        
        # Deproject + TF → base_link
        base_coords = iface.deproject_and_transform(pixel_x, pixel_y, depth_mm)
        if base_coords is None:
            return {"status": "error", "message": "Deproject/TF failed"}
        bx, by, bz = base_coords
        
        # Centroid Z correction
        grasp_z = (cfg.table_z + bz) / 2.0
        if grasp_z < cfg.table_z:
            return {"status": "error", "message": f"Grasp Z ({grasp_z:.3f}) below table ({cfg.table_z:.3f})"}
        
        self.logger.log(f"Pick: pixel({pixel_x},{pixel_y}) depth={depth_mm}mm → base({bx:.3f},{by:.3f},{bz:.3f}), grasp_z={grasp_z:.3f}, width={width_cm:.1f}cm")
        
        # Downward-facing orientation
        qx, qy, qz, qw = -0.2924, 0.0, 0.0, 0.9563
        
        # Open gripper
        if not iface.open_gripper():
            return {"status": "error", "message": "Failed to open gripper"}
        
        # Approach above target
        if not iface.move_to_pose(bx, by, grasp_z + cfg.approach_height, qx, qy, qz, qw):
            return {"status": "error", "message": "Failed to reach approach position"}
        
        # Descend to grasp
        if not iface.move_to_pose(bx, by, grasp_z, qx, qy, qz, qw):
            return {"status": "error", "message": "Failed to reach grasp position"}
        
        # Grip sized to object
        if not iface.grip_for_width(width_m):
            return {"status": "error", "message": "Failed to close gripper"}
        
        # Lift
        if not iface.move_to_pose(bx, by, grasp_z + cfg.lift_height, qx, qy, qz, qw):
            return {"status": "error", "message": "Failed to lift"}
        
        self.logger.log("Pick completed")
        return {"status": "success", "message": "Pick completed", "base_coords": [bx, by, bz], "grasp_z": grasp_z, "width_m": width_m}
    
    @action
    def place_at_pixel_coords(
        self,
        pixel_x: int,
        pixel_y: int,
        depth_mm: int,
    ) -> dict:
        """
        Place held object at pixel coordinates.
        
        Args:
            pixel_x, pixel_y: Target location in color image pixels
            depth_mm: Depth at target in millimeters
        """
        cfg = self.config
        iface = self.robot_interface
        
        # Deproject + TF → base_link
        base_coords = iface.deproject_and_transform(pixel_x, pixel_y, depth_mm)
        if base_coords is None:
            return {"status": "error", "message": "Deproject/TF failed"}
        bx, by, bz = base_coords
        
        place_z = (cfg.table_z + bz) / 2.0
        qx, qy, qz, qw = -0.2924, 0.0, 0.0, 0.9563
        
        self.logger.log(f"Place: pixel({pixel_x},{pixel_y}) depth={depth_mm}mm → base({bx:.3f},{by:.3f},{bz:.3f}), place_z={place_z:.3f}")
        
        # Approach above target
        if not iface.move_to_pose(bx, by, place_z + cfg.approach_height, qx, qy, qz, qw):
            return {"status": "error", "message": "Failed to reach approach position"}
        
        # Descend
        if not iface.move_to_pose(bx, by, place_z, qx, qy, qz, qw):
            return {"status": "error", "message": "Failed to reach place position"}
        
        # Release
        if not iface.open_gripper():
            return {"status": "error", "message": "Failed to open gripper"}
        
        # Lift away
        if not iface.move_to_pose(bx, by, place_z + cfg.approach_height, qx, qy, qz, qw):
            return {"status": "error", "message": "Failed to lift after placing"}
        
        self.logger.log("Place completed")
        return {"status": "success", "message": "Place completed", "base_coords": [bx, by, bz]}
    
    # ========================================================================
    # LOCATION-BASED PICK AND PLACE (existing)
    # ========================================================================
    
    def pick_from_location(self, location_id: str) -> dict:
        """Pick object from predefined location."""
        try:
            location = self.location_client.get_location(location_id)
            node_name = self.node_definition.node_name
            if node_name not in location.representations:
                return {"status": "error", "message": f"No representation for {node_name}"}
            
            reps = location.representations[node_name]
            
            if not self.robot_interface.open_gripper():
                return {"status": "error", "message": "Failed to open gripper"}
            if not self.robot_interface.move_to_joints(reps["raised"]):
                return {"status": "error", "message": "Failed to move to raised"}
            if not self.robot_interface.move_to_joints(reps["lowered"]):
                return {"status": "error", "message": "Failed to move to lowered"}
            if not self.robot_interface.move_gripper(-0.5):
                return {"status": "error", "message": "Failed to close gripper"}
            
            resource_id = location.resource_id
            if resource_id:
                self.resource_client.push(self.gripper.resource_id, resource_id)
            
            if not self.robot_interface.move_to_joints(reps["raised"]):
                return {"status": "error", "message": "Failed to return to raised"}
            
            return {"status": "success", "message": "Pick completed", "resource_id": resource_id}
            
        except Exception as e:
            self.logger.log_error(f"Pick failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def place_at_location(self, location_id: str) -> dict:
        """Place held object at predefined location."""
        try:
            popped_resource = None
            try:
                popped, _ = self.resource_client.pop(self.gripper.resource_id)
                popped_resource = popped
            except Exception:
                pass
            
            location = self.location_client.get_location(location_id)
            node_name = self.node_definition.node_name
            if node_name not in location.representations:
                return {"status": "error", "message": f"No representation for {node_name}"}
            
            reps = location.representations[node_name]
            
            if not self.robot_interface.move_to_joints(reps["raised"]):
                return {"status": "error", "message": "Failed to move to raised"}
            if not self.robot_interface.move_to_joints(reps["lowered"]):
                return {"status": "error", "message": "Failed to move to lowered"}
            if not self.robot_interface.open_gripper():
                return {"status": "error", "message": "Failed to open gripper"}
            
            if popped_resource:
                self.location_client.attach_resource(
                    location_id=location_id, resource_id=popped_resource.resource_id
                )
            
            if not self.robot_interface.move_to_joints(reps["raised"]):
                return {"status": "error", "message": "Failed to return to raised"}
            
            return {
                "status": "success", "message": "Place completed",
                "resource_id": popped_resource.resource_id if popped_resource else None
            }
            
        except Exception as e:
            self.logger.log_error(f"Place failed: {e}")
            return {"status": "error", "message": str(e)}
    
    @action
    def swap_blocks(self, location_a_id: str, location_b_id: str, temp_location_id: str) -> dict:
        """Swap blocks between two locations using temporary location."""
        try:
            for src, dst, label in [
                (location_a_id, temp_location_id, "A→Temp"),
                (location_b_id, location_a_id, "B→A"),
                (temp_location_id, location_b_id, "Temp→B"),
            ]:
                result = self.pick_from_location(src)
                if result["status"] != "success":
                    return {"status": "error", "message": f"{label} pick failed: {result['message']}"}
                result = self.place_at_location(dst)
                if result["status"] != "success":
                    return {"status": "error", "message": f"{label} place failed: {result['message']}"}
            
            return {"status": "success", "message": "Swap completed"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    node_url = os.getenv("DOFBOT_PRO_1_URL", "http://192.168.1.200:2000/")
    node = DofbotRosNode(node_config=DofbotRosConfig(node_url=node_url))
    node.start_node()
