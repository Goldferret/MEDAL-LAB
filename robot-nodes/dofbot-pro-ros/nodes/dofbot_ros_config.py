"""
ROS-Based DOFBOT Pro Configuration

Configuration for the ROS/MoveIT-based DOFBOT Pro robot node.
"""

from madsci.common.types.node_types import RestNodeConfig


class DofbotRosConfig(RestNodeConfig):
    """Configuration for ROS-based DOFBOT Pro Robot Node."""
    
    node_definition: str = "nodes/default.node.yaml"
    """Path to node definition file."""
    
    # ========================================================================
    # MOVEIT PLANNING PARAMETERS
    # ========================================================================
    
    allow_replanning: bool = True
    """Allow MoveIT to replan if initial plan fails."""
    
    planning_time: float = 5.0
    """Maximum time (seconds) allowed for motion planning."""
    
    num_planning_attempts: int = 10
    """Number of planning attempts before giving up."""
    
    goal_position_tolerance: float = 0.01
    """Position tolerance (meters) for reaching goal."""
    
    goal_orientation_tolerance: float = 0.01
    """Orientation tolerance (radians) for reaching goal."""
    
    goal_tolerance: float = 0.01
    """Overall goal tolerance."""
    
    max_velocity_scaling_factor: float = 0.4
    """Maximum velocity scaling (0.0-1.0, where 1.0 is full speed)."""
    
    max_acceleration_scaling_factor: float = 0.4
    """Maximum acceleration scaling (0.0-1.0, where 1.0 is full acceleration)."""
    
    # ========================================================================
    # ROBOT POSITIONS
    # ========================================================================
    
    home_position: list[float] = [0.0, 0.9633, -1.5708, -1.5708, 0.0]
    """Home/init position joint angles in radians."""
    
    # ========================================================================
    # PICK AND PLACE PARAMETERS
    # ========================================================================
    
    table_z: float = 0.003
    """Table surface height in base_link frame (meters). Calibrated across workspace."""
    
    approach_height: float = 0.10
    """Height above target to approach from (meters)."""
    
    lift_height: float = 0.08
    """Height to lift to after grasping (meters)."""
    
    gripper_squeeze_cm: float = 1.0
    """How much tighter than object width to close gripper (cm)."""
    
    min_object_width_cm: float = 2.0
    """Minimum object width to accept (cm). Rejects cables/noise."""
    
    # Gripper calibration: linear fit from empirical measurements
    # gap_cm = gripper_slope * grip_joint + gripper_intercept
    gripper_slope: float = -3.9318
    """Gripper calibration slope (cm per radian)."""
    
    gripper_intercept: float = 0.7795
    """Gripper calibration intercept (cm)."""
    
    # ========================================================================
    # CAMERA
    # ========================================================================
    
    camera_intrinsics_path: str = "calibration/camera_intrinsics.json"
    """Path to camera intrinsics JSON file (relative to working directory)."""
