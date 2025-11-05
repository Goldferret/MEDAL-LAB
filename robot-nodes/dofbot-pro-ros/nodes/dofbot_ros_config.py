"""
ROS-Based DOFBOT Pro Configuration

Configuration for the ROS/MoveIT-based DOFBOT Pro robot node.
"""

from madsci.common.types.node_types import RestNodeConfig


class DofbotRosConfig(RestNodeConfig):
    """Configuration for ROS-based DOFBOT Pro Robot Node."""
    
    # node_url is set at runtime from ROBOT_NODE_URL environment variable
    # See __main__ block in dofbot_ros_node.py
    
    node_definition: str = "nodes/default.node.yaml"
    """Path to node definition file."""
    
    # ========================================================================
    # MOVEIT PLANNING PARAMETERS
    # ========================================================================
    # These parameters configure MoveIT's motion planning behavior
    # Values from manufacturer's example (arm_test.py)
    
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
    
    max_velocity_scaling_factor: float = 3.0
    """Maximum velocity scaling (0.0-1.0, where 1.0 is full speed)."""
    
    max_acceleration_scaling_factor: float = 3.0
    """Maximum acceleration scaling (0.0-1.0, where 1.0 is full acceleration)."""
    
    # ========================================================================
    # ROBOT POSITIONS
    # ========================================================================
    # Predefined joint positions for common robot poses
    # Format: [joint1, joint2, joint3, joint4, joint5] in radians
    
    home_position: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    """Home position - all joints at zero."""
    
    # Additional positions can be added as needed:
    # scan_position: list[float] = [0.79, 0.79, -1.57, -1.57, 0.0]
    # pickup_position: list[float] = [0.3, 0.3, -1.0, -1.0, 0.0]
