"""
ROS/MoveIT Interface for DOFBOT Pro

Handles ROS communication and MoveIT motion planning for the DOFBOT Pro robot.
Uses action client interface to communicate with host's move_group server.
"""

from typing import Optional
import threading
import rospy
import actionlib
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal, Constraints, JointConstraint
from sensor_msgs.msg import JointState
from madsci.client.event_client import EventClient
from dofbot_ros_config import DofbotRosConfig


class DofbotRosInterface:
    """Interface for controlling DOFBOT Pro via ROS and MoveIT action client."""
    
    def __init__(
        self, 
        config: DofbotRosConfig,
        logger: Optional[EventClient] = None
    ) -> None:
        """
        Initialize ROS node and MoveIT action client.
        
        Args:
            config: Configuration with MoveIT parameters
            logger: Event logger for status messages
        """
        self.config = config
        self.logger = logger or EventClient()
        
        # Initialize ROS node (disable_signals=True because we're in a thread)
        rospy.init_node("dofbot_madsci_node", anonymous=True, disable_signals=True)
        self.logger.log("ROS node initialized")
        
        # Create action client for move_group
        self.move_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        self.logger.log("Waiting for move_group action server...")
        self.move_client.wait_for_server()
        self.logger.log("Connected to move_group action server")
        
        # Subscribe to joint states for current position
        self.current_joint_state = None
        self.joint_state_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self._joint_state_callback
        )
        
        # Start background thread to process ROS callbacks
        self.spinner = threading.Thread(target=rospy.spin, daemon=True)
        self.spinner.start()
        
        # Wait for first joint state message
        rospy.sleep(0.5)
        self.logger.log("MoveIT action client initialized and configured")
    
    def _joint_state_callback(self, msg: JointState) -> None:
        """Store latest joint state."""
        self.current_joint_state = msg
    
    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """
        Move robot to specified joint positions using MoveIT.
        
        Args:
            joint_positions: List of 5 joint angles in radians
            
        Returns:
            True if movement successful, False otherwise
        """
        try:
            # Create goal message
            goal = MoveGroupGoal()
            
            # Set planning group
            goal.request.group_name = "arm_group"
            
            # Set planning parameters
            goal.request.num_planning_attempts = self.config.num_planning_attempts
            goal.request.allowed_planning_time = self.config.planning_time
            goal.request.max_velocity_scaling_factor = self.config.max_velocity_scaling_factor
            goal.request.max_acceleration_scaling_factor = self.config.max_acceleration_scaling_factor
            
            # Create joint constraints for goal
            constraints = Constraints()
            joint_names = ['Arm1_Joint', 'Arm2_Joint', 'Arm3_Joint', 'Arm4_Joint', 'Arm5_Joint']
            
            for i, (name, position) in enumerate(zip(joint_names, joint_positions)):
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = position
                constraint.tolerance_above = self.config.goal_tolerance
                constraint.tolerance_below = self.config.goal_tolerance
                constraint.weight = 1.0
                constraints.joint_constraints.append(constraint)
            
            goal.request.goal_constraints.append(constraints)
            
            # Set planning options
            goal.planning_options.plan_only = False  # Plan and execute
            
            # Send goal and wait for result
            self.logger.log(f"Sending movement goal to {joint_positions}")
            self.move_client.send_goal(goal)
            
            # Wait for result with timeout
            finished = self.move_client.wait_for_result(rospy.Duration(30.0))
            
            if not finished:
                self.logger.log_error("Movement timed out after 30 seconds")
                return False
            
            # Check result
            result = self.move_client.get_result()
            state = self.move_client.get_state()
            
            if state == actionlib.GoalStatus.SUCCEEDED:
                self.logger.log("Movement completed successfully")
                return True
            else:
                self.logger.log_error(f"Movement failed with state: {state}")
                return False
            
        except Exception as e:
            self.logger.log_error(f"Movement failed: {str(e)}")
            return False
    
    def get_current_joints(self) -> list[float]:
        """
        Get current joint positions.
        
        Returns:
            List of 5 current joint angles in radians
        """
        try:
            if self.current_joint_state is None:
                self.logger.log_error("No joint state received yet")
                return []
            
            # Extract arm joint positions (first 5 joints)
            return list(self.current_joint_state.position[:5])
            
        except Exception as e:
            self.logger.log_error(f"Failed to get joint values: {str(e)}")
            return []
