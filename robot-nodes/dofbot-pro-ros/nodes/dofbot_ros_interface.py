"""
ROS/MoveIT Interface for DOFBOT Pro

Handles ROS communication and MoveIT motion planning for the DOFBOT Pro robot.
Uses MoveIT service calls (/compute_ik, /plan_kinematic_path) and action client
for trajectory execution. This avoids moveit_commander C++ bindings which
segfault in Docker/conda environments.
"""

from typing import Optional
import json
import threading
import numpy as np
import rospy
import actionlib
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform
from moveit_msgs.msg import (
    MoveGroupAction, MoveGroupGoal, Constraints, JointConstraint,
    PositionIKRequest,
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import JointState, Image
from madsci.client.event_client import EventClient
from dofbot_ros_config import DofbotRosConfig

ARM_JOINT_NAMES = ['Arm1_Joint', 'Arm2_Joint', 'Arm3_Joint', 'Arm4_Joint', 'Arm5_Joint']
EEF_LINK = 'Gripping_point_Link'
PLANNING_GROUP = 'arm_group'
GRIPPER_GROUP = 'grip_group'
DEPTH_SAMPLE_RADIUS = 3


class DofbotRosInterface:
    """Interface for controlling DOFBOT Pro via ROS and MoveIT action client."""
    
    def __init__(
        self, 
        config: DofbotRosConfig,
        logger: Optional[EventClient] = None
    ) -> None:
        self.config = config
        self.logger = logger or EventClient()
        
        # Initialize ROS node
        rospy.init_node("dofbot_madsci_node", anonymous=True, disable_signals=True)
        self.logger.log("ROS node initialized")
        
        # Load camera intrinsics
        with open(config.camera_intrinsics_path) as f:
            calib = json.load(f)
        self.fx = calib['camera_matrix'][0][0]
        self.fy = calib['camera_matrix'][1][1]
        self.cx = calib['camera_matrix'][0][2]
        self.cy = calib['camera_matrix'][1][2]
        self.logger.log(f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        
        # Action client for move_group (arm + gripper)
        self.move_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        self.logger.log("Waiting for move_group action server...")
        self.move_client.wait_for_server()
        self.logger.log("Connected to move_group action server")
        
        self.gripper_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        
        # MoveIT service proxies for IK and planning
        self.logger.log("Waiting for MoveIT services...")
        rospy.wait_for_service('/compute_ik', timeout=10.0)
        rospy.wait_for_service('/plan_kinematic_path', timeout=10.0)
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.plan_service = rospy.ServiceProxy('/plan_kinematic_path', GetMotionPlan)
        self.logger.log("MoveIT services connected")
        
        # TF buffer for camera→base_link transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.logger.log("TF listener initialized")
        
        # Joint state subscriber
        self.current_joint_state = None
        self.joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback
        )
        
        # Background ROS spinner
        self.spinner = threading.Thread(target=rospy.spin, daemon=True)
        self.spinner.start()
        
        rospy.sleep(1.0)  # Wait for TF and joint states
        self.logger.log("DOFBOT ROS interface ready")
    
    def _joint_state_callback(self, msg: JointState) -> None:
        self.current_joint_state = msg
    
    # ========================================================================
    # DEPROJECTION & TF
    # ========================================================================
    
    def deproject_and_transform(self, pixel_x: int, pixel_y: int, depth_mm: int) -> Optional[tuple]:
        """
        Deproject pixel + depth to base_link frame.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates in color image
            depth_mm: Depth value in millimeters
            
        Returns:
            (x, y, z) in base_link frame, or None on failure.
        """
        depth_m = depth_mm / 1000.0
        if depth_m <= 0:
            self.logger.log_error(f"Invalid depth: {depth_mm}mm")
            return None
        
        # Deproject pixel → camera frame
        cam_x = (pixel_x - self.cx) * depth_m / self.fx
        cam_y = (pixel_y - self.cy) * depth_m / self.fy
        cam_z = depth_m
        
        # TF transform camera → base_link
        try:
            point = PointStamped()
            point.header.frame_id = 'camera_color_optical_frame'
            point.header.stamp = rospy.Time(0)
            point.point.x = cam_x
            point.point.y = cam_y
            point.point.z = cam_z
            
            transformed = self.tf_buffer.transform(point, 'base_link', rospy.Duration(2.0))
            bx = transformed.point.x
            by = transformed.point.y
            bz = transformed.point.z
            self.logger.log(f"Deproject+TF: pixel({pixel_x},{pixel_y}) depth={depth_mm}mm → base({bx:.3f},{by:.3f},{bz:.3f})")
            return (bx, by, bz)
            
        except Exception as e:
            self.logger.log_error(f"TF transform failed: {e}")
            return None
    
    def pixel_width_to_meters(self, width_px: float, depth_mm: int) -> float:
        """Convert pixel width to meters using depth and camera intrinsics."""
        return width_px * (depth_mm / 1000.0) / self.fx
    
    # ========================================================================
    # CORE MOTION
    # ========================================================================
    
    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """Move arm to joint positions via MoveGroupAction."""
        try:
            goal = MoveGroupGoal()
            goal.request.group_name = PLANNING_GROUP
            goal.request.num_planning_attempts = self.config.num_planning_attempts
            goal.request.allowed_planning_time = self.config.planning_time
            goal.request.max_velocity_scaling_factor = self.config.max_velocity_scaling_factor
            goal.request.max_acceleration_scaling_factor = self.config.max_acceleration_scaling_factor
            
            constraints = Constraints()
            for name, position in zip(ARM_JOINT_NAMES, joint_positions):
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = position
                jc.tolerance_above = self.config.goal_tolerance
                jc.tolerance_below = self.config.goal_tolerance
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)
            
            goal.request.goal_constraints.append(constraints)
            goal.planning_options.plan_only = False
            
            self.move_client.send_goal(goal)
            finished = self.move_client.wait_for_result(rospy.Duration(30.0))
            
            if not finished:
                self.logger.log_error("Movement timed out")
                return False
            
            return self.move_client.get_state() == actionlib.GoalStatus.SUCCEEDED
            
        except Exception as e:
            self.logger.log_error(f"move_to_joints failed: {e}")
            return False
    
    def move_to_pose(self, x: float, y: float, z: float,
                     qx: float = 0.0, qy: float = 0.0,
                     qz: float = 0.0, qw: float = 1.0) -> bool:
        """
        Move end effector to Cartesian pose in base_link frame.
        Uses /compute_ik → /plan_kinematic_path → execute.
        """
        # Step 1: IK
        ik_req = PositionIKRequest()
        ik_req.group_name = PLANNING_GROUP
        ik_req.avoid_collisions = True
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        ik_req.pose_stamped = pose
        
        try:
            ik_resp = self.ik_service(ik_req)
        except rospy.ServiceException as e:
            self.logger.log_error(f"IK service call failed: {e}")
            return False
        
        if ik_resp.error_code.val != 1:
            self.logger.log_error(f"IK failed (code {ik_resp.error_code.val}) for pose ({x:.3f}, {y:.3f}, {z:.3f})")
            return False
        
        # Extract arm joints from IK solution
        joint_map = dict(zip(ik_resp.solution.joint_state.name,
                             ik_resp.solution.joint_state.position))
        joint_positions = [joint_map[name] for name in ARM_JOINT_NAMES]
        
        self.logger.log(f"IK solved for ({x:.3f}, {y:.3f}, {z:.3f}) → joints {[round(j,3) for j in joint_positions]}")
        return self.move_to_joints(joint_positions)
    
    # ========================================================================
    # GRIPPER
    # ========================================================================
    
    def move_gripper(self, grip_joint: float) -> bool:
        """Move gripper to specified joint position. Clamped to [-1.5, 0.0]."""
        grip_joint = max(-1.5, min(0.0, grip_joint))
        
        try:
            goal = MoveGroupGoal()
            goal.request.group_name = GRIPPER_GROUP
            goal.request.num_planning_attempts = self.config.num_planning_attempts
            goal.request.allowed_planning_time = self.config.planning_time
            goal.request.max_velocity_scaling_factor = self.config.max_velocity_scaling_factor
            goal.request.max_acceleration_scaling_factor = self.config.max_acceleration_scaling_factor
            
            constraints = Constraints()
            jc = JointConstraint()
            jc.joint_name = 'grip_joint'
            jc.position = grip_joint
            jc.tolerance_above = self.config.goal_tolerance
            jc.tolerance_below = self.config.goal_tolerance
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
            goal.request.goal_constraints.append(constraints)
            goal.planning_options.plan_only = False
            
            self.gripper_client.send_goal(goal)
            finished = self.gripper_client.wait_for_result(rospy.Duration(10.0))
            
            if not finished:
                self.logger.log_error("Gripper timed out")
                return False
            
            return self.gripper_client.get_state() == actionlib.GoalStatus.SUCCEEDED
            
        except Exception as e:
            self.logger.log_error(f"Gripper failed: {e}")
            return False
    
    def grip_for_width(self, width_m: float) -> bool:
        """Close gripper to match object width with squeeze margin."""
        width_cm = width_m * 100.0
        target_gap_cm = width_cm - self.config.gripper_squeeze_cm
        grip_joint = (target_gap_cm - self.config.gripper_intercept) / self.config.gripper_slope
        grip_joint = max(-1.5, min(0.0, grip_joint))
        self.logger.log(f"Gripping: width={width_cm:.1f}cm, gap={target_gap_cm:.1f}cm, joint={grip_joint:.3f}rad")
        return self.move_gripper(grip_joint)
    
    def open_gripper(self) -> bool:
        return self.move_gripper(-1.5)
    
    # ========================================================================
    # CAMERA
    # ========================================================================
    
    def get_current_joints(self) -> list[float]:
        if self.current_joint_state is None:
            return []
        return list(self.current_joint_state.position[:5])
    
    def capture_camera_image(self):
        """Capture color image. Returns numpy BGR array or None."""
        from rosbags.image import message_to_cvimage
        try:
            msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=5.0)
            return message_to_cvimage(msg, 'bgr8')
        except Exception as e:
            self.logger.log_error(f"Color capture failed: {e}")
            return None
    
    def capture_depth_image(self):
        """Capture depth image. Returns numpy uint16 array (mm) or None."""
        try:
            msg = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5.0)
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except Exception as e:
            self.logger.log_error(f"Depth capture failed: {e}")
            return None
