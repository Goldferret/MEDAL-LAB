#!/usr/bin/env python3
"""
MoveIt-based Calibration Script
Uses pre-collected poses and MoveIt action client to perform stable hand-eye calibration.
"""

import rospy
import json
import sys
import signal
import actionlib
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal, Constraints, JointConstraint
from sensor_msgs.msg import JointState
from easy_handeye_msgs.srv import TakeSample, ComputeCalibration, SetAlgorithm
from std_srvs.srv import Empty

class MoveItCalibrator:
    def __init__(self, poses_file, namespace='/dofbot_eye_to_hand_eye_on_base/'):
        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        rospy.init_node('moveit_calibrator', disable_signals=True)
        self.shutdown_requested = False
        
        # Load poses
        rospy.loginfo(f"Loading poses from {poses_file}")
        with open(poses_file, 'r') as f:
            data = json.load(f)
        
        self.poses = data['poses']
        rospy.loginfo(f"Loaded {len(self.poses)} poses")
        
        # Initialize MoveIt action client (same as dofbot_ros_interface.py)
        rospy.loginfo("Connecting to move_group action server...")
        self.move_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        self.move_client.wait_for_server()
        rospy.loginfo("Connected to move_group action server")
        
        # CRITICAL: Send gripper close command to update SimulationToMachine's internal state
        # This ensures when arm moves (5 joints), gripper stays at 180 instead of reverting to 30
        rospy.loginfo("Setting gripper to closed position (180) in SimulationToMachine...")
        gripper_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        gripper_client.wait_for_server()
        
        gripper_goal = MoveGroupGoal()
        gripper_goal.request.group_name = "grip_group"
        gripper_goal.request.num_planning_attempts = 3
        gripper_goal.request.allowed_planning_time = 3.0
        
        gripper_constraint = JointConstraint()
        gripper_constraint.joint_name = 'grip_joint'
        gripper_constraint.position = 0.0  # 0 radians = 180 servo = closed
        gripper_constraint.tolerance_above = 0.1
        gripper_constraint.tolerance_below = 0.1
        gripper_constraint.weight = 1.0
        
        gripper_constraints = Constraints()
        gripper_constraints.joint_constraints.append(gripper_constraint)
        gripper_goal.request.goal_constraints.append(gripper_constraints)
        gripper_goal.planning_options.plan_only = False
        
        gripper_client.send_goal(gripper_goal)
        gripper_client.wait_for_result(rospy.Duration(5.0))
        rospy.loginfo("Gripper closed - SimulationToMachine state updated")
        
        # Easy handeye services
        self.namespace = namespace
        rospy.loginfo("Waiting for easy_handeye services...")
        rospy.wait_for_service(namespace + 'take_sample')
        rospy.wait_for_service(namespace + 'compute_calibration')
        rospy.wait_for_service(namespace + 'save_calibration')
        rospy.wait_for_service(namespace + 'set_algorithm')
        
        self.take_sample_srv = rospy.ServiceProxy(namespace + 'take_sample', TakeSample)
        self.compute_srv = rospy.ServiceProxy(namespace + 'compute_calibration', ComputeCalibration)
        self.save_srv = rospy.ServiceProxy(namespace + 'save_calibration', Empty)
        
        # Switch to Park algorithm (Tsai-Lenz fails with eye-on-base inversion trick)
        set_alg = rospy.ServiceProxy(namespace + 'set_algorithm', SetAlgorithm)
        resp = set_alg('OpenCV/Park')
        rospy.loginfo(f"Set algorithm to OpenCV/Park: success={resp.success}")
        
        # Publisher to override arm_driver's noisy servo reads with commanded positions
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
        self.joint_names = ['Arm1_Joint', 'Arm2_Joint', 'Arm3_Joint', 'Arm4_Joint', 'Arm5_Joint', 'grip_joint']
        
        rospy.loginfo("All services ready!")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        rospy.loginfo("\n\nShutdown requested (Ctrl+C). Stopping calibration...")
        self.shutdown_requested = True
        if hasattr(self, 'move_client'):
            self.move_client.cancel_all_goals()
        sys.exit(0)
    
    def move_to_pose(self, joint_positions):
        """Move robot to specified joint configuration using MoveIt action client"""
        if self.shutdown_requested:
            return False
        
        rospy.loginfo(f"Planning move to: {[f'{j:.3f}' for j in joint_positions]}")
        
        try:
            # Create goal message (same as dofbot_ros_interface.py)
            goal = MoveGroupGoal()
            
            # Set planning group
            goal.request.group_name = "arm_group"
            
            # Set planning parameters
            goal.request.num_planning_attempts = 5
            goal.request.allowed_planning_time = 5.0
            goal.request.max_velocity_scaling_factor = 0.3
            goal.request.max_acceleration_scaling_factor = 0.3
            
            # Create joint constraints for goal
            constraints = Constraints()
            joint_names = ['Arm1_Joint', 'Arm2_Joint', 'Arm3_Joint', 'Arm4_Joint', 'Arm5_Joint']
            
            for name, position in zip(joint_names, joint_positions):
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = position
                constraint.tolerance_above = 0.01
                constraint.tolerance_below = 0.01
                constraint.weight = 1.0
                constraints.joint_constraints.append(constraint)
            
            goal.request.goal_constraints.append(constraints)
            
            # Set planning options
            goal.planning_options.plan_only = False  # Plan and execute
            
            # Send goal and wait for result
            rospy.loginfo("Executing movement...")
            self.move_client.send_goal(goal)
            
            # Wait for result with timeout
            finished = self.move_client.wait_for_result(rospy.Duration(30.0))
            
            if self.shutdown_requested:
                return False
            
            if not finished:
                rospy.logerr("Movement timed out after 30 seconds")
                return False
            
            # Check result
            state = self.move_client.get_state()
            
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Move completed successfully")
                return True
            else:
                rospy.logerr(f"Movement failed with state: {state}")
                return False
            
        except Exception as e:
            rospy.logerr(f"Movement failed: {e}")
            return False
    
    def take_sample(self):
        """Call easy_handeye TakeSample service"""
        if self.shutdown_requested:
            return False
        
        try:
            response = self.take_sample_srv()
            num_samples = len(response.samples.hand_world_samples)  # It's a list, not .transforms
            rospy.loginfo(f"Sample taken! Total samples: {num_samples}")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"TakeSample service failed: {e}")
            return False
    
    def compute_calibration(self):
        """Compute calibration from collected samples"""
        if self.shutdown_requested:
            return False
        
        rospy.loginfo("Computing calibration...")
        try:
            response = self.compute_srv()
            if response.valid:
                rospy.loginfo("Calibration computed successfully!")
                rospy.loginfo(f"Transform:\n{response.calibration}")
                return True
            else:
                rospy.logerr("Calibration computation failed!")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"ComputeCalibration service failed: {e}")
            return False
    
    def save_calibration(self):
        """Save calibration to file"""
        if self.shutdown_requested:
            return False
        
        rospy.loginfo("Saving calibration...")
        try:
            self.save_srv()
            rospy.loginfo("Calibration saved!")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"SaveCalibration service failed: {e}")
            return False
    
    def run_calibration(self):
        """Execute full calibration sequence"""
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("Starting MoveIt-based calibration")
        rospy.loginfo(f"Total poses to visit: {len(self.poses)}")
        rospy.loginfo("Press Ctrl+C at any time to stop")
        rospy.loginfo("="*60 + "\n")
        
        successful_samples = 0
        failed_samples = 0
        
        for i, pose_data in enumerate(self.poses):
            if self.shutdown_requested:
                rospy.loginfo("Calibration stopped by user")
                break
            
            rospy.loginfo(f"\n--- Pose {i+1}/{len(self.poses)} ---")
            
            # Get joint positions (only first 5 for arm, exclude gripper)
            joint_positions = pose_data['joint_positions'][:5]
            
            # Move to pose
            if not self.move_to_pose(joint_positions):
                if self.shutdown_requested:
                    break
                rospy.logwarn(f"Failed to move to pose {i+1}, skipping...")
                failed_samples += 1
                continue
            
            # Wait for robot to settle and TF to stabilize
            rospy.loginfo("Waiting 2 seconds for robot to settle...")
            rospy.sleep(2.0)
            
            if self.shutdown_requested:
                break
            
            # Publish commanded joint positions to override arm_driver's noisy servo reads
            js = JointState()
            js.header.stamp = rospy.Time.now()
            js.header.frame_id = "joint_states"
            js.name = self.joint_names
            # Use commanded arm positions + grip_joint closed (servo 180 = 1.5708 rad)
            js.position = list(joint_positions) + [1.5708]
            self.joint_pub.publish(js)
            rospy.loginfo(f"Published commanded joint_states: {[f'{p:.4f}' for p in js.position]}")
            rospy.sleep(0.5)  # Let robot_state_publisher update TF
            
            if self.shutdown_requested:
                break
            
            # Take sample
            if self.take_sample():
                successful_samples += 1
            else:
                rospy.logwarn(f"Failed to take sample at pose {i+1}")
                failed_samples += 1
        
        if self.shutdown_requested:
            rospy.loginfo("\nCalibration interrupted. Exiting...")
            return False
        
        # Summary
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("Sample collection complete!")
        rospy.loginfo(f"Successful: {successful_samples}")
        rospy.loginfo(f"Failed: {failed_samples}")
        rospy.loginfo("="*60 + "\n")
        
        if successful_samples < 10:
            rospy.logwarn("Less than 10 samples collected. Calibration may be poor quality.")
            try:
                proceed = input("Proceed with calibration anyway? (y/n): ").strip().lower()
                if proceed != 'y':
                    rospy.loginfo("Calibration aborted by user")
                    return False
            except (KeyboardInterrupt, EOFError):
                rospy.loginfo("\nCalibration aborted by user")
                return False
        
        # Compute calibration
        if not self.compute_calibration():
            return False
        
        # Save calibration
        try:
            save = input("\nSave calibration? (y/n): ").strip().lower()
            if save == 'y':
                return self.save_calibration()
            else:
                rospy.loginfo("Calibration not saved")
                return False
        except (KeyboardInterrupt, EOFError):
            rospy.loginfo("\nCalibration not saved")
            return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 moveit_calibration.py <poses_file.json>")
        print("Example: python3 moveit_calibration.py calibration_poses.json")
        sys.exit(1)
    
    poses_file = sys.argv[1]
    
    try:
        calibrator = MoveItCalibrator(poses_file)
        calibrator.run_calibration()
    except (KeyboardInterrupt, SystemExit):
        print("\nCalibration stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
