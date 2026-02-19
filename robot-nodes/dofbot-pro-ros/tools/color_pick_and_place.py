#!/usr/bin/env python3
"""
Color-based pick-and-place for DOFBOT Pro.
Uses touch_test7 pattern: moveit_commander for arm, action client for gripper.
"""

import rospy
import numpy as np
import cv2
import json
import sys
import actionlib
from pathlib import Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import moveit_commander
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal, Constraints, JointConstraint

# ============================================================================
# CONFIGURATION
# ============================================================================

COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([160, 50, 100]), 'upper': np.array([180, 255, 255])}
    ],
    'blue': {'lower': np.array([100, 150, 100]), 'upper': np.array([130, 255, 255])},
    'green': {'lower': np.array([40, 60, 60]), 'upper': np.array([90, 255, 255])},
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
}

APPROACH_HEIGHT = 0.10
LIFT_HEIGHT = 0.08
TABLE_Z = 0.003  # Table surface height in base_link frame (meters)

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 1000
DEPTH_SAMPLE_RADIUS = 5

PLACE_LOCATIONS = {
    'red': [0.15, -0.10, 0.02],
    'blue': [0.15, -0.05, 0.02],
    'green': [0.15, 0.0, 0.02],
    'yellow': [0.15, 0.05, 0.02]
}

# ============================================================================
# MAIN CLASS
# ============================================================================

class ColorPickAndPlace:
    def __init__(self, visualize_only=False):
        rospy.init_node('color_pick_and_place', anonymous=True)
        self.visualize_only = visualize_only
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None

        # Load camera intrinsics
        intrinsics_path = Path.home() / "medal-lab/robot-nodes/dofbot-pro-ros/calibration/camera_intrinsics.json"
        with open(intrinsics_path) as f:
            calib = json.load(f)
        self.fx = calib['camera_matrix'][0][0]
        self.fy = calib['camera_matrix'][1][1]
        self.cx = calib['camera_matrix'][0][2]
        self.cy = calib['camera_matrix'][1][2]
        rospy.loginfo(f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        if not visualize_only:
            # Arm: moveit_commander (same as touch_test7)
            moveit_commander.roscpp_initialize(sys.argv)
            self.arm = moveit_commander.MoveGroupCommander("arm_group")
            self.arm.allow_replanning(True)
            self.arm.set_planning_time(10)
            self.arm.set_num_planning_attempts(20)
            self.arm.set_goal_position_tolerance(0.005)
            self.arm.set_goal_orientation_tolerance(3.14)
            self.arm.set_max_velocity_scaling_factor(0.4)
            self.arm.set_max_acceleration_scaling_factor(0.4)
            rospy.loginfo(f"End effector link: {self.arm.get_end_effector_link()}")

            # Gripper: action client (same as touch_test7)
            self.gripper_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
            self.gripper_client.wait_for_server()

            # TF
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Image subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self._color_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self._depth_cb)

        rospy.loginfo("Waiting for images...")
        rospy.sleep(2.0)
        if self.color_image is None or self.depth_image is None:
            rospy.logerr("Failed to receive images!")
            sys.exit(1)

        rospy.loginfo(f"Initialized in {'visualization' if visualize_only else 'pick-and-place'} mode")

    def _color_cb(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # ========================================================================
    # MOTION (touch_test7 pattern)
    # ========================================================================

    def move_to_position(self, x, y, z):
        """Move arm to position (moveit_commander, same as touch_test7)."""
        self.arm.set_position_target([x, y, z])
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        if ok:
            rospy.sleep(1.0)
        return ok

    def grip_position_for_width(self, width_m):
        """Convert object width (meters) to grip_joint position with grip pressure."""
        width_cm = width_m * 100.0
        target_gap_cm = width_cm - 1.0  # Close 1cm tighter than object to grip
        # Linear fit from calibration: gap_cm = -3.9318 * grip_joint + 0.7795
        pos = (target_gap_cm - 0.7795) / -3.9318
        return max(-1.5, min(0.0, pos))

    def control_gripper(self, close=True, width_m=None):
        """Control gripper via action client. If width_m provided, closes to fit object."""
        gg = MoveGroupGoal()
        gg.request.group_name = "grip_group"
        gg.request.num_planning_attempts = 3
        gg.request.allowed_planning_time = 3.0

        jc = JointConstraint()
        jc.joint_name = "grip_joint"
        if close and width_m is not None:
            jc.position = self.grip_position_for_width(width_m)
            rospy.loginfo(f"Gripper closing to {jc.position:.3f} rad for {width_m*100:.1f}cm object")
        elif close:
            jc.position = -0.2  # Default gentle close
        else:
            jc.position = -1.5
        jc.tolerance_above = 0.1
        jc.tolerance_below = 0.1
        jc.weight = 1.0

        cons = Constraints()
        cons.joint_constraints.append(jc)
        gg.request.goal_constraints.append(cons)
        gg.planning_options.plan_only = False

        self.gripper_client.send_goal(gg)
        self.gripper_client.wait_for_result(rospy.Duration(5))
        rospy.sleep(1.0)

    # ========================================================================
    # DETECTION
    # ========================================================================

    def detect_blocks(self, color_name):
        """Detect all blocks of specified color."""
        if self.color_image is None or self.depth_image is None:
            return []

        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

        ranges = COLOR_RANGES[color_name]
        if not isinstance(ranges, list):
            ranges = [ranges]

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            mask |= cv2.inRange(hsv, r['lower'], r['upper'])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue

                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                pos_3d = self.pixel_to_3d(cx, cy)
                if pos_3d is not None:
                    depth_m = np.median(self.depth_image[max(0,cy-DEPTH_SAMPLE_RADIUS):cy+DEPTH_SAMPLE_RADIUS+1,
                                                          max(0,cx-DEPTH_SAMPLE_RADIUS):cx+DEPTH_SAMPLE_RADIUS+1]
                                        [self.depth_image[max(0,cy-DEPTH_SAMPLE_RADIUS):cy+DEPTH_SAMPLE_RADIUS+1,
                                                          max(0,cx-DEPTH_SAMPLE_RADIUS):cx+DEPTH_SAMPLE_RADIUS+1] > 0]) / 1000.0
                    # boundingRect width (axis-aligned, inflated by rotation)
                    br_width = w * depth_m / self.fx
                    # minAreaRect width (rotation-invariant, tighter fit)
                    rect = cv2.minAreaRect(contour)
                    mar_w = min(rect[1])  # Shorter side = width for near-square objects
                    mar_width = mar_w * depth_m / self.fx
                    rospy.loginfo(f"  boundingRect width: {br_width*100:.1f}cm | minAreaRect width: {mar_width*100:.1f}cm")
                    if mar_width < 0.02:  # Skip objects smaller than 2cm
                        rospy.loginfo(f"  Skipping: too small ({mar_width*100:.1f}cm < 2.0cm)")
                        continue
                    blocks.append({'pixel': (cx, cy), 'position': pos_3d, 'area': area, 'width': mar_width})
                    rospy.loginfo(f"Detected {color_name} block at pixel ({cx}, {cy}) -> 3D: {pos_3d}, width: {mar_width*100:.1f}cm")

        return blocks

    def pixel_to_3d(self, px, py):
        """Convert pixel to 3D position in base_link frame."""
        y_min = max(0, py - DEPTH_SAMPLE_RADIUS)
        y_max = min(self.depth_image.shape[0], py + DEPTH_SAMPLE_RADIUS + 1)
        x_min = max(0, px - DEPTH_SAMPLE_RADIUS)
        x_max = min(self.depth_image.shape[1], px + DEPTH_SAMPLE_RADIUS + 1)

        depth_region = self.depth_image[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0]

        if len(valid_depths) == 0:
            rospy.logwarn(f"No valid depth at pixel ({px}, {py})")
            return None

        depth_m = np.median(valid_depths) / 1000.0

        x_cam = (px - self.cx) * depth_m / self.fx
        y_cam = (py - self.cy) * depth_m / self.fy
        z_cam = depth_m

        if self.visualize_only:
            return [x_cam, y_cam, z_cam]

        point_cam = PointStamped()
        point_cam.header.frame_id = "camera_color_optical_frame"
        point_cam.header.stamp = rospy.Time(0)
        point_cam.point.x = x_cam
        point_cam.point.y = y_cam
        point_cam.point.z = z_cam

        try:
            point_base = self.tf_buffer.transform(point_cam, "base_link", rospy.Duration(1.0))
            return [point_base.point.x, point_base.point.y, point_base.point.z]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF transform failed: {e}")
            return None

    # ========================================================================
    # PICK AND PLACE
    # ========================================================================

    def pick(self, position, width_m=None):
        x, y, z = position
        grasp_z = (TABLE_Z + z) / 2.0  # Centroid of block

        if grasp_z < TABLE_Z:
            rospy.logerr(f"Grasp z={grasp_z:.4f}m is below table z={TABLE_Z:.4f}m, skipping")
            return False

        rospy.loginfo(f"Picking at ({x:.3f}, {y:.3f}, {z:.3f}), grasp_z={grasp_z:.4f}m")

        rospy.loginfo("Opening gripper...")
        self.control_gripper(close=False)

        rospy.loginfo("Moving above target...")
        if not self.move_to_position(x, y, z + APPROACH_HEIGHT):
            rospy.logerr("Failed to move above target")
            return False

        rospy.loginfo("Lowering to grasp...")
        if not self.move_to_position(x, y, grasp_z):
            rospy.logerr("Failed to lower to grasp")
            return False

        rospy.sleep(0.5)  # Pause to let block settle between fingers

        rospy.loginfo("Closing gripper...")
        self.control_gripper(close=True, width_m=width_m)

        rospy.loginfo("Lifting...")
        if not self.move_to_position(x, y, z + LIFT_HEIGHT):
            rospy.logerr("Failed to lift")
            return False

        rospy.loginfo("Pick complete")
        return True

    def place(self, position):
        x, y, z = position
        rospy.loginfo(f"Placing at ({x:.3f}, {y:.3f}, {z:.3f})")

        if not self.move_to_position(x, y, z + APPROACH_HEIGHT):
            rospy.logerr("Failed to move above place")
            return False

        if not self.move_to_position(x, y, z + 0.01):
            rospy.logerr("Failed to lower to place")
            return False

        rospy.loginfo("Opening gripper...")
        self.control_gripper(close=False)

        rospy.loginfo("Lifting...")
        if not self.move_to_position(x, y, z + LIFT_HEIGHT):
            rospy.logerr("Failed to lift after place")
            return False

        rospy.loginfo("Place complete")
        return True

    def run(self, color_name):
        rospy.loginfo(f"=== Starting pick-and-place for {color_name} blocks ===")

        blocks = self.detect_blocks(color_name)
        if not blocks:
            rospy.logwarn(f"No {color_name} blocks detected")
            return

        rospy.loginfo(f"Found {len(blocks)} {color_name} block(s)")

        place_pos = PLACE_LOCATIONS.get(color_name)
        if place_pos is None:
            rospy.logerr(f"No place location defined for {color_name}")
            return

        for i, block in enumerate(blocks):
            rospy.loginfo(f"\n--- Block {i+1}/{len(blocks)} ---")

            if not self.pick(block['position'], width_m=block.get('width')):
                rospy.logerr(f"Failed to pick block {i+1}")
                continue

            place_offset = [place_pos[0], place_pos[1] + i * 0.04, place_pos[2]]
            if not self.place(place_offset):
                rospy.logerr(f"Failed to place block {i+1}")
                continue

            rospy.loginfo(f"Block {i+1} complete!")

        rospy.loginfo(f"=== Finished {color_name} blocks ===\n")

        rospy.loginfo("Returning to init position...")
        self.arm.set_named_target("init")
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

    def visualize_detection(self, color_name):
        if self.color_image is None:
            return

        blocks = self.detect_blocks(color_name)

        vis = self.color_image.copy()
        for block in blocks:
            px, py = block['pixel']
            cv2.circle(vis, (px, py), 10, (0, 255, 0), 2)
            cv2.putText(vis, color_name, (px + 15, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(f"{color_name} Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 color_pick_and_place.py <color> [--visualize]")
        print("Colors: red, blue, green, yellow")
        sys.exit(1)

    color = sys.argv[1].lower()
    visualize = '--visualize' in sys.argv

    if color not in COLOR_RANGES:
        print(f"Unknown color: {color}. Available: {list(COLOR_RANGES.keys())}")
        sys.exit(1)

    picker = ColorPickAndPlace(visualize_only=visualize)

    if visualize:
        picker.visualize_detection(color)
    else:
        picker.run(color)

if __name__ == '__main__':
    main()
