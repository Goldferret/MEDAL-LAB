#!/usr/bin/env python3
"""
Phase 2: Hand-Eye Calibration for DOFBOT Pro + Orbbec DCW2

Manual physical positioning with 70mm ArUco marker mounted at gripper TCP.
User positions robot by hand, script captures synchronized image + FK pose pairs,
then solves hand-eye calibration using Daniilidis method.

Usage:
    python tools/hand_eye_calibration.py

Output:
    calibration/hand_eye_transform.json - Camera-to-base transformation matrix
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from dofbot_pro_info.srv import kinemarics, kinemaricsRequest
from scipy.spatial.transform import Rotation

# Import arm controller for torque control
sys.path.append('/home/jetson/dofbot_ws-custom_scripts/scripts')
from arm_controller import ArmController

# Marker configuration
MARKER_ID = 0
MARKER_SIZE = 0.07  # 70mm in meters
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def load_intrinsic_calibration():
    """Load camera intrinsics from Phase 1."""
    try:
        with open("calibration/camera_intrinsics.json", "r") as f:
            data = json.load(f)
        
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["distortion_coefficients"])
        
        print(f"✓ Loaded intrinsics with RMS error: {data['rms_error']:.4f}")
        return camera_matrix, dist_coeffs
        
    except FileNotFoundError:
        raise ValueError("Phase 1 intrinsics not found. Run intrinsic_calibration.py first.")


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix.
    
    DOFBOT Pro Convention: Intrinsic ZYX rotations
    ===============================================
    The robot's FK service returns RPY in intrinsic ZYX convention, meaning:
    1. Yaw rotates around Z-axis
    2. Pitch rotates around the NEW Y-axis (after yaw rotation)
    3. Roll rotates around the NEW X-axis (after yaw+pitch rotations)
    
    This is also called "Tait-Bryan angles" in aerospace.
    
    Args:
        roll (float): Rotation around X-axis in radians
        pitch (float): Rotation around Y-axis in radians
        yaw (float): Rotation around Z-axis in radians
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=False).as_matrix()
    return R


def get_robot_tcp_pose(arm_controller):
    """Get robot end-effector pose using FK service with actual servo positions."""
    try:
        rospy.wait_for_service('/get_kinemarics', timeout=5.0)
        kinematics_service = rospy.ServiceProxy('/get_kinemarics', kinemarics)
        
        # Read actual servo positions from hardware
        print("  Reading servo positions...")
        current_state = arm_controller.read_current_state()
        
        # Extract joint positions (servos 1-5 are arm joints)
        # FK service expects DEGREES (not radians!)
        joint_positions = [
            current_state[1],
            current_state[2],
            current_state[3],
            current_state[4],
            current_state[5]
        ]
        
        print(f"  Joint angles (deg): [{joint_positions[0]:.1f}, {joint_positions[1]:.1f}, {joint_positions[2]:.1f}, {joint_positions[3]:.1f}, {joint_positions[4]:.1f}]")
        
        # Call FK service with degrees
        request = kinemaricsRequest()
        request.cur_joint1 = joint_positions[0]
        request.cur_joint2 = joint_positions[1]
        request.cur_joint3 = joint_positions[2]
        request.cur_joint4 = joint_positions[3]
        request.cur_joint5 = joint_positions[4]
        request.kin_name = "fk"
        
        response = kinematics_service(request)
        
        # Build transformation matrix
        T_base_to_tcp = np.eye(4)
        T_base_to_tcp[0, 3] = response.x
        T_base_to_tcp[1, 3] = response.y
        T_base_to_tcp[2, 3] = response.z
        
        # Convert RPY to rotation matrix
        R = euler_to_rotation_matrix(response.Roll, response.Pitch, response.Yaw)
        T_base_to_tcp[:3, :3] = R
        
        return T_base_to_tcp, response.x, response.y, response.z
        
    except Exception as e:
        print(f"❌ Failed to get robot pose: {e}")
        return None, None, None, None


def detect_aruco_marker(image, camera_matrix, dist_coeffs, marker_id=MARKER_ID, marker_size=MARKER_SIZE):
    """
    Detect 70mm ArUco marker and return 3D pose.
    
    Args:
        image: Input image (BGR)
        camera_matrix: Camera intrinsics from Phase 1
        dist_coeffs: Camera distortion
        marker_id: ArUco marker ID (default: 0)
        marker_size: Physical marker size in meters (default: 0.07 = 70mm)
    
    Returns:
        T_marker2cam: 4×4 transformation (marker→camera)
        corner_count: Always 4 for single marker
        msg: Status message
    """
    # Undistort image first
    image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    # Detect ArUco marker (OpenCV 4.2.0 API)
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(
        image_undistorted, aruco_dict, parameters=detector_params
    )
    
    if ids is None:
        return None, 0, "No ArUco markers detected"
    
    # Find requested marker ID
    marker_index = None
    for i, detected_id in enumerate(ids.flatten()):
        if detected_id == marker_id:
            marker_index = i
            break
    
    if marker_index is None:
        detected_ids = ids.flatten().tolist()
        return None, 0, f"Marker ID {marker_id} not found (detected: {detected_ids})"
    
    # Estimate 3D pose
    marker_corners = corners[marker_index]
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        marker_corners.reshape(1, 4, 2),
        marker_size,
        camera_matrix,
        dist_coeffs
    )
    
    # Convert to transformation matrix
    R, _ = cv2.Rodrigues(rvec[0])
    T_marker2cam = np.eye(4)
    T_marker2cam[:3, :3] = R
    T_marker2cam[:3, 3] = tvec[0].flatten()
    
    return T_marker2cam, 4, "Marker detected (4 corners)"


def collect_calibration_data_manual(camera_matrix, dist_coeffs, arm_controller):
    """
    Collect hand-eye calibration data using manual physical positioning.
    User positions robot by hand, script captures data with torque control.
    """
    rospy.init_node('hand_eye_calibration', anonymous=True)
    bridge = CvBridge()
    
    robot_poses = []
    marker_poses = []
    pose_stats = {
        "total_attempted": 0,
        "successful": 0,
        "failed": 0,
        "failed_poses": []
    }
    
    print("\n" + "="*70)
    print("HAND-EYE CALIBRATION: 70mm ArUco Marker at TCP")
    print("="*70)
    print("\n⚠️  MARKER MOUNTED AT GRIPPER TCP")
    print("   - Marker is centered on gripper closing point")
    print("   - Gripper must stay fully closed during calibration")
    print("   - Wrist rotation (joint 5) is now ENCOURAGED for diversity")
    print()
    print("Workflow for each pose:")
    print("  1. Robot torque will be DISABLED (robot goes limp)")
    print("  2. Physically move robot to described position")
    print("  3. Press Enter #1 → Joints LOCK")
    print("  4. Wait for servos to settle (~1 second)")
    print("  5. Press Enter #2 → Image CAPTURED")
    print("  6. Script validates detection and saves data")
    print()
    
    # Define 20 calibration poses with wrist rotation
    poses = [
        # GROUP 1: PITCH VARIATIONS (3 poses)
        ("PITCH NEUTRAL", "Gripper forward level, wrist center rotation"),
        ("PITCH DOWN 20°", "Tilt wrist down 20°, wrist rotate left 30°"),
        ("PITCH UP 20°", "Tilt wrist up 20°, wrist rotate right 30°"),
        
        # GROUP 2: BASE ROTATION VARIATIONS (3 poses)
        ("BASE LEFT 30°", "Rotate base left 30°, wrist neutral"),
        ("BASE CENTER", "Base centered, arm forward, wrist center"),
        ("BASE RIGHT 30°", "Rotate base right 30°, wrist neutral"),
        
        # GROUP 3: HEIGHT VARIATIONS (3 poses)
        ("HEIGHT LOW", "Lower arm to 15cm above table, wrist rotate 45°"),
        ("HEIGHT MID", "Mid-height 25cm above table, wrist rotate -45°"),
        ("HEIGHT HIGH", "Raise arm to 35cm, wrist center"),
        
        # GROUP 4: WRIST ROTATION EMPHASIS (3 poses)
        ("WRIST LEFT 60°", "Keep arm level, rotate wrist left 60°"),
        ("WRIST CENTER", "Wrist at 0° center rotation"),
        ("WRIST RIGHT 60°", "Keep arm level, rotate wrist right 60°"),
        
        # GROUP 5: COMBINED POSES (8 poses)
        ("COMBO 1", "Base left + wrist right + pitch down"),
        ("COMBO 2", "Base right + wrist left + pitch up"),
        ("COMBO 3", "Left side low height, wrist 45°"),
        ("COMBO 4", "Right side high height, wrist -45°"),
        ("COMBO 5", "Forward diagonal, mid height, wrist center"),
        ("COMBO 6", "Back diagonal, pitch up, wrist rotate"),
        ("COMBO 7", "Extreme left, low height, wrist 60°"),
        ("COMBO 8", "Extreme right, high height, wrist -60°"),
    ]
    
    for pose_id, (pose_name, description) in enumerate(poses, start=1):
        print("\n" + "="*70)
        print(f"POSE {pose_id}/20: {pose_name}")
        print("="*70)
        print(f"Description: {description}")
        print("⚠️  Remember: Keep gripper fully closed")
        print()
        
        pose_stats["total_attempted"] += 1
        
        # Step 1: Disable torque
        print("Disabling torque...")
        arm_controller.set_torque(False)
        print("✓ Robot unlocked - move to position")
        
        input(f"\nPress ENTER when robot is positioned for '{pose_name}'...")
        
        # Step 2: Enable torque and lock
        print("\nLocking joints...")
        arm_controller.set_torque(True)
        time.sleep(1.0)  # Servo settling time
        print("✓ Joints locked and settled")
        
        input("Press ENTER to capture image and robot pose...")
        
        # Step 3: Get robot TCP pose
        print("\nReading robot pose...")
        tcp_pose, x, y, z = get_robot_tcp_pose(arm_controller)
        if tcp_pose is None:
            print("❌ FK service failed")
            retry = input("Retry this pose? (y/n): ")
            if retry.lower() == 'y':
                pose_id -= 1
                pose_stats["total_attempted"] -= 1
                continue
            else:
                pose_stats["failed"] += 1
                pose_stats["failed_poses"].append(f"{pose_id}: {pose_name} (FK failed)")
                continue
        
        # Step 4: Capture image
        print("Capturing image...")
        try:
            ros_image = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=3.0)
            cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except Exception as e:
            print(f"❌ Failed to capture image: {e}")
            retry = input("Retry this pose? (y/n): ")
            if retry.lower() == 'y':
                pose_id -= 1
                pose_stats["total_attempted"] -= 1
                continue
            else:
                pose_stats["failed"] += 1
                pose_stats["failed_poses"].append(f"{pose_id}: {pose_name} (Image capture failed)")
                continue
        
        # Step 5: Detect ArUco marker
        print("Detecting ArUco marker...")
        marker_pose, corner_count, msg = detect_aruco_marker(
            cv_image, camera_matrix, dist_coeffs
        )
        
        if marker_pose is None:
            print(f"❌ {msg}")
            print("  Suggestions:")
            print("  - Ensure marker is fully visible in camera view")
            print("  - Check lighting (avoid shadows/glare)")
            print("  - Verify marker is not edge-on to camera")
            retry = input("Retry this pose? (y/n): ")
            if retry.lower() == 'y':
                pose_id -= 1
                pose_stats["total_attempted"] -= 1
                continue
            else:
                pose_stats["failed"] += 1
                pose_stats["failed_poses"].append(f"{pose_id}: {pose_name} (Detection failed)")
                continue
        
        # Step 6: Store successful capture
        robot_poses.append(tcp_pose)
        marker_poses.append(marker_pose)
        pose_stats["successful"] += 1
        
        print(f"✓ Pose {pose_id} captured successfully!")
        print(f"  TCP position: [{x:.3f}, {y:.3f}, {z:.3f}] m")
        print(f"  Marker corners: {corner_count}")
        print(f"  Progress: {len(robot_poses)}/20 poses collected")
    
    # Print collection summary
    print("\n" + "="*70)
    print("DATA COLLECTION SUMMARY")
    print("="*70)
    print(f"Total poses attempted: {pose_stats['total_attempted']}")
    print(f"Successful captures: {pose_stats['successful']}")
    print(f"Failed captures: {pose_stats['failed']}")
    if pose_stats['failed_poses']:
        print("\nFailed poses:")
        for failed in pose_stats['failed_poses']:
            print(f"  - {failed}")
    print("="*70)
    
    return robot_poses, marker_poses


def solve_hand_eye_calibration(robot_poses, marker_poses):
    """
    Solve AX=XB hand-eye calibration using Daniilidis dual quaternion method.
    
    COORDINATE FRAME CONVENTIONS:
    - robot_poses: 4x4 matrices, Base→TCP (URDF passive convention: p_tcp = T @ p_base)
    - marker_poses: 4x4 matrices, Marker→Camera (from ArUco detection)
    - R_gripper2base/t_gripper2base: TCP→Base (active convention: p_base = R @ p_tcp + t)
    - R_target2cam/t_target2cam: Marker→Camera
    
    EYE-TO-HAND CALIBRATION (camera fixed overhead):
    - FK returns Base→TCP in URDF passive convention
    - OpenCV expects TCP→Base in active convention (gripper→base)
    - We invert FK output to convert between conventions
    
    Returns: Camera-to-base transformation
    """
    if len(robot_poses) < 3:
        raise ValueError("Insufficient data for calibration (minimum 3 poses required)")
    
    print(f"\nSolving hand-eye calibration with {len(robot_poses)} poses...")
    
    # Extract rotation and translation components
    R_gripper2base = []    # TCP→Base (active convention for OpenCV)
    t_gripper2base = []
    R_target2cam = []      # Marker→Camera (from ArUco)
    t_target2cam = []
    
    for robot_pose, marker_pose in zip(robot_poses, marker_poses):
        # FK returns Base→TCP (URDF passive convention)
        # OpenCV needs TCP→Base (active convention)
        T_base2tcp = robot_pose
        T_tcp2base = np.linalg.inv(T_base2tcp)
        
        R_gripper2base.append(T_tcp2base[:3, :3])
        t_gripper2base.append(T_tcp2base[:3, 3])
        
        # Marker-to-camera is correct as-is
        R_target2cam.append(marker_pose[:3, :3])
        t_target2cam.append(marker_pose[:3, 3])
    
    # Solve hand-eye calibration using Daniilidis method
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    
    # Construct transformation matrix
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()
    
    print("✓ Hand-eye calibration solved (Daniilidis method)")
    
    # Validate result with sanity checks
    translation = T_cam2base[:3, 3]
    distance = np.linalg.norm(translation)
    
    print(f"\n📊 Calibration Result:")
    print(f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] m")
    print(f"  Distance: {distance:.3f} m")
    
    # Sanity checks
    if abs(distance - 0.51) < 0.10:
        print(f"  ✅ Magnitude reasonable (~0.51m)")
    else:
        print(f"  ⚠️  Magnitude unexpected (expected ~0.51m, got {distance:.3f}m)")
    
    if translation[2] > 0:
        print(f"  ✅ Z-sign correct (camera above base, +Z is up)")
    else:
        print(f"  ⚠️  Z-sign unexpected (expected positive Z for camera above base)")
    
    return T_cam2base


def compute_reprojection_error(robot_poses, marker_poses, T_cam2base):
    """
    Compute reprojection error with marker at TCP (no offset estimation needed).
    
    Conventions:
      - robot_poses[i]: T_base2tcp_i (Base→TCP from FK, URDF passive convention)
      - marker_poses[i]: T_marker2cam_i (Marker→Camera from ArUco)
      - T_cam2base: Camera→Base from hand-eye calibration
    
    Since marker is mounted at TCP, prediction chain is:
      marker_in_base = T_base2tcp @ marker_in_tcp
      marker_in_cam = inv(T_cam2base) @ marker_in_base
    
    Therefore:
      T_marker2cam_pred = inv(T_cam2base) @ T_base2tcp
    
    Compare with actual:
      error = || T_marker2cam_pred[:3,3] - T_marker2cam_actual[:3,3] ||
    """
    print("\nComputing reprojection error...")
    
    errors_mm = []
    
    for i, (T_base2tcp, T_marker2cam_actual) in enumerate(zip(robot_poses, marker_poses)):
        # FK returns Base→TCP (URDF passive convention)
        # Prediction chain: marker→base→camera
        T_marker2cam_pred = np.linalg.inv(T_cam2base) @ T_base2tcp
        
        # Debug output for first pose
        if i == 0:
            print(f"\n  [DEBUG] Pose 0:")
            print(f"    Actual   t_marker2cam: {T_marker2cam_actual[:3, 3]}")
            print(f"    Predicted t_marker2cam: {T_marker2cam_pred[:3, 3]}")
            print(f"    Error (mm): {np.linalg.norm(T_marker2cam_pred[:3, 3] - T_marker2cam_actual[:3, 3]) * 1000:.2f}")
        
        # Compute position error (no offset to estimate)
        err_mm = np.linalg.norm(
            T_marker2cam_pred[:3, 3] - T_marker2cam_actual[:3, 3]
        ) * 1000
        errors_mm.append(err_mm)
    
    mean_error = float(np.mean(errors_mm))
    max_error = float(np.max(errors_mm))
    
    print("✓ Reprojection error computed")
    print(f"  Mean error: {mean_error:.2f} mm")
    print(f"  Max error: {max_error:.2f} mm")
    
    # Quality rating
    if mean_error < 5.0:
        quality = "EXCELLENT"
        quality_icon = "🟢"
    elif mean_error < 10.0:
        quality = "GOOD"
        quality_icon = "🟢"
    elif mean_error < 15.0:
        quality = "ACCEPTABLE"
        quality_icon = "🟡"
    else:
        quality = "POOR - Consider recalibrating"
        quality_icon = "🔴"
    
    print(f"  Quality: {quality_icon} {quality}")
    
    return mean_error, max_error, quality


def save_hand_eye_results(T_cam2base, mean_error, max_error, quality, num_poses):
    """Save hand-eye calibration results to JSON."""
    os.makedirs("calibration", exist_ok=True)
    
    calibration_data = {
        "camera_to_base_transform": T_cam2base.tolist(),
        "calibration_info": {
            "method": "daniilidis_dual_quaternion",
            "num_poses": num_poses,
            "collection_method": "manual_physical_positioning",
            "target_type": "aruco_marker_at_tcp",
            "marker_parameters": {
                "marker_id": MARKER_ID,
                "marker_size_mm": int(MARKER_SIZE * 1000),
                "dictionary": "DICT_4X4_50",
                "mounting": "centered_at_gripper_tcp"
            }
        },
        "quality_metrics": {
            "mean_reprojection_error_mm": float(mean_error),
            "max_reprojection_error_mm": float(max_error),
            "quality_rating": quality
        }
    }
    
    with open("calibration/hand_eye_transform.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\n✓ Hand-eye calibration saved to calibration/hand_eye_transform.json")
    return calibration_data


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("DOFBOT Pro Phase 2: Hand-Eye Calibration")
        print("70mm ArUco Marker at TCP (Manual Positioning)")
        print("="*70)
        
        # Initialize arm controller for torque control
        print("\nInitializing arm controller...")
        arm_controller = ArmController()
        print("✓ Arm controller initialized")
        
        # Load intrinsics from Phase 1
        camera_matrix, dist_coeffs = load_intrinsic_calibration()
        
        print(f"✓ ArUco marker parameters: ID {MARKER_ID}, Size {int(MARKER_SIZE*1000)}mm")
        
        # Collect calibration data via manual positioning
        print("\nStarting data collection...")
        robot_poses, marker_poses = collect_calibration_data_manual(
            camera_matrix, dist_coeffs, arm_controller
        )
        
        if len(robot_poses) < 8:
            print(f"\n❌ Insufficient data for calibration (minimum 8 poses required, got {len(robot_poses)})")
            print("Please run the script again and collect more poses.")
            exit(1)
        
        # Solve hand-eye calibration
        T_cam2base = solve_hand_eye_calibration(robot_poses, marker_poses)
        
        # Compute reprojection error for validation
        mean_error, max_error, quality = compute_reprojection_error(
            robot_poses, marker_poses, T_cam2base
        )
        
        # Save results
        save_hand_eye_results(T_cam2base, mean_error, max_error, quality, len(robot_poses))
        
        # Re-enable torque before exiting
        print("\nRe-enabling torque...")
        arm_controller.set_torque(True)
        print("✓ Torque enabled")
        
        print("\n" + "="*70)
        print("🎉 Phase 2: Hand-Eye Calibration Completed Successfully!")
        print("="*70)
        print(f"\nCalibration Quality: {quality}")
        print(f"Mean Error: {mean_error:.2f} mm")
        print(f"Max Error: {max_error:.2f} mm")
        print(f"\nExpected: Mean error <10mm for good calibration")
        print("\nNext Steps:")
        print("1. Verify hand_eye_transform.json was created")
        print("2. Remove ArUco marker from gripper")
        print("3. Proceed to Phase 3: MADSci Integration")
        print("4. Run validation tests with actual grasping tasks")
        
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
        print("Re-enabling torque for safety...")
        try:
            arm_controller.set_torque(True)
        except:
            pass
        exit(0)
    except Exception as e:
        print(f"\n❌ Calibration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nRe-enabling torque for safety...")
        try:
            arm_controller.set_torque(True)
        except:
            pass
        exit(1)
