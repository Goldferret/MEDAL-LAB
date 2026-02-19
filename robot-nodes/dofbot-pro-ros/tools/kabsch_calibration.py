#!/usr/bin/env python3
"""
Kabsch (translation-only) Hand-Eye Calibration

Solves for T_base_cam using only the TRANSLATION component of ArUco detections,
bypassing unreliable ArUco rotation estimates that corrupt cv2.calibrateHandEye.

Background:
  Single planar ArUco markers have unreliable rotation estimates, especially at
  oblique viewing angles or low resolution (e.g. 70mm marker at 640x480). The
  translation (distance/position) is reliable since it's based on marker size +
  triangulation. Standard calibrateHandEye uses both rotation and translation,
  so bad rotations corrupt the entire solve.

  This script uses the Kabsch algorithm (rigid body point-cloud registration)
  which only needs corresponding 3D point pairs:
    - Gripper position in base frame (from forward kinematics)
    - Marker position in camera frame (ArUco translation only)

  The relationship is: p_cam = R_cam_base @ p_base + t_cam_base
  Kabsch finds the optimal R and t minimizing sum of squared residuals.

Requirements:
  - easy_handeye node running with samples already collected
  - Samples use eye-on-base convention (easy_handeye feeds inverted robot transforms)

Usage:
  1. Collect samples normally via easy_handeye (moveit_calibration.py or GUI)
  2. Run: python3 kabsch_calibration.py [--save] [--namespace /dofbot_eye_to_hand_eye_on_base/]
"""

import argparse
import numpy as np
import yaml
import subprocess
from scipy.spatial.transform import Rotation as R


def to_T(sample):
    """Convert easy_handeye sample (translation + quaternion) to 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat([
        sample["rotation"]["x"], sample["rotation"]["y"],
        sample["rotation"]["z"], sample["rotation"]["w"]
    ]).as_matrix()
    T[:3, 3] = [sample["translation"]["x"], sample["translation"]["y"], sample["translation"]["z"]]
    return T


def kabsch(A, B):
    """
    Kabsch algorithm: find R, t such that B ≈ R @ A + t (least-squares).
    A: Nx3 source points, B: Nx3 target points.
    Returns: R (3x3 rotation), t (3, translation)
    """
    cA, cB = np.mean(A, axis=0), np.mean(B, axis=0)
    H = (A - cA).T @ (B - cB)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    Rot = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cB - Rot @ cA
    return Rot, t


def main():
    parser = argparse.ArgumentParser(description="Kabsch hand-eye calibration")
    parser.add_argument("--save", action="store_true", help="Save result to easy_handeye yaml")
    parser.add_argument("--namespace", default="/dofbot_eye_to_hand_eye_on_base/",
                        help="easy_handeye namespace")
    args = parser.parse_args()

    # Get samples from easy_handeye
    result = subprocess.run(
        ["rosservice", "call", args.namespace + "get_sample_list"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to get samples: {result.stderr}")
        return

    data = yaml.safe_load(result.stdout)
    hw = data["samples"]["hand_world_samples"]
    cm = data["samples"]["camera_marker_samples"]
    print(f"Loaded {len(hw)} samples")

    if len(hw) < 3:
        print("Need at least 3 samples")
        return

    # Extract point correspondences
    # hw[i] = T_gripper_base (easy_handeye inverted robot chain for eye-on-base)
    # cm[i] = T_camera_marker (camera tracking_base_frame -> marker)
    p_base, p_cam = [], []
    for i in range(len(hw)):
        T_bg = np.linalg.inv(to_T(hw[i]))  # T_base_gripper
        p_base.append(T_bg[:3, 3])
        p_cam.append([cm[i]["translation"]["x"], cm[i]["translation"]["y"], cm[i]["translation"]["z"]])

    p_base, p_cam = np.array(p_base), np.array(p_cam)

    # Solve: p_cam = R_cam_base @ p_base + t_cam_base
    Rot_cb, t_cb = kabsch(p_base, p_cam)
    T_cam_base = np.eye(4)
    T_cam_base[:3, :3] = Rot_cb
    T_cam_base[:3, 3] = t_cb
    T_base_cam = np.linalg.inv(T_cam_base)

    # Results
    q = R.from_matrix(T_base_cam[:3, :3]).as_quat()  # xyzw
    rpy = R.from_matrix(T_base_cam[:3, :3]).as_euler("xyz", degrees=True)
    cam_z = T_base_cam[:3, :3] @ np.array([0, 0, 1])
    angle_from_down = np.degrees(np.arccos(np.clip(-cam_z[2], -1, 1)))

    print(f"\n=== T_base_cam ===")
    print(f"Translation: x={T_base_cam[0,3]:.4f}  y={T_base_cam[1,3]:.4f}  z={T_base_cam[2,3]:.4f}")
    print(f"Quaternion (xyzw): {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    print(f"RPY (deg): roll={rpy[0]:.1f}  pitch={rpy[1]:.1f}  yaw={rpy[2]:.1f}")
    print(f"Camera z-axis angle from straight down: {angle_from_down:.1f}°")

    # Per-sample residuals
    print(f"\n=== Residuals ===")
    residuals = [np.linalg.norm((Rot_cb @ p_base[i] + t_cb) - p_cam[i]) for i in range(len(p_base))]
    for i, r in enumerate(residuals):
        print(f"  S{i+1:2d}: {r*1000:.1f}mm")
    print(f"  Mean: {np.mean(residuals)*1000:.1f}mm  Median: {np.median(residuals)*1000:.1f}mm")

    if args.save:
        # Save to calibration directory
        save_path = "/home/jetson/medal-lab/robot-nodes/dofbot-pro-ros/calibration/hand_eye_transform.yaml"
        cal = {
            "parameters": {
                "eye_on_hand": False,
                "robot_base_frame": "base_link",
                "robot_effector_frame": "Gripping_point_Link",
                "tracking_base_frame": "camera_link",
                "tracking_marker_frame": "aruco_marker_frame"
            },
            "transformation": {
                "x": float(T_base_cam[0, 3]),
                "y": float(T_base_cam[1, 3]),
                "z": float(T_base_cam[2, 3]),
                "qx": float(q[0]),
                "qy": float(q[1]),
                "qz": float(q[2]),
                "qw": float(q[3]),
            }
        }
        with open(save_path, "w") as f:
            yaml.dump(cal, f, default_flow_style=False)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
