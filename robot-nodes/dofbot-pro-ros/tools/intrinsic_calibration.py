#!/usr/bin/env python3
"""
Phase 1: Intrinsic Camera Calibration for DOFBOT Pro + Orbbec DCW2

This script performs professional intrinsic calibration using ChArUco boards
to fix lens distortion across the entire camera field of view.

Hardware Requirements:
- ChArUco board: 5×7 grid, 35mm squares, 26mm markers, DICT_4X4_50
- Orbbec DCW2 camera at ~47cm height
- Rigid backing for ChArUco board (A4 paper + cardboard)

Usage:
    python tools/intrinsic_calibration.py

Output:
    calibration/camera_intrinsics.json - Camera matrix, distortion coefficients, quality metrics
"""

import cv2
import numpy as np
import json
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def create_charuco_board():
    """Create ChArUco board with actual printed dimensions."""
    # Actual printed dimensions measured from physical board
    squares_x, squares_y = 5, 7
    square_length = 0.036  # 36mm squares (measured)
    marker_length = 0.027  # 27mm markers (measured)
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, aruco_dict
    )
    return board, aruco_dict


def validate_image_quality(image, board, aruco_dict):
    """Validate image quality with industry-standard thresholds."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) < 4:  # Minimum markers for ChArUco
        return False, f"Too few markers detected: {len(corners)}"
    
    # 2. Interpolate ChArUco corners
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if not ret or len(charuco_corners) < 20:  # Minimum corners for good calibration
        return False, f"Insufficient ChArUco corners: {len(charuco_corners) if ret else 0}"
    
    # 3. Check image sharpness (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Empirical threshold for sharpness
        return False, f"Image too blurry: {laplacian_var:.1f}"
    
    return True, f"Good quality - {len(charuco_corners)} corners detected"


def capture_calibration_images():
    """Interactive image capture with quality validation."""
    rospy.init_node('intrinsic_calibration', anonymous=True)
    bridge = CvBridge()
    
    # Create ChArUco board
    board, aruco_dict = create_charuco_board()
    
    # Storage for calibration data
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    
    print("ChArUco Intrinsic Calibration")
    print("=============================")
    print("Move the board to different positions and orientations.")
    print("Ensure the entire board is visible in each image.")
    print("Target: 15 good images for robust calibration")
    print()
    
    for i in range(15):  # Target 15 good images
        input(f"Position {i+1}/15: Press Enter to capture...")
        
        # Capture image from ROS topic
        try:
            ros_image = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=5.0)
            cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
            image_size = cv_image.shape[:2][::-1]  # (width, height)
        except Exception as e:
            print(f"❌ Failed to capture image from camera: {e}")
            continue
        
        # Validate image quality
        is_good, message = validate_image_quality(cv_image, board, aruco_dict)
        
        if is_good:
            print(f"✓ {message} - Image accepted")
            
            # Extract ChArUco corners for calibration
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            
        else:
            print(f"❌ {message}")
            retry = input("Try again? (y/n): ")
            if retry.lower() == 'y':
                i -= 1  # Retry same position
            else:
                print("Skipping this position...")
    
    return all_charuco_corners, all_charuco_ids, image_size, board


def perform_calibration(all_corners, all_ids, image_size, board):
    """Perform camera calibration with error analysis."""
    print(f"\nPerforming calibration with {len(all_corners)} images...")
    
    # Calibrate camera using ChArUco
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )
    
    if not ret:
        raise ValueError("Calibration failed - insufficient data")
    
    print(f"Calibration completed with RMS error: {ret:.4f} pixels")
    
    # Validate calibration quality (Perplexity thresholds)
    if ret < 0.3:
        quality = "EXCELLENT"
        quality_color = "🟢"
    elif ret < 0.7:
        quality = "ACCEPTABLE"
        quality_color = "🟡"
    else:
        quality = "POOR - Consider retaking images"
        quality_color = "🔴"
    
    print(f"Calibration quality: {quality_color} {quality}")
    
    return camera_matrix, dist_coeffs, ret


def save_calibration_results(camera_matrix, dist_coeffs, rms_error):
    """Save calibration results to JSON file."""
    os.makedirs("calibration", exist_ok=True)
    
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "rms_error": float(rms_error),
        "image_size": [640, 480],
        "calibration_method": "charuco",
        "board_parameters": {
            "squares_x": 5,
            "squares_y": 7,
            "square_length_mm": 36,
            "marker_length_mm": 27,
            "dictionary": "DICT_4X4_50"
        },
        "quality_assessment": {
            "rms_error_pixels": float(rms_error),
            "quality_rating": "EXCELLENT" if rms_error < 0.3 else "ACCEPTABLE" if rms_error < 0.7 else "POOR",
            "focal_length_x": float(camera_matrix[0, 0]),
            "focal_length_y": float(camera_matrix[1, 1]),
            "principal_point_x": float(camera_matrix[0, 2]),
            "principal_point_y": float(camera_matrix[1, 2]),
            "distortion_k1": float(dist_coeffs[0][0]),
            "distortion_k2": float(dist_coeffs[0][1]),
            "distortion_p1": float(dist_coeffs[0][2]),
            "distortion_p2": float(dist_coeffs[0][3]),
            "distortion_k3": float(dist_coeffs[0][4])
        }
    }
    
    with open("calibration/camera_intrinsics.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"✓ Calibration saved to calibration/camera_intrinsics.json")
    return calibration_data


def print_calibration_summary(calibration_data):
    """Print human-readable calibration summary."""
    print("\n" + "="*50)
    print("CALIBRATION SUMMARY")
    print("="*50)
    
    quality = calibration_data["quality_assessment"]
    
    print(f"RMS Error: {quality['rms_error_pixels']:.4f} pixels ({quality['quality_rating']})")
    print(f"Focal Length: fx={quality['focal_length_x']:.1f}, fy={quality['focal_length_y']:.1f}")
    print(f"Principal Point: cx={quality['principal_point_x']:.1f}, cy={quality['principal_point_y']:.1f}")
    print(f"Distortion: k1={quality['distortion_k1']:.6f}, k2={quality['distortion_k2']:.6f}")
    
    print("\nNext Steps:")
    if quality['quality_rating'] == "EXCELLENT":
        print("🟢 Calibration is excellent! Proceed to Phase 2: Hand-Eye Calibration")
    elif quality['quality_rating'] == "ACCEPTABLE":
        print("🟡 Calibration is acceptable. You may proceed to Phase 2 or retake for better quality.")
    else:
        print("🔴 Calibration quality is poor. Please retake images with better lighting and board positioning.")
    
    print(f"\nCalibration file: calibration/camera_intrinsics.json")
    print("Ready to proceed to Phase 2: Hand-Eye Calibration")


if __name__ == "__main__":
    try:
        print("DOFBOT Pro Intrinsic Camera Calibration")
        print("Phase 1 of Professional Camera-Robot Calibration")
        print("=" * 60)
        print()
        
        # Capture calibration images
        corners, ids, image_size, board = capture_calibration_images()
        
        if len(corners) < 8:
            print("❌ Insufficient images for calibration (minimum 8 required)")
            print("Please run the script again and capture more images.")
            exit(1)
        
        # Perform calibration
        camera_matrix, dist_coeffs, rms_error = perform_calibration(corners, ids, image_size, board)
        
        # Save results
        calibration_data = save_calibration_results(camera_matrix, dist_coeffs, rms_error)
        
        # Print summary
        print_calibration_summary(calibration_data)
        
        print("\n🎉 Phase 1: Intrinsic calibration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
        exit(0)
    except Exception as e:
        print(f"\n❌ Calibration failed: {str(e)}")
        print("Please check your setup and try again.")
        exit(1)
