#!/usr/bin/env python3
"""
ArUco Marker Detection for MoveIt Hand-Eye Calibration
Detects ArUco marker and publishes TF transform for calibration
"""

import rospy
import cv2
import numpy as np
import tf2_ros
import json
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

class ArucoMarkerDetector:
    def __init__(self):
        rospy.init_node('aruco_marker_detector')
        
        # Parameters
        self.marker_id = rospy.get_param('~marker_id', 0)
        self.marker_size = rospy.get_param('~marker_size', 0.070)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.marker_frame = rospy.get_param('~marker_frame', 'aruco_marker_frame')
        self.aruco_dict = cv2.aruco.DICT_4X4_50
        
        # Load camera intrinsics from JSON file
        calib_file = '/home/jetson/medal-lab/robot-nodes/dofbot-pro-ros/calibration/camera_intrinsics.json'
        try:
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
                self.camera_matrix = np.array(calib_data['camera_matrix'])
                self.dist_coeffs = np.array(calib_data['distortion_coefficients'][0])
            rospy.loginfo(f"Camera intrinsics loaded from {calib_file}")
            rospy.loginfo(f"  fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
            rospy.loginfo(f"  Distortion: {self.dist_coeffs}")
        except Exception as e:
            rospy.logwarn(f"Failed to load intrinsics from file: {e}")
            rospy.logwarn("Will use camera_info topic instead")
            self.camera_matrix = None
            self.dist_coeffs = None
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        
        rospy.loginfo(f"ArUco Marker Detector started (marker_id={self.marker_id}, size={self.marker_size}m)")
    
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera intrinsics loaded from camera_info topic")
    
    def image_callback(self, msg):
        if self.camera_matrix is None:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # Use raw image - let ArUco handle distortion correction internally
            
            aruco_dict = cv2.aruco.Dictionary_get(self.aruco_dict)
            detector_params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(
                cv_image, aruco_dict, parameters=detector_params
            )
            
            if ids is not None and self.marker_id in ids:
                marker_index = np.where(ids.flatten() == self.marker_id)[0][0]
                marker_corners = corners[marker_index]
                
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                R, _ = cv2.Rodrigues(rvec[0])
                T_marker_camera = np.eye(4)
                T_marker_camera[:3, :3] = R
                T_marker_camera[:3, 3] = tvec[0].flatten()
                
                self.publish_transform(T_marker_camera, msg.header.stamp)
                
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def publish_transform(self, T_marker_camera, timestamp):
        """Publish TF transform from camera to marker"""
        translation = T_marker_camera[:3, 3]
        R = T_marker_camera[:3, :3]
        
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        
        transform = TransformStamped()
        transform.header.stamp = timestamp
        transform.header.frame_id = self.camera_frame
        transform.child_frame_id = self.marker_frame
        
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(transform)

if __name__ == '__main__':
    try:
        detector = ArucoMarkerDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
