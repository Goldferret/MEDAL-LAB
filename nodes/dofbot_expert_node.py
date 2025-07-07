"""A robot arm module for DOFBOT Pro robotic arm with expert trajectory collection.

CAMERA PIPELINE MANAGEMENT:
- Camera pipeline is configured during node startup but NOT started
- Pipeline only starts when recording begins (start_recording action)
- Pipeline stops when recording ends (stop_recording action)
- This prevents continuous frame generation and buffer overflow warnings
- Manual pipeline control available via start_camera_pipeline/stop_camera_pipeline actions
"""

import time
import json
import os
import datetime
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple

from madsci.common.types.action_types import ActionFailed, ActionResult, ActionSucceeded
from madsci.common.types.node_types import RestNodeConfig, NodeDefinition
from madsci.node_module.helpers import action
from madsci.node_module.rest_node_module import RestNode
from madsci.common.types.node_types import NodeStatus
from pydantic import Field, AnyUrl
from Arm_Lib import Arm_Device

# Orbbec DaiBai DCW2 camera support via pyorbbecsdk
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, PointCloudFilter
    ORBBEC_SDK_AVAILABLE = True
    print("✓ pyorbbecsdk imported successfully")
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    print("✗ pyorbbecsdk not available. Orbbec camera functionality will be disabled.")


class RobotArmConfig(RestNodeConfig):
    """Configuration for the robot arm node module."""

    device_number: int = 0
    """The device number of the robot arm."""
    data_collection_path: str = "./expert_trajectories"
    """Path where expert trajectory data will be stored."""
    rgb_camera_index: int = 0
    """Index of the RGB camera device (video0 for Orbbec DaBai DCW2)."""
    depth_camera_index: int = 1
    """Index of the depth camera device (currently metadata only)."""
    enable_orbbec_camera: bool = True
    """Whether to enable Orbbec DaiBai DCW2 camera via SDK (requires pyorbbecsdk)."""
    record_frequency: float = 10.0
    """Frequency in Hz to record data during trajectory collection."""

class RobotArmInterface:
    """Interface for the DOFBOT Pro robot arm with expert trajectory collection."""

    status_code: int = 0

    device_number: int = 0
    """An identifier for the robot we are controlling."""
    joint_angles: list[int] = [0, 0, 0, 0, 0]
    """The joint angles of the robot."""
    joint_velocities: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    """The joint velocities of the robot (estimated)."""
    gripper_closed: bool = False
    """The state of the gripper, open or closed."""
    is_moving: bool = False
    """Whether the robot is currently moving."""
    arm = Arm_Device()
    """Initializes the arm device for the robot"""
    torque_state: bool = False
    """Whether the robot has its torque on or off"""
    approach_method: dict[str, list: int] = {
        "directly_above": [40, 60, 0], # Basic lean-on-top approach (servos 2, 3, 4)
        "angled_above": [67, 0, 45], # lean-forward with gripper at 45 degree angle 4: 44
    }
    """Define pre-existing approach methods"""
    return_method: dict[str, list: int] = {
        "directly_above": [180, 0, 0], # Basic lean-on-top return (servos 2, 3, 4)
        "angled_above": [165, 0, 62], # lean-back with arm parallel to table
    }
    """Define pre-existing return/move-back-up methods"""
    grabbing_method: dict[str, int] = {
        "horizontal": 90, # Pinch from left-to-right of item (x axis)
        "vertical": 0 # Pinch from top-to-bottom of item (y axis)
    }
    """Define pre-existing grabbing methods"""

    # Camera and data collection attributes
    orbbec_pipeline = None
    """Orbbec SDK pipeline for both RGB and depth data"""
    orbbec_config = None
    """Orbbec SDK configuration"""
    point_cloud_filter = None
    """Point cloud filter for 3D data generation"""
    frame_to_bgr_image = None
    """Frame conversion utility function"""
    _pipeline_started: bool = False
    """Whether the camera pipeline is currently running"""
    recording: bool = False
    """Whether we're currently recording expert trajectories"""
    current_trajectory_data: Dict[str, Any] = {}
    """Current trajectory data being recorded"""
    last_angles: list[int] = [0, 0, 0, 0, 0]
    """Last recorded joint angles for velocity calculation"""
    last_timestamp: float = 0.0
    """Last timestamp for velocity calculation"""
    
    def __init__(self, device_number: int = 0, data_path: str = "./expert_trajectories", 
                 rgb_camera_index: int = 0, depth_camera_index: int = 1, enable_orbbec: bool = True):
        """Initialize the robot arm interface.
        
        Args:
            device_number: The device number of the robot arm
            data_path: Path to store expert trajectory data
            rgb_camera_index: Index of the RGB camera device (not used with Orbbec SDK)
            depth_camera_index: Index of the depth camera device (not used with Orbbec SDK)
            enable_orbbec: Whether to enable Orbbec DaiBai DCW2 camera via SDK
        """
        self.device_number = device_number
        self.data_path = data_path
        self.rgb_camera_index = rgb_camera_index
        self.depth_camera_index = depth_camera_index
        self.enable_orbbec = enable_orbbec and ORBBEC_SDK_AVAILABLE
        
        # Initialize the arm
        self.joint_angles = self._get_all_angles()
        # Ensure last_angles has valid values (replace None with 0)
        self.last_angles = [angle if angle is not None else 0 for angle in self.joint_angles.copy()]
        self.last_timestamp = time.time()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
        
        # Load camera calibration if available
        self.camera_calibration = self._load_camera_calibration()
        
        # Initialize cameras
        self._init_orbbec_camera()

    def _init_orbbec_camera(self):
        """Initialize the Orbbec DaiBai DCW2 camera configuration but don't start the pipeline yet."""
        if not self.enable_orbbec:
            print("Orbbec camera disabled in configuration")
            return
            
        try:
            # Initialize pipeline and configuration
            self.orbbec_pipeline = Pipeline()
            self.orbbec_config = Config()
            
            # Initialize point cloud filter
            self.point_cloud_filter = PointCloudFilter()
            
            # Initialize frame conversion utility
            def frame_to_bgr_image(frame):
                """Convert Orbbec frame to BGR image format."""
                if frame is None:
                    return None
                    
                width = frame.get_width()
                height = frame.get_height()
                color_format = frame.get_format()
                data = np.asanyarray(frame.get_data())
                
                try:
                    if color_format == OBFormat.RGB:
                        image = np.resize(data, (height, width, 3))
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif color_format == OBFormat.BGR:
                        image = np.resize(data, (height, width, 3))
                    elif color_format == OBFormat.YUYV:
                        image = np.resize(data, (height, width, 2))
                        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
                    elif color_format == OBFormat.MJPG:
                        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    elif color_format == OBFormat.UYVY:
                        image = np.resize(data, (height, width, 2))
                        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
                    else:
                        print(f"Unsupported color format: {color_format}")
                        return None
                    return image
                except Exception as e:
                    print(f"Error converting frame to BGR: {e}")
                    return None
            
            self.frame_to_bgr_image = frame_to_bgr_image
            
            # Configure color stream (RGB)
            try:
                color_profiles = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                if color_profiles and color_profiles.get_count() > 0:
                    print(f"Found {color_profiles.get_count()} color profiles")
                    
                    # Try to get RGB format at 640x480@10fps (lower FPS for recording efficiency)
                    color_profile = None
                    try:
                        color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 10)
                        if color_profile:
                            print("✓ Using RGB 640x480@10fps color profile")
                    except:
                        pass
                    
                    # Fallback to 30fps if 10fps not available
                    if color_profile is None:
                        try:
                            color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                            if color_profile:
                                print("✓ Using RGB 640x480@30fps color profile")
                        except:
                            pass
                    
                    # Fallback to default profile
                    if color_profile is None:
                        try:
                            color_profile = color_profiles.get_default_video_stream_profile()
                            if color_profile:
                                print("✓ Using default color profile")
                        except:
                            pass
                    
                    if color_profile:
                        self.orbbec_config.enable_stream(color_profile)
                        print(f"Color stream configured: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
                    else:
                        print("✗ No suitable color profile found")
                else:
                    print("✗ No color profiles available")
            except Exception as e:
                print(f"Error configuring color stream: {e}")
            
            # Configure depth stream
            try:
                depth_profiles = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                if depth_profiles and depth_profiles.get_count() > 0:
                    print(f"Found {depth_profiles.get_count()} depth profiles")
                    
                    # Try to get a lower FPS depth profile for recording efficiency
                    depth_profile = None
                    try:
                        # Try 10fps first with Y16 format
                        depth_profile = depth_profiles.get_video_stream_profile(640, 480, OBFormat.Y16, 10)
                        if depth_profile:
                            print("✓ Using 640x480@10fps depth profile")
                    except:
                        pass
                    
                    # Fallback to default depth profile
                    if depth_profile is None:
                        try:
                            depth_profile = depth_profiles.get_default_video_stream_profile()
                            if depth_profile:
                                print("✓ Using default depth profile")
                        except:
                            pass
                    
                    if depth_profile:
                        self.orbbec_config.enable_stream(depth_profile)
                        print(f"Depth stream configured: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps")
                    else:
                        print("✗ No suitable depth profile found")
                else:
                    print("✗ No depth profiles available")
            except Exception as e:
                print(f"Error configuring depth stream: {e}")
            
            print("✓ Orbbec DaiBai DCW2 camera configured (pipeline not started)")
            self._pipeline_started = False
                    
        except Exception as e:
            print(f"Error configuring Orbbec camera: {e}")
            self.orbbec_pipeline = None
            self.orbbec_config = None
            self.point_cloud_filter = None
    def _start_camera_pipeline(self) -> bool:
        """Start the camera pipeline for recording. Returns True if successful."""
        if self.orbbec_pipeline is None:
            print("✗ Camera not configured")
            return False
            
        if self._pipeline_started:
            print("✓ Camera pipeline already running")
            return True
            
        try:
            self.orbbec_pipeline.start(self.orbbec_config)
            self._pipeline_started = True
            print("✓ Started camera pipeline for recording")
            
            # Brief test to ensure pipeline is working
            try:
                frames = self.orbbec_pipeline.wait_for_frames(1000)
                if frames:
                    print("✓ Camera pipeline test successful")
                    return True
                else:
                    print("⚠ Camera pipeline test returned no frames")
                    return True  # Still consider it started, might just need time
            except Exception as e:
                print(f"⚠ Camera pipeline test failed: {e}")
                return True  # Still consider it started
                
        except Exception as e:
            print(f"✗ Failed to start camera pipeline: {e}")
            self._pipeline_started = False
            return False
    
    def _stop_camera_pipeline(self) -> None:
        """Stop the camera pipeline to free resources."""
        if self.orbbec_pipeline is None or not self._pipeline_started:
            return
            
        try:
            self.orbbec_pipeline.stop()
            self._pipeline_started = False
            print("✓ Stopped camera pipeline")
        except Exception as e:
            print(f"Error stopping camera pipeline: {e}")
            # Mark as stopped anyway to prevent stuck state
            self._pipeline_started = False
    def _cleanup_cameras(self):
        """Clean up camera resources."""
        self._stop_camera_pipeline()
        if self.orbbec_pipeline is not None:
            print("✓ Camera resources cleaned up")
        """Load camera calibration data if available."""
        calibration_path = os.path.join(self.data_path, "camera_calibration.json")
        
        # Default calibration values for Orbbec DaiBai DCW2 (approximate)
        default_calibration = {
            "fx": 525.0,
            "fy": 525.0,
            "cx": 320.0,
            "cy": 240.0,
            "depth_scale": 1000.0,
            "width": 640,
            "height": 480
        }
        
        try:
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    calibration_data = json.load(f)
                    
                # Extract values from camera matrix format if present
                if "camera_matrix" in calibration_data:
                    matrix = calibration_data["camera_matrix"]
                    return {
                        "fx": matrix[0][0],
                        "fy": matrix[1][1],
                        "cx": matrix[0][2],
                        "cy": matrix[1][2],
                        "depth_scale": calibration_data.get("depth_scale", 1000.0),
                        "width": calibration_data.get("image_width", 640),
                        "height": calibration_data.get("image_height", 480)
                    }
                else:
                    return calibration_data
            else:
                print(f"No calibration file found at {calibration_path}, using default values")
                return default_calibration
                
        except Exception as e:
            print(f"Error loading camera calibration: {e}, using default values")
            return default_calibration

    def get_device_number(self) -> int:
        """Get the robot number."""
        return self.device_number
    
    def _get_all_angles(self):
        """Read angles for servos 1 through 6."""
        angles = []
        for i in range(1, 7):
            try:
                angle = self.arm.Arm_serial_servo_read(i)
                # If angle is None (I2C error, pos==0, or out of range), use fallback
                if angle is None:
                    if hasattr(self, 'joint_angles') and len(self.joint_angles) >= i:
                        angle = self.joint_angles[i-1]  # Use last known angle
                        print(f"Warning: Servo {i} read failed, using last known angle: {angle}")
                    else:
                        angle = 90  # Use middle position as fallback
                        print(f"Warning: Servo {i} read failed, using default angle: {angle}")
                angles.append(angle)
            except Exception as e:
                print(f"Error reading servo {i} angle: {e}")
                # Use last known angle or middle position as fallback
                if hasattr(self, 'joint_angles') and len(self.joint_angles) >= i:
                    angles.append(self.joint_angles[i-1])
                else:
                    angles.append(90)  # Middle position fallback
        return angles
        
    def _update_joint_velocities(self):
        """Update joint velocities based on angle changes."""
        current_time = time.time()
        dt = current_time - self.last_timestamp
        if dt > 0:
            current_angles = self._get_all_angles()[:5]  # Exclude gripper
            
            # Ensure we have valid angles and last_angles
            if all(angle is not None for angle in current_angles) and all(angle is not None for angle in self.last_angles):
                self.joint_velocities = [(current - last) / dt for current, last in zip(current_angles, self.last_angles)]
                self.last_angles = current_angles
            else:
                # If we have None values, keep previous velocities or set to zero
                print("Warning: Some joint angles are None, skipping velocity update")
                if not hasattr(self, 'joint_velocities') or len(self.joint_velocities) != 5:
                    self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0]
                # Update last_angles with valid values only
                self.last_angles = [angle if angle is not None else last for angle, last in zip(current_angles, self.last_angles)]
            
            self.last_timestamp = current_time

    def change_torque_state(self) -> None:
        """Changes the robot's torque"""
        self.torque_state = not self.torque_state
        self.arm.Arm_serial_set_torque(1 if self.torque_state else 0)

    def capture_rgb_image(self) -> Optional[np.ndarray]:
        """Capture an RGB image from the Orbbec camera via SDK.
        
        Note: This method requires the camera pipeline to be running.
        It will not start the pipeline automatically to avoid resource waste.
        """
        if self.orbbec_pipeline is None:
            print("✗ Camera not configured")
            return None
            
        if not self._pipeline_started:
            print("✗ Camera pipeline not started. Use start_recording() to begin camera capture.")
            return None
            
        try:
            # Wait for frames with timeout
            frames = self.orbbec_pipeline.wait_for_frames(1000)  # 1 second timeout
            if not frames:
                return None
                
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            
            # Convert frame to BGR image
            bgr_image = self.frame_to_bgr_image(color_frame)
            return bgr_image
            
        except Exception as e:
            print(f"Error capturing RGB image: {e}")
            return None

    def _load_camera_calibration(self) -> Dict[str, Any]:
        """Load camera calibration data if available."""
        calibration_path = os.path.join(self.data_path, "camera_calibration.json")
        
        # Default calibration values for Orbbec DaiBai DCW2 (approximate)
        default_calibration = {
            "fx": 525.0,
            "fy": 525.0,
            "cx": 320.0,
            "cy": 240.0,
            "depth_scale": 1000.0,
            "width": 640,
            "height": 480,
            "camera_matrix": [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        try:
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r') as f:
                    calibration_data = json.load(f)
                    
                # Extract values from camera matrix format if present
                if "camera_matrix" in calibration_data:
                    matrix = calibration_data["camera_matrix"]
                    # Keep ALL original data and add simplified fields
                    result = calibration_data.copy()  # Preserve everything
                    result.update({
                        "fx": matrix[0][0],
                        "fy": matrix[1][1],
                        "cx": matrix[0][2],
                        "cy": matrix[1][2],
                        "depth_scale": calibration_data.get("depth_scale", 1000.0),
                        "width": calibration_data.get("image_width", 640),
                        "height": calibration_data.get("image_height", 480)
                    })
                    return result
                else:
                    # If no camera_matrix, add it from individual parameters
                    fx = calibration_data.get("fx", 525.0)
                    fy = calibration_data.get("fy", 525.0)
                    cx = calibration_data.get("cx", 320.0)
                    cy = calibration_data.get("cy", 240.0)
                    
                    result = calibration_data.copy()
                    result.update({
                        "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        "distortion_coefficients": calibration_data.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])
                    })
                    return result
            else:
                print(f"No calibration file found at {calibration_path}, using default values")
                return default_calibration
                
        except Exception as e:
            print(f"Error loading camera calibration: {e}, using default values")
            return default_calibration

    def capture_depth_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture depth image and generate point cloud from the Orbbec DaiBai DCW2 camera via SDK.
        
        Note: This method requires the camera pipeline to be running.
        
        Returns:
            Tuple of (depth_image, depth_colormap, point_cloud_data) or (None, None, None) if not available
        """
        if self.orbbec_pipeline is None:
            print("✗ Camera not configured")
            return None, None, None
            
        if not self._pipeline_started:
            print("✗ Camera pipeline not started. Use start_recording() to begin camera capture.")
            return None, None, None
            
        try:
            # Wait for frames with timeout
            frames = self.orbbec_pipeline.wait_for_frames(1000)  # 1 second timeout
            if not frames:
                return None, None, None
                
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None, None, None
            
            # Convert to numpy array
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            
            # Create a colorized version of the depth map for visualization
            # Normalize depth values to 0-255 range for better visualization
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Generate point cloud data using the SDK's point cloud filter
            point_cloud_data = self._generate_point_cloud_sdk(frames)
            
            return depth_image, depth_colormap, point_cloud_data
            
        except Exception as e:
            print(f"Error capturing depth data: {e}")
            return None, None, None
    
    def _generate_point_cloud_sdk(self, frames) -> Optional[np.ndarray]:
        """Generate a point cloud using the Orbbec SDK's PointCloudFilter.
        
        Args:
            frames: Frame set containing depth (and optionally color) frames
            
        Returns:
            Point cloud as Nx3 numpy array (x, y, z coordinates) or None
        """
        if self.point_cloud_filter is None:
            return None
            
        try:
            # Generate point cloud using SDK
            point_cloud_frame = self.point_cloud_filter.process(frames)
            if not point_cloud_frame:
                return None
            
            # Convert to numpy array
            point_data = np.frombuffer(point_cloud_frame.get_data(), dtype=np.float32)
            
            # Reshape to Nx3 (x, y, z coordinates)
            # Each point is 3 floats (x, y, z)
            num_points = len(point_data) // 3
            if num_points * 3 != len(point_data):
                print(f"Warning: Point cloud data size mismatch. Expected multiple of 3, got {len(point_data)}")
                return None
                
            points_3d = point_data.reshape((num_points, 3))
            
            # Filter out invalid points (NaN, inf, or zero depth)
            valid_mask = np.isfinite(points_3d).all(axis=1) & (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 10.0)
            valid_points = points_3d[valid_mask]
            
            return valid_points
            
        except Exception as e:
            print(f"Error generating point cloud with SDK: {e}")
            # Fallback to manual point cloud generation
            return self._generate_point_cloud_manual(frames)
    
    def _generate_point_cloud_manual(self, frames) -> Optional[np.ndarray]:
        """Generate a point cloud manually from depth frame using calibrated camera parameters.
        
        Args:
            frames: Frame set containing depth frame
            
        Returns:
            Point cloud as Nx3 numpy array (x, y, z coordinates) or None
        """
        try:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
                
            # Convert to numpy array
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            
            # Use calibrated camera parameters
            fx = self.camera_calibration["fx"]
            fy = self.camera_calibration["fy"]
            cx = self.camera_calibration["cx"]
            cy = self.camera_calibration["cy"]
            depth_scale = self.camera_calibration["depth_scale"]
            
            height, width = depth_image.shape
            
            # Create coordinate grids
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert depth image to meters
            # Orbbec depth values are typically in millimeters
            depth_m = depth_image.astype(np.float32) / depth_scale
            
            # Filter out invalid depth values
            valid_mask = (depth_m > 0.1) & (depth_m < 10.0)  # Keep depths between 10cm and 10m
            
            # Calculate 3D coordinates using pinhole camera model
            x = (u - cx) * depth_m / fx
            y = (v - cy) * depth_m / fy
            z = depth_m
            
            # Stack coordinates and filter valid points
            points_3d = np.stack([x, y, z], axis=-1)
            valid_points = points_3d[valid_mask]
            
            # Additional filtering to remove outliers
            if len(valid_points) > 0:
                # Remove points that are too far from the median (simple outlier removal)
                median_z = np.median(valid_points[:, 2])
                z_std = np.std(valid_points[:, 2])
                outlier_mask = np.abs(valid_points[:, 2] - median_z) < 3 * z_std
                valid_points = valid_points[outlier_mask]
            
            return valid_points
            
        except Exception as e:
            print(f"Error generating manual point cloud: {e}")
            return None
    def capture_synchronized_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized RGB and depth data from the Orbbec camera.
        
        Note: This method requires the camera pipeline to be running.
        
        Returns:
            Tuple of (rgb_image, depth_image, depth_colormap, point_cloud_data)
        """
        if self.orbbec_pipeline is None:
            print("✗ Camera not configured")
            return None, None, None, None
            
        if not self._pipeline_started:
            print("✗ Camera pipeline not started. Use start_recording() to begin camera capture.")
            return None, None, None, None
            
        try:
            # Wait for frames with timeout
            frames = self.orbbec_pipeline.wait_for_frames(1000)  # 1 second timeout
            if not frames:
                return None, None, None, None
            
            # Get RGB frame
            rgb_image = None
            color_frame = frames.get_color_frame()
            if color_frame:
                rgb_image = self.frame_to_bgr_image(color_frame)
            
            # Get depth data
            depth_image = None
            depth_colormap = None
            point_cloud_data = None
            
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                # Convert to numpy array
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                
                # Create colorized depth map
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # Generate point cloud
                point_cloud_data = self._generate_point_cloud_sdk(frames)
            
            return rgb_image, depth_image, depth_colormap, point_cloud_data
            
        except Exception as e:
            print(f"Error capturing synchronized data: {e}")
            return None, None, None, None
            
    def start_recording(self, annotation: str = "") -> str:
        """Start recording an expert trajectory.
        
        Args:
            annotation: Text description of the trajectory being recorded
            
        Returns:
            ID of the trajectory being recorded
        """
        if self.recording:
            print("Already recording a trajectory. Stop the current recording first.")
            return self.current_trajectory_data.get("trajectory_id", "")
        
        # Start the camera pipeline when recording begins
        if self.orbbec_pipeline and not self._pipeline_started:
            if not self._start_camera_pipeline():
                print("⚠ Failed to start camera pipeline, continuing without camera data")
            
        # Generate a unique ID for this trajectory
        trajectory_id = f"trajectory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trajectory_dir = os.path.join(self.data_path, trajectory_id)
        os.makedirs(trajectory_dir, exist_ok=True)
        os.makedirs(os.path.join(trajectory_dir, "rgb_images"), exist_ok=True)
        os.makedirs(os.path.join(trajectory_dir, "depth_images"), exist_ok=True)
        os.makedirs(os.path.join(trajectory_dir, "point_clouds"), exist_ok=True)
        
        self.current_trajectory_data = {
            "trajectory_id": trajectory_id,
            "annotation": annotation,
            "start_time": time.time(),
            "joint_states": [],
            "actions": [],
            "images": []
        }
        
        self.recording = True
        print(f"Started recording trajectory {trajectory_id}")
        return trajectory_id
        
    def stop_recording(self) -> Optional[str]:
        """Stop recording the current expert trajectory and save the data.
        
        Returns:
            ID of the saved trajectory or None if no recording was active
        """
        if not self.recording:
            print("No active recording to stop.")
            return None
            
        trajectory_id = self.current_trajectory_data["trajectory_id"]
        trajectory_dir = os.path.join(self.data_path, trajectory_id)
        
        # Add end time
        self.current_trajectory_data["end_time"] = time.time()
        self.current_trajectory_data["duration"] = (
            self.current_trajectory_data["end_time"] - self.current_trajectory_data["start_time"]
        )
        
        # Save trajectory metadata and joint states
        with open(os.path.join(trajectory_dir, "trajectory_data.json"), "w") as f:
            json.dump(self.current_trajectory_data, f, indent=2)
        
        # Stop the camera pipeline when recording ends
        if self.orbbec_pipeline and self._pipeline_started:
            self._stop_camera_pipeline()
            
        self.recording = False
        print(f"Stopped and saved trajectory {trajectory_id}")
        return trajectory_id
        
    def record_data_point(self, action_name: str = None, action_params: Dict = None):
        """Record a data point in the current trajectory.
        
        Args:
            action_name: Name of the action being performed (if any)
            action_params: Parameters of the action being performed (if any)
        """
        if not self.recording:
            return
            
        timestamp = time.time()
        trajectory_id = self.current_trajectory_data["trajectory_id"]
        trajectory_dir = os.path.join(self.data_path, trajectory_id)
        
        # Update joint velocities
        self._update_joint_velocities()
        
        # Get current joint states
        joint_angles = self._get_all_angles()
        
        # Capture synchronized images for better data quality
        rgb_image, depth_image, depth_colormap, point_cloud_data = self.capture_synchronized_data()
        
        # Fallback to individual capture if synchronized capture fails
        if rgb_image is None:
            rgb_image = self.capture_rgb_image()
        if depth_image is None:
            depth_image, depth_colormap, point_cloud_data = self.capture_depth_data()
        
        # Generate frame ID
        frame_id = f"frame_{len(self.current_trajectory_data['joint_states']):06d}"
        
        # Save images if available
        image_data = {"frame_id": frame_id, "timestamp": timestamp}
        
        if rgb_image is not None:
            rgb_path = os.path.join(trajectory_dir, "rgb_images", f"{frame_id}.jpg")
            cv2.imwrite(rgb_path, rgb_image)
            image_data["rgb_path"] = f"rgb_images/{frame_id}.jpg"
            
        if depth_image is not None:
            # Save raw depth image
            depth_path = os.path.join(trajectory_dir, "depth_images", f"{frame_id}_depth.png")
            cv2.imwrite(depth_path, depth_image)
            image_data["depth_path"] = f"depth_images/{frame_id}_depth.png"
            
            # Save colorized depth map for visualization
            if depth_colormap is not None:
                depth_colormap_path = os.path.join(trajectory_dir, "depth_images", f"{frame_id}_depth_colormap.jpg")
                cv2.imwrite(depth_colormap_path, depth_colormap)
                image_data["depth_colormap_path"] = f"depth_images/{frame_id}_depth_colormap.jpg"
            
            # Save point cloud data
            if point_cloud_data is not None:
                point_cloud_path = os.path.join(trajectory_dir, "point_clouds", f"{frame_id}_pointcloud.npy")
                np.save(point_cloud_path, point_cloud_data)
                image_data["point_cloud_path"] = f"point_clouds/{frame_id}_pointcloud.npy"
                image_data["point_cloud_size"] = len(point_cloud_data)
        
        # Record joint state
        joint_state = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "joint_angles": joint_angles[:5],  # First 5 are arm joints
            "gripper_angle": joint_angles[5],  # 6th is gripper
            "joint_velocities": self.joint_velocities,
            "gripper_closed": self.gripper_closed
        }
        
        # Record action if provided
        if action_name:
            action_data = {
                "timestamp": timestamp,
                "frame_id": frame_id,
                "action_name": action_name,
                "action_params": action_params or {}
            }
            self.current_trajectory_data["actions"].append(action_data)
        
        # Add data to trajectory
        self.current_trajectory_data["joint_states"].append(joint_state)
        self.current_trajectory_data["images"].append(image_data)

    def move_single_joint(self, servo_id: int, angle: int, record_action: bool = True) -> None:
        """Move the robot to the specified joint angles."""
        if self.is_moving:
            raise RuntimeError("Robot is already moving.")
        if servo_id < 1 or servo_id > 6:
            raise ValueError("Invalid servo id.")
        self.is_moving = True

        # Record action start if recording
        if record_action and self.recording:
            self.record_data_point(
                action_name="move_single_joint",
                action_params={"servo_id": servo_id, "angle": angle}
            )

        angles = self.joint_angles.copy()
        angles[servo_id -1] = angle
        angles.append(145 if self.gripper_closed else 90)
        try:
            self.arm.Arm_serial_servo_write6_array(angles, 2000)
            
            # Record data points during movement if recording
            if self.recording:
                start_time = time.time()
                interval = 1.0 / 10.0  # 10Hz recording
                while time.time() - start_time < 2.0:  # 2 second movement
                    self.record_data_point()
                    time.sleep(interval)
            else:
                time.sleep(2)
            
        except Exception as e:
            print(f"An error occurred in move_single_joint (servo: {servo_id}): {e}")

        self.is_moving = False
        self.joint_angles[servo_id -1] = angle
        
        # Record action end if recording
        if record_action and self.recording:
            self.record_data_point()

    def move_all_joints(self, angles: list[int], record_action: bool = True) -> None:
        """Move the robot to the specified joint angles (at the same time)."""
        if self.is_moving:
            raise RuntimeError("Robot is already moving.")
        if len(angles) != 5:
            raise ValueError("Expected 5 joint angles.")
        self.is_moving = True

        # Record action start if recording
        if record_action and self.recording:
            self.record_data_point(
                action_name="move_all_joints",
                action_params={"angles": angles}
            )

        angles_with_gripper = angles.copy()
        angles_with_gripper.append(145 if self.gripper_closed else 90)

        try:
            self.arm.Arm_serial_servo_write6_array(angles_with_gripper, 2000)
            
            # Record data points during movement if recording
            if self.recording:
                start_time = time.time()
                interval = 1.0 / 10.0  # 10Hz recording
                while time.time() - start_time < 2.0:  # 2 second movement
                    self.record_data_point()
                    time.sleep(interval)
            else:
                time.sleep(2)
                
        except Exception as e:
            print(f"An error occurred in move_all_joints: {e}")

        self.is_moving = False
        self.joint_angles = angles[:5]
        
        # Record action end if recording
        if record_action and self.recording:
            self.record_data_point()

    def close_gripper(self, record_action: bool = True) -> None:
        """Close the gripper."""
        if self.is_moving:
            raise RuntimeError("Robot is already moving.")
        self.is_moving = True

        # Record action start if recording
        if record_action and self.recording:
            self.record_data_point(
                action_name="close_gripper",
                action_params={}
            )

        try:
            # Using static "145 degrees" to close the gripper
            self.arm.Arm_serial_servo_write(6, 145, 1000)
            
            # Record data points during movement if recording
            if self.recording:
                start_time = time.time()
                interval = 1.0 / 10.0  # 10Hz recording
                while time.time() - start_time < 1.0:  # 1 second movement
                    self.record_data_point()
                    time.sleep(interval)
            else:
                time.sleep(1)
                
        except Exception as e:
            print(f"An error occurred in close_gripper: {e}")

        self.is_moving = False
        self.gripper_closed = True
        
        # Record action end if recording
        if record_action and self.recording:
            self.record_data_point()

    def open_gripper(self, record_action: bool = True) -> None:
        """Open the gripper."""
        if self.is_moving:
            raise RuntimeError("Robot is already moving.")
        self.is_moving = True

        # Record action start if recording
        if record_action and self.recording:
            self.record_data_point(
                action_name="open_gripper",
                action_params={}
            )

        try:
            # Using static "90 degrees" to open the gripper
            self.arm.Arm_serial_servo_write(6, 90, 1000)
            
            # Record data points during movement if recording
            if self.recording:
                start_time = time.time()
                interval = 1.0 / 10.0  # 10Hz recording
                while time.time() - start_time < 1.0:  # 1 second movement
                    self.record_data_point()
                    time.sleep(interval)
            else:
                time.sleep(1)
                
        except Exception as e:
            print(f"An error occurred in open_gripper: {e}")

        self.is_moving = False
        self.gripper_closed = False
        
        # Record action end if recording
        if record_action and self.recording:
            self.record_data_point()

    def pick_from_source(self, source_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True) -> None:
        """Grab an item from the given location"""
        if self.is_moving:
            raise RuntimeError("Robot is already moving.")
        # Checks to see if the provided angle is valid
        if source_angle > 180 or source_angle < 0:
            raise RuntimeError("Invalid Source Angle Provided")

        # Record high-level action start if recording
        if record_action and self.recording:
            self.record_data_point(
                action_name="pick_from_source",
                action_params={"source_angle": source_angle}
            )

        # Concatinates servo angles for place and return movements (set manually for now)
        pick_method = [source_angle] + self.approach_method[approach_key] + [self.grabbing_method[grab_key]]
        return_method = [source_angle] + self.return_method[return_key] + [self.grabbing_method[grab_key]]

        self.move_all_joints([90, 180, 0, 0, 90], record_action=False)  # Move to grabber position
        self.move_single_joint(1, source_angle, record_action=False)  # Rotate to face source location
        self.move_all_joints(pick_method, record_action=False)  # Lean forward to grab
        self.close_gripper(record_action=False)  # Close gripper
        self.move_all_joints(return_method, record_action=False)  # Move back up
        
        # Record high-level action end if recording
        if record_action and self.recording:
            self.record_data_point()

    def place_at_target(self, target_angle: int, approach_key: str, return_key: str, grab_key: str, record_action: bool = True) -> None:
        """Place an item from the given location"""
        if self.is_moving:
            raise RuntimeError("Robot is already moving.")
        # Checks to see if the provided angle is valid
        if target_angle > 180 or target_angle < 0:
            raise RuntimeError("Invalid Target Angle Provided")

        # Record high-level action start if recording
        if record_action and self.recording:
            self.record_data_point(
                action_name="place_at_target",
                action_params={"target_angle": target_angle}
            )
        # Concatinates servo angles for place and return movements (set manually for now)
        place_method = [target_angle] + self.approach_method[approach_key] + [self.grabbing_method[grab_key]]
        return_method = [target_angle] + self.return_method[return_key] + [self.grabbing_method[grab_key]]

        self.move_single_joint(1, target_angle, record_action=False)  # Rotate to face target location
        self.move_all_joints(place_method, record_action=False)  # Lean forward to place
        self.open_gripper(record_action=False)  # Open gripper
        self.move_all_joints(return_method, record_action=False)  # Move back up
        self.move_all_joints([90, 180, 0, 0, 90], record_action=False)  # Move back to grabber position
        
        # Record high-level action end if recording
        if record_action and self.recording:
            self.record_data_point()

class RobotArmNode(RestNode):
    """A robot arm node module with expert trajectory collection capabilities."""

    robot_interface: RobotArmInterface = None
    config_model = RobotArmConfig
    node_status = NodeStatus()
    node_state = {
        "joint_angles": [0, 0, 0, 0, 0],
        "joint_velocities": [0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_closed": False,
        "recording": False,
        "current_trajectory_id": None,
        "camera_pipeline_started": False
    }

    def startup_handler(self) -> None:
        """Handle the startup of the node."""
        # Get config values with defaults if attributes are missing
        device_number = getattr(self.config, 'device_number', 0)
        data_path = getattr(self.config, 'data_collection_path', "./expert_trajectories")
        rgb_camera_index = getattr(self.config, 'rgb_camera_index', 0)  # Not used with Orbbec SDK
        depth_camera_index = getattr(self.config, 'depth_camera_index', 1)  # Not used with Orbbec SDK
        enable_orbbec = getattr(self.config, 'enable_orbbec_camera', True)  # Enabled by default if SDK available
        
        self.logger.log_info(f"Connecting to robot {device_number}...")
        self.robot_interface = RobotArmInterface(
            device_number=device_number,
            data_path=data_path,
            rgb_camera_index=rgb_camera_index,
            depth_camera_index=depth_camera_index,
            enable_orbbec=enable_orbbec
        )
        self.logger.log_info(f"Connected to robot {self.robot_interface.get_device_number()}")


    def shutdown_handler(self) -> None:
        """Handle the shutdown of the node."""
        # Use safe attribute access like in startup_handler
        device_number = getattr(self.config, 'device_number', 0)
        self.logger.log_info(f"Disconnecting from robot {device_number}...")
        
        # Stop recording if active
        if self.robot_interface.recording:
            self.robot_interface.stop_recording()
            
        # Clean up camera resources
        if hasattr(self.robot_interface, "_cleanup_cameras"):
            self.robot_interface._cleanup_cameras()
            
        del self.robot_interface
        self.logger.log_info(f"Disconnected from robot {device_number}")

    def state_handler(self) -> None:
        """This is where you can implement logic to periodically update the node's public-facing state information."""
        if self.robot_interface is not None:
            self.node_state = {
                "joint_angles": self.robot_interface.joint_angles,
                "joint_velocities": self.robot_interface.joint_velocities,
                "gripper_closed": self.robot_interface.gripper_closed,
                "recording": self.robot_interface.recording,
                "current_trajectory_id": self.robot_interface.current_trajectory_data.get("trajectory_id") if self.robot_interface.recording else None,
                "camera_pipeline_started": self.robot_interface._pipeline_started
            }
        else:
            self.node_state = {
                "joint_angles": None, 
                "joint_velocities": None,
                "gripper_closed": None,
                "recording": False,
                "current_trajectory_id": None,
                "camera_pipeline_started": False
            }

    def status_handler(self) -> None:
        """
        This is where you can implement logic to periodically update the node's status information.
        """
        if self.robot_interface is not None and self.robot_interface.is_moving:
            self.node_status.busy = True
        else:
            self.node_status.busy = len(self.node_status.running_actions) > 0

    @action(name="move_joint", description="Move the robot's joint by the specified joint angle")
    def move_joints(self, servo_id: int, joint_angle: int) -> ActionResult:
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
        if self.robot_interface.is_moving:
            self.logger.log_error("Robot is already moving")
            return ActionFailed(errors="Robot is already moving")
        self.robot_interface.change_torque_state() # Turns the torque on
        self.robot_interface.move_single_joint(servo_id, joint_angle)
        self.robot_interface.change_torque_state() # Turns the torque off
        self.logger.log_info(f"Moved servo {servo_id} to {joint_angle} degrees")
        return ActionSucceeded()
    
    @action(name="move_all_joints", description="Move all the robot's joints by the specified angles")
    def move_all_joints(self, joint_angles: list[int]) -> ActionResult:
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
        if self.robot_interface.is_moving:
            self.logger.log_error("Robot is already moving")
            return ActionFailed(errors="Robot is already moving")
        if len(joint_angles) != 5:
            self.logger.log_error("Invalid number of joint angles. Expected 5.")
            return ActionFailed(errors="Invalid number of joint angles. Expected 5.")
        self.robot_interface.change_torque_state() # Turns the torque on
        self.robot_interface.move_all_joints(joint_angles)
        self.robot_interface.change_torque_state() # Turns the torque off
        self.logger.log_info(f"Moved robot to joint angles: {joint_angles}")
        return ActionSucceeded()
    
    @action(name="grabber_position", description="Move the robot to the grabber position")
    def grabber_position(self) -> ActionResult:
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
        if self.robot_interface.is_moving:
            self.logger.log_error("Robot is already moving")
            return ActionFailed(errors="Robot is already moving")
        self.robot_interface.change_torque_state() # Turns the torque on
        self.robot_interface.move_all_joints([90, 180, 0, 0, 90])
        self.robot_interface.change_torque_state() # Turns the torque off
        self.logger.log_info(f"Moved robot to grabber position")
        return ActionSucceeded()
        
    @action(name="transfer", description="Transfer item from source to target location")
    def transfer(self, locations: list[int], movement_keys: list[str]) -> ActionResult:
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
        if self.robot_interface.is_moving:
            self.logger.log_error("Robot is already moving")
            return ActionFailed(errors="Robot is already moving")
        if len(locations) != 2:
            self.logger.log_error("Invalid number of location arguments. Expected 2.")
            return ActionFailed(errors="Invalid number of location arguments. Expected 2.")
        self.robot_interface.change_torque_state() # Turns the torque on
        self.robot_interface.pick_from_source(locations[0], *movement_keys) # Grab item from the provided location
        self.robot_interface.place_at_target(locations[1], *movement_keys) # Place item at the provided location
        self.robot_interface.change_torque_state() # Turns torque off
        self.logger.log_info(f"Moved item from {locations[0]} degrees to {locations[1]} degrees")
        return ActionSucceeded()
        
    @action(name="start_recording", description="Start camera pipeline and begin recording expert trajectory data")
    def start_recording(self, annotation: str = "", task_description: str = "") -> ActionResult:
        """Start camera pipeline and begin recording expert trajectory data with annotations."""
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
            
        if self.robot_interface.recording:
            self.logger.log_error("Already recording a trajectory")
            return ActionFailed(errors="Already recording a trajectory")
        
        try:
            # Step 1: Start camera pipeline first
            self.logger.log_info("Starting camera pipeline for recording...")
            pipeline_success = self.robot_interface._start_camera_pipeline()
            if not pipeline_success:
                self.logger.log_warning("Failed to start camera pipeline, continuing without camera data")
            
            # Step 2: Start recording trajectory
            trajectory_id = self.robot_interface.start_recording(annotation)
            self.logger.log_info(f"Started recording expert trajectory: {trajectory_id}")
            
            # Step 3: Record initial annotation with task description
            if task_description:
                full_annotation = f"{annotation} - Task: {task_description}"
            else:
                full_annotation = annotation
                
            if full_annotation:
                self.robot_interface.record_data_point(
                    action_name="annotation",
                    action_params={"text": full_annotation}
                )
                self.logger.log_info(f"Recorded initial annotation: {full_annotation}")
            
            return ActionSucceeded(data={
                "trajectory_id": trajectory_id,
                "camera_pipeline_started": pipeline_success,
                "annotation_recorded": bool(full_annotation)
            })
            
        except Exception as e:
            self.logger.log_error(f"Error starting recording session: {e}")
            # Try to clean up if something went wrong
            try:
                if self.robot_interface.recording:
                    self.robot_interface.stop_recording()
                if self.robot_interface._pipeline_started:
                    self.robot_interface._stop_camera_pipeline()
            except:
                pass
            return ActionFailed(errors=str(e))
        
    @action(name="stop_recording", description="Stop recording and stop camera pipeline")
    def stop_recording(self) -> ActionResult:
        """Stop recording expert trajectory data and stop camera pipeline."""
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
            
        if not self.robot_interface.recording:
            self.logger.log_error("No active recording to stop")
            return ActionFailed(errors="No active recording to stop")
        
        try:
            # Step 1: Stop recording trajectory
            trajectory_id = self.robot_interface.stop_recording()
            self.logger.log_info(f"Stopped recording expert trajectory: {trajectory_id}")
            
            # Step 2: Stop camera pipeline
            pipeline_stopped = False
            if self.robot_interface._pipeline_started:
                self.robot_interface._stop_camera_pipeline()
                pipeline_stopped = True
                self.logger.log_info("Stopped camera pipeline")
            
            return ActionSucceeded(data={
                "trajectory_id": trajectory_id,
                "camera_pipeline_stopped": pipeline_stopped
            })
            
        except Exception as e:
            self.logger.log_error(f"Error stopping recording session: {e}")
            return ActionFailed(errors=str(e))

    @action(name="test_camera_capture", description="Test camera capture functionality")
    def test_camera_capture(self) -> ActionResult:
        """Test the camera capture functionality and return detailed status."""
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
            
        if not ORBBEC_SDK_AVAILABLE:
            return ActionFailed(errors="pyorbbecsdk not available")
            
        try:
            # Test synchronized capture
            rgb_image, depth_image, depth_colormap, point_cloud_data = self.robot_interface.capture_synchronized_data()
            
            result = {
                "rgb_available": rgb_image is not None,
                "depth_available": depth_image is not None,
                "point_cloud_available": point_cloud_data is not None,
                "camera_initialized": self.robot_interface.orbbec_pipeline is not None
            }
            
            if rgb_image is not None:
                result["rgb_shape"] = rgb_image.shape
                
            if depth_image is not None:
                result["depth_shape"] = depth_image.shape
                result["depth_range"] = {
                    "min": int(np.min(depth_image[depth_image > 0])) if np.any(depth_image > 0) else 0,
                    "max": int(np.max(depth_image))
                }
                
            if point_cloud_data is not None:
                result["point_cloud_size"] = len(point_cloud_data)
                result["point_cloud_bounds"] = {
                    "x_range": [float(np.min(point_cloud_data[:, 0])), float(np.max(point_cloud_data[:, 0]))],
                    "y_range": [float(np.min(point_cloud_data[:, 1])), float(np.max(point_cloud_data[:, 1]))],
                    "z_range": [float(np.min(point_cloud_data[:, 2])), float(np.max(point_cloud_data[:, 2]))]
                }
            
            self.logger.log_info(f"Camera test completed: RGB={result['rgb_available']}, Depth={result['depth_available']}, PointCloud={result['point_cloud_available']}")
            return ActionSucceeded(data=result)
            
        except Exception as e:
            self.logger.log_error(f"Error testing camera capture: {e}")
            return ActionFailed(errors=str(e))
            
    @action(name="calibrate_depth_camera", description="Perform real camera calibration using checkerboard pattern")
    def calibrate_depth_camera(self, 
                              checkerboard_width: int = 10,
                              checkerboard_height: int = 7,
                              square_size_mm: float = 25.0,
                              num_images: int = 15,
                              capture_delay: float = 2.0,
                              save_calibration: bool = True) -> ActionResult:
        """Perform real camera calibration using checkerboard pattern.
        
        Args:
            checkerboard_width: Number of internal corners horizontally (default: 10 for 11x8 board)
            checkerboard_height: Number of internal corners vertically (default: 7 for 11x8 board)
            square_size_mm: Physical size of checkerboard squares in millimeters (default: 25.0)
            num_images: Number of calibration images to capture (default: 15)
            capture_delay: Delay between image captures in seconds (default: 2.0)
            save_calibration: Whether to save calibration data to file (default: True)
            
        Returns:
            ActionResult with calibration data and quality metrics
            
        Instructions:
            1. Print an 11x8 checkerboard pattern (10x7 internal corners)
            2. Ensure the checkerboard is flat and well-lit
            3. Move the checkerboard to different positions and angles during capture
            4. Keep the checkerboard fully visible in the camera frame
        """
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
            
        if not ORBBEC_SDK_AVAILABLE:
            return ActionFailed(errors="pyorbbecsdk not available")
        
        # Start camera pipeline if not already running
        pipeline_was_started = self.robot_interface._pipeline_started
        if not pipeline_was_started:
            if not self.robot_interface._start_camera_pipeline():
                return ActionFailed(errors="Failed to start camera pipeline for calibration")
        
        try:
            checkerboard_size = (checkerboard_width, checkerboard_height)
            
            # Prepare object points (3D points in real world space)
            objp = np.zeros((checkerboard_width * checkerboard_height, 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_width, 0:checkerboard_height].T.reshape(-1, 2)
            objp *= square_size_mm
            
            # Arrays to store object points and image points from all images
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            # Create calibration images directory
            cal_images_dir = os.path.join(self.robot_interface.data_path, "calibration_images")
            os.makedirs(cal_images_dir, exist_ok=True)
            
            captured_images = 0
            failed_captures = 0
            max_failed_attempts = num_images * 3  # Allow some failed attempts
            
            self.logger.log_info(f"Starting camera calibration...")
            self.logger.log_info(f"Checkerboard: {checkerboard_width}x{checkerboard_height} internal corners (11x8 squares)")
            self.logger.log_info(f"Square size: {square_size_mm}mm")
            self.logger.log_info(f"Target images: {num_images}")
            self.logger.log_info("Move the checkerboard to different positions and angles between captures")
            
            while captured_images < num_images and failed_captures < max_failed_attempts:
                # Capture image
                rgb_image = self.robot_interface.capture_rgb_image()
                if rgb_image is None:
                    failed_captures += 1
                    self.logger.log_warning("Failed to capture image, retrying...")
                    time.sleep(1)
                    continue
                
                # Convert to grayscale for corner detection
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                
                if ret:
                    # Refine corner positions for sub-pixel accuracy
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Store the points
                    objpoints.append(objp)
                    imgpoints.append(corners_refined)
                    captured_images += 1
                    
                    # Save the calibration image with corners drawn
                    img_with_corners = rgb_image.copy()
                    cv2.drawChessboardCorners(img_with_corners, checkerboard_size, corners_refined, ret)
                    
                    image_filename = f"calibration_{captured_images:02d}.jpg"
                    image_path = os.path.join(cal_images_dir, image_filename)
                    cv2.imwrite(image_path, img_with_corners)
                    
                    self.logger.log_info(f"Captured calibration image {captured_images}/{num_images}")
                    
                    if captured_images < num_images:
                        self.logger.log_info(f"Move checkerboard to new position. Next capture in {capture_delay} seconds...")
                        time.sleep(capture_delay)
                else:
                    failed_captures += 1
                    if failed_captures % 5 == 0:  # Log every 5th failure
                        self.logger.log_warning(f"Checkerboard not detected in image. Failed attempts: {failed_captures}")
                        self.logger.log_info("Ensure checkerboard is fully visible, well-lit, and flat")
                    time.sleep(0.5)
            
            if captured_images < num_images:
                error_msg = f"Only captured {captured_images}/{num_images} valid images. Calibration requires more images."
                self.logger.log_error(error_msg)
                return ActionFailed(errors=error_msg)
            
            self.logger.log_info("Performing camera calibration calculation...")
            
            # Perform camera calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if not ret:
                return ActionFailed(errors="Camera calibration calculation failed")
            
            # Calculate reprojection error
            total_error = 0
            max_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                                camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
                max_error = max(max_error, error)
            
            mean_reprojection_error = total_error / len(objpoints)
            
            # Extract individual parameters for easier access
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            
            # Create comprehensive calibration data
            calibration_data = {
                # Core OpenCV calibration results
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.flatten().tolist(),
                
                # Image dimensions
                "image_width": gray.shape[1],
                "image_height": gray.shape[0],
                
                # Individual parameters for point cloud generation
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                
                # Depth camera specific
                "depth_scale": 1000.0,  # Orbbec typically uses mm, adjust if needed
                
                # Quality metrics
                "mean_reprojection_error": float(mean_reprojection_error),
                "max_reprojection_error": float(max_error),
                "calibration_success": True,
                
                # Calibration metadata
                "calibration_date": datetime.datetime.now().isoformat(),
                "num_images_used": captured_images,
                "checkerboard_size": list(checkerboard_size),
                "square_size_mm": square_size_mm,
                "opencv_version": cv2.__version__,
                
                # Camera information
                "camera_model": "pinhole",
                "sensor_type": "RGB",
                "device_info": {
                    "manufacturer": "Orbbec",
                    "model": "DaiBai DCW2",
                    "calibration_method": "opencv_checkerboard"
                }
            }
            
            # Evaluate calibration quality
            quality_assessment = self._assess_calibration_quality(mean_reprojection_error, max_error)
            calibration_data["quality_assessment"] = quality_assessment
            
            if save_calibration:
                # Save calibration data
                calibration_path = os.path.join(self.robot_interface.data_path, "camera_calibration.json")
                with open(calibration_path, "w") as f:
                    json.dump(calibration_data, f, indent=2)
                
                # Reload calibration in the robot interface
                self.robot_interface.camera_calibration = self.robot_interface._load_camera_calibration()
                
                self.logger.log_info(f"Saved camera calibration to {calibration_path}")
            
            # Log calibration results
            self.logger.log_info("Camera calibration completed successfully!")
            self.logger.log_info(f"Mean reprojection error: {mean_reprojection_error:.3f} pixels")
            self.logger.log_info(f"Max reprojection error: {max_error:.3f} pixels")
            self.logger.log_info(f"Quality assessment: {quality_assessment}")
            self.logger.log_info(f"Focal lengths: fx={fx:.1f}, fy={fy:.1f}")
            self.logger.log_info(f"Principal point: cx={cx:.1f}, cy={cy:.1f}")
            
            return ActionSucceeded(data=calibration_data)
            
        except Exception as e:
            self.logger.log_error(f"Error during camera calibration: {e}")
            return ActionFailed(errors=str(e))
            
        finally:
            # Stop camera pipeline if we started it
            if not pipeline_was_started and self.robot_interface._pipeline_started:
                self.robot_interface._stop_camera_pipeline()
    
    def _assess_calibration_quality(self, mean_error: float, max_error: float) -> str:
        """Assess the quality of camera calibration based on reprojection errors."""
        if mean_error < 0.5 and max_error < 1.0:
            return "Excellent"
        elif mean_error < 1.0 and max_error < 2.0:
            return "Good"
        elif mean_error < 1.5 and max_error < 3.0:
            return "Acceptable"
        else:
            return "Poor - Consider recalibrating with better images"
            
    @action(name="validate_calibration", description="Validate current camera calibration quality")
    def validate_calibration(self) -> ActionResult:
        """Validate the current camera calibration by testing point cloud accuracy."""
        if self.robot_interface is None:
            self.logger.log_error("Robot interface not initialized")
            return ActionFailed(errors="Robot interface not initialized")
            
        try:
            calibration = self.robot_interface.camera_calibration
            
            # Check if calibration data exists and has required fields
            required_fields = ["fx", "fy", "cx", "cy", "camera_matrix", "distortion_coefficients"]
            missing_fields = [field for field in required_fields if field not in calibration]
            
            if missing_fields:
                return ActionFailed(errors=f"Missing calibration fields: {missing_fields}")
            
            # Basic parameter validation
            validation_results = {
                "calibration_loaded": True,
                "calibration_date": calibration.get("calibration_date", "Unknown"),
                "image_dimensions": f"{calibration.get('image_width', 'Unknown')}x{calibration.get('image_height', 'Unknown')}",
                "focal_lengths": f"fx={calibration['fx']:.1f}, fy={calibration['fy']:.1f}",
                "principal_point": f"cx={calibration['cx']:.1f}, cy={calibration['cy']:.1f}",
                "reprojection_error": calibration.get("mean_reprojection_error", "Unknown"),
                "quality_assessment": calibration.get("quality_assessment", "Unknown"),
                "warnings": []
            }
            
            # Check for reasonable parameter values
            fx, fy = calibration["fx"], calibration["fy"]
            cx, cy = calibration["cx"], calibration["cy"]
            width, height = calibration.get("image_width", 640), calibration.get("image_height", 480)
            
            # Validate focal lengths (should be positive and reasonable for the sensor)
            if fx <= 0 or fy <= 0:
                validation_results["warnings"].append("Invalid focal lengths (should be positive)")
            elif fx < 200 or fx > 1000 or fy < 200 or fy > 1000:
                validation_results["warnings"].append("Focal lengths seem unusual for this camera")
            
            # Validate principal point (should be near image center)
            if abs(cx - width/2) > width/4 or abs(cy - height/2) > height/4:
                validation_results["warnings"].append("Principal point far from image center")
            
            # Check reprojection error if available
            if "mean_reprojection_error" in calibration:
                error = calibration["mean_reprojection_error"]
                if error > 2.0:
                    validation_results["warnings"].append(f"High reprojection error: {error:.3f} pixels")
            
            # Test point cloud generation if camera is available
            if self.robot_interface._pipeline_started:
                try:
                    _, _, _, point_cloud = self.robot_interface.capture_synchronized_data()
                    if point_cloud is not None:
                        validation_results["point_cloud_test"] = {
                            "success": True,
                            "num_points": len(point_cloud),
                            "point_range": {
                                "x": [float(np.min(point_cloud[:, 0])), float(np.max(point_cloud[:, 0]))],
                                "y": [float(np.min(point_cloud[:, 1])), float(np.max(point_cloud[:, 1]))],
                                "z": [float(np.min(point_cloud[:, 2])), float(np.max(point_cloud[:, 2]))]
                            }
                        }
                    else:
                        validation_results["point_cloud_test"] = {"success": False, "error": "No point cloud generated"}
                except Exception as e:
                    validation_results["point_cloud_test"] = {"success": False, "error": str(e)}
            else:
                validation_results["point_cloud_test"] = {"success": False, "error": "Camera pipeline not running"}
            
            # Overall assessment
            if len(validation_results["warnings"]) == 0:
                validation_results["overall_status"] = "Good"
            elif len(validation_results["warnings"]) <= 2:
                validation_results["overall_status"] = "Acceptable with warnings"
            else:
                validation_results["overall_status"] = "Poor - recommend recalibration"
            
            self.logger.log_info(f"Calibration validation complete. Status: {validation_results['overall_status']}")
            if validation_results["warnings"]:
                for warning in validation_results["warnings"]:
                    self.logger.log_warning(f"Calibration warning: {warning}")
            
            # Debug logging to see what we're actually returning
            self.logger.log_info(f"DEBUG: validation_results type: {type(validation_results)}")
            self.logger.log_info(f"DEBUG: validation_results keys: {list(validation_results.keys())}")
            self.logger.log_info(f"DEBUG: overall_status value: {validation_results['overall_status']}")
            
            return ActionSucceeded(data=validation_results)
            
        except Exception as e:
            self.logger.log_error(f"Error validating calibration: {e}")
            return ActionFailed(errors=str(e))

    @action(name="generate_position_variations", description="Generate and preview position variations for dataset collection")
    def generate_position_variations(self, 
                                   num_positions: int = 12,
                                   num_approaches: int = 3,
                                   num_wrist_rotations: int = 3) -> ActionResult:
        """Generate and return position variations without executing them."""
        
        try:
            # Generate all variations
            positions = self.robot_interface.generate_angular_positions(num_positions)
            
            variations_preview = []
            total_combinations = 0
            
            for pos_name, angle in positions.items():
                approach_variations = self.robot_interface.generate_approach_variations(angle)
                wrist_rotations = self.robot_interface.generate_wrist_rotations()
                
                position_data = {
                    "position_name": pos_name,
                    "base_angle": angle,
                    "approach_variations": approach_variations[:num_approaches],
                    "wrist_rotations": wrist_rotations[:num_wrist_rotations],
                    "combinations_for_position": num_approaches * num_wrist_rotations
                }
                
                variations_preview.append(position_data)
                total_combinations += position_data["combinations_for_position"]
            
            self.logger.log_info(f"Generated {total_combinations} total trajectory variations across {num_positions} positions")
            
            return ActionSucceeded(data={
                "total_combinations": total_combinations,
                "num_positions": num_positions,
                "num_approaches": num_approaches,
                "num_wrist_rotations": num_wrist_rotations,
                "position_variations": variations_preview,
                "estimated_time_minutes": total_combinations * 0.5  # Rough estimate: 30 seconds per trajectory
            })
            
        except Exception as e:
            return ActionFailed(errors=str(e))


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    
    # Load environment variables from root directory
    root_dir = Path(__file__).parent.parent
    load_dotenv(root_dir / '.env')
    
    # Get configuration from environment
    definition_path = Path(os.getenv("NODE_DEFINITION", "nodes/default.node.yaml"))
    node_url = os.getenv("ROBOT_NODE_URL", "http://localhost:2000")
    
    # Load node definition
    node_definition = None
    if definition_path.exists():
        node_definition = NodeDefinition.from_yaml(definition_path)
    
    # Create custom config with node_url
    node_config = RobotArmConfig(node_url=node_url)
    
    # Initialize and start node
    robot_arm_node = RobotArmNode(
        node_definition=node_definition,
        node_config=node_config
    )
    
    print(f"Starting DOFBOT Pro Expert Node (definition: {definition_path}, url: {node_url})")
    robot_arm_node.start_node()
