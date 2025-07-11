"""
Camera Manager Component for DOFBOT Pro

Handles all camera-related operations including:
- Orbbec depth camera initialization and management
- Frame capture and processing
- Camera calibration loading
- Pipeline management
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from queue import Queue, Empty
import threading
from functools import wraps

# Try to import Orbbec SDK
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, FrameSet, OBError
    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    Pipeline = None
    Config = None
    OBSensorType = None
    OBFormat = None
    FrameSet = None
    OBError = None


def require_valid_frame(func):
    """Decorator to ensure frame is valid before processing."""
    @wraps(func)
    def wrapper(self, frame, *args, **kwargs):
        if frame is None:
            return None
        return func(self, frame, *args, **kwargs)
    return wrapper


def handle_camera_errors(operation_name: str, return_value=None):
    """Decorator for consistent camera error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.log_error(f"Error in {operation_name}: {e}")
                return return_value
        return wrapper
    return decorator


class CameraManager:
    """Manages camera operations for DOFBOT Pro robot."""
    
    def __init__(self, logger, data_path: str = "./captures"):
        """Initialize camera manager.
        
        Args:
            logger: Logger instance for logging
            data_path: Path for saving captured data
        """
        self.logger = logger
        self.data_path = data_path
        
        # Camera state
        self.orbbec_pipeline = None
        self.orbbec_config = None
        self._frame_callback_active = False
        self._frameset_queue = Queue(maxsize=10)
        self._simple_capture = None  # For simple RGB capture
        
        # Initialize camera if available
        if ORBBEC_SDK_AVAILABLE:
            self._init_orbbec_camera()
        else:
            self.logger.log_warning("Orbbec SDK not available - using fallback camera")
    
    @handle_camera_errors("Orbbec camera initialization")
    def _init_orbbec_camera(self):
        """Initialize Orbbec depth camera with always-on pipeline."""
        if not ORBBEC_SDK_AVAILABLE:
            return
        
        self.orbbec_pipeline = Pipeline()
        self.orbbec_config = Config()
        
        # Setup color and depth streams
        self._setup_color_stream()
        self._setup_depth_stream()
        
        # Start pipeline with callback
        self.orbbec_pipeline.start(self.orbbec_config, self._on_frame_callback)
        self._frame_callback_active = True
        self.logger.log_info("Orbbec camera pipeline started successfully")
    
    def _setup_color_stream(self):
        """Setup color stream with fallback options."""
        color_profiles = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if not color_profiles or color_profiles.get_count() == 0:
            self.logger.log_error("No color profiles available")
            return
        
        self.logger.log_debug(f"Found {color_profiles.get_count()} color profiles")
        
        # Try different profile configurations in order of preference
        profile_configs = [
            {"width": 640, "height": 480, "format": OBFormat.RGB, "fps": 15},
            {"width": 640, "height": 480, "format": OBFormat.RGB, "fps": 10},
            {"width": 640, "height": 0, "format": OBFormat.RGB, "fps": 30},  # Any height
        ]
        
        color_profile = self._try_profile_configs(color_profiles, profile_configs, "color")
        
        # Final fallback to default profile
        if color_profile is None:
            try:
                color_profile = color_profiles.get_default_video_stream_profile()
                if color_profile:
                    self.logger.log_info("Using default color profile")
            except (OBError, Exception):
                pass
        
        if color_profile:
            self.orbbec_config.enable_stream(color_profile)
            self.logger.log_info(f"Color stream configured: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
        else:
            self.logger.log_error("No suitable color profile found")
    
    def _setup_depth_stream(self):
        """Setup depth stream with fallback options."""
        depth_profiles = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if not depth_profiles or depth_profiles.get_count() == 0:
            self.logger.log_error("No depth profiles available")
            return
        
        self.logger.log_debug(f"Found {depth_profiles.get_count()} depth profiles")
        
        # Try different profile configurations in order of preference
        profile_configs = [
            {"width": 640, "height": 400, "format": OBFormat.Y11, "fps": 15},
            {"width": 640, "height": 400, "format": OBFormat.Y12, "fps": 15},
            {"width": 640, "height": 0, "format": OBFormat.Y16, "fps": 30},  # Any height
        ]
        
        depth_profile = self._try_profile_configs(depth_profiles, profile_configs, "depth")
        
        # Final fallback to default profile
        if depth_profile is None:
            try:
                depth_profile = depth_profiles.get_default_video_stream_profile()
                if depth_profile:
                    self.logger.log_info("Using default depth profile")
            except (OBError, Exception):
                pass
        
        if depth_profile:
            self.orbbec_config.enable_stream(depth_profile)
            self.logger.log_info(f"Depth stream configured: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps")
        else:
            self.logger.log_error("No suitable depth profile found")
    
    def _try_profile_configs(self, profiles, configs: List[Dict], stream_type: str):
        """Try multiple profile configurations with fallbacks.
        
        Args:
            profiles: Stream profile list from Orbbec SDK
            configs: List of profile configuration dictionaries
            stream_type: Type of stream for logging ("color" or "depth")
            
        Returns:
            Video stream profile or None if all configs fail
        """
        for config in configs:
            try:
                profile = profiles.get_video_stream_profile(
                    config["width"], config["height"], config["format"], config["fps"]
                )
                if profile:
                    self.logger.log_info(f"Using {stream_type} profile: {config['width']}x{config['height']}@{config['fps']}fps")
                    return profile
            except (OBError, Exception):
                continue
        
        return None
    
    @handle_camera_errors("frame callback", return_value=None)
    def _on_frame_callback(self, frames: FrameSet):
        """Callback for processing incoming frames from Orbbec camera."""
        if not self._frame_callback_active:
            return
        
        # Add frames to queue (non-blocking)
        if not self._frameset_queue.full():
            self._frameset_queue.put_nowait(frames)
    
    @require_valid_frame
    @handle_camera_errors("frame conversion", return_value=None)
    def frame_to_bgr_image(self, frame) -> Optional[np.ndarray]:
        """Convert Orbbec frame to BGR image format.
        
        Args:
            frame: Orbbec color frame
            
        Returns:
            BGR image as numpy array or None if conversion fails
        """
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        
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
            self.logger.log_error(f"Unsupported color format: {color_format}")
            return None
        
        return image
    
    @handle_camera_errors("RGB image capture", return_value=None)
    def capture_rgb_image(self) -> Optional[np.ndarray]:
        """Capture a single RGB image from the always-on camera pipeline.
        
        Returns:
            BGR image as numpy array or None if capture fails
        """
        if not ORBBEC_SDK_AVAILABLE or not self._frame_callback_active:
            return None
        
        try:
            # Get frame from queue with timeout
            frameset = self._frameset_queue.get(timeout=2.0)
            if frameset is None:
                return None
            
            color_frame = frameset.get_color_frame()
            if not color_frame:
                return None
            
            return self.frame_to_bgr_image(color_frame)
            
        except Empty:
            self.logger.log_warning("No frames available in queue")
            return None
    
    @handle_camera_errors("simple frame capture", return_value=None)
    def capture_frame_simple(self) -> Optional[np.ndarray]:
        """Capture frame using simple OpenCV VideoCapture (Yahboom's approach).
        
        This is more reliable for basic RGB capture without depth.
        
        Returns:
            BGR image as numpy array or None if capture fails
        """
        if self._simple_capture is None:
            self._simple_capture = cv2.VideoCapture(0)
            self._simple_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._simple_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.logger.log_info("Initialized simple camera capture")
        
        ret, frame = self._simple_capture.read()
        return frame if ret else None
    
    @handle_camera_errors("depth data capture", return_value=(None, None, None))
    def capture_depth_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture depth data from Orbbec camera.
        
        Returns:
            Tuple of (depth_image, depth_colormap, point_cloud) or (None, None, None) if failed
        """
        if not ORBBEC_SDK_AVAILABLE or not self._frame_callback_active:
            return None, None, None
        
        try:
            # Get frame from queue
            frameset = self._frameset_queue.get(timeout=2.0)
            if frameset is None:
                return None, None, None
            
            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                return None, None, None
            
            # Convert to numpy array
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            
            # Create a colorized version of the depth map for visualization
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Generate point cloud (simplified version)
            point_cloud = self._generate_point_cloud_simple(depth_image)
            
            return depth_image, depth_colormap, point_cloud
            
        except Empty:
            self.logger.log_warning("No frames available for depth capture")
            return None, None, None
    
    @handle_camera_errors("point cloud generation", return_value=None)
    def _generate_point_cloud_simple(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate a simple point cloud from depth image.
        
        Args:
            depth_image: Depth image as numpy array
            
        Returns:
            Point cloud as numpy array or None if generation fails
        """
        # Use default camera parameters if calibration not available
        fx, fy = 525.0, 525.0  # Default focal lengths
        cx, cy = 320.0, 240.0  # Default principal point
        depth_scale = 1000.0   # Default depth scale
        
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth image to meters
        depth_meters = depth_image.astype(np.float32) / depth_scale
        
        # Calculate 3D coordinates
        x = (u - cx) * depth_meters / fx
        y = (v - cy) * depth_meters / fy
        z = depth_meters
        
        # Stack coordinates and filter out invalid points
        points = np.stack((x, y, z), axis=-1)
        valid_mask = (depth_image > 0) & (depth_image < 10000)  # Filter reasonable depth values
        point_cloud = points[valid_mask]
        
        return point_cloud
    
    def load_camera_calibration(self) -> Dict[str, Any]:
        """Load camera calibration parameters.
        
        Returns:
            Dictionary containing calibration parameters
        """
        # Default calibration parameters
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
            calibration_file = Path(self.data_path) / "camera_calibration.json"
            if calibration_file.exists():
                with open(calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                
                # Extract camera matrix if available
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
                self.logger.log_info("No calibration file found, using defaults")
                return default_calibration
                
        except Exception as e:
            self.logger.log_error(f"Error loading camera calibration: {e}")
            return default_calibration
    
    @handle_camera_errors("camera pipeline stop")
    def stop_camera_pipeline(self) -> None:
        """Stop the camera pipeline and cleanup resources."""
        self._frame_callback_active = False
        
        if self.orbbec_pipeline:
            self.orbbec_pipeline.stop()
            self.logger.log_info("Orbbec camera pipeline stopped")
        
        if self._simple_capture:
            self._simple_capture.release()
            self.logger.log_info("Simple camera capture released")
    
    def cleanup(self):
        """Cleanup all camera resources."""
        self.stop_camera_pipeline()
        
        # Clear queue
        while not self._frameset_queue.empty():
            try:
                self._frameset_queue.get_nowait()
            except Empty:
                break
        
        self.logger.log_info("Camera manager cleanup completed")
    
    def is_available(self) -> bool:
        """Check if camera is available and working.
        
        Returns:
            True if camera is available, False otherwise
        """
        if ORBBEC_SDK_AVAILABLE and self._frame_callback_active:
            return True
        elif self._simple_capture and self._simple_capture.isOpened():
            return True
        else:
            return False
    
    def get_queue_size(self) -> int:
        """Get current frame queue size for debugging.
        
        Returns:
            Number of frames in queue
        """
        return self._frameset_queue.qsize()
