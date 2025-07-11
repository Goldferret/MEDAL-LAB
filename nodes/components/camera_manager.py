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
import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from queue import Queue, Empty
import threading
from functools import wraps

# Try to import Orbbec SDK
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, FrameSet, OBError, OBAlignMode
    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    Pipeline = None
    Config = None
    OBSensorType = None
    OBFormat = None
    FrameSet = None
    OBError = None
    OBAlignMode = None


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
        self._frameset_queue = Queue(maxsize=30)  # Increased from 10 to 30 for 10Hz recording
        
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
        
        # Enable frame synchronization if supported (optional for this device)
        try:
            self.orbbec_pipeline.enable_frame_sync()
            self.logger.log_info("Frame synchronization enabled")
        except Exception as e:
            self.logger.log_info("Frame sync not supported - using alignment only")
        
        # Set alignment mode for better RGB-Depth correspondence (more important than sync)
        try:
            # OBAlignMode is imported with the * import from pyorbbecsdk
            self.orbbec_config.set_align_mode(OBAlignMode.HW_MODE)  # Hardware alignment if available
            self.logger.log_info("Hardware alignment enabled")
        except Exception as e:
            try:
                self.orbbec_config.set_align_mode(OBAlignMode.SW_MODE)  # Software fallback
                self.logger.log_info("Software alignment enabled")
            except Exception as e2:
                self.logger.log_info("Alignment not available - using raw frames")
        
        # Start pipeline with callback
        self.orbbec_pipeline.start(self.orbbec_config, self._on_frame_callback)
        self._frame_callback_active = True
        
        # Pipeline warmup - wait for stable frame production
        self.logger.log_info("Warming up camera pipeline...")
        warmup_start = time.time()
        
        # Wait for queue to fill up and stabilize
        target_warmup_frames = 10  # Wait for 10 frames to ensure stability
        while self._frameset_queue.qsize() < target_warmup_frames and (time.time() - warmup_start) < 5.0:
            time.sleep(0.1)  # Wait 100ms between checks
        
        # Additional stability wait - let a few more frames accumulate
        time.sleep(0.5)
        
        warmup_time = time.time() - warmup_start
        queue_size = self._frameset_queue.qsize()
        self.logger.log_info(f"Camera pipeline ready after {warmup_time:.1f}s with {queue_size} frames queued")
        self.logger.log_info("Orbbec camera pipeline started successfully with 10Hz optimization")
    
    def _setup_color_stream(self):
        """Setup color stream optimized for 10Hz recording."""
        color_profiles = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if not color_profiles or color_profiles.get_count() == 0:
            self.logger.log_error("No color profiles available")
            return
        
        self.logger.log_debug(f"Found {color_profiles.get_count()} color profiles")
        
        # Optimized profile configurations for 10Hz recording
        # Based on DaBai DCW2 specs: RGB 1920×1080 @ 5/10/15/30 fps
        profile_configs = [
            {"width": 1920, "height": 1080, "format": OBFormat.MJPG, "fps": 15},  # Optimal for 10Hz
            {"width": 1920, "height": 1080, "format": OBFormat.MJPG, "fps": 10},  # Exact match
            {"width": 640, "height": 480, "format": OBFormat.RGB, "fps": 15},     # Lower res fallback
            {"width": 640, "height": 480, "format": OBFormat.RGB, "fps": 10},     # Lower res exact
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
        """Setup depth stream optimized for 10Hz recording."""
        depth_profiles = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if not depth_profiles or depth_profiles.get_count() == 0:
            self.logger.log_error("No depth profiles available")
            return
        
        self.logger.log_debug(f"Found {depth_profiles.get_count()} depth profiles")
        
        # Optimized profile configurations for 10Hz recording
        # Based on DaBai DCW2 specs: Depth 640×400 @ 5/10/15 fps
        profile_configs = [
            {"width": 640, "height": 400, "format": OBFormat.Y16, "fps": 15},  # Optimal for 10Hz
            {"width": 640, "height": 400, "format": OBFormat.Y16, "fps": 10},  # Exact match
            {"width": 640, "height": 400, "format": OBFormat.Y11, "fps": 15},  # Alternative format
            {"width": 640, "height": 400, "format": OBFormat.Y12, "fps": 15},  # Alternative format
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
        """Optimized callback for processing incoming frames from Orbbec camera."""
        if not self._frame_callback_active or frames is None:
            return
        
        # Optimized queue management for 10Hz recording
        # Keep queue at ~70% capacity to ensure frames are always available
        target_queue_size = int(self._frameset_queue.maxsize * 0.7)  # ~21 frames
        
        # If queue is above target, remove oldest frames to make room
        while self._frameset_queue.qsize() >= target_queue_size:
            try:
                self._frameset_queue.get_nowait()  # Remove oldest frame
            except Empty:
                break
        
        # Add new frameset (non-blocking)
        try:
            self._frameset_queue.put_nowait(frames)
        except:
            # Queue still full somehow, skip this frame
            pass
    
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
    
    @handle_camera_errors("depth data capture", return_value=(None, None, None, None))
    def capture_synchronized_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized RGB, depth, and point cloud data from single frameset.
        
        This is the standardized interface method for recording and other operations
        that need complete camera data. Uses single frameset to ensure synchronization.
        
        Returns:
            Tuple of (rgb_image, depth_image, depth_colormap, point_cloud_data)
            Any element can be None if capture failed or not available
        """
        if not ORBBEC_SDK_AVAILABLE or not self._frame_callback_active:
            return None, None, None, None
        
        try:
            # Get single frameset from queue with appropriate timeout for 10Hz recording
            # 500ms timeout allows for occasional delays while maintaining 10Hz performance
            frameset = self._frameset_queue.get(timeout=0.5)  # 500ms timeout
            if frameset is None:
                return None, None, None, None
            
            # Extract all data from the same synchronized frameset
            rgb_image = None
            depth_image = None
            depth_colormap = None
            point_cloud_data = None
            
            # Get RGB frame
            color_frame = frameset.get_color_frame()
            if color_frame:
                rgb_image = self.frame_to_bgr_image(color_frame)
            
            # Get depth frame and process all depth-related data
            depth_frame = frameset.get_depth_frame()
            if depth_frame:
                # Convert depth frame to numpy array
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape((height, width))
                
                # Create colorized depth map
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # Generate point cloud if we have both color and depth
                if color_frame and depth_frame:
                    try:
                        # Get camera parameters for point cloud generation
                        camera_param = self._get_camera_param()
                        if camera_param:
                            # Use Orbbec's built-in point cloud generation
                            points = frameset.get_point_cloud(camera_param)
                            if points is not None and len(points) > 0:
                                # Convert to numpy array format
                                point_cloud_data = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float32)
                    except Exception as e:
                        # Point cloud generation failed, continue without it
                        pass
            
            return rgb_image, depth_image, depth_colormap, point_cloud_data
            
        except Empty:
            # No frames available - this is normal at 10Hz if camera is slower
            return None, None, None, None
        except Exception as e:
            self.logger.log_error(f"Error in synchronized capture: {e}")
            return None, None, None, None
    
    def _get_camera_param(self):
        """Get camera parameters for point cloud generation."""
        if not hasattr(self, '_camera_param'):
            try:
                if self.orbbec_pipeline:
                    device = self.orbbec_pipeline.get_device()
                    self._camera_param = device.get_camera_param()
                    return self._camera_param
            except Exception:
                self._camera_param = None
                return None
        return self._camera_param
    
    
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
    
    def save_debug_frame(self, bgr_image: np.ndarray, filename_prefix: str, data_path: str = "./captures") -> bool:
        """Save debug frame for analysis.
        
        Args:
            bgr_image: BGR image to save
            filename_prefix: Prefix for the filename (e.g., "scan_debug_pos180_attempt1")
            data_path: Directory to save the image
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_filename = f"{filename_prefix}_{timestamp}.jpg"
            frame_path = Path(data_path) / frame_filename
            frame_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(frame_path), bgr_image)
            self.logger.log_info(f"Saved debug frame: {frame_filename}")
            return True
        except Exception as e:
            self.logger.log_warning(f"Failed to save debug frame: {e}")
            return False
    
    def save_color_mask_debug(self, bgr_image: np.ndarray, color: str, filename_prefix: str, 
                             vision_detector, data_path: str = "./captures") -> bool:
        """Save color mask for debugging color detection.
        
        Args:
            bgr_image: Original BGR image
            color: Color to create mask for
            filename_prefix: Prefix for the filename (e.g., "color_mask_red_pos180_attempt1")
            vision_detector: Vision detector instance for creating masks
            data_path: Directory to save the mask
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert to HSV for color detection
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            
            # Create color mask using vision detector
            color_mask = vision_detector.create_color_mask(hsv_image, color)
            
            # Create a 3-channel version for saving (white mask on black background)
            color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            
            # Save the mask
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            mask_filename = f"{filename_prefix}_{timestamp}.jpg"
            mask_path = Path(data_path) / mask_filename
            mask_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(mask_path), color_mask_bgr)
            self.logger.log_info(f"Saved color mask: {mask_filename}")
            return True
            
        except Exception as e:
            self.logger.log_warning(f"Failed to save color mask: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if camera is available and working.
        
        Returns:
            True if camera is available, False otherwise
        """
        return ORBBEC_SDK_AVAILABLE and self._frame_callback_active
    
    def get_queue_size(self) -> int:
        """Get current frame queue size for debugging.
        
        Returns:
            Number of frames in queue
        """
        return self._frameset_queue.qsize()
