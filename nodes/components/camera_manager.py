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

# Try to import Orbbec SDK (basic components first)
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

# Try to import additional SDK components for device properties
ORBBEC_PROPERTIES_AVAILABLE = False
if ORBBEC_SDK_AVAILABLE:
    try:
        from pyorbbecsdk import OBPropertyID
        ORBBEC_PROPERTIES_AVAILABLE = True
    except ImportError:
        ORBBEC_PROPERTIES_AVAILABLE = False
        OBPropertyID = None


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
        
        # Pipeline mode management
        self._pipeline_mode = "recording"  # "recording" or "scanning"
        self.recording_pipeline = None
        self.recording_config = None
        self.scanning_pipeline = None
        self.scanning_config = None
        
        # Recording state tracking for scanning frames
        self._was_recording_before_scan = False
        self._scanning_frames_buffer = []
        self._scanning_session_id = None
        self._is_scanning_active = False  # Flag to indicate scanning is in progress
        
        # Depth filtering components (using device properties instead of filter classes)
        self.depth_filtering_enabled = False
        self.device = None  # Will store device reference for property access
        
        # Initialize camera if available
        if ORBBEC_SDK_AVAILABLE:
            self._init_dual_pipeline_system()
            # Initialize depth filtering after pipeline setup
            self._init_depth_filtering()
        else:
            self.logger.log_warning("Orbbec SDK not available - using fallback camera")
    
    @handle_camera_errors("Dual pipeline system initialization")
    def _init_dual_pipeline_system(self):
        """Initialize both recording and scanning pipelines."""
        if not ORBBEC_SDK_AVAILABLE:
            return
        
        # Initialize recording pipeline (main pipeline)
        self.recording_pipeline = Pipeline()
        self.recording_config = Config()
        self._setup_recording_streams()
        
        # Initialize scanning pipeline (optimized for speed)
        self.scanning_pipeline = Pipeline()
        self.scanning_config = Config()
        self._setup_scanning_streams_optimized()
        
        # Set current pipeline references for backward compatibility
        self.orbbec_pipeline = self.recording_pipeline
        self.orbbec_config = self.recording_config
        
        # Start in recording mode
        self._start_recording_pipeline()
        
        self.logger.log_info("Dual pipeline system initialized successfully")
    
    def _init_depth_filtering(self):
        """Initialize depth filtering using device properties instead of filter classes."""
        if not ORBBEC_SDK_AVAILABLE:
            self.logger.log_info("Orbbec SDK not available - depth filtering disabled")
            return
            
        try:
            # Get device reference from pipeline
            if hasattr(self.scanning_pipeline, 'get_device'):
                self.device = self.scanning_pipeline.get_device()
            elif hasattr(self.recording_pipeline, 'get_device'):
                self.device = self.recording_pipeline.get_device()
            else:
                self.logger.log_info("Cannot get device reference - depth filtering disabled")
                return
            
            # Enable software depth filtering if available
            if ORBBEC_PROPERTIES_AVAILABLE and self.device:
                try:
                    # Enable software depth filter (equivalent to spatial filtering)
                    self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, True)
                    self.logger.log_info("Software depth filtering enabled")
                    self.depth_filtering_enabled = True
                except Exception as e:
                    self.logger.log_info(f"Software depth filtering not supported: {e}")
            
            if not self.depth_filtering_enabled:
                self.logger.log_info("Using enhanced software-based depth processing")
                self.depth_filtering_enabled = True  # Enable our custom processing
                
        except Exception as e:
            self.logger.log_error(f"Failed to initialize depth filtering: {e}")
            self.depth_filtering_enabled = False
    
    def _apply_depth_filtering(self, depth_frame):
        """Apply enhanced depth processing using available methods."""
        if not self.depth_filtering_enabled or depth_frame is None:
            return depth_frame
        
        # Since hardware filters aren't available, we'll enhance the existing processing
        # The actual filtering will be done in the processing pipeline with OpenCV
        return depth_frame
    
    def _setup_recording_streams(self):
        """Setup recording pipeline streams (same as original setup)."""
        # Setup color and depth streams for recording
        self._setup_color_stream_for_pipeline(self.recording_pipeline, self.recording_config, "recording")
        self._setup_depth_stream_for_pipeline(self.recording_pipeline, self.recording_config, "recording")
        
        # Enable frame synchronization if supported
        try:
            self.recording_pipeline.enable_frame_sync()
            self.logger.log_info("Recording pipeline: Frame synchronization enabled")
        except Exception as e:
            self.logger.log_info("Recording pipeline: Frame sync not supported - using alignment only")
        
        # Set alignment mode for better RGB-Depth correspondence
        try:
            self.recording_config.set_align_mode(OBAlignMode.HW_MODE)
            self.logger.log_info("Recording pipeline: Hardware alignment enabled")
        except Exception as e:
            try:
                self.recording_config.set_align_mode(OBAlignMode.SW_MODE)
                self.logger.log_info("Recording pipeline: Software alignment enabled")
            except Exception as e2:
                self.logger.log_info("Recording pipeline: Alignment not available - using raw frames")
    
    def _setup_scanning_streams_optimized(self):
        """Setup scanning pipeline with optimized configuration for speed."""
        # Use the same standardized setup methods as recording pipeline
        self._setup_color_stream_for_pipeline(self.scanning_pipeline, self.scanning_config, "scanning")
        self._setup_depth_stream_for_pipeline(self.scanning_pipeline, self.scanning_config, "scanning")
        
        # Enable frame synchronization if supported (copied from recording pipeline)
        try:
            self.scanning_pipeline.enable_frame_sync()
            self.logger.log_info("Scanning pipeline: Frame synchronization enabled")
        except Exception as e:
            self.logger.log_info("Scanning pipeline: Frame sync not supported - using alignment only")
        
        # Set alignment mode for better RGB-Depth correspondence (copied from recording pipeline)
        try:
            self.scanning_config.set_align_mode(OBAlignMode.HW_MODE)
            self.logger.log_info("Scanning pipeline: Hardware alignment enabled")
        except Exception as e:
            try:
                self.scanning_config.set_align_mode(OBAlignMode.SW_MODE)
                self.logger.log_info("Scanning pipeline: Software alignment enabled")
            except Exception as e2:
                self.logger.log_info("Scanning pipeline: Alignment not available - using raw frames")
        
        self.logger.log_info("Scanning pipeline configured for RGB+Depth operation")
    
    def _setup_color_stream_for_pipeline(self, pipeline, config, pipeline_name):
        """Setup color stream for a specific pipeline."""
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if not color_profiles or color_profiles.get_count() == 0:
            self.logger.log_error(f"No color profiles available for {pipeline_name} pipeline")
            return
        
        self.logger.log_info(f"Found {color_profiles.get_count()} color profiles for {pipeline_name}")
        
        # Use recording-optimized profiles for recording pipeline
        if pipeline_name == "recording":
            profile_configs = [
                {"width": 1920, "height": 1080, "format": OBFormat.MJPG, "fps": 15},
                {"width": 1920, "height": 1080, "format": OBFormat.MJPG, "fps": 10},
                {"width": 640, "height": 480, "format": OBFormat.RGB, "fps": 15},
                {"width": 640, "height": 480, "format": OBFormat.RGB, "fps": 10},
            ]
        else:  # scanning pipeline
            profile_configs = [
                {"width": 1280, "height": 720, "format": OBFormat.MJPG, "fps": 15},
                {"width": 640, "height": 480, "format": OBFormat.MJPG, "fps": 15},
            ]
        
        color_profile = self._try_profile_configs(color_profiles, profile_configs, f"{pipeline_name}_color")
        
        # Final fallback to default profile
        if color_profile is None:
            try:
                color_profile = color_profiles.get_default_video_stream_profile()
                if color_profile:
                    self.logger.log_info(f"Using default color profile for {pipeline_name}")
            except (OBError, Exception):
                pass
        
        if color_profile:
            config.enable_stream(color_profile)
            self.logger.log_info(f"{pipeline_name.title()} color stream: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
        else:
            self.logger.log_error(f"No suitable color profile found for {pipeline_name} pipeline")
    
    def _setup_depth_stream_for_pipeline(self, pipeline, config, pipeline_name):
        """Setup depth stream for a specific pipeline."""
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if not depth_profiles or depth_profiles.get_count() == 0:
            self.logger.log_error(f"No depth profiles available for {pipeline_name} pipeline")
            return
        
        self.logger.log_info(f"Found {depth_profiles.get_count()} depth profiles for {pipeline_name}")
        
        # Depth profile configurations
        profile_configs = [
            {"width": 640, "height": 400, "format": OBFormat.Y16, "fps": 15},
            {"width": 640, "height": 480, "format": OBFormat.Y16, "fps": 15},
            {"width": 320, "height": 240, "format": OBFormat.Y16, "fps": 30},
        ]
        
        depth_profile = self._try_profile_configs(depth_profiles, profile_configs, f"{pipeline_name}_depth")
        
        # Final fallback to default profile
        if depth_profile is None:
            try:
                depth_profile = depth_profiles.get_default_video_stream_profile()
                if depth_profile:
                    self.logger.log_info(f"Using default depth profile for {pipeline_name}")
            except (OBError, Exception):
                pass
        
        if depth_profile:
            config.enable_stream(depth_profile)
            self.logger.log_info(f"{pipeline_name.title()} depth stream: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps")
        else:
            self.logger.log_error(f"No suitable depth profile found for {pipeline_name} pipeline")
    
    def _start_recording_pipeline(self):
        """Start the recording pipeline with callback."""
        try:
            # Start pipeline with callback for continuous recording
            self.recording_pipeline.start(self.recording_config, self._on_frame_callback)
            self._frame_callback_active = True
            self._pipeline_mode = "recording"
            
            # Pipeline warmup
            self.logger.log_info("Warming up recording pipeline...")
            if self._wait_for_queue_refill(target_frames=10, timeout_seconds=5.0):
                self.logger.log_info("Recording pipeline ready with fresh frames")
            else:
                self.logger.log_warning("Recording pipeline warmup timeout")
            
        except Exception as e:
            self.logger.log_error(f"Failed to start recording pipeline: {e}")
            raise
    
    def _try_profile_configs(self, profiles, configs: List[Dict], stream_type: str):
        """Try different profile configurations until one works."""
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
    
    def is_scanning_active(self) -> bool:
        """Check if scanning is currently active (for recording integration)."""
        return self._is_scanning_active
    
    def _clear_frameset_queue(self):
        """Clear all frames from the frameset queue."""
        cleared_count = 0
        while not self._frameset_queue.empty():
            try:
                self._frameset_queue.get_nowait()
                cleared_count += 1
            except Empty:
                break
        
        if cleared_count > 0:
            self.logger.log_info(f"Cleared {cleared_count} old frames from queue")
    
    def _wait_for_queue_refill(self, target_frames=10, timeout_seconds=5.0):
        """Wait for the frameset queue to refill after pipeline restart."""
        warmup_start = time.time()
        
        # Wait for queue to fill up and stabilize
        while self._frameset_queue.qsize() < target_frames and (time.time() - warmup_start) < timeout_seconds:
            time.sleep(0.1)
        
        time.sleep(0.5)  # Additional stability wait
        
        warmup_time = time.time() - warmup_start
        queue_size = self._frameset_queue.qsize()
        self.logger.log_info(f"Queue refilled after {warmup_time:.1f}s with {queue_size} frames")
        return queue_size >= target_frames
    
    def switch_to_scanning_mode(self, action_context=None):
        """Switch from recording to scanning pipeline mode."""
        if self._pipeline_mode == "scanning":
            self.logger.log_info("Already in scanning mode")
            return True
        
        try:
            # Record if we were recording before switching
            self._was_recording_before_scan = (self._pipeline_mode == "recording" and self._frame_callback_active)
            
            # Initialize scanning session
            self._scanning_session_id = f"scan_{int(time.time() * 1000)}"
            self._scanning_frames_buffer = []
            self._is_scanning_active = True  # Set scanning flag
            
            self.logger.log_info(f"Switching to scanning mode (session: {self._scanning_session_id})")
            
            # Stop recording pipeline
            if self._pipeline_mode == "recording":
                self._frame_callback_active = False
                self.recording_pipeline.stop()
                time.sleep(0.2)  # Allow pipeline to fully stop
                self.logger.log_info("Recording pipeline stopped for scanning")
            
            # Start scanning pipeline (polling mode, no callback)
            self.scanning_pipeline.start(self.scanning_config)
            time.sleep(0.3)  # Allow pipeline to stabilize
            self._pipeline_mode = "scanning"
            
            self.logger.log_info("Successfully switched to scanning mode")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to switch to scanning mode: {e}")
            # Clear scanning flag on failure
            self._is_scanning_active = False
            # Try to recover by restarting recording pipeline
            self._recover_to_recording_mode()
            return False
    
    def switch_to_recording_mode(self):
        """Switch from scanning to recording pipeline mode."""
        if self._pipeline_mode == "recording":
            self.logger.log_info("Already in recording mode")
            return True
        
        try:
            self.logger.log_info("Switching back to recording mode")
            
            # Stop scanning pipeline
            if self._pipeline_mode == "scanning":
                self.scanning_pipeline.stop()
                time.sleep(0.2)  # Allow pipeline to fully stop
                self.logger.log_info("Scanning pipeline stopped")
            
            # Save scanning frames if we were recording before
            if self._was_recording_before_scan and self._scanning_frames_buffer:
                self._save_scanning_session_data()
            
            # Restart recording pipeline if it was active before
            if self._was_recording_before_scan:
                # Clear old frames from queue before restarting
                self._clear_frameset_queue()
                
                # Restart recording pipeline
                self.recording_pipeline.start(self.recording_config, self._on_frame_callback)
                self._frame_callback_active = True
                
                # Wait for queue to refill with fresh frames
                self.logger.log_info("Waiting for recording pipeline to refill with fresh frames...")
                if self._wait_for_queue_refill(target_frames=10, timeout_seconds=5.0):
                    self.logger.log_info("Recording pipeline restarted with fresh frame queue")
                else:
                    self.logger.log_warning("Recording pipeline queue refill timeout - may have stale frames initially")
            
            self._pipeline_mode = "recording"
            
            # Clean up scanning session
            self._scanning_frames_buffer = []
            self._scanning_session_id = None
            self._was_recording_before_scan = False
            self._is_scanning_active = False  # Clear scanning flag
            
            self.logger.log_info("Successfully switched back to recording mode")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to switch to recording mode: {e}")
            self._recover_to_recording_mode()
            return False
    
    def _recover_to_recording_mode(self):
        """Emergency recovery to recording mode."""
        try:
            self.logger.log_warning("Attempting emergency recovery to recording mode")
            
            # Stop both pipelines
            try:
                self.scanning_pipeline.stop()
            except:
                pass
            try:
                self.recording_pipeline.stop()
            except:
                pass
            
            time.sleep(0.5)
            
            # Restart recording pipeline
            if self._was_recording_before_scan:
                self.recording_pipeline.start(self.recording_config, self._on_frame_callback)
                self._frame_callback_active = True
            
            # Clear scanning state
            self._pipeline_mode = "recording"
            self._is_scanning_active = False
            self.logger.log_info("Emergency recovery completed")
            
        except Exception as e:
            self.logger.log_error(f"Emergency recovery failed: {e}")
            # Ensure scanning flag is cleared even on failure
            self._is_scanning_active = False
    def grab_fresh_scanning_frame(self, timeout_ms=1000, save_for_recording=True):
        """Grab a fresh frame from scanning pipeline with optional recording."""
        if self._pipeline_mode != "scanning":
            self.logger.log_error("Cannot grab scanning frame - not in scanning mode")
            return None, None, None, None
        
        try:
            # Get fresh frameset from scanning pipeline
            frameset = self.scanning_pipeline.wait_for_frames(timeout_ms)
            if frameset is None:
                self.logger.log_warning(f"No frame received within {timeout_ms}ms timeout")
                return None, None, None, None
            
            # Process the frameset
            rgb_image, depth_image, depth_colormap, point_cloud = self._process_scanning_frameset(frameset)
            
            # Add to scanning frames buffer if recording
            if save_for_recording and self._was_recording_before_scan and rgb_image is not None:
                frame_data = {
                    "timestamp": time.time(),
                    "session_id": self._scanning_session_id,
                    "rgb_shape": rgb_image.shape if rgb_image is not None else None,
                    "depth_shape": depth_image.shape if depth_image is not None else None,
                    "frame_index": len(self._scanning_frames_buffer)
                }
                self._scanning_frames_buffer.append(frame_data)
                self.logger.log_info(f"Added frame {frame_data['frame_index']} to scanning buffer")
            
            return rgb_image, depth_image, depth_colormap, point_cloud
            
        except Exception as e:
            self.logger.log_error(f"Error grabbing fresh scanning frame: {e}")
            return None, None, None, None
    
    def _process_scanning_frameset(self, frameset):
        """Process scanning frameset with enhanced depth filtering."""
        rgb_image = None
        depth_image = None
        depth_colormap = None
        point_cloud = None

        try:
            # Process RGB frame (primary for scanning)
            color_frame = frameset.get_color_frame()
            if color_frame:
                rgb_image = self.frame_to_bgr_image(color_frame)

            # Process depth frame with enhanced filtering
            depth_frame = frameset.get_depth_frame()
            if depth_frame:
                try:
                    filtered_depth_frame = self._apply_depth_filtering(depth_frame)
                    
                    width = filtered_depth_frame.get_width()
                    height = filtered_depth_frame.get_height()
                    scale = filtered_depth_frame.get_depth_scale()
                    
                    depth_data = np.frombuffer(filtered_depth_frame.get_data(), dtype=np.uint16)
                    depth_image_raw = depth_data.reshape((height, width))
                    
                    if self.depth_filtering_enabled:
                        depth_image_raw = self._apply_enhanced_depth_processing(depth_image_raw, rgb_image)
                    
                    depth_image_corrected = self._correct_uint16_overflow_artifacts(depth_image_raw)
                    
                    depth_image = depth_image_corrected.astype(np.float32) * scale
                    
                    MIN_DEPTH, MAX_DEPTH = 20, 3000
                    depth_image = np.where((depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH), depth_image, 0)
                        
                    # Create colorized depth map
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        
                    self.logger.log_info(f"Enhanced depth processing: filtering={self.depth_filtering_enabled}, scale={scale}")
                    
                except Exception as e:
                    self.logger.log_error(f"Enhanced depth processing failed: {e}")

        except Exception as e:
            self.logger.log_error(f"Error processing enhanced scanning frameset: {e}")

        return rgb_image, depth_image, depth_colormap, point_cloud
    
    def _apply_enhanced_depth_processing(self, depth_image_raw, rgb_image=None):
        """Apply enhanced OpenCV-based depth processing to reduce artifacts."""
        try:
            # Convert to float for processing
            depth_float = depth_image_raw.astype(np.float32)
            
            # Apply bilateral filter for edge-preserving smoothing
            if rgb_image is not None:
                # Use RGB as guide for joint bilateral filtering (if available)
                try:
                    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'jointBilateralFilter'):
                        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) if len(rgb_image.shape) == 3 else rgb_image
                        if gray.shape != depth_float.shape:
                            gray = cv2.resize(gray, (depth_float.shape[1], depth_float.shape[0]))
                        
                        depth_float = cv2.ximgproc.jointBilateralFilter(
                            joint=gray.astype(np.float32),
                            src=depth_float,
                            d=5, sigmaColor=50, sigmaSpace=50
                        )
                        self.logger.log_debug("Applied joint bilateral filter")
                    else:
                        # Fallback to standard bilateral filter
                        depth_float = cv2.bilateralFilter(depth_float, 5, 50, 50)
                        self.logger.log_debug("Applied bilateral filter")
                except Exception as e:
                    self.logger.log_debug(f"Bilateral filtering failed: {e}")
            
            # Apply median filter to reduce salt-and-pepper noise
            depth_float = cv2.medianBlur(depth_float.astype(np.uint16), 3).astype(np.float32)
            
            # Apply morphological operations to fill small holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = (depth_float > 0).astype(np.uint8)
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Fill small holes using inpainting
            holes_mask = (mask_closed - mask).astype(np.uint8)
            if np.any(holes_mask):
                depth_normalized = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_inpainted = cv2.inpaint(depth_normalized, holes_mask, 3, cv2.INPAINT_TELEA)
                depth_float = depth_inpainted.astype(np.float32) * (np.max(depth_float) / 255.0)
            
            return depth_float.astype(np.uint16)
            
        except Exception as e:
            self.logger.log_error(f"Enhanced depth processing failed: {e}")
            return depth_image_raw
    
    def _correct_uint16_overflow_artifacts(self, depth_raw):
        """
        Correct uint16 overflow artifacts with enhanced debugging
        """
        corrected = depth_raw.copy()
        rows_corrected = 0
        
        # Add comprehensive debugging
        total_pixels = depth_raw.size
        valid_pixels = np.sum(depth_raw > 0)
        
        self.logger.log_info(f"Depth data analysis: {valid_pixels}/{total_pixels} valid pixels")
        self.logger.log_info(f"Depth range: min={np.min(depth_raw[depth_raw > 0])}, max={np.max(depth_raw)}")
        
        # Analyze the full distribution
        if valid_pixels > 0:
            valid_data = depth_raw[depth_raw > 0]
            percentiles = np.percentile(valid_data, [1, 5, 10, 90, 95, 99])
            self.logger.log_info(f"Depth percentiles - 1%:{percentiles[0]}, 5%:{percentiles[1]}, "
                               f"10%:{percentiles[2]}, 90%:{percentiles[3]}, 95%:{percentiles[4]}, 99%:{percentiles[5]}")
        
        for row in range(depth_raw.shape[0]):
            row_data = depth_raw[row, :]
            valid_data = row_data[row_data > 0]
            
            if len(valid_data) < 10:
                continue
            
            # Dynamic threshold detection instead of fixed values
            row_min, row_max = np.min(valid_data), np.max(valid_data)
            row_range = row_max - row_min
            
            # Look for bimodal pattern: significant gap in distribution
            if row_range > 10000:  # Significant range suggests potential overflow
                # Count values in lower 20% and upper 20% of range
                lower_threshold = row_min + (row_range * 0.2)
                upper_threshold = row_max - (row_range * 0.2)
                
                very_low_count = np.sum(valid_data < lower_threshold)
                high_count = np.sum(valid_data > upper_threshold)
                middle_count = np.sum((valid_data >= lower_threshold) & (valid_data <= upper_threshold))
                
                # Check for bimodal distribution (low values + high values, few middle values)
                if (very_low_count > len(valid_data) * 0.1 and 
                    high_count > len(valid_data) * 0.1 and
                    middle_count < len(valid_data) * 0.3):
                    
                    self.logger.log_info(f"Row {row}: Potential overflow detected - "
                                       f"range={row_range}, low={very_low_count}, high={high_count}, middle={middle_count}")
                    
                    # Interpolate problematic row
                    if 0 < row < depth_raw.shape[0] - 1:
                        above_row = depth_raw[row-1, :].astype(np.uint32)
                        below_row = depth_raw[row+1, :].astype(np.uint32)
                        interpolated = ((above_row + below_row) / 2).astype(np.uint16)
                        corrected[row, :] = interpolated
                        rows_corrected += 1
                    else:
                        corrected[row, :] = 0
                        rows_corrected += 1
        
        if rows_corrected > 0:
            self.logger.log_info(f"Corrected uint16 overflow in {rows_corrected} rows")
        else:
            self.logger.log_info("No overflow patterns detected - thresholds may need adjustment")
        
        return corrected
    
    def _get_current_experiment_directory(self):
        """Get the current experiment directory if recording is active."""
        try:
            # Try to find the most recent experiment directory
            data_path = Path(self.data_path)
            if not data_path.exists():
                return None
            
            # Look for experiment directories (experiment_YYYYMMDD_HHMMSS format)
            experiment_dirs = [d for d in data_path.iterdir() 
                             if d.is_dir() and d.name.startswith("experiment_")]
            
            if not experiment_dirs:
                return None
            
            # Get the most recent experiment directory
            latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
            
            # Verify it has the expected structure
            if (latest_experiment / "rgb_images").exists() or (latest_experiment / "depth_images").exists():
                return latest_experiment
            
            return None
            
        except Exception as e:
            self.logger.log_info(f"Could not determine current experiment directory: {e}")
            return None
    
    def save_scanning_detection_result(self, rgb_image, color_mask, detection_result, position, object_type, color):
        """Save detection results during scanning for analysis - integrated with main recording."""
        if not self._was_recording_before_scan or self._scanning_session_id is None:
            return False
        
        try:
            # Check if we have an active recording experiment to integrate with
            experiment_dir = self._get_current_experiment_directory()
            
            if experiment_dir and experiment_dir.exists():
                # Save to main experiment directory (integrated with recording)
                self.logger.log_info(f"Saving detection results to experiment directory: {experiment_dir}")
                
                # Create detection subdirectory in experiment
                detection_dir = experiment_dir / "scanning_detections"
                detection_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time() * 1000)
                
                # Save detection image (with bounding boxes if available)
                detection_filename = f"detection_{self._scanning_session_id}_pos{position}_{object_type}_{color}_{timestamp}.jpg"
                detection_path = detection_dir / detection_filename
                
                # Draw detection results on image if found
                detection_image = rgb_image.copy()
                if detection_result and len(detection_result) >= 2 and detection_result[1] > 0:
                    if len(detection_result) >= 1 and detection_result[0] is not None:
                        centroid = detection_result[0]
                        if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                            cv2.circle(detection_image, (int(centroid[0]), int(centroid[1])), 10, (0, 255, 0), 3)
                            cv2.putText(detection_image, f"{object_type}:{color} ({detection_result[1]:.2f})", 
                                      (int(centroid[0])+15, int(centroid[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imwrite(str(detection_path), detection_image)
                
                # Save color mask
                mask_path = None
                if color_mask is not None:
                    mask_filename = f"color_mask_{self._scanning_session_id}_pos{position}_{color}_{timestamp}.jpg"
                    mask_path = detection_dir / mask_filename
                    cv2.imwrite(str(mask_path), color_mask)
                
                # Save detection metadata to experiment directory
                detection_metadata = {
                    "session_id": self._scanning_session_id,
                    "position": position,
                    "object_type": object_type,
                    "color": color,
                    "detection_result": detection_result,
                    "timestamp": time.time(),
                    "detection_image": str(detection_path),
                    "color_mask": str(mask_path) if mask_path else None,
                    "integrated_with_recording": True
                }
                
                # Append to experiment's detection results
                metadata_file = experiment_dir / "scanning_detection_results.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        all_detections = json.load(f)
                else:
                    all_detections = []
                
                all_detections.append(detection_metadata)
                
                with open(metadata_file, 'w') as f:
                    json.dump(all_detections, f, indent=2)
                
                self.logger.log_info(f"Detection result integrated with experiment at pos{position}")
                
            else:
                # Fallback to separate scanning directory if no active recording
                self.logger.log_info("No active recording experiment - saving detection to separate directory")
                session_dir = Path(self.data_path) / "scanning_sessions" / self._scanning_session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                
                # Save detection image (with bounding boxes if available)
                detection_filename = f"detection_pos{position}_{object_type}_{color}.jpg"
                detection_path = session_dir / detection_filename
                
                # Draw detection results on image if found
                detection_image = rgb_image.copy()
                if detection_result and len(detection_result) >= 2 and detection_result[1] > 0:
                    if len(detection_result) >= 1 and detection_result[0] is not None:
                        centroid = detection_result[0]
                        if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                            cv2.circle(detection_image, (int(centroid[0]), int(centroid[1])), 10, (0, 255, 0), 3)
                            cv2.putText(detection_image, f"{object_type}:{color} ({detection_result[1]:.2f})", 
                                      (int(centroid[0])+15, int(centroid[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imwrite(str(detection_path), detection_image)
                
                # Save color mask
                if color_mask is not None:
                    mask_filename = f"color_mask_pos{position}_{color}.jpg"
                    mask_path = session_dir / mask_filename
                    cv2.imwrite(str(mask_path), color_mask)
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Error saving detection result: {e}")
            return False
    
    def _save_hybrid_detection_summary(self, directory, filename, detection_result, color, object_type, timestamp):
        """Save detection summary text file.
        
        Args:
            directory: Directory path to save to
            filename: Filename for the summary
            detection_result: Detection result tuple or None
            color: Target color
            object_type: Target object type
            timestamp: Timestamp for the detection
        """
        try:
            with open(directory / filename, 'w') as f:
                f.write(f"Hybrid Detection Summary\n")
                f.write(f"========================\n")
                f.write(f"Target: {color} {object_type}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Detection successful: {detection_result is not None}\n")
                
                if detection_result:
                    centroid, confidence = detection_result
                    f.write(f"Position: {centroid}\n")
                    f.write(f"Confidence: {confidence:.3f}\n")
                else:
                    f.write(f"No valid detection found\n")
            
            self.logger.log_info(f"Saved hybrid detection summary: {filename}")
        except Exception as e:
            self.logger.log_error(f"Failed to save hybrid detection summary {filename}: {e}")
    
    
    
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
                
                # CRITICAL FIX: Get the depth scale factor
                scale = depth_frame.get_depth_scale()
                
                # Get raw uint16 depth data
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image_raw = depth_data.reshape((height, width))
                
                # Convert to actual depth values in millimeters and keep as float32
                depth_image = depth_image_raw.astype(np.float32) * scale
                
                # Apply depth range filtering (same as scanning pipeline)
                MIN_DEPTH = 20   # 20mm minimum
                MAX_DEPTH = 3000 # 3000mm maximum (3m for DaBai DCW2)
                depth_image = np.where((depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH), depth_image, 0)
                
                # Create colorized depth map using actual depth values (normalize from float32)
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
        """Enhanced cleanup with dual pipeline management."""
        try:
            # Save any pending scanning session data
            if self._scanning_frames_buffer and self._scanning_session_id:
                self._save_scanning_session_data()
            
            # Stop both pipelines
            self._frame_callback_active = False
            
            if self.recording_pipeline:
                try:
                    self.recording_pipeline.stop()
                    self.logger.log_info("Recording pipeline stopped")
                except:
                    pass
            
            if self.scanning_pipeline:
                try:
                    self.scanning_pipeline.stop()
                    self.logger.log_info("Scanning pipeline stopped")
                except:
                    pass
            
            # Stop legacy pipeline if it exists
            if self.orbbec_pipeline and self.orbbec_pipeline != self.recording_pipeline:
                try:
                    self.orbbec_pipeline.stop()
                except:
                    pass
            
            # Clear queue
            while not self._frameset_queue.empty():
                try:
                    self._frameset_queue.get_nowait()
                except Empty:
                    break
            
            # Clear scanning session data
            self._scanning_frames_buffer = []
            self._scanning_session_id = None
            self._was_recording_before_scan = False
            self._pipeline_mode = "recording"
            
            self.logger.log_info("Enhanced dual pipeline cleanup completed")
            
        except Exception as e:
            self.logger.log_error(f"Error during enhanced cleanup: {e}")
    
    def _save_scanning_session_data(self):
        """Save scanning session data integrated with main recording system."""
        if not self._scanning_frames_buffer or not self._scanning_session_id:
            return
        
        try:
            # Check if we have an active recording experiment to integrate with
            experiment_dir = self._get_current_experiment_directory()
            
            if experiment_dir and experiment_dir.exists():
                # Save to main experiment directory (integrated with recording)
                self.logger.log_info(f"Saving scanning session to experiment directory: {experiment_dir}")
                
                # Create session summary integrated with experiment
                session_summary = {
                    "session_id": self._scanning_session_id,
                    "timestamp": time.time(),
                    "frame_count": len(self._scanning_frames_buffer),
                    "frames": self._scanning_frames_buffer,
                    "experiment_directory": str(experiment_dir),
                    "integrated_with_recording": True,
                    "scanning_type": "action_level_pipeline_switching"
                }
                
                # Save session summary to experiment directory
                summary_file = experiment_dir / f"scanning_session_{self._scanning_session_id}.json"
                with open(summary_file, 'w') as f:
                    json.dump(session_summary, f, indent=2)
                
                # Also update the main experiment metadata to include scanning info
                self._update_experiment_metadata_with_scanning(experiment_dir, session_summary)
                
                self.logger.log_info(f"Scanning session {self._scanning_session_id} integrated with experiment recording ({len(self._scanning_frames_buffer)} frames)")
                
            else:
                # Fallback to separate scanning directory if no active recording
                self.logger.log_info("No active recording experiment - saving session to separate directory")
                session_dir = Path(self.data_path) / "scanning_sessions" / self._scanning_session_id
                
                # Create session summary
                session_summary = {
                    "session_id": self._scanning_session_id,
                    "timestamp": time.time(),
                    "frame_count": len(self._scanning_frames_buffer),
                    "frames": self._scanning_frames_buffer,
                    "session_directory": str(session_dir),
                    "integrated_with_recording": False,
                    "scanning_type": "action_level_pipeline_switching"
                }
                
                # Save session summary
                summary_file = session_dir / "session_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(session_summary, f, indent=2)
                
                self.logger.log_info(f"Scanning session {self._scanning_session_id} saved separately with {len(self._scanning_frames_buffer)} frames")
            
        except Exception as e:
            self.logger.log_error(f"Error saving scanning session data: {e}")
    
    def _update_experiment_metadata_with_scanning(self, experiment_dir, session_summary):
        """Update experiment metadata to include scanning session information."""
        try:
            metadata_file = experiment_dir / "experiment_metadata.json"
            
            # Load existing metadata or create new
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "experiment_id": experiment_dir.name,
                    "created_at": time.time(),
                    "scanning_sessions": []
                }
            
            # Add scanning session info
            if "scanning_sessions" not in metadata:
                metadata["scanning_sessions"] = []
            
            metadata["scanning_sessions"].append({
                "session_id": session_summary["session_id"],
                "timestamp": session_summary["timestamp"],
                "frame_count": session_summary["frame_count"],
                "scanning_type": session_summary["scanning_type"]
            })
            
            metadata["last_updated"] = time.time()
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.log_info(f"Updated experiment metadata with scanning session {session_summary['session_id']}")
            
        except Exception as e:
            self.logger.log_info(f"Could not update experiment metadata: {e}")
    
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
