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
        
        # Initialize camera if available
        if ORBBEC_SDK_AVAILABLE:
            self._init_dual_pipeline_system()
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
        
        self.logger.log_info("Scanning pipeline configured for RGB+Depth operation")
    
    def _setup_color_stream_for_pipeline(self, pipeline, config, pipeline_name):
        """Setup color stream for a specific pipeline."""
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if not color_profiles or color_profiles.get_count() == 0:
            self.logger.log_error(f"No color profiles available for {pipeline_name} pipeline")
            return
        
        self.logger.log_debug(f"Found {color_profiles.get_count()} color profiles for {pipeline_name}")
        
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
        
        self.logger.log_debug(f"Found {depth_profiles.get_count()} depth profiles for {pipeline_name}")
        
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
            self.logger.log_debug(f"Cleared {cleared_count} old frames from queue")
    
    def _wait_for_queue_refill(self, target_frames=10, timeout_seconds=5.0):
        """Wait for the frameset queue to refill after pipeline restart."""
        warmup_start = time.time()
        
        # Wait for queue to fill up and stabilize
        while self._frameset_queue.qsize() < target_frames and (time.time() - warmup_start) < timeout_seconds:
            time.sleep(0.1)
        
        time.sleep(0.5)  # Additional stability wait
        
        warmup_time = time.time() - warmup_start
        queue_size = self._frameset_queue.qsize()
        self.logger.log_debug(f"Queue refilled after {warmup_time:.1f}s with {queue_size} frames")
        return queue_size >= target_frames
    
    def switch_to_scanning_mode(self, action_context=None):
        """Switch from recording to scanning pipeline mode."""
        if self._pipeline_mode == "scanning":
            self.logger.log_debug("Already in scanning mode")
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
                self.logger.log_debug("Recording pipeline stopped for scanning")
            
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
            self.logger.log_debug("Already in recording mode")
            return True
        
        try:
            self.logger.log_info("Switching back to recording mode")
            
            # Stop scanning pipeline
            if self._pipeline_mode == "scanning":
                self.scanning_pipeline.stop()
                time.sleep(0.2)  # Allow pipeline to fully stop
                self.logger.log_debug("Scanning pipeline stopped")
            
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
                    self.logger.log_debug("Recording pipeline restarted with fresh frame queue")
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
            
            # Save frame data for recording if requested and we were recording before
            if save_for_recording and self._was_recording_before_scan and rgb_image is not None:
                frame_data = {
                    "timestamp": time.time(),
                    "session_id": self._scanning_session_id,
                    "rgb_shape": rgb_image.shape if rgb_image is not None else None,
                    "depth_shape": depth_image.shape if depth_image is not None else None,
                    "frame_index": len(self._scanning_frames_buffer)
                }
                
                # Save actual frame data to disk and store metadata
                if self._save_scanning_frame_to_disk(rgb_image, depth_image, depth_colormap, frame_data):
                    self._scanning_frames_buffer.append(frame_data)
                    self.logger.log_debug(f"Scanning frame {frame_data['frame_index']} saved for recording")
            
            return rgb_image, depth_image, depth_colormap, point_cloud
            
        except Exception as e:
            self.logger.log_error(f"Error grabbing fresh scanning frame: {e}")
            return None, None, None, None
    
    def _process_scanning_frameset(self, frameset):
        """Process scanning frameset (RGB-only optimized)."""
        rgb_image = None
        depth_image = None
        depth_colormap = None
        point_cloud = None
        
        try:
            # Process RGB frame (primary for scanning)
            color_frame = frameset.get_color_frame()
            if color_frame:
                rgb_image = self.frame_to_bgr_image(color_frame)
                if rgb_image is not None:
                    self.logger.log_debug(f"Processed scanning RGB frame: {rgb_image.shape}")
            
            # Process depth frame if available (optional for scanning)
            depth_frame = frameset.get_depth_frame()
            if depth_frame:
                try:
                    width = depth_frame.get_width()
                    height = depth_frame.get_height()
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_image = depth_data.reshape((height, width))
                    
                    # Create colorized depth map
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                except Exception as e:
                    self.logger.log_debug(f"Error processing scanning depth frame: {e}")
        
        except Exception as e:
            self.logger.log_error(f"Error processing scanning frameset: {e}")
        
        return rgb_image, depth_image, depth_colormap, point_cloud
    
    def _save_scanning_frame_to_disk(self, rgb_image, depth_image, depth_colormap, frame_data):
        """Save scanning frame data to main recording directory for integration."""
        try:
            session_id = frame_data["session_id"]
            frame_index = frame_data["frame_index"]
            timestamp = frame_data["timestamp"]
            
            # Check if we have an active recording experiment to integrate with
            experiment_dir = self._get_current_experiment_directory()
            
            if experiment_dir and experiment_dir.exists():
                # Save to main experiment directory (integrated with recording)
                self.logger.log_debug(f"Saving scanning frame to experiment directory: {experiment_dir}")
                
                # Save RGB image to experiment's rgb_images directory
                if rgb_image is not None:
                    rgb_dir = experiment_dir / "rgb_images"
                    rgb_dir.mkdir(exist_ok=True)
                    rgb_filename = f"scan_{session_id}_frame_{frame_index:04d}_{int(timestamp*1000)}.jpg"
                    rgb_path = rgb_dir / rgb_filename
                    cv2.imwrite(str(rgb_path), rgb_image)
                    frame_data["rgb_file"] = str(rgb_path)
                    frame_data["integrated_with_recording"] = True
                
                # Save depth image to experiment's depth_images directory
                if depth_image is not None:
                    depth_dir = experiment_dir / "depth_images"
                    depth_dir.mkdir(exist_ok=True)
                    depth_filename = f"scan_{session_id}_frame_{frame_index:04d}_{int(timestamp*1000)}.png"
                    depth_path = depth_dir / depth_filename
                    cv2.imwrite(str(depth_path), depth_image)
                    frame_data["depth_file"] = str(depth_path)
                
                # Save depth colormap for visualization
                if depth_colormap is not None:
                    colormap_filename = f"scan_{session_id}_frame_{frame_index:04d}_{int(timestamp*1000)}_colormap.jpg"
                    colormap_path = rgb_dir / colormap_filename
                    cv2.imwrite(str(colormap_path), depth_colormap)
                    frame_data["depth_colormap_file"] = str(colormap_path)
                    
            else:
                # Fallback to separate scanning directory if no active recording
                self.logger.log_debug("No active recording experiment - saving to separate scanning directory")
                session_dir = Path(self.data_path) / "scanning_sessions" / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                
                # Save RGB image
                if rgb_image is not None:
                    rgb_filename = f"frame_{frame_index:04d}_rgb.jpg"
                    rgb_path = session_dir / rgb_filename
                    cv2.imwrite(str(rgb_path), rgb_image)
                    frame_data["rgb_file"] = str(rgb_path)
                    frame_data["integrated_with_recording"] = False
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Error saving scanning frame to disk: {e}")
            return False
    
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
            self.logger.log_debug(f"Could not determine current experiment directory: {e}")
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
                self.logger.log_debug(f"Saving detection results to experiment directory: {experiment_dir}")
                
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
                
                self.logger.log_debug(f"Detection result integrated with experiment at pos{position}")
                
            else:
                # Fallback to separate scanning directory if no active recording
                self.logger.log_debug("No active recording experiment - saving detection to separate directory")
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
                    self.logger.log_debug("Recording pipeline stopped")
                except:
                    pass
            
            if self.scanning_pipeline:
                try:
                    self.scanning_pipeline.stop()
                    self.logger.log_debug("Scanning pipeline stopped")
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
                self.logger.log_debug(f"Saving scanning session to experiment directory: {experiment_dir}")
                
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
                self.logger.log_debug("No active recording experiment - saving session to separate directory")
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
                
            self.logger.log_debug(f"Updated experiment metadata with scanning session {session_summary['session_id']}")
            
        except Exception as e:
            self.logger.log_debug(f"Could not update experiment metadata: {e}")
    
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
