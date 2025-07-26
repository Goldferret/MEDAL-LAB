"""
Experiment Logger Component for MEDAL-LAB

Handles all experiment data logging and persistence including:
- Scanning event logging
- Debug image saving
- Experiment metadata management
- Summary file generation
- JSON data persistence
- 10Hz recording coordination
- Trajectory data management
"""

import cv2
import numpy as np
import time
import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import wraps


def handle_logging_errors(operation_name: str):
    """Decorator for consistent logging error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.log_error(f"Error in {operation_name}: {e}")
                return False
        return wrapper
    return decorator


class ExperimentLogger:
    """Centralized component for all experiment data logging and summaries."""
    
    def __init__(self, logger, data_path: str = "./captures"):
        """Initialize experiment logger.
        
        Args:
            logger: Logger instance for logging
            data_path: Base path for data storage
        """
        self.logger = logger
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # Current experiment tracking
        self.current_experiment_dir = None
        self.current_session_id = None
        
        # Recording state tracking
        self.recording = False
        self.current_trajectory_data = None
    
    @handle_logging_errors("scanning event logging")
    def log_scanning_event(self, original_frame: np.ndarray, depth_frame: np.ndarray, depth_colormap: np.ndarray, scan_data: Dict[str, Any]) -> bool:
        """Log complete scanning event with all associated data.
        
        Args:
            original_frame: Original RGB frame from camera
            depth_frame: Original Depth frame from camera
            depth_colormap: Original Colormap from Depth frame
            scan_data: Dictionary containing all scan information
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self._ensure_experiment_directory():
            return False
        
        try:
            position = scan_data["position"]
            detection_result = scan_data["detection_result"]
            debug_images = scan_data["debug_images"]
            frame_metadata = scan_data["frame_metadata"]
            
            timestamp = frame_metadata["timestamp"]
            object_type = frame_metadata["object_type"]
            color = frame_metadata["color"]
            
            # Create scanning event subdirectory
            scanning_dir = self.current_experiment_dir / "scanning_events"
            scanning_dir.mkdir(exist_ok=True)
            
            # Generate unique identifier for this scanning event
            event_id = f"pos{position}_{color}_{object_type}_{int(timestamp * 1000)}"
            
            # Save debug images
            debug_paths = self._save_scanning_debug_images(
                scanning_dir, event_id, original_frame, depth_frame, depth_colormap, debug_images
            )
            
            # Create event record
            event_record = {
                "event_id": event_id,
                "timestamp": timestamp,
                "position": position,
                "object_type": object_type,
                "color": color,
                "detection_successful": detection_result is not None,
                "detection_result": {
                    "centroid": detection_result[0] if detection_result else None,
                    "confidence": detection_result[1] if detection_result else 0.0
                },
                "debug_image_paths": debug_paths,
                "frame_metadata": frame_metadata
            }
            
            # Append to experiment scanning log
            self._append_to_scanning_log(event_record)
            
            self.logger.log_debug(f"Scanning event logged: {event_id}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Error logging scanning event: {e}")
            return False
    
    def _save_scanning_debug_images(self, scanning_dir: Path, event_id: str, 
                                   original_frame: np.ndarray, depth_frame: np.ndarray, depth_colormap: np.ndarray, debug_images: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Save all debug images for a scanning event.
        
        Args:
            scanning_dir: Directory to save images in
            event_id: Unique identifier for this event
            original_frame: Original RGB frame
            depth_frame: Original Depth frame
            depth_colormap: Original Depth Colormap
            debug_images: Dictionary of debug images
            
        Returns:
            Dictionary mapping image types to file paths
        """
        debug_paths = {}
        
        try:
            # Save original frame
            original_path = scanning_dir / f"{event_id}_01_original.jpg"
            if self.save_frame(original_frame, str(original_path), "original"):
                debug_paths["original"] = str(original_path)
                
            # Save depth frame
            depth_path = scanning_dir / f"{event_id}_02_depth.png"
            if self.save_frame(depth_frame, str(depth_path), "depth"):
                debug_paths["depth"] = str(depth_path)
                
            # Save depth colormap
            colormap_path = scanning_dir / f"{event_id}_03_colormap.jpg"
            if self.save_frame(depth_colormap, str(colormap_path), "colormap"):
                debug_paths["colormap"] = str(colormap_path)
            
            # Save debug images
            image_mapping = {
                "initial_mask": "04_mask_initial",
                "final_mask": "07_mask_final", 
                "detection_visualization": "09_detections"
            }
            
            for debug_key, filename_part in image_mapping.items():
                if debug_key in debug_images:
                    image_path = scanning_dir / f"{event_id}_{filename_part}.jpg"
                    
                    # Convert grayscale masks to BGR for saving
                    image_to_save = debug_images[debug_key]
                    if len(image_to_save.shape) == 2:  # Grayscale mask
                        image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_GRAY2BGR)
                    
                    if self.save_frame(image_to_save, str(image_path), debug_key):
                        debug_paths[debug_key] = str(image_path)
            
            # Save summary
            summary_path = scanning_dir / f"{event_id}_10_summary.txt"
            self._save_scanning_event_summary(summary_path, event_id, debug_images, debug_paths)
            debug_paths["summary"] = str(summary_path)
            
        except Exception as e:
            self.logger.log_error(f"Error saving debug images for {event_id}: {e}")
        
        return debug_paths
    
    @handle_logging_errors("JSON data saving")
    def save_json_data(self, data: Dict[str, Any], file_path: str, data_type: str = "data") -> bool:
        """Save JSON data for analysis.
        
        Args:
            data: Dictionary data to save
            file_path: Full path where to save the JSON file
            data_type: Type of data for logging purposes (e.g., "calibration", "metadata", "trajectory")
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.log_debug(f"Saved {data_type} JSON: {Path(file_path).name}")
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to save {data_type} JSON {file_path}: {e}")
            return False
    
    @handle_logging_errors("frame saving")
    def save_frame(self, image: np.ndarray, file_path: str, image_type: str = "debug") -> bool:
        """Save frame for analysis.
        
        Args:
            image: Image array to save
            file_path: Full path where to save the image
            image_type: Type of image for logging purposes (e.g., "mask", "detection", "original", "recording", "depth")
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Special handling for depth images
            if image_type == "depth" and image is not None:
                # Convert to millimeters and save as 16-bit PNG
                depth_mm = (image * 1000).astype(np.uint16)
                cv2.imwrite(file_path, depth_mm)
            else:
                cv2.imwrite(file_path, image)
                
            self.logger.log_debug(f"Saved {image_type} image: {Path(file_path).name}")
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to save {image_type} image {file_path}: {e}")
            return False
    
    def _save_scanning_event_summary(self, summary_path: Path, event_id: str, 
                                   debug_images: Dict[str, np.ndarray], debug_paths: Dict[str, str]):
        """Save scanning event summary text file.
        
        Args:
            summary_path: Path to save summary file
            event_id: Event identifier
            debug_images: Debug images dictionary
            debug_paths: Paths where images were saved
        """
        try:
            with open(summary_path, 'w') as f:
                f.write(f"Scanning Event Summary\n")
                f.write(f"=====================\n")
                f.write(f"Event ID: {event_id}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
                
                f.write(f"Debug Images Generated:\n")
                for debug_key in debug_images.keys():
                    f.write(f"  - {debug_key}: {debug_paths.get(debug_key, 'Not saved')}\n")
                
                f.write(f"\nTotal debug files: {len(debug_paths)}\n")
            
            self.logger.log_debug(f"Saved scanning event summary: {summary_path.name}")
        except Exception as e:
            self.logger.log_error(f"Failed to save scanning event summary {summary_path}: {e}")
    
    def _append_to_scanning_log(self, event_record: Dict[str, Any]):
        """Append scanning event to experiment log.
        
        Args:
            event_record: Event data to append
        """
        try:
            log_file = self.current_experiment_dir / "scanning_events.json"
            
            # Load existing log or create new
            if log_file.exists():
                with open(log_file, 'r') as f:
                    scanning_log = json.load(f)
            else:
                scanning_log = []
            
            # Append new event
            scanning_log.append(event_record)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(scanning_log, f, indent=2)
            
            self.logger.log_debug(f"Appended event to scanning log: {event_record['event_id']}")
            
        except Exception as e:
            self.logger.log_error(f"Error appending to scanning log: {e}")
    
    def _ensure_experiment_directory(self) -> bool:
        """Ensure we have a valid experiment directory to log to.
        
        Returns:
            True if directory is available, False otherwise
        """
        # For now, create a simple experiment directory
        # This will be enhanced when integrated with the recording system
        if self.current_experiment_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_experiment_dir = self.data_path / f"experiment_{timestamp}"
            self.current_experiment_dir.mkdir(exist_ok=True)
            self.logger.log_info(f"Created experiment directory: {self.current_experiment_dir}")
        
        return self.current_experiment_dir.exists()
    
    def set_experiment_directory(self, experiment_dir: Path):
        """Set the current experiment directory for logging.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.current_experiment_dir = experiment_dir
        self.logger.log_info(f"Experiment directory set to: {experiment_dir}")
    
    def get_experiment_directory(self) -> Optional[Path]:
        """Get the current experiment directory.
        
        Returns:
            Current experiment directory path or None
        """
        return self.current_experiment_dir
    
    # ========================================
    # RECORDING FUNCTIONALITY
    # ========================================
    
    def start_recording(self, experiment_id: str = None) -> str:
        """Start coordinated recording session.
        
        Args:
            experiment_id: Optional experiment identifier
            
        Returns:
            Experiment ID for the recording session
        """
        if self.recording:
            self.logger.log_warning("Recording already active. Stopping previous recording.")
            self.stop_recording()
        
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"experiment_{timestamp}"
        
        # Create experiment directory
        experiment_dir = self.data_path / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized data storage
        (experiment_dir / "rgb_images").mkdir(exist_ok=True)
        (experiment_dir / "depth_images").mkdir(exist_ok=True)
        (experiment_dir / "point_clouds").mkdir(exist_ok=True)
        
        # Initialize trajectory data structure
        self.current_trajectory_data = {
            "trajectory_id": experiment_id,
            "start_time": time.time(),
            "joint_states": [],
            "images": [],
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "data_format_version": "1.0"
            }
        }
        
        self.current_experiment_dir = experiment_dir
        self.recording = True
        self.logger.log_info(f"Started recording experiment {experiment_id}")
        return experiment_id
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Stop coordinated recording and save trajectory data.
        
        Returns:
            Dictionary with trajectory data or None if no recording was active
        """
        if not self.recording:
            self.logger.log_warning("No active recording to stop.")
            return None
        
        experiment_id = self.current_trajectory_data["trajectory_id"]
        experiment_dir = self.current_experiment_dir
        
        # Finalize trajectory data
        self.current_trajectory_data["end_time"] = time.time()
        self.current_trajectory_data["duration"] = (
            self.current_trajectory_data["end_time"] - self.current_trajectory_data["start_time"]
        )
        
        # Save trajectory data to JSON
        trajectory_file = experiment_dir / "trajectory_data.json"
        with open(trajectory_file, "w") as f:
            json.dump(self.current_trajectory_data, f, indent=2)
        
        self.recording = False
        self.logger.log_info(f"Stopped and saved experiment {experiment_id}")
        
        # Return copy of trajectory data for the caller
        trajectory_copy = self.current_trajectory_data.copy()
        self.current_trajectory_data = None
        return trajectory_copy
    
    def record_data_point(self, action_name: str = "", action_params: Dict = None, 
                         joint_angles: List[int] = None, gripper_closed: bool = False, 
                         is_moving: bool = False, camera_manager=None):
        """Record coordinated data point with movement and camera data.
        
        NOTE: Current implementation uses synchronous cv2.imwrite() which may block 
        10Hz recording pipeline. Future enhancement should implement asynchronous saving.
        
        Args:
            action_name: Name of the action being performed
            action_params: Parameters of the action being performed
            joint_angles: Current joint angles from movement controller
            gripper_closed: Gripper state from movement controller
            is_moving: Movement state from movement controller
            camera_manager: Camera manager instance for capturing images
        """
        if not self.recording:
            return
        
        try:
            timestamp = time.time()
            frame_id = f"frame_{len(self.current_trajectory_data['joint_states']):06d}"
            
            # Get movement data
            joint_state = {
                "timestamp": timestamp,
                "joint_angles": joint_angles.copy() if joint_angles else [],
                "gripper_closed": gripper_closed,
                "is_moving": is_moving,
                "action_name": action_name,
                "action_params": action_params or {}
            }
            
            # Capture images from camera manager (skip during scanning)
            image_data = {"frame_id": frame_id, "timestamp": timestamp}
            
            # Check if scanning is active - if so, skip image capture but note it
            if camera_manager and camera_manager.is_scanning_active():
                image_data["scanning_active"] = True
                image_data["note"] = "Image capture skipped - scanning in progress"
                self.logger.log_debug(f"Skipping image capture for {frame_id} - scanning active")
            elif camera_manager:
                try:
                    # Get synchronized camera data
                    rgb_image, depth_image, depth_colormap, point_cloud = camera_manager.capture_synchronized_data()
                    
                    # Save RGB image (synchronous - may block 10Hz pipeline)
                    if rgb_image is not None:
                        rgb_path = self.current_experiment_dir / "rgb_images" / f"{frame_id}.jpg"
                        cv2.imwrite(str(rgb_path), rgb_image)
                        image_data["rgb_path"] = f"rgb_images/{frame_id}.jpg"
                    
                    # Save depth data (synchronous - may block 10Hz pipeline)
                    if depth_image is not None:
                        depth_path = self.current_experiment_dir / "depth_images" / f"{frame_id}_depth.png"
                        # Save depth image with 16-bit precision to preserve depth values
                        cv2.imwrite(str(depth_path), (depth_image * 1000).astype(np.uint16))
                        image_data["depth_path"] = f"depth_images/{frame_id}_depth.png"
                        
                        # Save colorized depth map
                        if depth_colormap is not None:
                            colormap_path = self.current_experiment_dir / "depth_images" / f"{frame_id}_depth_colormap.jpg"
                            cv2.imwrite(str(colormap_path), depth_colormap)
                            image_data["depth_colormap_path"] = f"depth_images/{frame_id}_depth_colormap.jpg"
                    
                    # Save point cloud (synchronous - may block 10Hz pipeline)
                    if point_cloud is not None:
                        pc_path = self.current_experiment_dir / "point_clouds" / f"{frame_id}_pointcloud.npy"
                        np.save(str(pc_path), point_cloud)
                        image_data["point_cloud_path"] = f"point_clouds/{frame_id}_pointcloud.npy"
                        image_data["point_cloud_size"] = len(point_cloud)
                        
                except Exception as e:
                    self.logger.log_error(f"Error capturing images during recording: {e}")
            
            # Store coordinated data (joint data always recorded, images only when not scanning)
            self.current_trajectory_data["joint_states"].append(joint_state)
            self.current_trajectory_data["images"].append(image_data)
            
        except Exception as e:
            self.logger.log_error(f"Error recording data point: {e}")
    
    def is_recording(self) -> bool:
        """Check if recording is currently active.
        
        Returns:
            True if recording is active, False otherwise
        """
        return self.recording
    
    def get_current_trajectory_data(self) -> Optional[Dict[str, Any]]:
        """Get current trajectory data (copy).
        
        Returns:
            Copy of current trajectory data or None if not recording
        """
        if self.current_trajectory_data:
            return self.current_trajectory_data.copy()
        return None
