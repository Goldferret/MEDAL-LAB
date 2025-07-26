#!/usr/bin/env python3
"""
Object Detection Parameter Calibration Tool

Interactive tool for calibrating object detection parameters in the MEDAL-LAB system.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import sys
import logging

# Add the MEDAL-LAB nodes directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "nodes"))

# Create a simple logger class that mimics the MADSci EventClient
class SimpleLogger:
    """A simple logger class that mimics the MADSci EventClient for standalone tools."""
    
    def __init__(self, name="CalibrationTool", level=logging.INFO):
        """Initialize the logger with the given name and level."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    # MADSci EventClient compatible methods only
    def log_debug(self, message, **kwargs):
        """Log a debug message (MADSci EventClient compatibility)."""
        self.logger.debug(message)
    
    def log_info(self, message, **kwargs):
        """Log an info message (MADSci EventClient compatibility)."""
        self.logger.info(message)
    
    def log_warning(self, message, **kwargs):
        """Log a warning message (MADSci EventClient compatibility)."""
        self.logger.warning(message)
    
    def log_error(self, message, **kwargs):
        """Log an error message (MADSci EventClient compatibility)."""
        self.logger.error(message)
    
    def log_critical(self, message, **kwargs):
        """Log a critical message (MADSci EventClient compatibility)."""
        self.logger.critical(message)

# Import robot arm config for parameter format reference
try:
    from robot_arm_config import RobotArmConfig
except ImportError:
    # If we can't import it, we'll just skip loading default config
    RobotArmConfig = None

class ObjectDetectionCalibrationTool:
    def __init__(self, camera_id=0):
        # Create a logger
        self.logger = SimpleLogger("ObjectDetectionCalibrationTool")
        self.logger.log_info("Initializing Object Detection Calibration Tool")
        
        # Initialize OpenCV camera
        self.logger.log_info(f"Connecting to camera with device ID: {camera_id}")
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            self.logger.log_error(f"Failed to open camera with device ID: {camera_id}")
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Load default configuration if available
        if RobotArmConfig:
            self.logger.log_info("Loading default configuration")
            self.config = RobotArmConfig()
        else:
            self.logger.log_warning("Could not load RobotArmConfig, using built-in defaults")
            self.config = None
        
        # Create window and trackbars with resizable window
        cv2.namedWindow("Object Detection Calibration", cv2.WINDOW_NORMAL)
        # Set initial window size to be more manageable (800x600)
        cv2.resizeWindow("Object Detection Calibration", 800, 600)
        
        # Edge detection parameters
        self.logger.log_debug("Creating trackbars for parameter adjustment")
        cv2.createTrackbar("Gaussian Kernel", "Object Detection Calibration", 5, 15, self.update_params)
        cv2.createTrackbar("Canny Low", "Object Detection Calibration", 20, 255, self.update_params)
        cv2.createTrackbar("Canny High", "Object Detection Calibration", 60, 255, self.update_params)
        
        # Shape detection parameters
        cv2.createTrackbar("Polygon Epsilon x100", "Object Detection Calibration", 1, 10, self.update_params)
        cv2.createTrackbar("Min Area", "Object Detection Calibration", 400, 5000, self.update_params)
        cv2.createTrackbar("Rectangle Extent x100", "Object Detection Calibration", 50, 100, self.update_params)
        
        # Aspect ratio parameters
        cv2.createTrackbar("Aspect Ratio Min x10", "Object Detection Calibration", 7, 15, self.update_params)
        cv2.createTrackbar("Aspect Ratio Max x10", "Object Detection Calibration", 13, 20, self.update_params)
        
        # Hybrid approach parameters
        cv2.createTrackbar("Color-Edge Weight x10", "Object Detection Calibration", 5, 10, self.update_params)
        cv2.createTrackbar("Morph Kernel Size", "Object Detection Calibration", 3, 15, self.update_params)
        
        # Visualization mode
        cv2.createTrackbar("Visualization Mode", "Object Detection Calibration", 0, 5, lambda x: None)
        
        # Color selection
        self.colors = ["red", "green", "blue", "yellow"]
        self.current_color_idx = 2  # Default to blue
        cv2.createTrackbar("Color", "Object Detection Calibration", self.current_color_idx, 3, self.update_color)
        
        # Object type selection
        self.object_types = ["cube", "rectangular_prism"]
        self.current_object_idx = 0  # Default to cube
        cv2.createTrackbar("Object Type", "Object Detection Calibration", self.current_object_idx, 1, self.update_object)
        
        # Initialize parameters
        self.update_params(0)
        
    def update_color(self, value):
        self.current_color_idx = value
        
    def update_object(self, value):
        self.current_object_idx = value
        
    def update_params(self, _):
        # Get values from trackbars
        kernel_size = cv2.getTrackbarPos("Gaussian Kernel", "Object Detection Calibration")
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd number
            
        canny_low = cv2.getTrackbarPos("Canny Low", "Object Detection Calibration")
        canny_high = cv2.getTrackbarPos("Canny High", "Object Detection Calibration")
        
        polygon_epsilon = cv2.getTrackbarPos("Polygon Epsilon x100", "Object Detection Calibration") / 100.0
        min_area = cv2.getTrackbarPos("Min Area", "Object Detection Calibration")
        rectangle_extent = cv2.getTrackbarPos("Rectangle Extent x100", "Object Detection Calibration") / 100.0
        
        aspect_ratio_min = cv2.getTrackbarPos("Aspect Ratio Min x10", "Object Detection Calibration") / 10.0
        aspect_ratio_max = cv2.getTrackbarPos("Aspect Ratio Max x10", "Object Detection Calibration") / 10.0
        
        color_edge_weight = cv2.getTrackbarPos("Color-Edge Weight x10", "Object Detection Calibration") / 10.0
        morph_kernel_size = cv2.getTrackbarPos("Morph Kernel Size", "Object Detection Calibration")
        if morph_kernel_size % 2 == 0:
            morph_kernel_size += 1  # Ensure odd number
        
        # Update parameters
        self.params = {
            "gaussian_kernel": (kernel_size, kernel_size),
            "canny_thresholds": (canny_low, canny_high),
            "polygon_epsilon": polygon_epsilon,
            "min_area": min_area,
            "rectangle_extent": rectangle_extent,
            "aspect_ratio_square": (aspect_ratio_min, aspect_ratio_max),
            "color_edge_weight": color_edge_weight,
            "morph_kernel_size": morph_kernel_size
        }
    
    def process_frame(self, frame):
        # Get current parameters
        kernel_size = self.params["gaussian_kernel"]
        canny_low, canny_high = self.params["canny_thresholds"]
        polygon_epsilon = self.params["polygon_epsilon"]
        min_area = self.params["min_area"]
        rectangle_extent = self.params["rectangle_extent"]
        aspect_ratio_min, aspect_ratio_max = self.params["aspect_ratio_square"]
        color_edge_weight = self.params["color_edge_weight"]
        morph_kernel_size = self.params["morph_kernel_size"]
        
        # Get current color and object type
        color = self.colors[self.current_color_idx]
        object_type = self.object_types[self.current_object_idx]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = self._create_color_mask(hsv, color)
        
        # Hybrid approach
        hybrid_mask = cv2.addWeighted(
            edges, 1 - color_edge_weight,
            color_mask, color_edge_weight,
            0
        )
        
        # Apply morphological operations
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        morph_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate extent
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Validate shape
            if (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max and 
                extent >= rectangle_extent):
                valid_contours.append((contour, (x, y, w, h), area))
        
        # Create visualization based on selected mode
        viz_mode = cv2.getTrackbarPos("Visualization Mode", "Object Detection Calibration")
        
        if viz_mode == 0:  # Original
            result = frame.copy()
        elif viz_mode == 1:  # Edges
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif viz_mode == 2:  # Color Mask
            result = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        elif viz_mode == 3:  # Hybrid Mask
            result = cv2.cvtColor(morph_mask, cv2.COLOR_GRAY2BGR)
        elif viz_mode == 4:  # Contours
            result = frame.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        else:  # Final
            result = frame.copy()
            
        # Draw valid contours and bounding boxes
        if viz_mode == 5:  # Final visualization
            for contour, (x, y, w, h), area in valid_contours:
                # Draw contour
                cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Draw centroid
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)
                
                # Add text
                cv2.putText(result, f"{area:.0f}px", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add parameter info
        cv2.putText(result, f"Color: {color}, Object: {object_type}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(result, f"Valid Objects: {len(valid_contours)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize the result for display to fit better on screen
        # Scale down to max width of 800 pixels while maintaining aspect ratio
        height, width = result.shape[:2]
        max_width = 800
        if width > max_width:
            scale_factor = max_width / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return result
    
    def _create_color_mask(self, hsv_image, color):
        """Create color mask for the specified color."""
        if color == "red":
            # Red wraps around the hue spectrum
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([160, 100, 100])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            return cv2.bitwise_or(mask1, mask2)
        elif color == "green":
            lower = np.array([40, 40, 40])
            upper = np.array([80, 255, 255])
        elif color == "blue":
            lower = np.array([100, 50, 50])
            upper = np.array([130, 255, 255])
        elif color == "yellow":
            lower = np.array([20, 100, 100])
            upper = np.array([40, 255, 255])
        else:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            
        return cv2.inRange(hsv_image, lower, upper)
    
    def save_parameters(self):
        """Print current parameters to the console in a format suitable for robot_arm_config.py."""
        # Create configuration dictionary for display
        config = {
            "vision_thresholds": {
                "min_area": self.params["min_area"],
                "canny_thresholds": list(self.params["canny_thresholds"]),
                "polygon_epsilon": self.params["polygon_epsilon"],
                "rectangle_extent": self.params["rectangle_extent"],
                "aspect_ratio_square": list(self.params["aspect_ratio_square"]),
                "gaussian_kernel": list(self.params["gaussian_kernel"]),
                "morph_kernel_size": self.params["morph_kernel_size"],
                "color_edge_weight": self.params["color_edge_weight"]
            }
        }
        
        # Print Python code format for robot_arm_config.py
        self.logger.log_info("\nCopy this code into your robot_arm_config.py:")
        self.logger.log_info("    # Vision detection thresholds (centralized from vision_detector)")
        self.logger.log_info("    vision_thresholds: Dict[str, Any] = {")
        self.logger.log_info(f"        \"min_area\": {self.params['min_area']},                    # Minimum contour area")
        self.logger.log_info(f"        \"min_radius\": 15,                   # Minimum circle radius")
        self.logger.log_info(f"        \"confidence_weights\": (0.6, 0.4),   # Shape vs color confidence weighting")
        self.logger.log_info(f"        \"canny_thresholds\": {tuple(self.params['canny_thresholds'])},      # Edge detection thresholds")
        self.logger.log_info(f"        \"polygon_epsilon\": {self.params['polygon_epsilon']:.3f},            # Polygon approximation factor")
        self.logger.log_info(f"        \"rectangle_extent\": {self.params['rectangle_extent']:.2f},            # Minimum rectangle fill ratio")
        self.logger.log_info(f"        \"aspect_ratio_square\": {tuple(self.params['aspect_ratio_square'])},  # Square aspect ratio range")
        self.logger.log_info(f"        \"morphology_iterations\": 2,         # Erosion/dilation iterations")
        self.logger.log_info(f"        \"gaussian_kernel\": {tuple(self.params['gaussian_kernel'])},          # Gaussian blur kernel size")
        self.logger.log_info(f"        \"edge_margin\": 50,                  # Minimum distance from image edges")
        self.logger.log_info("    }")
        
        # Print additional parameters for vision_detector.py
        self.logger.log_info("\nAdditional parameters for vision_detector.py:")
        self.logger.log_info(f"morph_kernel_size = {self.params['morph_kernel_size']}")
        self.logger.log_info(f"color_edge_weight = {self.params['color_edge_weight']}")
        
        self.logger.log_info("\nParameters printed to console. Copy and paste them into your configuration files.")
        return config
    
    def run(self):
        self.logger.log_info("Starting Object Detection Parameter Calibration Tool")
        self.logger.log_info("------------------------------------------")
        self.logger.log_info("Controls:")
        self.logger.log_info("  ESC - Exit")
        self.logger.log_info("  S - Save parameters")
        self.logger.log_info("  R - Reset parameters")
        self.logger.log_info("  1-6 - Change visualization mode")
        self.logger.log_info("\nVisualization Modes:")
        self.logger.log_info("  1 - Original")
        self.logger.log_info("  2 - Edges")
        self.logger.log_info("  3 - Color Mask")
        self.logger.log_info("  4 - Hybrid Mask")
        self.logger.log_info("  5 - Contours")
        self.logger.log_info("  6 - Final Detection")
        
        frame_count = 0
        while True:
            # Capture frame from OpenCV camera
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.logger.log_error("Failed to capture frame from camera")
                break
            
            frame_count += 1
            if frame_count % 100 == 0:  # Log every 100 frames to show it's working
                self.logger.log_debug(f"Processed {frame_count} frames")
            
            # Process frame
            result = self.process_frame(frame)
            
            # Show result
            cv2.imshow("Object Detection Calibration", result)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # Save parameters
                config = self.save_parameters()
                self.logger.log_info("Parameters printed above. Copy and paste them into your configuration files.")
            elif key == ord('r'):  # Reset parameters
                self.update_params(0)
                self.logger.log_info("Parameters reset to default values.")
            elif key >= ord('1') and key <= ord('6'):  # Change visualization mode
                cv2.setTrackbarPos("Visualization Mode", "Object Detection Calibration", key - ord('1'))
                mode_names = ["Original", "Edges", "Color Mask", "Hybrid Mask", "Contours", "Final Detection"]
                self.logger.log_info(f"Visualization mode changed to: {mode_names[key - ord('1')]}")
        
        # Clean up
        cv2.destroyAllWindows()
        self.camera.release()
        self.logger.log_info("Object Detection Calibration Tool closed.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Object Detection Parameter Calibration Tool")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--camera-id", type=int, default=0, 
                        help="Camera device ID (default: 0)")
    args = parser.parse_args()
    
    # Set up logging level
    log_level = getattr(logging, args.log_level)
    
    try:
        # Create and run the tool
        tool = ObjectDetectionCalibrationTool(camera_id=args.camera_id)
        tool.logger.logger.setLevel(log_level)
        tool.run()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Try using a different camera ID with --camera-id parameter")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCalibration tool interrupted by user")
        sys.exit(0)
