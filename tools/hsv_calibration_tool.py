#!/usr/bin/env python3
"""
HSV Color Calibration Tool for DOFBOT Pro

This tool helps calibrate HSV color ranges for the standard DOFBOT Pro colored blocks
(red, green, blue, yellow cubes and rectangular prisms) under your specific lighting conditions.

Usage:
    python3 hsv_calibration_tool.py

Features:
- Click on objects to sample HSV values
- Real-time HSV range adjustment with trackbars
- Export calibrated ranges for use in robot actions
- Test detection with live preview
- Save/load calibration profiles

Requirements:
- OpenCV with camera access
- pyorbbecsdk (if using Orbbec camera)
- Standard DOFBOT Pro colored blocks in view
"""

import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

# Try to import Orbbec SDK for camera access
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    ORBBEC_AVAILABLE = True
    print("✓ Orbbec SDK available - will use depth camera")
except ImportError:
    ORBBEC_AVAILABLE = False
    print("⚠ Orbbec SDK not available - will use default camera")

class HSVCalibrationTool:
    def __init__(self, use_orbbec=True, camera_index=0):
        """Initialize the HSV calibration tool."""
        self.use_orbbec = use_orbbec and ORBBEC_AVAILABLE
        self.camera_index = camera_index
        
        # Current HSV ranges being calibrated
        self.hsv_ranges = {
            "red": {"lower": [0, 50, 50], "upper": [10, 255, 255]},
            "green": {"lower": [40, 40, 40], "upper": [80, 255, 255]},
            "blue": {"lower": [100, 50, 50], "upper": [130, 255, 255]},
            "yellow": {"lower": [20, 100, 100], "upper": [30, 255, 255]}
        }
        
        # Red needs special handling (wraps around HSV hue)
        self.red_ranges = [
            {"lower": [0, 50, 50], "upper": [10, 255, 255]},      # Lower red
            {"lower": [170, 50, 50], "upper": [180, 255, 255]}   # Upper red
        ]
        
        # Current color being calibrated
        self.current_color = "red"
        self.current_red_range = 0  # For red's dual ranges
        
        # Sampling data
        self.sample_points = []
        self.sample_hsv_values = []
        
        # Camera setup
        self.camera = None
        self.orbbec_pipeline = None
        self.setup_camera()
        
        # Create windows and trackbars
        self.setup_ui()
        
        print("\n🎯 HSV Calibration Tool Ready!")
        print("📋 Instructions:")
        print("1. Select color to calibrate using keys: R(red), G(green), B(blue), Y(yellow)")
        print("2. Click on objects of that color to sample HSV values")
        print("3. Adjust HSV ranges using trackbars")
        print("4. Press 'T' to test current ranges")
        print("5. Press 'S' to save calibration")
        print("6. Press 'L' to load previous calibration")
        print("7. Press 'Q' to quit")
        print(f"\n🔴 Currently calibrating: {self.current_color.upper()}")
    
    def setup_camera(self):
        """Setup camera (Orbbec or standard webcam)."""
        if self.use_orbbec:
            try:
                self.orbbec_pipeline = Pipeline()
                config = Config()
                
                # Configure color stream
                profile_list = self.orbbec_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                if profile_list and profile_list.get_count() > 0:
                    color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 15)
                    if color_profile:
                        config.enable_stream(color_profile)
                        print("✓ Orbbec color stream configured: 640x480@15fps")
                
                self.orbbec_pipeline.start(config)
                print("✓ Orbbec camera initialized")
                return
            except Exception as e:
                print(f"⚠ Orbbec camera failed: {e}")
                print("Falling back to standard camera...")
        
        # Fallback to standard camera
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"✓ Standard camera initialized (index {self.camera_index})")
    
    def setup_ui(self):
        """Create OpenCV windows and trackbars."""
        # Main windows
        cv2.namedWindow('HSV Calibration Tool', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('HSV Controls', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Mask Preview', cv2.WINDOW_AUTOSIZE)
        
        # Mouse callback for clicking on objects
        cv2.setMouseCallback('HSV Calibration Tool', self.mouse_callback)
        
        # Create trackbars for HSV adjustment
        self.create_trackbars()
        
        # Position windows
        cv2.moveWindow('HSV Calibration Tool', 50, 50)
        cv2.moveWindow('HSV Controls', 700, 50)
        cv2.moveWindow('Mask Preview', 50, 600)
    
    def create_trackbars(self):
        """Create HSV adjustment trackbars."""
        color = self.current_color
        
        if color == "red":
            # Red has dual ranges
            range_idx = self.current_red_range
            lower = self.red_ranges[range_idx]["lower"]
            upper = self.red_ranges[range_idx]["upper"]
            window_title = f'HSV Controls - RED (Range {range_idx + 1}/2)'
        else:
            lower = self.hsv_ranges[color]["lower"]
            upper = self.hsv_ranges[color]["upper"]
            window_title = f'HSV Controls - {color.upper()}'
        
        # Destroy existing window and recreate with new title
        cv2.destroyWindow('HSV Controls')
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        
        # Create trackbars
        cv2.createTrackbar('H Min', window_title, lower[0], 179, self.on_trackbar)
        cv2.createTrackbar('S Min', window_title, lower[1], 255, self.on_trackbar)
        cv2.createTrackbar('V Min', window_title, lower[2], 255, self.on_trackbar)
        cv2.createTrackbar('H Max', window_title, upper[0], 179, self.on_trackbar)
        cv2.createTrackbar('S Max', window_title, upper[1], 255, self.on_trackbar)
        cv2.createTrackbar('V Max', window_title, upper[2], 255, self.on_trackbar)
        
        # Move window to consistent position
        cv2.moveWindow(window_title, 700, 50)
    
    def on_trackbar(self, val):
        """Handle trackbar changes."""
        color = self.current_color
        window_title = f'HSV Controls - {color.upper()}'
        
        if color == "red":
            window_title = f'HSV Controls - RED (Range {self.current_red_range + 1}/2)'
            range_data = self.red_ranges[self.current_red_range]
        else:
            range_data = self.hsv_ranges[color]
        
        # Update HSV values from trackbars
        try:
            range_data["lower"][0] = cv2.getTrackbarPos('H Min', window_title)
            range_data["lower"][1] = cv2.getTrackbarPos('S Min', window_title)
            range_data["lower"][2] = cv2.getTrackbarPos('V Min', window_title)
            range_data["upper"][0] = cv2.getTrackbarPos('H Max', window_title)
            range_data["upper"][1] = cv2.getTrackbarPos('S Max', window_title)
            range_data["upper"][2] = cv2.getTrackbarPos('V Max', window_title)
        except:
            pass  # Window might not exist yet
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for HSV sampling."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Sample HSV value at clicked point
            if hasattr(self, 'current_frame'):
                hsv_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
                hsv_value = hsv_frame[y, x]
                
                self.sample_points.append((x, y))
                self.sample_hsv_values.append(hsv_value)
                
                print(f"📍 Sampled {self.current_color} at ({x}, {y}): HSV = {hsv_value}")
                
                # Auto-adjust HSV ranges based on samples
                self.auto_adjust_ranges()
    
    def auto_adjust_ranges(self):
        """Automatically adjust HSV ranges based on sampled values."""
        if not self.sample_hsv_values:
            return
        
        # Get HSV values for current color samples
        hsv_array = np.array(self.sample_hsv_values)
        
        # Calculate min/max with some margin
        h_min, h_max = int(hsv_array[:, 0].min()), int(hsv_array[:, 0].max())
        s_min, s_max = int(hsv_array[:, 1].min()), int(hsv_array[:, 1].max())
        v_min, v_max = int(hsv_array[:, 2].min()), int(hsv_array[:, 2].max())
        
        # Add margins for robustness
        h_margin = max(5, (h_max - h_min) // 4)
        s_margin = max(20, (s_max - s_min) // 4)
        v_margin = max(20, (v_max - v_min) // 4)
        
        # Apply margins
        h_min = max(0, h_min - h_margin)
        h_max = min(179, h_max + h_margin)
        s_min = max(0, s_min - s_margin)
        s_max = min(255, s_max + s_margin)
        v_min = max(0, v_min - v_margin)
        v_max = min(255, v_max + v_margin)
        
        # Update ranges
        color = self.current_color
        if color == "red":
            # Handle red's special case (hue wrapping)
            if h_max > 170 or h_min < 10:
                # Spans across hue boundary
                self.red_ranges[0]["lower"] = [0, s_min, v_min]
                self.red_ranges[0]["upper"] = [min(10, h_max), s_max, v_max]
                self.red_ranges[1]["lower"] = [max(170, h_min), s_min, v_min]
                self.red_ranges[1]["upper"] = [179, s_max, v_max]
            else:
                # Normal red range
                range_idx = 0 if h_max <= 90 else 1
                self.red_ranges[range_idx]["lower"] = [h_min, s_min, v_min]
                self.red_ranges[range_idx]["upper"] = [h_max, s_max, v_max]
        else:
            self.hsv_ranges[color]["lower"] = [h_min, s_min, v_min]
            self.hsv_ranges[color]["upper"] = [h_max, s_max, v_max]
        
        # Update trackbars
        self.update_trackbars()
        
        print(f"🔧 Auto-adjusted {color} range: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
    
    def update_trackbars(self):
        """Update trackbar positions to match current HSV ranges."""
        color = self.current_color
        
        if color == "red":
            window_title = f'HSV Controls - RED (Range {self.current_red_range + 1}/2)'
            range_data = self.red_ranges[self.current_red_range]
        else:
            window_title = f'HSV Controls - {color.upper()}'
            range_data = self.hsv_ranges[color]
        
        try:
            cv2.setTrackbarPos('H Min', window_title, range_data["lower"][0])
            cv2.setTrackbarPos('S Min', window_title, range_data["lower"][1])
            cv2.setTrackbarPos('V Min', window_title, range_data["lower"][2])
            cv2.setTrackbarPos('H Max', window_title, range_data["upper"][0])
            cv2.setTrackbarPos('S Max', window_title, range_data["upper"][1])
            cv2.setTrackbarPos('V Max', window_title, range_data["upper"][2])
        except:
            pass
    
    def capture_frame(self):
        """Capture frame from camera."""
        if self.use_orbbec and self.orbbec_pipeline:
            try:
                frames = self.orbbec_pipeline.wait_for_frames(100)
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        # Convert Orbbec frame to OpenCV format
                        width = color_frame.get_width()
                        height = color_frame.get_height()
                        data = np.asanyarray(color_frame.get_data())
                        
                        if color_frame.get_format() == OBFormat.RGB:
                            image = data.reshape((height, width, 3))
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        else:
                            image = data.reshape((height, width, 3))
                        
                        return image
            except:
                pass
        
        # Fallback to standard camera
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                return frame
        
        return None
    
    def create_mask(self, hsv_frame, color):
        """Create mask for specified color."""
        if color == "red":
            # Red uses dual ranges
            mask1 = cv2.inRange(hsv_frame, 
                               np.array(self.red_ranges[0]["lower"]), 
                               np.array(self.red_ranges[0]["upper"]))
            mask2 = cv2.inRange(hsv_frame, 
                               np.array(self.red_ranges[1]["lower"]), 
                               np.array(self.red_ranges[1]["upper"]))
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(hsv_frame,
                              np.array(self.hsv_ranges[color]["lower"]),
                              np.array(self.hsv_ranges[color]["upper"]))
    
    def switch_color(self, new_color):
        """Switch to calibrating a different color."""
        if new_color != self.current_color:
            self.current_color = new_color
            self.current_red_range = 0
            self.sample_points.clear()
            self.sample_hsv_values.clear()
            self.create_trackbars()
            print(f"\n🔴 Now calibrating: {new_color.upper()}")
            print("Click on objects of this color to sample HSV values")
    
    def switch_red_range(self):
        """Switch between red's dual ranges."""
        if self.current_color == "red":
            self.current_red_range = 1 - self.current_red_range
            self.create_trackbars()
            print(f"🔴 Switched to RED range {self.current_red_range + 1}/2")
    
    def test_detection(self):
        """Test current HSV ranges on all colors."""
        print("\n🧪 Testing detection with current ranges...")
        
        frame = self.capture_frame()
        if frame is None:
            print("❌ Could not capture frame for testing")
            return
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        test_frame = frame.copy()
        
        colors = ["red", "green", "blue", "yellow"]
        color_bgr = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255)
        }
        
        for color in colors:
            mask = self.create_mask(hsv_frame, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects_found = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    # Draw bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(test_frame, (x, y), (x + w, y + h), color_bgr[color], 2)
                    cv2.putText(test_frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color], 2)
                    objects_found += 1
            
            print(f"  {color}: {objects_found} objects detected")
        
        cv2.imshow('Detection Test', test_frame)
        print("Press any key to close test window...")
        cv2.waitKey(0)
        cv2.destroyWindow('Detection Test')
    
    def save_calibration(self, filename=None):
        """Save current HSV calibration to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hsv_calibration_{timestamp}.json"
        
        # Prepare calibration data
        calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "description": "HSV color ranges for DOFBOT Pro colored blocks",
            "colors": {}
        }
        
        # Add regular colors
        for color in ["green", "blue", "yellow"]:
            calibration_data["colors"][color] = {
                "lower": self.hsv_ranges[color]["lower"],
                "upper": self.hsv_ranges[color]["upper"]
            }
        
        # Add red with dual ranges
        calibration_data["colors"]["red"] = {
            "ranges": [
                {
                    "lower": self.red_ranges[0]["lower"],
                    "upper": self.red_ranges[0]["upper"]
                },
                {
                    "lower": self.red_ranges[1]["lower"],
                    "upper": self.red_ranges[1]["upper"]
                }
            ]
        }
        
        # Save to file
        filepath = Path(__file__).parent / filename
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 Calibration saved to: {filepath}")
        return filepath
    
    def load_calibration(self, filename=None):
        """Load HSV calibration from file."""
        if filename is None:
            # Find most recent calibration file
            calibration_files = list(Path(__file__).parent.glob("hsv_calibration_*.json"))
            if not calibration_files:
                print("❌ No calibration files found")
                return False
            
            filename = max(calibration_files, key=os.path.getctime)
        
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            # Load regular colors
            for color in ["green", "blue", "yellow"]:
                if color in calibration_data["colors"]:
                    self.hsv_ranges[color]["lower"] = calibration_data["colors"][color]["lower"]
                    self.hsv_ranges[color]["upper"] = calibration_data["colors"][color]["upper"]
            
            # Load red ranges
            if "red" in calibration_data["colors"] and "ranges" in calibration_data["colors"]["red"]:
                red_data = calibration_data["colors"]["red"]["ranges"]
                if len(red_data) >= 2:
                    self.red_ranges[0]["lower"] = red_data[0]["lower"]
                    self.red_ranges[0]["upper"] = red_data[0]["upper"]
                    self.red_ranges[1]["lower"] = red_data[1]["lower"]
                    self.red_ranges[1]["upper"] = red_data[1]["upper"]
            
            self.update_trackbars()
            print(f"📂 Calibration loaded from: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False
    
    def export_for_robot(self, filename="robot_hsv_ranges.json"):
        """Export calibration in format suitable for robot actions."""
        # Format for robot code
        robot_ranges = {}
        
        # Regular colors (single range format)
        for color in ["green", "blue", "yellow"]:
            robot_ranges[color] = (
                tuple(self.hsv_ranges[color]["lower"]),
                tuple(self.hsv_ranges[color]["upper"])
            )
        
        # Red (dual range format)
        robot_ranges["red"] = [
            (tuple(self.red_ranges[0]["lower"]), tuple(self.red_ranges[0]["upper"])),
            (tuple(self.red_ranges[1]["lower"]), tuple(self.red_ranges[1]["upper"]))
        ]
        
        # Save robot-compatible format
        filepath = Path(__file__).parent / filename
        with open(filepath, 'w') as f:
            json.dump(robot_ranges, f, indent=2)
        
        print(f"🤖 Robot-compatible ranges exported to: {filepath}")
        
        # Also print Python code format
        print("\n📋 Copy this code into your robot actions:")
        print("color_hsv_ranges = {")
        for color in ["green", "blue", "yellow"]:
            lower = tuple(self.hsv_ranges[color]["lower"])
            upper = tuple(self.hsv_ranges[color]["upper"])
            print(f'    "{color}": ({lower}, {upper}),')
        
        print('    "red": [')
        for i, range_data in enumerate(self.red_ranges):
            lower = tuple(range_data["lower"])
            upper = tuple(range_data["upper"])
            print(f'        ({lower}, {upper}),')
        print('    ]')
        print("}")
        
        return filepath
    
    def run(self):
        """Main calibration loop."""
        print("\n🚀 Starting HSV calibration...")
        
        while True:
            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                print("❌ Could not capture frame")
                break
            
            self.current_frame = frame.copy()
            
            # Convert to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for current color
            mask = self.create_mask(hsv_frame, self.current_color)
            
            # Draw sample points
            display_frame = frame.copy()
            for point in self.sample_points:
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            
            # Add instructions
            cv2.putText(display_frame, f"Calibrating: {self.current_color.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Click on objects to sample HSV", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, "R/G/B/Y: switch color, T: test, S: save, Q: quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.current_color == "red":
                cv2.putText(display_frame, f"Red Range: {self.current_red_range + 1}/2 (Space: switch)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frames
            cv2.imshow('HSV Calibration Tool', display_frame)
            cv2.imshow('Mask Preview', mask)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):
                self.switch_color("red")
            elif key == ord('g'):
                self.switch_color("green")
            elif key == ord('b'):
                self.switch_color("blue")
            elif key == ord('y'):
                self.switch_color("yellow")
            elif key == ord(' ') and self.current_color == "red":
                self.switch_red_range()
            elif key == ord('t'):
                self.test_detection()
            elif key == ord('s'):
                filepath = self.save_calibration()
                self.export_for_robot()
            elif key == ord('l'):
                self.load_calibration()
            elif key == ord('c'):
                # Clear samples
                self.sample_points.clear()
                self.sample_hsv_values.clear()
                print(f"🧹 Cleared samples for {self.current_color}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.release()
        
        if self.orbbec_pipeline:
            try:
                self.orbbec_pipeline.stop()
            except:
                pass
        
        cv2.destroyAllWindows()
        print("\n👋 HSV Calibration Tool closed")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="HSV Color Calibration Tool for DOFBOT Pro")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-orbbec", action="store_true", help="Don't use Orbbec camera")
    args = parser.parse_args()
    
    try:
        tool = HSVCalibrationTool(use_orbbec=not args.no_orbbec, camera_index=args.camera)
        tool.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
