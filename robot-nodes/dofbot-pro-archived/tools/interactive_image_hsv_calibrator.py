#!/usr/bin/env python3
"""
Interactive HSV Calibration Tool for Saved Images

This tool allows you to calibrate HSV color ranges using the exact same images
that the scanning pipeline captures, ensuring perfect alignment between 
calibration and detection conditions.

Features:
- Load multiple images from experiment directories
- Cycle between images using keyboard controls
- Interactive HSV range adjustment with trackbars
- Click on image to sample HSV values
- Export calibrated ranges in config format
- Real-time mask preview

Usage:
    python tools/interactive_image_hsv_calibrator.py [image_directory]
    
Controls:
    - Click on image: Sample HSV values at that point
    - 'n': Next image
    - 'p': Previous image  
    - 'r': Reset HSV ranges to defaults
    - 's': Save current ranges to file
    - 'q': Quit
"""

import cv2
import numpy as np
import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any

class InteractiveImageHSVCalibrator:
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        self.current_image_index = 0
        self.current_image = None
        self.current_hsv = None
        
        # HSV range variables
        self.h_min = 0
        self.h_max = 179
        self.s_min = 0
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255
        
        # Click tracking and HSV sampling
        self.click_points = []
        self.sample_hsv_values = []
        
        # Window names
        self.main_window = "HSV Calibrator - Original Image"
        self.mask_window = "HSV Calibrator - Color Mask"
        self.control_window = "HSV Controls"
        
        # Color being calibrated
        self.color_name = "unknown"
        
        # Results storage
        self.calibrated_ranges = {}
        
        print(f"Loaded {len(self.image_paths)} images for calibration")
        print("Controls:")
        print("  Click on image: Sample HSV values (green dots will appear)")
        print("  'n': Next image")
        print("  'p': Previous image")
        print("  'r': Reset ranges")
        print("  's': Save ranges")
        print("  'q': Quit")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to sample HSV values."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_hsv is not None:
            # Get HSV value at clicked point
            hsv_value = self.current_hsv[y, x]
            h, s, v = hsv_value
            
            # Store click point and HSV value
            self.click_points.append((x, y))
            self.sample_hsv_values.append([h, s, v])
            
            print(f"📍 Clicked at ({x}, {y}) - HSV: ({h}, {s}, {v})")
            
            # Auto-adjust ranges based on all samples
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
        
        # Update trackbars
        cv2.setTrackbarPos("H Min", self.control_window, h_min)
        cv2.setTrackbarPos("H Max", self.control_window, h_max)
        cv2.setTrackbarPos("S Min", self.control_window, s_min)
        cv2.setTrackbarPos("S Max", self.control_window, s_max)
        cv2.setTrackbarPos("V Min", self.control_window, v_min)
        cv2.setTrackbarPos("V Max", self.control_window, v_max)
        
        # Update ranges and display
        self.update_ranges()
        
        print(f"🔧 Auto-adjusted {self.color_name} range: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            
    def trackbar_callback(self, val):
        """Handle trackbar changes."""
        self.update_ranges()
        
    def update_ranges(self):
        """Update HSV ranges from trackbars and refresh display."""
        self.h_min = cv2.getTrackbarPos("H Min", self.control_window)
        self.h_max = cv2.getTrackbarPos("H Max", self.control_window)
        self.s_min = cv2.getTrackbarPos("S Min", self.control_window)
        self.s_max = cv2.getTrackbarPos("S Max", self.control_window)
        self.v_min = cv2.getTrackbarPos("V Min", self.control_window)
        self.v_max = cv2.getTrackbarPos("V Max", self.control_window)
        
        self.update_display()
        
    def load_image(self, index: int):
        """Load image at specified index."""
        if 0 <= index < len(self.image_paths):
            image_path = self.image_paths[index]
            self.current_image = cv2.imread(image_path)
            
            if self.current_image is not None:
                # Convert to HSV
                self.current_hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
                
                # Clear previous click points when loading new image
                self.click_points = []
                
                # Update window title with current image info
                image_name = os.path.basename(image_path)
                title = f"HSV Calibrator - {image_name} ({index + 1}/{len(self.image_paths)})"
                cv2.setWindowTitle(self.main_window, title)
                
                print(f"Loaded: {image_name}")
                self.update_display()
                return True
            else:
                print(f"Failed to load image: {image_path}")
                return False
        return False
        
    def create_mask(self, hsv_image):
        """Create mask using current HSV ranges, handling red's dual ranges."""
        if self.color_name == "red":
            # Red needs dual ranges to handle hue wrapping
            # Lower red range (0-10)
            lower_red1 = np.array([0, self.s_min, self.v_min])
            upper_red1 = np.array([10, self.s_max, self.v_max])
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            
            # Upper red range (170-179)
            lower_red2 = np.array([170, self.s_min, self.v_min])
            upper_red2 = np.array([179, self.s_max, self.v_max])
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            
            # Combine both red ranges
            return cv2.bitwise_or(mask1, mask2)
        else:
            # Single range for other colors
            lower_hsv = np.array([self.h_min, self.s_min, self.v_min])
            upper_hsv = np.array([self.h_max, self.s_max, self.v_max])
            return cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
    def update_display(self):
        """Update the display with current image and mask."""
        if self.current_image is None or self.current_hsv is None:
            return
            
        # Create mask with current HSV ranges
        mask = self.create_mask(self.current_hsv)
        
        # Create colored mask for better visualization
        colored_mask = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
        
        # Display original image with click points
        display_image = self.current_image.copy()
        
        # Draw green dots for click points
        for point in self.click_points:
            cv2.circle(display_image, point, 5, (0, 255, 0), -1)
        
        # Add HSV range text overlay
        if self.color_name == "red":
            range_text = f"RED: S[{self.s_min}-{self.s_max}], V[{self.v_min}-{self.v_max}] (dual H ranges)"
        else:
            range_text = f"{self.color_name.upper()}: H[{self.h_min}-{self.h_max}], S[{self.s_min}-{self.s_max}], V[{self.v_min}-{self.v_max}]"
        cv2.putText(display_image, range_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Count matching pixels
        matching_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (matching_pixels / total_pixels) * 100
        
        pixel_text = f"Matching: {matching_pixels}/{total_pixels} ({percentage:.1f}%)"
        cv2.putText(display_image, pixel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add click instructions
        click_text = f"Clicks: {len(self.click_points)} (Click on {self.color_name} cube)"
        cv2.putText(display_image, click_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(self.main_window, display_image)
        cv2.imshow(self.mask_window, colored_mask)
        
    def setup_windows(self):
        """Setup OpenCV windows and trackbars."""
        # Create windows
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.mask_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        
        # Set mouse callback
        cv2.setMouseCallback(self.main_window, self.mouse_callback)
        
        # Initialize ranges based on color
        if self.color_name == "red":
            # Red uses S and V ranges, H is handled by dual ranges
            self.h_min, self.h_max = 0, 179  # Full hue range (not used for red)
            self.s_min, self.s_max = 50, 255
            self.v_min, self.v_max = 50, 255
        elif self.color_name == "green":
            self.h_min, self.h_max = 40, 80
            self.s_min, self.s_max = 40, 255
            self.v_min, self.v_max = 40, 255
        elif self.color_name == "blue":
            self.h_min, self.h_max = 100, 130
            self.s_min, self.s_max = 50, 255
            self.v_min, self.v_max = 50, 255
        elif self.color_name == "yellow":
            self.h_min, self.h_max = 20, 30
            self.s_min, self.s_max = 100, 255
            self.v_min, self.v_max = 100, 255
        else:
            # Default ranges
            self.h_min, self.h_max = 0, 179
            self.s_min, self.s_max = 0, 255
            self.v_min, self.v_max = 0, 255
        
        # Create trackbars
        cv2.createTrackbar("H Min", self.control_window, self.h_min, 179, self.trackbar_callback)
        cv2.createTrackbar("H Max", self.control_window, self.h_max, 179, self.trackbar_callback)
        cv2.createTrackbar("S Min", self.control_window, self.s_min, 255, self.trackbar_callback)
        cv2.createTrackbar("S Max", self.control_window, self.s_max, 255, self.trackbar_callback)
        cv2.createTrackbar("V Min", self.control_window, self.v_min, 255, self.trackbar_callback)
        cv2.createTrackbar("V Max", self.control_window, self.v_max, 255, self.trackbar_callback)
        
        # Resize control window
        cv2.resizeWindow(self.control_window, 400, 300)
        
    def reset_ranges(self):
        """Reset HSV ranges to defaults for current color."""
        if self.color_name == "red":
            self.s_min, self.s_max = 50, 255
            self.v_min, self.v_max = 50, 255
        elif self.color_name == "green":
            self.h_min, self.h_max = 40, 80
            self.s_min, self.s_max = 40, 255
            self.v_min, self.v_max = 40, 255
        elif self.color_name == "blue":
            self.h_min, self.h_max = 100, 130
            self.s_min, self.s_max = 50, 255
            self.v_min, self.v_max = 50, 255
        elif self.color_name == "yellow":
            self.h_min, self.h_max = 20, 30
            self.s_min, self.s_max = 100, 255
            self.v_min, self.v_max = 100, 255
        else:
            self.h_min, self.h_max = 0, 179
            self.s_min, self.s_max = 0, 255
            self.v_min, self.v_max = 0, 255
            
        # Update trackbars
        cv2.setTrackbarPos("H Min", self.control_window, self.h_min)
        cv2.setTrackbarPos("H Max", self.control_window, self.h_max)
        cv2.setTrackbarPos("S Min", self.control_window, self.s_min)
        cv2.setTrackbarPos("S Max", self.control_window, self.s_max)
        cv2.setTrackbarPos("V Min", self.control_window, self.v_min)
        cv2.setTrackbarPos("V Max", self.control_window, self.v_max)
        
        # Clear samples and update display
        self.click_points = []
        self.sample_hsv_values = []
        self.update_display()
        print(f"Reset {self.color_name} HSV ranges to defaults")
        
    def save_ranges(self):
        """Save current HSV ranges."""
        if not self.color_name or self.color_name == "unknown":
            self.color_name = input("Enter color name for these ranges: ").strip().lower()
            
        if self.color_name == "red":
            # Red uses dual ranges
            current_range = [
                ((0, self.s_min, self.v_min), (10, self.s_max, self.v_max)),
                ((170, self.s_min, self.v_min), (179, self.s_max, self.v_max))
            ]
        else:
            # Single range for other colors
            current_range = ((self.h_min, self.s_min, self.v_min), (self.h_max, self.s_max, self.v_max))
            
        self.calibrated_ranges[self.color_name] = current_range
        
        # Save to file
        output_file = f"calibrated_hsv_ranges_{self.color_name}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "color": self.color_name,
                "hsv_range": current_range,
                "config_format": f'"{self.color_name}": {current_range},',
                "click_points": len(self.click_points),
                "sample_count": len(self.sample_hsv_values)
            }, f, indent=2)
            
        print(f"Saved {self.color_name} HSV range: {current_range}")
        print(f"Config format: \"{self.color_name}\": {current_range},")
        print(f"Saved to: {output_file}")
        print(f"Based on {len(self.click_points)} click samples")
        
    def run(self):
        """Main calibration loop."""
        if not self.image_paths:
            print("No images to calibrate!")
            return
            
        self.setup_windows()
        
        # Load first image
        if not self.load_image(0):
            print("Failed to load first image!")
            return
            
        print(f"Starting calibration with {len(self.image_paths)} images")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Next image
                next_index = (self.current_image_index + 1) % len(self.image_paths)
                if self.load_image(next_index):
                    self.current_image_index = next_index
            elif key == ord('p'):
                # Previous image
                prev_index = (self.current_image_index - 1) % len(self.image_paths)
                if self.load_image(prev_index):
                    self.current_image_index = prev_index
            elif key == ord('r'):
                self.reset_ranges()
            elif key == ord('s'):
                self.save_ranges()
                
        cv2.destroyAllWindows()
        
        # Print final summary
        if self.calibrated_ranges:
            print("\nCalibrated Ranges Summary:")
            for color, range_tuple in self.calibrated_ranges.items():
                print(f'"{color}": {range_tuple},')

def find_scanning_images(experiment_dir: str, color: str = None) -> List[str]:
    """Find scanning images in experiment directory."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return []
        
    # Look for scanning event images
    scanning_events_dir = experiment_path / "scanning_events"
    if not scanning_events_dir.exists():
        print(f"No scanning_events directory found in: {experiment_dir}")
        return []
        
    # Find original images
    pattern = "*_01_original.jpg"
    if color:
        pattern = f"*{color}*_01_original.jpg"
        
    image_paths = list(scanning_events_dir.glob(pattern))
    
    if not image_paths:
        print(f"No scanning images found with pattern: {pattern}")
        return []
        
    # Sort by position (extract position from filename)
    def extract_position(path):
        filename = path.name
        if "pos" in filename:
            try:
                pos_part = filename.split("pos")[1].split("_")[0]
                return int(pos_part)
            except:
                return 0
        return 0
        
    image_paths.sort(key=extract_position)
    
    print(f"Found {len(image_paths)} scanning images:")
    for path in image_paths:
        print(f"  {path.name}")
        
    return [str(path) for path in image_paths]

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_image_hsv_calibrator.py <color_directory>")
        print("Examples:")
        print("  python interactive_image_hsv_calibrator.py red")
        print("  python interactive_image_hsv_calibrator.py blue")
        print("  python interactive_image_hsv_calibrator.py yellow")
        print("  python interactive_image_hsv_calibrator.py green")
        print()
        print("Expected directory structure:")
        print("  tools/")
        print("    red/")
        print("      pos180_red_cube_*.jpg")
        print("      pos135_red_cube_*.jpg")
        print("      ...")
        print("    blue/")
        print("      pos45_blue_cube_*.jpg")
        print("      ...")
        return
        
    color_dir = sys.argv[1]
    color_path = Path(color_dir)
    
    if not color_path.exists():
        print(f"Color directory not found: {color_dir}")
        print("Please create the directory and add your cube images.")
        return
        
    # Find all image files in the color directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(color_path.glob(ext))
        
    if not image_paths:
        print(f"No images found in directory: {color_dir}")
        print("Please add your cube images to this directory.")
        return
        
    # Sort by position if possible
    def extract_position(path):
        filename = path.name.lower()
        if "pos" in filename:
            try:
                pos_part = filename.split("pos")[1].split("_")[0]
                return int(pos_part)
            except:
                return 999
        return 999
        
    image_paths.sort(key=extract_position)
    
    print(f"Found {len(image_paths)} images in {color_dir}/ directory:")
    for path in image_paths:
        print(f"  {path.name}")
        
    # Create and run calibrator
    calibrator = InteractiveImageHSVCalibrator([str(p) for p in image_paths])
    calibrator.color_name = color_dir  # Use directory name as color name
        
    calibrator.run()

if __name__ == "__main__":
    main()
