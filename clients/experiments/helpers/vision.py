#!/usr/bin/env python3
"""
Vision processing helper for block detection.

Detects colored blocks in a 2x3 grid layout (using first 4 positions) using OpenCV.

IMPORTANT ARCHITECTURAL NOTE:
This module runs on the CLIENT and processes images retrieved from the Data Manager.
It does NOT access ROS topics or robot hardware directly - that's the robot node's job.

MADSci Pattern:
    Robot Node: Captures image from camera → Stores in Data Manager → Returns datapoint ID
    Client: Retrieves image from Data Manager → Processes with this module → Makes decisions

This separation enables:
    - Distributed compute (offload heavy processing from robot)
    - Portability (client doesn't need ROS installed)
    - Scalability (one client can process data from multiple robots)
"""
import cv2 as cv
import numpy as np


def detect_blocks(image):
    """
    Detect colored blocks in a 2x3 grid (first 4 positions) from a provided image.
    
    Args:
        image: Numpy array in BGR format (from Data Manager)
    
    Returns:
        list: Array of length 4, where each index represents a position (0-3).
              Position layout: [0, 1, 2]  (top row: left, middle, right)
                               [3, 4, 5]  (bottom row: left only)
              Each element contains: 'red', 'yellow', 'green', or 'blue'
    
    Note:
        This function does NOT access ROS topics or robot hardware.
        It only processes the image data passed to it.
    """
    # HSV color ranges (from implementation plan Section 2.2)
    colors = {
        'red': ([0, 50, 50], [10, 255, 255]),      # Note: red wraps around hue
        'yellow': ([20, 50, 50], [35, 255, 255]),
        'blue': ([85, 40, 40], [130, 255, 255]),
        'green': ([35, 40, 40], [90, 255, 255])
    }
    
    # Validate input
    if image is None:
        print("✗ No image provided")
        return [None, None, None, None]
    
    # Resize to standard dimensions
    img_width, img_height = 640, 480
    frame = cv.resize(image, (img_width, img_height))
    print(f"✓ Processing image: {frame.shape[1]}x{frame.shape[0]}")
    
    # Define quadrant dimensions (2 rows x 3 columns, using first 4 positions)
    # Layout: [0, 1, 2]  <- top row (we use all 3)
    #         [3, 4, 5]  <- bottom row (we only use position 3)
    col_width = img_width // 3   # 213 pixels per column
    row_height = img_height // 2  # 240 pixels per row
    
    # Define the 4 quadrants we care about
    quadrants = [
        (0, 0, col_width, row_height),              # Q0: top-left
        (col_width, 0, col_width*2, row_height),    # Q1: top-middle
        (col_width*2, 0, img_width, row_height),    # Q2: top-right
        (0, row_height, col_width, img_height)      # Q3: bottom-left
    ]
    
    # Initialize result array
    quadrant_colors = [None] * 4
    
    # For each quadrant, find the most confident color
    for q_idx, (x1, y1, x2, y2) in enumerate(quadrants):
        # Extract quadrant image
        quadrant_img = frame[y1:y2, x1:x2]
        hsv_quadrant = cv.cvtColor(quadrant_img, cv.COLOR_BGR2HSV)
        
        # Calculate percentage for each color in this quadrant
        color_percentages = {}
        total_pixels = quadrant_img.shape[0] * quadrant_img.shape[1]
        
        for color_name, (lower_hsv, upper_hsv) in colors.items():
            lower = np.array(lower_hsv)
            upper = np.array(upper_hsv)
            
            # Special handling for red (wraps around hue circle)
            if color_name == 'red':
                # Red range 1: 0-10
                lower1 = np.array([0, 50, 50])
                upper1 = np.array([10, 255, 255])
                mask1 = cv.inRange(hsv_quadrant, lower1, upper1)
                
                # Red range 2: 170-180
                lower2 = np.array([170, 50, 50])
                upper2 = np.array([180, 255, 255])
                mask2 = cv.inRange(hsv_quadrant, lower2, upper2)
                
                # Combine both red ranges
                mask = cv.bitwise_or(mask1, mask2)
            else:
                mask = cv.inRange(hsv_quadrant, lower, upper)
            
            # Apply morphological operations to reduce noise
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            
            # Count pixels
            color_pixels = cv.countNonZero(mask)
            percentage = (color_pixels / total_pixels) * 100
            color_percentages[color_name] = percentage
        
        # Pick the color with highest percentage
        if color_percentages:
            best_color = max(color_percentages.items(), key=lambda x: x[1])
            quadrant_colors[q_idx] = best_color[0]
            print(f"  Q{q_idx}: {best_color[0].upper():6s} ({best_color[1]:.1f}%)")
        else:
            # Fallback
            quadrant_colors[q_idx] = 'red'
            print(f"  Q{q_idx}: RED (fallback)")
    
    print(f"Final detection: {quadrant_colors}")
    return quadrant_colors
