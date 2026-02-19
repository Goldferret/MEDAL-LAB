#!/usr/bin/env python3
"""
Test vision-guided pick and place.

Captures color + depth images from the robot, detects a colored block,
and commands the robot to pick it up and place it at a target location.

Usage:
    python test_pick_and_place.py [color]
    
    color: red, blue, green, yellow (default: red)
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('../../.env.global')

from madsci.client import WorkcellClient, DataClient, ExperimentClient
from madsci.common.types.experiment_types import ExperimentDesign
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

# ============================================================================
# CONFIG
# ============================================================================

WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
DATA_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8004")
EXPERIMENT_URL = os.getenv("EXPERIMENT_SERVER_URL", "http://localhost:8002")
ROBOT_NODE = "DOFBOT_Pro_1"

# Scan position: camera looking down at workspace
SCAN_POSITION = [-0.07, 0.2, -1.5, -1.5, 0.0]

# HSV ranges (from color_pick_and_place.py, tuned for DOFBOT workspace)
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([160, 50, 100]), 'upper': np.array([180, 255, 255])}
    ],
    'blue': {'lower': np.array([100, 150, 100]), 'upper': np.array([130, 255, 255])},
    'green': {'lower': np.array([40, 60, 60]), 'upper': np.array([90, 255, 255])},
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
}

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 1000
MIN_WIDTH_PX = 15  # Minimum object width in pixels to reject noise
DEPTH_SAMPLE_RADIUS = 5

# Place locations in pixel space (approximate center-right of workspace)
# These are rough — adjust based on your workspace layout
PLACE_PIXEL = (500, 350)

# ============================================================================
# DETECTION (from color_pick_and_place.py, adapted for client-side)
# ============================================================================

def detect_block(color_image, depth_image, color_name):
    """
    Detect the largest block of specified color.
    
    Args:
        color_image: BGR numpy array
        depth_image: uint16 numpy array (mm)
        color_name: 'red', 'blue', 'green', or 'yellow'
    
    Returns:
        dict with pixel_x, pixel_y, depth_mm, width_px or None
    """
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    ranges = COLOR_RANGES[color_name]
    if not isinstance(ranges, list):
        ranges = [ranges]
    
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, r['lower'], r['upper'])
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA):
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(w) / h if h > 0 else 0
        if aspect < 0.5 or aspect > 2.0:
            continue
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # minAreaRect for rotation-invariant width
        rect = cv2.minAreaRect(contour)
        width_px = min(rect[1])  # Shorter side
        if width_px < MIN_WIDTH_PX:
            print(f"  Skipping: too small ({width_px:.0f}px)")
            continue
        
        # Sample depth (median over region, skip zeros)
        r = DEPTH_SAMPLE_RADIUS
        y_min = max(0, cy - r)
        y_max = min(depth_image.shape[0], cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(depth_image.shape[1], cx + r + 1)
        region = depth_image[y_min:y_max, x_min:x_max]
        valid = region[region > 0]
        if len(valid) == 0:
            print(f"  Skipping: no valid depth at ({cx}, {cy})")
            continue
        depth_mm = int(np.median(valid))
        
        if best is None or area > best['area']:
            best = {
                'pixel_x': cx,
                'pixel_y': cy,
                'depth_mm': depth_mm,
                'width_px': float(width_px),
                'area': area,
            }
    
    return best


def get_place_depth(depth_image, px, py):
    """Get median depth at a pixel location for placing."""
    r = DEPTH_SAMPLE_RADIUS
    y_min = max(0, py - r)
    y_max = min(depth_image.shape[0], py + r + 1)
    x_min = max(0, px - r)
    x_max = min(depth_image.shape[1], px + r + 1)
    region = depth_image[y_min:y_max, x_min:x_max]
    valid = region[region > 0]
    if len(valid) == 0:
        return None
    return int(np.median(valid))


# ============================================================================
# MAIN
# ============================================================================

def main():
    color_name = sys.argv[1] if len(sys.argv) > 1 else 'red'
    if color_name not in COLOR_RANGES:
        print(f"Unknown color: {color_name}. Choose from: {list(COLOR_RANGES.keys())}")
        sys.exit(1)
    
    print(f"=== Vision-Guided Pick & Place Test ({color_name}) ===\n")
    
    # Initialize clients
    workcell = WorkcellClient(WORKCELL_URL)
    data = DataClient(DATA_URL)
    experiment_client = ExperimentClient(EXPERIMENT_URL)
    
    # Start experiment
    experiment = experiment_client.start_experiment(
        experiment_design=ExperimentDesign(
            experiment_name=f"Pick and Place Test - {color_name}",
            experiment_description=f"Vision-guided pick and place of {color_name} block"
        ),
        run_name=f"pick_place_{color_name}",
        run_description=f"Detect and pick {color_name} block, place at target"
    )
    print(f"Experiment started: {experiment.experiment_id}\n")
    
    # Step 1: Capture images (overhead camera, no movement needed)
    print("Step 1: Capturing images from overhead camera...")
    scan_workflow = WorkflowDefinition(
        name="Scan",
        description="Capture color + depth from overhead camera",
        steps=[
            StepDefinition(
                name="capture",
                node=ROBOT_NODE,
                action="capture_camera_image",
                args={}
            )
        ]
    )
    result = workcell.start_workflow(scan_workflow)
    
    # Get datapoint IDs from capture step
    capture_step = [s for s in result.steps if s.name == "capture"][0]
    
    # json_result is a wrapper datapoint ID containing the actual result dict
    wrapper_id = capture_step.result.datapoints.json_result
    capture_result = data.get_datapoint_value(wrapper_id)
    
    color_id = capture_result["color_datapoint_id"]
    depth_id = capture_result["depth_datapoint_id"]
    
    print(f"  Color datapoint: {color_id}")
    print(f"  Depth datapoint: {depth_id}")
    
    # Step 2: Download images
    print("\nStep 2: Downloading images...")
    tmp = Path(tempfile.mkdtemp())
    
    color_path = tmp / "color.jpg"
    data.save_datapoint_value(color_id, str(color_path))
    color_image = cv2.imread(str(color_path))
    print(f"  Color image: {color_image.shape}")
    
    depth_path = tmp / "depth.npy"
    data.save_datapoint_value(depth_id, str(depth_path))
    depth_image = np.load(str(depth_path))
    print(f"  Depth image: {depth_image.shape}, dtype={depth_image.dtype}")
    
    # Step 3: Detect block
    print(f"\nStep 3: Detecting {color_name} block...")
    block = detect_block(color_image, depth_image, color_name)
    
    if block is None:
        print(f"  ✗ No {color_name} block found!")
        experiment_client.end_experiment(experiment.experiment_id)
        sys.exit(1)
    
    print(f"  ✓ Found at pixel ({block['pixel_x']}, {block['pixel_y']})")
    print(f"    Depth: {block['depth_mm']}mm")
    print(f"    Width: {block['width_px']:.0f}px")
    print(f"    Area:  {block['area']:.0f}px²")
    
    # Step 4: Get place location depth
    place_px, place_py = PLACE_PIXEL
    place_depth = get_place_depth(depth_image, place_px, place_py)
    if place_depth is None:
        print(f"  ✗ No valid depth at place location ({place_px}, {place_py})")
        experiment_client.end_experiment(experiment.experiment_id)
        sys.exit(1)
    print(f"\n  Place target: pixel ({place_px}, {place_py}), depth={place_depth}mm")
    
    # Step 5: Pick and place
    print("\nStep 4: Executing pick and place...")
    pick_place_workflow = WorkflowDefinition(
        name="Pick and Place",
        description=f"Pick {color_name} block and place at target",
        steps=[
            StepDefinition(
                name="pick",
                node=ROBOT_NODE,
                action="pick_at_pixel_coords",
                args={
                    "pixel_x": block['pixel_x'],
                    "pixel_y": block['pixel_y'],
                    "depth_mm": block['depth_mm'],
                    "width_px": block['width_px'],
                }
            ),
            StepDefinition(
                name="place",
                node=ROBOT_NODE,
                action="place_at_pixel_coords",
                args={
                    "pixel_x": place_px,
                    "pixel_y": place_py,
                    "depth_mm": place_depth,
                }
            ),
            StepDefinition(
                name="home",
                node=ROBOT_NODE,
                action="home_robot",
                args={}
            ),
        ]
    )
    result = workcell.start_workflow(pick_place_workflow)
    
    # Check results
    for step in result.steps:
        status = step.result.json_result.get("status", "unknown") if step.result.json_result else "no response"
        print(f"  {step.name}: {status}")
    
    # End experiment
    experiment_client.end_experiment(experiment.experiment_id)
    print(f"\n=== Test Complete ===")
    print(f"Experiment ID: {experiment.experiment_id}")


if __name__ == "__main__":
    main()
