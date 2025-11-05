#!/usr/bin/env python3
"""Quick test of vision helper with real robot image."""
import os
import sys
sys.path.insert(0, '.')

from madsci.client import WorkcellClient, DataClient
from helpers.block_permutations.workflows import create_scan_workflow
from helpers.vision import detect_blocks

# Get service URLs
WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
DATA_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8004")

print("=== Vision Helper Test ===\n")

# Step 1: Initialize clients
print("Step 1: Initializing clients...")
workcell_client = WorkcellClient(WORKCELL_URL)
data_client = DataClient(DATA_URL)
print("✓ Clients initialized\n")

# Step 2: Run scan workflow (move to home + capture)
print("Step 2: Running scan workflow...")
print("  - Moving robot to home position")
print("  - Capturing image from camera")
scan_workflow = create_scan_workflow()
result = workcell_client.start_workflow(scan_workflow)
print("✓ Workflow completed\n")

# Step 3: Retrieve image from Data Manager
print("Step 3: Retrieving image from Data Manager...")
# The action returns a datapoint ID which is stored as a ValueDataPoint
# We need to get that value first, then use it to get the actual image
capture_step = [step for step in result.steps if step.name == "capture"][0]
wrapper_id = capture_step.result.datapoints.json_result

print(f"  Wrapper ID: {wrapper_id}")
# Get the actual image datapoint ID
image_id = data_client.get_datapoint_value(wrapper_id)
print(f"  Image ID: {image_id}")

# Save the image file locally
import tempfile
from pathlib import Path
temp_dir = Path(tempfile.mkdtemp())
image_path = temp_dir / "captured_image.jpg"
data_client.save_datapoint_value(image_id, str(image_path))

# Load the image with OpenCV
import cv2
image_data = cv2.imread(str(image_path))

print(f"  Image data type: {type(image_data)}")
if image_data is None or not hasattr(image_data, 'shape'):
    print(f"  Unexpected data: {image_data}")
    raise ValueError(f"Expected numpy array, got: {type(image_data)}")
print(f"  Image shape: {image_data.shape}")
print(f"  Image dtype: {image_data.dtype}")
print("✓ Image retrieved\n")

# Step 4: Process image with vision helper
print("Step 4: Processing image with detect_blocks()...")
colors = detect_blocks(image_data)
print(f"\n✓ Detection complete!")
print(f"\nResult: {colors}")
print(f"Layout: [0, 1]  <- Top row")
print(f"        [2, 3]  <- Bottom row")
print(f"\nColors detected:")
print(f"  Q0 (top-left):     {colors[0]}")
print(f"  Q1 (top-right):    {colors[1]}")
print(f"  Q2 (bottom-left):  {colors[2]}")
print(f"  Q3 (bottom-right): {colors[3]}")

print("\n=== Test Complete ===")
