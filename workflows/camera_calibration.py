#!/usr/bin/python

from madsci.client.workcell_client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition
import time
import os
from dotenv import load_dotenv

# Load environment variables from root directory
import pathlib
root_dir = pathlib.Path(__file__).parent.parent
load_dotenv(root_dir / '.env')

workcell_client = WorkcellClient(
    workcell_server_url=os.getenv("WORKCELL_MANAGER_URL", "http://localhost:8005")
)

# Camera Calibration workflow
calibration_workflow = WorkflowDefinition(
    name="Camera Calibration Workflow",
    description="Calibrate the robot arm's depth camera using 11x8 checkerboard pattern",
    steps=[
        StepDefinition(
            name="Calibrate Depth Camera",
            node="DOFBOT_Pro_1",  # Must match your node name in workcell config
            action="calibrate_depth_camera",
            args={
                "checkerboard_width": 10,      # 11x8 board = 10x7 internal corners
                "checkerboard_height": 7,
                "square_size_mm": 25.0,
                "num_images": 15,
                "capture_delay": 3.0,          # Give yourself time to reposition checkerboard
                "save_calibration": True
            }
        )
    ]
)

print("Starting camera calibration workflow...")
print("Make sure you have:")
print("1. An 11x8 checkerboard pattern printed and mounted on rigid surface")
print("2. 25mm squares (or adjust square_size_mm parameter)")
print("3. Good lighting on the checkerboard")
print("4. Camera pipeline will start automatically during calibration")
print()

# Submit and run the calibration workflow
result = workcell_client.submit_workflow(workflow=calibration_workflow)

print(f"Calibration workflow completed!")
print(f"Workflow ID: {result.workflow_id}")
print(f"Status: {result.status}")

if hasattr(result, 'result') and result.result:
    calibration_data = result.result
    if 'mean_reprojection_error' in calibration_data:
        print(f"Mean reprojection error: {calibration_data['mean_reprojection_error']:.3f} pixels")
    if 'quality_assessment' in calibration_data:
        print(f"Quality assessment: {calibration_data['quality_assessment']}")
    if 'num_images_used' in calibration_data:
        print(f"Images used: {calibration_data['num_images_used']}")

print("\nCalibration data saved to: ./expert_trajectories/camera_calibration.json")
print("Calibration images saved to: ./expert_trajectories/calibration_images/")
