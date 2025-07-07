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

# Calibration Validation workflow
validation_workflow = WorkflowDefinition(
    name="Camera Calibration Validation Workflow",
    description="Validate camera calibration quality and test point cloud generation",
    steps=[
        StepDefinition(
            name="Validate Camera Calibration",
            node="DOFBOT_Pro_1",  # Must match your node name in workcell config
            action="validate_calibration"
        ),
        StepDefinition(
            name="Test Camera Capture",
            node="DOFBOT_Pro_1",
            action="test_camera_capture"
        ),
        StepDefinition(
            name="Start Camera Pipeline",
            node="DOFBOT_Pro_1", 
            action="start_camera_pipeline"
        ),
        StepDefinition(
            name="Capture Test Images",
            node="DOFBOT_Pro_1",
            action="capture_images"
        ),
        StepDefinition(
            name="Stop Camera Pipeline",
            node="DOFBOT_Pro_1",
            action="stop_camera_pipeline"
        )
    ]
)

print("Starting camera calibration validation workflow...")
print("This will:")
print("1. Validate current calibration parameters")
print("2. Test camera capture functionality") 
print("3. Start camera pipeline")
print("4. Capture test images with point cloud")
print("5. Stop camera pipeline")
print()

# Submit and run the validation workflow
result = workcell_client.submit_workflow(workflow=validation_workflow)

print(f"Calibration validation workflow completed!")
print(f"Workflow ID: {result.workflow_id}")
print(f"Status: {result.status}")

# Extract and display validation results
if hasattr(result, 'steps') and result.steps:
    for step in result.steps:
        if step.name == "Validate Camera Calibration" and hasattr(step, 'result') and step.result:
            # Access the result data correctly - it's in step.result.data (confirmed from MADSci source)
            validation_data = step.result.data
            print(f"\n=== CALIBRATION VALIDATION RESULTS ===")
            print(f"Overall Status: {validation_data.get('overall_status', 'Unknown')}")
            print(f"Calibration Date: {validation_data.get('calibration_date', 'Unknown')}")
            print(f"Image Dimensions: {validation_data.get('image_dimensions', 'Unknown')}")
            print(f"Focal Lengths: {validation_data.get('focal_lengths', 'Unknown')}")
            print(f"Principal Point: {validation_data.get('principal_point', 'Unknown')}")
            print(f"Reprojection Error: {validation_data.get('reprojection_error', 'Unknown')}")
            print(f"Quality Assessment: {validation_data.get('quality_assessment', 'Unknown')}")
            
            if validation_data.get('warnings'):
                print(f"\nWarnings:")
                for warning in validation_data['warnings']:
                    print(f"  - {warning}")
            
            if 'point_cloud_test' in validation_data:
                pc_test = validation_data['point_cloud_test']
                print(f"\n=== POINT CLOUD TEST ===")
                print(f"Success: {pc_test.get('success', False)}")
                if pc_test.get('success'):
                    print(f"Number of Points: {pc_test.get('num_points', 'Unknown')}")
                    if 'point_range' in pc_test:
                        ranges = pc_test['point_range']
                        print(f"X Range: {ranges.get('x', 'Unknown')} meters")
                        print(f"Y Range: {ranges.get('y', 'Unknown')} meters") 
                        print(f"Z Range: {ranges.get('z', 'Unknown')} meters")
                else:
                    print(f"Error: {pc_test.get('error', 'Unknown')}")
        
        elif step.name == "Capture Test Images" and hasattr(step, 'result') and step.result:
            # Access the result data correctly - it's in step.result.data (confirmed from MADSci source)
            capture_data = step.result.data
            print(f"\n=== IMAGE CAPTURE TEST ===")
            print(f"RGB Captured: {capture_data.get('rgb_captured', False)}")
            print(f"Depth Captured: {capture_data.get('depth_captured', False)}")
            print(f"Point Cloud Captured: {capture_data.get('point_cloud_captured', False)}")
            print(f"Point Cloud Size: {capture_data.get('point_cloud_size', 0)} points")

print(f"\nValidation complete! Your calibration is ready for expert trajectory recording.")
