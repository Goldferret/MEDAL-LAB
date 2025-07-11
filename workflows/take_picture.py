#!/usr/bin/python

from madsci.client.workcell_client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition
import os
from dotenv import load_dotenv

# Load environment variables from root directory
import pathlib
root_dir = pathlib.Path(__file__).parent.parent
load_dotenv(root_dir / '.env')

workcell_client = WorkcellClient(
    workcell_server_url=os.getenv("WORKCELL_MANAGER_URL", "http://localhost:8005")
)

# Create workflow for single image capture
_demo_workflow = WorkflowDefinition(
    name=f"Single Image Capture",
    description=f"Capture a single image from the robot's camera",
    steps=[
        StepDefinition(
            name="Capture Single Image",
            node="DOFBOT_Pro_1",
            action="capture_single_image"
        )
    ]
)

demo_workflow = WorkflowDefinition(
    name=f"Testing new action scanning",
    description=f"Testing the new scanning action",
    steps=[
        StepDefinition(
            name="Test Center Action",
            node="DOFBOT_Pro_1",
            action="scan_for_target",
            args={
                "object_type": "cube", # cube or rectangular_prism
                "color": "blue"
            }
        )
    ]
)

print("Starting single image capture workflow...")
print(f"Taking a picture...")

# Submit and run the image capture workflow
result = workcell_client.submit_workflow(workflow=demo_workflow)

print(f"Image capture workflow completed!")
print(f"Workflow ID: {result.workflow_id}")
print(f"Status: {result.status}")

# Check if the workflow was successful using the correct status attributes
if hasattr(result.status, 'completed') and result.status.completed and not result.status.failed:
    print("✓ Image successfully captured and saved to captures/ directory")
    print("  Check captures/capture_YYYYMMDD_HHMMSS.jpg for the captured image")
elif hasattr(result.status, 'ok') and result.status.ok:
    print("✓ Image successfully captured and saved to captures/ directory")
    print("  Check captures/capture_YYYYMMDD_HHMMSS.jpg for the captured image")
else:
    print("✗ Image capture failed - check node logs for details")
    if hasattr(result.status, 'failed') and result.status.failed:
        print(f"  Workflow failed: {result.status}")
    else:
        print(f"  Unexpected status: {result.status}")
