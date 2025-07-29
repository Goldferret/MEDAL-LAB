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

# Create workflow for this demonstration
demo_workflow = WorkflowDefinition(
    name=f"Recording Demo",
    description=f"Collect expert trajectory data from Transfer action",
    steps=[
        StepDefinition(
            name="Start Recording with Camera",
            node="DOFBOT_Pro_1",
            action="start_recording",
            args={
                "annotation": f"Recording Demo: Go to \"Home Position\" and then attempt a scan_for_target action."
            }
        ),
        StepDefinition(
            name="Move to Grabber Position",
            node="DOFBOT_Pro_1",
            action="grabber_position"
        ),
        StepDefinition(
            name="Scan for Target",
            node="DOFBOT_Pro_1",
            action="scan_for_target",
            args={
                "object_type": "cube", # cube or rectangular_prism
                "color": "yellow"
            }
        ),
        StepDefinition(
            name="Stop Recording with Camera",
            node="DOFBOT_Pro_1",
            action="stop_recording"
        )
    ]
)

print("Starting recording workflow...")

# Submit and run the recording workflow
result = workcell_client.submit_workflow(workflow=demo_workflow)

print(f"Recording workflow completed!")
print(f"Workflow ID: {result.workflow_id}")
print(f"Status: {result.status}")
print("Experiment data has been recorded and saved.")
