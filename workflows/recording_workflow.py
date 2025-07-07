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

# Current location of item (angle)
current_location = 90
# Next location of item (angle)
next_location = 90
# Approach method, Retreat/Return method, and Grab orientation (in order)
movement_keys = ["angled_above", "angled_above", "horizontal"]

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
                "annotation": f"Recording Demo: Transfer from {current_location} degrees to {next_location} degrees"
            }
        ),
        StepDefinition(
            name="Execute Transfer",
            node="DOFBOT_Pro_1",
            action="transfer",
            args={
                "locations": [current_location, next_location],
                "movement_keys": movement_keys
            }
        ),
        StepDefinition(
            name="Stop Recording and Camera",
            node="DOFBOT_Pro_1",
            action="stop_recording"
        )
    ]
)

print("Starting recording workflow...")
print(f"Transfer from {current_location} degrees to {next_location} degrees")
print("This will record expert trajectory data for training")

# Submit and run the recording workflow
result = workcell_client.submit_workflow(workflow=demo_workflow)

print(f"Recording workflow completed!")
print(f"Workflow ID: {result.workflow_id}")
print(f"Status: {result.status}")
print("Expert trajectory data has been recorded and saved.")
