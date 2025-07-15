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

# Target joint angles: [servo1, servo2, servo3, servo4, servo5]
target_angles = [135, 110, 5, 0, 90]

# Create workflow for moving robot to specific position
move_position_workflow = WorkflowDefinition(
    name=f"Move to Position {target_angles}",
    description=f"Move robot to joint angles: Servo1={target_angles[0]}°, Servo2={target_angles[1]}°, Servo3={target_angles[2]}°, Servo4={target_angles[3]}°, Servo5={target_angles[4]}°",
    steps=[
        StepDefinition(
            name="Move All Joints to Target Position",
            node="DOFBOT_Pro_1",
            action="move_all_joints",
            args={
                "joint_angles": target_angles
            }
        )
    ]
)

print("Starting robot movement workflow...")
print(f"Moving robot to position: {target_angles}")
print(f"  Servo 1 (Base): {target_angles[0]}°")
print(f"  Servo 2 (Shoulder): {target_angles[1]}°") 
print(f"  Servo 3 (Elbow): {target_angles[2]}°")
print(f"  Servo 4 (Wrist): {target_angles[3]}°")
print(f"  Servo 5 (Gripper): {target_angles[4]}°")

# Submit and run the movement workflow
result = workcell_client.submit_workflow(workflow=move_position_workflow)

print(f"Robot movement workflow completed!")
print(f"Workflow ID: {result.workflow_id}")
print(f"Status: {result.status}")

# Check if the workflow was successful using the correct status attributes
if hasattr(result.status, 'completed') and result.status.completed and not result.status.failed:
    print("✓ Robot successfully moved to target position")
    print(f"  Final position: {target_angles}")
elif hasattr(result.status, 'ok') and result.status.ok:
    print("✓ Robot successfully moved to target position")
    print(f"  Final position: {target_angles}")
else:
    print("✗ Robot movement failed - check node logs for details")
    if hasattr(result.status, 'failed') and result.status.failed:
        print(f"  Workflow failed: {result.status}")
    else:
        print(f"  Unexpected status: {result.status}")
