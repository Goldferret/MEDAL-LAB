#!/usr/bin/env python3
"""
Simple demo workflow showcasing DOFBOT Pro ROS node movement.

This workflow demonstrates basic robot control by moving the arm through
two poses and returning to home position.
"""

import os
from madsci.client.workcell_client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

# Get workcell manager URL from environment
WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
ROBOT_NODE = "DOFBOT_Pro_1"

# Define two poses (matching arm_test.py - in radians for ROS)
pose_1 = [0.79, 0.79, -1.57, -1.57, 0.0]
pose_2 = [0.3, 0.3, -1.0, -1.0, 0.0]

# Create workflow definition
demo_workflow = WorkflowDefinition(
    name="DOFBOT Pro Demo",
    description="Demonstrate robot movement through two poses and return to home",
    steps=[
        StepDefinition(
            name="Move to Pose 1",
            node=ROBOT_NODE,
            action="move_to_position",
            args={"joint_positions": pose_1}
        ),
        StepDefinition(
            name="Move to Pose 2",
            node=ROBOT_NODE,
            action="move_to_position",
            args={"joint_positions": pose_2}
        ),
        StepDefinition(
            name="Return to Home",
            node=ROBOT_NODE,
            action="home_robot",
            args={}
        )
    ]
)

def main():
    print("=== DOFBOT Pro Demo Workflow ===\n")
    print(f"Connecting to workcell manager at {WORKCELL_URL}...")
    
    # Connect to workcell manager
    workcell_client = WorkcellClient(WORKCELL_URL)
    
    print(f"Robot node: {ROBOT_NODE}")
    print(f"Pose 1: {pose_1}")
    print(f"Pose 2: {pose_2}\n")
    
    # Submit and run the workflow
    print("Submitting workflow...")
    result = workcell_client.start_workflow(demo_workflow)
    
    print(f"\nWorkflow completed!")
    print(f"Workflow ID: {result.workflow_id}")
    print(f"Status: {result.status}")
    
    # Check workflow success
    if hasattr(result.status, 'completed') and result.status.completed:
        print("✅ Demo workflow completed successfully")
    elif hasattr(result.status, 'failed') and result.status.failed:
        print("❌ Demo workflow failed - check node logs for details")
    else:
        print(f"⚠️  Unexpected status: {result.status}")

if __name__ == "__main__":
    main()
