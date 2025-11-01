#!/usr/bin/env python3
"""
Demo experiment showcasing MADSci Experiment Manager integration.

This experiment demonstrates:
1. Creating and starting an experiment
2. Running multiple workflows as part of the experiment
3. Querying experiment status
4. Completing the experiment
"""

import os
from madsci.client import ExperimentClient, WorkcellClient
from madsci.common.types.experiment_types import ExperimentDesign
from madsci.common.types.auth_types import OwnershipInfo
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

# Get service URLs from environment
EXPERIMENT_URL = os.getenv("EXPERIMENT_SERVER_URL", "http://localhost:8002")
WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
ROBOT_NODE = "DOFBOT_Pro_1"

# Define robot poses for the experiment
home_pose = [0.0, 0.0, 0.0, 0.0, 0.0]
pose_1 = [0.79, 0.79, -1.57, -1.57, 0.0]
pose_2 = [0.3, 0.3, -1.0, -1.0, 0.0]
pose_3 = [0.5, 0.5, -1.2, -1.2, 0.0]


def create_movement_workflow(name: str, description: str, target_pose: list) -> WorkflowDefinition:
    """Create a simple movement workflow."""
    return WorkflowDefinition(
        name=name,
        description=description,
        steps=[
            StepDefinition(
                name=f"Move to {name}",
                node=ROBOT_NODE,
                action="move_to_position",
                args={"joint_positions": target_pose}
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
    print("=== DOFBOT Pro Experiment Demo ===\n")
    
    # Step 1: Create experiment design
    print("Step 1: Creating experiment design...")
    experiment_design = ExperimentDesign(
        experiment_name="DOFBOT Movement Study",
        experiment_description="Systematic study of robot movement through multiple poses",
        resource_conditions=[],  # No pre-conditions for this simple demo
        ownership_info=OwnershipInfo()  # Minimal ownership info
    )
    print("✓ Experiment design created\n")
    
    # Step 2: Connect to Experiment Manager and start experiment
    print(f"Step 2: Connecting to Experiment Manager at {EXPERIMENT_URL}...")
    experiment_client = ExperimentClient(EXPERIMENT_URL)
    
    print("Starting experiment run...")
    experiment = experiment_client.start_experiment(
        experiment_design=experiment_design,
        run_name="Demo Run 1",
        run_description="First demonstration run of movement study"
    )
    print(f"✓ Experiment started: {experiment.experiment_id}\n")
    
    # Step 3: Connect to Workcell Manager
    print(f"Step 3: Connecting to Workcell Manager at {WORKCELL_URL}...")
    workcell_client = WorkcellClient(WORKCELL_URL)
    print("✓ Connected to workcell\n")
    
    # Step 4: Run multiple workflows as part of the experiment
    print("Step 4: Executing experimental workflows...")
    
    workflows = [
        ("Pose 1 Test", "Test movement to pose 1", pose_1),
        ("Pose 2 Test", "Test movement to pose 2", pose_2),
        ("Pose 3 Test", "Test movement to pose 3", pose_3),
    ]
    
    workflow_results = []
    for i, (name, desc, pose) in enumerate(workflows, 1):
        print(f"\n  Workflow {i}/3: {name}")
        print(f"  Target pose: {pose}")
        
        workflow_def = create_movement_workflow(name, desc, pose)
        result = workcell_client.start_workflow(workflow_def)
        workflow_results.append(result)
        
        if result.status.completed:
            print(f"  ✓ Completed (ID: {result.workflow_id})")
        elif result.status.failed:
            print(f"  ✗ Failed (ID: {result.workflow_id})")
        else:
            print(f"  ⚠ Unexpected status: {result.status.description}")
    
    print("\n✓ All workflows executed\n")
    
    # Step 5: Query experiment status
    print("Step 5: Querying experiment status...")
    updated_experiment = experiment_client.get_experiment(experiment.experiment_id)
    print(f"  Experiment ID: {updated_experiment.experiment_id}")
    print(f"  Run Name: {updated_experiment.run_name}")
    print(f"  Status: {updated_experiment.status}")
    print(f"  Started: {updated_experiment.started_at}")
    print("✓ Experiment status retrieved\n")
    
    # Step 6: End the experiment
    print("Step 6: Ending experiment...")
    final_experiment = experiment_client.end_experiment(experiment.experiment_id)
    print(f"✓ Experiment ended at: {final_experiment.ended_at}\n")
    
    # Summary
    print("=== Experiment Summary ===")
    print(f"Experiment: {experiment_design.experiment_name}")
    print(f"Run: {experiment.run_name}")
    print(f"Workflows executed: {len(workflow_results)}")
    successful = sum(1 for r in workflow_results if r.status.completed)
    failed = sum(1 for r in workflow_results if r.status.failed)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nExperiment ID: {experiment.experiment_id}")
    print("\n✅ Experiment demo completed successfully!")


if __name__ == "__main__":
    main()
