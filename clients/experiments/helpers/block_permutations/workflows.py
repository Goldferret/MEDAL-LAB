"""Dynamic workflow generation for block permutation experiment."""
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition
from .algorithms import calculate_swaps

# Robot node name
ROBOT_NODE = "DOFBOT_Pro_1"


def generate_workflow_for_permutation(target_perm, current_arrangement, location_ids, trial_count):
    """
    Generate dynamic workflow to reach target permutation.
    
    Args:
        target_perm: Target arrangement to reach
        current_arrangement: Current block arrangement
        location_ids: Mapping of position names to location IDs
        trial_count: Current trial number
        
    Returns:
        WorkflowDefinition with swap steps + capture step
        
    Notes:
        - Skips redundant swaps where pos_a == pos_b
        - If no swaps needed, workflow only contains capture step
        - Each swap uses temp location for three-way exchange
    """
    swaps = calculate_swaps(current_arrangement, target_perm)
    steps = []
    
    # Add swap steps (skip redundant same-position swaps)
    for i, (pos_a, pos_b) in enumerate(swaps):
        if pos_a == pos_b:
            print(f"  ⚠ Skipping redundant swap: pos_{pos_a} to pos_{pos_b}")
            continue
            
        steps.append(StepDefinition(
            name=f"swap_{i}",
            node=ROBOT_NODE,
            action="swap_blocks",
            args={
                "location_a_id": location_ids[f"pos_{pos_a}"],
                "location_b_id": location_ids[f"pos_{pos_b}"],
                "temp_location_id": location_ids["temp"]
            }
        ))
    
    # Move to scan position before capturing
    scan_position = [-0.07, 0.2, -1.5, -1.5, 0.0]
    steps.append(StepDefinition(
        name="move_to_scan",
        node=ROBOT_NODE,
        action="move_to_position",
        args={"joint_positions": scan_position}
    ))
    
    # Add capture step
    steps.append(StepDefinition(
        name="capture_final",
        node=ROBOT_NODE,
        action="capture_camera_image",
        args={},
        data_labels={"image": "arrangement_image"}
    ))
    
    return WorkflowDefinition(
        name=f"test_permutation_{trial_count}",
        description=f"Test permutation: {target_perm}",
        steps=steps
    )


def create_scan_workflow():
    """
    Create workflow to move robot to scan position and capture image.
    
    Returns:
        WorkflowDefinition for initial scanning
        
    Note:
        Uses scan position from BlockProblem4Blocks.py: [-0.07, 0.2, -1.5, -1.5, 0.0]
        This positions the camera looking down at the block workspace.
    """
    # Scan position from BlockProblem4Blocks.py
    scan_position = [-0.07, 0.2, -1.5, -1.5, 0.0]
    
    return WorkflowDefinition(
        name="Initial Scan",
        description="Move to scan position and capture image",
        steps=[
            StepDefinition(
                name="move_scan",
                node=ROBOT_NODE,
                action="move_to_position",
                args={"joint_positions": scan_position}
            ),
            StepDefinition(
                name="capture",
                node=ROBOT_NODE,
                action="capture_camera_image",
                args={},
                data_labels={"image": "initial_scan"}
            )
        ]
    )
