"""
Dynamic workflow generation for block combination solver experiment.

Generates workflows for:
- Initial scanning
- Moving blocks between positions
- Testing each combination
"""
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition
from .algorithms import generate_move_sequence

# Robot node name
ROBOT_NODE = "DOFBOT_Pro_1"

# Scan position from BlockPositions.py
SCAN_POSITION = [-0.07, 0.2, -1.5, -1.5, 0.0]


def create_scan_workflow():
    """
    Create workflow to move robot to scan position and capture image.
    
    Returns:
        WorkflowDefinition for initial scanning
    """
    return WorkflowDefinition(
        name="Initial Scan",
        description="Move to scan position and capture image",
        steps=[
            StepDefinition(
                name="move_scan",
                node=ROBOT_NODE,
                action="move_to_position",
                args={"joint_positions": SCAN_POSITION}
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


def create_move_workflow(from_pos, to_pos, location_ids, move_name="move"):
    """
    Create workflow to move single block between positions.
    
    Args:
        from_pos: Source position (index 0-5 or 'temp')
        to_pos: Destination position (index 0-5 or 'temp')
        location_ids: Mapping of position names to location IDs
        move_name: Name for this move step
        
    Returns:
        WorkflowDefinition with pick and place steps
    """
    # Convert position indices to location names
    if from_pos == 'temp':
        from_location = location_ids['temp']
        from_name = 'temp'
    else:
        from_location = location_ids[f'pos_{from_pos}']
        from_name = f'pos_{from_pos}'
    
    if to_pos == 'temp':
        to_location = location_ids['temp']
        to_name = 'temp'
    else:
        to_location = location_ids[f'pos_{to_pos}']
        to_name = f'pos_{to_pos}'
    
    return WorkflowDefinition(
        name=f"Move {from_name} to {to_name}",
        description=f"Move block from {from_name} to {to_name}",
        steps=[
            StepDefinition(
                name=f"{move_name}_pick",
                node=ROBOT_NODE,
                action="pick_from_location",
                args={"location_id": from_location}
            ),
            StepDefinition(
                name=f"{move_name}_place",
                node=ROBOT_NODE,
                action="place_at_location",
                args={"location_id": to_location}
            )
        ]
    )


def generate_rearrangement_workflow(
    current_arrangement, 
    target_arrangement, 
    location_ids, 
    attempt_count
):
    """
    Generate workflow to rearrange blocks to target arrangement.
    
    Uses cycle-based algorithm to minimize number of moves.
    Always ends with scan + capture to verify arrangement.
    
    Args:
        current_arrangement: Current block arrangement (list of colors)
        target_arrangement: Target arrangement to reach
        location_ids: Mapping of position names to location IDs
        attempt_count: Current attempt number (for naming)
        
    Returns:
        WorkflowDefinition with all move steps + final scan + capture
    """
    # Generate optimal move sequence
    move_sequence = generate_move_sequence(current_arrangement, target_arrangement)
    
    steps = []
    
    # Add move steps
    for move_num, (from_pos, to_pos) in enumerate(move_sequence):
        # Convert position to location ID
        if from_pos == 'temp':
            from_location = location_ids['temp']
        else:
            from_location = location_ids[f'pos_{from_pos}']
        
        if to_pos == 'temp':
            to_location = location_ids['temp']
        else:
            to_location = location_ids[f'pos_{to_pos}']
        
        # Add pick step
        steps.append(StepDefinition(
            name=f"move_{move_num}_pick",
            node=ROBOT_NODE,
            action="pick_from_location",
            args={"location_id": from_location}
        ))
        
        # Add place step
        steps.append(StepDefinition(
            name=f"move_{move_num}_place",
            node=ROBOT_NODE,
            action="place_at_location",
            args={"location_id": to_location}
        ))
    
    # Move to scan position before capturing
    steps.append(StepDefinition(
        name="move_to_scan",
        node=ROBOT_NODE,
        action="move_to_position",
        args={"joint_positions": SCAN_POSITION}
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
        name=f"test_combination_{attempt_count}",
        description=f"Attempt {attempt_count}: Test combination {target_arrangement}",
        steps=steps
    )

