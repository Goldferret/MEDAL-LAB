"""Location setup and management for block permutation experiment."""
"""
Location setup and management for block permutation experiment.

Defines 5 block positions (pos_0 through pos_3 + temp) with joint angle
representations for the DOFBOT Pro robot.
"""
from madsci.common.types.location_types import Location
from madsci.common.utils import new_ulid_str

# Robot node name
ROBOT_NODE = "DOFBOT_Pro_1"

# Position configurations from BlockProblem4Blocks.py
# Note: Lowered positions adjusted to -1.0 for better block reach
POSITION_CONFIGS = {
    "pos_0": {
        "normal": [0.3, -0.8, -0.45, -1.35, 0.25],
        "raised": [0.3, -0.5, -0.45, -1.35, 0.25],
        "lowered": [0.3, -1.0, -0.45, -1.35, 0.25]
    },
    "pos_1": {
        "normal": [0.0, -0.8, -0.45, -1.4, 0.0],
        "raised": [0.0, -0.5, -0.45, -1.4, 0.0],
        "lowered": [0.0, -1.0, -0.45, -1.4, 0.0]
    },
    "pos_2": {
        "normal": [-0.3, -0.8, -0.45, -1.35, -0.25],
        "raised": [-0.3, -0.5, -0.45, -1.35, -0.25],
        "lowered": [-0.3, -1.0, -0.45, -1.35, -0.25]
    },
    "pos_3": {
        "normal": [0.4, -0.4, -1.15, -1.3, 0.4],
        "raised": [0.4, -0.3, -1.15, -1.3, 0.4],
        "lowered": [0.4, -0.45, -1.15, -1.3, 0.4]
    },
    "temp": {
        "normal": [0.65, -0.8, -0.45, -1.35, 0.25],
        "raised": [0.65, -0.5, -0.45, -1.35, 0.25],
        "lowered": [0.65, -0.925, -0.45, -1.35, 0.25]
    }
}


def setup_locations(location_client):
    """
    Create locations for block positions.
    
    Args:
        location_client: LocationClient instance
        
    Returns:
        dict: Mapping of position names to location IDs
    """
    location_ids = {}
    
    for pos_name, representations in POSITION_CONFIGS.items():
        location = Location(
            location_id=new_ulid_str(),
            name=f"block_{pos_name}",
            description=f"Block position {pos_name}"
        )
        created = location_client.add_location(location)
        
        location_client.set_representations(
            location_id=location.location_id,
            node_name=ROBOT_NODE,
            representation=representations
        )
        
        location_ids[pos_name] = location.location_id
        print(f"  ✓ Created location: {pos_name}")
    
    return location_ids
