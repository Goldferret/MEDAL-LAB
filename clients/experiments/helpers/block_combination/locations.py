"""
Location setup and management for block combination solver experiment.

Defines 6 block positions (pos_0 through pos_5) plus temp position with joint angle
representations for the DOFBOT Pro robot. Uses positions from BlockPositions.py.
"""
from madsci.common.types.location_types import Location
from madsci.common.utils import new_ulid_str

# Robot node name
ROBOT_NODE = "DOFBOT_Pro_1"

# Position configurations from BlockPositions.py
# 6 positions in 2x3 grid layout plus temporary holding position
POSITION_CONFIGS = {
    # Top row (positions 0-2)
    "pos_0": {  # Top row, right
        "normal": [0.3, -0.8, -0.45, -1.35, 0.35],
        "raised": [0.3, -0.5, -0.45, -1.35, 0.35],
        "lowered": [0.3, -1.0, -0.25, -1.3, 0.35]
    },
    "pos_1": {  # Top row, center
        "normal": [0.0, -0.8, -0.45, -1.4, 0.0],
        "raised": [0.0, -0.5, -0.45, -1.4, 0.0],
        "lowered": [0.0, -0.925, -0.35, -1.35, 0.0]
    },
    "pos_2": {  # Top row, left
        "normal": [-0.3, -0.8, -0.45, -1.35, -0.35],
        "raised": [-0.3, -0.5, -0.45, -1.35, -0.35],
        "lowered": [-0.3, -1.0, -0.25, -1.37, -0.35]
    },
    
    # Bottom row (positions 3-5)
    "pos_3": {  # Bottom row, right
        "normal": [0.4, -0.4, -1.15, -1.3, 0.5],
        "raised": [0.4, 0.0, -1.15, -1.3, 0.5],
        "lowered": [0.4, -0.5, -1.05, -1.2, 0.5]
    },
    "pos_4": {  # Bottom row, center
        "normal": [0.0, -0.3, -1.2, -1.4, 0.0],
        "raised": [0.0, 0.1, -1.2, -1.4, 0.0],
        "lowered": [0.0, -0.4, -1.2, -1.2, 0.0]
    },
    "pos_5": {  # Bottom row, left
        "normal": [-0.5, -0.3, -1.2, -1.3, -0.5],
        "raised": [-0.5, 0.1, -1.2, -1.3, -0.5],
        "lowered": [-0.35, -0.4, -1.2, -1.15, -0.4]
    },
    
    # Temporary holding position
    "temp": {
        "normal": [-0.8, -0.1, -1.2, -1.2, 0.0],
        "raised": [-0.8, 0.3, -1.2, -1.2, 0.0],
        "lowered": [-0.8, -0.4, -1.2, -1.2, 0.0]
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
            description=f"Block position {pos_name} (2x3 grid)"
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

