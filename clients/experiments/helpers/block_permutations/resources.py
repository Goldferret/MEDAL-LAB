"""Resource creation and cleanup for block permutation experiment."""
"""
Resource creation and cleanup for block permutation experiment.

Dynamically creates colored block resources based on vision detection
and manages their lifecycle (creation, attachment to locations, deletion).
"""
from datetime import datetime
from madsci.common.types.resource_types import Asset


def create_block_resources(resource_client, location_client, colors, location_ids):
    """
    Dynamically create block resources based on detection.
    
    Args:
        resource_client: ResourceClient instance
        location_client: LocationClient instance
        colors: List of detected colors
        location_ids: Mapping of position names to location IDs
        
    Returns:
        dict: Mapping of resource names to resource IDs
    """
    resource_ids = {}
    
    for i, color in enumerate(colors):
        if color is None:
            continue
            
        block = Asset(
            resource_name=f"{color}_block",
            resource_class="colored_block",
            attributes={
                "color": color,
                "initial_position": i,
                "detected_at": datetime.now().isoformat()
            }
        )
        created = resource_client.add_resource(block)
        
        location_id = location_ids[f"pos_{i}"]
        location_client.attach_resource(
            location_id=location_id,
            resource_id=created.resource_id
        )
        
        resource_ids[f"{color}_block"] = created.resource_id
        print(f"  ✓ Created resource: {color}_block at position {i}")
    
    return resource_ids


def cleanup_resources(location_client, resource_client, location_ids, resource_ids):
    """
    Delete all created resources and locations.
    
    Args:
        location_client: LocationClient instance
        resource_client: ResourceClient instance
        location_ids: Mapping of position names to location IDs
        resource_ids: Mapping of resource names to resource IDs
    """
    for name, res_id in resource_ids.items():
        resource_client.remove_resource(res_id)
        print(f"  ✓ Deleted resource: {name}")
    
    for name, loc_id in location_ids.items():
        location_client.delete_location(loc_id)
        print(f"  ✓ Deleted location: {name}")
