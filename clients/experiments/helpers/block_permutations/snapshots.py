"""Configuration snapshot for block permutation experiment."""
from madsci.common.types.datapoint_types import ValueDataPoint


def snapshot_configuration(data_client, location_client, resource_client, 
                          location_ids, resource_ids, target, final, trials, image_ids, experiment_id):
    """
    Create snapshot of experiment configuration.
    
    Args:
        data_client: DataClient instance
        location_client: LocationClient instance
        resource_client: ResourceClient instance
        location_ids: Mapping of position names to location IDs
        resource_ids: Mapping of resource names to resource IDs
        target: Target arrangement
        final: Final arrangement
        trials: Total number of trials
        image_ids: List of image datapoint IDs captured during experiment
        experiment_id: Experiment ID for ownership tracking
        
    Returns:
        str: Datapoint ID of snapshot
    """
    config = {
        "locations": {
            name: location_client.get_location(loc_id).model_dump()
            for name, loc_id in location_ids.items()
        },
        "resources": {
            name: resource_client.get_resource(res_id).model_dump()
            for name, res_id in resource_ids.items()
        },
        "experiment_data": {
            "target_arrangement": target,
            "final_arrangement": final,
            "total_trials": trials,
            "image_datapoint_ids": image_ids
        }
    }
    
    snapshot = ValueDataPoint(
        label="experiment_configuration",
        value=config,
        ownership_info={"experiment_id": experiment_id}
    )
    
    submitted = data_client.submit_datapoint(snapshot)
    return submitted.datapoint_id
