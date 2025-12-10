"""
Configuration snapshot for block combination solver experiment.

Captures complete experiment state including all locations, resources,
experiment results, timing statistics, and image datapoints.
"""
from madsci.common.types.datapoint_types import ValueDataPoint


def snapshot_configuration(
    data_client, 
    location_client, 
    resource_client,
    location_ids, 
    resource_ids, 
    target_arrangement, 
    final_arrangement,
    total_attempts,
    combinations_tried,
    solution_found,
    image_datapoint_ids,
    attempt_times,
    experiment_id
):
    """
    Create snapshot of experiment configuration and results.
    
    Args:
        data_client: DataClient instance
        location_client: LocationClient instance
        resource_client: ResourceClient instance
        location_ids: Mapping of position names to location IDs
        resource_ids: Mapping of resource names to resource IDs
        target_arrangement: Target arrangement that was searched for
        final_arrangement: Final arrangement at end of experiment
        total_attempts: Total number of attempts made
        combinations_tried: Total unique combinations tested
        solution_found: Whether solution was found
        image_datapoint_ids: List of image datapoint IDs captured
        attempt_times: List of time taken for each attempt
        experiment_id: Experiment ID for ownership tracking
        
    Returns:
        str: Datapoint ID of snapshot
    """
    # Calculate timing statistics
    total_time = sum(attempt_times) if attempt_times else 0
    avg_time = total_time / len(attempt_times) if attempt_times else 0
    min_time = min(attempt_times) if attempt_times else 0
    max_time = max(attempt_times) if attempt_times else 0
    
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
            "target_arrangement": target_arrangement,
            "final_arrangement": final_arrangement,
            "total_attempts": total_attempts,
            "unique_combinations_tried": combinations_tried,
            "solution_found": solution_found,
            "image_datapoint_ids": image_datapoint_ids
        },
        "timing_statistics": {
            "total_time_seconds": total_time,
            "average_time_per_attempt": avg_time,
            "fastest_attempt": min_time,
            "slowest_attempt": max_time,
            "attempt_times": attempt_times
        }
    }
    
    snapshot = ValueDataPoint(
        label="experiment_configuration",
        value=config,
        ownership_info={"experiment_id": experiment_id}
    )
    
    submitted = data_client.submit_datapoint(snapshot)
    return submitted.datapoint_id

