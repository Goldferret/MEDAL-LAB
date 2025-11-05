#!/usr/bin/env python3
"""
4-Block Permutation Search Experiment.

Systematically tests block arrangements to find target configuration.
Demonstrates MADSci patterns for high-throughput experimentation.
"""

import os
from madsci.client import (
    ExperimentClient, LocationClient, ResourceClient,
    DataClient, WorkcellClient
)
from madsci.common.types.experiment_types import ExperimentDesign, ExperimentStatus

# Import helper functions
from helpers.vision import detect_blocks
from helpers.block_permutations.locations import setup_locations
from helpers.block_permutations.resources import create_block_resources, cleanup_resources
from helpers.block_permutations.algorithms import generate_target, generate_permutations
from helpers.block_permutations.workflows import generate_workflow_for_permutation, create_scan_workflow
from helpers.block_permutations.snapshots import snapshot_configuration

# Get service URLs from environment
EXPERIMENT_URL = os.getenv("EXPERIMENT_SERVER_URL", "http://localhost:8002")
WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
LOCATION_URL = os.getenv("LOCATION_SERVER_URL", "http://localhost:8006")
RESOURCE_URL = os.getenv("RESOURCE_SERVER_URL", "http://localhost:8003")
DATA_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8004")

# Configuration
MAX_ATTEMPTS = 3  # Limited for initial testing


def main():
    print("=== 4-Block Permutation Search Experiment ===\n")
    
    # Track state for error handling
    experiment = None
    experiment_client = None
    location_client = None
    resource_client = None
    location_ids = {}
    resource_ids = {}
    image_datapoint_ids = []  # Track all captured images
    
    try:
        # Step 1: Create experiment design
        print("Step 1: Creating experiment design...")
        experiment_design = ExperimentDesign(
            experiment_name="4-Block Permutation Search",
            experiment_description="Find target block arrangement through systematic testing"
        )
        print("✓ Experiment design created\n")
        
        # Step 2: Start experiment
        print(f"Step 2: Starting experiment...")
        experiment_client = ExperimentClient(EXPERIMENT_URL)
        experiment = experiment_client.start_experiment(
            experiment_design=experiment_design,
            run_name="Permutation Run 1",
            run_description="Initial permutation search run"
        )
        print(f"✓ Experiment started: {experiment.experiment_id}\n")
        
        # Step 3: Initialize clients
        print("Step 3: Initializing MADSci clients...")
        location_client = LocationClient(LOCATION_URL)
        resource_client = ResourceClient(RESOURCE_URL)
        data_client = DataClient(DATA_URL)
        workcell_client = WorkcellClient(WORKCELL_URL)
        print("✓ Clients initialized\n")
        
        # Step 4: Setup locations
        print("Step 4: Setting up block locations...")
        location_ids = setup_locations(location_client)
        print(f"✓ Created {len(location_ids)} locations\n")
        
        # Step 5: Initial scan and resource creation
        print("Step 5: Scanning workspace and creating resources...")
        
        # Robot captures image and stores in Data Manager
        scan_workflow = create_scan_workflow()
        scan_result = workcell_client.start_workflow(scan_workflow)
        
        # Two-level datapoint retrieval (action returns wrapper, wrapper contains actual image ID)
        capture_step = [step for step in scan_result.steps if step.name == "capture"][0]
        wrapper_id = capture_step.result.datapoints.json_result
        image_id = data_client.get_datapoint_value(wrapper_id)
        image_datapoint_ids.append(image_id)  # Track image ID
        
        # Client retrieves image data from Data Manager
        print("  Retrieving image from Data Manager...")
        import tempfile
        from pathlib import Path
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / "scan_image.jpg"
        data_client.save_datapoint_value(image_id, str(image_path))
        
        # Load image with OpenCV
        import cv2
        image_data = cv2.imread(str(image_path))
        
        # Client processes image locally (distributed compute pattern)
        print("  Processing image on client...")
        current_arrangement = detect_blocks(image_data)
        print(f"  Detected blocks: {current_arrangement}")
        
        resource_ids = create_block_resources(
            resource_client, location_client, 
            current_arrangement, location_ids
        )
        print(f"✓ Created {len(resource_ids)} block resources\n")
        
        # Step 6: Generate target and permutations
        print("Step 6: Generating target arrangement...")
        target_arrangement = generate_target(current_arrangement)
        print(f"  Target: {target_arrangement}")
        permutations = generate_permutations(current_arrangement)
        print(f"✓ Generated {len(permutations)} permutations to test\n")
        
        # Step 7: Run experiment loop
        print(f"Step 7: Testing permutations (max {MAX_ATTEMPTS})...")
        match_found = False
        trial_count = 0
        
        for perm in permutations[:MAX_ATTEMPTS]:
            trial_count += 1
            print(f"\n  Trial {trial_count}: Testing {perm}")
            
            # Generate and submit workflow
            workflow_def = generate_workflow_for_permutation(
                perm, current_arrangement, location_ids, trial_count
            )
            result = workcell_client.start_workflow(workflow_def)
            
            # Two-level datapoint retrieval
            capture_step = [step for step in result.steps if step.name == "capture_final"][0]
            wrapper_id = capture_step.result.datapoints.json_result
            image_id = data_client.get_datapoint_value(wrapper_id)
            image_datapoint_ids.append(image_id)  # Track image ID
            
            # Save and load image
            trial_image_path = temp_dir / f"trial_{trial_count}.jpg"
            data_client.save_datapoint_value(image_id, str(trial_image_path))
            image_data = cv2.imread(str(trial_image_path))
            
            # Process image on client
            detected = detect_blocks(image_data)
            
            # Update current arrangement to the intended permutation
            # (Don't trust vision detection for tracking state)
            current_arrangement = perm
            
            # Check if detected matches target (for success detection)
            if detected == target_arrangement:
                match_found = True
                print(f"  ✓ MATCH FOUND! Detected: {detected}")
                break
            else:
                print(f"  ✗ No match. Detected: {detected}, Expected: {perm}")
        
        print(f"\n✓ Experiment loop completed ({trial_count} trials)\n")
        
        # Step 8: Snapshot and cleanup
        print("Step 8: Creating configuration snapshot...")
        snapshot_id = snapshot_configuration(
            data_client, location_client, resource_client,
            location_ids, resource_ids, 
            target_arrangement, current_arrangement, trial_count,
            image_datapoint_ids,  # Pass tracked image IDs
            experiment.experiment_id  # Pass experiment ID for ownership
        )
        print(f"✓ Snapshot created: {snapshot_id}\n")
        
        print("Step 9: Cleaning up resources...")
        cleanup_resources(location_client, resource_client, location_ids, resource_ids)
        print("✓ Cleanup complete\n")
        
        # Step 10: End experiment
        print("Step 10: Ending experiment...")
        final_experiment = experiment_client.end_experiment(experiment.experiment_id)
        print(f"✓ Experiment ended at: {final_experiment.ended_at}\n")
        
        # Summary
        print("=== Experiment Summary ===")
        print(f"Success: {match_found}")
        print(f"Trials: {trial_count}")
        print(f"Target: {target_arrangement}")
        print(f"Final: {current_arrangement}")
        print(f"Snapshot ID: {snapshot_id}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print("\n✅ Experiment completed!")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}\n")
        
        # End experiment with FAILED status if it was started
        if experiment and experiment_client:
            try:
                print("Ending experiment with FAILED status...")
                experiment_client.end_experiment(
                    experiment.experiment_id,
                    status=ExperimentStatus.FAILED
                )
                print("✓ Experiment ended\n")
            except Exception as end_error:
                print(f"⚠ Could not end experiment: {end_error}\n")
        
        # Attempt cleanup if resources were created
        if location_ids and resource_ids and location_client and resource_client:
            try:
                print("Attempting cleanup...")
                cleanup_resources(location_client, resource_client, location_ids, resource_ids)
                print("✓ Cleanup complete\n")
            except Exception as cleanup_error:
                print(f"⚠ Cleanup failed: {cleanup_error}\n")
        
        raise  # Re-raise to show full traceback


if __name__ == "__main__":
    main()
