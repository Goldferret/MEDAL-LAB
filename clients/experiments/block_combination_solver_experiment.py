#!/usr/bin/env python3
"""
6-Block Combination Search Experiment (MADSci-Compatible Port)

Ported from BlockCombinationSolver.py to MADSci architecture.

Problem:
  Given 6 blocks of 4 colors, search for a randomly generated target combination.
  
Solution:
  - Generates random target pattern after initial scan
  - Shuffles blocks through random combinations
  - Uses efficient cycle-based block movement algorithm
  - Tracks tried combinations to avoid repeats
  - Runs autonomously until match is found or maximum attempts reached

Key Changes from Original:
  - Robot control via MADSci workflows instead of direct ROS calls
  - Vision processing on client (experiment) instead of robot node
  - Location-based positioning via Location Manager
  - Resource tracking via Resource Manager
  - All experiment data stored in Data Manager
"""

import os
import time
from madsci.client import (
    ExperimentClient, LocationClient, ResourceClient,
    DataClient, WorkcellClient
)
from madsci.common.types.experiment_types import ExperimentDesign, ExperimentStatus

# Import helper functions
from helpers.vision import detect_blocks
from helpers.block_combination.locations import setup_locations
from helpers.block_combination.resources import create_block_resources, cleanup_resources
from helpers.block_combination.algorithms import (
    generate_target, 
    generate_random_combination,
    mark_combination_tried,
    is_combination_tried
)
from helpers.block_combination.workflows import (
    create_scan_workflow,
    generate_rearrangement_workflow
)
from helpers.block_combination.snapshots import snapshot_configuration

# Get service URLs from environment
EXPERIMENT_URL = os.getenv("EXPERIMENT_SERVER_URL", "http://localhost:8002")
WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
LOCATION_URL = os.getenv("LOCATION_SERVER_URL", "http://localhost:8006")
RESOURCE_URL = os.getenv("RESOURCE_SERVER_URL", "http://localhost:8003")
DATA_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8004")

# Configuration
MAX_ATTEMPTS = 50  # Maximum attempts before giving up
NUM_BLOCKS = 6     # 6 blocks in 2x3 grid


def main():
    print("="*60)
    print("6-Block Combination Search Experiment")
    print("="*60)
    print()
    
    # Track state for error handling
    experiment = None
    experiment_client = None
    location_client = None
    resource_client = None
    location_ids = {}
    resource_ids = {}
    image_datapoint_ids = []
    attempt_times = []
    
    # Experiment state
    current_arrangement = None
    target_arrangement = None
    tried_combinations = set()
    solution_found = False
    attempt_count = 0
    
    try:
        # =====================================================================
        # STEP 1: Create Experiment
        # =====================================================================
        print("Step 1: Creating experiment design...")
        experiment_design = ExperimentDesign(
            experiment_name="6-Block Combination Search",
            experiment_description="Autonomous search for target block arrangement through random permutations"
        )
        print("✓ Experiment design created\n")
        
        # =====================================================================
        # STEP 2: Start Experiment
        # =====================================================================
        print("Step 2: Starting experiment...")
        experiment_client = ExperimentClient(EXPERIMENT_URL)
        experiment = experiment_client.start_experiment(
            experiment_design=experiment_design,
            run_name="Combination Search Run 1",
            run_description="Autonomous combination search with cycle-based movement"
        )
        print(f"✓ Experiment started: {experiment.experiment_id}\n")
        
        # Start timing
        experiment_start_time = time.time()
        
        # =====================================================================
        # STEP 3: Initialize MADSci Clients
        # =====================================================================
        print("Step 3: Initializing MADSci clients...")
        location_client = LocationClient(LOCATION_URL)
        resource_client = ResourceClient(RESOURCE_URL)
        data_client = DataClient(DATA_URL)
        workcell_client = WorkcellClient(WORKCELL_URL)
        print("✓ Clients initialized\n")
        
        # =====================================================================
        # STEP 4: Setup Locations (6 positions + temp)
        # =====================================================================
        print("Step 4: Setting up block locations...")
        location_ids = setup_locations(location_client)
        print(f"✓ Created {len(location_ids)} locations\n")
        
        # =====================================================================
        # STEP 5: Initial Scan and Resource Creation
        # =====================================================================
        print("Step 5: Scanning workspace and creating resources...")
        
        # Robot captures image via workflow
        scan_workflow = create_scan_workflow()
        scan_result = workcell_client.start_workflow(scan_workflow)
        
        # Two-level datapoint retrieval
        capture_step = [step for step in scan_result.steps if step.name == "capture"][0]
        wrapper_id = capture_step.result.datapoints.json_result
        image_id = data_client.get_datapoint_value(wrapper_id)
        image_datapoint_ids.append(image_id)
        
        # Client retrieves and processes image
        print("  Retrieving image from Data Manager...")
        import tempfile
        from pathlib import Path
        import cv2
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / "scan_image.jpg"
        data_client.save_datapoint_value(image_id, str(image_path))
        
        # Load and process image
        image_data = cv2.imread(str(image_path))
        print("  Processing image on client...")
        current_arrangement = detect_blocks(image_data)
        
        # Ensure we have exactly 6 blocks
        if len(current_arrangement) < NUM_BLOCKS:
            current_arrangement.extend([None] * (NUM_BLOCKS - len(current_arrangement)))
        current_arrangement = current_arrangement[:NUM_BLOCKS]
        
        print(f"  Detected blocks: {current_arrangement}")
        
        # Validate detection
        if all(c is None for c in current_arrangement):
            raise ValueError("No blocks detected in initial scan")
        
        # Create resources based on detected blocks
        resource_ids = create_block_resources(
            resource_client, location_client,
            current_arrangement, location_ids
        )
        print(f"✓ Created {len(resource_ids)} block resources\n")
        
        # =====================================================================
        # STEP 6: Generate Target Pattern
        # =====================================================================
        print("Step 6: Generating target arrangement...")
        target_arrangement = generate_target(current_arrangement)
        print(f"  Target: {target_arrangement}")
        print("  (This is the 'hidden' pattern the experiment will search for)\n")
        
        # =====================================================================
        # STEP 7: Check Initial Configuration
        # =====================================================================
        print("Step 7: Checking initial configuration...")
        attempt_count = 1
        mark_combination_tried(current_arrangement, tried_combinations)
        
        print(f"\n{'='*60}")
        print(f"Attempt #{attempt_count}")
        print(f"Testing: {current_arrangement}")
        print(f"{'='*60}")
        
        attempt_start_time = time.time()
        
        if current_arrangement == target_arrangement:
            attempt_time = time.time() - attempt_start_time
            attempt_times.append(attempt_time)
            solution_found = True
            print("✓ MATCH FOUND on initial scan!")
            print(f"  Attempt time: {attempt_time:.2f} seconds")
        else:
            attempt_time = time.time() - attempt_start_time
            attempt_times.append(attempt_time)
            print(f"✗ No match. Current: {current_arrangement}, Target: {target_arrangement}")
            print(f"  Attempt time: {attempt_time:.2f} seconds")
        
        # =====================================================================
        # STEP 8: Search Loop
        # =====================================================================
        if not solution_found:
            print(f"\nStep 8: Searching for matching combination (max {MAX_ATTEMPTS} attempts)...")
        
        while not solution_found and attempt_count < MAX_ATTEMPTS:
            attempt_count += 1
            attempt_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Attempt #{attempt_count}")
            print(f"Combinations tried so far: {len(tried_combinations)}")
            print(f"{'='*60}")
            
            # Generate new untried combination
            new_combination = generate_random_combination(
                current_arrangement, 
                tried_combinations
            )
            
            if new_combination is None:
                print("⚠ Warning: No more unique combinations to try")
                break
            
            mark_combination_tried(new_combination, tried_combinations)
            print(f"Testing: {new_combination}")
            
            # Generate and execute workflow to reach new combination
            workflow_def = generate_rearrangement_workflow(
                current_arrangement,
                new_combination,
                location_ids,
                attempt_count
            )
            
            print(f"  Executing {len(workflow_def.steps)//2 - 1} block moves...")
            result = workcell_client.start_workflow(workflow_def)
            
            # Two-level datapoint retrieval for verification image
            capture_step = [step for step in result.steps if step.name == "capture_final"][0]
            wrapper_id = capture_step.result.datapoints.json_result
            image_id = data_client.get_datapoint_value(wrapper_id)
            image_datapoint_ids.append(image_id)
            
            # Save and load image for verification
            trial_image_path = temp_dir / f"trial_{attempt_count}.jpg"
            data_client.save_datapoint_value(image_id, str(trial_image_path))
            image_data = cv2.imread(str(trial_image_path))
            
            # Detect blocks in verification image
            detected = detect_blocks(image_data)
            if len(detected) < NUM_BLOCKS:
                detected.extend([None] * (NUM_BLOCKS - len(detected)))
            detected = detected[:NUM_BLOCKS]
            
            # Update current arrangement (trust workflow execution)
            current_arrangement = new_combination
            
            # Check if we found the solution
            if detected == target_arrangement:
                solution_found = True
                print("✓ MATCH FOUND!")
                print(f"  Detected: {detected}")
                print(f"  Target:   {target_arrangement}")
            else:
                print(f"✗ No match.")
                print(f"  Detected: {detected}")
                print(f"  Expected: {new_combination}")
                print(f"  (Trusting workflow - state updated to: {current_arrangement})")
            
            attempt_time = time.time() - attempt_start_time
            attempt_times.append(attempt_time)
            print(f"  Attempt time: {attempt_time:.2f} seconds")
        
        # =====================================================================
        # STEP 9: Calculate Statistics
        # =====================================================================
        experiment_end_time = time.time()
        total_time = experiment_end_time - experiment_start_time
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Solution found: {solution_found}")
        print(f"Total attempts: {attempt_count}")
        print(f"Unique combinations tried: {len(tried_combinations)}")
        print(f"Target arrangement: {target_arrangement}")
        print(f"Final arrangement: {current_arrangement}")
        print()
        print("TIMING STATISTICS")
        print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        if attempt_times:
            print(f"  Average time per attempt: {sum(attempt_times)/len(attempt_times):.2f} seconds")
            print(f"  Fastest attempt: {min(attempt_times):.2f} seconds")
            print(f"  Slowest attempt: {max(attempt_times):.2f} seconds")
        print(f"{'='*60}\n")
        
        # =====================================================================
        # STEP 10: Snapshot Configuration
        # =====================================================================
        print("Step 10: Creating configuration snapshot...")
        snapshot_id = snapshot_configuration(
            data_client, location_client, resource_client,
            location_ids, resource_ids,
            target_arrangement, current_arrangement,
            attempt_count, len(tried_combinations),
            solution_found, image_datapoint_ids,
            attempt_times, experiment.experiment_id
        )
        print(f"✓ Snapshot created: {snapshot_id}\n")
        
        # =====================================================================
        # STEP 11: Cleanup Resources
        # =====================================================================
        print("Step 11: Cleaning up resources...")
        cleanup_resources(location_client, resource_client, location_ids, resource_ids)
        print("✓ Cleanup complete\n")
        
        # =====================================================================
        # STEP 12: End Experiment
        # =====================================================================
        print("Step 12: Ending experiment...")
        final_experiment = experiment_client.end_experiment(experiment.experiment_id)
        print(f"✓ Experiment ended at: {final_experiment.ended_at}\n")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Success: {solution_found}")
        print(f"Attempts: {attempt_count}/{MAX_ATTEMPTS}")
        print(f"Target: {target_arrangement}")
        print(f"Final: {current_arrangement}")
        print(f"Snapshot ID: {snapshot_id}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Total time: {total_time:.2f} seconds")
        print("="*60)
        print("\n✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}\n")
        
        # End experiment with FAILED status
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

