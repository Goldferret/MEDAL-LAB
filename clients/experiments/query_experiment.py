#!/usr/bin/env python3
"""
Query and view experiment results.

Usage:
    python query_experiment.py                    # List last 3 experiments
    python query_experiment.py <experiment_id>    # View details and download images
"""

import sys
import os
import requests
from pathlib import Path
from madsci.client import ExperimentClient, DataClient

# Configuration
MAX_EXPERIMENTS = 3  # Number of recent experiments to show in list mode

# Get service URLs from environment
EXPERIMENT_URL = os.getenv("EXPERIMENT_SERVER_URL", "http://localhost:8002")
DATA_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8004").rstrip('/')


def list_recent_experiments(experiment_client, limit=MAX_EXPERIMENTS):
    """List the most recent experiments in a table."""
    experiments = experiment_client.get_experiments(number=limit)
    
    # Sort by started_at descending (most recent first)
    experiments = sorted(experiments, key=lambda e: e.started_at or "", reverse=True)
    
    if not experiments:
        print("No experiments found.")
        return
    
    print("\n=== Recent Experiments ===\n")
    print(f"{'ID':<28} {'Name':<30} {'Status':<12} {'Started':<20}")
    print("-" * 100)
    
    for exp in experiments:
        exp_id = exp.experiment_id
        name = (exp.experiment_design.experiment_name[:28] 
                if exp.experiment_design and exp.experiment_design.experiment_name 
                else exp.run_name[:28] if exp.run_name else "N/A")
        status = str(exp.status).split('.')[-1] if exp.status else "N/A"
        started = exp.started_at.strftime("%Y-%m-%d %H:%M:%S") if exp.started_at else "N/A"
        
        print(f"{exp_id:<28} {name:<30} {status:<12} {started:<20}")
    
    print("\nUse: python query_experiment.py <experiment_id> to view details and download images\n")


def view_experiment_details(experiment_client, data_client, experiment_id):
    """View detailed results and download images for a specific experiment."""
    try:
        experiment = experiment_client.get_experiment(experiment_id)
    except Exception as e:
        print(f"Error: Could not find experiment {experiment_id}")
        print(f"Details: {e}")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT DETAILS")
    print("="*80)
    
    # Basic info
    print(f"\nExperiment ID: {experiment.experiment_id}")
    if experiment.experiment_design:
        print(f"Name: {experiment.experiment_design.experiment_name}")
        print(f"Description: {experiment.experiment_design.experiment_description}")
    print(f"Run Name: {experiment.run_name}")
    print(f"Status: {experiment.status}")
    print(f"Started: {experiment.started_at}")
    print(f"Ended: {experiment.ended_at}")
    
    # Query snapshot and extract images
    print("\n" + "-"*80)
    print("EXPERIMENT SNAPSHOT & IMAGES")
    print("-"*80)
    
    try:
        # Query for experiment configuration snapshot
        selector = {
            "label": "experiment_configuration",
            "ownership_info.experiment_id": experiment_id
        }
        
        response = requests.post(
            f"{DATA_URL}/datapoints/query",
            json=selector,
            timeout=10
        )
        response.raise_for_status()
        snapshots = response.json()
        
        if not snapshots:
            print("\nNo configuration snapshot found for this experiment.")
        else:
            # Get the snapshot (should be exactly one)
            snapshot_id = list(snapshots.keys())[0]
            print(f"\nSnapshot ID: {snapshot_id}")
            
            # Retrieve snapshot data
            snapshot_data = data_client.get_datapoint_value(snapshot_id)
            
            # Extract experiment data
            exp_data = snapshot_data.get("experiment_data", {})
            print(f"Target Arrangement: {exp_data.get('target_arrangement')}")
            print(f"Final Arrangement: {exp_data.get('final_arrangement')}")
            print(f"Total Trials: {exp_data.get('total_trials')}")
            
            # Extract and download images
            image_ids = exp_data.get("image_datapoint_ids", [])
            
            if not image_ids:
                print("\nNo images recorded in snapshot.")
            else:
                print(f"\nFound {len(image_ids)} image(s) in snapshot")
                
                # Create output directory
                output_dir = Path("outputs") / experiment_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Download each image
                for idx, image_id in enumerate(image_ids, 1):
                    if idx == 1:
                        image_name = f"initial_scan_{image_id[:8]}.jpg"
                    else:
                        image_name = f"trial_{idx-1}_{image_id[:8]}.jpg"
                    
                    image_path = output_dir / image_name
                    
                    print(f"  Downloading image {idx}/{len(image_ids)}: {image_id[:8]}...")
                    data_client.save_datapoint_value(image_id, str(image_path))
                    print(f"    Saved to: {image_path}")
                
                print(f"\n✓ All images saved to: {output_dir.absolute()}")
            
    except Exception as e:
        print(f"\nError retrieving snapshot: {e}")
    
    print("\n" + "="*80 + "\n")


def main():
    # Initialize clients
    experiment_client = ExperimentClient(EXPERIMENT_URL)
    data_client = DataClient(DATA_URL)
    
    # Check if experiment ID provided
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        view_experiment_details(experiment_client, data_client, experiment_id)
    else:
        list_recent_experiments(experiment_client)


if __name__ == "__main__":
    main()
