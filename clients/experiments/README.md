# Experiments

Autonomous experiments using the MADSci framework with DOFBOT Pro robotic arm.

## Available Experiments

### Block Permutation Search

Systematically tests different block arrangements to find a target configuration. Demonstrates MADSci patterns for high-throughput experimentation.

**What it does:**
1. Scans workspace and detects 4 colored blocks
2. Generates a random target arrangement
3. Tests permutations until target is found
4. Tracks all data through MADSci managers

**What it teaches:**
- Experiment lifecycle management
- Dynamic resource creation from vision
- Distributed compute (robot does physical work, client does analysis)
- Workflow orchestration and data provenance

## Quick Start

### Prerequisites

- MADSci core services running (see `madsci-core/README.md`)
- Robot node running on Jetson (see `robot-nodes/dofbot-pro-ros/README.md`)
- Client container running (see `clients/README.md`)

### Run the Experiment

```bash
# From client container
cd /workspace/experiments
python block_permutation_experiment.py
```

The experiment will:
- Create locations and resources automatically
- Run up to 3 trials (configurable via `MAX_ATTEMPTS`)
- Print progress for each step
- Clean up all resources when done

### Expected Output

```
=== 4-Block Permutation Search Experiment ===

Step 1: Creating experiment design...
✓ Experiment design created

Step 2: Starting experiment...
✓ Experiment started: 01JKXXX...

...

Trial 1: Testing ['red', 'blue', 'yellow', 'green']
  ✗ No match. Detected: ['red', 'blue', 'yellow', 'green']

Trial 2: Testing ['blue', 'red', 'green', 'yellow']
  ✗ No match. Detected: ['blue', 'red', 'green', 'yellow']

Trial 3: Testing ['green', 'yellow', 'red', 'blue']
  ✓ MATCH FOUND! Detected: ['green', 'yellow', 'red', 'blue']

=== Experiment Summary ===
Success: True
Trials: 3
Target: ['green', 'yellow', 'red', 'blue']
Final: ['green', 'yellow', 'red', 'blue']
Experiment ID: 01JKXXX...

✅ Experiment completed!
```

## Configuration

Edit `block_permutation_experiment.py` to adjust:

```python
MAX_ATTEMPTS = 3  # Number of permutations to test (max 24)
```

## Troubleshooting

**Vision detection issues:**
- Ensure blocks are in 2x3 grid layout (using first 4 positions)
- Check lighting conditions (HSV ranges in `helpers/vision.py`)
- Verify camera is publishing: `rostopic hz /camera/color/image_raw`

**Robot movement issues:**
- Check robot node is running: `docker ps` on Jetson
- Verify joint positions in `helpers/block_permutations/locations.py`
- Check gripper calibration (open: -1.25, closed: -0.5)

**Experiment fails to start:**
- Verify all MADSci services are running: `cd madsci-core && docker compose ps`
- Check environment variables in `.env.global`
- Ensure network connectivity between client, host, and Jetson

## Helper Modules

The experiment uses modular helpers for clean code organization:

- `helpers/vision.py` - OpenCV block detection (client-side processing)
- `helpers/block_permutations/algorithms.py` - Swap calculation and permutation generation
- `helpers/block_permutations/workflows.py` - Dynamic workflow generation
- `helpers/block_permutations/locations.py` - Block position setup
- `helpers/block_permutations/resources.py` - Resource creation and cleanup
- `helpers/block_permutations/snapshots.py` - Configuration archival

## Architecture

**Key Pattern:** The robot captures images and stores them in the Data Manager. The client retrieves images and processes them locally. This distributed compute pattern offloads heavy processing from the robot and enables scalability.
