# 6-Block Combination Solver Experiment

## Overview

MADSci-compatible port of `BlockCombinationSolver.py` for autonomous block arrangement search.

**Problem**: Given 6 blocks of 4 colors in a 2x3 grid, find a randomly generated target combination.

**Solution**: Autonomously shuffle blocks through random permutations using efficient cycle-based movement until a match is found.

## Architecture

This experiment follows the MADSci integration patterns described in `ROS_Node_and_Experiment_Integration_Guide.md`.

### Files Structure

```
clients/experiments/
  block_combination_solver_experiment.py    # Main experiment script
  helpers/
    block_combination/                      # Organized helper modules
      __init__.py
      locations.py                          # 6 positions + temp (from BlockPositions.py)
      resources.py                          # 6 colored block resources
      algorithms.py                         # Cycle-based permutation + combination tracking
      workflows.py                          # Dynamic workflow generation
      snapshots.py                          # Configuration and timing capture
```

### Key Components

#### 1. **Location Setup** (`locations.py`)
- **7 locations**: 6 grid positions (pos_0 to pos_5) + 1 temporary holding position
- **Layout**: 2x3 grid (top row: 0,1,2 | bottom row: 3,4,5)
- **Representations**: Each location has `raised`, `lowered`, and `normal` joint angles
- **Source**: Position definitions from `reference-scripts/BlockPositions.py`

#### 2. **Resource Management** (`resources.py`)
- **Dynamic creation**: Resources created based on vision detection
- **Block naming**: `{color}_block_pos{i}` (e.g., "red_block_pos0")
  - Position index included since multiple blocks can have same color
- **Attributes**: Color, initial position, detection timestamp

#### 3. **Search Algorithms** (`algorithms.py`)
- **Random target generation**: Creates shuffled target different from initial state
- **Combination tracking**: Uses set of tuples to avoid retrying combinations
- **Cycle-based permutation**: Efficient algorithm that:
  1. Builds position mapping (current → target)
  2. Finds cycles in the permutation
  3. Uses temporary position to rotate blocks through cycles
  4. Minimizes total number of moves

#### 4. **Workflow Generation** (`workflows.py`)
- **Scan workflow**: Move to scan position + capture image
- **Rearrangement workflow**: 
  - Generated dynamically based on desired target
  - Uses cycle algorithm to minimize moves
  - Each workflow ends with scan + capture for verification

#### 5. **Data Capture** (`snapshots.py`)
- **Experiment state**: Locations, resources, arrangements, results
- **Timing statistics**: Total time, average/min/max per attempt
- **Image datapoints**: All captured images linked in snapshot

## Key Differences from Original

### Original (`BlockCombinationSolver.py`)
```python
# Direct ROS/MoveIT control
arm_group.set_joint_value_target(target_joints)
arm_group.plan()
arm_group.execute(plan)

# Direct vision processing
colors = detect_blocks(timeout=5.0)

# Hardcoded positions
self.block_positions_raised[from_pos]
```

### MADSci Port (`block_combination_solver_experiment.py`)
```python
# Workflow-based control
workflow = WorkflowDefinition(steps=[...])
result = workcell_client.start_workflow(workflow)

# Distributed vision processing
image_id = data_client.get_datapoint_value(wrapper_id)
image_data = cv2.imread(image_path)
colors = detect_blocks(image_data)

# Location-based positions
location_id = location_ids[f'pos_{from_pos}']
location_client.get_location(location_id)
```

### Benefits of Port
1. **Separation of concerns**: Robot node handles hardware, experiment handles logic
2. **Data provenance**: All images and results stored in Data Manager
3. **Resource tracking**: Know which block is where at all times
4. **Reproducibility**: Complete experiment state captured in snapshot
5. **Multi-robot ready**: Locations support multiple robot representations

## Usage

### Prerequisites
1. MADSci core services running (Experiment, Resource, Location, Data, Workcell managers)
2. DOFBOT Pro robot node running (`dofbot_ros_node.py`)
3. 6 colored blocks arranged in 2x3 grid
4. Robot camera positioned for overhead view

### Running the Experiment

```bash
cd clients/experiments
python3 block_combination_solver_experiment.py
```

### Configuration

Edit constants in `block_combination_solver_experiment.py`:

```python
MAX_ATTEMPTS = 50  # Maximum attempts before giving up
NUM_BLOCKS = 6     # Number of blocks (6 for 2x3 grid)
```

### Environment Variables

Set service URLs if not using defaults:

```bash
export EXPERIMENT_SERVER_URL=http://localhost:8002
export WORKCELL_SERVER_URL=http://localhost:8005
export LOCATION_SERVER_URL=http://localhost:8006
export RESOURCE_SERVER_URL=http://localhost:8003
export DATA_SERVER_URL=http://localhost:8004
```

## Experiment Flow

1. **Initialize**: Create experiment, setup clients
2. **Setup locations**: Create 7 locations (6 grid + 1 temp)
3. **Initial scan**: Capture image, detect 6 blocks, create resources
4. **Generate target**: Create random target arrangement (hidden from solver)
5. **Check initial**: Test if initial configuration matches target
6. **Search loop**:
   - Generate random untried combination
   - Calculate optimal move sequence (cycle-based)
   - Execute rearrangement workflow
   - Capture and analyze result image
   - Check if target found
   - Track timing for each attempt
7. **Complete**: Snapshot results, cleanup resources, end experiment

## Experiment Output

### Console Output
```
============================================================
6-Block Combination Search Experiment
============================================================

Step 1: Creating experiment design...
✓ Experiment design created

Step 2: Starting experiment...
✓ Experiment started: 01HN...

...

============================================================
EXPERIMENT COMPLETE
============================================================
Solution found: True
Total attempts: 12
Unique combinations tried: 12
Target arrangement: ['red', 'blue', 'yellow', 'green', 'red', 'blue']
Final arrangement: ['red', 'blue', 'yellow', 'green', 'red', 'blue']

TIMING STATISTICS
  Total time: 156.32 seconds (2.61 minutes)
  Average time per attempt: 13.03 seconds
  Fastest attempt: 11.45 seconds
  Slowest attempt: 15.67 seconds
============================================================
```

### Data Outputs
1. **Experiment record**: Start/end times, status, run metadata
2. **Location records**: 7 locations with robot-specific representations
3. **Resource records**: 6 block resources with color and position attributes
4. **Image datapoints**: All captured images (initial + per attempt)
5. **Configuration snapshot**: Complete experiment state with timing statistics

## Algorithm Details

### Cycle-Based Permutation

The algorithm efficiently rearranges blocks by finding cycles in the permutation:

**Example**:
- Current: `[A, B, C, D, E, F]`
- Target:  `[C, A, E, B, F, D]`
- Mapping: `0→1, 1→3, 2→0, 3→5, 4→2, 5→4`
- Cycles: `[0,1,3,5,4,2]` (one large cycle)

**Move sequence**:
1. Move position 0 to temp (creates empty space)
2. Move position 2 → position 0
3. Move position 4 → position 2
4. Move position 5 → position 4
5. Move position 3 → position 5
6. Move position 1 → position 3
7. Move temp → position 1

**Result**: 7 moves instead of 6 naive swaps (same efficiency, but generalizes to multiple cycles)

### Combination Tracking

- Combinations stored as tuples in a set: `{('red', 'blue', ...), ...}`
- Before testing, check if combination already tried
- Avoids infinite loops and duplicate work
- Maximum possible combinations: 6! = 720 (but many are equivalent due to color repetition)

## Troubleshooting

### "No blocks detected in initial scan"
- Check camera positioning and focus
- Verify blocks are in view and well-lit
- Check vision detection thresholds in `helpers/vision.py`

### "Failed to move to raised position"
- Check joint angles in `locations.py`
- Verify no collisions with workspace
- Test positions individually using robot node actions

### "No more unique combinations to try"
- Increased `MAX_ATTEMPTS` if needed
- Check that target is actually achievable with detected blocks
- Verify combination tracking logic

### Workflow execution errors
- Check robot node is running and reachable
- Verify workcell configuration includes DOFBOT_Pro_1
- Check resource tracking (gripper should be empty between moves)

## Extending the Experiment

### Add More Positions
1. Define new positions in `locations.py`
2. Update `NUM_BLOCKS` constant
3. Test positions with robot node

### Change Search Strategy
1. Modify `generate_random_combination()` in `algorithms.py`
2. Options: systematic enumeration, genetic algorithm, heuristic search
3. Keep combination tracking to avoid repeats

### Add Vision Verification
1. In experiment loop, compare detected vs expected arrangement
2. Add correction logic if mismatch detected
3. Log discrepancies in snapshot

### Multiple Target Search
1. Generate multiple targets
2. Search for any match (OR condition)
3. Track which target was found

## Performance Expectations

**Typical performance** (6 blocks, 4 colors):
- **Average solve time**: 2-5 minutes
- **Attempts to solution**: 5-30 (depends on luck)
- **Time per attempt**: 10-15 seconds
  - Movement: 8-12 seconds
  - Vision processing: 1-2 seconds
  - Overhead: 1 second

**Factors affecting performance**:
- Number of unique colors (fewer colors = more equivalent combinations)
- Robot velocity scaling (faster = quicker but less reliable)
- Vision processing time (depends on image resolution)
- Number of moves per rearrangement (depends on cycle complexity)

## Related Files

- **Guide**: `docs/ROS_Node_and_Experiment_Integration_Guide.md`
- **Original script**: `reference-scripts/BlockCombinationSolver.py`
- **Position definitions**: `reference-scripts/BlockPositions.py`
- **Robot node**: `robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py`
- **4-block version**: `clients/experiments/block_permutation_experiment.py`

## References

- MADSci documentation: [Link to docs]
- MoveIT planning: https://moveit.ros.org/
- Cycle decomposition: https://en.wikipedia.org/wiki/Cyclic_permutation

