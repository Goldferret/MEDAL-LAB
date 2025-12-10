# Block Combination Solver - Complete File Structure

## Visual Overview

```
MEDAL-LAB/
│
├── docs/
│   ├── ROS_Node_and_Experiment_Integration_Guide.md    [REFERENCE GUIDE]
│   ├── BlockCombinationSolver_Port_Summary.md          [PORT SUMMARY]
│   └── BlockCombinationSolver_File_Structure.md        [THIS FILE]
│
├── reference-scripts/                                   [ORIGINAL CODE]
│   ├── BlockCombinationSolver.py                       (Original - 556 lines)
│   └── BlockPositions.py                               (Position definitions)
│
├── robot-nodes/
│   └── dofbot-pro-ros/
│       └── nodes/
│           ├── dofbot_ros_node.py                      [REUSED - Already MADSci-compliant]
│           ├── dofbot_ros_interface.py
│           └── dofbot_ros_config.py
│
└── clients/
    └── experiments/
        ├── block_combination_solver_experiment.py       [NEW - Main experiment]
        │
        ├── helpers/
        │   ├── vision.py                                [SHARED - Vision utilities]
        │   │
        │   ├── block_permutations/                      [EXISTING - 4-block experiment]
        │   │   ├── __init__.py
        │   │   ├── locations.py
        │   │   ├── resources.py
        │   │   ├── algorithms.py
        │   │   ├── workflows.py
        │   │   └── snapshots.py
        │   │
        │   └── block_combination/                       [NEW - 6-block experiment]
        │       ├── __init__.py                          [NEW]
        │       ├── locations.py                         [NEW]
        │       ├── resources.py                         [NEW]
        │       ├── algorithms.py                        [NEW]
        │       ├── workflows.py                         [NEW]
        │       └── snapshots.py                         [NEW]
        │
        ├── block_permutation_experiment.py              [EXISTING - 4-block version]
        └── README_block_combination_solver.md           [NEW - Documentation]
```

## File Descriptions

### 📘 Documentation Files

#### `ROS_Node_and_Experiment_Integration_Guide.md`
- **Purpose**: Complete integration guide you created
- **Content**: Architecture patterns, porting instructions, examples
- **Use**: Reference for future ports

#### `BlockCombinationSolver_Port_Summary.md`
- **Purpose**: Summary of this specific port
- **Content**: Decisions, comparisons, changes, testing
- **Use**: Understand what was ported and why

#### `BlockCombinationSolver_File_Structure.md`
- **Purpose**: Visual file organization (this file)
- **Content**: Directory tree, file descriptions, relationships
- **Use**: Navigate the port files

#### `README_block_combination_solver.md`
- **Purpose**: User-facing documentation
- **Content**: Usage, configuration, troubleshooting
- **Use**: Run and configure the experiment

### 📜 Reference Scripts (Original Code)

#### `reference-scripts/BlockCombinationSolver.py`
- **Type**: Original standalone script
- **Lines**: 556
- **Pattern**: Direct ROS control, monolithic class
- **Status**: Preserved as reference, not used in MADSci

#### `reference-scripts/BlockPositions.py`
- **Type**: Position definitions
- **Content**: Joint angles for 6 positions + temp
- **Usage**: Values copied into `helpers/block_combination/locations.py`

### 🤖 Robot Node (Reused)

#### `robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py`
- **Type**: MADSci robot node
- **Status**: **Already exists, reused as-is**
- **Actions exposed**:
  - `move_to_position()` - Joint movement
  - `pick_from_location()` - Pick with resource tracking
  - `place_at_location()` - Place with resource tracking
  - `capture_camera_image()` - Image capture
  - `open_gripper()`, `close_gripper()` - Gripper control

### 🧪 Experiment Files (New)

#### `clients/experiments/block_combination_solver_experiment.py` ⭐
- **Type**: Main experiment orchestration
- **Lines**: 420
- **Structure**: 12-step MADSci pattern
- **Imports from**: All 6 helper modules
- **Key functions**: `main()` - Complete experiment flow

### 🛠️ Helper Modules (New)

#### `helpers/block_combination/__init__.py`
- **Type**: Package initialization
- **Lines**: 5
- **Content**: Module docstring

#### `helpers/block_combination/locations.py`
- **Type**: Location setup
- **Lines**: 78
- **Key constant**: `POSITION_CONFIGS` - 7 positions (6 grid + temp)
- **Key function**: `setup_locations()` - Creates all locations

#### `helpers/block_combination/resources.py`
- **Type**: Resource lifecycle
- **Lines**: 66
- **Key functions**:
  - `create_block_resources()` - Create 6 blocks
  - `cleanup_resources()` - Delete all

#### `helpers/block_combination/algorithms.py` ⭐
- **Type**: Search algorithms
- **Lines**: 174
- **Key functions**:
  - `generate_target()` - Random target
  - `generate_random_combination()` - Untried combination
  - `generate_move_sequence()` - Cycle-based permutation
  - `build_position_mapping()` - Current→target mapping
  - `find_permutation_cycles()` - Cycle detection

#### `helpers/block_combination/workflows.py`
- **Type**: Workflow generation
- **Lines**: 120
- **Key functions**:
  - `create_scan_workflow()` - Initial scan
  - `generate_rearrangement_workflow()` - Dynamic rearrangement

#### `helpers/block_combination/snapshots.py`
- **Type**: State capture
- **Lines**: 72
- **Key function**: `snapshot_configuration()` - Complete state + timing

## Import Relationships

```
block_combination_solver_experiment.py
  │
  ├─ from helpers.vision import detect_blocks
  │   └─ Shared vision processing (used by both experiments)
  │
  ├─ from helpers.block_combination.locations import setup_locations
  │   └─ Creates 7 locations with joint angle representations
  │
  ├─ from helpers.block_combination.resources import
  │   ├─ create_block_resources     (6 colored blocks)
  │   └─ cleanup_resources           (delete all)
  │
  ├─ from helpers.block_combination.algorithms import
  │   ├─ generate_target             (random target)
  │   ├─ generate_random_combination (untried shuffle)
  │   ├─ mark_combination_tried      (track attempts)
  │   └─ is_combination_tried        (check if tried)
  │
  ├─ from helpers.block_combination.workflows import
  │   ├─ create_scan_workflow                (scan + capture)
  │   └─ generate_rearrangement_workflow     (moves + verify)
  │
  └─ from helpers.block_combination.snapshots import
      └─ snapshot_configuration      (save complete state)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT SCRIPT                         │
│         block_combination_solver_experiment.py               │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─→ setup_locations() ──────→ Location Manager
             │                              (7 locations created)
             │
             ├─→ create_scan_workflow() ──→ Workcell Manager ──→ Robot Node
             │                              (robot captures)        │
             │                                                       ↓
             │                              Data Manager ←──────  (image stored)
             │                                   │
             ←─────────────────────────────────  (retrieve image)
             │
             ├─→ detect_blocks(image) ──────→ (process locally)
             │                                   │
             │                                   ↓
             │                              (detect 6 colors)
             │
             ├─→ create_block_resources() ─→ Resource Manager
             │                              (6 blocks created)
             │
             ├─→ generate_target() ─────────→ (random target)
             │
             │   [LOOP: Until solution found]
             │   │
             │   ├─→ generate_random_combination() ──→ (new shuffle)
             │   │
             │   ├─→ generate_move_sequence() ───────→ (optimal moves)
             │   │
             │   ├─→ generate_rearrangement_workflow() → Workcell → Robot
             │   │                                           │
             │   │                                           ↓
             │   │                                    (rearrange blocks)
             │   │                                           │
             │   │                                           ↓
             │   │                                    Data Manager
             │   │                                      (capture image)
             │   │                                           │
             │   ←───────────────────────────────────────────┘
             │   │
             │   ├─→ detect_blocks(image) ────────→ (verify result)
             │   │
             │   └─→ [Check if matches target]
             │
             └─→ snapshot_configuration() ──→ Data Manager
                                             (save complete state)
```

## Size Comparison

### Original Monolithic
```
BlockCombinationSolver.py:        556 lines
BlockPositions.py:                 72 lines (imported)
───────────────────────────────────────────
Total:                            628 lines
```

### MADSci Port
```
Main experiment:                  420 lines
Helper modules:
  - locations.py:                  78 lines
  - resources.py:                  66 lines
  - algorithms.py:                174 lines
  - workflows.py:                 120 lines
  - snapshots.py:                  72 lines
───────────────────────────────────────────
Total implementation:             930 lines
Documentation:
  - README_block_combination:     ~500 lines
  - Port summary:                 ~400 lines
───────────────────────────────────────────
Total with docs:                ~1830 lines
```

**Analysis**:
- More code, but better organized
- Self-documenting structure
- Reusable components
- Comprehensive documentation
- Easier to maintain and extend

## Execution Flow

### 1. Initialization
```
block_combination_solver_experiment.py
  ↓
main()
  ↓
ExperimentClient.start_experiment()
  ↓
Initialize clients (Location, Resource, Data, Workcell)
```

### 2. Setup
```
setup_locations(location_client)
  ↓ [for each position in POSITION_CONFIGS]
  ├─ Create Location object
  ├─ Add to Location Manager
  └─ Set representations for DOFBOT_Pro_1
```

### 3. Initial Scan
```
create_scan_workflow()
  ↓
WorkflowDefinition([move_to_scan, capture])
  ↓
workcell_client.start_workflow()
  ↓
Robot executes → Returns image ID
  ↓
Experiment retrieves → Processes locally
  ↓
create_block_resources() → 6 resources created
```

### 4. Search Loop
```
generate_target() → Random target
  ↓
While not found and attempts < MAX:
  │
  ├─ generate_random_combination() → Untried shuffle
  │
  ├─ generate_move_sequence() → Optimal moves
  │
  ├─ generate_rearrangement_workflow() → Dynamic workflow
  │
  ├─ workcell_client.start_workflow() → Execute
  │
  ├─ Retrieve and process result image
  │
  └─ Check if matches target
```

### 5. Completion
```
snapshot_configuration() → Save all data
  ↓
cleanup_resources() → Delete locations & resources
  ↓
experiment_client.end_experiment() → Finalize
```

## Testing Structure

### Unit Tests (Recommended)
```python
tests/
  test_block_combination/
    test_algorithms.py          # Test pure functions
      - test_generate_move_sequence()
      - test_find_permutation_cycles()
      - test_combination_tracking()
    
    test_locations.py           # Test location setup
      - test_position_configs()
      - test_setup_locations()
    
    test_workflows.py           # Test workflow generation
      - test_scan_workflow()
      - test_rearrangement_workflow()
```

### Integration Tests
```python
tests/
  integration/
    test_full_experiment.py     # End-to-end test
      - test_simple_target()    # Known solution
      - test_cleanup_on_failure()
      - test_snapshot_completeness()
```

## How to Navigate

### Want to understand the port?
1. Read: `BlockCombinationSolver_Port_Summary.md`
2. Compare: `reference-scripts/BlockCombinationSolver.py` vs new files
3. Study: `ROS_Node_and_Experiment_Integration_Guide.md`

### Want to run the experiment?
1. Read: `README_block_combination_solver.md`
2. Check prerequisites
3. Run: `python3 block_combination_solver_experiment.py`

### Want to modify the search algorithm?
1. Edit: `helpers/block_combination/algorithms.py`
2. Modify: `generate_random_combination()` or `generate_move_sequence()`
3. Test with simple cases first

### Want to add new positions?
1. Edit: `helpers/block_combination/locations.py`
2. Add to: `POSITION_CONFIGS`
3. Update: `NUM_BLOCKS` in experiment script

### Want to understand workflows?
1. Study: `helpers/block_combination/workflows.py`
2. See: `generate_rearrangement_workflow()`
3. Trace: How steps are created from move sequences

## Key Design Decisions

### ✅ Why separate helper modules?
- **Reason**: Follow guide's recommendation for organization
- **Benefit**: Each file has single responsibility
- **Pattern**: Same structure as `block_permutations/`

### ✅ Why reuse robot node?
- **Reason**: Already provides needed actions (pick, place, move)
- **Benefit**: No duplicate code, consistent behavior
- **Pattern**: Node is hardware abstraction, not experiment-specific

### ✅ Why client-side vision processing?
- **Reason**: Guide pattern - nodes produce data, experiments consume
- **Benefit**: Heavy processing off robot, enables distributed compute
- **Pattern**: Robot captures → Data Manager stores → Client processes

### ✅ Why cycle-based algorithm?
- **Reason**: Preserve original's efficiency
- **Benefit**: Minimal moves, proven algorithm
- **Pattern**: Pure function, easily testable

### ✅ Why combination tracking?
- **Reason**: Avoid infinite loops, prevent duplicate work
- **Benefit**: Guaranteed progress, better statistics
- **Pattern**: Set-based, O(1) lookup

## Common Questions

**Q: Can I use this for 4 blocks?**
A: Yes! Change `NUM_BLOCKS = 4` and only create 4 positions in `locations.py`

**Q: Can I use different positions?**
A: Yes! Edit `POSITION_CONFIGS` in `locations.py` with your joint angles

**Q: Can I run multiple experiments in parallel?**
A: Yes! Each experiment creates its own resources with unique IDs

**Q: Can I use a different robot?**
A: Yes! Update `ROBOT_NODE` constant and create node with same actions

**Q: Can I change the search strategy?**
A: Yes! Modify `generate_random_combination()` in `algorithms.py`

**Q: How do I visualize the search?**
A: Retrieve images from Data Manager and create visualization script

**Q: Can I add real-time monitoring?**
A: Yes! Add logging/streaming in experiment script or create separate monitor

**Q: How do I replay an experiment?**
A: Load snapshot from Data Manager, contains all images and states

## Next Steps

### To Run
1. Ensure MADSci services running
2. Start DOFBOT robot node
3. Run experiment script
4. Monitor console output
5. Check snapshot in Data Manager

### To Extend
1. Study helper modules
2. Identify what to change
3. Make focused modifications
4. Test incrementally
5. Document changes

### To Learn
1. Read integration guide
2. Compare both experiments (4-block vs 6-block)
3. Trace a single attempt through all layers
4. Experiment with modifications
5. Create your own experiment using patterns

## Summary

✅ **Complete port** following MADSci patterns
✅ **Well-organized** into logical modules  
✅ **Fully documented** with multiple guides
✅ **Ready to use** with clear instructions
✅ **Easy to extend** with modular design
✅ **Follows best practices** from integration guide

The port preserves all functionality while adding significant value through better architecture, complete data provenance, and comprehensive documentation.

