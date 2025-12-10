# MADSci ROS Node and Experiment Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Porting Guide](#porting-guide)
5. [Complete Porting Checklist](#complete-porting-checklist)

---

## Overview

This guide explains the architecture of a MADSci-integrated ROS robotics system, using the DOFBOT Pro robot and block permutation experiment as a reference implementation.

### System Components

The system consists of two main layers:

#### 1. **Robot Node Layer** (Robot Hardware Interface)
- **Purpose**: Exposes robot capabilities as MADSci-compliant actions
- **Files**: `dofbot_ros_node.py`, `dofbot_ros_interface.py`, `dofbot_ros_config.py`
- **Location**: `robot-nodes/dofbot-pro-ros/nodes/`
- **Responsibility**: Translates MADSci action calls into ROS/hardware commands

#### 2. **Experiment Layer** (High-Level Logic)
- **Purpose**: Orchestrates robot actions to accomplish scientific goals
- **Files**: `block_permutation_experiment.py`, helper modules
- **Location**: `clients/experiments/`
- **Responsibility**: Defines experimental logic, workflows, and data analysis

### Data Flow

```
Experiment Client
    ↓ (submits WorkflowDefinition)
Workcell Manager
    ↓ (routes actions to appropriate nodes)
Robot Node (RestNode)
    ↓ (translates to hardware commands)
ROS Interface
    ↓ (sends ROS messages)
Robot Hardware
```

---

## Architecture Deep Dive

### The Three-Layer Architecture

#### Layer 1: MADSci Node (`dofbot_ros_node.py`)
- **Inherits from**: `RestNode` (MADSci framework class)
- **Exposes**: RESTful API endpoints for robot actions
- **Manages**: Resource tracking (gripper), location lookup, logging
- **Pattern**: Thin orchestration layer - minimal business logic

#### Layer 2: Hardware Interface (`dofbot_ros_interface.py`)
- **Purpose**: Isolate hardware communication from MADSci framework
- **Handles**: ROS connection, action client management, message translation
- **Pattern**: Direct hardware control - no MADSci dependencies beyond logging

#### Layer 3: Configuration (`dofbot_ros_config.py`)
- **Purpose**: Centralize all tunable parameters
- **Contains**: Motion parameters, positions, network settings
- **Pattern**: Single source of truth for robot behavior

### Key Design Patterns

#### 1. **Action Decorator Pattern**
```python
@action
def move_to_position(self, joint_positions: list[float]) -> dict:
    # MADSci automatically registers this as a callable action
```
- Converts methods into workflow-callable actions
- Handles serialization/deserialization automatically
- Provides error handling and logging

#### 2. **Resource Tracking Pattern**
```python
# Create gripper resource in startup
self.gripper = self.resource_client.add_resource(Slot(...))

# Push/pop resources during pick/place
self.resource_client.push(self.gripper.resource_id, block_resource_id)
popped, updated = self.resource_client.pop(self.gripper.resource_id)
```
- Tracks what the robot is holding
- Enables workflow coordination across nodes
- Maintains experiment data provenance

#### 3. **Location Representation Pattern**
```python
representations = {
    "raised": [0.3, -0.5, -0.45, -1.35, 0.25],
    "lowered": [0.3, -1.0, -0.45, -1.35, 0.25]
}
location_client.set_representations(location_id, "DOFBOT_Pro_1", representations)
```
- Each location stores multiple representations per robot node
- Enables multi-robot coordination (different robots, different joint angles)
- Centralizes position management

#### 4. **Datapoint Flow Pattern**
```python
# Robot captures and stores
datapoint = FileDataPoint(label="camera_capture", path=temp_path)
submitted = self.data_client.submit_datapoint(datapoint)
return submitted.datapoint_id  # Return ID, not data

# Experiment retrieves and processes
image_path = data_client.save_datapoint_value(image_id, local_path)
image_data = cv2.imread(image_path)
processed = detect_blocks(image_data)
```
- Robot nodes produce data, don't analyze it
- Experiments retrieve and process data
- Enables distributed compute pattern

---

## File-by-File Breakdown

### 1. `dofbot_ros_node.py` - MADSci Robot Node

**Purpose**: Bridge between MADSci workflow system and physical robot

#### Key Sections

##### A. Class Definition and Initialization
```python
class DofbotRosNode(RestNode):
    robot_interface: DofbotRosInterface = None
    config_model = DofbotRosConfig
```
- Inherits `RestNode` to gain MADSci capabilities
- Links to configuration model for validation
- Declares interface as class attribute

##### B. Startup Handler (Lines 21-46)
```python
def startup_handler(self) -> None:
    self.robot_interface = DofbotRosInterface(self.config, self.logger)
    self.location_client = LocationClient()
    self.gripper = self.resource_client.add_resource(Slot(...))
```
**Purpose**: One-time initialization when node starts
- Creates hardware interface
- Initializes MADSci clients
- Registers gripper resource for tracking

**Why it matters**: Ensures all connections are established before accepting actions

##### C. Action Methods (Lines 64-246)

###### Movement Actions
- `move_to_position()` - Direct joint control
- `home_robot()` - Return to safe position
- `get_current_position()` - Query joint state

**Pattern**: All follow same structure:
1. Validate inputs
2. Log intent
3. Call interface method
4. Return standardized dict with status

###### Gripper Actions
- `open_gripper()` - Release objects
- `close_gripper()` - Grasp objects

**Note**: Hardcoded positions ([-1.25, -1.25, -1.25] and [-0.5, -0.5, -0.5])
- Consider moving these to config for hardware-specific tuning

###### Perception Action
- `capture_camera_image()` - Returns datapoint ID

**Critical Pattern**:
```python
cv_image = self.robot_interface.capture_camera_image()
temp_path = Path(temp_file.name)
cv2.imwrite(str(temp_path), cv_image)
datapoint = FileDataPoint(label="camera_capture", path=str(temp_path))
submitted = self.data_client.submit_datapoint(datapoint)
os.remove(temp_path)
return submitted.datapoint_id  # Return ID, not image!
```
- Node doesn't process image, just stores it
- Returns reference (ID) for experiment to retrieve
- Temporary file cleaned up after Data Manager copies it

##### D. Compound Actions (Lines 247-446)

###### `pick_from_location()` - Multi-step Pick Operation
```python
def pick_from_location(self, location_id: str) -> dict:
    location = self.location_client.get_location(location_id)
    representations = location.representations[node_name]
    
    # Open gripper
    # Move to raised position
    # Move to lowered position
    # Close gripper
    # Push resource to gripper tracking
    # Return to raised position
```

**Key Points**:
- Retrieves location's joint angle representation for this specific robot
- Sequences hardware calls (open, move, close)
- Updates resource tracking via `resource_client.push()`
- Returns standardized status dict

###### `place_at_location()` - Multi-step Place Operation
Similar to pick, but reversed:
1. Pop resource from gripper tracking
2. Move to position
3. Open gripper
4. Attach resource to location
5. Return to raised

###### `swap_blocks()` - Orchestrated Multi-Action
```python
def swap_blocks(self, location_a_id, location_b_id, temp_location_id):
    # A → Temp
    pick_from_location(location_a_id)
    place_at_location(temp_location_id)
    
    # B → A
    pick_from_location(location_b_id)
    place_at_location(location_a_id)
    
    # Temp → B
    pick_from_location(temp_location_id)
    place_at_location(location_b_id)
```

**Design Note**: This could be a workflow, but it's implemented as a single action because:
- It's a common, atomic operation in this domain
- Reduces workflow complexity
- Provides better error handling (all-or-nothing)

##### E. Main Block (Lines 449-458)
```python
if __name__ == "__main__":
    node_url = os.getenv("DOFBOT_PRO_1_URL", "http://192.168.1.200:2000/")
    node = DofbotRosNode(node_config=DofbotRosConfig(node_url=node_url))
    node.start_node()
```
- Reads URL from environment (supports deployment configuration)
- Starts node server (blocks indefinitely)

---

### 2. `dofbot_ros_interface.py` - ROS Communication Layer

**Purpose**: Isolate ROS-specific code from MADSci framework

#### Key Sections

##### A. Initialization (Lines 21-65)
```python
def __init__(self, config, logger):
    rospy.init_node("dofbot_madsci_node", anonymous=True, disable_signals=True)
    self.move_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
    self.move_client.wait_for_server()
    self.gripper_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
    self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
    self.spinner = threading.Thread(target=rospy.spin, daemon=True)
    self.spinner.start()
```

**Critical Details**:
- `disable_signals=True` - Required because ROS node runs in a thread (RestNode is main thread)
- `anonymous=True` - Allows multiple nodes if needed
- Two action clients - one for arm, one for gripper (both use same MoveIT server)
- Background thread for ROS callbacks (non-blocking)
- Wait for first joint state before returning

##### B. Move to Joints (Lines 71-136)
```python
def move_to_joints(self, joint_positions: list[float]) -> bool:
    goal = MoveGroupGoal()
    goal.request.group_name = "arm_group"
    goal.request.num_planning_attempts = self.config.num_planning_attempts
    goal.request.allowed_planning_time = self.config.planning_time
    
    # Create joint constraints
    for name, position in zip(joint_names, joint_positions):
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = position
        constraints.joint_constraints.append(constraint)
    
    self.move_client.send_goal(goal)
    finished = self.move_client.wait_for_result(rospy.Duration(30.0))
    return state == actionlib.GoalStatus.SUCCEEDED
```

**Key Concepts**:
- Uses MoveIT action server (not direct motor control)
- Joint constraints specify goal positions
- Blocks until movement completes or times out
- Returns simple bool (success/failure)

**MoveIT Parameters** (from config):
- `num_planning_attempts` - How many times to try finding a path
- `planning_time` - Max seconds per planning attempt
- `max_velocity_scaling_factor` - Speed multiplier (1.0 = full speed)
- `max_acceleration_scaling_factor` - Acceleration multiplier

##### C. Get Current Joints (Lines 138-155)
```python
def get_current_joints(self) -> list[float]:
    return list(self.current_joint_state.position[:5])
```
- Returns cached joint state (updated by subscriber callback)
- Only returns first 5 joints (arm), not gripper joints

##### D. Move Gripper (Lines 157-219)
```python
def move_gripper(self, gripper_positions: list[float]) -> bool:
    goal.request.group_name = "grip_group"
    constraint.joint_name = 'grip_joint'
    constraint.position = gripper_positions[0]
```
- Same pattern as arm movement, but uses "grip_group"
- Only controls main `grip_joint` (others are mechanically linked)
- Shorter timeout (10 seconds vs 30)

##### E. Camera Capture (Lines 221-237)
```python
def capture_camera_image(self):
    from rosbags.image import message_to_cvimage
    img_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=5.0)
    return message_to_cvimage(img_msg, 'bgr8')
```
- Waits for single message from camera topic
- Converts ROS Image message to OpenCV format (BGR)
- Returns numpy array directly (interface layer returns data, not IDs)

---

### 3. `dofbot_ros_config.py` - Configuration Model

**Purpose**: Centralized, validated configuration

#### Structure

```python
class DofbotRosConfig(RestNodeConfig):
    node_definition: str = "nodes/default.node.yaml"
    
    # MoveIT parameters
    allow_replanning: bool = True
    planning_time: float = 5.0
    num_planning_attempts: int = 10
    goal_tolerance: float = 0.01
    max_velocity_scaling_factor: float = 3.0
    max_acceleration_scaling_factor: float = 3.0
    
    # Positions
    home_position: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
```

**Key Points**:
- Inherits from `RestNodeConfig` (provides `node_url` field)
- Type hints enable validation
- Default values document expected behavior
- Comments explain units and ranges

**Note on Velocity/Acceleration Scaling**:
- Values > 1.0 seem unusual (typically 0.0-1.0)
- May be robot-specific or experimental values
- Document reasoning if intentional

---

### 4. `block_permutation_experiment.py` - Experiment Orchestration

**Purpose**: High-level experiment logic using robot node

#### Experiment Flow

##### Phase 1: Setup (Lines 47-77)
```python
experiment_design = ExperimentDesign(...)
experiment_client = ExperimentClient(EXPERIMENT_URL)
experiment = experiment_client.start_experiment(experiment_design, run_name, run_description)

location_client = LocationClient(LOCATION_URL)
resource_client = ResourceClient(RESOURCE_URL)
data_client = DataClient(DATA_URL)
workcell_client = WorkcellClient(WORKCELL_URL)

location_ids = setup_locations(location_client)
```

**Critical Pattern**: Initialize experiment FIRST, then create resources
- Experiment ID needed for resource ownership tracking
- Enables cleanup if experiment fails mid-execution

##### Phase 2: Initial Scan (Lines 79-113)
```python
scan_workflow = create_scan_workflow()
scan_result = workcell_client.start_workflow(scan_workflow)

# Two-level datapoint retrieval
wrapper_id = capture_step.result.datapoints.json_result
image_id = data_client.get_datapoint_value(wrapper_id)

# Retrieve and process on client
data_client.save_datapoint_value(image_id, local_path)
image_data = cv2.imread(local_path)
current_arrangement = detect_blocks(image_data)

# Create resources based on detection
resource_ids = create_block_resources(resource_client, location_client, 
                                      current_arrangement, location_ids)
```

**Key Design Decisions**:
- Robot moves to scan position and captures (workflow)
- Client processes image (not robot node)
- Resources created dynamically based on vision detection
- Resources immediately attached to locations

##### Phase 3: Generate Test Cases (Lines 115-120)
```python
target_arrangement = generate_target(current_arrangement)
permutations = generate_permutations(current_arrangement)
```
- Generates random target (could be parameter)
- Creates all possible permutations in random order
- Could implement smarter search algorithms here

##### Phase 4: Experiment Loop (Lines 122-162)
```python
for perm in permutations[:MAX_ATTEMPTS]:
    # Generate workflow with swap operations
    workflow_def = generate_workflow_for_permutation(perm, current_arrangement, location_ids, trial_count)
    
    # Execute workflow
    result = workcell_client.start_workflow(workflow_def)
    
    # Retrieve and process image
    wrapper_id = capture_step.result.datapoints.json_result
    image_id = data_client.get_datapoint_value(wrapper_id)
    data_client.save_datapoint_value(image_id, local_path)
    image_data = cv2.imread(local_path)
    detected = detect_blocks(image_data)
    
    # Update state (trust workflow, not vision)
    current_arrangement = perm
    
    # Check success
    if detected == target_arrangement:
        break
```

**Important Patterns**:
- Workflow generation is dynamic (based on current state)
- Trusts workflow execution over vision detection for state tracking
- Vision used only for success verification
- Each trial tracked with images

##### Phase 5: Cleanup (Lines 165-183)
```python
snapshot_id = snapshot_configuration(data_client, location_client, resource_client,
                                     location_ids, resource_ids, 
                                     target_arrangement, current_arrangement, 
                                     trial_count, image_datapoint_ids, 
                                     experiment.experiment_id)

cleanup_resources(location_client, resource_client, location_ids, resource_ids)

final_experiment = experiment_client.end_experiment(experiment.experiment_id)
```

**Critical Steps**:
1. Snapshot configuration (preserves experiment state)
2. Delete resources and locations (cleanup)
3. End experiment (finalizes timing/status)

---

### 5. Helper Modules (`helpers/block_permutations/`)

#### File Structure
```
helpers/
  block_permutations/
    __init__.py
    locations.py      # Location setup
    resources.py      # Resource lifecycle
    algorithms.py     # Permutation logic
    workflows.py      # Dynamic workflow generation
    snapshots.py      # Configuration capture
```

#### `locations.py` - Location Management
```python
POSITION_CONFIGS = {
    "pos_0": {
        "normal": [0.3, -0.8, -0.45, -1.35, 0.25],
        "raised": [0.3, -0.5, -0.45, -1.35, 0.25],
        "lowered": [0.3, -1.0, -0.45, -1.35, 0.25]
    },
    # ... more positions
}

def setup_locations(location_client):
    for pos_name, representations in POSITION_CONFIGS.items():
        location = Location(location_id=new_ulid_str(), name=f"block_{pos_name}")
        created = location_client.add_location(location)
        location_client.set_representations(location.location_id, ROBOT_NODE, representations)
        location_ids[pos_name] = location.location_id
```

**Key Points**:
- Hardcoded positions for this specific robot/workspace
- Three representations per location (normal, raised, lowered)
- Robot node name hardcoded ("DOFBOT_Pro_1")
- Returns dict mapping position names to IDs

#### `resources.py` - Resource Lifecycle
```python
def create_block_resources(resource_client, location_client, colors, location_ids):
    for i, color in enumerate(colors):
        block = Asset(resource_name=f"{color}_block", resource_class="colored_block",
                     attributes={"color": color, "initial_position": i})
        created = resource_client.add_resource(block)
        location_client.attach_resource(location_id=location_ids[f"pos_{i}"], 
                                        resource_id=created.resource_id)
```

**Pattern**: Resources created based on vision detection
- Dynamic creation (not predefined)
- Attributes store metadata (color, initial position)
- Immediately attached to locations

#### `workflows.py` - Dynamic Workflow Generation
```python
def generate_workflow_for_permutation(target_perm, current_arrangement, location_ids, trial_count):
    swaps = calculate_swaps(current_arrangement, target_perm)
    steps = []
    
    for i, (pos_a, pos_b) in enumerate(swaps):
        if pos_a == pos_b:  # Skip redundant swaps
            continue
        steps.append(StepDefinition(
            name=f"swap_{i}",
            node=ROBOT_NODE,
            action="swap_blocks",
            args={
                "location_a_id": location_ids[f"pos_{pos_a}"],
                "location_b_id": location_ids[f"pos_{pos_b}"],
                "temp_location_id": location_ids["temp"]
            }
        ))
    
    # Always end with scan + capture
    steps.append(StepDefinition(name="move_to_scan", node=ROBOT_NODE, 
                               action="move_to_position", 
                               args={"joint_positions": scan_position}))
    steps.append(StepDefinition(name="capture_final", node=ROBOT_NODE, 
                               action="capture_camera_image", args={}))
    
    return WorkflowDefinition(name=f"test_permutation_{trial_count}", steps=steps)
```

**Critical Pattern**: Workflows are generated programmatically
- Not hardcoded - calculated based on desired outcome
- Minimal swaps calculated algorithmically
- Consistent structure (swaps → scan → capture)

#### `algorithms.py` - Permutation Logic
```python
def calculate_swaps(current, target):
    working = current.copy()
    swaps = []
    for i in range(len(working)):
        if working[i] == target[i]:
            continue
        for j in range(i + 1, len(working)):
            if working[j] == target[i]:
                working[i], working[j] = working[j], working[i]
                swaps.append((i, j))
                break
    return swaps
```
- Greedy algorithm (not necessarily optimal)
- Returns list of position pairs to swap
- Simple and deterministic

#### `snapshots.py` - Configuration Capture
```python
def snapshot_configuration(data_client, location_client, resource_client,
                          location_ids, resource_ids, target, final, trials, 
                          image_ids, experiment_id):
    config = {
        "locations": {name: location_client.get_location(loc_id).model_dump() 
                     for name, loc_id in location_ids.items()},
        "resources": {name: resource_client.get_resource(res_id).model_dump()
                     for name, res_id in resource_ids.items()},
        "experiment_data": {
            "target_arrangement": target,
            "final_arrangement": final,
            "total_trials": trials,
            "image_datapoint_ids": image_ids
        }
    }
    snapshot = ValueDataPoint(label="experiment_configuration", value=config,
                              ownership_info={"experiment_id": experiment_id})
    submitted = data_client.submit_datapoint(snapshot)
    return submitted.datapoint_id
```

**Purpose**: Preserves full experiment state
- All locations with their representations
- All resources with their attributes
- Experiment results and metadata
- Links to all captured images
- Owned by experiment (for lifecycle management)

---

## Porting Guide

### Overview of Porting Process

**Goal**: Convert standalone Python robotics code into MADSci-integrated node + experiment

**Input**: Python script that directly controls robot hardware
**Output**: 
- Robot node (exposes hardware as MADSci actions)
- Experiment script (orchestrates actions via workflows)
- Helper modules (organize complex logic)

### Step-by-Step Porting Process

---

### STEP 1: Analyze Existing Code

#### 1.1 Identify Hardware Interfaces
Look for imports and connections to hardware:
```python
# Examples of what to look for:
import serial  # Direct serial connection
import rospy   # ROS connection
import cv2     # Camera access
from some_robot_library import RobotArm  # Proprietary SDK
```

**Task**: List all hardware interfaces and their initialization code

#### 1.2 Identify Actions (Motion/Perception)
Extract distinct operations the robot performs:
```python
# Example from standalone code:
def move_to_home():
    arm.move_joints([0, 0, 0, 0, 0])
    
def pick_block(x, y, z):
    arm.move_to_cartesian(x, y, z)
    gripper.close()
    arm.move_to_cartesian(x, y, z + 50)
```

**Task**: List all distinct actions with their parameters

#### 1.3 Identify High-Level Logic
Separate algorithmic/decision logic from hardware control:
```python
# High-level logic (stays in experiment):
for block in all_blocks:
    if block.color == target_color:
        pick_block(block.x, block.y, block.z)
        place_block(target.x, target.y, target.z)
        
# Hardware control (moves to node):
def pick_block(x, y, z):
    # ... hardware commands ...
```

**Task**: Identify what logic belongs in experiment vs node

#### 1.4 Identify Configuration Parameters
Extract all tunable values:
```python
# Examples:
ROBOT_IP = "192.168.1.100"
HOME_POSITION = [0, 0, 0, 0, 0]
GRIPPER_OPEN = 1.2
GRIPPER_CLOSED = 0.5
MOTION_SPEED = 0.3
CAMERA_EXPOSURE = 50
```

**Task**: List all parameters and their current values

---

### STEP 2: Create Robot Node Structure

#### 2.1 Create Node Directory
```
robot-nodes/
  your-robot-name/
    nodes/
      __init__.py
      your_robot_node.py         # Main node
      your_robot_interface.py    # Hardware interface
      your_robot_config.py       # Configuration
      default.node.yaml          # Node definition
    docker-compose.yml           # Deployment config
    README.md                    # Documentation
```

#### 2.2 Create Node Definition YAML
File: `nodes/default.node.yaml`
```yaml
node_name: Your_Robot_1
node_description: Your robot description
node_class: your_robot_type
capabilities:
  - movement
  - manipulation
  - vision
attributes:
  manufacturer: YourManufacturer
  model: YourModel
  serial_number: "12345"
```

**Purpose**: Metadata for node registration with MADSci

---

### STEP 3: Implement Configuration

#### 3.1 Create Config Class
File: `nodes/your_robot_config.py`

```python
"""Configuration for Your Robot Node."""
from madsci.common.types.node_types import RestNodeConfig

class YourRobotConfig(RestNodeConfig):
    """Configuration for Your Robot Node."""
    
    node_definition: str = "nodes/default.node.yaml"
    
    # === HARDWARE CONNECTION ===
    robot_ip: str = "192.168.1.100"
    robot_port: int = 5000
    connection_timeout: float = 5.0
    
    # === MOTION PARAMETERS ===
    default_speed: float = 0.3
    default_acceleration: float = 0.2
    position_tolerance: float = 0.01  # meters or radians
    
    # === GRIPPER PARAMETERS ===
    gripper_open_position: float = 1.2
    gripper_closed_position: float = 0.5
    gripper_force: float = 50.0  # Newtons
    
    # === PREDEFINED POSITIONS ===
    home_position: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    scan_position: list[float] = [0.79, 0.79, -1.57, -1.57, 0.0]
    
    # === CAMERA PARAMETERS ===
    camera_topic: str = "/camera/color/image_raw"  # If using ROS
    camera_exposure: int = 50
    camera_gain: int = 100
```

**Tips**:
- Group related parameters with comments
- Include units in docstrings or variable names
- Use type hints for validation
- Provide sensible defaults

---

### STEP 4: Implement Hardware Interface

#### 4.1 Create Interface Class
File: `nodes/your_robot_interface.py`

```python
"""Hardware interface for Your Robot."""
from typing import Optional
from madsci.client.event_client import EventClient
from your_robot_config import YourRobotConfig

# Import hardware-specific libraries
import your_robot_sdk  # Example

class YourRobotInterface:
    """Direct hardware interface for Your Robot."""
    
    def __init__(
        self, 
        config: YourRobotConfig,
        logger: Optional[EventClient] = None
    ) -> None:
        """
        Initialize hardware connection.
        
        Args:
            config: Configuration with hardware parameters
            logger: Event logger for status messages
        """
        self.config = config
        self.logger = logger or EventClient()
        
        # Initialize hardware connection
        try:
            self.robot = your_robot_sdk.RobotArm(
                ip=config.robot_ip,
                port=config.robot_port
            )
            self.robot.connect(timeout=config.connection_timeout)
            self.logger.log("Robot connection established")
            
            # Initialize other hardware (gripper, camera, etc.)
            self.gripper = self.robot.get_gripper()
            self.camera = your_robot_sdk.Camera(config.camera_topic)
            
            self.logger.log("Hardware interface initialized")
            
        except Exception as e:
            self.logger.log_error(f"Hardware initialization failed: {str(e)}")
            raise
    
    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """
        Move robot to specified joint positions.
        
        Args:
            joint_positions: List of joint angles
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.log(f"Moving to joints: {joint_positions}")
            
            # Call hardware-specific method
            result = self.robot.move_joints(
                positions=joint_positions,
                speed=self.config.default_speed,
                acceleration=self.config.default_acceleration
            )
            
            # Wait for completion
            result.wait_for_completion(timeout=30.0)
            
            if result.is_success():
                self.logger.log("Movement completed successfully")
                return True
            else:
                self.logger.log_error(f"Movement failed: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.log_error(f"Movement error: {str(e)}")
            return False
    
    def get_current_joints(self) -> list[float]:
        """Get current joint positions."""
        try:
            return self.robot.get_joint_positions()
        except Exception as e:
            self.logger.log_error(f"Failed to get joints: {str(e)}")
            return []
    
    def move_gripper(self, position: float) -> bool:
        """Move gripper to specified position."""
        try:
            self.logger.log(f"Moving gripper to: {position}")
            result = self.gripper.move_to(position)
            result.wait_for_completion(timeout=10.0)
            return result.is_success()
        except Exception as e:
            self.logger.log_error(f"Gripper error: {str(e)}")
            return False
    
    def capture_camera_image(self):
        """
        Capture image from camera.
        
        Returns:
            numpy array in BGR format, or None on failure
        """
        try:
            self.logger.log("Capturing camera image")
            image = self.camera.capture()
            return image
        except Exception as e:
            self.logger.log_error(f"Camera capture failed: {str(e)}")
            return None
    
    def __del__(self):
        """Clean up hardware connection."""
        try:
            if hasattr(self, 'robot'):
                self.robot.disconnect()
        except:
            pass
```

**Key Principles**:
1. **Only hardware code here** - No MADSci resource tracking, location lookup, etc.
2. **Return simple types** - bool, list[float], numpy arrays
3. **Log everything** - Helps debugging hardware issues
4. **Handle exceptions** - Don't crash node on hardware errors
5. **Blocking is OK** - Wait for movements to complete

---

### STEP 5: Implement Robot Node

#### 5.1 Create Node Class
File: `nodes/your_robot_node.py`

```python
"""MADSci Robot Node for Your Robot."""
from typing import Any
from madsci.node_module.rest_node_module import RestNode
from madsci.node_module.helpers import action
from madsci.common.types.resource_types import Slot
from your_robot_config import YourRobotConfig
from your_robot_interface import YourRobotInterface


class YourRobotNode(RestNode):
    """MADSci-compliant robot node for Your Robot."""
    
    robot_interface: YourRobotInterface = None
    config_model = YourRobotConfig
    
    def startup_handler(self) -> None:
        """Initialize robot interface and resources."""
        # Initialize hardware interface
        self.robot_interface = YourRobotInterface(self.config, self.logger)
        self.logger.log("Robot interface initialized")
        
        # Initialize Location Manager client
        from madsci.client import LocationClient
        self.location_client = LocationClient()
        self.logger.log("Location client initialized")
        
        # Create gripper resource for tracking what robot is holding
        self.gripper = self.resource_client.add_resource(
            Slot(
                resource_name=f"robot_gripper_{self.node_definition.node_name}",
                resource_class="robot_gripper",
                capacity=1,
                attributes={
                    "gripper_type": "your_robot_gripper",
                    "description": "Robot gripper for holding objects"
                }
            )
        )
        
        self.logger.log(f"Gripper resource initialized: {self.gripper.resource_id}")
        self.logger.log("Robot node startup complete")
    
    def shutdown_handler(self) -> None:
        """Clean up robot interface."""
        self.logger.log("Shutting down robot node")
        if self.robot_interface:
            del self.robot_interface
    
    def state_handler(self) -> None:
        """Report current robot state."""
        if self.robot_interface:
            current_joints = self.robot_interface.get_current_joints()
            self.node_state = {
                "joint_positions": current_joints,
                "gripper_resource_id": self.gripper.resource_id if self.gripper else None,
                "num_joints": len(current_joints)
            }
    
    # ========================================================================
    # BASIC ACTIONS
    # ========================================================================
    
    @action
    def move_to_position(self, joint_positions: list[float]) -> dict:
        """Move robot to specified joint positions."""
        # Validate
        expected_joints = 5  # Adjust for your robot
        if len(joint_positions) != expected_joints:
            return {
                "status": "error",
                "message": f"Expected {expected_joints} joints, got {len(joint_positions)}"
            }
        
        # Execute
        self.logger.log(f"Moving to position: {joint_positions}")
        success = self.robot_interface.move_to_joints(joint_positions)
        
        # Return
        if success:
            return {
                "status": "success",
                "message": "Movement completed",
                "final_position": joint_positions
            }
        else:
            return {
                "status": "error",
                "message": "Movement failed - check logs for details"
            }
    
    @action
    def get_current_position(self) -> dict:
        """Get current robot joint positions."""
        current_joints = self.robot_interface.get_current_joints()
        
        if current_joints:
            return {
                "status": "success",
                "joint_positions": current_joints,
                "num_joints": len(current_joints)
            }
        else:
            return {
                "status": "error",
                "message": "Failed to get current position"
            }
    
    @action
    def home_robot(self) -> dict:
        """Move robot to home position."""
        home_position = self.config.home_position
        self.logger.log(f"Moving to home position: {home_position}")
        
        success = self.robot_interface.move_to_joints(home_position)
        
        if success:
            return {
                "status": "success",
                "message": "Robot homed successfully",
                "home_position": home_position
            }
        else:
            return {
                "status": "error",
                "message": "Failed to home robot"
            }
    
    @action
    def open_gripper(self) -> dict:
        """Open the gripper."""
        self.logger.log("Opening gripper")
        success = self.robot_interface.move_gripper(self.config.gripper_open_position)
        
        if success:
            return {"status": "success", "message": "Gripper opened"}
        else:
            return {"status": "error", "message": "Failed to open gripper"}
    
    @action
    def close_gripper(self) -> dict:
        """Close the gripper."""
        self.logger.log("Closing gripper")
        success = self.robot_interface.move_gripper(self.config.gripper_closed_position)
        
        if success:
            return {"status": "success", "message": "Gripper closed"}
        else:
            return {"status": "error", "message": "Failed to close gripper"}
    
    @action
    def capture_camera_image(self) -> dict:
        """
        Capture image from camera and return as datapoint.
        
        Returns:
            Datapoint ID of captured image
        """
        import tempfile
        import os
        from pathlib import Path
        import cv2
        from madsci.common.types.datapoint_types import FileDataPoint
        
        try:
            self.logger.log("Capturing camera image")
            
            # Get image from hardware
            cv_image = self.robot_interface.capture_camera_image()
            
            if cv_image is None:
                return {
                    "status": "error",
                    "message": "Failed to capture image from camera"
                }
            
            # Write to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = Path(temp_file.name)
            temp_file.close()
            cv2.imwrite(str(temp_path), cv_image)
            
            # Submit to Data Manager
            datapoint = FileDataPoint(
                label="camera_capture",
                path=str(temp_path)
            )
            submitted = self.data_client.submit_datapoint(datapoint)
            
            # Delete temp file
            os.remove(temp_path)
            
            self.logger.log(f"Image captured: {submitted.datapoint_id}")
            
            # Return datapoint ID (not image data!)
            return submitted.datapoint_id
            
        except Exception as e:
            self.logger.log_error(f"Failed to capture image: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to capture image: {str(e)}"
            }
    
    # ========================================================================
    # COMPOUND ACTIONS (using Location Manager)
    # ========================================================================
    
    @action
    def pick_from_location(self, location_id: str) -> dict:
        """Pick object from specified location."""
        try:
            self.logger.log(f"Picking from location: {location_id}")
            
            # Get location representations
            location = self.location_client.get_location(location_id)
            node_name = self.node_definition.node_name
            
            if node_name not in location.representations:
                return {
                    "status": "error",
                    "message": f"No representation found for node {node_name}"
                }
            
            representations = location.representations[node_name]
            
            # Execute pick sequence
            # 1. Open gripper
            if not self.robot_interface.move_gripper(self.config.gripper_open_position):
                return {"status": "error", "message": "Failed to open gripper"}
            
            # 2. Move to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to move to raised position"}
            
            # 3. Move to lowered position
            if not self.robot_interface.move_to_joints(representations["lowered"]):
                return {"status": "error", "message": "Failed to move to lowered position"}
            
            # 4. Close gripper
            if not self.robot_interface.move_gripper(self.config.gripper_closed_position):
                return {"status": "error", "message": "Failed to close gripper"}
            
            # 5. Update resource tracking
            resource_id = location.resource_id
            if resource_id:
                self.resource_client.push(self.gripper.resource_id, resource_id)
                self.logger.log(f"Pushed resource {resource_id} to gripper")
            
            # 6. Return to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to return to raised position"}
            
            return {
                "status": "success",
                "message": "Pick completed successfully",
                "resource_id": resource_id
            }
            
        except Exception as e:
            self.logger.log_error(f"Pick failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Pick failed: {str(e)}"
            }
    
    @action
    def place_at_location(self, location_id: str) -> dict:
        """Place held object at specified location."""
        try:
            self.logger.log(f"Placing at location: {location_id}")
            
            # Pop resource from gripper
            popped_resource = None
            try:
                popped, updated_gripper = self.resource_client.pop(self.gripper.resource_id)
                popped_resource = popped
                self.logger.log(f"Popped resource {popped.resource_id} from gripper")
            except Exception as e:
                self.logger.log_warning(f"No resource in gripper: {str(e)}")
            
            # Get location representations
            location = self.location_client.get_location(location_id)
            node_name = self.node_definition.node_name
            
            if node_name not in location.representations:
                return {
                    "status": "error",
                    "message": f"No representation found for node {node_name}"
                }
            
            representations = location.representations[node_name]
            
            # Execute place sequence
            # 1. Move to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to move to raised position"}
            
            # 2. Move to lowered position
            if not self.robot_interface.move_to_joints(representations["lowered"]):
                return {"status": "error", "message": "Failed to move to lowered position"}
            
            # 3. Open gripper
            if not self.robot_interface.move_gripper(self.config.gripper_open_position):
                return {"status": "error", "message": "Failed to open gripper"}
            
            # 4. Attach resource to location
            if popped_resource:
                self.location_client.attach_resource(
                    location_id=location_id,
                    resource_id=popped_resource.resource_id
                )
                self.logger.log(f"Attached resource {popped_resource.resource_id} to location")
            
            # 5. Return to raised position
            if not self.robot_interface.move_to_joints(representations["raised"]):
                return {"status": "error", "message": "Failed to return to raised position"}
            
            return {
                "status": "success",
                "message": "Place completed successfully",
                "resource_id": popped_resource.resource_id if popped_resource else None
            }
            
        except Exception as e:
            self.logger.log_error(f"Place failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Place failed: {str(e)}"
            }


if __name__ == "__main__":
    import os
    
    # Get node URL from environment
    node_url = os.getenv("YOUR_ROBOT_URL", "http://192.168.1.100:2000/")
    
    node = YourRobotNode(node_config=YourRobotConfig(node_url=node_url))
    node.start_node()
```

**Action Design Principles**:
1. **Return dicts** with `status` and `message` fields
2. **Validate inputs** before calling hardware
3. **Log intentions** before executing
4. **Handle errors gracefully** - return error status, don't crash
5. **Update resource tracking** for pick/place operations

---

### STEP 6: Create Experiment Structure

#### 6.1 Create Experiment Directory
```
clients/
  experiments/
    your_experiment.py           # Main experiment
    helpers/
      your_experiment/
        __init__.py
        locations.py              # Location definitions
        resources.py              # Resource management
        workflows.py              # Workflow generation
        algorithms.py             # Experiment logic
        snapshots.py              # Configuration capture
    test_your_experiment.py      # Unit tests
```

#### 6.2 Create Helper Modules

##### `helpers/your_experiment/locations.py`
```python
"""Location setup for your experiment."""
from madsci.common.types.location_types import Location
from madsci.common.utils import new_ulid_str

ROBOT_NODE = "Your_Robot_1"  # Match node_name in YAML

# Define positions for your workspace
POSITION_CONFIGS = {
    "position_1": {
        "raised": [0.0, -0.5, -0.3, -1.0, 0.0],   # Above position
        "lowered": [0.0, -0.9, -0.3, -1.0, 0.0]   # At position
    },
    "position_2": {
        "raised": [0.3, -0.5, -0.3, -1.0, 0.3],
        "lowered": [0.3, -0.9, -0.3, -1.0, 0.3]
    },
    # ... more positions
}

def setup_locations(location_client):
    """Create locations with representations."""
    location_ids = {}
    
    for pos_name, representations in POSITION_CONFIGS.items():
        location = Location(
            location_id=new_ulid_str(),
            name=f"workspace_{pos_name}",
            description=f"Workspace position {pos_name}"
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
```

**How to define positions**:
1. Manually jog robot to desired position
2. Record joint angles
3. Add to POSITION_CONFIGS
4. Test with robot node's `move_to_position` action

##### `helpers/your_experiment/resources.py`
```python
"""Resource creation and cleanup."""
from madsci.common.types.resource_types import Asset

def create_resources(resource_client, location_client, items, location_ids):
    """Create resources based on detected items."""
    resource_ids = {}
    
    for i, item in enumerate(items):
        if item is None:
            continue
        
        resource = Asset(
            resource_name=f"item_{i}",
            resource_class="experiment_item",
            attributes={
                "item_type": item.get("type"),
                "initial_position": i
            }
        )
        created = resource_client.add_resource(resource)
        
        # Attach to location
        location_id = location_ids[f"position_{i}"]
        location_client.attach_resource(
            location_id=location_id,
            resource_id=created.resource_id
        )
        
        resource_ids[f"item_{i}"] = created.resource_id
        print(f"  ✓ Created resource: item_{i}")
    
    return resource_ids

def cleanup_resources(location_client, resource_client, location_ids, resource_ids):
    """Delete all created resources and locations."""
    for name, res_id in resource_ids.items():
        resource_client.remove_resource(res_id)
        print(f"  ✓ Deleted resource: {name}")
    
    for name, loc_id in location_ids.items():
        location_client.delete_location(loc_id)
        print(f"  ✓ Deleted location: {name}")
```

##### `helpers/your_experiment/workflows.py`
```python
"""Dynamic workflow generation."""
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

ROBOT_NODE = "Your_Robot_1"

def create_scan_workflow(scan_position):
    """Create workflow to scan workspace."""
    return WorkflowDefinition(
        name="Initial Scan",
        description="Move to scan position and capture image",
        steps=[
            StepDefinition(
                name="move_scan",
                node=ROBOT_NODE,
                action="move_to_position",
                args={"joint_positions": scan_position}
            ),
            StepDefinition(
                name="capture",
                node=ROBOT_NODE,
                action="capture_camera_image",
                args={},
                data_labels={"image": "scan_image"}
            )
        ]
    )

def create_manipulation_workflow(source_location_id, target_location_id):
    """Create workflow to move item from source to target."""
    return WorkflowDefinition(
        name="Move Item",
        description=f"Move item from {source_location_id} to {target_location_id}",
        steps=[
            StepDefinition(
                name="pick",
                node=ROBOT_NODE,
                action="pick_from_location",
                args={"location_id": source_location_id}
            ),
            StepDefinition(
                name="place",
                node=ROBOT_NODE,
                action="place_at_location",
                args={"location_id": target_location_id}
            )
        ]
    )
```

#### 6.3 Create Main Experiment
File: `experiments/your_experiment.py`

```python
#!/usr/bin/env python3
"""Your Experiment Description."""

import os
from madsci.client import (
    ExperimentClient, LocationClient, ResourceClient,
    DataClient, WorkcellClient
)
from madsci.common.types.experiment_types import ExperimentDesign, ExperimentStatus

# Import helpers
from helpers.your_experiment.locations import setup_locations
from helpers.your_experiment.resources import create_resources, cleanup_resources
from helpers.your_experiment.workflows import create_scan_workflow, create_manipulation_workflow
from helpers.your_experiment.snapshots import snapshot_configuration

# Service URLs from environment
EXPERIMENT_URL = os.getenv("EXPERIMENT_SERVER_URL", "http://localhost:8002")
WORKCELL_URL = os.getenv("WORKCELL_SERVER_URL", "http://localhost:8005")
LOCATION_URL = os.getenv("LOCATION_SERVER_URL", "http://localhost:8006")
RESOURCE_URL = os.getenv("RESOURCE_SERVER_URL", "http://localhost:8003")
DATA_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8004")


def main():
    print("=== Your Experiment ===\n")
    
    # State tracking
    experiment = None
    experiment_client = None
    location_client = None
    resource_client = None
    location_ids = {}
    resource_ids = {}
    
    try:
        # Step 1: Create experiment design
        print("Step 1: Creating experiment...")
        experiment_design = ExperimentDesign(
            experiment_name="Your Experiment",
            experiment_description="Description of what this experiment does"
        )
        
        experiment_client = ExperimentClient(EXPERIMENT_URL)
        experiment = experiment_client.start_experiment(
            experiment_design=experiment_design,
            run_name="Run 1",
            run_description="First experimental run"
        )
        print(f"✓ Experiment started: {experiment.experiment_id}\n")
        
        # Step 2: Initialize clients
        print("Step 2: Initializing clients...")
        location_client = LocationClient(LOCATION_URL)
        resource_client = ResourceClient(RESOURCE_URL)
        data_client = DataClient(DATA_URL)
        workcell_client = WorkcellClient(WORKCELL_URL)
        print("✓ Clients initialized\n")
        
        # Step 3: Setup locations
        print("Step 3: Setting up locations...")
        location_ids = setup_locations(location_client)
        print(f"✓ Created {len(location_ids)} locations\n")
        
        # Step 4: Initial scan
        print("Step 4: Scanning workspace...")
        scan_position = [0.0, 0.2, -1.5, -1.5, 0.0]  # Your scan position
        scan_workflow = create_scan_workflow(scan_position)
        scan_result = workcell_client.start_workflow(scan_workflow)
        
        # Retrieve image
        capture_step = [step for step in scan_result.steps if step.name == "capture"][0]
        wrapper_id = capture_step.result.datapoints.json_result
        image_id = data_client.get_datapoint_value(wrapper_id)
        
        # Process image
        import tempfile
        from pathlib import Path
        import cv2
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / "scan.jpg"
        data_client.save_datapoint_value(image_id, str(image_path))
        image_data = cv2.imread(str(image_path))
        
        # Your image processing logic here
        detected_items = process_image(image_data)
        print(f"  Detected {len(detected_items)} items\n")
        
        # Step 5: Create resources
        print("Step 5: Creating resources...")
        resource_ids = create_resources(
            resource_client, location_client, 
            detected_items, location_ids
        )
        print(f"✓ Created {len(resource_ids)} resources\n")
        
        # Step 6: Run experiment logic
        print("Step 6: Running experiment...")
        
        # Example: Move items between locations
        source_loc = location_ids["position_1"]
        target_loc = location_ids["position_2"]
        
        workflow = create_manipulation_workflow(source_loc, target_loc)
        result = workcell_client.start_workflow(workflow)
        
        print("✓ Experiment completed\n")
        
        # Step 7: Snapshot configuration
        print("Step 7: Creating snapshot...")
        snapshot_id = snapshot_configuration(
            data_client, location_client, resource_client,
            location_ids, resource_ids, experiment.experiment_id
        )
        print(f"✓ Snapshot: {snapshot_id}\n")
        
        # Step 8: Cleanup
        print("Step 8: Cleaning up...")
        cleanup_resources(location_client, resource_client, location_ids, resource_ids)
        print("✓ Cleanup complete\n")
        
        # Step 9: End experiment
        print("Step 9: Ending experiment...")
        final_experiment = experiment_client.end_experiment(experiment.experiment_id)
        print(f"✓ Experiment ended\n")
        
        print("✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}\n")
        
        # Cleanup on failure
        if experiment and experiment_client:
            try:
                experiment_client.end_experiment(
                    experiment.experiment_id,
                    status=ExperimentStatus.FAILED
                )
            except:
                pass
        
        if location_ids and resource_ids and location_client and resource_client:
            try:
                cleanup_resources(location_client, resource_client, location_ids, resource_ids)
            except:
                pass
        
        raise


def process_image(image_data):
    """
    Process image to detect items.
    
    Replace this with your actual image processing logic.
    """
    # Example placeholder
    return [
        {"type": "item_a", "position": (100, 100)},
        {"type": "item_b", "position": (200, 200)}
    ]


if __name__ == "__main__":
    main()
```

---

### STEP 7: Common Porting Patterns

#### Pattern 1: Converting Direct Hardware Calls

**Before (Standalone)**:
```python
# Direct hardware control
robot.move_joints([0.0, 0.0, 0.0, 0.0, 0.0])
gripper.close()
robot.move_joints([0.3, -0.5, -0.3, -1.0, 0.0])
gripper.open()
```

**After (MADSci Workflow)**:
```python
# Workflow definition
workflow = WorkflowDefinition(
    name="Pick and Place",
    steps=[
        StepDefinition(
            name="move_home",
            node="Your_Robot_1",
            action="move_to_position",
            args={"joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0]}
        ),
        StepDefinition(
            name="close_gripper",
            node="Your_Robot_1",
            action="close_gripper",
            args={}
        ),
        StepDefinition(
            name="move_pickup",
            node="Your_Robot_1",
            action="move_to_position",
            args={"joint_positions": [0.3, -0.5, -0.3, -1.0, 0.0]}
        ),
        StepDefinition(
            name="open_gripper",
            node="Your_Robot_1",
            action="open_gripper",
            args={}
        )
    ]
)
result = workcell_client.start_workflow(workflow)
```

#### Pattern 2: Converting Vision Processing

**Before (Standalone)**:
```python
# Direct camera access
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
results = detect_objects(frame)
camera.release()
```

**After (MADSci)**:
```python
# Robot captures image
workflow = WorkflowDefinition(
    name="Capture",
    steps=[
        StepDefinition(
            name="capture",
            node="Your_Robot_1",
            action="capture_camera_image",
            args={}
        )
    ]
)
result = workcell_client.start_workflow(workflow)

# Client processes image
capture_step = result.steps[0]
wrapper_id = capture_step.result.datapoints.json_result
image_id = data_client.get_datapoint_value(wrapper_id)

image_path = Path("temp_image.jpg")
data_client.save_datapoint_value(image_id, str(image_path))
frame = cv2.imread(str(image_path))

# Your processing logic
results = detect_objects(frame)
```

#### Pattern 3: Converting Position Management

**Before (Standalone)**:
```python
# Hardcoded positions in main code
PICKUP_POS = [0.3, -0.8, -0.45, -1.35, 0.25]
DROPOFF_POS = [-0.3, -0.8, -0.45, -1.35, -0.25]

robot.move_joints(PICKUP_POS)
```

**After (MADSci)**:
```python
# Location-based positioning
location_ids = setup_locations(location_client)

# Use location ID in workflow
workflow = WorkflowDefinition(
    name="Pick",
    steps=[
        StepDefinition(
            name="pick",
            node="Your_Robot_1",
            action="pick_from_location",
            args={"location_id": location_ids["pickup_position"]}
        )
    ]
)
```

**Benefits**:
- Positions stored centrally in Location Manager
- Easy to update without code changes
- Multi-robot support (different representations per robot)

#### Pattern 4: Converting Loops and Conditionals

**Before (Standalone)**:
```python
for i in range(10):
    robot.move_joints(positions[i])
    
    if sensor.detect():
        gripper.close()
        break
```

**After (MADSci)**:
```python
# Loop in experiment code
for i in range(10):
    # Generate workflow for this iteration
    workflow = WorkflowDefinition(
        name=f"trial_{i}",
        steps=[
            StepDefinition(
                name="move",
                node="Your_Robot_1",
                action="move_to_position",
                args={"joint_positions": positions[i]}
            ),
            StepDefinition(
                name="capture",
                node="Your_Robot_1",
                action="capture_camera_image",
                args={}
            )
        ]
    )
    result = workcell_client.start_workflow(workflow)
    
    # Process result in experiment code
    image_data = retrieve_image(result)
    if detect_condition(image_data):
        # Execute gripper close workflow
        close_workflow = WorkflowDefinition(...)
        workcell_client.start_workflow(close_workflow)
        break
```

**Key Principle**: 
- Loops and decisions live in experiment code
- Each iteration submits a workflow
- Process results on client, make decisions, generate next workflow

---

## Complete Porting Checklist

### Pre-Porting Analysis
- [ ] List all hardware interfaces (robot arm, gripper, camera, sensors)
- [ ] List all distinct actions the robot performs
- [ ] Identify high-level logic vs hardware control
- [ ] Extract all configuration parameters
- [ ] Document position definitions (joint angles, coordinates)
- [ ] Identify image processing / analysis code

### Robot Node Implementation
- [ ] Create node directory structure
- [ ] Create `default.node.yaml` with node metadata
- [ ] Implement config class (`your_robot_config.py`)
  - [ ] Hardware connection parameters
  - [ ] Motion parameters
  - [ ] Predefined positions
  - [ ] Camera/sensor parameters
- [ ] Implement interface class (`your_robot_interface.py`)
  - [ ] Hardware initialization
  - [ ] Movement methods
  - [ ] Gripper methods
  - [ ] Camera capture method
  - [ ] Current state query methods
- [ ] Implement node class (`your_robot_node.py`)
  - [ ] `startup_handler()` - Initialize interface and clients
  - [ ] `shutdown_handler()` - Clean up connections
  - [ ] `state_handler()` - Report current state
  - [ ] Basic actions: move, home, get_position
  - [ ] Gripper actions: open, close
  - [ ] Perception actions: capture_camera_image
  - [ ] Compound actions: pick_from_location, place_at_location
  - [ ] Domain-specific actions (e.g., swap_blocks)
- [ ] Test node standalone
  - [ ] Can connect to hardware
  - [ ] All actions work as expected
  - [ ] Error handling works

### Experiment Implementation
- [ ] Create experiment directory structure
- [ ] Create helper modules
  - [ ] `locations.py` - Define workspace positions
  - [ ] `resources.py` - Resource creation/cleanup
  - [ ] `workflows.py` - Dynamic workflow generation
  - [ ] `algorithms.py` - Experiment-specific logic
  - [ ] `snapshots.py` - Configuration capture
- [ ] Implement main experiment script
  - [ ] Experiment initialization
  - [ ] Client setup
  - [ ] Location setup
  - [ ] Initial scan and resource creation
  - [ ] Experiment loop
  - [ ] Snapshot and cleanup
  - [ ] Error handling
- [ ] Port image processing logic
  - [ ] Extract image from datapoint
  - [ ] Process on client (not robot node)
  - [ ] Use results to make decisions
- [ ] Port algorithmic logic
  - [ ] Keep decision-making in experiment
  - [ ] Generate workflows dynamically
  - [ ] Handle results and iterate

### Testing
- [ ] Test robot node actions individually
- [ ] Test location representations (robot reaches correct positions)
- [ ] Test resource tracking (pick/place updates correctly)
- [ ] Test image capture and retrieval
- [ ] Test complete experiment end-to-end
- [ ] Test error handling (what happens if movement fails?)
- [ ] Test cleanup (resources/locations deleted properly)

### Documentation
- [ ] Document node actions and parameters
- [ ] Document location definitions
- [ ] Document workflow patterns
- [ ] Document image processing requirements
- [ ] Create README for robot node
- [ ] Create README for experiment

---

## Additional Files and Folder Structure

### Complete Directory Structure

```
your-project/
├── robot-nodes/
│   └── your-robot/
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── default.node.yaml
│       │   ├── your_robot_node.py
│       │   ├── your_robot_interface.py
│       │   └── your_robot_config.py
│       ├── docker-compose.yml
│       └── README.md
│
├── clients/
│   └── experiments/
│       ├── your_experiment.py
│       ├── helpers/
│       │   ├── __init__.py
│       │   ├── vision.py                    # Shared vision utilities
│       │   └── your_experiment/
│       │       ├── __init__.py
│       │       ├── locations.py
│       │       ├── resources.py
│       │       ├── workflows.py
│       │       ├── algorithms.py
│       │       └── snapshots.py
│       ├── test_your_experiment.py
│       └── README.md
│
├── madsci-core/
│   └── managers/
│       └── workcell/
│           └── your_workcell.workcell.yaml  # Workcell configuration
│
└── README.md
```

### Workcell Configuration

File: `madsci-core/managers/workcell/your_workcell.workcell.yaml`

```yaml
workcell_name: YourWorkcell
workcell_description: Your experimental workcell configuration

nodes:
  - node_name: Your_Robot_1
    node_url: http://192.168.1.100:2000/
    capabilities:
      - movement
      - manipulation
      - vision

# Optional: Define node groups for multi-robot coordination
node_groups:
  robots:
    - Your_Robot_1
```

### Docker Compose (Optional)

File: `robot-nodes/your-robot/docker-compose.yml`

```yaml
version: '3.8'

services:
  your_robot_node:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: your_robot_node
    network_mode: host
    environment:
      - YOUR_ROBOT_URL=http://192.168.1.100:2000/
      - EXPERIMENT_SERVER_URL=http://localhost:8002
      - RESOURCE_SERVER_URL=http://localhost:8003
      - DATA_SERVER_URL=http://localhost:8004
      - LOCATION_SERVER_URL=http://localhost:8006
    volumes:
      - ./nodes:/app/nodes
    restart: unless-stopped
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Processing Images in Robot Node
**Wrong**:
```python
@action
def capture_and_detect(self) -> dict:
    image = self.robot_interface.capture_camera_image()
    detected = detect_blocks(image)  # ❌ Processing in node
    return {"detected": detected}
```

**Correct**:
```python
# In robot node:
@action
def capture_camera_image(self) -> dict:
    image = self.robot_interface.capture_camera_image()
    # ... store in Data Manager ...
    return datapoint_id  # Return ID only

# In experiment:
image_path = data_client.save_datapoint_value(image_id, local_path)
image = cv2.imread(image_path)
detected = detect_blocks(image)  # ✓ Processing in experiment
```

### Pitfall 2: Hardcoding Robot Node Names
**Wrong**:
```python
workflow = WorkflowDefinition(
    steps=[
        StepDefinition(
            name="move",
            node="DOFBOT_Pro_1",  # ❌ Hardcoded
            action="move_to_position",
            args={}
        )
    ]
)
```

**Correct**:
```python
# In locations.py or workflows.py:
ROBOT_NODE = "DOFBOT_Pro_1"  # Single constant

workflow = WorkflowDefinition(
    steps=[
        StepDefinition(
            name="move",
            node=ROBOT_NODE,  # ✓ Uses constant
            action="move_to_position",
            args={}
        )
    ]
)
```

### Pitfall 3: Not Handling Two-Level Datapoint Retrieval
**Wrong**:
```python
capture_step = result.steps[0]
image_data = capture_step.result  # ❌ This is a wrapper ID
```

**Correct**:
```python
capture_step = result.steps[0]
wrapper_id = capture_step.result.datapoints.json_result  # Get wrapper
image_id = data_client.get_datapoint_value(wrapper_id)   # Get actual ID
data_client.save_datapoint_value(image_id, local_path)   # Retrieve data
```

### Pitfall 4: Not Updating Resource Tracking
**Wrong**:
```python
def pick_from_location(self, location_id: str) -> dict:
    # ... move to location, close gripper ...
    return {"status": "success"}  # ❌ Didn't update tracking
```

**Correct**:
```python
def pick_from_location(self, location_id: str) -> dict:
    # ... move to location, close gripper ...
    resource_id = location.resource_id
    if resource_id:
        self.resource_client.push(self.gripper.resource_id, resource_id)  # ✓ Update tracking
    return {"status": "success", "resource_id": resource_id}
```

### Pitfall 5: Forgetting Cleanup on Failure
**Wrong**:
```python
def main():
    experiment = experiment_client.start_experiment(...)
    locations = setup_locations(...)
    resources = create_resources(...)
    # ... experiment ...
    # ❌ No try/except, resources orphaned on error
```

**Correct**:
```python
def main():
    experiment = None
    locations = {}
    resources = {}
    
    try:
        experiment = experiment_client.start_experiment(...)
        locations = setup_locations(...)
        resources = create_resources(...)
        # ... experiment ...
    except Exception as e:
        # Cleanup on failure
        if experiment:
            experiment_client.end_experiment(experiment.experiment_id, status=FAILED)
        if locations and resources:
            cleanup_resources(location_client, resource_client, locations, resources)
        raise
```

---

## Summary

### Key Principles

1. **Separation of Concerns**
   - Robot node: Hardware control + resource tracking
   - Experiment: High-level logic + data analysis

2. **Resource Tracking**
   - Always update gripper/location resources during pick/place
   - Enables workflow coordination and data provenance

3. **Location-Based Positioning**
   - Store positions in Location Manager, not code
   - Each robot node has its own representation

4. **Datapoint Flow**
   - Nodes produce data (return IDs)
   - Experiments consume data (retrieve and process)

5. **Dynamic Workflows**
   - Generate workflows programmatically
   - Adapt based on state and results

6. **Error Handling**
   - Always clean up resources on failure
   - Use try/except in experiments
   - Return error status in nodes (don't crash)

### Benefits of MADSci Integration

- **Multi-Robot Coordination**: Multiple nodes work together via workflows
- **Data Provenance**: Track which node produced which data, when
- **Experiment Reproducibility**: Workflows and configurations stored in Data Manager
- **Distributed Compute**: Heavy processing on client, not robot node
- **Centralized Resource Management**: One source of truth for locations/resources
- **Error Recovery**: Standardized cleanup and error handling

---

## Next Steps

1. **Start Simple**: Port basic movement actions first
2. **Test Incrementally**: Test each action before moving to next
3. **Add Complexity**: Add perception, then compound actions
4. **Iterate**: Refine based on experimental needs

For questions or issues, refer to:
- MADSci documentation
- Reference implementation (DOFBOT Pro + block permutation experiment)
- MADSci community support channels

