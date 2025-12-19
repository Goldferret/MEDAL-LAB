# ROS-Based DOFBOT Pro Robot Node

MADSci-compliant robot node using ROS and MoveIT for DOFBOT Pro control.

## Architecture

**Host (Jetson):** Runs ROS Master, MoveIT, arm_driver, camera drivers  
**Container:** Runs MADSci node that communicates with host ROS services via action clients

This separation keeps hardware drivers on the host while providing a clean MADSci interface. The container uses:
- **Action clients** for MoveIT communication (pure Python, no C++ dependencies)
- **Message package mounting** from host for MD5 compatibility
- **Host networking** for seamless ROS communication
- **Auto-reconnection** to MADSci services (survives MADSci restarts)

## Prerequisites

- NVIDIA Jetson Orin Nano with DOFBOT Pro
- ROS Noetic installed on host
- DOFBOT Pro ROS packages (MoveIT, arm_driver)
- Docker and Docker Compose (V2)

### Installing Docker Compose V2 on Jetson

```bash
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.30.3/docker-compose-linux-aarch64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
docker compose version  # Verify installation
```

## Quick Start

Start ROS services and the robot node on the Jetson:

```bash
# 1. Start all ROS services in tmux session (~100 seconds)
make ros-up

# 2. Start MADSci robot node container
make robot-up NODE=dofbot-pro-ros
```

**View ROS service logs (optional):**
```bash
make ros-attach  # Attach to tmux session
make ros-status  # Check ROS service status
```

**tmux commands while attached:**
- `Ctrl+b` then `d` - Detach (services keep running)
- `Ctrl+b` then `0-3` - Switch to window (0=moveit, 1=arm_driver, 2=sim_bridge, 3=camera)
- `Ctrl+b` then `n` - Next window
- `Ctrl+b` then `p` - Previous window

**Verify ROS is running:**
```bash
rostopic list  # Should show topics like /joint_states, /move_group/*
rosnode list   # Should show /move_group, /Arm_Driver_Node_*
```

**Stop services:**
```bash
make robot-down NODE=dofbot-pro-ros  # Stop robot node
make ros-down                         # Stop ROS services
```

## Container Setup

The container connects to the host's ROS services via network and auto-starts the MADSci node.

### Environment Variables

Configure in two files:

**1. `../../.env.global` (MADSci network configuration):**
```bash
# Define this robot node's IP address in the Robot Node(s) section
DOFBOT_PRO_1_HOST=192.168.1.77
DOFBOT_PRO_1_PORT=2000
DOFBOT_PRO_1_NAME=DOFBOT_Pro_1
```

**2. `.env` (ROS and node configuration):**
```bash
# MADSci Node URL (bind to all interfaces for external access)
ROBOT_NODE_URL=http://0.0.0.0:2000

# ROS Configuration
ROS_MASTER_URI=http://localhost:11311
ROS_DISTRO=noetic
```

## Development Workflow

### Normal Operation
```bash
make ros-up                        # Start ROS services
make robot-up NODE=dofbot-pro-ros  # Start robot node
make robot-logs NODE=dofbot-pro-ros # Watch logs
```

### Code Changes
```bash
# 1. Edit code in ./nodes/ on host
# 2. Restart to pick up changes
make robot-restart NODE=dofbot-pro-ros
make robot-logs NODE=dofbot-pro-ros  # Watch restart
```

### Debugging
```bash
# Stop auto-start temporarily
make robot-down NODE=dofbot-pro-ros

# Edit docker-compose.yml: change command to 'tail -f /dev/null'
cd robot-nodes/dofbot-pro-ros
docker compose --env-file=../../.env.global up -d

# Run manually with debugging
docker exec -it dofbot-pro-ros-node bash
/opt/conda/envs/rosenv/bin/python -u nodes/dofbot_ros_node.py
```

### Expected Startup Output
```
INFO: Uvicorn running on http://0.0.0.0:2000
INFO: ROS node initialized
INFO: Connected to move_group action server
INFO: Robot interface initialized
INFO: Gripper resource initialized
INFO: DOFBOT Pro robot node startup complete
```

**Verify node is ready:**
```bash
curl http://192.168.1.77:2000/status
# Should return: {"ready": true, "errored": false, ...}
```

## Code Structure

```
nodes/
├── dofbot_ros_config.py      # MoveIT parameters, robot positions
├── dofbot_ros_interface.py   # ROS/MoveIT action client wrapper
└── dofbot_ros_node.py         # MADSci node (actions, lifecycle)
```

**Key components:**
- **Config:** MoveIT planning parameters, home position
- **Interface:** Action client for move_group, joint state subscriber with spinner thread
- **Node:** MADSci actions (move_to_position, home_robot, get_current_position)

## Available Actions

### Basic Actions

- **`move_to_position`** - Move robot to specified joint positions (5 angles in radians)
- **`home_robot`** - Move robot to home position
- **`get_current_position`** - Query current joint positions

### Gripper Actions

- **`open_gripper`** - Open gripper to release objects
- **`close_gripper`** - Close gripper to grasp objects

### Vision Actions

- **`capture_camera_image`** - Capture image from camera and store in Data Manager

### Compound Actions

- **`swap_blocks`** - Swap blocks between two locations using a temporary location (requires location IDs)

### Testing Basic Actions

```bash
# Home robot
curl -X POST http://192.168.1.77:2000/action/home_robot \
  -H "Content-Type: application/json" -d '{"args": {}}'

# Move to position (radians)
curl -X POST http://192.168.1.77:2000/action/move_to_position \
  -H "Content-Type: application/json" \
  -d '{"args": {"joint_positions": [0.1, 0.1, 0.1, 0.1, 0.1]}}'

# Get current position
curl -X POST http://192.168.1.77:2000/action/get_current_position \
  -H "Content-Type: application/json" -d '{"args": {}}'
```

**Note:** For advanced actions (gripper, camera, swap), see the block permutation experiment in `../../clients/experiments/` which demonstrates these actions in complete workflows.

## Resource Tracking

The node creates a gripper slot resource (`dofbot_gripper_<node_name>`) that tracks what the robot is holding. Workflows can:
- Query gripper contents via Resource Manager
- Push/pop items to track pick/place operations
- Monitor gripper state in real-time

## Technical Details

### ROS Integration
- **Action Client:** Pure Python communication with move_group
- **Spinner Thread:** Background thread processes ROS callbacks for joint state subscriber
- **Message Compatibility:** Host's ROS message packages mounted into container for MD5 hash matching
- **Networking:** `ROS_HOSTNAME=localhost` with `network_mode: host` for seamless communication

### Joint Names
Robot uses: `Arm1_Joint`, `Arm2_Joint`, `Arm3_Joint`, `Arm4_Joint`, `Arm5_Joint` (from URDF)

### Container Architecture
- **Base:** Python 3.8 slim (minimal Debian)
- **Conda:** Python 3.9 environment with rospy/actionlib from RoboStack
- **Mounts:** Host ROS message packages + application code
- **User:** Runs as `madsci` user (matches host UID/GID)
- **Name:** `dofbot-pro-ros-node`

## Troubleshooting

**ROS services not running:**
- Start services: `make ros-up` (~100 seconds)
- Check status: `make ros-status`
- View logs: `make ros-attach`
- Verify topics: `rostopic list` (should show /joint_states, /move_group/*)

**Container exits immediately:**
- Check logs: `make robot-logs NODE=dofbot-pro-ros`
- Verify ROS services are running: `make ros-status`
- Ensure `ROBOT_NODE_URL` is set in `.env`

**"No joint state received yet" errors:**
- Verify `arm_driver.py` is running: `make ros-attach` (check arm_driver window)
- Check: `rostopic hz /joint_states` (should show ~10 Hz)
- Restart ROS services: `make ros-down && make ros-up`

**MoveIT planning fails:**
- Check that `demo.launch` is running: `make ros-attach` (check moveit window)
- Verify joint names match URDF (Arm1_Joint, etc.)
- Ensure joint positions are within limits
- Note: MoveIT takes ~90 seconds to fully initialize

**"Connection refused" on port 2000:**
- Verify node started: `make robot-logs NODE=dofbot-pro-ros`
- Check `ROBOT_NODE_URL=http://0.0.0.0:2000` in `.env`
- Ensure no firewall blocking port 2000

**MD5 hash mismatch errors:**
- Verify message package mounts in docker-compose.yml
- Check that host has ROS Noetic installed at `/opt/ros/noetic`

**Node shows "errored: true":**
- Usually means MADSci services are unavailable
- Verify MADSci core is running: `make madsci-logs`
- Node will auto-recover when services come back online

## Related

- **MADSci Core:** `../../madsci-core/` - Framework services
- **Clients:** `../../clients/` - Submit workflows  
- **Archived:** `../dofbot-pro-archived/` - Original hardware-level implementation
