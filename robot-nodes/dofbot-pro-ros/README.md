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

## Host Setup

Start ROS services on the Jetson **before** starting the container:

```bash
# Terminal 1: ROS Master
roscore

# Terminal 2: MoveIT simulator
roslaunch dofbot_pro_config demo.launch

# Terminal 3: Arm driver (publishes joint states)
rosrun dofbot_pro_info arm_driver.py

# Terminal 4: Simulation bridge
rosrun arm_moveit_demo SimulationToMachine.py

# Optional: Camera
roslaunch orbbec_camera dabai_dcw2.launch
```

**Verify ROS is running:**
```bash
rostopic list  # Should show topics like /joint_states, /move_group/*
rosnode list   # Should show /move_group, /Arm_Driver_Node_*
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

### Starting the Node

All robot node operations are managed through the root Makefile.

**Start container (auto-starts node):**
```bash
cd /path/to/medal-lab
make robot-up
```

**Stop container:**
```bash
make robot-down
```

**Restart container:**
```bash
make robot-restart
```

**Watch logs:**
```bash
make robot-logs
```

**Expected startup output:**
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

## Development Workflow

### Normal Operation
```bash
make robot-up              # Start container + node
make robot-logs            # Watch logs
```

### Code Changes
```bash
# 1. Edit code in ./nodes/ on host
# 2. Restart to pick up changes
make robot-restart
make robot-logs            # Watch restart
```

### Debugging
```bash
# Stop auto-start temporarily
make robot-down

# Edit docker-compose.yml: change command to 'tail -f /dev/null'
cd robot-nodes/dofbot-pro-ros
docker compose --env-file=../../.env.global up -d

# Run manually with debugging
docker exec -it dofbot-pro-ros-node bash
/opt/conda/envs/rosenv/bin/python -u nodes/dofbot_ros_node.py
```

### If Node Crashes
Container will exit (no auto-restart). Check logs, fix issue, then:
```bash
make robot-up  # Restart container + node
```

## Auto-Reconnection

The robot node automatically reconnects to MADSci services if they restart:
- Registers resources with Resource Manager on startup
- Re-registers if connection is lost
- Survives `make madsci-restart` without manual intervention
- Transitions from error state to ready state automatically

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

Test actions via HTTP:

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

**Container exits immediately:**
- Check logs: `make robot-logs`
- Verify ROS services are running on host
- Ensure `ROBOT_NODE_URL` is set in `.env`

**"No joint state received yet" errors:**
- Verify `arm_driver.py` is running and publishing to `/joint_states`
- Check: `rostopic hz /joint_states` (should show ~10 Hz)
- Ensure `ROS_HOSTNAME=localhost` in docker-compose.yml

**MoveIT planning fails:**
- Check that `demo.launch` is running
- Verify joint names match URDF (Arm1_Joint, etc.)
- Check MoveIT logs for planning errors
- Ensure joint positions are within limits

**"Connection refused" on port 2000:**
- Verify node started: `make robot-logs`
- Check `ROBOT_NODE_URL=http://0.0.0.0:2000` in `.env`
- Ensure no firewall blocking port 2000

**MD5 hash mismatch errors:**
- Verify message package mounts in docker-compose.yml
- Check that host has ROS Noetic installed at `/opt/ros/noetic`

**Node shows "errored: true":**
- Usually means MADSci services are unavailable
- Verify MADSci core is running: `make madsci-logs`
- Node will auto-recover when services come back online

## Next Steps

- Add gripper control actions (open/close)
- Integrate camera via ROS topics for vision
- Create pick/place workflows
- Add collision avoidance
- Implement trajectory recording/playback

## Related

- **MADSci Core:** `../../madsci-core/` - Framework services
- **Clients:** `../../clients/` - Submit workflows  
- **Archived:** `../dofbot-pro-archived/` - Original hardware-level implementation
