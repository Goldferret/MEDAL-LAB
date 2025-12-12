# MADSci Block Combination Solver - Deployment Guide

## System Architecture

### Two-Computer Setup

```
┌─────────────────────────────────────────────────────────────────┐
│                    CENTRAL LINUX COMPUTER                        │
│                  (Workcell Manager Host)                         │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │          MADSci Core Services (Docker)                  │     │
│  │  - Experiment Manager    (port 8002)                    │     │
│  │  - Resource Manager      (port 8003)                    │     │
│  │  - Data Manager          (port 8004)                    │     │
│  │  - Workcell Manager      (port 8005)                    │     │
│  │  - Location Manager      (port 8006)                    │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │          Experiment Client                              │     │
│  │  - block_combination_solver_experiment.py               │     │
│  │  - helpers/block_combination/                           │     │
│  │  - helpers/vision.py                                    │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                   │
└───────────────────────────┬───────────────────────────────────┬─┘
                            │                                   │
                      Network (WiFi/Ethernet)                  │
                            │                                   │
┌───────────────────────────┴─────────────────────────────┐     │
│              ROBOT ARM COMPUTER                          │     │
│           (Onboard computer on DOFBOT Pro)               │     │
│                                                           │     │
│  ┌─────────────────────────────────────────────────┐    │     │
│  │  ROS/MoveIT Stack                                │    │     │
│  │  - ROS Noetic                                    │    │     │
│  │  - MoveIT Commander                              │    │     │
│  │  - /move_group action server                     │    │     │
│  │  - /joint_states topic                           │    │     │
│  │  - Camera driver (/camera/color/image_raw)      │    │     │
│  └─────────────────────────────────────────────────┘    │     │
│                                                           │     │
│  ┌─────────────────────────────────────────────────┐    │     │
│  │  MADSci Robot Node                               │    │     │
│  │  - dofbot_ros_node.py                            │    │     │
│  │  - dofbot_ros_interface.py                       │    │     │
│  │  - dofbot_ros_config.py                          │    │     │
│  │  - default.node.yaml                             │    │     │
│  │                                                   │    │     │
│  │  REST API Server (port 2000)                     │◄───┼─────┘
│  │  Connects to central MADSci services             │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  Hardware:                                                │
│  - DOFBOT Pro robot arm                                  │
│  - Orbbec depth camera                                   │
└───────────────────────────────────────────────────────────┘
```

---

## What Goes Where

### Central Linux Computer (Your Current Machine)

**Already in place** - The repository you have:

```
MEDAL-LAB/
├── madsci-core/                          # Core services
│   ├── docker-compose.yml
│   └── managers/
│
├── clients/
│   └── experiments/
│       ├── block_combination_solver_experiment.py
│       └── helpers/
│
└── docs/
```

**Runs**:
- MADSci core services (via Docker)
- Experiment scripts (Python)
- Data storage and management

### Robot Arm Computer

**What you need to transfer** - Only the robot node files:

```
robot-nodes/
└── dofbot-pro-ros/
    ├── nodes/
    │   ├── dofbot_ros_node.py          # Main node
    │   ├── dofbot_ros_interface.py     # ROS interface
    │   ├── dofbot_ros_config.py        # Configuration
    │   └── default.node.yaml           # Node metadata
    │
    └── requirements.txt                 # Python dependencies
```

**Runs**:
- ROS/MoveIT (already installed by manufacturer)
- MADSci robot node (Python script)
- Camera drivers

---

## Step-by-Step Setup

### PART 1: Central Computer Setup

#### 1.1 Install MADSci Core Services

```bash
# On central Linux computer
cd ~/MEDAL-LAB/madsci-core

# Start all MADSci services
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# experiment-manager    running    0.0.0.0:8002
# resource-manager      running    0.0.0.0:8003
# data-manager          running    0.0.0.0:8004
# workcell-manager      running    0.0.0.0:8005
# location-manager      running    0.0.0.0:8006
```

#### 1.2 Configure Workcell

Edit `madsci-core/managers/workcell/example_wc.workcell.yaml`:

```yaml
workcell_name: BlockSolverWorkcell
workcell_description: 6-block combination solver workcell

nodes:
  - node_name: DOFBOT_Pro_1
    node_url: http://192.168.1.234:2000/    # Your robot's IP
    capabilities:
      - movement
      - manipulation
      - vision
```

**Note**: Using your robot's IP address: `192.168.1.234`

#### 1.3 Install Python Dependencies (if not already done)

```bash
cd ~/MEDAL-LAB/clients/experiments
pip3 install -r requirements.txt

# Typical requirements:
# madsci-client
# opencv-python
# numpy
```

---

### PART 2: Robot Computer Setup

#### 2.1 Connect to Robot Computer

The DOFBOT Pro has an onboard computer (usually Jetson Nano or similar).

**Option A: Direct connection** (keyboard/monitor to robot)
```bash
# Already logged in to robot computer
```

**Option B: SSH from central computer**
```bash
# From central computer
ssh robot@192.168.1.234    # Your robot's IP
# Password: (provided by manufacturer)
```

#### 2.2 Create MADSci Node Directory

```bash
# On robot computer
mkdir -p ~/madsci_node
cd ~/madsci_node
```

#### 2.3 Transfer Robot Node Files

**Option A: Using SCP** (from central computer)
```bash
# From central Linux computer
cd ~/MEDAL-LAB
scp -r robot-nodes/dofbot-pro-ros/nodes robot@192.168.1.234:~/madsci_node/
```

**Option B: Using Git** (on robot computer)
```bash
# On robot computer
cd ~/madsci_node
git clone https://github.com/your-org/MEDAL-LAB.git
cd MEDAL-LAB/robot-nodes/dofbot-pro-ros
```

**Option C: Manual copy via USB**
```bash
# Copy robot-nodes/dofbot-pro-ros/nodes/ to USB drive
# On robot computer:
cp -r /media/usb/nodes ~/madsci_node/
```

#### 2.4 Install Python Dependencies on Robot

```bash
# On robot computer
cd ~/madsci_node/nodes

# Install madsci client
pip3 install madsci-client

# Install other dependencies
pip3 install opencv-python numpy
```

#### 2.5 Create Environment Configuration

Create `~/madsci_node/.env` on robot:

```bash
# On robot computer
cat > ~/madsci_node/.env << 'EOF'
# Central computer IP (MADSci services location)
CENTRAL_COMPUTER_IP=192.168.1.100     # UPDATE THIS!

# MADSci service URLs
EXPERIMENT_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8002
RESOURCE_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8003
DATA_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8004
LOCATION_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8006

# This robot's URL (robot's own IP)
DOFBOT_PRO_1_URL=http://192.168.1.234:2000/
EOF
```

**Important**: Update `CENTRAL_COMPUTER_IP` with your central Linux computer's IP address

#### 2.6 Verify ROS is Running

```bash
# On robot computer
# Check if ROS is running
rostopic list

# Expected output should include:
# /joint_states
# /move_group/...
# /camera/color/image_raw
# (and many others)

# If not running, start ROS/MoveIT (command depends on your setup):
# roslaunch dofbot_pro moveit.launch
```

---

## Running the System

### Startup Sequence

#### Step 1: Start Central Services (Central Computer)

```bash
# On central Linux computer
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d

# Verify all services running
docker-compose ps
```

#### Step 2: Start Robot Node (Robot Computer)

```bash
# On robot computer
cd ~/madsci_node/nodes

# Source environment variables
export $(cat ../.env | xargs)

# Start the robot node
python3 dofbot_ros_node.py
```

**Expected output**:
```
ROS node initialized
Waiting for move_group action server...
Connected to move_group action server
Robot interface initialized
Location client initialized
Gripper resource initialized: 01HN...
Robot node startup complete
Starting REST server on http://192.168.1.234:2000
```

**Leave this running!** The robot node must stay running to accept commands.

#### Step 3: Verify Robot Node Connection (Central Computer)

```bash
# On central Linux computer
# Test robot node is reachable
curl http://192.168.1.234:2000/health

# Expected response:
# {"status": "healthy", "node_name": "DOFBOT_Pro_1"}
```

#### Step 4: Run Experiment (Central Computer)

```bash
# On central Linux computer
cd ~/MEDAL-LAB/clients/experiments

# Set environment variables (if not already set)
export EXPERIMENT_SERVER_URL=http://localhost:8002
export WORKCELL_SERVER_URL=http://localhost:8005
export LOCATION_SERVER_URL=http://localhost:8006
export RESOURCE_SERVER_URL=http://localhost:8003
export DATA_SERVER_URL=http://localhost:8004

# Run the experiment
python3 block_combination_solver_experiment.py
```

**Expected output**:
```
============================================================
6-Block Combination Search Experiment
============================================================

Step 1: Creating experiment design...
✓ Experiment design created

Step 2: Starting experiment...
✓ Experiment started: 01HN...

[Experiment continues...]
```

---

## Running in Background (Optional)

### Robot Node as Service

To keep the robot node running automatically, create a systemd service:

```bash
# On robot computer
sudo nano /etc/systemd/system/madsci-robot-node.service
```

Add this content:

```ini
[Unit]
Description=MADSci Robot Node for DOFBOT Pro
After=network.target roscore.service

[Service]
Type=simple
User=robot
WorkingDirectory=/home/robot/madsci_node/nodes
EnvironmentFile=/home/robot/madsci_node/.env
ExecStart=/usr/bin/python3 /home/robot/madsci_node/nodes/dofbot_ros_node.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable madsci-robot-node
sudo systemctl start madsci-robot-node

# Check status
sudo systemctl status madsci-robot-node

# View logs
journalctl -u madsci-robot-node -f
```

---

## Network Configuration

### Option 1: Both on Same Network (Recommended)

**Setup**:
- Central computer: `192.168.1.100` (update to your actual IP)
- Robot computer: `192.168.1.234` (your robot)
- Both connected to same WiFi/Ethernet

**Configuration**:
- No special routing needed
- Direct communication works

### Option 2: Robot Connects to Central via WiFi

**Setup**:
- Central computer: Wired Ethernet `192.168.1.100` (update to your actual IP)
- Robot computer: WiFi `192.168.1.234` (your robot)

**Configuration**:
- Ensure WiFi and Ethernet on same subnet
- May need to configure router/firewall

### Option 3: Remote Access

**Setup**:
- Central computer: University network
- Robot computer: Lab network
- VPN or port forwarding required

**Configuration**:
- Set up VPN between networks, or
- Configure port forwarding for MADSci services

---

## Troubleshooting

### Robot Node Won't Start

**Error**: `Connection refused to MADSci services`

**Solution**:
```bash
# Check central services are running
curl http://<central-ip>:8002/health

# If not reachable, check firewall
sudo ufw allow 8002:8006/tcp
```

**Error**: `ROS master not found`

**Solution**:
```bash
# On robot computer
# Check ROS is running
rostopic list

# If not, start ROS
roslaunch dofbot_pro moveit.launch
```

**Error**: `Failed to connect to move_group`

**Solution**:
```bash
# Check MoveIT is running
rostopic list | grep move_group

# Restart MoveIT if needed
rosnode kill /move_group
roslaunch dofbot_pro moveit.launch
```

### Experiment Can't Reach Robot

**Error**: `Failed to start workflow - node not found`

**Solution**:
```bash
# 1. Check robot node is running
curl http://<robot-ip>:2000/health

# 2. Check workcell configuration
cat ~/MEDAL-LAB/madsci-core/managers/workcell/example_wc.workcell.yaml
# Verify node_url matches robot's IP

# 3. Restart workcell manager
cd ~/MEDAL-LAB/madsci-core
docker-compose restart workcell-manager
```

### Camera Not Working

**Error**: `Failed to capture image from camera`

**Solution**:
```bash
# On robot computer
# Check camera topic
rostopic echo /camera/color/image_raw --noarr

# If no data, check camera driver
lsusb | grep Orbbec

# Restart camera driver
rosnode kill /camera_driver
roslaunch orbbec_camera astra.launch
```

### Network Timeout Issues

**Symptoms**: Workflows start but timeout

**Solution**:
```bash
# Increase timeouts in dofbot_ros_interface.py
# Edit on robot computer:
nano ~/madsci_node/nodes/dofbot_ros_interface.py

# Find line:
finished = self.move_client.wait_for_result(rospy.Duration(30.0))

# Increase to:
finished = self.move_client.wait_for_result(rospy.Duration(60.0))
```

---

## Complete Startup Script

### On Central Computer

Create `~/start_madsci.sh`:

```bash
#!/bin/bash
echo "Starting MADSci Core Services..."
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d
sleep 5

echo "Checking service health..."
for port in 8002 8003 8004 8005 8006; do
    curl -s http://localhost:$port/health || echo "Service on port $port not ready"
done

echo ""
echo "✓ MADSci services started"
echo "Next: Start robot node on robot computer, then run experiments"
```

Make executable:
```bash
chmod +x ~/start_madsci.sh
```

### On Robot Computer

Create `~/start_robot_node.sh`:

```bash
#!/bin/bash
echo "Starting MADSci Robot Node..."
cd ~/madsci_node/nodes

# Load environment
export $(cat ../.env | xargs)

# Check ROS is running
if ! rostopic list > /dev/null 2>&1; then
    echo "Error: ROS not running. Start ROS first."
    exit 1
fi

# Start robot node
python3 dofbot_ros_node.py
```

Make executable:
```bash
chmod +x ~/start_robot_node.sh
```

---

## Quick Reference

### Daily Operation

**1. Morning startup:**
```bash
# Central computer
~/start_madsci.sh

# Robot computer (via SSH or direct)
ssh robot@192.168.1.77
~/start_robot_node.sh
```

**2. Run experiment:**
```bash
# Central computer
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

**3. Evening shutdown:**
```bash
# Robot computer - Ctrl+C to stop node

# Central computer
cd ~/MEDAL-LAB/madsci-core
docker-compose down
```

### Important IP Addresses

**Update these for your setup:**

| Component | IP Address | Port | Update Location |
|-----------|-----------|------|-----------------|
| Central Computer | `192.168.1.100` (update yours) | - | Robot's `.env` file |
| Robot Computer | `192.168.1.234` (your robot) | 2000 | Workcell YAML, robot's `.env` |
| Experiment Manager | Central IP | 8002 | Automatic |
| Workcell Manager | Central IP | 8005 | Automatic |

### File Locations Summary

| File | Central Computer | Robot Computer |
|------|------------------|----------------|
| MADSci Services | ✓ (Docker) | - |
| Experiments | ✓ | - |
| Helpers | ✓ | - |
| Robot Node | - | ✓ |
| ROS/MoveIT | - | ✓ (pre-installed) |

---

## Testing the Setup

### Test 1: Services Health Check

```bash
# On central computer
curl http://localhost:8002/health  # Experiment Manager
curl http://localhost:8005/health  # Workcell Manager
curl http://192.168.1.234:2000/health  # Robot Node
```

All should return `{"status": "healthy", ...}`

### Test 2: Robot Movement Test

```bash
# On central computer
cd ~/MEDAL-LAB/clients/experiments

# Create test script
cat > test_robot.py << 'EOF'
from madsci.client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

workcell = WorkcellClient("http://localhost:8005")

workflow = WorkflowDefinition(
    name="Test Movement",
    steps=[
        StepDefinition(
            name="home",
            node="DOFBOT_Pro_1",
            action="home_robot",
            args={}
        )
    ]
)

result = workcell.start_workflow(workflow)
print(f"Result: {result.status}")
EOF

python3 test_robot.py
# Should output: Result: completed
```

### Test 3: Full Experiment (Dry Run)

```bash
# Modify MAX_ATTEMPTS to 1 for quick test
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
# Should complete 1 attempt successfully
```

---

## Summary

### What Runs Where

**Central Computer**:
- ✓ MADSci core services (Docker containers)
- ✓ Experiment scripts
- ✓ Data storage
- ✓ Decision making and orchestration

**Robot Computer**:
- ✓ MADSci robot node (Python script)
- ✓ ROS/MoveIT stack
- ✓ Hardware drivers
- ✓ Takes commands from central computer

### Communication Flow

```
Experiment (Central)
    ↓ HTTP
Workcell Manager (Central)
    ↓ HTTP
Robot Node (Robot) ← REST API on port 2000
    ↓ ROS
Hardware (Robot)
```

### You're Ready When...

- ✓ Central services all show "healthy"
- ✓ Robot node starts without errors
- ✓ Test movement works
- ✓ Experiment can reach robot node

Now you can run the full block combination solver experiment!

