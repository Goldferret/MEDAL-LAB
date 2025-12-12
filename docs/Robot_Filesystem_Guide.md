# Robot Filesystem Guide - What Runs Where

## Current Situation

You have the **entire MEDAL-LAB repository** on the robot computer via git. This is actually fine! But only a tiny portion of it will actually execute on the robot.

---

## What the Robot Actually Runs

### Files That Execute on Robot

**Only these 4 files run on the robot:**

```
MEDAL-LAB/
└── robot-nodes/
    └── dofbot-pro-ros/
        └── nodes/
            ├── dofbot_ros_node.py         ← RUNS ON ROBOT
            ├── dofbot_ros_interface.py    ← RUNS ON ROBOT
            ├── dofbot_ros_config.py       ← RUNS ON ROBOT
            └── default.node.yaml          ← RUNS ON ROBOT
```

**That's it!** Just these 4 files totaling ~800 lines of code.

---

## What the Robot Does NOT Run

### Files That Stay on Robot But Don't Execute

Everything else in the repo can be there but won't run:

```
MEDAL-LAB/
├── madsci-core/                    [Runs on CENTRAL, not robot]
│   ├── docker-compose.yml
│   └── managers/
│
├── clients/                        [Runs on CENTRAL, not robot]
│   └── experiments/
│       ├── block_combination_solver_experiment.py
│       └── helpers/
│
├── docs/                           [Just documentation]
│   ├── Deployment_Guide.md
│   └── ...
│
├── reference-scripts/              [Reference only]
│
└── robot-nodes/
    ├── dofbot-pro-archived/        [Archived, not used]
    └── dofbot-pro-ros/
        ├── docker-compose.yml      [Not needed (no Docker on robot)]
        ├── Dockerfile              [Not needed]
        ├── README.md               [Just documentation]
        └── nodes/                  [THIS IS THE ONLY DIRECTORY USED]
```

---

## Recommended Setup Options

### Option 1: Keep Full Repo (Simplest)

**Current state: Full MEDAL-LAB repo on robot**

**Advantages:**
- Easy to update with `git pull`
- Can reference docs on robot
- Consistent with central computer
- No risk of missing dependencies

**Disadvantages:**
- Uses ~50MB extra disk space (negligible)
- Cluttered directory structure

**What to do:**
```bash
# On robot computer
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes

# Create startup script
cat > ~/start_robot_node.sh << 'EOF'
#!/bin/bash
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes
export $(cat ~/robot_env/.env | xargs)
python3 dofbot_ros_node.py
EOF

chmod +x ~/start_robot_node.sh
```

**Run it:**
```bash
~/start_robot_node.sh
```

**RECOMMENDED** - Easiest to maintain with git

---

### Option 2: Minimal Install (Clean but manual)

**Only copy the 4 files needed**

**Advantages:**
- Clean filesystem
- Clear what's running
- Minimal disk usage

**Disadvantages:**
- Must manually sync when files change
- Can't use git pull
- More maintenance work

**Setup:**
```bash
# On robot computer
mkdir -p ~/madsci_node
cd ~/madsci_node

# Copy just the 4 files from existing repo
cp ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py .
cp ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_interface.py .
cp ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_config.py .
cp ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/default.node.yaml .

# Create env file
mkdir -p ~/madsci_node/.env_dir
nano ~/madsci_node/.env_dir/.env
# (add environment variables)
```

**NOT RECOMMENDED** - More work to maintain

---

## What You Should Do

### Recommended Approach: Keep Full Repo + Simple Script

Since you already have the full repo via git, just create a simple startup script:

```bash
# On robot computer (via SSH)
ssh robot@192.168.1.234

# Create environment directory
mkdir -p ~/robot_config

# Create environment file
cat > ~/robot_config/.env << 'EOF'
# UPDATE THIS LINE with your central computer IP!
CENTRAL_COMPUTER_IP=192.168.1.XXX

EXPERIMENT_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8002
RESOURCE_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8003
DATA_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8004
LOCATION_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8006
DOFBOT_PRO_1_URL=http://192.168.1.234:2000/
EOF

# Create startup script
cat > ~/start_robot_node.sh << 'EOF'
#!/bin/bash
echo "Starting MADSci Robot Node..."
echo ""

# Load environment variables
export $(cat ~/robot_config/.env | xargs)

# Navigate to node directory
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes

# Check ROS is running
if ! rostopic list > /dev/null 2>&1; then
    echo "❌ Error: ROS not running!"
    echo "Please start ROS/MoveIT first"
    exit 1
fi

echo "✓ ROS is running"
echo "✓ Starting robot node..."
echo ""

# Start the node
python3 dofbot_ros_node.py
EOF

chmod +x ~/start_robot_node.sh
```

**Update the central computer IP:**
```bash
nano ~/robot_config/.env
# Change 192.168.1.XXX to your central computer's actual IP
```

**To run:**
```bash
~/start_robot_node.sh
```

---

## File Execution Summary

### On Robot Computer

**What runs:**
```python
# This is the ONLY Python script that executes:
python3 ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py

# Which imports:
from dofbot_ros_interface import DofbotRosInterface
from dofbot_ros_config import DofbotRosConfig
# And reads: default.node.yaml
```

**What it does:**
1. Starts a REST API server on port 2000
2. Connects to ROS/MoveIT (already running on robot)
3. Connects to MADSci services on central computer
4. Waits for commands from central computer
5. Translates commands to ROS messages

**What does NOT run on robot:**
- ❌ Experiment scripts (`block_combination_solver_experiment.py`)
- ❌ Helper modules (`helpers/block_combination/`)
- ❌ MADSci core services (these are Docker containers on central)
- ❌ Vision processing (happens on central after image transfer)

### On Central Computer

**What runs:**
```python
# Experiment scripts run here:
python3 ~/MEDAL-LAB/clients/experiments/block_combination_solver_experiment.py

# Which imports:
from helpers.block_combination.locations import setup_locations
from helpers.block_combination.workflows import create_scan_workflow
# etc... all the helper modules

# And uses:
from madsci.client import WorkcellClient, LocationClient, etc.
```

**Plus:**
- Docker containers with all MADSci services
- Data storage and processing
- Workflow orchestration

---

## Dependencies on Robot

### Python Packages Needed on Robot

```bash
# On robot computer:
pip3 install madsci-client opencv-python numpy

# That's it! Just 3 packages for the robot node
```

### Python Packages NOT Needed on Robot

The robot does NOT need packages used by experiments:
- ❌ scikit-image (used by vision processing on central)
- ❌ pandas (used by experiment analysis on central)
- ❌ matplotlib (used by visualization on central)
- ❌ Any experiment-specific packages

---

## Updating the Robot Code

### If Using Full Repo (Recommended)

```bash
# On robot computer
cd ~/MEDAL-LAB
git pull

# Restart robot node
# Ctrl+C in terminal running the node
~/start_robot_node.sh
```

✅ Easy and clean!

### If Using Minimal Install

```bash
# From central computer
cd ~/MEDAL-LAB
scp robot-nodes/dofbot-pro-ros/nodes/*.py robot@192.168.1.234:~/madsci_node/
scp robot-nodes/dofbot-pro-ros/nodes/*.yaml robot@192.168.1.234:~/madsci_node/

# Then restart node on robot
```

⚠️ Manual and error-prone

---

## Disk Space Analysis

**Full MEDAL-LAB repo:** ~50MB
- `robot-nodes/`: 2MB (code)
- `clients/`: 5MB (experiments + helpers)
- `madsci-core/`: 1MB (configs)
- `docs/`: 2MB (documentation)
- `.git/`: 40MB (git history)

**Minimal install:** ~0.1MB
- Just 4 Python/YAML files

**Verdict:** Even on embedded systems, 50MB is trivial. Keep the full repo!

---

## Directory Structure on Robot

### Current Setup (Full Repo)
```
/home/robot/
├── MEDAL-LAB/                          [Full git repo]
│   └── robot-nodes/
│       └── dofbot-pro-ros/
│           └── nodes/                  [The only directory that matters]
│               ├── dofbot_ros_node.py
│               ├── dofbot_ros_interface.py
│               ├── dofbot_ros_config.py
│               └── default.node.yaml
│
├── robot_config/                       [Separate config directory]
│   └── .env                            [Environment variables]
│
└── start_robot_node.sh                 [Startup script]
```

### What You Actually Access
```bash
# You run this:
~/start_robot_node.sh

# Which executes this:
~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py

# Using config from:
~/robot_config/.env
```

---

## Testing Your Setup

### Verify Correct Files Are Running

```bash
# On robot, while node is running:
ps aux | grep dofbot_ros_node.py

# Should show:
# python3 /home/robot/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py
```

### Verify No Experiments Running on Robot

```bash
# These should NOT be running on robot:
ps aux | grep experiment.py
ps aux | grep docker

# Should return nothing (or just the grep command itself)
```

### Verify Network Connections

```bash
# Robot node should be listening on port 2000:
sudo netstat -tlnp | grep 2000

# Should show:
# tcp  0  0  0.0.0.0:2000  0.0.0.0:*  LISTEN  12345/python3
```

---

## Quick Reference

### What You Need to Know

| Component | Where It Lives | Where It Runs |
|-----------|---------------|---------------|
| Robot node files | Robot filesystem | Robot computer |
| Experiment scripts | Central filesystem | Central computer |
| Helper modules | Central filesystem | Central computer |
| MADSci services | Central Docker | Central computer |
| Vision processing | Central filesystem | Central computer |
| Data storage | Central Docker volumes | Central computer |
| Workflow orchestration | Central Docker | Central computer |

### What You Need to Run on Robot

```bash
# Only this one command:
~/start_robot_node.sh

# Which runs:
python3 ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/dofbot_ros_node.py
```

That's literally it! Everything else runs on central computer.

---

## Summary

### Current Setup is Good!

Having the full MEDAL-LAB repo on the robot via git is actually the **recommended approach** because:

1. Easy to update (`git pull`)
2. Consistent with central computer
3. Documentation available on robot
4. No risk of version mismatches
5. Negligible disk space usage

### What You Need to Do

1. **Create environment file**: `~/robot_config/.env`
2. **Create startup script**: `~/start_robot_node.sh`
3. **Update central computer IP** in the env file
4. **Run the startup script**: `~/start_robot_node.sh`

### What You DON'T Need to Do

1. Delete anything from the repo
2. Copy files around
3. Install experiment dependencies on robot
4. Run Docker on robot
5. Run any experiment scripts on robot

### The Robot's Job

The robot computer has **ONE job**:
- Run `dofbot_ros_node.py` which listens for commands and controls hardware
- Everything else (decisions, data, orchestration) happens on central computer

---

## Final Recommendation

**Keep your current setup** (full repo via git) and just:

1. Create `~/robot_config/.env` with network settings
2. Create `~/start_robot_node.sh` startup script  
3. Run `~/start_robot_node.sh` when you need the robot

Simple, maintainable, and follows best practices!

