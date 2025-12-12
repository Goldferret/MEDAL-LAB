# Simple Answer: What Runs on the Robot?

## TL;DR

You already have the full MEDAL-LAB repo on the robot via git. **That's perfect!**

**Only these 4 files actually execute on the robot:**

```
~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/
├── dofbot_ros_node.py         [The main script]
├── dofbot_ros_interface.py    [Imported by main script]
├── dofbot_ros_config.py       [Imported by main script]
└── default.node.yaml          [Read by main script]
```

Everything else in the repo just sits there unused (which is fine!).

---

## What You Need to Do

### One-Time Setup on Robot

```bash
# SSH to robot
ssh robot@192.168.1.234

# Create config directory
mkdir -p ~/robot_config

# Create environment file (UPDATE THE IP!)
cat > ~/robot_config/.env << 'EOF'
CENTRAL_COMPUTER_IP=192.168.1.XXX    # Change this!
EXPERIMENT_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8002
RESOURCE_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8003
DATA_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8004
LOCATION_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8006
DOFBOT_PRO_1_URL=http://192.168.1.234:2000/
EOF

# Create startup script
cat > ~/start_robot_node.sh << 'EOF'
#!/bin/bash
export $(cat ~/robot_config/.env | xargs)
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes
python3 dofbot_ros_node.py
EOF

chmod +x ~/start_robot_node.sh

# Install dependencies (REQUIRED!)
pip3 install madsci-client opencv-python numpy

# Verify madsci is installed
python3 -c "import madsci; print('MADSci OK')"
```

### Daily Operation

**On robot:**
```bash
ssh robot@192.168.1.234
~/start_robot_node.sh
```
Leave this running!

**On central:**
```bash
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

---

## What Executes Where

### Robot Computer
**Runs:** `dofbot_ros_node.py`
**Job:** Listen for commands, control hardware
**CPU:** ~5% (just waiting for commands)

### Central Computer  
**Runs:** Experiment scripts, MADSci services
**Job:** Decision making, data processing, orchestration
**CPU:** ~20-40% (doing all the thinking)

---

## File Locations

### On Robot (192.168.1.234)
```
/home/robot/
├── MEDAL-LAB/                    [Full repo via git - good!]
│   └── robot-nodes/.../nodes/    [Only these 4 files run]
├── robot_config/.env             [Your config - create this]
└── start_robot_node.sh           [Startup script - create this]
```

### On Central  
```
~/MEDAL-LAB/
├── madsci-core/                  [Docker services - runs here]
├── clients/experiments/          [Experiments - run here]
└── robot-nodes/                  [Reference only on central]
```

---

## Visual Flow

```
Central Computer                     Robot Computer
┌─────────────────┐                 ┌─────────────────┐
│                 │                 │                 │
│  Experiment     │                 │  dofbot_ros     │
│  Script         │                 │  _node.py       │
│                 │                 │                 │
│  (You run this) │                 │  (Start once,   │
│                 │                 │   leave running)│
└────────┬────────┘                 └────────▲────────┘
         │                                   │
         │  HTTP Commands                    │
         │  "Pick from location X"           │
         └───────────────────────────────────┘
                    Network
         ┌───────────────────────────────────┐
         │                                   │
         ▼                                   │
┌────────┴────────┐                 ┌────────┴────────┐
│                 │                 │                 │
│  Workcell       │                 │  ROS/MoveIT     │
│  Manager        │                 │                 │
│                 │                 │  (Controls      │
│  (Routes        │                 │   motors)       │
│   commands)     │                 │                 │
└─────────────────┘                 └─────────────────┘
```

---

## What You DON'T Need to Do

- Don't delete anything from the robot's MEDAL-LAB repo  
- Don't copy files around  
- Don't run experiments on robot  
- Don't run Docker on robot  
- Don't install experiment dependencies on robot  

---

## Summary

**Current state:** Good!
- Full repo on robot via git
- Easy to update with `git pull`

**What you need:**
- Create `~/robot_config/.env` (update central IP)
- Create `~/start_robot_node.sh` (startup script)
- Run `~/start_robot_node.sh` when needed

**That's it!**

See full details: `docs/Robot_Filesystem_Guide.md`

