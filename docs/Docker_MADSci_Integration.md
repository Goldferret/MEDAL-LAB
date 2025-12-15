# Integrating with Docker-Based MADSci Environment

## Your Current Setup

You have a MADSci Docker container accessed via the `madsci` alias:

```bash
madsci is aliased to `xhost +local:docker >/dev/null; 
    docker run --privileged -it --rm 
    --net=host 
    -e DISPLAY=$DISPLAY 
    -e QT_X11_NO_MITSHM=1 
    -v /tmp/.X11-unix:/tmp/.X11-unix 
    --mount type=bind,src="/home/gabriel/madsci/madsci-demo",dst="/home/madsci/madsci-demo" 
    ghcr.io/ad-sdl/madsci /bin/bash'
```

This means MADSci is pre-installed in a Docker container, not directly on your system.

---

## Quick Fix: Install OpenCV in Container (Temporary)

If you get `ModuleNotFoundError: No module named 'cv2'`, you need to install OpenCV.

**Temporary fix (inside container):**
```bash
# Inside madsci container
pip3 install opencv-python-headless numpy

# Then run experiment
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

**Note:** This is lost when you exit the container (--rm flag). See [Permanent Solution](#permanent-solution-custom-docker-image) below.

---

## How Our Work Fits In

### Two-Part Architecture

**1. Robot Node (Runs on HOST - Outside Container)**
- Location: `~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes/`
- Needs direct ROS access
- Runs directly on robot computer (NOT in Docker)

**2. Experiment Scripts (Run INSIDE Container)**
- Location: `~/MEDAL-LAB/clients/experiments/`
- Need MADSci Python libraries
- Should run inside the `madsci` Docker container

---

## Updated Setup Instructions

### Step 1: Mount MEDAL-LAB in MADSci Container

Modify your `madsci` alias to include your MEDAL-LAB directory:

```bash
# Edit your ~/.bashrc or ~/.zshrc
nano ~/.bashrc

# Find the madsci alias and add another mount for MEDAL-LAB:
alias madsci='
    xhost +local:docker >/dev/null;
    docker run --privileged -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,src="/home/gabriel/madsci/madsci-demo",dst="/home/madsci/madsci-demo" \
    --mount type=bind,src="/home/gabriel/Documents/medal-lab-bill/MEDAL-LAB",dst="/home/madsci/MEDAL-LAB" \
    ghcr.io/ad-sdl/madsci /bin/bash'

# Save and reload
source ~/.bashrc
```

**Key change:** Added mount for your MEDAL-LAB directory so experiments can access it inside the container.

### Step 2: Start MADSci Core Services (Still on Host)

The MADSci core services (Experiment Manager, Workcell Manager, etc.) likely run in separate Docker containers:

```bash
# On central computer (outside madsci container)
cd ~/Documents/medal-lab-bill/MEDAL-LAB/madsci-core
docker compose up -d
```

These services need to run on the host so both the container and robot node can reach them.

### Step 3: Start Robot Node (On Robot - Outside Container)

The robot node must run directly on the robot (NOT in Docker) because it needs direct access to ROS and hardware:

```bash
# SSH to robot
ssh jetson@ubuntu

# Install MADSci client ON THE ROBOT HOST (not in container)
pip3 install --user madsci-client opencv-python numpy

# Create startup script (if not already done)
cat > ~/start_robot_node.sh << 'EOF'
#!/bin/bash
export $(cat ~/robot_config/.env | xargs)
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes
python3 dofbot_ros_node.py
EOF

chmod +x ~/start_robot_node.sh

# Run it
~/start_robot_node.sh
```

### Step 4: Run Experiments (Inside MADSci Container)

Experiments should run inside the MADSci Docker container:

```bash
# On central computer
# Enter MADSci container
madsci

# Now you're INSIDE the container
# Navigate to MEDAL-LAB (mounted from host)
cd ~/MEDAL-LAB/clients/experiments

# Run experiment
python3 block_combination_solver_experiment.py
```

---

## Complete Workflow

### On Central Computer

**Terminal 1: Start MADSci Services**
```bash
# Outside container
cd ~/Documents/medal-lab-bill/MEDAL-LAB/madsci-core
docker compose up -d

# Verify services
curl http://localhost:8002/health
curl http://localhost:8005/health
```

**Terminal 2: Enter MADSci Container and Run Experiment**
```bash
# Enter container
madsci

# Inside container - verify MADSci is available
python3 -c "import madsci; print('MADSci OK')"

# Navigate to experiments
cd ~/MEDAL-LAB/clients/experiments

# Run experiment
python3 block_combination_solver_experiment.py
```

### On Robot Computer

**Terminal 3: Run Robot Node**
```bash
# SSH to robot
ssh jetson@ubuntu

# Start robot node (runs on host, NOT in container)
~/start_robot_node.sh
```

---

## Why This Architecture?

```
┌─────────────────────────────────────────────────────────┐
│              Central Computer                            │
│                                                           │
│  ┌──────────────────────────────┐                        │
│  │   MADSci Docker Container    │                        │
│  │   (via 'madsci' alias)       │                        │
│  │                              │                        │
│  │  - Experiment scripts        │                        │
│  │  - Helper modules            │                        │
│  │  - MADSci clients            │                        │
│  │                              │                        │
│  │  Mounted: ~/MEDAL-LAB        │                        │
│  └──────────┬───────────────────┘                        │
│             │                                             │
│             │ HTTP (localhost)                            │
│             ↓                                             │
│  ┌──────────────────────────────┐                        │
│  │  MADSci Services (Separate   │                        │
│  │  Docker Compose)             │                        │
│  │  - Experiment Manager        │                        │
│  │  - Workcell Manager          │                        │
│  │  - etc.                      │                        │
│  └──────────┬───────────────────┘                        │
└─────────────┼───────────────────────────────────────────┘
              │
              │ Network
              │
┌─────────────┼───────────────────────────────────────────┐
│             │         Robot Computer                      │
│             ↓                                             │
│  ┌──────────────────────────────┐                        │
│  │  Robot Node (On Host)        │                        │
│  │  NOT in Docker!              │                        │
│  │                              │                        │
│  │  - dofbot_ros_node.py        │                        │
│  │  - Direct ROS access         │                        │
│  │  - Hardware control          │                        │
│  └──────────────────────────────┘                        │
└───────────────────────────────────────────────────────────┘
```

**Why robot node is NOT in Docker:**
- Needs direct access to ROS topics
- Needs direct access to hardware (motors, camera)
- Needs real-time performance
- ROS master runs on host, not in container

**Why experiments ARE in Docker:**
- MADSci libraries pre-installed
- Consistent environment
- No dependency conflicts
- Easy to update

---

## Checking Your Setup

### 1. Verify MADSci Container Works

```bash
# Enter container
madsci

# Inside container - check MADSci
python3 << 'EOF'
from madsci.client import ExperimentClient, WorkcellClient
print("MADSci imports successful!")
EOF

# Check mounted directory
ls ~/MEDAL-LAB/clients/experiments
# Should show your experiment files

# Exit container
exit
```

### 2. Verify Services Are Running

```bash
# Outside container
docker ps
# Should show madsci-core services running

curl http://localhost:8002/health
curl http://localhost:8005/health
```

### 3. Verify Robot Node Can Connect

```bash
# On robot
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes

# Check it can import (should work now with --user install)
python3 -c "from madsci.node_module.rest_node_module import RestNode; print('OK')"

# Try running (will need ROS running)
python3 dofbot_ros_node.py
```

---

## Troubleshooting

### Problem: "No such file or directory" in container

**Symptom:** Can't find MEDAL-LAB inside container

**Fix:** Make sure you updated the `madsci` alias to mount MEDAL-LAB:
```bash
# Check your alias
type madsci | grep MEDAL-LAB

# If not there, update ~/.bashrc and source it
```

### Problem: Robot node can't import madsci

**Symptom:** `ModuleNotFoundError: No module named 'madsci'` on robot

**Fix:** Install on robot HOST (not in container):
```bash
# On robot, outside any container
pip3 install --user madsci-client opencv-python numpy
```

### Problem: Experiment can't reach services

**Symptom:** Connection refused to localhost:8002

**Fix:** 
1. Make sure services are running: `docker compose ps`
2. Use `--net=host` in madsci alias (already there)
3. Services must be on same network as container

### Problem: Robot node can't reach services

**Symptom:** Robot node shows connection errors

**Fix:** Check environment variables in `~/robot_config/.env`:
```bash
# Should point to central computer, not localhost
CENTRAL_COMPUTER_IP=192.168.1.XXX  # Your central computer IP
```

---

## Updated Quick Start

### Initial Setup (One-Time)

**1. Update madsci alias (Central Computer)**
```bash
nano ~/.bashrc
# Add MEDAL-LAB mount to madsci alias
source ~/.bashrc
```

**2. Create .env.global (Central Computer)**
```bash
cd ~/Documents/medal-lab-bill/MEDAL-LAB
cat > .env.global << 'EOF'
EXPERIMENT_SERVER_PORT=8002
RESOURCE_SERVER_PORT=8003
DATA_SERVER_PORT=8004
WORKCELL_SERVER_PORT=8005
LOCATION_SERVER_PORT=8006
EVENT_SERVER_PORT=8007
LAB_SERVER_PORT=8008
MINIO_CONSOLE_PORT=9001
EOF
```

**3. Install dependencies on robot (Robot Computer)**
```bash
# SSH to robot
ssh jetson@ubuntu
pip3 install --user madsci-client opencv-python numpy
```

### Daily Operation

**1. Start Services (Central)**
```bash
cd ~/Documents/medal-lab-bill/MEDAL-LAB/madsci-core
docker compose up -d
```

**2. Start Robot Node (Robot)**
```bash
ssh jetson@ubuntu
~/start_robot_node.sh
```

**3. Run Experiment (Central - Inside Container)**
```bash
madsci  # Enter container
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

---

## Summary of Changes

### What Runs Where

| Component | Location | Environment |
|-----------|----------|-------------|
| MADSci Core Services | Central | Docker Compose |
| Experiment Scripts | Central | MADSci Docker Container |
| Helper Modules | Central | MADSci Docker Container |
| Robot Node | Robot | Host (NOT Docker) |
| ROS/MoveIT | Robot | Host |

### Key Points

1. **Experiments run IN the MADSci container** (has MADSci pre-installed)
2. **Robot node runs ON the robot host** (needs direct ROS access)
3. **Services run in separate containers** (Docker Compose)
4. **Mount MEDAL-LAB in madsci alias** so experiments can access files
5. **Install madsci-client on robot HOST** (not in container)

This architecture leverages your existing Docker setup while adding the new block combination solver experiment!

---

## Permanent Solution: Custom Docker Image

The base MADSci image doesn't include OpenCV. Create an extended image with all needed dependencies.

### Option 1: Build Extended Image

**1. Create Dockerfile:**

File: `~/MEDAL-LAB/docker/Dockerfile.madsci-extended`
```dockerfile
# Extended MADSci Docker Image with OpenCV
FROM ghcr.io/ad-sdl/madsci:latest

# Install additional Python packages
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    numpy \
    scikit-image

WORKDIR /home/madsci
CMD ["/bin/bash"]
```

**2. Build the image:**
```bash
cd ~/Documents/medal-lab-bill/MEDAL-LAB/docker
docker build -t madsci-extended:latest -f Dockerfile.madsci-extended .
```

**3. Update your madsci alias:**
```bash
nano ~/.bashrc

# Change this line:
# ghcr.io/ad-sdl/madsci /bin/bash'
# To:
# madsci-extended:latest /bin/bash'

# Full updated alias:
alias madsci='
    xhost +local:docker >/dev/null;
    docker run --privileged -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,src="/home/gabriel/madsci/madsci-demo",dst="/home/madsci/madsci-demo" \
    --mount type=bind,src="/home/gabriel/Documents/medal-lab-bill/MEDAL-LAB",dst="/home/madsci/MEDAL-LAB" \
    madsci-extended:latest /bin/bash'

# Save and reload
source ~/.bashrc
```

**4. Test:**
```bash
madsci
python3 -c "import cv2; print('OpenCV OK!')"
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

### Option 2: Quick Startup Script (Install on Each Run)

If you don't want to build a custom image, create a startup script:

File: `~/MEDAL-LAB/docker/madsci-with-deps.sh`
```bash
#!/bin/bash
# Start MADSci container and install dependencies

xhost +local:docker >/dev/null
docker run --privileged -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,src="/home/gabriel/madsci/madsci-demo",dst="/home/madsci/madsci-demo" \
    --mount type=bind,src="/home/gabriel/Documents/medal-lab-bill/MEDAL-LAB",dst="/home/madsci/MEDAL-LAB" \
    ghcr.io/ad-sdl/madsci /bin/bash -c "
        pip3 install -q opencv-python-headless numpy scikit-image && 
        echo 'Dependencies installed.' && 
        exec /bin/bash
    "
```

Make it executable:
```bash
chmod +x ~/MEDAL-LAB/docker/madsci-with-deps.sh
```

Use it:
```bash
~/MEDAL-LAB/docker/madsci-with-deps.sh
# Waits a moment while installing, then gives you shell with deps ready
```

### Option 3: Install Once Per Session

Add to your workflow:

```bash
# Enter container
madsci

# Install dependencies (fast after first time due to caching)
pip3 install -q opencv-python-headless numpy

# Run experiments
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

---

## Recommended Approach

**Best:** Build custom image (Option 1)
- Dependencies always available
- Fastest startup
- Most professional

**Quick:** Install on each session (Option 3)
- No image building required
- Takes ~10 seconds per session
- Good for testing

**Middle:** Startup script (Option 2)
- No image building
- Automatic installation
- Good compromise

