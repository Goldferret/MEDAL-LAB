# Quick Start Checklist - Block Combination Solver

## Pre-Deployment Checklist

### Hardware
- [ ] DOFBOT Pro robot arm powered on
- [ ] Camera connected and powered
- [ ] Robot computer accessible (SSH or direct)
- [ ] Central Linux computer ready
- [ ] Both computers on same network

### Network Information
```
Central Computer IP: _________________ (e.g., 192.168.1.100)
Robot Computer IP:   192.168.1.234    (your robot)
```

---

## One-Time Setup

### On Central Computer

#### Install MADSci Services
```bash
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d
```
- [ ] All services running (`docker-compose ps`)

#### Configure Workcell
Edit: `madsci-core/managers/workcell/example_wc.workcell.yaml`
```yaml
nodes:
  - node_name: DOFBOT_Pro_1
    node_url: http://192.168.1.234:2000/  # Your robot IP
```
- [ ] Robot IP address updated

### On Robot Computer

#### Transfer Files
```bash
# From central computer:
scp -r robot-nodes/dofbot-pro-ros/nodes robot@192.168.1.234:~/madsci_node/
```
- [ ] Files transferred to `~/madsci_node/nodes/`

#### Install Dependencies
```bash
# On robot computer:
pip3 install madsci-client opencv-python numpy
```
- [ ] Dependencies installed

#### Create Environment File
```bash
# On robot computer:
nano ~/madsci_node/.env
```

Add:
```bash
CENTRAL_COMPUTER_IP=192.168.1.100        # Update with your central computer IP
EXPERIMENT_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8002
RESOURCE_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8003
DATA_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8004
LOCATION_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8006
DOFBOT_PRO_1_URL=http://192.168.1.234:2000/  # Your robot IP
```
- [ ] Environment file created
- [ ] Central computer IP address updated

---

## Daily Operation

### Step 1: Start Central Services
```bash
# On central computer:
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d
```
- [ ] Services started
- [ ] Health check passes: `curl http://localhost:8002/health`

### Step 2: Start Robot Node
```bash
# On robot computer (via SSH or direct):
cd ~/madsci_node/nodes
export $(cat ../.env | xargs)
python3 dofbot_ros_node.py
```
- [ ] Robot node running (leave terminal open)
- [ ] No errors in output
- [ ] Shows "Robot node startup complete"

### Step 3: Verify Connection
```bash
# From central computer:
curl http://192.168.1.234:2000/health
```
- [ ] Returns `{"status": "healthy"}`

### Step 4: Run Experiment
```bash
# On central computer:
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```
- [ ] Experiment starts
- [ ] Robot moves
- [ ] Images captured
- [ ] Experiment completes

---

## Quick Test Script

### Test Robot Connection
```bash
# On central computer:
python3 << EOF
from madsci.client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

wc = WorkcellClient("http://localhost:8005")
workflow = WorkflowDefinition(
    name="Test",
    steps=[StepDefinition(
        name="home",
        node="DOFBOT_Pro_1", 
        action="home_robot",
        args={}
    )]
)
result = wc.start_workflow(workflow)
print(f"[PASS] Test passed!" if result.status == "completed" else "[FAIL] Test failed")
EOF
```
- [ ] Test passes

---

## Troubleshooting Quick Fixes

### Robot Node Won't Connect
```bash
# Check services reachable from robot:
curl http://192.168.1.100:8002/health
# If fails: Check firewall on central computer
```

### Experiment Can't Find Robot
```bash
# Restart workcell manager:
cd ~/MEDAL-LAB/madsci-core
docker-compose restart workcell-manager
```

### ROS Not Running
```bash
# On robot computer:
rostopic list
# If empty: Start ROS/MoveIT (check manufacturer docs)
```

### Camera Not Working
```bash
# On robot computer:
rostopic echo /camera/color/image_raw --noarr
# If no data: Restart camera driver
```

---

## Shutdown Procedure

### End of Day
```bash
# 1. Stop experiment (if running) - Ctrl+C

# 2. Stop robot node - Ctrl+C in robot terminal

# 3. Stop central services:
cd ~/MEDAL-LAB/madsci-core
docker-compose down
```

---

## Emergency Stop

### If Robot Misbehaves
1. **Physical E-stop** - Press robot's emergency stop button
2. **Kill robot node** - Ctrl+C in robot terminal
3. **Stop all** - Turn off robot power

### If Experiment Hangs
1. **Ctrl+C** - Stop experiment script
2. **Check robot node** - Still running? Look for errors
3. **Restart services** - `docker-compose restart`

---

## Contact Information

**Fill in your details:**

```
Lab Manager: _________________________
Emergency:   _________________________
Robot IT:    _________________________
Network:     _________________________
```

---

## File Locations

| Item | Location |
|------|----------|
| Experiment scripts | `~/MEDAL-LAB/clients/experiments/` |
| Robot node | `~/madsci_node/nodes/` (robot computer) |
| Workcell config | `~/MEDAL-LAB/madsci-core/managers/workcell/` |
| Logs | `journalctl -u madsci-robot-node` (if using service) |

---

## Common Commands

### Check Status
```bash
# Central services
docker-compose ps

# Robot node
curl http://192.168.1.234:2000/health

# View experiment results
# (Access Data Manager UI or query via API)
```

### View Logs
```bash
# Central services
docker-compose logs -f workcell-manager

# Robot node (if running as service)
journalctl -u madsci-robot-node -f
```

### Restart Everything
```bash
# Central
cd ~/MEDAL-LAB/madsci-core
docker-compose restart

# Robot
# Ctrl+C then restart python3 dofbot_ros_node.py
```

---

## Success Indicators

### Everything Working When:
- [ ] All Docker services show "healthy"
- [ ] Robot node shows no errors
- [ ] `curl` to robot returns healthy
- [ ] Test workflow completes
- [ ] Experiment runs and completes
- [ ] Robot moves smoothly
- [ ] Images captured successfully

### Something Wrong If:
- [ ] Any service unreachable
- [ ] Robot node shows connection errors
- [ ] Workflows timeout
- [ ] Robot doesn't move
- [ ] No images captured

---

## Performance Benchmarks

**Normal operation:**
- Experiment startup: < 10 seconds
- Per attempt: 10-15 seconds
- Total experiment: 2-5 minutes (depends on luck)

**If slower:**
- Check network latency: `ping 192.168.1.234`
- Check robot load: `top` on robot computer
- Check service load: `docker stats` on central

---

## Version Information

**Record your setup:**

```
MADSci version:     _____________
ROS version:        _____________
Robot firmware:     _____________
Python version:     _____________
Last updated:       _____________
```

---

## Notes

```
_____________________________________________________________

_____________________________________________________________

_____________________________________________________________

_____________________________________________________________

_____________________________________________________________
```

---

## Step-by-Step: Running the Experiment

### Complete Workflow from Power-On to Results

#### 1. Power Up (5 minutes)

```bash
# Turn on robot arm
# Wait for boot-up complete (LED indicators)
# Verify camera is connected
```

- [ ] Robot powered on and booted
- [ ] Camera light on
- [ ] No error beeps from robot

#### 2. Start Central Services (2 minutes)

```bash
# On central computer - Terminal 1
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d

# Wait 5 seconds for startup
sleep 5

# Verify all services healthy
curl http://localhost:8002/health  # Should return {"status": "healthy"}
curl http://localhost:8005/health  # Should return {"status": "healthy"}
```

- [ ] All Docker containers running
- [ ] Health checks pass
- [ ] No error messages in logs

#### 3. Start Robot Node (1 minute)

```bash
# On central computer - Terminal 2
ssh robot@192.168.1.234

# Once connected to robot:
~/start_robot_node.sh

# Expected output:
# "ROS node initialized"
# "Connected to move_group action server"
# "Robot node startup complete"
# "Starting REST server on http://192.168.1.234:2000"
```

- [ ] SSH connection successful
- [ ] Robot node starts without errors
- [ ] Shows "startup complete" message
- [ ] **Leave this terminal running!**

#### 4. Verify Connection (30 seconds)

```bash
# On central computer - Terminal 3 (new terminal)
curl http://192.168.1.234:2000/health

# Should return:
# {"status": "healthy", "node_name": "DOFBOT_Pro_1"}
```

- [ ] Robot responds to health check
- [ ] Status shows "healthy"

#### 5. Prepare Workspace (2 minutes)

**Physical setup:**
- [ ] Place 6 colored blocks in 2x3 grid on workspace
- [ ] Ensure blocks are visible to camera
- [ ] Clear workspace of obstacles
- [ ] Ensure good lighting

**Block arrangement example:**
```
[Red]   [Blue]  [Yellow]
[Green] [Red]   [Blue]
```
(Colors can be any combination of red, blue, yellow, green)

#### 6. Run Quick Test (1 minute)

```bash
# On central computer - Terminal 3
python3 << 'EOF'
from madsci.client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

wc = WorkcellClient("http://localhost:8005")
workflow = WorkflowDefinition(
    name="Pre-flight Test",
    steps=[StepDefinition(
        name="home",
        node="DOFBOT_Pro_1",
        action="home_robot",
        args={}
    )]
)
result = wc.start_workflow(workflow)
print(f"\n[PASS] Test PASSED - Ready to run experiment!" if result.status == "completed" else "[FAIL] Test FAILED - Check robot node")
EOF
```

- [ ] Robot moves to home position
- [ ] Test prints "[PASS] Test PASSED"
- [ ] No errors or warnings

#### 7. Run the Experiment (2-5 minutes)

```bash
# On central computer - Terminal 3
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

**What you'll see:**

```
============================================================
6-Block Combination Search Experiment
============================================================

Step 1: Creating experiment design...
[OK] Experiment design created

Step 2: Starting experiment...
[OK] Experiment started: 01HN...

Step 3: Initializing MADSci clients...
[OK] Clients initialized

Step 4: Setting up block locations...
  [OK] Created location: pos_0
  [OK] Created location: pos_1
  [... continues ...]

Step 5: Scanning workspace and creating resources...
  Retrieving image from Data Manager...
  Processing image on client...
  Detected blocks: ['red', 'blue', 'yellow', 'green', 'red', 'blue']
  [OK] Created 6 block resources

Step 6: Generating target arrangement...
  Target: ['blue', 'red', 'green', 'yellow', 'blue', 'red']

Step 7: Checking initial configuration...
============================================================
Attempt #1
Testing: ['red', 'blue', 'yellow', 'green', 'red', 'blue']
============================================================
[X] No match. Current: [...], Target: [...]
  Attempt time: 12.45 seconds

[Robot will now shuffle blocks and test combinations...]

============================================================
Attempt #2
Combinations tried so far: 1
============================================================
Testing: ['blue', 'red', 'green', 'yellow', 'blue', 'red']
  Executing 7 block moves...
[Robot moves blocks...]
[OK] MATCH FOUND!
  Detected: ['blue', 'red', 'green', 'yellow', 'blue', 'red']
  Target:   ['blue', 'red', 'green', 'yellow', 'blue', 'red']
  Attempt time: 14.32 seconds

[... continues through cleanup ...]

============================================================
EXPERIMENT COMPLETE
============================================================
Solution found: True
Total attempts: 2
[... timing statistics ...]

[SUCCESS] Experiment completed successfully!
```

**During the experiment:**
- [ ] Robot moves to scan position
- [ ] Robot picks and places blocks
- [ ] Console shows progress for each attempt
- [ ] Blocks are rearranged between attempts

#### 8. Review Results (1 minute)

**Console output shows:**
- [ ] Total attempts made
- [ ] Whether solution was found
- [ ] Timing statistics
- [ ] Snapshot ID (for data retrieval)

**Check the summary:**
```
FINAL SUMMARY
============================================================
Success: True
Attempts: 2/50
Target: ['blue', 'red', 'green', 'yellow', 'blue', 'red']
Final: ['blue', 'red', 'green', 'yellow', 'blue', 'red']
Snapshot ID: 01HN...
Experiment ID: 01HN...
Total time: 26.77 seconds
============================================================
```

#### 9. Shutdown (1 minute)

```bash
# Terminal 3 (experiment terminal)
# Experiment completes automatically, no action needed

# Terminal 2 (robot node)
# Press Ctrl+C to stop robot node

# Terminal 1 (central services)
cd ~/MEDAL-LAB/madsci-core
docker-compose down

# Power off robot (optional)
# Press power button on robot arm
```

- [ ] Experiment completed or stopped
- [ ] Robot node stopped (Ctrl+C)
- [ ] Central services stopped
- [ ] Robot powered down (if done for the day)

---

## Troubleshooting During Experiment

### Robot Doesn't Move

**Stop and check:**
```bash
# In robot terminal (Terminal 2):
# Look for error messages like:
# "Movement timed out"
# "Failed to reach position"

# Test ROS manually on robot:
rostopic list
rostopic echo /joint_states --noarr
```

**Fix:** Restart ROS/MoveIT, then restart robot node

### Experiment Can't Detect Blocks

**Symptoms:**
```
Detected blocks: [None, None, None, None, None, None]
Error: No blocks detected in initial scan
```

**Fix:**
- Check lighting (too bright/dark?)
- Verify camera is working: `rostopic echo /camera/color/image_raw --noarr`
- Adjust block positions (blocks too close together?)
- Check color thresholds in `helpers/vision.py`

### Experiment Hangs

**Symptoms:**
- Console stops updating
- Robot stops moving mid-workflow

**Fix:**
1. **Ctrl+C** in experiment terminal
2. Check robot node for errors (Terminal 2)
3. Restart services: `docker-compose restart workcell-manager`
4. Re-run experiment

### Blocks End Up in Wrong Positions

**Symptoms:**
- Blocks physically moved but tracking is wrong
- "Resource not found" errors

**Fix:**
- This is usually OK - experiment trusts workflow execution
- Check for mechanical issues (gripper not closing properly?)
- Verify joint angles in `helpers/block_combination/locations.py`

---

## Quick Command Summary

### Full Run Sequence (Copy-Paste Ready)

```bash
# === Terminal 1: Central Services ===
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d
sleep 5
curl http://localhost:8005/health

# === Terminal 2: Robot Node ===
ssh robot@192.168.1.234
~/start_robot_node.sh
# Leave running!

# === Terminal 3: Experiment ===
# Wait for robot node to show "startup complete"
curl http://192.168.1.234:2000/health
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py

# === When Done ===
# Terminal 2: Ctrl+C
# Terminal 1: 
cd ~/MEDAL-LAB/madsci-core
docker-compose down
```

---

## Expected Timeline

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Power up | 1-2 min | Robot boots, services start |
| Initial scan | 5 sec | Camera captures, blocks detected |
| Per attempt | 10-15 sec | Robot rearranges blocks, scans |
| Total experiment | 2-5 min | Depends on luck (1-30 attempts) |
| Shutdown | 30 sec | Cleanup and services stop |

**Total time from power-on to results: ~5-10 minutes**

---

**Keep this checklist handy for daily operations!**

For detailed instructions, see: `docs/Deployment_Guide.md`

