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
    node_url: http://192.168.1.234:2000/  # ← Your robot IP
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
CENTRAL_COMPUTER_IP=192.168.1.100        # ← Update with your central computer IP
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
print(f"✓ Test passed!" if result.status == "completed" else "✗ Test failed")
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

### ✓ Everything Working When:
- [ ] All Docker services show "healthy"
- [ ] Robot node shows no errors
- [ ] `curl` to robot returns healthy
- [ ] Test workflow completes
- [ ] Experiment runs and completes
- [ ] Robot moves smoothly
- [ ] Images captured successfully

### ✗ Something Wrong If:
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

**Keep this checklist handy for daily operations!** 📋

For detailed instructions, see: `docs/Deployment_Guide.md`

