# Your Specific Setup Commands

**Robot IP: 192.168.1.234**

## One-Time Setup

### 1. Configure Workcell (Central Computer)

Edit this file:
```bash
nano ~/MEDAL-LAB/madsci-core/managers/workcell/example_wc.workcell.yaml
```

Set to:
```yaml
workcell_name: BlockSolverWorkcell
workcell_description: 6-block combination solver workcell

nodes:
  - node_name: DOFBOT_Pro_1
    node_url: http://192.168.1.234:2000/
    capabilities:
      - movement
      - manipulation
      - vision
```

### 2. Setup Robot (You Already Have the Repo!)

Since you already have MEDAL-LAB via git on the robot, you just need config:

SSH to robot:
```bash
ssh robot@192.168.1.234
```

Create environment file:
```bash
mkdir -p ~/robot_config
cat > ~/robot_config/.env << 'EOF'
# UPDATE THE NEXT LINE WITH YOUR CENTRAL COMPUTER'S IP!
CENTRAL_COMPUTER_IP=192.168.1.XXX

EXPERIMENT_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8002
RESOURCE_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8003
DATA_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8004
LOCATION_SERVER_URL=http://${CENTRAL_COMPUTER_IP}:8006
DOFBOT_PRO_1_URL=http://192.168.1.234:2000/
EOF
```

**IMPORTANT**: Replace `192.168.1.XXX` with your central computer's IP!

To find your central computer's IP:
```bash
# On central computer
hostname -I | awk '{print $1}'
```

### 3. Create Robot Startup Script

Still on robot (via SSH):
```bash
cat > ~/start_robot_node.sh << 'EOF'
#!/bin/bash
echo "Starting MADSci Robot Node..."
export $(cat ~/robot_config/.env | xargs)
cd ~/MEDAL-LAB/robot-nodes/dofbot-pro-ros/nodes
python3 dofbot_ros_node.py
EOF

chmod +x ~/start_robot_node.sh
```

### 4. Install Dependencies on Robot

```bash
# Still on robot:
pip3 install madsci-client opencv-python numpy
```

---

## Daily Operation Commands

### Start Everything (Copy-Paste Ready)

**Terminal 1 - Central Computer** (Start MADSci services):
```bash
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d
sleep 5
echo "Services started! Check health:"
curl http://localhost:8002/health
curl http://localhost:8005/health
```

**Terminal 2 - Robot** (SSH and start robot node):
```bash
ssh robot@192.168.1.234
~/start_robot_node.sh
```
*Leave this running!*

**Terminal 3 - Central Computer** (Verify connection):
```bash
# Wait for robot node to finish starting, then test:
curl http://192.168.1.234:2000/health
# Should return: {"status": "healthy", "node_name": "DOFBOT_Pro_1"}
```

**Terminal 4 - Central Computer** (Run experiment):
```bash
cd ~/MEDAL-LAB/clients/experiments
python3 block_combination_solver_experiment.py
```

---

## Quick Test

After starting services and robot node, test with:

```bash
# On central computer:
python3 << 'EOF'
from madsci.client import WorkcellClient
from madsci.common.types.workflow_types import WorkflowDefinition
from madsci.common.types.step_types import StepDefinition

wc = WorkcellClient("http://localhost:8005")
workflow = WorkflowDefinition(
    name="Connection Test",
    steps=[StepDefinition(
        name="home",
        node="DOFBOT_Pro_1",
        action="home_robot",
        args={}
    )]
)
result = wc.start_workflow(workflow)
print(f"\n{'✓ SUCCESS!' if result.status == 'completed' else '✗ FAILED'}")
print(f"Robot moved to home position!\n")
EOF
```

If this works, you're ready to run the full experiment!

---

## Troubleshooting Your Setup

### Can't reach robot node?
```bash
# Check robot is reachable:
ping 192.168.1.234

# Check robot node port:
curl http://192.168.1.234:2000/health

# If fails, check robot node is running (Terminal 2)
```

### Can't reach central services?
```bash
# On central computer:
docker-compose ps    # All should show "running"

# If not:
cd ~/MEDAL-LAB/madsci-core
docker-compose restart
```

### Experiment can't find robot?
```bash
# Restart workcell manager:
cd ~/MEDAL-LAB/madsci-core
docker-compose restart workcell-manager

# Verify workcell config:
cat managers/workcell/example_wc.workcell.yaml
# Should show: http://192.168.1.234:2000/
```

---

## Your Network Setup

```
┌─────────────────────────────┐
│   Central Computer          │
│   IP: 192.168.1.XXX         │  ← Fill in your IP
│   Ports: 8002-8006          │
│   (MADSci services)         │
└─────────────┬───────────────┘
              │
              │ Network
              │
┌─────────────┴───────────────┐
│   Robot Computer            │
│   IP: 192.168.1.234         │  ← Your robot
│   Port: 2000                │
│   (Robot node)              │
└─────────────────────────────┘
```

---

## Complete Startup Script

Save this as `~/start_everything.sh` on central computer:

```bash
#!/bin/bash
echo "=========================================="
echo "Starting MADSci Block Combination Solver"
echo "=========================================="
echo ""

# Start services
echo "1. Starting MADSci services..."
cd ~/MEDAL-LAB/madsci-core
docker-compose up -d
sleep 5

# Check services
echo "2. Checking service health..."
SERVICES_OK=true
for port in 8002 8003 8004 8005 8006; do
    if curl -sf http://localhost:$port/health > /dev/null; then
        echo "   ✓ Port $port OK"
    else
        echo "   ✗ Port $port FAILED"
        SERVICES_OK=false
    fi
done

if [ "$SERVICES_OK" = false ]; then
    echo ""
    echo "❌ Some services failed to start!"
    exit 1
fi

echo ""
echo "✓ MADSci services running"
echo ""
echo "Next steps:"
echo "  1. SSH to robot: ssh robot@192.168.1.234"
echo "  2. Run: ~/start_robot_node.sh"
echo "  3. Then run experiment: cd ~/MEDAL-LAB/clients/experiments && python3 block_combination_solver_experiment.py"
echo ""
```

Make executable:
```bash
chmod +x ~/start_everything.sh
```

---

## Summary Checklist

- [ ] Workcell YAML updated with `http://192.168.1.234:2000/`
- [ ] Robot `.env` file created with central computer IP
- [ ] Files transferred to robot
- [ ] Dependencies installed on robot
- [ ] Can ping robot: `ping 192.168.1.234` ✓
- [ ] Services healthy on central
- [ ] Robot node running and showing "startup complete"
- [ ] Test connection successful
- [ ] Ready to run experiment! 🚀

---

**Need help?** See:
- Full guide: `docs/Deployment_Guide.md`
- Quick checklist: `docs/Quick_Start_Checklist.md`

