# MEDAL-LAB

**Autonomous laboratory automation using MADSci framework with DOFBOT Pro robotic arms and intelligent workflow orchestration.**

**MADSci Version:** v0.5.0

## What is MEDAL-LAB?

MEDAL-LAB is a working implementation of the MADSci framework for autonomous laboratory operations. It provides a distributed system for controlling robots, managing experiments, and collecting data across multiple devices in a laboratory environment.

The system uses a three-device architecture: robot nodes run on Jetson hardware for real-time control, MADSci core services run on a host computer for orchestration, and client workstations submit workflows and monitor experiments.

## Hardware Requirements

- **DOFBOT Pro** robotic arm with ROS/MoveIT support
- **Orbbec DaiBai DCW2** depth camera (or compatible Orbbec camera)
- **NVIDIA Jetson Orin Nano** for robot node execution
- **Host computer** for MADSci core services (Docker required)
- **Development workstation** for workflow submission (Docker required)

## Repository Structure

```
MEDAL-LAB/
├── madsci-core/                      # MADSci framework infrastructure
│   ├── docker-compose.yml                # Core services orchestration
│   ├── README.md                         # Setup and deployment guide
│   ├── managers/                         # Service configurations
│   └── tools/                            # Validation scripts
├── robot-nodes/                      # Robot implementations
│   ├── dofbot-pro-ros/                   # ROS-based DOFBOT (current)
│   └── dofbot-pro-archived/              # Hardware-level implementation (legacy)
├── clients/                          # Client workstation setup
│   ├── docker-compose.yml                # Client container
│   ├── workflows/                        # Workflow scripts
│   └── experiments/                      # Experiment definitions
└── .env.global.example               # Network configuration template
```

## Quick Start

### 1. Configure Network Settings

```bash
# Copy environment template
cp .env.global.example .env.global

# Edit with your network configuration
# - MADSCI_CORE_HOST: IP of host computer running MADSci services
# - DOFBOT_PRO_1_HOST: IP of Jetson running robot node
```

### 2. Start MADSci Core Services

On your host computer:

```bash
make madsci-up
cd madsci-core && ./tools/validate-deployment.sh  # Verify all services running
```

See [madsci-core/README.md](madsci-core/README.md) for detailed setup.

### 3. Start Robot Node

On your Jetson device, start ROS services then the MADSci node container. See [robot-nodes/dofbot-pro-ros/README.md](robot-nodes/dofbot-pro-ros/README.md) for complete instructions.

```bash
# Start ROS services (MoveIT, arm driver, camera) - takes ~100 seconds
make ros-up

# Start MADSci robot node
make robot-up NODE=dofbot-pro-ros
```

**View ROS logs (optional):**
```bash
make ros-attach  # Attach to tmux session
make ros-status  # Check ROS service status
```

**tmux commands while attached:**
- `Ctrl+b` then `d` - Detach (services keep running)
- `Ctrl+b` then `0-3` - Switch to window (0=moveit, 1=arm_driver, 2=sim_bridge, 3=camera)
- `Ctrl+b` then `n` - Next window
- `Ctrl+b` then `p` - Previous window

### 4. Submit Workflows and Experiments

From your development workstation:

```bash
make client-up
make client-shell

# Inside container - run a workflow:
python workflows/demo_workflow.py

# Or run an experiment:
python experiments/demo_experiment.py
```

See [clients/README.md](clients/README.md) for workflow and experiment development.

## Network Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Client Workstation│    │   MADSci Core Host  │    │   Robot Node        │
│   (Development PC)  │    │   (Host Computer)   │    │   (Jetson Orin)     │
│                     │    │                     │    │                     │
│ • Workflow Scripts  │◄──►│ • Workcell Manager  │◄──►│ • ROS + MoveIT      │
│ • Experiment Mgmt   │    │ • Resource Manager  │    │ • MADSci Node       │
│ • Data Queries      │    │ • Event Manager     │    │ • DOFBOT Pro        │
│ • MADSci Client     │    │ • Data Manager      │    │ • Orbbec Camera     │
│                     │    │ • MongoDB + Redis   │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Documentation

### Component Setup
- **[MADSci Core](madsci-core/README.md)** - Framework services deployment and configuration
- **[Robot Nodes](robot-nodes/dofbot-pro-ros/README.md)** - ROS-based robot node setup and development
- **[Clients](clients/README.md)** - Workflow submission and experiment management
- **[Archived Implementation](robot-nodes/dofbot-pro-archived/README.md)** - Legacy hardware-level robot control

### Workflows and Tools
- **[Workflows](clients/workflows/README.md)** - Available workflows and usage examples
- **[Diagnostic Tools](robot-nodes/dofbot-pro-archived/archived-tools/README.md)** - Camera calibration and hardware diagnostics

### External Resources
- **[MADSci Framework](https://github.com/AD-SDL/MADSci)** - Main framework repository and documentation

## System Features

- **Distributed Architecture**: Robot control, orchestration, and workflow submission on separate devices
- **ROS Integration**: MoveIT motion planning with MADSci workflow orchestration
- **Resource Management**: Track samples, consumables, and robot gripper contents
- **Event Logging**: Complete audit trail of all system operations
- **Data Collection**: Synchronized camera and robot state recording
- **Workflow Orchestration**: Submit complex multi-step experiments from client workstations

## Support

For setup issues:
- Check component-specific README files for detailed troubleshooting
- Verify network configuration in `.env.global`
- Use validation scripts in `madsci-core/tools/`
- Review diagnostic tools in archived implementation for camera issues

For MADSci framework questions, see the [main MADSci repository](https://github.com/AD-SDL/MADSci).
