# MADSCI-LAB

**Autonomous laboratory framework for orchestrating robots, AI agents, and resources through intelligent workflows with comprehensive tracking and experimental coordination.**

MADSCI-LAB is a practical implementation of the MADSci framework designed for autonomous scientific experimentation using physical robotics hardware. This repository provides a complete working example of autonomous laboratory automation with DOFBOT Pro robotic arms, integrated camera systems, and intelligent workflow orchestration.

## 🤖 Hardware Requirements

- **DOFBOT Pro** robotic arm
- **Orbbec DaiBai DCW2** camera (or compatible Orbbec camera)
- **NVIDIA Jetson Orin Nano** (for robot node execution)
- **Host computer** (for workcell manager and client operations)

## 📋 Prerequisites

This repository works alongside the main MADSci framework in a distributed architecture:

- **MADSci Framework**: The main framework components run as Docker containers
- **Python 3.10**: Required for the robot node environment on Jetson
- **Docker**: Required for workcell manager and client operations

## 🏗️ Repository Structure

```
MADSCI-LAB/
├── .env.example                       # Example environment configuration
├── ARCHITECTURE.md                    # Core framework architecture design
├── captures/                          # Camera calibration and data capture
│   └── example_camera_calibration.json    # Example calibration parameters
├── workflows/                         # Workflow examples and scripts
│   ├── recording_workflow.py             # Data collection with synchronized recording
│   ├── transfer.py                       # Basic robot transfer operations
│   ├── camera_calibration.py             # Camera calibration workflow
│   └── validate_calibration.py           # Calibration validation workflow
├── managers/                          # Workcell configuration
│   └── example_wc.workcell.yaml          # Example workcell setup
├── nodes/                            # Robot nodes and hardware control
│   ├── dofbot_expert_node.py             # Main DOFBOT Pro node with camera integration
│   ├── Arm_Lib.py                        # Low-level arm control library
│   ├── angle_finder.py                   # Servo position utility
│   └── default.node.yaml                 # Default node configuration
└── tools/                            # Hardware testing and diagnostic tools
    ├── diagnose_orbbec_device.py         # Orbbec camera diagnostic script
    ├── fix_orbbec_device.sh              # Automated Orbbec device fix script
    └── test_cameras_final.py             # Comprehensive camera testing script
```

## 🚀 Quick Start

### 0. Environment Configuration

First, set up your network and path configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual configuration
# WORKCELL_MANAGER_URL=http://YOUR_WORKCELL_MANAGER_IP:8005
# ROBOT_NODE_URL=http://YOUR_JETSON_IP:2000
# MADSCI_PATH=/path/to/your/MADSci/repository
```

## 🔧 Component Setup

MADSCI-LAB uses a distributed architecture with three main components running on separate devices:

### 1. 🤖 Robot Node (NVIDIA Jetson Orin Nano)

The robot node runs directly on the Jetson hardware and controls the DOFBOT Pro arm and camera.

**Dependencies:**
- Python 3.10 virtual environment (required for Jetson compatibility)
- MADSci framework (`pip install madsci`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- smbus2 (`pip install smbus2`) - for I2C servo communication
- python-dotenv (`pip install python-dotenv`) - for .env file support
- pyorbbecsdk-1.3.1 (install from source)
- Additional dependencies automatically installed with MADSci:
  - Pydantic (for data validation)
  - FastAPI (for REST API functionality)
  - Other MADSci framework dependencies

**Setup:**
```bash
# Create and activate virtual environment
python3.10 -m venv madsci_env
source madsci_env/bin/activate

# Install core dependencies
pip install madsci opencv-python numpy smbus2 python-dotenv

# Install pyorbbecsdk-1.3.1 from source
# 1. Download source code from GitHub releases
wget https://github.com/orbbec/pyorbbecsdk/archive/refs/tags/v1.3.1.tar.gz
tar -xzf v1.3.1.tar.gz
cd pyorbbecsdk-1.3.1

# 2. Install build dependencies
pip install -r requirements.txt

# 3. Build the SDK
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install

# 4. IMPORTANT: After building, export PYTHONPATH to include the install directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/

# 5. Install udev rules for camera device access
cd ../
export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
sudo bash ./scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger

# 6. You can now cd back to your MADSCI-LAB repository and run the robot node
# The PYTHONPATH export allows pyorbbecsdk to be imported from anywhere
```

**Start the robot node:**
```bash
# From the pyorbbecsdk directory with PYTHONPATH set
# Set NODE_DEFINITION and ROBOT_NODE_URL in your .env file, then:
python /path/to/madsci-lab/nodes/dofbot_expert_node.py
```

### 2. 🏭 Workcell Manager (Host Computer)

The workcell manager orchestrates workflows and manages the laboratory resources using Docker containers.

**Dependencies:**
- Docker (>= 20.10) and Docker Compose (>= 2.0)
- Bash shell
- MADSci repository cloned locally
- MADSci Docker images (automatically pulled):
  - `ghcr.io/ad-sdl/madsci:latest`
  - `ghcr.io/ad-sdl/madsci_dashboard:latest`
- Database containers (automatically managed):
  - MongoDB 8.0
  - Redis 7.4
  - PostgreSQL 17
  - MinIO (object storage)

**Setup:**
```bash
# Start workcell manager with all required services
./managers/start_workcell.sh
```

**Setup:**
```bash
# Start workcell manager with all required services
./managers/start_workcell.sh
```

**Automatic Configuration:**
The `start_workcell.sh` script automatically configures the workcell definition with IP addresses from your `.env` file:
- Extracts the host and port from `WORKCELL_MANAGER_URL` 
- Uses `ROBOT_NODE_URL` for the robot node configuration
- Creates a properly configured workcell definition file

This eliminates the need to manually sync IP addresses between `.env` and the workcell YAML file.

**View logs:**
```bash
# View workcell manager logs
docker logs -f workcell_manager
```

### 3. 💻 Workcell Client (Development Machine)

The workcell client submits and monitors workflows using the official MADSci Docker image.

**Dependencies:**
- Docker
- Network access to workcell manager

**Setup:**
```bash
# Start interactive MADSci container with entire repository mounted
docker run -it --network host \
  -v $(pwd):/medal-lab \
  -w /medal-lab \
  ghcr.io/ad-sdl/madsci:latest bash

# Inside container, run workflows
python workflows/transfer.py
python workflows/recording_workflow.py
python workflows/camera_calibration.py
python workflows/validate_calibration.py
```

## 📖 Workflow Examples

### Basic Transfer Operation
```python
# workflows/transfer.py
# Demonstrates simple robot transfer between positions
```

### Data Collection with Recording
```python
# workflows/recording_workflow.py  
# Synchronized camera recording during robot operations for training data collection
```

### Camera Calibration
```python
# workflows/camera_calibration.py
# Automated camera calibration workflow
```

## 🔧 Configuration

### Environment Variables
- Copy `.env.example` to `.env` and update with your network configuration
- `WORKCELL_MANAGER_URL` - IP address and port of your workcell manager host
- `ROBOT_NODE_URL` - IP address and port of your Jetson Orin Nano
- `NODE_DEFINITION` - Path to your node definition YAML file
- `MADSCI_PATH` - Path to your cloned MADSci repository

### Node Configuration
- `nodes/default.node.yaml` - Default node settings
- `nodes/angle_finder.py` - Utility for manual position setup

### Workcell Configuration  
- `managers/example_wc.workcell.yaml` - Workcell setup with nodes and locations

### Camera Calibration
- `captures/example_camera_calibration.json` - Example calibration parameters

## 🌐 Network Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Workcell Client   │    │  Workcell Manager   │    │    Robot Node       │
│  (Development PC)   │    │   (Host Computer)   │    │  (Jetson Orin Nano) │
│                     │    │                     │    │                     │
│ • Docker Container  │◄──►│ • Docker Compose    │◄──►│ • Python 3.10       │
│ • MADSci Image      │    │ • Workcell Manager  │    │ • DOFBOT Pro        │
│ • Workflow Scripts  │    │ • Resource Manager  │    │ • Orbbec Camera     │
│                     │    │ • Event Manager     │    │ • pyorbbecsdk       │
│                     │    │ • Redis + MongoDB   │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🛠️ Troubleshooting

### Camera Issues
If you encounter problems with the Orbbec DaiBai DCW2 camera:

```bash
# Run diagnostic script to identify issues
python3 tools/diagnose_orbbec_device.py

# Run automated fix script for common issues
./tools/fix_orbbec_device.sh

# Test camera functionality
python3 tools/test_cameras_final.py
```

Common camera issues and solutions:
- **"No device found"**: Check USB connection, run fix script for udev rules
- **Permission denied**: Add user to video group, restart system
- **Import errors**: Ensure pyorbbecsdk is installed with correct PYTHONPATH

### Node Connection Issues
- Verify `.env` file configuration matches your network setup
- Check that robot node URL is accessible from workcell manager
- Ensure no firewall blocking the configured ports

## 📚 Documentation

- **[Architecture Design](ARCHITECTURE.md)** - Complete framework architecture with hybrid digital twin design
- **[MADSci Framework](https://github.com/AD-SDL/MADSci)** - Main framework repository

## 🔬 Current Features

- **Autonomous Robot Control**: DOFBOT Pro integration with precise servo control
- **Camera Integration**: Orbbec camera support with calibration workflows  
- **Workflow Orchestration**: MADSci-based workflow management and execution
- **Data Collection**: Synchronized recording for AI training data generation
- **Resource Management**: Comprehensive tracking of laboratory assets and states
- **Event Logging**: Complete experimental traceability and debugging support

## 🚧 Development Status

This repository represents an active implementation of the MADSci framework architecture. While core functionality is operational, development is ongoing to implement the complete feature set described in the architecture documentation.

**Currently Implemented:**
- Basic robot control and workflows
- Camera integration and calibration
- Data collection workflows
- MADSci framework integration

**In Development:**
- Complete hybrid digital twin implementation
- Advanced multi-robot coordination
- Enhanced AI agent integration
- Comprehensive resource management features

## 🤝 Contributing

This project is under active development. The architecture design in `ARCHITECTURE.md` provides the roadmap for full implementation.

## 📄 License

[Add your license information here]

## 🆘 Support

For issues related to:
- **MADSci Framework**: See the main MADSci repository
- **Hardware Setup**: Check the component setup sections above
- **Architecture Questions**: Review `ARCHITECTURE.md`
