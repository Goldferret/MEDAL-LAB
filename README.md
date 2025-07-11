# MEDAL-LAB

**Autonomous laboratory framework for orchestrating robots, AI agents, and resources through intelligent workflows with comprehensive tracking and experimental coordination.**

MEDAL-LAB is a practical implementation of the MADSci framework designed for autonomous scientific experimentation using physical robotics hardware. This repository provides a complete working example of autonomous laboratory automation with DOFBOT Pro robotic arms, integrated camera systems, and intelligent workflow orchestration.

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
MEDAL-LAB/
├── .env.example                       # Example environment configuration
├── ARCHITECTURE.md                    # Core framework architecture design
├── captures/                          # Data capture and camera calibration
│   ├── example_camera_calibration.json    # Example calibration parameters
│   ├── experiment_YYYYMMDD_HHMMSS/        # Recorded experiment data (git-ignored)
│   │   ├── rgb_images/                     # RGB camera frames
│   │   ├── depth_images/                   # Depth camera data
│   │   ├── point_clouds/                   # 3D point cloud data
│   │   └── trajectory_data.json            # Joint states and metadata
│   └── capture_YYYYMMDD_HHMMSS.jpg        # Single image captures (git-ignored)
├── workflows/                         # Workflow examples and scripts
│   ├── recording_workflow.py             # Data collection with synchronized recording
│   ├── transfer.py                       # Basic robot transfer operations
│   ├── camera_calibration.py             # Camera calibration workflow
│   ├── validate_calibration.py           # Calibration validation workflow
│   └── take_picture.py                   # Single image capture workflow
├── managers/                          # Workcell configuration
│   └── example_wc.workcell.yaml          # Example workcell setup
├── nodes/                            # Robot nodes and hardware control
│   ├── dofbot_modular_node.py            # Modular DOFBOT Pro node (MADSci compliant)
│   ├── robot_arm_config.py               # Centralized configuration management
│   ├── robot_arm_interface.py            # High-level robot coordination
│   ├── components/                        # Modular component architecture
│   │   ├── camera_manager.py              # Camera operations and pipeline management
│   │   ├── vision_detector.py             # Computer vision and object detection
│   │   ├── movement_controller.py         # Robot movement and servo control
│   │   └── calibration_manager.py         # Camera and system calibration
│   ├── Arm_Lib.py                        # Low-level arm control library
│   └── default.node.yaml                 # Default node configuration
└── tools/                            # Hardware testing and diagnostic tools
    ├── README.md                         # Comprehensive tools documentation
    ├── diagnose_orbbec_device.py         # Orbbec camera diagnostic script
    ├── fix_orbbec_device.sh              # Automated Orbbec device fix script
    ├── test_cameras_final.py             # Comprehensive camera testing script
    ├── hsv_calibration_tool.py           # Interactive HSV color range calibration
    ├── calibrate_depth_camera.py         # Interactive camera calibration tool
    └── validate_camera_calibration.py    # Camera calibration validation tool
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

MEDAL-LAB uses a distributed architecture with three main components running on separate devices:

### 1. 🤖 Robot Node (NVIDIA Jetson Orin Nano)

The robot node runs directly on the Jetson hardware and controls the DOFBOT Pro arm and camera using a **modular, component-based architecture** for better maintainability and MADSci compliance.

**Architecture Features:**
- **Modular Components**: Separate managers for camera, vision, movement, and calibration
- **MADSci Compliant**: Full REST API with proper action patterns and state management
- **Clean Separation**: Operational actions vs setup/maintenance tools
- **Robust Error Handling**: Comprehensive safety checks and exception management
- **Configurable**: Centralized configuration with environment variable support

**Dependencies:**
- Python 3.10 virtual environment (required for Jetson compatibility)
- MADSci framework (`pip install madsci.client madsci.node_module`)
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
pip install madsci.client madsci.node_module opencv-python numpy smbus2 python-dotenv

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

# 6. You can now cd back to your MEDAL-LAB repository and run the robot node
# The PYTHONPATH export allows pyorbbecsdk to be imported from anywhere
```

**Start the robot node:**
```bash
# From the pyorbbecsdk directory with PYTHONPATH set
# Set NODE_DEFINITION and ROBOT_NODE_URL in your .env file, then:
python /path/to/medal-lab/nodes/dofbot_modular_node.py
```

**Available Actions:**
- **Movement**: `move_joint`, `move_all_joints`, `grabber_position`, `transfer`
- **Vision**: `scan_for_target`, `center_on_target`
- **Camera**: `capture_single_image`, `test_camera_capture`
- **Recording**: `start_recording`, `stop_recording`
- **Status**: `get_robot_status`, `reset_movement_state`

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
# Demonstrates simple robot transfer between positions using the modular node
```

### Data Collection with Recording
```python
# workflows/recording_workflow.py  
# Synchronized camera recording during robot operations for training data collection
# Data automatically saved to captures/experiment_YYYYMMDD_HHMMSS/ directories
```

### Camera Calibration
```python
# workflows/camera_calibration.py
# Automated camera calibration workflow (deprecated - use tools/calibrate_depth_camera.py)
```

### Single Image Capture Workflow
```python
# workflows/take_picture.py
# Workflow for capturing single images using the robot node's capture_single_image action
```

### Single Image Capture
The robot node includes a `capture_single_image` action that:
- Captures a single RGB image from the camera
- Saves it as `captures/capture_YYYYMMDD_HHMMSS.jpg`
- Returns image metadata and file information

**Note**: All experimental data (experiment folders and single captures) are automatically excluded from git commits to protect your personal research data.

## 🔧 Configuration

### Environment Variables
- Copy `.env.example` to `.env` and update with your network configuration
- `WORKCELL_MANAGER_URL` - IP address and port of your workcell manager host
- `ROBOT_NODE_URL` - IP address and port of your Jetson Orin Nano
- `NODE_DEFINITION` - Path to your node definition YAML file
- `MADSCI_PATH` - Path to your cloned MADSci repository

### Node Configuration
- `nodes/default.node.yaml` - Default node settings
- `nodes/robot_arm_config.py` - Centralized configuration with vision thresholds, HSV ranges, and movement parameters

### Workcell Configuration  
- `managers/example_wc.workcell.yaml` - Workcell setup with nodes and locations

### Camera Calibration
- `captures/example_camera_calibration.json` - Example calibration parameters
- Use `tools/calibrate_depth_camera.py` for interactive calibration
- Experiment data automatically saved to `captures/experiment_*/` directories
- Single image captures saved as `captures/capture_*.jpg` files

## 🌐 Network Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Workcell Client   │    │  Workcell Manager   │    │    Robot Node       │
│  (Development PC)   │    │   (Host Computer)   │    │  (Jetson Orin Nano) │
│                     │    │                     │    │                     │
│ • Docker Container  │◄──►│ • Docker Compose    │◄──►│ • Modular Node      │
│ • MADSci Image      │    │ • Workcell Manager  │    │ • Component-Based   │
│ • Workflow Scripts   │    │ • Resource Manager  │    │ • DOFBOT Pro        │
│                     │    │ • Event Manager     │    │ • Orbbec Camera     │
│                     │    │ • Redis + MongoDB   │    │ • MADSci Compliant  │
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
- **[Tools Documentation](tools/README.md)** - Comprehensive guide to all setup and diagnostic tools
- **[MADSci Framework](https://github.com/AD-SDL/MADSci)** - Main framework repository

## 🔬 Current Features

- **Modular Robot Control**: Component-based DOFBOT Pro integration with clean architecture
- **MADSci Compliance**: Full REST API with proper action patterns and state management
- **Camera Integration**: Orbbec camera support with comprehensive calibration tools
- **Vision System**: HSV-based object detection with configurable color ranges
- **Workflow Orchestration**: MADSci-based workflow management and execution
- **Data Collection**: Synchronized recording for AI training data generation
- **Resource Management**: Comprehensive tracking of laboratory assets and states
- **Event Logging**: Complete experimental traceability and debugging support
- **Diagnostic Tools**: Comprehensive hardware testing and troubleshooting utilities

## 🏗️ Architecture Improvements

### Modular Node Design
The robot node now uses a clean, component-based architecture:

- **`dofbot_modular_node.py`**: MADSci-compliant REST interface with proper action patterns
- **`robot_arm_interface.py`**: High-level coordination between components
- **`components/`**: Modular components for camera, vision, movement, and calibration
- **`robot_arm_config.py`**: Centralized configuration management

### Benefits
- **Maintainability**: Each component has a single responsibility
- **Testability**: Components can be tested independently
- **Extensibility**: Easy to add new capabilities
- **MADSci Compliance**: Proper separation of operational actions vs setup tools
- **Error Handling**: Comprehensive safety checks and exception management

## 🚧 Development Status

This repository represents an active implementation of the MADSci framework architecture. The core functionality is operational with a robust, modular design.

**Currently Implemented:**
- Modular robot control with component-based architecture
- Full MADSci compliance with proper REST API patterns
- Camera integration and comprehensive calibration tools
- Vision-based object detection and tracking
- Data collection workflows with synchronized recording
- Comprehensive diagnostic and setup tools

**In Development:**
- Complete hybrid digital twin implementation
- Advanced multi-robot coordination
- Enhanced AI agent integration
- Comprehensive resource management features

## 🤝 Contributing

This project is under active development. The architecture design in `ARCHITECTURE.md` provides the roadmap for full implementation.

When contributing:
- Follow the modular component architecture
- Maintain MADSci compliance patterns
- Add appropriate tools for setup/maintenance tasks
- Update documentation for new features

## 📄 License

[Add your license information here]

## 🆘 Support

For issues related to:
- **MADSci Framework**: See the main MADSci repository
- **Hardware Setup**: Check the component setup sections and tools documentation
- **Architecture Questions**: Review `ARCHITECTURE.md`
- **Diagnostic Tools**: See `tools/README.md` for comprehensive troubleshooting guides
