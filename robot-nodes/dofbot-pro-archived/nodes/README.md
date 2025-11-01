# MEDAL-LAB Robot Nodes

This directory contains the robot control system for MEDAL-LAB, implementing a modular, component-based architecture for DOFBOT Pro robotic arm control with integrated camera systems.

## 🏗️ Architecture Overview

The robot node system uses a clean, modular architecture that separates concerns and provides robust, MADSci-compliant robot control:

```
┌─────────────────────────────────────────────────────────────────┐
│                    dofbot_modular_node.py                       │
│              (MADSci REST API Interface)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                robot_arm_interface.py                           │
│            (High-Level Coordination Layer)                      │
└─────┬─────────┬─────────┬─────────┬─────────┬───────────────────┘
      │         │         │         │         │
┌─────▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼─────────┐
│ Camera  │ │Vision │ │Movement│ │Calib. │ │Configuration│
│Manager  │ │Detect.│ │Control │ │Manager│ │   Manager   │
└─────────┘ └───────┘ └───────┘ └───────┘ └─────────────┘
```

## 📁 File Structure

```
nodes/
├── README.md                      # This file - comprehensive technical documentation
├── dofbot_modular_node.py         # MADSci-compliant REST API interface
├── robot_arm_interface.py         # High-level robot coordination layer
├── robot_arm_config.py            # Centralized configuration management
├── default.node.yaml             # Default node configuration
├── Arm_Lib.py                     # Low-level arm control library
└── components/                    # Modular component architecture
    ├── camera_manager.py          # Dual-pipeline camera operations
    ├── vision_detector.py         # Computer vision and object detection
    ├── movement_controller.py     # Robot movement and servo control
    ├── calibration_manager.py     # Camera and system calibration
    └── experiment_logger.py       # Experiment data logging and persistence
```

## 🔧 Component Architecture

### Core Components

#### `dofbot_modular_node.py` - MADSci REST Interface
- **Purpose**: MADSci-compliant REST API providing standardized robot actions
- **Responsibilities**:
  - Action endpoint definitions with proper MADSci patterns
  - Parameter validation and error handling
  - Resource cleanup on action failures
  - State management and status reporting
- **Key Features**:
  - Automatic resource cleanup helper (`_cleanup_on_failure`)
  - Comprehensive parameter validation
  - Proper MADSci action result patterns
  - Integration with all robot components

#### `robot_arm_interface.py` - Coordination Layer
- **Purpose**: High-level coordination between all robot components
- **Responsibilities**:
  - Component lifecycle management
  - Action-level operations with intelligent pipeline switching
  - Recording session management
  - Cross-component coordination
- **Key Features**:
  - Dual-pipeline camera system coordination
  - Action-level scanning with fresh frame guarantee
  - Integrated recording and movement coordination
  - Resource cleanup and error recovery

### Component Modules

#### `components/camera_manager.py` - Dual-Pipeline Camera System
- **Purpose**: Manages Orbbec camera operations with optimized pipelines
- **Architecture**:
  - **Recording Pipeline**: Optimized for high-quality 10Hz data collection
  - **Scanning Pipeline**: Optimized for speed and responsiveness
  - **Action-Level Switching**: Intelligent pipeline selection per operation
- **Key Features**:
  - Fresh frame guarantee through pipeline switching
  - 30-frame queue for continuous operation
  - Synchronized RGB, depth, and point cloud capture
  - Debug visualization with automatic frame saving
  - Comprehensive error handling and recovery

#### `components/vision_detector.py` - Computer Vision
- **Purpose**: Advanced object detection and visual processing with hybrid approach
- **Architecture**:
  - **Hybrid Detection**: Combines Yahboom's proven methods with MEDAL-LAB precision
  - **Multi-Stage Pipeline**: Preprocessing → Morphological ops → Detection → Validation
  - **Enhanced Debugging**: Comprehensive debug visualization with actual object sizing
- **Capabilities**:
  - HSV-based color detection with configurable ranges
  - Object shape recognition (cube, rectangular prism) with geometric validation
  - Dual-range red color support (handles HSV hue wraparound)
  - Real-time debug visualization with mask and contour saving
  - Radius-based detection visualization (shows actual object size)
- **Key Features**:
  - Configurable HSV color ranges with edge filtering
  - Automatic debug frame saving with comprehensive metadata
  - Robust contour detection with confidence scoring
  - Integration with experiment logger for coordinated data flow
  - Enhanced error handling with detailed logging

#### `components/movement_controller.py` - Robot Movement
- **Purpose**: Low-level robot movement and servo control
- **Capabilities**:
  - Individual joint control with angle validation
  - Coordinated multi-joint movements
  - Torque state management
  - Movement recording integration
- **Key Features**:
  - Safety checks and angle validation
  - Smooth movement execution
  - Integration with recording system
  - Error recovery and state management

#### `components/calibration_manager.py` - System Calibration
- **Purpose**: Camera and system calibration management
- **Capabilities**:
  - Camera intrinsic and extrinsic calibration
  - Calibration data persistence
  - Validation and verification workflows
- **Key Features**:
  - Automated calibration workflows
  - Calibration data validation
  - Integration with camera system
  - Persistent calibration storage

#### `components/experiment_logger.py` - Experiment Data Management
- **Purpose**: Centralized experiment data logging and persistence
- **Architecture**:
  - **Coordinated Logging**: Integrates with all system components
  - **Structured Data**: JSON-based experiment metadata and event logging
  - **Debug Integration**: Automatic debug image saving with proper organization
- **Capabilities**:
  - Scanning event logging with comprehensive metadata
  - Debug image saving with timestamped organization
  - Experiment summary generation and persistence
  - 10Hz recording coordination and trajectory data management
- **Key Features**:
  - JSON data persistence with structured schemas
  - Automatic directory creation and file organization
  - Integration with camera manager for synchronized data flow
  - Error handling with graceful degradation
  - Comprehensive logging for experiment reproducibility

### Configuration Management

#### `robot_arm_config.py` - Centralized Configuration
- **Purpose**: Single source of truth for all robot configuration
- **Includes**:
  - Vision thresholds and HSV color ranges
  - Movement parameters and joint limits
  - Camera settings and pipeline configurations
  - Scanning positions and timing parameters
- **Benefits**:
  - Environment variable support
  - Type validation with Pydantic
  - Easy configuration updates
  - Consistent settings across components

## 🚀 Available Actions

The robot node provides the following MADSci-compliant actions:

### Movement Actions
- **`move_joint`**: Move individual robot joint by specified angle
- **`move_all_joints`**: Move all joints to specified angles simultaneously
- **`grabber_position`**: Move robot to predefined grabber position
- **`transfer`**: Transfer items between specified locations

### Vision and Detection Actions
- **`scan_for_target`**: Scan working area to find and face specified colored object
- **`center_on_target`**: Center camera on colored object using PID control

### Camera Actions
- **`capture_single_image`**: Capture single RGB image from camera
- **`test_camera_capture`**: Test camera capture functionality

### Recording Actions
- **`start_recording`**: Begin recording expert trajectory data
- **`stop_recording`**: Stop recording and save trajectory data

### Status Actions
- **`get_robot_status`**: Get comprehensive robot status information
- **`reset_movement_state`**: Reset movement state if stuck

## 🔄 Dual-Pipeline System

### Recording Pipeline
- **Optimization**: High-quality, synchronized data collection
- **Frame Rate**: Consistent 10Hz performance
- **Data Types**: RGB, depth, point cloud, and trajectory data
- **Use Cases**: Expert demonstration recording, training data collection

### Scanning Pipeline
- **Optimization**: Speed and responsiveness for real-time detection
- **Frame Rate**: Variable, optimized for detection speed
- **Data Types**: RGB frames for vision processing
- **Use Cases**: Object detection, target scanning, real-time vision

### Pipeline Switching
- **Action-Level**: Automatic pipeline selection based on operation type
- **Fresh Frame Guarantee**: Pipeline switching ensures current position data
- **Queue Management**: Intelligent frame queue clearing and refilling
- **Performance**: Optimized for each specific use case

## 🛡️ Error Handling and Resource Management

### Automatic Resource Cleanup
- **Recording Sessions**: Automatic cleanup on action failures
- **Pipeline Management**: Proper pipeline shutdown and restart
- **Memory Management**: Queue clearing and buffer management
- **Error Recovery**: Graceful degradation and recovery

### Safety Features
- **Parameter Validation**: Comprehensive input validation
- **Joint Limits**: Angle validation and safety checks
- **Torque Management**: Automatic torque state management
- **Exception Handling**: Robust error handling throughout

## 🧪 Testing and Validation

### Component Testing
Each component can be tested independently:
- Camera system: Use tools in `../tools/` directory
- Movement system: Individual joint testing available
- Vision system: Debug visualization automatically enabled
- Integration: Comprehensive integration tests available

### Debug Features
- **Automatic Debug Frames**: Vision processing frames saved automatically
- **Color Mask Visualization**: HSV detection masks saved for troubleshooting
- **Comprehensive Logging**: Detailed logging throughout all components
- **Status Reporting**: Real-time status information available

## 🔍 Troubleshooting

### Common Issues
1. **Camera Connection**: Check USB connection and udev rules
2. **Permission Errors**: Ensure user is in video group
3. **Import Errors**: Verify pyorbbecsdk PYTHONPATH is set correctly
4. **Movement Issues**: Check servo connections and power supply

### Debug Tools
- Use diagnostic scripts in `../tools/` directory
- Enable debug visualization in vision components
- Check comprehensive logs for detailed error information
- Use status actions to verify component states

## 🚀 Extending the System

### Adding New Components
1. Create new component in `components/` directory
2. Follow existing component patterns
3. Integrate with `robot_arm_interface.py`
4. Add configuration to `robot_arm_config.py`
5. Expose actions through `dofbot_modular_node.py`

### Adding New Actions
1. Define action in `dofbot_modular_node.py`
2. Use proper MADSci action patterns
3. Add parameter validation
4. Include resource cleanup handling
5. Update this documentation

## 📚 Related Documentation

- **Main README**: High-level system overview and setup
- **Tools README**: Diagnostic and testing tools
- **Architecture Document**: Complete system architecture design
- **MADSci Documentation**: Framework-specific information
