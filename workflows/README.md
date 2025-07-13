# MEDAL-LAB Workflows

This directory contains workflow examples and scripts that demonstrate how to orchestrate robot operations using the MADSci framework. Each workflow showcases different capabilities and use cases for autonomous laboratory operations.

## 📁 Available Workflows

### `transfer.py` - Basic Transfer Operation
Demonstrates simple robot transfer operations between predefined positions using the modular node. This workflow shows the fundamental movement capabilities and serves as a starting point for understanding robot coordination.

**Key Features:**
- Basic robot movement coordination
- Position-to-position transfers
- Simple MADSci workflow patterns
- Error handling and status reporting

### `recording_workflow.py` - Data Collection with Recording
Synchronized camera recording during robot operations for training data collection. This workflow captures expert demonstrations by recording robot movements along with synchronized camera data.

**Key Features:**
- 10Hz synchronized data collection
- RGB, depth, and point cloud capture
- Trajectory data recording with joint states
- Automatic data organization in timestamped directories
- Integration with dual-pipeline camera system

**Data Output:**
- `captures/experiment_YYYYMMDD_HHMMSS/rgb_images/` - RGB camera frames
- `captures/experiment_YYYYMMDD_HHMMSS/depth_images/` - Depth camera data  
- `captures/experiment_YYYYMMDD_HHMMSS/point_clouds/` - 3D point cloud data
- `captures/experiment_YYYYMMDD_HHMMSS/trajectory_data.json` - Joint states and metadata

### `take_picture.py` - Single Image Capture
Workflow for capturing single images using the robot node's `capture_single_image` action. Useful for quick image capture, testing camera functionality, or collecting individual reference images.

**Key Features:**
- Single RGB image capture
- Automatic timestamped file naming
- Image metadata collection
- Quick camera functionality testing

**Data Output:**
- `captures/capture_YYYYMMDD_HHMMSS.jpg` - Single RGB image with timestamp

## 🚀 Running Workflows

### Prerequisites
- Robot node running on Jetson hardware
- Workcell manager running with proper configuration
- MADSci client environment set up

### Execution
```bash
# From within MADSci Docker container or configured environment
python workflows/transfer.py
python workflows/recording_workflow.py
python workflows/take_picture.py
```

## 🔧 Creating Custom Workflows

### Basic Workflow Pattern
All workflows follow the MADSci pattern:
1. Initialize workcell client
2. Define workflow steps
3. Execute actions on robot node
4. Handle results and errors
5. Clean up resources

### Best Practices
- Use proper error handling for robot actions
- Implement resource cleanup in finally blocks
- Validate action results before proceeding
- Use descriptive logging for debugging
- Follow MADSci action patterns consistently

## 📊 Data Management

### Automatic Git Exclusion
All experimental data is automatically excluded from git commits:
- `captures/experiment_*/` directories (recording data)
- `captures/capture_*.jpg` files (single images)

This protects your research data while allowing workflow scripts to be version controlled.

### Data Organization
- **Timestamped Directories**: Each recording session creates a unique timestamped directory
- **Structured Data**: RGB, depth, point cloud, and trajectory data are organized in separate subdirectories
- **Metadata Preservation**: Joint states, timestamps, and experimental parameters are saved alongside sensor data

## 🔍 Troubleshooting

### Common Issues
1. **Robot Node Connection**: Ensure robot node is running and accessible
2. **Workcell Manager**: Verify workcell manager is properly configured
3. **Camera Permissions**: Check camera access and udev rules
4. **Data Directory**: Ensure captures directory exists and is writable

### Debug Information
- Check robot node logs for action execution details
- Verify workcell manager status and resource availability
- Use diagnostic tools in `../tools/` directory for hardware validation
- Enable verbose logging in workflow scripts for detailed execution traces

## 📚 Related Documentation

- **Main README**: Complete system setup and overview
- **Nodes README**: Robot node architecture and technical details
- **Tools README**: Diagnostic and calibration tools
- **MADSci Documentation**: Framework-specific information and patterns
