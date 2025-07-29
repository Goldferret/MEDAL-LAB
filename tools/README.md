# DOFBOT Pro Tools Directory

This directory contains standalone diagnostic, calibration, and testing tools for the DOFBOT Pro robot system. These tools are designed for setup, maintenance, and troubleshooting, separate from normal robot operations.

## Tools Overview

### 🧪 Performance Testing Tools

#### `test_10hz_recording.py`
**Purpose:** Test the camera's ability to consistently capture synchronized RGB and depth data at 10Hz

**Features:**
- Tests 50 consecutive frame captures at 10Hz
- Monitors queue status and frame availability
- Measures actual FPS vs target 10Hz
- Reports success rate and identifies bottlenecks
- Essential for validating robot recording workflows

**Usage:**
```bash
cd /path/to/MEDAL-LAB
python3 tools/test_10hz_recording.py
```

**Expected Results:**
- Success rate: 94%+ (47+ successful captures out of 50)
- Actual FPS: ~10.0
- Queue stability: ~20-21 frames consistently

### 🔧 Hardware Diagnostic Tools

#### `diagnose_orbbec_device.py`
**Purpose:** Comprehensive diagnostic tool for Orbbec depth camera issues

**Features:**
- Device detection and enumeration
- USB connection diagnostics
- Permission and udev rules checking
- Firmware compatibility verification
- Process conflict detection
- Detailed troubleshooting guidance

**Usage:**
```bash
python3 diagnose_orbbec_device.py
```

**Common Issues Diagnosed:**
- "No device found" errors
- USB permissions problems
- Device conflicts
- Driver installation issues

---

#### `fix_orbbec_device.sh`
**Purpose:** Automated fix script for common Orbbec device detection issues

**Features:**
- Automatic LD_LIBRARY_PATH configuration
- USB permissions repair
- Process conflict resolution
- System configuration fixes
- Udev rules installation

**Usage:**
```bash
./fix_orbbec_device.sh
```

**What It Fixes:**
- Library path issues
- USB device permissions
- Conflicting processes
- Missing system configurations

---

### 🔄 Integration Testing Tools

#### `test_recording_scanning_integration.py`
**Purpose:** Test recording and scanning pipeline integration

**Features:**
- Pipeline switching validation
- State management testing
- Recording continuity verification
- Scanning mode functionality
- Error recovery testing

**Usage:**
```bash
python3 test_recording_scanning_integration.py
```

#### `test_queue_clearing.py`
**Purpose:** Test frame queue clearing during pipeline transitions

**Features:**
- Queue state monitoring
- Stale frame detection
- Fresh frame validation
- Pipeline transition timing
- Memory management verification

**Usage:**
```bash
python3 test_queue_clearing.py
```

---

### 🎯 Calibration Tools

#### `calibrate_depth_camera.py`
**Purpose:** Interactive camera calibration using checkerboard patterns

**Features:**
- Live camera preview with pattern detection
- Automatic corner detection and refinement
- Configurable checkerboard parameters
- Progress tracking and image saving
- Comprehensive calibration calculation
- JSON output compatible with robot node

**Usage:**
```bash
# Basic calibration
./calibrate_depth_camera.py

# Custom parameters
./calibrate_depth_camera.py --width 9 --height 6 --square-size 30.0 --num-images 20
```

**Requirements:**
- Printed checkerboard pattern
- Good lighting conditions
- Stable camera mount

---

#### `validate_camera_calibration.py`
**Purpose:** Validation and testing of existing camera calibration

**Features:**
- Parameter validation and reasonableness checks
- Live undistortion preview
- Checkerboard-based accuracy testing
- Reprojection error analysis
- Visual comparison tools

**Usage:**
```bash
# Basic validation
./validate_camera_calibration.py

# Full testing suite
./validate_camera_calibration.py --test-undistortion --test-checkerboard
```

---

#### `interactive_image_hsv_calibrator.py`
**Purpose:** Interactive HSV color range calibration tool with real-time preview

**Features:**
- Real-time HSV range adjustment with trackbars
- Live camera feed with color mask overlay
- Interactive threshold tuning for optimal detection
- Export calibrated ranges to configuration
- Support for complex colors like red (dual-range)
- Visual feedback for color detection quality

**Usage:**
```bash
python3 interactive_image_hsv_calibrator.py
```

**Controls:**
- Use trackbars to adjust HSV ranges in real-time
- Press 'q' to quit and save current settings
- Press 'r' to reset to default ranges
- Live preview shows original image and color mask

---

#### `test_depth_vision_detection.py`
**Purpose:** Test depth-first 3D object detection with real camera data

**Features:**
- Tests the complete depth-first detection pipeline
- 3D object segmentation with table plane fitting
- Shape classification and dimension validation
- Color validation on detected 3D objects
- Debug visualization showing detection process
- Coordinate translation verification (depth→RGB space)

**Usage:**
```bash
python3 test_depth_vision_detection.py
```

**What It Tests:**
- Depth image preprocessing and noise reduction
- Table plane fitting and object extraction
- 3D shape classification (cubes vs rectangular prisms)
- Color matching within detected object regions
- Coordinate space translation accuracy

---

#### `test_rotated_cube_detection.py`
**Purpose:** Specialized test for detecting rotated cubes at various angles

**Features:**
- Tests cube detection at different rotation angles
- Validates dimension estimation for rotated objects
- Aspect ratio analysis for diamond-shaped projections
- Rotation-aware validation thresholds
- Performance metrics for rotated vs face-on cubes

**Usage:**
```bash
python3 test_rotated_cube_detection.py
```

**Validation Focus:**
- 45° rotated cubes (diamond projection)
- Perspective distortion handling
- Dimension estimation accuracy
- Shape confidence scoring

---

#### `object_detection_calibration.py`
**Purpose:** Comprehensive object detection calibration and validation tool

**Features:**
- Multi-object detection testing
- Calibration parameter optimization
- Detection accuracy measurement
- Threshold validation across scenarios
- Performance benchmarking

**Usage:**
```bash
python3 object_detection_calibration.py
```

----

#### `position_cycler.py`
**Purpose:** Cycles robot through scan positions for systematic HSV calibration

**Features:**
- Automated position cycling through [180°, 135°, 90°, 45°, 0°] servo angles
- Direct Arm_Lib.py integration with proper torque management
- Interactive Enter-to-advance interface for controlled calibration
- Perfect companion tool for `interactive_image_hsv_calibrator.py`
- Safe torque enable/disable on startup/shutdown

**Usage:**
```bash
cd /path/to/MEDAL-LAB/tools
python3 position_cycler.py
```

**Workflow:**
1. Robot moves to first position [180, 110, 5, 0, 90]
2. Press Enter to advance to next position in sequence
3. Use with HSV calibration tool in parallel for comprehensive color sampling
4. Quit safely with 'Q' - automatically disables robot torque

**Integration with HSV Calibration:**
- Run `position_cycler.py` in one terminal
- Run `interactive_image_hsv_calibrator.py` in another terminal
- Advance robot positions while sampling colors at each angle
- Captures lighting variations across all robot viewing positions

---

## Quick Start Guide

### 1. Initial Setup and Diagnostics
```bash
# Check camera hardware
python3 diagnose_orbbec_device.py

# Fix any detected issues
./fix_orbbec_device.sh

# Test camera functionality
```

### 2. Camera Calibration
```bash
# Calibrate depth camera (requires printed checkerboard)
./calibrate_depth_camera.py --num-images 15

# Validate calibration quality
./validate_camera_calibration.py --test-undistortion
```

### 3. Vision System Setup
```bash
# Calibrate HSV ranges for your lighting
python3 interactive_image_hsv_calibrator.py

# Test vision-based robot actions
curl -X POST http://localhost:2000/action/scan_for_target \
  -H "Content-Type: application/json" \
  -d '{"object_type": "cube", "color": "red"}'
```

## Troubleshooting Workflow

### Camera Not Detected
1. Run `diagnose_orbbec_device.py` for detailed analysis
2. Apply fixes with `fix_orbbec_device.sh`
4. Check physical connections and USB power

### Poor Vision Performance
1. Recalibrate camera with `calibrate_depth_camera.py`
2. Validate calibration with `validate_camera_calibration.py`
3. Adjust HSV ranges with `interactive_image_hsv_calibrator.py`
4. Test under actual working conditions

### Robot Actions Failing
1. Check camera status: `curl http://localhost:2000/state`
2. Verify calibration files are in robot's data directory
3. Test individual components with diagnostic tools
4. Review robot logs for specific error messages

## File Outputs

### Calibration Data
- `calibration_data/camera_calibration.json` - Camera intrinsic parameters
- `calibration_data/calibration_image_*.jpg` - Calibration reference images
- `hsv_ranges_calibrated.json` - Color detection parameters

### Diagnostic Reports
- Console output with detailed analysis
- Temporary test images for verification
- Performance metrics and benchmarks

### Integration Files
- JSON configuration files compatible with robot node
- Backup copies with timestamps
- Validation test results

## Tool Dependencies

### Required Python Packages
```bash
pip install pyorbbecsdk opencv-python numpy
```

### System Requirements
- Ubuntu/Linux with USB 3.0 support
- Orbbec SDK installed and configured
- Proper udev rules for device permissions
- Sufficient USB power (3.0 recommended)

### Hardware Requirements
- Orbbec DaiBai DCW2 depth camera
- USB 3.0 connection
- Printed checkerboard pattern (for calibration)
- DOFBOT Pro colored blocks (for HSV calibration)

## Best Practices

### Environment Setup
1. **Lighting:** Use consistent, diffuse lighting for calibration
2. **Stability:** Ensure camera is securely mounted
3. **Power:** Use USB 3.0 ports with adequate power
4. **Interference:** Avoid USB hubs and extension cables

### Calibration Quality
1. **Camera Calibration:** Aim for <0.5 pixel reprojection error
2. **HSV Ranges:** Test under actual working lighting conditions
3. **Validation:** Always validate calibration before deployment
4. **Maintenance:** Recalibrate if vision accuracy degrades

### Troubleshooting Approach
1. **Start with Hardware:** Use diagnostic tools first
2. **Systematic Testing:** Test each component individually
3. **Document Issues:** Save diagnostic outputs for reference
4. **Incremental Fixes:** Apply one fix at a time and retest

## Command Reference

### diagnose_orbbec_device.py
```bash
python3 diagnose_orbbec_device.py
# No arguments - runs full diagnostic suite
```

### fix_orbbec_device.sh
```bash
./fix_orbbec_device.sh
# No arguments - applies all common fixes
```

```bash
# No arguments - runs comprehensive camera tests
```

### calibrate_depth_camera.py
```bash
./calibrate_depth_camera.py [OPTIONS]

Options:
  --width INT          Checkerboard width in corners (default: 10)
  --height INT         Checkerboard height in corners (default: 7)
  --square-size FLOAT  Square size in millimeters (default: 25.0)
  --num-images INT     Number of calibration images (default: 15)
  --delay FLOAT        Delay between captures in seconds (default: 2.0)
  --output-dir PATH    Output directory (default: ./calibration_data)
```

### validate_camera_calibration.py
```bash
./validate_camera_calibration.py [OPTIONS]

Options:
  --calibration-file PATH  Path to calibration file
  --test-undistortion      Show undistortion preview
  --test-checkerboard      Test with live checkerboard
  --duration FLOAT         Test duration in seconds (default: 10.0)
```

### interactive_image_hsv_calibrator.py
```bash
python3 interactive_image_hsv_calibrator.py
# Interactive tool - no command line options
# Use trackbars for real-time HSV calibration
```

### position_cycler.py
```bash
python3 position_cycler.py
# Interactive tool - no command line options
# Press Enter to cycle positions, Q to quit
# Requires robot hardware connection
```

## Support and Maintenance

### Regular Maintenance
- Run diagnostics monthly or after system changes
- Recalibrate camera if moved or after major updates
- Update HSV ranges if lighting conditions change
- Validate calibration before important experiments

### Getting Help
1. Check tool output messages and error codes
2. Review this README for troubleshooting steps
3. Run diagnostic tools for detailed analysis
4. Check robot logs for integration issues

### Contributing
When adding new tools to this directory:
1. Follow the naming convention: `verb_noun_tool.py`
2. Include comprehensive docstrings and help text
3. Add executable permissions for main tools
4. Update this README with tool description
5. Test thoroughly before committing
