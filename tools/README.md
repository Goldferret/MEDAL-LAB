# DOFBOT Pro Tools Directory

This directory contains standalone diagnostic, calibration, and testing tools for the DOFBOT Pro robot system. These tools are designed for setup, maintenance, and troubleshooting, separate from normal robot operations.

## Tools Overview

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

### 📷 Camera Testing Tools

#### `test_cameras_final.py`
**Purpose:** Comprehensive camera functionality testing and validation

**Features:**
- Live RGB and depth stream testing
- Frame rate and resolution verification
- Image quality assessment
- Stream synchronization testing
- Performance benchmarking
- Multiple capture method testing

**Usage:**
```bash
python3 test_cameras_final.py
```

**Test Coverage:**
- Color stream functionality
- Depth stream functionality
- Frame synchronization
- Capture reliability
- Performance metrics

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

#### `hsv_calibration_tool.py`
**Purpose:** Interactive HSV color range calibration for object detection

**Features:**
- Click-to-sample HSV values from objects
- Real-time HSV range adjustment with trackbars
- Live detection preview and testing
- Export calibrated ranges for robot use
- Support for standard DOFBOT Pro colored blocks

**Usage:**
```bash
python3 hsv_calibration_tool.py
```

**Supported Objects:**
- Red, green, blue, yellow cubes
- Rectangular prisms
- Custom colored objects

---

## Quick Start Guide

### 1. Initial Setup and Diagnostics
```bash
# Check camera hardware
python3 diagnose_orbbec_device.py

# Fix any detected issues
./fix_orbbec_device.sh

# Test camera functionality
python3 test_cameras_final.py
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
python3 hsv_calibration_tool.py

# Test vision-based robot actions
curl -X POST http://localhost:2000/action/scan_for_target \
  -H "Content-Type: application/json" \
  -d '{"object_type": "cube", "color": "red"}'
```

## Troubleshooting Workflow

### Camera Not Detected
1. Run `diagnose_orbbec_device.py` for detailed analysis
2. Apply fixes with `fix_orbbec_device.sh`
3. Verify with `test_cameras_final.py`
4. Check physical connections and USB power

### Poor Vision Performance
1. Recalibrate camera with `calibrate_depth_camera.py`
2. Validate calibration with `validate_camera_calibration.py`
3. Adjust HSV ranges with `hsv_calibration_tool.py`
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

### test_cameras_final.py
```bash
python3 test_cameras_final.py
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

### hsv_calibration_tool.py
```bash
python3 hsv_calibration_tool.py
# Interactive tool - no command line options
# Use mouse and trackbars for calibration
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
