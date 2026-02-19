# Overhead Camera Calibration for DOFBOT Pro

Complete guide for calibrating an overhead (eye-on-base) camera setup for vision-guided manipulation with the DOFBOT Pro robot arm.

## Overview

This calibration process has 5 phases:

0. **Install Dependencies** - Install TRAC-IK and apply patches
1. **Camera Intrinsics** - Calibrate focal length and lens distortion
2. **Hand-Eye Sample Collection** - Collect synchronized robot pose + marker detection pairs
3. **Kabsch Calibration** - Compute final transform using translation-only method
4. **Deploy Transform** - Publish calibration as static TF transform

**Total time:** ~60-90 minutes

---

## Phase 0: Install Dependencies and Apply Patches

**Purpose:** Install required software and apply patches before calibration.

### Install TRAC-IK Solver

The robot node uses TRAC-IK for inverse kinematics (better success rate than default KDL solver).

```bash
cd ~/dofbot_pro_ws/src
git clone https://github.com/traclabs/trac_ik.git
cd ~/dofbot_pro_ws
catkin_make
source devel/setup.bash
```

### Apply Patches

Apply patches that configure TRAC-IK, fix MoveIt settings, and add ArUco detector:

```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros/patches
bash install.sh
```

**What gets patched:**
- `kinematics.yaml` - TRAC-IK solver with position_only_ik
- `DOFBOT_Pro-V24.srdf` - Adds Gripping_point_Link to arm_group
- `demo.launch` - Increased MoveIt start tolerance, RViz disabled
- `arm_driver.py` - Patched servo driver
- `aruco_marker_detector.py` - ArUco marker detection node

### Rebuild Workspace

```bash
cd ~/dofbot_pro_ws
catkin_make
source devel/setup.bash
```

**Verify patches applied:**
```bash
grep -q "trac_ik" ~/dofbot_pro_ws/src/dofbot_pro_config/config/kinematics.yaml && echo "✓ TRAC-IK configured"
```

---

## Phase 1: Camera Intrinsics Calibration

### Hardware
- **Camera:** Orbbec DaBai DCW2 (or compatible)
- **Robot:** DOFBOT Pro with ROS/MoveIt
- **Mounting:** Rigid overhead camera mount looking down at workspace
- **Materials:**
  - ChArUco board: 5×7 grid, 36mm squares, 27mm markers, DICT_4X4_50
  - ArUco marker (ID 0, DICT_4X4_50) for end-effector attachment
  - Rigid backing (cardboard/foam board) for ChArUco board
  - Calipers for verifying print dimensions

### Software
- ROS Noetic
- OpenCV with ArUco support
- scipy (for Kabsch algorithm)
- easy_handeye (installed from source)

### Installation

**Install easy_handeye from source:**
```bash
cd ~/dofbot_pro_ws/src
git clone https://github.com/IFL-CAMP/easy_handeye.git
cd ~/dofbot_pro_ws
catkin_make
source devel/setup.bash
```

**Copy corrected launch file:**
```bash
cp ~/medal-lab/robot-nodes/dofbot-pro-ros/patches/easy_handeye_calibration.launch \
   ~/dofbot_pro_ws/src/dofbot_pro_config/launch/
```

**Copy ArUco detector:**
```bash
cp ~/medal-lab/robot-nodes/dofbot-pro-ros/patches/aruco_marker_detector.py \
   ~/dofbot_pro_ws/src/dofbot_pro_config/scripts/
chmod +x ~/dofbot_pro_ws/src/dofbot_pro_config/scripts/aruco_marker_detector.py
```

---

## Phase 1: Camera Intrinsics Calibration

**Purpose:** Calibrate camera's focal length and lens distortion parameters.

### Setup

1. **Print ChArUco board:**
   - Pattern: 5×7 grid, 36mm squares, 27mm markers, DICT_4X4_50
   - Print at **100% scale** (ACTUAL SIZE) on A4 paper
   - **Verify dimensions:** Use calipers to confirm squares are 36mm
   - Mount on rigid backing (cardboard or foam board)

2. **Start ROS services:**
   ```bash
   cd ~/medal-lab
   make ros-up
   ```

   **⚠️ IMPORTANT:** Stop the camera TF publisher (window 5) to avoid conflicts:
   ```bash
   make ros-attach
   # Press Ctrl+b then 5 (go to camera_tf window)
   # Press Ctrl+C to stop
   # Press Ctrl+b then d to detach
   ```

### Calibration Process

```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros
source ~/dofbot_pro_ws/devel/setup.bash
python3 tools/intrinsic_calibration.py
```

**Workflow:**
1. Position ChArUco board at various positions and angles in camera view
2. Press **ENTER** to capture each image (aim for 10-15 images)
3. Vary distance, rotation, and position for good coverage
4. Press **Q** when done to compute calibration

**Output:** `calibration/camera_intrinsics.json`

**Quality Targets:**
- ✅ **Excellent:** RMS error < 0.3 pixels
- ✅ **Good:** RMS error < 0.5 pixels
- ❌ **Poor:** RMS error > 0.5 pixels (recalibrate with more diverse images)

---

## Phase 2: Hand-Eye Sample Collection

**Purpose:** Collect synchronized robot pose + ArUco marker detection pairs for calibration.

### Setup

1. **Attach ArUco marker to end-effector:**
   - Use marker ID 0, DICT_4X4_50 dictionary
   - Size: 70mm (or adjust `marker_size` parameter in launch file)
   - Mount rigidly to robot end-effector (not gripper fingers)

2. **Position camera overhead:**
   - Camera should look down at workspace
   - Ensure marker is visible throughout robot's workspace

3. **Start ROS services** (if not already running):
   ```bash
   cd ~/medal-lab
   make ros-up
   ```

   **⚠️ IMPORTANT:** Stop camera TF publisher (window 5):
   ```bash
   make ros-attach
   # Press Ctrl+b then 5
   # Press Ctrl+C
   # Press Ctrl+b then d
   ```

### Launch Calibration

```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros
source ~/dofbot_pro_ws/devel/setup.bash
roslaunch dofbot_pro_config easy_handeye_calibration.launch
```

This launch file automatically starts:
- ArUco marker detector (publishes `camera_color_optical_frame` → `aruco_marker_frame` TF)
- easy_handeye calibration node (no GUI)

### Collect Samples

**Option A: Automated with predefined poses (recommended)**
```bash
python3 tools/hand_eye_calibration.py
```

This script:
- Loads predefined poses from `tools/calibration_poses.json`
- Moves robot to each pose via MoveIt
- Captures image + FK pose automatically
- Collects 15-20 samples

**Option B: Manual positioning**
```bash
python3 tools/easy_handeye_sample_collector.py
```

For each sample:
1. Manually position robot so marker is visible to camera
2. Press **ENTER** to capture sample
3. Repeat for 15-20 diverse poses

**Tips for good samples:**
- Vary marker position (center, edges, corners of workspace)
- Vary marker orientation (different roll/pitch/yaw angles)
- Ensure marker is clearly visible in all samples
- Aim for 15-20 successful samples

**Output:** Samples stored in `~/.ros/easy_handeye/dofbot_eye_to_hand_eye_on_base_samples.yaml`

---

## Phase 3: Kabsch Calibration (Final Transform)

**Purpose:** Compute `base_link` → `camera_color_optical_frame` transform using translation-only method.

### Why Kabsch?

Standard `cv2.calibrateHandEye()` requires reliable rotation estimates from ArUco marker detection. However:
- Small markers (70mm) at low resolution (640×480) have unreliable rotation estimates
- Bad rotations corrupt the entire calibration solve
- Kabsch algorithm uses **only translation** (position), which is reliable
- Translation is based on marker size + triangulation, not orientation

**This is a temporary solution** until upgrading to:
- Higher resolution (1920×1080)
- Larger marker (100mm)
- Better lighting conditions

### Run Calibration

```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros
python3 tools/kabsch_calibration.py --save
```

**What it does:**
1. Reads samples from `~/.ros/easy_handeye/dofbot_eye_to_hand_eye_on_base_samples.yaml`
2. Extracts translation-only point pairs:
   - Gripper position in base frame (from FK)
   - Marker position in camera frame (from ArUco detection)
3. Solves for optimal rigid transform using Kabsch algorithm
4. Saves result to `calibration/hand_eye_transform.yaml`

**Output:** `calibration/hand_eye_transform.yaml`

**Quality Targets:**
- ✅ **Excellent:** Mean residual < 5mm
- ✅ **Good:** Mean residual < 10mm
- ❌ **Poor:** Mean residual > 10mm (collect more diverse samples)

---

## Phase 4: Deploy Transform

**Purpose:** Publish calibration as static TF transform for runtime use.

### Extract Calibration Values

```bash
cat ~/medal-lab/robot-nodes/dofbot-pro-ros/calibration/hand_eye_transform.yaml
```

Example output:
```yaml
transformation:
  x: -0.0328
  y: 0.1170
  z: 0.6667
  qx: 0.998065
  qy: -0.058126
  qz: -0.011025
  qw: -0.019111
```

### Update Environment Variables

Edit `~/medal-lab/.env.global` and add/update:

```bash
# Camera Transform (base_link → camera_color_optical_frame)
CAMERA_TRANSFORM_X=-0.0328
CAMERA_TRANSFORM_Y=0.1170
CAMERA_TRANSFORM_Z=0.6667
CAMERA_TRANSFORM_QX=0.998065
CAMERA_TRANSFORM_QY=-0.058126
CAMERA_TRANSFORM_QZ=-0.011025
CAMERA_TRANSFORM_QW=-0.019111
```

### Restart ROS Services

```bash
cd ~/medal-lab
make ros-down
make ros-up
```

This will:
1. Start all ROS services (MoveIt, arm driver, camera)
2. Publish static TF transform: `base_link` → `camera_color_optical_frame` (window 5)

### Verify Transform

Check that the transform is being published:

```bash
make ros-attach
# Press Ctrl+b then 5 (go to camera_tf window)
# Should see: rosrun tf static_transform_publisher ...
# Press Ctrl+b then d to detach
```

### Install TRAC-IK Solver

The robot node uses TRAC-IK for inverse kinematics (better than default KDL solver). Install from source:

```bash
cd ~/dofbot_pro_ws/src
git clone https://github.com/traclabs/trac_ik.git
cd ~/dofbot_pro_ws
catkin_make
source devel/setup.bash
```

**Apply patches** (includes TRAC-IK configuration):
```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros/patches
bash install.sh
```

**Restart ROS services** to load TRAC-IK:
```bash
cd ~/medal-lab
make ros-down
make ros-up
```

---

## Verification

Test the calibration with vision-guided pick and place:

```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros
source ~/dofbot_pro_ws/devel/setup.bash
python3 tools/color_pick_and_place.py red
```

**Expected behavior:**
- Robot detects colored block in camera image
- Transforms pixel coordinates to base_link frame
- Successfully picks and places block

**If pick fails:**
- Check that Z coordinates are reasonable (0.02-0.05m for table height)
- Verify TF transform is publishing correctly
- Re-run calibration with more diverse samples

---

## Troubleshooting

### Phase 1: Intrinsic Calibration

**"Too few markers detected"**
- Improve lighting (avoid glare and shadows)
- Ensure board is flat and fully visible
- Move board closer to camera
- Check print quality (squares should be sharp)

**High RMS error (>0.5 pixels)**
- Collect more images (15-20)
- Increase diversity (more angles, distances, positions)
- Verify board dimensions with calipers
- Check that board is mounted rigidly

### Phase 2: Sample Collection

**"ArUco marker not detected"**
- Improve lighting conditions
- Ensure marker is not occluded
- Move robot to position marker more directly toward camera
- Verify marker ID and dictionary match (ID 0, DICT_4X4_50)

**"FK service failed"**
- Check ROS is running: `rosservice list | grep kinemarics`
- Verify joint states: `rostopic echo /joint_states`
- Restart ROS services: `make ros-down && make ros-up`

**"TF transform failed"**
- Verify ArUco detector is running: `rostopic echo /tf | grep aruco`
- Check camera intrinsics loaded correctly (see detector node output)
- Ensure camera_tf publisher (window 5) is stopped during calibration

### Phase 3: Kabsch Calibration

**High residual error (>10mm)**
- Collect more samples (aim for 18-20)
- Increase rotational diversity of samples
- Verify Phase 1 intrinsics quality (RMS < 0.5)
- Check that marker was rigidly attached (no wobble)

**"No samples found"**
- Verify samples were saved: `ls ~/.ros/easy_handeye/`
- Check easy_handeye namespace matches: `dofbot_eye_to_hand_eye_on_base`
- Re-run Phase 2 sample collection

### Phase 4: Deployment

**Pick and place fails with "IK failed"**
- Check Z coordinates are reasonable (not 0.6m+)
- Verify correct frame: `camera_color_optical_frame` (not `camera_link`)
- Confirm static TF publisher is running (window 5)
- Test transform: `rostopic echo /tf_static | grep camera`

**"TF lookup failed"**
- Restart ROS services to ensure TF publisher starts
- Check `.env.global` values are correct (no typos)
- Verify Makefile publishes to `camera_color_optical_frame`

---

## When to Recalibrate

Recalibrate if:
- Camera is moved or remounted
- Camera focus or zoom is adjusted
- Workspace layout changes significantly
- Pick accuracy degrades over time (>5mm error)
- Upgrading to higher resolution or larger marker

---

## Files Reference

**Calibration Scripts:**
- `tools/intrinsic_calibration.py` - Phase 1: Camera intrinsics
- `tools/hand_eye_calibration.py` - Phase 2: Automated sample collection
- `tools/kabsch_calibration.py` - Phase 3: Final transform computation

**Configuration:**
- `tools/calibration_poses.json` - Predefined robot poses for sample collection
- `patches/easy_handeye_calibration.launch` - Launch file with ArUco detector
- `patches/aruco_marker_detector.py` - ArUco marker detection node

**Outputs:**
- `calibration/camera_intrinsics.json` - Camera matrix and distortion coefficients
- `calibration/hand_eye_transform.yaml` - Final camera-to-base transform
- `~/.ros/easy_handeye/dofbot_eye_to_hand_eye_on_base_samples.yaml` - Collected samples

**Deployment:**
- `~/medal-lab/.env.global` - Environment variables for TF publisher
- Makefile window 5 - Static TF publisher (`base_link` → `camera_color_optical_frame`)

---

## Additional Resources

- [easy_handeye documentation](https://github.com/IFL-CAMP/easy_handeye)
- [OpenCV ArUco documentation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [Kabsch algorithm explanation](https://en.wikipedia.org/wiki/Kabsch_algorithm)
- [Hand-eye calibration theory](https://en.wikipedia.org/wiki/Hand_eye_calibration_problem)
