# DOFBOT Pro Calibration Tools

Camera calibration tools for DOFBOT Pro + Orbbec DCW2. Run scripts in sequence: generate board → Phase 1 (intrinsics) → Phase 2 (hand-eye).

## Quick Start

### 1. Generate ChArUco Board

```bash
python3 tools/generate_charuco.py
```

**Output:** `charuco_board_5x7_DICT_4X4_50.png`

**Print at 100% scale** (ACTUAL SIZE) on A4 matte paper. Verify squares are 35mm with calipers. Mount on rigid backing.

---

### 2. Phase 1: Intrinsic Calibration

Calibrates camera intrinsics (focal length, distortion).

```bash
# Start ROS services (MoveIT, arm driver, camera) - runs in tmux
cd /home/jetson/medal-lab
make ros-up

# Run calibration
cd /home/jetson/medal-lab/robot-nodes/dofbot-pro-ros
source ~/dofbot_pro_ws/devel/setup.bash
python3 tools/intrinsic_calibration.py
```

**Workflow:**
- Move ChArUco board to different positions/angles
- Press **SPACE** to capture (10-15 images)
- Press **Q** to finish and compute

**Output:** `calibration/camera_intrinsics.json`

**Target:** RMS error < 0.3 pixels (excellent), < 0.5 pixels (good)

---

### 3. Phase 2: Hand-Eye Calibration

Computes camera-to-robot transformation. **Mount ChArUco board on robot wrist** (not gripper).

```bash
# Start ROS services (MoveIT, arm driver, camera) - runs in tmux
cd /home/jetson/medal-lab
make ros-up

# Run calibration
cd /home/jetson/medal-lab/robot-nodes/dofbot-pro-ros
source ~/dofbot_pro_ws/devel/setup.bash
python3 tools/hand_eye_calibration.py
```

**Workflow (20 poses, ~45-60 min):**
1. Torque disabled → manually position robot
2. Press Enter #1 → joints lock
3. Press Enter #2 → capture image + pose
4. Retry if detection fails

**⚠️ CRITICAL:** Keep gripper roll (joint 6) at 0° throughout to avoid collision.

**Output:** `calibration/hand_eye_transform.json`

**Target:** Mean reprojection error < 1.0mm (excellent), < 2.0mm (acceptable)

---

## Common Issues

**Phase 1: "Too few markers detected"**
- Improve lighting, ensure board is flat and visible
- Move board closer or check print quality

**Phase 2: "ChArUco detection failed"**
- Adjust robot position for better camera visibility
- Verify board is still rigid and well-lit

**Phase 2: "FK service failed"**
- Check ROS is running: `rosservice list | grep kinemarics`
- Verify joint states: `rostopic echo /joint_states`

**High reprojection error (>2mm)**
- Collect more poses (18-20 successful)
- Increase rotational diversity
- Verify Phase 1 quality and joint 6 stayed at 0°

---

## Dependencies

- Python: `opencv-python`, `numpy`, `scipy`, `rospy`, `cv_bridge`
- ROS: Camera topic `/camera/color/image_raw`, FK service `/get_kinemarics`
- Arm controller: `/home/jetson/dofbot_ws-custom_scripts/scripts/arm_controller.py`
