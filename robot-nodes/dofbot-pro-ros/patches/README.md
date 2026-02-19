# DOFBOT Pro ROS Workspace Patches

Modified files for `dofbot_pro_ws` that fix IK, MoveIt config, and add the ArUco detector.

**What's patched:**
- `kinematics.yaml` — TRAC-IK solver with `position_only_ik: true` (replaces KDL)
- `DOFBOT_Pro-V24.srdf` — Adds `Gripping_point_Link` to `arm_group` so IK targets fingertips, not wrist
- `demo.launch` — Increased MoveIt start tolerance to 0.05 rad
- `arm_driver.py` — Patched servo driver
- `aruco_marker_detector.py` — ArUco marker detection node

**Usage:**
```bash
cd ~/medal-lab/robot-nodes/dofbot-pro-ros/patches
bash install.sh
```

See `CALIBRATION_LOG.md` for full details on why each patch was needed.
