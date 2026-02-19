#!/bin/bash
# Install DOFBOT Pro patches into the ROS workspace.
# Run from: ~/medal-lab/robot-nodes/dofbot-pro-ros/patches/
#
# These patches apply fixes for:
#   - TRAC-IK with position_only_ik (kinematics.yaml)
#   - Gripping_point_Link in arm_group (SRDF)
#   - MoveIt start tolerance (demo.launch)
#   - Patched arm_driver (arm_driver.py)
#   - ArUco marker detector (aruco_marker_detector.py)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS=~/dofbot_pro_ws

echo "Installing patches from $SCRIPT_DIR into $WS..."

cp "$SCRIPT_DIR/arm_driver.py"              "$WS/src/dofbot_pro_info/scripts/arm_driver.py"
cp "$SCRIPT_DIR/demo.launch"                "$WS/src/dofbot_pro_config/launch/demo.launch"
cp "$SCRIPT_DIR/DOFBOT_Pro-V24.srdf"        "$WS/src/dofbot_pro_config/config/DOFBOT_Pro-V24.srdf"
cp "$SCRIPT_DIR/kinematics.yaml"            "$WS/src/dofbot_pro_config/config/kinematics.yaml"
cp "$SCRIPT_DIR/aruco_marker_detector.py"   "$WS/src/dofbot_pro_config/scripts/aruco_marker_detector.py"

echo "Done. Patches installed."
echo "Rebuild workspace if needed: cd $WS && catkin_make"
