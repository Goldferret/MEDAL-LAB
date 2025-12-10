#!/usr/bin/env python
# coding: utf-8

"""
Block Positions Configuration

This file contains all the joint angle definitions for block positions used
across multiple scripts (BlockProblem6Blocks4.py, 2x3Setup.py, BlockCombinationSolver.py).

Centralized location makes it easy to debug and adjust positions - changes here
will automatically affect all scripts that import from this module.

Position Types:
- Normal: Standard height for block positions
- Raised: Higher positions for collision-free traversal
- Lowered: Lower positions for reliable pickup/placement
"""

# ==================================================
# Pickup Position (where blocks are initially placed)
# ==================================================
pickup_position = [1.5, -0.8, -0.65, -1.5, 0.0]
pickup_position_raised = [1.5, -0.5, -0.65, -1.5, 0.0]
# pickup_position_lowered = [1.5, -0.925, -0.65, -1.5, 0.0]
pickup_position_lowered = [1.5, -0.8, -0.65, -1.5, 0.0]

# ==================================================
# Temporary Holding Position
# ==================================================
#block_position_temp = [0.55, -0.8, -0.5, -1.25, 0.25]
#block_position_temp_raised = [0.55, -0.5, -0.5, -1.25, 0.25]
#block_position_temp_lowered = [0.55, -0.825, -0.6, -1.25, 0.5]

block_position_temp = [-0.8, -0.1, -1.2, -1.2, 0.0]
block_position_temp_raised = [-0.8, 0.3, -1.2, -1.2, 0.0]
block_position_temp_lowered = [-0.8, -0.4, -1.2, -1.2, 0.0]

# ==================================================
# Grid Positions 0-5 (2x3 layout)
# ==================================================

# Position 0 (top row, right)
block_position_0 = [0.3, -0.8, -0.45, -1.35, 0.35]
block_position_0_raised = [0.3, -0.5, -0.45, -1.35, 0.35]
block_position_0_lowered = [0.3, -1.0, -0.25, -1.3, 0.35]

# Position 1 (top row, center)
block_position_1 = [0.0, -0.8, -0.45, -1.4, 0.0]
block_position_1_raised = [0.0, -0.5, -0.45, -1.4, 0.0]
block_position_1_lowered = [0.0, -0.925, -0.35, -1.35, 0.0]

# Position 2 (top row, left)
block_position_2 = [-0.3, -0.8, -0.45, -1.35, -0.35]
block_position_2_raised = [-0.3, -0.5, -0.45, -1.35, -0.35]
block_position_2_lowered = [-0.3, -1.0, -0.25, -1.37, -0.35]

# Position 3 (bottom row, right)
block_position_3 = [0.4, -0.4, -1.15, -1.3, 0.5]
block_position_3_raised = [0.4, 0.0, -1.15, -1.3, 0.5]
block_position_3_lowered = [0.4, -0.5, -1.05, -1.2, 0.5]

# Position 4 (bottom row, center)
block_position_4 = [0.0, -0.3, -1.2, -1.4, 0.0]
block_position_4_raised = [0.0, 0.1, -1.2, -1.4, 0.0]
block_position_4_lowered = [0.0, -0.4, -1.2, -1.2, 0.0]

# Position 5 (bottom row, left)
block_position_5 = [-0.5, -0.3, -1.2, -1.3, -0.5]
block_position_5_raised = [-0.5, 0.1, -1.2, -1.3, -0.5]
block_position_5_lowered = [-0.35, -0.4, -1.2, -1.15, -0.4]

