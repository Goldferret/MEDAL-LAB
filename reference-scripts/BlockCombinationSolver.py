#!/usr/bin/env python
# coding: utf-8

"""
Block Combination Solver (Fully Autonomous)

Problem: 
  Given 6 blocks of 4 colors (red, yellow, green, blue), and a hidden 'correct' combination

Solution: 
  Shuffle blocks into combinations at random until a matching combination is found
  - Generates a random 'correct' pattern automatically after scanning
  - Uses efficient cycle-based block movement from BlockProblem6Blocks4.py
  - Remembers past combinations to avoid repeating them
  - Runs completely autonomously with zero human interaction

Block Interchangeability:
  Blocks of the same color are treated as interchangeable. Since we work with color
  names (not individual block IDs), patterns are matched by color sequence only.
  Example: [red, blue, red] matches [red, blue, red] regardless of which specific
  red block is in which position.
"""

import rospy
import sys
import os
import random
from time import sleep, time
from moveit_commander.move_group import MoveGroupCommander

sys.path.append(os.path.join(os.path.dirname(__file__), '../../dofbot_pro_RGBDCam/scripts'))
from block_find import detect_blocks
from BlockPositions import (
    block_position_temp, block_position_temp_raised, block_position_temp_lowered,
    block_position_0, block_position_0_raised, block_position_0_lowered,
    block_position_1, block_position_1_raised, block_position_1_lowered,
    block_position_2, block_position_2_raised, block_position_2_lowered,
    block_position_3, block_position_3_raised, block_position_3_lowered,
    block_position_4, block_position_4_raised, block_position_4_lowered,
    block_position_5, block_position_5_raised, block_position_5_lowered
)


class BlockCombinationSolver:
    def __init__(self):
        print("\n" + "="*60)
        print("Block Combination Solver - Initializing")
        print("="*60)
        
        # Initialize ROS node
        rospy.init_node("block_combination_solver")
        print("ROS node initialized")
        
        # Initialize position arrays
        self._init_positions()
        
        # Initialize arm and gripper
        self._init_arm_group()
        self._init_gripper_group()
        
        # Gripper angles
        self.gripper_open = [-1.25, -1.25]
        self.gripper_closed = [-0.5, -0.5]
        
        # Solver state
        self.current_blocks = None
        self.correct_pattern = None
        self.tried_combinations = set()
        self.attempt_count = 0
        self.solution_found = False
        
        # Time tracking
        self.start_time = None
        self.end_time = None
        self.attempt_times = []
        
        print("="*60 + "\n")
    
    def _init_positions(self):
        """Initialize all block position arrays"""
        self.block_positions = [
            block_position_0, block_position_1, block_position_2,
            block_position_3, block_position_4, block_position_5
        ]
        
        self.block_positions_raised = [
            block_position_0_raised, block_position_1_raised, block_position_2_raised,
            block_position_3_raised, block_position_4_raised, block_position_5_raised
        ]
        
        self.block_positions_lowered = [
            block_position_0_lowered, block_position_1_lowered, block_position_2_lowered,
            block_position_3_lowered, block_position_4_lowered, block_position_5_lowered
        ]
        
        self.block_position_temp_raised = block_position_temp_raised
        self.block_position_temp_lowered = block_position_temp_lowered
    
    def _init_arm_group(self):
        """Initialize arm group commander"""
        self.arm_group = MoveGroupCommander("arm_group")
        self.arm_group.allow_replanning(True)
        self.arm_group.set_planning_time(5)
        self.arm_group.set_num_planning_attempts(10)
        self.arm_group.set_goal_position_tolerance(0.01)
        self.arm_group.set_goal_orientation_tolerance(0.01)
        self.arm_group.set_goal_tolerance(0.01)
        self.arm_group.set_max_velocity_scaling_factor(1.0)
        self.arm_group.set_max_acceleration_scaling_factor(1.0)
        print("Arm group configured")
    
    def _init_gripper_group(self):
        """Initialize gripper group commander"""
        self.gripper = MoveGroupCommander("grip_group")
        self.gripper.allow_replanning(True)
        self.gripper.set_planning_time(5)
        self.gripper.set_num_planning_attempts(10)
        self.gripper.set_goal_tolerance(0.01)
        self.gripper.set_max_velocity_scaling_factor(1.0)
        self.gripper.set_max_acceleration_scaling_factor(1.0)
        print("Gripper group configured")
    
    # ==================================================
    # Basic Movement Functions
    # ==================================================
    
    def execute_movement(self, move_group, target_joints, action_name=""):
        """Execute a movement with retry logic"""
        move_group.set_joint_value_target(target_joints)
        for i in range(5):
            plan_success, plan_points, plan_time, plan_errors = move_group.plan()
            
            if len(plan_points.joint_trajectory.points) != 0:
                move_group.execute(plan_points)
                return True
        
        return False
    
    def open_gripper(self):
        """Open the gripper"""
        return self.execute_movement(self.gripper, self.gripper_open, "Open Gripper")
    
    def close_gripper(self):
        """Close the gripper"""
        return self.execute_movement(self.gripper, self.gripper_closed, "Close Gripper")
    
    def move_to_init_position(self):
        """Move arm to initial position"""
        self.arm_group.set_named_target("init")
        self.arm_group.go()
        sleep(0.3)
    
    # ==================================================
    # Scanning Functions
    # ==================================================
    
    def scan_blocks(self):
        """Scan and detect current block arrangement"""
        print("\nScanning for blocks...")
        
        # Move to scan position
        scan_joints = [-0.07, 0.2, -1.5, -1.5, 0.0]
        self.execute_movement(self.arm_group, scan_joints, "Scan position")
        sleep(0.5)
        
        # Detect blocks
        colors = detect_blocks(timeout=5.0)
        self.current_blocks = colors[:6] if colors else [None, None, None, None, None, None]
        
        print(f"Detected blocks: {self.current_blocks}")
        return self.current_blocks
    
    # ==================================================
    # Combination Management Functions
    # ==================================================
    
    def combination_to_tuple(self, combination):
        """Convert combination list to hashable tuple"""
        return tuple(combination)
    
    def is_combination_tried(self, combination):
        """Check if combination has been tried before"""
        return self.combination_to_tuple(combination) in self.tried_combinations
    
    def mark_combination_tried(self, combination):
        """Mark combination as tried"""
        self.tried_combinations.add(self.combination_to_tuple(combination))
    
    def generate_random_combination(self):
        """Generate a random combination that hasn't been tried"""
        max_attempts = 1000
        for _ in range(max_attempts):
            new_combination = self.current_blocks[:]
            random.shuffle(new_combination)
            
            if not self.is_combination_tried(new_combination):
                return new_combination
        
        return None
    
    # ==================================================
    # Block Movement Functions
    # ==================================================
    
    def move_block(self, from_pos, to_pos):
        """Move a single block from one position to another"""
        # Get positions
        if from_pos == 'temp':
            from_raised = self.block_position_temp_raised
            from_lowered = self.block_position_temp_lowered
        else:
            from_raised = self.block_positions_raised[from_pos]
            from_lowered = self.block_positions_lowered[from_pos]
        
        if to_pos == 'temp':
            to_raised = self.block_position_temp_raised
            to_lowered = self.block_position_temp_lowered
        else:
            to_raised = self.block_positions_raised[to_pos]
            to_lowered = self.block_positions_lowered[to_pos]
        
        # Execute movement
        self.open_gripper()
        sleep(0.1)
        self.execute_movement(self.arm_group, from_raised, "")
        sleep(0.1)
        self.execute_movement(self.arm_group, from_lowered, "")
        sleep(0.1)
        self.close_gripper()
        sleep(0.2)
        self.execute_movement(self.arm_group, from_raised, "")
        sleep(0.1)
        self.execute_movement(self.arm_group, to_raised, "")
        sleep(0.1)
        self.execute_movement(self.arm_group, to_lowered, "")
        sleep(0.1)
        self.open_gripper()
        sleep(0.1)
        self.execute_movement(self.arm_group, to_raised, "")
        sleep(0.1)
    
    def build_position_mapping(self, current_state, target_state):
        """Build optimal mapping from current to target state"""
        source_indices = {}
        target_indices = {}
        
        # Group by color
        for i in range(len(current_state)):
            color = current_state[i]
            if color not in source_indices:
                source_indices[color] = []
            source_indices[color].append(i)
        
        for i in range(len(target_state)):
            color = target_state[i]
            if color not in target_indices:
                target_indices[color] = []
            target_indices[color].append(i)
        
        # Create mapping
        mapping = {}
        for color in source_indices:
            src_list = sorted(source_indices[color])
            tgt_list = sorted(target_indices[color])
            for src, tgt in zip(src_list, tgt_list):
                mapping[src] = tgt
        
        return mapping
    
    def find_permutation_cycles(self, mapping, state_length):
        """Find all cycles in the permutation"""
        visited = [False] * state_length
        cycles = []
        
        for start in range(state_length):
            if visited[start] or mapping[start] == start:
                visited[start] = True
                continue
            
            cycle = []
            pos = start
            while not visited[pos]:
                visited[pos] = True
                cycle.append(pos)
                pos = mapping[pos]
            
            if len(cycle) > 1:
                cycles.append(cycle)
        
        return cycles
    
    def generate_move_sequence(self, target_combination):
        """Generate optimal move sequence to reach target combination"""
        current_state = self.current_blocks[:]
        target_state = target_combination[:]
        moves = []
        
        # Build mapping and find cycles
        mapping = self.build_position_mapping(current_state, target_state)
        cycles = self.find_permutation_cycles(mapping, len(current_state))
        
        if not cycles:
            return moves
        
        # Move first block to temp to create empty space
        temp_source_pos = cycles[0][0]
        temp_block_color = current_state[temp_source_pos]
        
        moves.append((temp_source_pos, 'temp'))
        current_state[temp_source_pos] = None
        empty_pos = temp_source_pos
        
        # Rotate all cycles using the empty space
        for cycle in cycles:
            if empty_pos in cycle:
                empty_idx_in_cycle = cycle.index(empty_pos)
                
                for i in range(len(cycle)):
                    prev_idx = (empty_idx_in_cycle - 1 - i) % len(cycle)
                    source_pos = cycle[prev_idx]
                    
                    if source_pos == temp_source_pos:
                        continue
                    
                    dest_pos = empty_pos
                    
                    if current_state[source_pos] is not None:
                        moves.append((source_pos, dest_pos))
                        current_state[dest_pos] = current_state[source_pos]
                        current_state[source_pos] = None
                        empty_pos = source_pos
            else:
                # Connect this cycle to the empty space
                cycle_entry = cycle[0]
                
                if current_state[cycle_entry] is not None:
                    moves.append((cycle_entry, empty_pos))
                    current_state[empty_pos] = current_state[cycle_entry]
                    current_state[cycle_entry] = None
                    empty_pos = cycle_entry
                
                cycle_start_idx = cycle.index(empty_pos)
                for i in range(len(cycle) - 1):
                    prev_idx = (cycle_start_idx - 1 - i) % len(cycle)
                    source_pos = cycle[prev_idx]
                    dest_pos = empty_pos
                    
                    if current_state[source_pos] is not None:
                        moves.append((source_pos, dest_pos))
                        current_state[dest_pos] = current_state[source_pos]
                        current_state[source_pos] = None
                        empty_pos = source_pos
        
        # Place temp block back
        moves.append(('temp', empty_pos))
        current_state[empty_pos] = temp_block_color
        
        return moves
    
    def execute_combination(self, target_combination):
        """Execute physical rearrangement to target combination"""
        print(f"\nArranging blocks to: {target_combination}")
        
        move_sequence = self.generate_move_sequence(target_combination)
        
        print(f"Executing {len(move_sequence)} moves")
        
        for move_num, (from_pos, to_pos) in enumerate(move_sequence, 1):
            print(f"  Move {move_num}/{len(move_sequence)}: {from_pos} → {to_pos}")
            self.move_block(from_pos, to_pos)
        
        # Update current state
        self.current_blocks = target_combination[:]
        print("Rearrangement complete")
    
    # ==================================================
    # Solution Check Functions
    # ==================================================
    
    def generate_correct_pattern(self):
        """Generate a random 'correct' pattern from scanned blocks"""
        self.correct_pattern = self.current_blocks[:]
        random.shuffle(self.correct_pattern)
        print("\n" + "="*60)
        print("Generated correct pattern (hidden from solver):")
        print(f"  {self.correct_pattern}")
        print("="*60)
        return self.correct_pattern
    
    def combinations_match(self, combination1, combination2):
        """
        Check if two combinations match.
        Since we work with color names, blocks of same color are 
        automatically treated as interchangeable.
        """
        if len(combination1) != len(combination2):
            return False
        
        return combination1 == combination2
    
    def check_solution(self):
        """Check if current combination matches the correct pattern"""
        is_correct = self.combinations_match(self.current_blocks, self.correct_pattern)
        
        if is_correct:
            print("\n" + "!"*60)
            print("MATCH FOUND!")
            print(f"Current:  {self.current_blocks}")
            print(f"Target:   {self.correct_pattern}")
            print("!"*60)
        else:
            print(f"  No match. Current: {self.current_blocks}")
        
        return is_correct
    
    # ==================================================
    # Timing Statistics
    # ==================================================
    
    def _print_timing_statistics(self):
        """Print timing statistics for the experiment"""
        if self.start_time is None or self.end_time is None:
            return
        
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*60)
        print("TIMING STATISTICS")
        print("="*60)
        print(f"Total elapsed time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        if self.attempt_times:
            avg_time = sum(self.attempt_times) / len(self.attempt_times)
            print(f"Total shuffles/attempts: {len(self.attempt_times)}")
            print(f"Average time per shuffle: {avg_time:.2f} seconds")
            print(f"Fastest shuffle: {min(self.attempt_times):.2f} seconds")
            print(f"Slowest shuffle: {max(self.attempt_times):.2f} seconds")
        
        print("="*60)
    
    # ==================================================
    # Main Solver Loop
    # ==================================================
    
    def solve(self):
        """Main solving loop - fully autonomous"""
        print("\n" + "="*60)
        print("Starting Combination Search (Autonomous)")
        print("="*60)
        
        # Start timing
        self.start_time = time()
        
        # Initial scan
        self.scan_blocks()
        
        if not self.current_blocks or all(c is None for c in self.current_blocks):
            print("Error: No blocks detected")
            return False
        
        # Generate the hidden 'correct' pattern
        self.generate_correct_pattern()
        
        # Mark initial combination as tried
        self.mark_combination_tried(self.current_blocks)
        self.attempt_count = 1
        
        # Check if initial configuration is solution
        print(f"\nAttempt #{self.attempt_count}")
        print(f"Testing: {self.current_blocks}")
        attempt_start_time = time()
        if self.check_solution():
            attempt_time = time() - attempt_start_time
            self.attempt_times.append(attempt_time)
            self.end_time = time()
            print("\nSolution found on initial scan!")
            self.solution_found = True
            self._print_timing_statistics()
            return True
        attempt_time = time() - attempt_start_time
        self.attempt_times.append(attempt_time)
        
        # Search loop
        while not self.solution_found:
            self.attempt_count += 1
            attempt_start_time = time()
            
            print(f"\n{'='*60}")
            print(f"Attempt #{self.attempt_count}")
            print(f"Combinations tried so far: {len(self.tried_combinations)}")
            print(f"{'='*60}")
            
            # Generate new combination
            new_combination = self.generate_random_combination()
            
            if new_combination is None:
                print("Error: No more unique combinations to try")
                self.end_time = time()
                self._print_timing_statistics()
                return False
            
            # Mark as tried
            self.mark_combination_tried(new_combination)
            
            # Execute rearrangement
            self.execute_combination(new_combination)
            
            # Check solution
            if self.check_solution():
                attempt_time = time() - attempt_start_time
                self.attempt_times.append(attempt_time)
                self.end_time = time()
                
                print("\n" + "="*60)
                print("SOLUTION FOUND!")
                print(f"Correct combination: {self.current_blocks}")
                print(f"Total attempts: {self.attempt_count}")
                print(f"Total combinations tried: {len(self.tried_combinations)}")
                print("="*60)
                self.solution_found = True
                self._print_timing_statistics()
                return True
            
            attempt_time = time() - attempt_start_time
            self.attempt_times.append(attempt_time)
            print(f"  Attempt time: {attempt_time:.2f} seconds")
        
        return False


def main():
    try:
        solver = BlockCombinationSolver()
        
        # Move to initial position
        solver.move_to_init_position()
        
        # Run solver
        solver.solve()
        
        # Return to initial position
        solver.move_to_init_position()
        
        print("\nProgram complete")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
