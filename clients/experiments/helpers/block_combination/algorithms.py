"""
Algorithms for block combination solver experiment.

Includes:
- Random target generation
- Combination tracking (avoid repeats)
- Cycle-based permutation algorithm for efficient block rearrangement
"""
import random


def generate_target(current):
    """
    Generate random target arrangement different from current.
    
    Args:
        current: Current block arrangement (list of colors)
        
    Returns:
        list: Target arrangement (shuffled, different from current)
    """
    target = current.copy()
    while target == current:
        random.shuffle(target)
    return target


def combination_to_tuple(combination):
    """
    Convert combination list to hashable tuple for tracking.
    
    Args:
        combination: List of colors
        
    Returns:
        tuple: Hashable representation
    """
    return tuple(combination)


def is_combination_tried(combination, tried_combinations):
    """
    Check if combination has been tried before.
    
    Args:
        combination: List of colors to check
        tried_combinations: Set of tried combination tuples
        
    Returns:
        bool: True if already tried
    """
    return combination_to_tuple(combination) in tried_combinations


def mark_combination_tried(combination, tried_combinations):
    """
    Mark combination as tried.
    
    Args:
        combination: List of colors
        tried_combinations: Set to update
    """
    tried_combinations.add(combination_to_tuple(combination))


def generate_random_combination(current_blocks, tried_combinations, max_attempts=1000):
    """
    Generate a random combination that hasn't been tried.
    
    Args:
        current_blocks: Current block arrangement
        tried_combinations: Set of tried combinations
        max_attempts: Maximum attempts to find untried combination
        
    Returns:
        list: New untried combination, or None if all tried
    """
    for _ in range(max_attempts):
        new_combination = current_blocks[:]
        random.shuffle(new_combination)
        
        if not is_combination_tried(new_combination, tried_combinations):
            return new_combination
    
    return None


def build_position_mapping(current_state, target_state):
    """
    Build optimal mapping from current to target state.
    
    Handles blocks of same color by creating deterministic mapping.
    
    Args:
        current_state: Current arrangement
        target_state: Target arrangement
        
    Returns:
        dict: Mapping from source index to target index
    """
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


def find_permutation_cycles(mapping, state_length):
    """
    Find all cycles in the permutation.
    
    A cycle is a sequence of positions where each maps to the next,
    and the last maps back to the first.
    
    Args:
        mapping: Position mapping dict
        state_length: Number of positions
        
    Returns:
        list: List of cycles (each cycle is a list of position indices)
    """
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


def generate_move_sequence(current_state, target_state):
    """
    Generate optimal move sequence using cycle-based algorithm.
    
    Strategy:
    1. Build position mapping (current -> target)
    2. Find cycles in the permutation
    3. Move first block to temp (creates empty space)
    4. Rotate all cycles using the empty space
    5. Place temp block back
    
    Args:
        current_state: Current arrangement
        target_state: Target arrangement
        
    Returns:
        list: Move sequence as list of (from_pos, to_pos) tuples
             Positions are indices (0-5) or 'temp'
    """
    # Work with copies to avoid modifying originals
    working_state = current_state[:]
    moves = []
    
    # Build mapping and find cycles
    mapping = build_position_mapping(current_state, target_state)
    cycles = find_permutation_cycles(mapping, len(current_state))
    
    if not cycles:
        # No moves needed - already at target
        return moves
    
    # Move first block to temp to create empty space
    temp_source_pos = cycles[0][0]
    temp_block_color = working_state[temp_source_pos]
    
    moves.append((temp_source_pos, 'temp'))
    working_state[temp_source_pos] = None
    empty_pos = temp_source_pos
    
    # Rotate all cycles using the empty space
    for cycle in cycles:
        if empty_pos in cycle:
            # Empty space is already in this cycle - rotate it
            empty_idx_in_cycle = cycle.index(empty_pos)
            
            for i in range(len(cycle)):
                prev_idx = (empty_idx_in_cycle - 1 - i) % len(cycle)
                source_pos = cycle[prev_idx]
                
                if source_pos == temp_source_pos:
                    continue
                
                dest_pos = empty_pos
                
                if working_state[source_pos] is not None:
                    moves.append((source_pos, dest_pos))
                    working_state[dest_pos] = working_state[source_pos]
                    working_state[source_pos] = None
                    empty_pos = source_pos
        else:
            # Connect this cycle to the empty space
            cycle_entry = cycle[0]
            
            if working_state[cycle_entry] is not None:
                moves.append((cycle_entry, empty_pos))
                working_state[empty_pos] = working_state[cycle_entry]
                working_state[cycle_entry] = None
                empty_pos = cycle_entry
            
            # Now rotate this cycle
            cycle_start_idx = cycle.index(empty_pos)
            for i in range(len(cycle) - 1):
                prev_idx = (cycle_start_idx - 1 - i) % len(cycle)
                source_pos = cycle[prev_idx]
                dest_pos = empty_pos
                
                if working_state[source_pos] is not None:
                    moves.append((source_pos, dest_pos))
                    working_state[dest_pos] = working_state[source_pos]
                    working_state[source_pos] = None
                    empty_pos = source_pos
    
    # Place temp block back into empty position
    moves.append(('temp', empty_pos))
    working_state[empty_pos] = temp_block_color
    
    return moves

