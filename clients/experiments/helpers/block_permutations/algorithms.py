"""Algorithms for block permutation experiment."""
import itertools
import random


def generate_target(current):
    """
    Generate random target arrangement different from current.
    
    Args:
        current: Current block arrangement
        
    Returns:
        list: Target arrangement (shuffled, different from current)
    """
    target = current.copy()
    while target == current:
        random.shuffle(target)
    return target


def generate_permutations(colors):
    """
    Generate all permutations in random order.
    
    Args:
        colors: List of block colors
        
    Returns:
        list: All permutations in randomized order
    """
    perms = list(itertools.permutations(colors))
    perms = [list(p) for p in perms]
    random.shuffle(perms)
    return perms


def calculate_swaps(current, target):
    """
    Calculate minimal swap sequence to transform current into target.
    
    Args:
        current: Current arrangement
        target: Target arrangement
        
    Returns:
        list: Swap sequence as [(pos_a, pos_b), ...]
        
    Note:
        Algorithm from BlockProblem4Blocks.py get_solution_path()
    """
    working = current.copy()
    swaps = []
    
    for i in range(len(working)):
        if working[i] == target[i]:
            continue
        
        for j in range(i + 1, len(working)):
            if working[j] == target[i]:
                working[i], working[j] = working[j], working[i]
                swaps.append((i, j))
                break
    
    return swaps
