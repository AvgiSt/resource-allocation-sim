"""Helper utilities and functions."""

import numpy as np
from typing import List, Tuple, Union
from pathlib import Path


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Path to directory
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalise_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Normalise probability array to sum to 1.
    
    Args:
        probs: Input probability array
        
    Returns:
        Normalised probability array
    """
    probs = np.array(probs)
    total = np.sum(probs)
    
    if total == 0:
        # If all probabilities are zero, return uniform distribution
        return np.ones_like(probs) / len(probs)
    
    return probs / total


def calculate_distribution_distance(
    distribution: Union[List[float], np.ndarray],
    capacity: Union[List[float], np.ndarray],
    num_agents: int
) -> float:
    """
    Calculate distance between current distribution and ideal distribution.
    
    Args:
        distribution: Current resource consumption distribution
        capacity: Resource capacities
        num_agents: Total number of agents
        
    Returns:
        Distance metric (lower is better)
    """
    distribution = np.array(distribution)
    capacity = np.array(capacity)
    
    # Normalise current distribution
    normalised_dist = distribution / num_agents
    
    # Calculate ideal distribution (proportional to capacity)
    if np.sum(capacity) > 0:
        normalised_cap = capacity / np.sum(capacity)
    else:
        normalised_cap = np.ones_like(capacity) / len(capacity)
    
    # Return L2 distance
    return np.linalg.norm(normalised_dist - normalised_cap)


def generate_parameter_grid(**param_ranges) -> List[dict]:
    """
    Generate parameter combinations for grid search.
    
    Args:
        **param_ranges: Dictionary of parameter names to lists of values
        
    Returns:
        List of parameter combinations
    """
    import itertools
    
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, combo)))
    
    return combinations


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def validate_capacity_format(
    capacity: Union[float, List[float], np.ndarray], 
    num_resources: int
) -> np.ndarray:
    """
    Validate and format capacity parameter.
    
    Args:
        capacity: Capacity specification
        num_resources: Number of resources
        
    Returns:
        Formatted capacity array
        
    Raises:
        ValueError: If capacity format is invalid
    """
    if isinstance(capacity, (int, float)):
        return np.full(num_resources, float(capacity))
    
    capacity_array = np.array(capacity, dtype=float)
    
    if len(capacity_array) != num_resources:
        raise ValueError(
            f"Capacity length ({len(capacity_array)}) must match "
            f"number of resources ({num_resources})"
        )
    
    if np.any(capacity_array < 0):
        raise ValueError("All capacities must be non-negative")
    
    return capacity_array 