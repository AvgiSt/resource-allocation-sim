"""Metrics for evaluating simulation performance."""

import numpy as np
from typing import List, Dict, Any, Union


def calculate_entropy(consumption: Union[List[float], np.ndarray]) -> float:
    """
    Calculate entropy of resource consumption distribution.
    
    Args:
        consumption: Resource consumption values
        
    Returns:
        Shannon entropy of the distribution
    """
    consumption = np.array(consumption, dtype=float)
    total_consumption = np.sum(consumption)
    
    if total_consumption == 0:
        return 0.0
    
    # Calculate probabilities (avoiding zero division)
    probabilities = consumption / total_consumption
    probabilities = probabilities[probabilities > 0]  # Remove zeros
    
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def calculate_gini_coefficient(consumption: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Gini coefficient of resource consumption distribution.
    
    Args:
        consumption: Resource consumption values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    consumption = np.array(consumption, dtype=float)
    consumption = np.sort(consumption)  # Sort in ascending order
    n = len(consumption)
    
    if n == 0 or np.sum(consumption) == 0:
        return 0.0
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(consumption)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def calculate_resource_utilisation(
    consumption: Union[List[float], np.ndarray],
    capacity: Union[List[float], np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Calculate resource utilization efficiency.
    
    Args:
        consumption: Resource consumption values
        capacity: Optional resource capacities for utilization rates
        
    Returns:
        Standard deviation of consumption (if no capacity) or utilization rates
    """
    consumption = np.array(consumption, dtype=float)
    
    if capacity is None:
        # Return standard deviation as efficiency measure
        return np.std(consumption)
    
    capacity = np.array(capacity, dtype=float)
    # Calculate utilization rates
    utilization = np.divide(
        consumption, 
        capacity, 
        out=np.zeros_like(consumption), 
        where=capacity != 0
    )
    return utilization


def calculate_convergence_speed(
    cost_history: List[float], 
    window_size: int = 10,
    threshold: float = 0.01
) -> int:
    """
    Calculate convergence speed based on cost variance.
    
    Args:
        cost_history: List of costs over time
        window_size: Window size for moving average
        threshold: Variance threshold for convergence
        
    Returns:
        Iteration number where convergence occurred, or total iterations
    """
    if len(cost_history) < window_size:
        return len(cost_history)
    
    costs = np.array(cost_history)
    
    # Calculate moving average
    moving_avg = np.convolve(costs, np.ones(window_size) / window_size, mode='valid')
    
    # Calculate variance of moving average in windows
    for i in range(len(moving_avg) - window_size + 1):
        window_var = np.var(moving_avg[i:i + window_size])
        if window_var < threshold:
            return i + window_size
    
    return len(cost_history)


def calculate_total_cost(
    consumption: Union[List[float], np.ndarray],
    capacity: Union[List[float], np.ndarray] = None
) -> float:
    """
    Calculate total system cost.
    
    Args:
        consumption: Resource consumption values
        capacity: Resource capacities
        
    Returns:
        Total cost value
    """
    consumption = np.array(consumption, dtype=float)
    
    if capacity is None:
        # Simple cost based on consumption
        return np.sum(consumption * np.exp(1 - consumption))
    
    capacity = np.array(capacity, dtype=float)
    total_cost = 0.0
    
    for i, (c, cap) in enumerate(zip(consumption, capacity)):
        if cap == 0:
            cost = 0.0
        elif c <= cap:
            cost = c * 1.0  # Base cost
        else:
            cost = c * np.exp(1 - c / cap)  # Exponential penalty
        total_cost += cost
    
    return total_cost


def calculate_system_metrics(
    results: Dict[str, Any],
    num_agents: int
) -> Dict[str, float]:
    """
    Calculate comprehensive system metrics from simulation results.
    
    Args:
        results: Simulation results dictionary
        num_agents: Number of agents in simulation
        
    Returns:
        Dictionary of calculated metrics
    """
    final_consumption = np.array(results.get('final_consumption', []))
    total_cost = results.get('total_cost', 0.0)
    
    metrics = {
        'entropy': calculate_entropy(final_consumption),
        'gini_coefficient': calculate_gini_coefficient(final_consumption),
        'resource_utilisation_std': calculate_resource_utilisation(final_consumption),
        'total_cost': total_cost,
        'mean_consumption': np.mean(final_consumption),
        'std_consumption': np.std(final_consumption),
        'max_consumption': np.max(final_consumption) if len(final_consumption) > 0 else 0,
        'min_consumption': np.min(final_consumption) if len(final_consumption) > 0 else 0
    }
    
    return metrics


def analyze_convergence_patterns(agent_results: Dict[int, Dict[str, List]]) -> Dict[str, Any]:
    """
    Analyze convergence patterns of agents.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        
    Returns:
        Dictionary with convergence analysis
    """
    convergence_data = {
        'agent_entropies': {},
        'convergence_times': {},
        'final_entropies': {}
    }
    
    for agent_id, data in agent_results.items():
        prob_history = data['prob']
        
        # Calculate entropy over time for this agent
        entropies = [calculate_entropy(probs) for probs in prob_history]
        convergence_data['agent_entropies'][agent_id] = entropies
        convergence_data['final_entropies'][agent_id] = entropies[-1] if entropies else 0
        
        # Find convergence time (when entropy drops below threshold)
        convergence_time = len(entropies)
        for i, entropy in enumerate(entropies):
            if entropy < 0.1:  # Low entropy threshold
                convergence_time = i
                break
        convergence_data['convergence_times'][agent_id] = convergence_time
    
    return convergence_data 