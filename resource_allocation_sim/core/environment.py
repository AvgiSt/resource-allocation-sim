"""Environment implementation for resource allocation simulation."""

import numpy as np
from typing import List, Union, Optional


class Environment:
    """
    Environment that manages resources and calculates costs.
    
    The environment tracks resource consumption and provides cost feedback
    to agents based on current resource loads.
    """
    
    def __init__(
        self, 
        num_resources: int, 
        capacity: Union[List[float], np.ndarray],
        cost_function: str = "quadratic"
    ):
        """
        Initialise the environment.
        
        Args:
            num_resources: Number of available resources
            capacity: Capacity limits for each resource
            cost_function: Type of cost function ('linear', 'quadratic', 'exponential')
        """
        self.num_resources = num_resources
        self.capacity = np.array(capacity)
        self.cost_function = cost_function
        
        if len(self.capacity) != num_resources:
            raise ValueError("Capacity array length must match number of resources")
        
        # Current resource consumption
        self.consumption = np.zeros(num_resources)
        
        # History tracking
        self.consumption_history = []
        self.cost_history = []
    
    def step(self, actions: List[int]) -> List[float]:
        """
        Process agent actions and return costs.
        
        Args:
            actions: List of resource selections from agents
            
        Returns:
            List of costs for each resource
        """
        # Reset consumption for this step
        self.consumption = np.zeros(self.num_resources)
        
        # Count selections for each resource
        for action in actions:
            if 0 <= action < self.num_resources:
                self.consumption[action] += 1
        
        # Calculate costs based on current consumption
        costs = self._calculate_costs()
        
        # Store history
        self.consumption_history.append(self.consumption.copy())
        self.cost_history.append(costs.copy())
        
        return costs.tolist()
    
    def _calculate_costs(self) -> np.ndarray:
        """
        Calculate costs based on current consumption and capacity.
        
        Returns:
            Array of costs for each resource
        """
        # Calculate load ratios (consumption / capacity)
        load_ratios = self.consumption / (self.capacity + 1e-12)  # Avoid division by zero
        
        if self.cost_function == "linear":
            costs = load_ratios
        elif self.cost_function == "quadratic":
            costs = load_ratios ** 2
        elif self.cost_function == "exponential":
            costs = np.exp(load_ratios) - 1
        else:
            raise ValueError(f"Unknown cost function: {self.cost_function}")
        
        # Ensure costs are non-negative
        costs = np.maximum(costs, 0.0)
        
        return costs
    
    def get_state(self) -> dict:
        """
        Get current environment state.
        
        Returns:
            Dictionary containing environment state information
        """
        return {
            'num_resources': self.num_resources,
            'capacity': self.capacity.tolist(),
            'consumption': self.consumption.tolist(),
            'cost_function': self.cost_function,
            'consumption_history': [c.tolist() for c in self.consumption_history],
            'cost_history': [c.tolist() for c in self.cost_history]
        }
    
    def reset(self) -> None:
        """Reset environment to initial state."""
        self.consumption = np.zeros(self.num_resources)
        self.consumption_history = []
        self.cost_history = []
    
    def get_current_costs(self) -> List[float]:
        """Get costs for current consumption levels."""
        return self._calculate_costs().tolist()
    
    def get_utilisation_ratios(self) -> List[float]:
        """Get current utilisation ratios (consumption / capacity)."""
        return (self.consumption / (self.capacity + 1e-12)).tolist()
    
    def is_overloaded(self, threshold: float = 1.0) -> List[bool]:
        """
        Check which resources are overloaded.
        
        Args:
            threshold: Overload threshold (default 1.0 = at capacity)
            
        Returns:
            List of boolean values indicating overload status
        """
        utilisation_ratios = self.consumption / (self.capacity + 1e-12)
        return (utilisation_ratios > threshold).tolist()
    
    def get_total_cost(self) -> float:
        """Get total cost across all resources."""
        costs = self._calculate_costs()
        return float(np.sum(costs))
    
    def __repr__(self) -> str:
        """String representation of environment."""
        return f"Environment(resources={self.num_resources}, capacity={self.capacity.tolist()})" 