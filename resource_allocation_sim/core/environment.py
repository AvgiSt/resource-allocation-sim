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
        relative_capacity: Union[List[float], np.ndarray],
        num_agents: int
    ):
        """
        Initialise the environment.
        
        Args:
            num_resources: Number of available resources
            relative_capacity: Relative capacity limits for each resource (fraction of agents)
            num_agents: Number of agents in the system (used to calculate actual capacity)
        """
        self.num_resources = num_resources
        self.num_agents = num_agents
        self.relative_capacity = np.array(relative_capacity)
        
        # Calculate actual capacity from relative capacity
        self.capacity = self.relative_capacity * num_agents
            
        if len(self.relative_capacity) != num_resources:
            raise ValueError("Relative capacity array length must match number of resources")
        
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
        # Calculate costs based on mathematical model:
        # L(r,t) = 1 if consumption <= capacity (no congestion)
        # L(r,t) = exp(1 - consumption/capacity) if consumption > capacity (congestion penalty)
        costs = np.zeros(self.num_resources)
        for road in range(self.num_resources):
            if self.consumption[road] == 0:
                costs[road] = 0.0 
            elif self.consumption[road] <= self.capacity[road]:
                costs[road] = 1.0  # No congestion
            else:
                costs[road] = np.exp(1 - (self.consumption[road] / self.capacity[road]))  # Congestion penalty
        
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
            'num_agents': self.num_agents,
            'relative_capacity': self.relative_capacity.tolist(),
            'actual_capacity': self.capacity.tolist(),  # For backward compatibility
            'consumption': self.consumption.tolist(),
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
        return f"Environment(resources={self.num_resources}, agents={self.num_agents}, relative_capacity={self.relative_capacity.tolist()}, actual_capacity={self.capacity.tolist()})" 