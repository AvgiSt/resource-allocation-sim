"""Environment managing resources and cost calculations."""

import numpy as np
import math
from typing import List, Union, Dict, Any


class Environment:
    """
    Environment that models resources with different capacities and
    calculates costs based on resource consumption.
    
    Attributes:
        num_resources: Number of available resources
        capacity: Capacity of each resource
        consumption: Current consumption of each resource
        all_costs: History of all costs
    """

    def __init__(
        self, 
        num_resources: int, 
        capacity: Union[float, List[float], np.ndarray]
    ):
        """
        Initialize the environment.
        
        Args:
            num_resources: Number of available resources
            capacity: Capacity for each resource (scalar or array)
        """
        self.num_resources = num_resources
        
        # Handle different capacity input formats
        if isinstance(capacity, (int, float)):
            self.capacity = np.full(num_resources, float(capacity))
        else:
            self.capacity = np.array(capacity, dtype=float)
            
        if len(self.capacity) != num_resources:
            raise ValueError(
                f"Capacity length ({len(self.capacity)}) must match "
                f"number of resources ({num_resources})"
            )
        
        self.consumption = np.zeros(num_resources)
        self.all_costs: List[float] = []
        
    def step(self, actions: List[int]) -> np.ndarray:
        """
        Update environment based on agent actions and calculate costs.
        
        Args:
            actions: List of actions (resource indices) selected by agents
            
        Returns:
            Array of costs for each resource
        """
        # Reset consumption for new iteration
        self.consumption = np.zeros(self.num_resources)
        
        # Update consumption based on actions
        for action in actions:
            if 0 <= action < self.num_resources:
                self.consumption[action] += 1
        
        # Calculate costs
        costs = self._calculate_costs()
        
        # Store costs for history
        for i, (cost, consumption) in enumerate(zip(costs, self.consumption)):
            self.all_costs.append(cost * consumption)
            
        return costs
    
    def _calculate_costs(self) -> np.ndarray:
        """
        Calculate costs based on current consumption and capacity.
        
        Returns:
            Array of costs for each resource
        """
        costs = np.zeros(self.num_resources)
        
        for i, (consumption, capacity) in enumerate(zip(self.consumption, self.capacity)):
            if capacity == 0:
                # Infinite cost for zero capacity resources if used
                costs[i] = float('inf') if consumption > 0 else 0.0
            elif consumption <= capacity:
                costs[i] = 1.0  # Base cost when under capacity
            else:
                # FIXED: Exponential cost increase when over capacity
                costs[i] = math.exp((consumption / capacity) - 1)
                
        return costs
    
    def reset(self) -> None:
        """Reset environment to initial state."""
        self.consumption = np.zeros(self.num_resources)
        self.all_costs = []
        
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            'num_resources': self.num_resources,
            'capacity': self.capacity.tolist(),
            'consumption': self.consumption.tolist(),
            'total_cost_history': len(self.all_costs),
            'current_costs': self._calculate_costs().tolist()
        }
    
    def get_utilization(self) -> np.ndarray:
        """
        Get current resource utilization rates.
        
        Returns:
            Array of utilization rates (consumption / capacity)
        """
        return np.divide(
            self.consumption, 
            self.capacity, 
            out=np.zeros_like(self.consumption), 
            where=self.capacity != 0
        )
    
    def get_total_cost(self) -> float:
        """Get total accumulated cost."""
        return sum(self.all_costs) 