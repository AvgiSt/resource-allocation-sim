"""Agent implementation for resource allocation simulation."""

import numpy as np
from typing import List, Optional


class Agent:
    """
    An agent that learns to select resources based on observed costs.
    
    Uses probability-based learning to adapt resource selection over time.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        num_resources: int, 
        weight: float,
        initialisation_method: str = "uniform"
    ):
        """
        Initialise an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            num_resources: Number of available resources
            weight: Learning weight parameter (0-1)
            initialisation_method: How to initialise probabilities
        """
        self.id = agent_id
        self.num_resources = num_resources
        self.weight = weight
        self.initialisation_method = initialisation_method
        
        # Initialise with uniform distribution
        if initialisation_method == "uniform":
            self.probabilities = np.ones(num_resources) / num_resources
        elif initialisation_method == "dirichlet":
            # Use Dirichlet distribution for more varied initialisation
            self.probabilities = np.random.dirichlet(np.ones(num_resources))
        elif initialisation_method == "softmax":
            # Random initialisation with softmax normalisation
            logits = np.random.normal(0, 1, num_resources)
            self.probabilities = self._softmax(logits)
        else:
            raise ValueError(f"Unknown initialisation method: {initialisation_method}")
        
        # History tracking
        self.action_history = []
        self.probability_history = []
        self.cost_history = []
    
    def select_action(self) -> int:
        """
        Select a resource based on current probabilities.
        
        Returns:
            Selected resource index
        """
        action = np.random.choice(self.num_resources, p=self.probabilities)
        self.action_history.append(action)
        return action
    
    def update_probabilities(self, costs: List[float]) -> None:
        """
        Update selection probabilities based on observed costs.
        
        Args:
            costs: Cost for each resource
        """
        if len(costs) != self.num_resources:
            raise ValueError("Number of costs must match number of resources")
        
        # Store cost information
        self.cost_history.append(costs.copy())
        
        # Update probabilities using weighted average with inverse costs
        # Lower costs should lead to higher probabilities
        inverse_costs = 1.0 / (np.array(costs) + 1e-10)  # Avoid division by zero
        
        # Normalise inverse costs to get target probabilities
        target_probabilities = inverse_costs / np.sum(inverse_costs)
        
        # Update probabilities with learning weight
        self.probabilities = (
            (1 - self.weight) * self.probabilities + 
            self.weight * target_probabilities
        )
        
        # Ensure probabilities sum to 1 (numerical stability)
        self.probabilities = self.probabilities / np.sum(self.probabilities)
        
        # Store probability history
        self.probability_history.append(self.probabilities.copy())
    
    def get_state(self) -> dict:
        """
        Get current agent state.
        
        Returns:
            Dictionary containing agent state information
        """
        return {
            'id': self.id,
            'probabilities': self.probabilities.copy(),
            'action_history': self.action_history.copy(),
            'probability_history': [p.copy() for p in self.probability_history],
            'cost_history': [c.copy() for c in self.cost_history],
            'num_resources': self.num_resources,
            'weight': self.weight,
            'initialisation_method': self.initialisation_method
        }
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        # Reset probabilities based on initialisation method
        if self.initialisation_method == "uniform":
            self.probabilities = np.ones(self.num_resources) / self.num_resources
        elif self.initialisation_method == "dirichlet":
            self.probabilities = np.random.dirichlet(np.ones(self.num_resources))
        elif self.initialisation_method == "softmax":
            logits = np.random.normal(0, 1, self.num_resources)
            self.probabilities = self._softmax(logits)
        
        # Clear history
        self.action_history = []
        self.probability_history = []
        self.cost_history = []
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax function to logits."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return f"Agent(id={self.id}, resources={self.num_resources}, weight={self.weight})"


# Convenience functions for creating agents with different initialisation methods
def create_uniform_agent(agent_id: int, num_resources: int, weight: float) -> Agent:
    """Create agent with uniform initialisation."""
    return Agent(agent_id, num_resources, weight, "uniform")


def create_dirichlet_agent(agent_id: int, num_resources: int, weight: float) -> Agent:
    """
    Create agent with Dirichlet-initialised probabilities.
    
    This provides more diverse initial probability distributions.
    """
    return Agent(agent_id, num_resources, weight, "dirichlet")


def create_softmax_agent(agent_id: int, num_resources: int, weight: float) -> Agent:
    """
    Create agent with softmax-initialised probabilities.
    
    Uses random normal logits passed through softmax for initialisation.
    """
    return Agent(agent_id, num_resources, weight, "softmax") 