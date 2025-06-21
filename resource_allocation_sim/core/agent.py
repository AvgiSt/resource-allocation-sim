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
        elif initialisation_method == "custom":
            # Custom initialisation - start with uniform, to be overridden by custom factory
            self.probabilities = np.ones(num_resources) / num_resources
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

    def update_probabilities(self, selected_resource: int, observed_cost: float) -> None:
        """
        Update selection probabilities based on observed cost for selected resource.
        
        Uses the mathematical model:
        λ = w_r * L(r,t)
        p(a|r) = λ·I + (1-λ)·p(a|r)
        
        Args:
            selected_resource: Index of the resource that was selected
            observed_cost: Cost observed for the selected resource (L(r,t))
        """
        # Store cost information (sparse representation)
        cost_info = {'resource': selected_resource, 'cost': observed_cost}
        self.cost_history.append(cost_info)
        
        # Calculate λ = w_r * L(r,t)
        # where w_r is the agent weight and L(r,t) is the observed cost
        lambda_factor = self.weight * observed_cost
        
        # Create unit vector I with 1 for selected resource, 0 for others
        unit_vector = np.zeros(self.num_resources)
        unit_vector[selected_resource] = 1.0
        
        # Apply probability update formula: p(a|r) = λ·I + (1-λ)·p(a|r)
        self.probabilities = lambda_factor * unit_vector + (1 - lambda_factor) * self.probabilities
        
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
            'cost_history': self.cost_history.copy(),  # Now contains dicts with resource and cost
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
        elif self.initialisation_method == "custom":
            # Custom initialisation - reset to uniform, custom factory should override
            self.probabilities = np.ones(self.num_resources) / self.num_resources
        
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