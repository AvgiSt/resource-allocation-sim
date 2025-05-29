"""Agent implementation with probability-based learning."""

import numpy as np
from typing import List, Optional


class Agent:
    """
    Agent that selects actions based on probability distributions and
    updates these probabilities based on observed costs.
    
    Attributes:
        id: Unique identifier for the agent
        num_resources: Number of available resources
        probabilities: Current probability distribution over actions
        action: Last selected action
        weight: Learning rate parameter
    """

    def __init__(
        self, 
        agent_id: int, 
        num_resources: int, 
        weight: float,
        initial_probabilities: Optional[np.ndarray] = None
    ):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            num_resources: Number of available resources
            weight: Learning rate parameter (0 < weight < 1)
            initial_probabilities: Optional initial probability distribution
        """
        self.id = agent_id
        self.num_resources = num_resources
        self.weight = weight
        self.action: Optional[int] = None
        
        if initial_probabilities is not None:
            self.probabilities = np.array(initial_probabilities, dtype=float)
        else:
            # Initialize with uniform distribution
            self.probabilities = np.full(num_resources, 1.0 / num_resources)
        
        # Ensure probabilities sum to 1
        self.probabilities = self.probabilities / np.sum(self.probabilities)

    def select_action(self) -> int:
        """
        Select an action based on current probability distribution.
        
        Returns:
            Selected action (resource index)
        """
        action = np.random.choice(self.num_resources, p=self.probabilities)
        self.action = action
        return action

    def update_probabilities(self, observed_cost: float) -> None:
        """
        Update probability distribution based on observed cost.
        
        Args:
            observed_cost: Cost observed for the last action
        """
        if self.action is None:
            raise ValueError("No action has been selected yet")
            
        # Create identity vector for selected action
        id_vector = np.zeros(self.num_resources)
        id_vector[self.action] = 1.0
        
        # FIXED: Decrease probability when cost is high
        # Use (1 - normalized_cost) as the target weight
        max_reasonable_cost = 10.0  # Reasonable upper bound
        normalized_cost = min(observed_cost, max_reasonable_cost) / max_reasonable_cost
        lambda_ = self.weight * (1 - normalized_cost)
        
        # Update: move towards action if cost is low, away if cost is high
        self.probabilities = (
            lambda_ * id_vector + (1 - lambda_) * self.probabilities
        )
        
        # Ensure probabilities remain valid
        self.probabilities = np.maximum(self.probabilities, 1e-10)
        self.probabilities = self.probabilities / np.sum(self.probabilities)

    def reset(self) -> None:
        """Reset agent to initial state."""
        self.probabilities = np.full(self.num_resources, 1.0 / self.num_resources)
        self.action = None

    def get_state(self) -> dict:
        """Get current agent state."""
        return {
            'id': self.id,
            'probabilities': self.probabilities.tolist(),
            'last_action': self.action,
            'weight': self.weight
        }

    @classmethod
    def from_dirichlet(
        cls, 
        agent_id: int, 
        num_resources: int, 
        weight: float,
        alpha: Optional[np.ndarray] = None
    ) -> 'Agent':
        """
        Create agent with Dirichlet-distributed initial probabilities.
        
        Args:
            agent_id: Unique identifier
            num_resources: Number of resources
            weight: Learning rate
            alpha: Dirichlet parameters (default: uniform)
            
        Returns:
            Agent with Dirichlet-initialized probabilities
        """
        if alpha is None:
            alpha = np.ones(num_resources)
        initial_probs = np.random.dirichlet(alpha)
        return cls(agent_id, num_resources, weight, initial_probs)

    @classmethod
    def from_softmax(
        cls,
        agent_id: int,
        num_resources: int, 
        weight: float,
        temperature: float = 1.0
    ) -> 'Agent':
        """
        Create agent with softmax-distributed initial probabilities.
        
        Args:
            agent_id: Unique identifier
            num_resources: Number of resources
            weight: Learning rate
            temperature: Softmax temperature parameter
            
        Returns:
            Agent with softmax-initialized probabilities
        """
        logits = np.random.randn(num_resources) / temperature
        initial_probs = np.exp(logits) / np.sum(np.exp(logits))
        return cls(agent_id, num_resources, weight, initial_probs) 