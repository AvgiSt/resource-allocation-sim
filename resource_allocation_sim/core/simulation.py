"""Main simulation runner with configurable parameters."""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from ..utils.config import Config
from .agent import Agent
from .environment import Environment


class SimulationRunner:
    """
    Main simulation runner that orchestrates agents and environment.
    
    Attributes:
        config: Simulation configuration
        agents: List of simulation agents
        environment: Simulation environment
        results: Collected simulation results
    """

    def __init__(
        self, 
        config: Optional[Config] = None,
        custom_agent_factory: Optional[Callable] = None
    ):
        """
        Initialise simulation runner.
        
        Args:
            config: Simulation configuration
            custom_agent_factory: Optional custom agent creation function
        """
        self.config = config or Config()
        self.custom_agent_factory = custom_agent_factory
        
        self.agents: List[Agent] = []
        self.environment: Optional[Environment] = None
        self.results: Dict[str, Any] = {}
        
    def setup(self) -> None:
        """Set up simulation components."""
        # Create environment with relative capacity
        self.environment = Environment(
            num_resources=self.config.num_resources,
            relative_capacity=self.config.relative_capacity,
            num_agents=self.config.num_agents
        )
        
        # Create agents
        self.agents = self._create_agents()
        
    def _create_agents(self) -> List[Agent]:
        """Create agents based on configuration."""
        agents = []
        
        if self.custom_agent_factory:
            # Use custom agent factory
            for i in range(self.config.num_agents):
                agent = self.custom_agent_factory(i, self.config)
                agents.append(agent)
        else:
            # Use default agent creation
            init_method = self.config.agent_initialisation_method
            
            for i in range(self.config.num_agents):
                if init_method == "uniform":
                    agent = Agent(i, self.config.num_resources, self.config.weight)
                elif init_method == "dirichlet":
                    agent = Agent(i, self.config.num_resources, self.config.weight, "dirichlet")
                elif init_method == "softmax":
                    agent = Agent(i, self.config.num_resources, self.config.weight, "softmax")
                elif init_method == "custom":
                    agent = Agent(i, self.config.num_resources, self.config.weight, "custom")
                else:
                    raise ValueError(f"Unknown initialisation method: {init_method}")
                    
                agents.append(agent)
                
        return agents
        
    def run(self) -> Dict[str, Any]:
        """
        Run the simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        if not self.environment or not self.agents:
            self.setup()
            
        # Initialise results storage
        agent_results = {
            agent.id: {"prob": [], "action": []} 
            for agent in self.agents
        }
        
        # Run simulation iterations
        for iteration in range(self.config.num_iterations):
            # Agents select actions based on probabilities
            actions = [agent.select_action() for agent in self.agents]
            
            # Environment step
            costs = self.environment.step(actions)
            
            # Record agent states
            for agent in self.agents:
                agent_results[agent.id]["prob"].append(agent.probabilities.copy())
                agent_results[agent.id]["action"].append(actions[agent.id])
            
            # Update agent probabilities - each agent only gets cost for their selected resource
            for i, agent in enumerate(self.agents):
                selected_resource = actions[i]
                observed_cost = costs[selected_resource]
                agent.update_probabilities(selected_resource, observed_cost)
        
        # Compile results
        self.results = {
            "agent_results": agent_results,
            "environment_state": self.environment.get_state(),
            "final_consumption": self.environment.consumption.tolist(),
            "total_cost": self.environment.get_total_cost(),
            "config": self.config.to_dict()
        }
        
        return self.results
    
    def run_multiple_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """
        Run multiple episodes of the simulation.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            List of results from each episode
        """
        all_results = []
        
        for episode in range(num_episodes):
            # Reset components
            if self.environment:
                self.environment.reset()
            for agent in self.agents:
                agent.reset()
                
            # Run episode
            episode_results = self.run()
            episode_results["episode"] = episode
            all_results.append(episode_results)
            
        return all_results
    
    def get_agents(self) -> List[Agent]:
        """Get list of agents."""
        return self.agents
    
    def get_environment(self) -> Environment:
        """Get environment."""
        if self.environment is None:
            raise ValueError("Environment not initialised. Call setup() first.")
        return self.environment 
    