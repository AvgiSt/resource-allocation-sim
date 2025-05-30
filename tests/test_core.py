"""Tests for core components."""

import unittest
import numpy as np
from resource_allocation_sim.core.agent import Agent
from resource_allocation_sim.core.environment import Environment
from resource_allocation_sim.core.simulation import SimulationRunner
from resource_allocation_sim.utils.config import Config


class TestAgent(unittest.TestCase):
    """Test Agent class functionality."""
    
    def test_initialisation(self):
        """Test agent initialisation."""
        agent = Agent(0, 3, 0.5)
        self.assertEqual(agent.id, 0)
        self.assertEqual(agent.num_resources, 3)
        self.assertEqual(agent.weight, 0.5)
        self.assertEqual(len(agent.probabilities), 3)
        np.testing.assert_almost_equal(np.sum(agent.probabilities), 1.0)
    
    def test_action_selection(self):
        """Test action selection."""
        agent = Agent(0, 3, 0.5)
        action = agent.select_action()
        self.assertIn(action, [0, 1, 2])
    
    def test_probability_update(self):
        """Test probability updates."""
        agent = Agent(0, 3, 0.5)
        initial_probs = agent.probabilities.copy()
        
        # Update with costs
        costs = [1.0, 0.5, 2.0]  # Resource 1 has lowest cost
        agent.update_probabilities(costs)
        
        # Probability for resource 1 should increase
        self.assertGreater(agent.probabilities[1], initial_probs[1])
    
    def test_reset(self):
        """Test agent reset."""
        agent = Agent(0, 3, 0.5)
        agent.select_action()
        agent.update_probabilities([1.0, 0.5, 2.0])
        
        agent.reset()
        self.assertEqual(len(agent.action_history), 0)
        self.assertEqual(len(agent.probability_history), 0)


class TestEnvironment(unittest.TestCase):
    """Test Environment class functionality."""
    
    def test_initialisation(self):
        """Test environment initialisation."""
        env = Environment(3, [1.0, 1.5, 0.8])
        self.assertEqual(env.num_resources, 3)
        np.testing.assert_array_equal(env.capacity, [1.0, 1.5, 0.8])
    
    def test_step(self):
        """Test environment step."""
        env = Environment(3, [1.0, 1.0, 1.0])
        actions = [0, 1, 0, 2]  # 2 agents select resource 0, 1 selects resource 1, 1 selects resource 2
        
        costs = env.step(actions)
        self.assertEqual(len(costs), 3)
        self.assertEqual(env.consumption[0], 2)  # Resource 0 selected twice
        self.assertEqual(env.consumption[1], 1)  # Resource 1 selected once
        self.assertEqual(env.consumption[2], 1)  # Resource 2 selected once
    
    def test_cost_calculation(self):
        """Test cost calculation."""
        env = Environment(2, [1.0, 2.0])
        actions = [0, 0, 1]  # Overload resource 0
        
        costs = env.step(actions)
        # Resource 0 should have higher cost due to overload
        self.assertGreater(costs[0], costs[1])


class TestSimulationRunner(unittest.TestCase):
    """Test SimulationRunner class functionality."""
    
    def test_initialisation(self):
        """Test simulation runner initialisation."""
        config = Config()
        runner = SimulationRunner(config)
        self.assertEqual(runner.config, config)
    
    def test_setup(self):
        """Test simulation setup."""
        config = Config()
        config.num_agents = 5
        config.num_resources = 3
        
        runner = SimulationRunner(config)
        runner.setup()
        
        self.assertEqual(len(runner.agents), 5)
        self.assertIsNotNone(runner.environment)
    
    def test_run(self):
        """Test simulation run."""
        config = Config()
        config.num_agents = 3
        config.num_resources = 2
        config.num_iterations = 10
        
        runner = SimulationRunner(config)
        results = runner.run()
        
        self.assertIn('agent_results', results)
        self.assertIn('final_consumption', results)
        self.assertIn('total_cost', results)


if __name__ == '__main__':
    unittest.main() 