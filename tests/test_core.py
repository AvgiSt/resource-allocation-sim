"""Tests for core components."""

import pytest
import numpy as np
from resource_allocation_sim.core.agent import Agent
from resource_allocation_sim.core.environment import Environment
from resource_allocation_sim.utils.config import Config


class TestAgent:
    """Test Agent class."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = Agent(0, 3, 0.5)
        assert agent.id == 0
        assert agent.num_resources == 3
        assert agent.weight == 0.5
        assert len(agent.probabilities) == 3
        assert abs(sum(agent.probabilities) - 1.0) < 1e-10

    def test_select_action(self):
        """Test action selection."""
        agent = Agent(0, 3, 0.5)
        action = agent.select_action()
        assert 0 <= action < 3
        
    def test_update_probabilities(self):
        """Test probability updates."""
        agent = Agent(0, 3, 0.5)
        initial_probs = agent.probabilities.copy()
        
        # Update with some costs
        costs = [1.0, 2.0, 0.5]
        agent.update_probabilities(costs)
        
        # Probabilities should have changed
        assert not np.allclose(initial_probs, agent.probabilities)
        # Should still sum to 1
        assert abs(sum(agent.probabilities) - 1.0) < 1e-10


class TestEnvironment:
    """Test Environment class."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = Environment(3, [1.0, 1.0, 1.0])
        assert env.num_resources == 3
        assert len(env.capacity) == 3
        
    def test_step(self):
        """Test environment step function."""
        env = Environment(3, [1.0, 1.0, 1.0])
        costs = env.step([0, 1, 2])
        assert len(costs) == 3
        assert all(cost >= 0 for cost in costs)
        
    def test_reset(self):
        """Test environment reset."""
        env = Environment(3, [1.0, 1.0, 1.0])
        env.step([0, 1, 2])  # Do some actions
        env.reset()
        assert all(c == 0 for c in env.consumption)


class TestConfig:
    """Test Config class."""
    
    def test_initialization(self):
        """Test config initialization."""
        config = Config()
        assert config.num_agents > 0
        assert config.num_resources > 0
        assert 0 < config.weight < 1
        
    def test_validation(self):
        """Test configuration validation."""
        config = Config()
        config.validate()  # Should not raise
        
        # Test invalid configurations
        config.num_agents = -1
        with pytest.raises(ValueError, match="num_agents must be positive"):
            config.validate()
            
        config.num_agents = 5  # Reset
        config.num_resources = 0
        with pytest.raises(ValueError, match="num_resources must be positive"):
            config.validate()
            
    def test_capacity_auto_adjustment(self):
        """Test automatic capacity adjustment."""
        config = Config()
        original_resources = config.num_resources
        
        # Change num_resources - capacity should auto-adjust
        config.num_resources = 5
        assert len(config.capacity) == 5
        
        # Reduce resources
        config.num_resources = 2
        assert len(config.capacity) == 2 