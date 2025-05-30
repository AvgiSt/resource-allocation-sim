"""Configuration management for simulations."""

import yaml
import numpy as np
from typing import Dict, Any, Union, List, Optional
from pathlib import Path


class Config:
    """
    Configuration class for simulation parameters.
    
    Supports loading from YAML files and provides sensible defaults.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialise configuration.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Override parameters
        """
        # Set defaults
        self._set_defaults()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
            
        # Apply any overrides
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def _set_defaults(self):
        """Set default configuration values."""
        # Simulation parameters
        self.num_iterations = 100
        self.num_agents = 5
        self.num_resources = 3
        self.weight = 0.6
        
        # Environment parameters
        self.capacity = [1.0, 1.0, 1.0]  # Equal capacity by default
        
        # Agent parameters
        self.agent_initialisation_method = "uniform"  # uniform, dirichlet, softmax
        
        # Experiment parameters
        self.num_episodes = 1
        self.random_seed = None
        
        # Output parameters
        self.save_results = True
        self.results_dir = "results"
        self.plot_results = True
        
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        for key, value in config_data.items():
            setattr(self, key, value)
            
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Auto-adjust capacity if num_resources changes
        if 'num_resources' in kwargs and hasattr(self, 'capacity'):
            if len(self.capacity) != self.num_resources:
                # Resize capacity array to match num_resources
                if len(self.capacity) > self.num_resources:
                    self.capacity = self.capacity[:self.num_resources]
                else:
                    # Extend with the last capacity value or 1.0
                    last_cap = self.capacity[-1] if self.capacity else 1.0
                    while len(self.capacity) < self.num_resources:
                        self.capacity.append(last_cap)
            
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
            
        if self.num_resources <= 0:
            raise ValueError("num_resources must be positive")
            
        if not 0 < self.weight < 1:
            raise ValueError("weight must be between 0 and 1")
            
        if len(self.capacity) != self.num_resources:
            raise ValueError("capacity length must match num_resources")
            
        if any(c < 0 for c in self.capacity):
            raise ValueError("all capacities must be non-negative")

    def __setattr__(self, name, value):
        """Override setattr to handle capacity auto-adjustment."""
        super().__setattr__(name, value)
        
        # Auto-adjust capacity when num_resources changes
        if name == 'num_resources' and hasattr(self, 'capacity'):
            if len(self.capacity) != self.num_resources:
                if len(self.capacity) > self.num_resources:
                    self.capacity = self.capacity[:self.num_resources]
                else:
                    last_cap = self.capacity[-1] if self.capacity else 1.0
                    while len(self.capacity) < self.num_resources:
                        self.capacity.append(last_cap)