"""
Resource Allocation Simulation Framework

A multi-agent simulation framework for studying resource allocation
with probability-based learning agents.
"""

__version__ = "1.0.0"
__author__ = "Avgi Stavrou"

# Core simulation components
from .core.agent import Agent
from .core.environment import Environment
from .core.simulation import SimulationRunner

# Evaluation tools
from .evaluation.metrics import (
    calculate_entropy,
    calculate_gini_coefficient,
    calculate_resource_utilization,
    calculate_total_cost
)

# Configuration
from .utils.config import Config

# Experiments
from .experiments.base_experiment import BaseExperiment
from .experiments.grid_search import GridSearchExperiment
from .experiments.parameter_sweep import ParameterSweepExperiment
from .experiments.capacity_analysis import CapacityAnalysisExperiment
from .experiments.comprehensive_study import ComprehensiveStudy

# Utilities
from .utils.io import save_results, load_results

# CLI
from .cli import cli

# Optional visualization imports with better error handling
__visualization_available__ = True
__ternary_available__ = False
__network_available__ = False

try:
    from .visualization.plots import plot_resource_distribution
    try:
        from .visualization.ternary import plot_ternary_distribution
        __ternary_available__ = True
    except ImportError:
        plot_ternary_distribution = None
    
    try:
        from .visualization.network import visualize_state_network
        __network_available__ = True
    except ImportError:
        visualize_state_network = None
        
except ImportError:
    __visualization_available__ = False
    plot_resource_distribution = None
    plot_ternary_distribution = None
    visualize_state_network = None

__all__ = [
    # Core
    "Agent",
    "Environment", 
    "SimulationRunner",
    "Config",
    
    # Experiments
    "BaseExperiment",
    "GridSearchExperiment",
    "ParameterSweepExperiment", 
    "CapacityAnalysisExperiment",
    "ComprehensiveStudy",
    
    # Evaluation
    "calculate_entropy",
    "calculate_gini_coefficient",
    "calculate_resource_utilization",
    "calculate_total_cost",
    
    # Utilities
    "save_results",
    "load_results",
    
    # CLI
    "cli",
    
    # Visualization (if available)
    "plot_resource_distribution",
    "plot_ternary_distribution",
    "visualize_state_network"
] 