"""Visualization tools for simulation results."""

from .plots import (
    plot_resource_distribution,
    plot_convergence_comparison,
    plot_parameter_sensitivity
)

try:
    from .ternary import plot_ternary_distribution
except ImportError:
    # mpltern not available
    plot_ternary_distribution = None

try:
    from .network import visualize_state_network  
except ImportError:
    # networkx not available
    visualize_state_network = None

__all__ = [
    "plot_resource_distribution",
    "plot_convergence_comparison", 
    "plot_parameter_sensitivity",
    "plot_ternary_distribution",
    "visualize_state_network"
] 