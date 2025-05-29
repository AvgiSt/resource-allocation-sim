"""Ternary plot utilities for 3-resource visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import mpltern
    HAS_MPLTERN = True
except ImportError:
    HAS_MPLTERN = False


def plot_ternary_distribution(
    agent_results: Dict[int, Dict[str, List]],
    save_path: Optional[str] = None,
    title: str = "Agent Distribution Evolution"
) -> plt.Figure:
    """
    Plot agent distributions on ternary diagram.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        save_path: Optional path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Extract final distributions
    distributions = []
    for agent_id, data in agent_results.items():
        if data['prob']:
            final_prob = data['prob'][-1]
            if len(final_prob) == 3:
                distributions.append(final_prob)
    
    if not distributions:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, 'No 3-resource data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    distributions = np.array(distributions)
    
    if HAS_MPLTERN:
        # Use mpltern for proper ternary plots
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'ternary'})
        
        scatter = ax.scatter(distributions[:, 0], distributions[:, 1], distributions[:, 2],
                           alpha=0.7, s=50, c=range(len(distributions)), cmap='viridis')
        
        ax.set_tlabel('Resource 1')
        ax.set_llabel('Resource 2') 
        ax.set_rlabel('Resource 3')
        ax.set_title(title)
        
        plt.colorbar(scatter, ax=ax, label='Agent ID')
        
    else:
        # Fallback to manual ternary projection
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define triangle vertices
        triangle = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3)/2]])
        
        # Normalize distributions
        distributions = distributions / distributions.sum(axis=1, keepdims=True)
        
        # Project to 2D
        projected = np.dot(distributions, triangle)
        
        # Draw triangle
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', linewidth=2)
        
        # Plot points
        scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                           alpha=0.7, s=50, c=range(len(distributions)), cmap='viridis')
        
        # Add labels
        label_offset = 0.05
        ax.text(triangle[0, 0] - label_offset, triangle[0, 1] - label_offset, 
                'Resource 2', ha='center', va='top', fontsize=12)
        ax.text(triangle[1, 0] + label_offset, triangle[1, 1] - label_offset, 
                'Resource 3', ha='center', va='top', fontsize=12)
        ax.text(triangle[2, 0], triangle[2, 1] + label_offset, 
                'Resource 1', ha='center', va='bottom', fontsize=12)
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        
        plt.colorbar(scatter, ax=ax, label='Agent ID')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_ternary_trajectory(
    agent_results: Dict[int, Dict[str, List]],
    agent_id: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trajectory of single agent on ternary diagram.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        agent_id: ID of agent to plot
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    if agent_id not in agent_results:
        raise ValueError(f"Agent {agent_id} not found in results")
    
    prob_history = agent_results[agent_id]['prob']
    if not prob_history or len(prob_history[0]) != 3:
        raise ValueError("Ternary plots require exactly 3 resources")
    
    trajectory = np.array(prob_history)
    
    if HAS_MPLTERN:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'ternary'})
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', alpha=0.7, linewidth=2)
        
        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  c='red', s=100, marker='s', label='End', zorder=5)
        
        ax.set_tlabel('Resource 1')
        ax.set_llabel('Resource 2')
        ax.set_rlabel('Resource 3')
        ax.set_title(f'Agent {agent_id} Trajectory')
        ax.legend()
        
    else:
        # Manual projection
        fig, ax = plt.subplots(figsize=(10, 8))
        
        triangle = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3)/2]])
        projected = np.dot(trajectory, triangle)
        
        # Draw triangle
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', linewidth=2)
        
        # Plot trajectory
        ax.plot(projected[:, 0], projected[:, 1], 'b-', alpha=0.7, linewidth=2)
        
        # Mark start and end
        ax.scatter(projected[0, 0], projected[0, 1], c='green', s=100, 
                  marker='o', label='Start', zorder=5)
        ax.scatter(projected[-1, 0], projected[-1, 1], c='red', s=100,
                  marker='s', label='End', zorder=5)
        
        # Labels
        label_offset = 0.05
        ax.text(triangle[0, 0] - label_offset, triangle[0, 1] - label_offset, 
                'Resource 2', ha='center', va='top', fontsize=12)
        ax.text(triangle[1, 0] + label_offset, triangle[1, 1] - label_offset, 
                'Resource 3', ha='center', va='top', fontsize=12)
        ax.text(triangle[2, 0], triangle[2, 1] + label_offset, 
                'Resource 1', ha='center', va='bottom', fontsize=12)
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Agent {agent_id} Trajectory')
        ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_ternary_grid(num_points: int = 20) -> np.ndarray:
    """
    Create a grid of points for ternary analysis.
    
    Args:
        num_points: Number of points along each edge
        
    Returns:
        Array of ternary coordinates
    """
    points = []
    step = 1.0 / (num_points - 1)
    
    for i in range(num_points):
        for j in range(num_points - i):
            a = i * step
            b = j * step
            c = 1.0 - a - b
            if c >= 0:
                points.append([a, b, c])
    
    return np.array(points) 