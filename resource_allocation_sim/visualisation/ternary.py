"""Ternary plot utilities for 3-resource visualisations."""

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
        
        # Normalise distributions
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


def plot_initial_conditions_barycentric(
    initial_condition_data: Union[List[str], Dict[str, List[float]]],
    save_path: Optional[str] = None,
    title: str = "Initial Probability Distributions in Barycentric Coordinates"
) -> plt.Figure:
    """
    Plot different initial condition types on barycentric coordinates.
    
    Args:
        initial_condition_data: Either list of condition type names or dict mapping names to coordinates
        save_path: Optional path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw triangle for barycentric coordinates
    triangle = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3)/2]])
    triangle_closed = np.vstack([triangle, triangle[0]])
    ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', linewidth=2)
    
    # Define complete initial condition mappings including all diagonal points
    conditions = {
        'uniform': [1/3, 1/3, 1/3],
        'diagonal_point_1': [0.4, 0.289, 0.311],
        'diagonal_point_2': [0.444, 0.256, 0.3],
        'diagonal_point_3': [0.489, 0.222, 0.289],
        'diagonal_point_4': [0.533, 0.189, 0.278],
        'diagonal_point_5': [0.578, 0.156, 0.266],
        'diagonal_point_6': [0.622, 0.122, 0.256],
        'diagonal_point_7': [0.667, 0.089, 0.244],
        'diagonal_point_8': [0.711, 0.056, 0.233],
        'diagonal_point_9': [0.756, 0.022, 0.222],
        'diagonal_point_10': [0.8, 0.0, 0.2],
        'edge_bias_12': [0.45, 0.45, 0.10],
        'edge_bias_13': [0.45, 0.10, 0.45], 
        'edge_bias_23': [0.10, 0.45, 0.45],
        'vertex_bias_1': [0.70, 0.15, 0.15],
        'vertex_bias_2': [0.15, 0.70, 0.15],
        'vertex_bias_3': [0.15, 0.15, 0.70],
        'diagonal_sample_fixed': [0.7, 0.2, 0.1],
        'diagonal_sample_varied': [0.5, 0.3, 0.2],
        'uniform_varied': [0.35, 0.33, 0.32]
    }
    
    # Handle both formats: list of names or dict of coordinates
    if isinstance(initial_condition_data, dict):
        # Direct coordinate data provided
        plot_data = initial_condition_data
    else:
        # List of condition names - use predefined mappings
        plot_data = {}
        for condition in initial_condition_data:
            if condition in conditions:
                plot_data[condition] = conditions[condition]
    
    # Define colors and markers for different condition types
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'gray', 'olive', 'navy', 'maroon']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
    
    plotted_conditions = []
    for i, (condition, probs) in enumerate(plot_data.items()):
        if len(probs) == 3 and abs(sum(probs) - 1.0) < 1e-6:  # Valid probability distribution
            # Convert to barycentric coordinates
            coord = np.dot(probs, triangle)
            
            # Choose marker style based on condition type
            if 'uniform' in condition:
                marker = 'o'
                size = 150
                alpha = 1.0
            elif 'diagonal_point' in condition:
                marker = '^'
                size = 100
                alpha = 0.8
            elif 'edge_bias' in condition:
                marker = 's'
                size = 120
                alpha = 0.9
            elif 'vertex_bias' in condition:
                marker = 'D'
                size = 120
                alpha = 0.9
            else:
                marker = 'o'
                size = 100
                alpha = 0.8
            
            # Create readable label
            if 'diagonal_point' in condition:
                point_num = condition.split('_')[-1]
                label = f"Diagonal {point_num}"
            elif 'edge_bias' in condition:
                bias_num = condition.split('_')[-1]
                label = f"Edge Bias {bias_num}"
            else:
                label = condition.replace('_', ' ').title()
            
            ax.scatter(coord[0], coord[1], c=colors[i % len(colors)], s=size, 
                      marker=marker, label=label, alpha=alpha, edgecolors='black', linewidth=0.5)
            plotted_conditions.append(condition)
    
    # Labels for vertices
    label_offset = 0.08
    ax.text(triangle[0, 0] - label_offset, triangle[0, 1] - label_offset, 
            'Resource 2\n[0,1,0]', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(triangle[1, 0] + label_offset, triangle[1, 1] - label_offset, 
            'Resource 3\n[0,0,1]', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(triangle[2, 0], triangle[2, 1] + label_offset, 
            'Resource 1\n[1,0,0]', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add grid lines for better readability
    for val in [0.2, 0.4, 0.6, 0.8]:
        # Lines parallel to each edge
        # Resource 1 = val lines
        p1 = np.dot([val, 1-val, 0], triangle)
        p2 = np.dot([val, 0, 1-val], triangle)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.2, linewidth=0.5)
        
        # Resource 2 = val lines  
        p1 = np.dot([1-val, val, 0], triangle)
        p2 = np.dot([0, val, 1-val], triangle)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.2, linewidth=0.5)
        
        # Resource 3 = val lines
        p1 = np.dot([1-val, 0, val], triangle)
        p2 = np.dot([0, 1-val, val], triangle)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.2, linewidth=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    
    if plotted_conditions:
        # Create legend with multiple columns to save space
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=10)
        legend.set_title("Initial Conditions", prop={'size': 12, 'weight': 'bold'})
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 