"""Analysis tools for individual agent behavior."""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Any, Tuple, Optional
from .metrics import calculate_entropy


def analyse_agent_convergence(agent_results: Dict[int, Dict[str, List]]) -> Dict[str, Any]:
    """
    Analyse convergence behavior of individual agents.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        
    Returns:
        Dictionary containing convergence analysis
    """
    analysis = {
        'convergence_times': {},
        'final_distributions': {},
        'entropy_evolution': {},
        'action_frequencies': {}
    }
    
    for agent_id, data in agent_results.items():
        prob_history = np.array(data['prob'])
        action_history = data['action']
        
        # Calculate entropy evolution
        entropies = [calculate_entropy(probs) for probs in prob_history]
        analysis['entropy_evolution'][agent_id] = entropies
        
        # Find convergence time (entropy below threshold for consecutive steps)
        convergence_time = _find_convergence_time(entropies, threshold=0.1, window=5)
        analysis['convergence_times'][agent_id] = convergence_time
        
        # Final probability distribution
        analysis['final_distributions'][agent_id] = prob_history[-1].tolist()
        
        # Action frequency analysis
        action_counts = {}
        for action in action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        analysis['action_frequencies'][agent_id] = action_counts
    
    return analysis


def _find_convergence_time(
    entropies: List[float], 
    threshold: float = 0.1, 
    window: int = 5
) -> int:
    """Find when agent converged based on entropy."""
    for i in range(len(entropies) - window + 1):
        if all(e < threshold for e in entropies[i:i + window]):
            return i
    return len(entropies)


def plot_probability_distribution(
    agent_results: Dict[int, Dict[str, List]],
    save_path: Optional[str] = None,
    show_legend: bool = True
) -> plt.Figure:
    """
    Plot probability distributions for all agents over time.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        save_path: Optional path to save plot
        show_legend: Whether to show legend
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(
        len(agent_results), 1, 
        figsize=(12, 3 * len(agent_results)),
        squeeze=False
    )
    
    for idx, (agent_id, data) in enumerate(agent_results.items()):
        ax = axes[idx, 0]
        prob_history = np.array(data['prob'])
        
        # Plot each resource probability over time
        for resource_idx in range(prob_history.shape[1]):
            ax.plot(
                prob_history[:, resource_idx], 
                label=f'Resource {resource_idx + 1}',
                linewidth=2
            )
        
        ax.set_title(f'Agent {agent_id} - Probability Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
        
        if show_legend and prob_history.shape[1] <= 10:  # Only show legend if not too many resources
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_visited_probabilities(
    agent_results: Dict[int, Dict[str, List]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot visited probabilities on a ternary plot (for 3-resource case).
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    # Check if we have exactly 3 resources
    first_agent_probs = list(agent_results.values())[0]['prob'][0]
    if len(first_agent_probs) != 3:
        raise ValueError("Ternary plot only supports 3 resources")
    
    # Triangle vertices for ternary plot
    triangle = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    triangle_path = np.array([triangle[0], triangle[1], triangle[2], triangle[0]])
    
    # Create subplots
    num_agents = len(agent_results)
    num_cols = min(3, num_agents)
    num_rows = math.ceil(num_agents / num_cols)
    
    fig, axes = plt.subplots(
        num_rows, num_cols, 
        figsize=(5 * num_cols, 5 * num_rows)
    )
    
    if num_agents == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (agent_id, data) in enumerate(agent_results.items()):
        ax = axes[idx]
        
        # Plot triangle
        ax.plot(triangle_path[:, 0], triangle_path[:, 1], 'k-', linewidth=2)
        
        # Annotate vertices
        labels = ['Resource 1', 'Resource 2', 'Resource 3']
        offsets = [[-10, -15], [10, -15], [0, 10]]
        for i, (label, offset) in enumerate(zip(labels, offsets)):
            ax.annotate(
                label, triangle[i], 
                textcoords="offset points", 
                xytext=offset,
                ha="center", fontsize=10
            )
        
        # Project probabilities onto triangle
        prob_history = np.array(data['prob'])
        projected_probs = np.dot(prob_history, triangle)
        
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(projected_probs)))
        
        # Plot trajectory
        ax.scatter(
            projected_probs[:, 0], projected_probs[:, 1],
            c=colors, s=50, alpha=0.7, edgecolors='white', linewidths=0.5
        )
        
        # Connect points with line
        ax.plot(
            projected_probs[:, 0], projected_probs[:, 1],
            '-', color='gray', alpha=0.3, linewidth=1
        )
        
        # Mark start and end
        ax.scatter(
            projected_probs[0, 0], projected_probs[0, 1],
            c='red', s=100, marker='s', label='Start'
        )
        ax.scatter(
            projected_probs[-1, 0], projected_probs[-1, 1],
            c='blue', s=100, marker='*', label='End'
        )
        
        ax.set_title(f'Agent {agent_id} - Probability Trajectory')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend()
    
    # Hide unused subplots
    for idx in range(num_agents, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_agent_entropy_evolution(
    agent_results: Dict[int, Dict[str, List]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot entropy evolution for each agent.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for agent_id, data in agent_results.items():
        prob_history = data['prob']
        entropies = [calculate_entropy(probs) for probs in prob_history]
        
        ax.plot(entropies, label=f'Agent {agent_id}', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('Agent Entropy Evolution Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_agent_diversity(agent_results: Dict[int, Dict[str, List]]) -> Dict[str, float]:
    """
    Calculate diversity metrics across agents.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        
    Returns:
        Dictionary of diversity metrics
    """
    final_distributions = []
    for data in agent_results.values():
        final_distributions.append(data['prob'][-1])
    
    final_distributions = np.array(final_distributions)
    
    # Calculate inter-agent distances
    distances = []
    for i in range(len(final_distributions)):
        for j in range(i + 1, len(final_distributions)):
            dist = np.linalg.norm(final_distributions[i] - final_distributions[j])
            distances.append(dist)
    
    diversity_metrics = {
        'mean_pairwise_distance': np.mean(distances) if distances else 0.0,
        'std_pairwise_distance': np.std(distances) if distances else 0.0,
        'max_pairwise_distance': np.max(distances) if distances else 0.0,
        'min_pairwise_distance': np.min(distances) if distances else 0.0
    }
    
    return diversity_metrics 