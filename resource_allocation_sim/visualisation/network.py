"""Network visualisation for state transitions and agent interactions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import warnings

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def visualise_state_network(
    agent_data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    layout: str = 'spring'
) -> plt.Figure:
    """
    Visualise agent state transitions as a network.
    
    Args:
        agent_data: List of agent dictionaries with state transitions
        save_path: Optional path to save the plot
        layout: Network layout algorithm ('spring', 'circular', 'random')
        
    Returns:
        Matplotlib figure object
    """
    try:
        return _create_networkx_visualisation(agent_data, save_path, layout)
    except ImportError:
        warnings.warn("NetworkX not available. Using fallback visualisation.")
        return _create_fallback_visualisation(agent_data, save_path)


def _create_networkx_visualisation(
    agent_data: List[Dict[str, Any]], 
    save_path: Optional[str] = None,
    layout: str = 'spring'
) -> plt.Figure:
    """Create network visualisation using NetworkX."""
    import networkx as nx
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Process agent data to extract state transitions
    edge_weights = {}
    
    for agent in agent_data:
        agent_id = agent.get('id', 0)
        history = agent.get('action_history', [])
        
        # Create transitions between consecutive actions
        for i in range(len(history) - 1):
            from_state = f"R{history[i]}"
            to_state = f"R{history[i + 1]}"
            
            edge = (from_state, to_state)
            edge_weights[edge] = edge_weights.get(edge, 0) + 1
    
    # Add edges to graph
    for (from_state, to_state), weight in edge_weights.items():
        G.add_edge(from_state, to_state, weight=weight)
    
    # Create visualisation
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # Draw network
    edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights_list) if edge_weights_list else 1
    
    # Normalise edge weights for visualisation
    edge_widths = [3 * (weight / max_weight) for weight in edge_weights_list]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          edge_color='gray', arrows=True, 
                          arrowsize=20, ax=ax)
    
    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    
    ax.set_title('State Transition Network')
    ax.axis('off')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _create_fallback_visualisation(
    agent_data: List[Dict[str, Any]], 
    save_path: Optional[str] = None
) -> plt.Figure:
    """Fallback visualisation without NetworkX."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract action sequences
    all_actions = []
    action_counts = {}
    
    for agent in agent_data:
        history = agent.get('action_history', [])
        all_actions.extend(history)
        
        for action in history:
            action_counts[action] = action_counts.get(action, 0) + 1
    
    # Plot action frequency
    if action_counts:
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        ax1.bar([f'R{a}' for a in actions], counts)
        ax1.set_title('Resource Selection Frequency')
        ax1.set_xlabel('Resource')
        ax1.set_ylabel('Selection Count')
    
    # Plot action sequence for first few agents
    for i, agent in enumerate(agent_data[:5]):
        history = agent.get('action_history', [])
        if history:
            ax2.plot(range(len(history)), history, 
                    marker='o', label=f'Agent {i}', alpha=0.7)
    
    ax2.set_title('Action Sequences (First 5 Agents)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Resource Selection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_state_transitions(
    agent_results: Dict[int, Dict[str, List]],
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze state transition patterns.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        threshold: Minimum probability threshold for state definition
        
    Returns:
        Dictionary containing transition analysis
    """
    analysis = {
        'states_per_agent': {},
        'convergence_states': {},
        'transition_statistics': {},
        'state_persistence': {}
    }
    
    all_states = set()
    all_transitions = []
    
    for agent_id, data in agent_results.items():
        prob_history = data['prob']
        states = _discretize_states(prob_history, threshold)
        transitions = _extract_transitions(states)
        
        analysis['states_per_agent'][agent_id] = len(set(states))
        analysis['convergence_states'][agent_id] = states[-1] if states else 'UNKNOWN'
        
        all_states.update(states)
        all_transitions.extend(transitions)
        
        # Analyze state persistence
        if states:
            persistence = _calculate_state_persistence(states)
            analysis['state_persistence'][agent_id] = persistence
    
    # Global statistics
    transition_counts = Counter(all_transitions)
    
    analysis['transition_statistics'] = {
        'total_unique_states': len(all_states),
        'total_transitions': len(all_transitions),
        'most_common_transitions': transition_counts.most_common(5),
        'avg_states_per_agent': np.mean(list(analysis['states_per_agent'].values())) if analysis['states_per_agent'] else 0
    }
    
    return analysis


def _discretize_states(prob_history: List[List[float]], threshold: float) -> List[str]:
    """Convert probability history to discrete states."""
    states = []
    for probs in prob_history:
        max_prob = max(probs)
        if max_prob > threshold:
            dominant = [i for i, p in enumerate(probs) if p == max_prob]
            if len(dominant) == 1:
                state = f"R{dominant[0]}"
            else:
                state = f"R{'-'.join(map(str, sorted(dominant)))}"
        else:
            state = "UNIFORM"
        states.append(state)
    return states


def _extract_transitions(states: List[str]) -> List[Tuple[str, str]]:
    """Extract state transitions from state sequence."""
    transitions = []
    for i in range(len(states) - 1):
        transitions.append((states[i], states[i + 1]))
    return transitions


def _calculate_state_persistence(states: List[str]) -> Dict[str, float]:
    """Calculate how long agents stay in each state."""
    persistence = {}
    current_state = None
    current_duration = 0
    state_durations = defaultdict(list)
    
    for state in states:
        if state == current_state:
            current_duration += 1
        else:
            if current_state is not None:
                state_durations[current_state].append(current_duration)
            current_state = state
            current_duration = 1
    
    # Add final state duration
    if current_state is not None:
        state_durations[current_state].append(current_duration)
    
    # Calculate average persistence for each state
    for state, durations in state_durations.items():
        persistence[state] = np.mean(durations)
    
    return persistence


def plot_transition_graph(
    agent_results: Dict[int, Dict[str, List]],
    agent_id: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot transition graph for a specific agent.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        agent_id: ID of agent to analyze
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    if agent_id not in agent_results:
        raise ValueError(f"Agent {agent_id} not found in results")
    
    data = agent_results[agent_id]
    action_history = data['action']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Action sequence over time
    ax1.plot(action_history, 'o-', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Selected Resource')
    ax1.set_title(f'Agent {agent_id} Action Sequence')
    ax1.grid(True, alpha=0.3)
    
    # Action frequency
    action_counts = Counter(action_history)
    resources = list(action_counts.keys())
    frequencies = list(action_counts.values())
    
    ax2.bar([f'Resource {r}' for r in resources], frequencies)
    ax2.set_xlabel('Resource')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Agent {agent_id} Action Frequency')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 