"""Network visualization for state transitions and agent interactions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def visualize_state_network(
    agent_results: Dict[int, Dict[str, List]],
    threshold: float = 0.1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize agent state transitions as a network.
    
    Args:
        agent_results: Dictionary of agent probability and action histories
        threshold: Minimum probability threshold for state definition
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    if not HAS_NETWORKX:
        # Fallback visualization without networkx
        return _create_simple_state_plot(agent_results, save_path)
    
    # Extract state transitions
    all_transitions = []
    state_counts = defaultdict(int)
    
    for agent_id, data in agent_results.items():
        prob_history = data['prob']
        states = _discretize_states(prob_history, threshold)
        transitions = _extract_transitions(states)
        all_transitions.extend(transitions)
        
        for state in states:
            state_counts[state] += 1
    
    # Build network
    G = nx.DiGraph()
    transition_counts = defaultdict(int)
    
    for from_state, to_state in all_transitions:
        transition_counts[(from_state, to_state)] += 1
        G.add_edge(from_state, to_state, weight=transition_counts[(from_state, to_state)])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    if len(G.nodes()) > 0:
        # Network layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node sizes based on frequency
        node_sizes = [state_counts[node] * 10 + 200 for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color='lightblue', alpha=0.7, ax=ax1)
        
        if len(G.edges()) > 0:
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [w/max_weight * 5 for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths,
                                  alpha=0.6, edge_color='gray', ax=ax1)
        
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    
    ax1.set_title('State Transition Network')
    ax1.axis('off')
    
    # Transition frequency histogram
    if transition_counts:
        frequencies = list(transition_counts.values())
        ax2.hist(frequencies, bins=min(20, len(frequencies)), alpha=0.7, edgecolor='black')
    else:
        ax2.text(0.5, 0.5, 'No transitions found', ha='center', va='center')
    
    ax2.set_xlabel('Transition Frequency')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Transition Frequencies')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _create_simple_state_plot(
    agent_results: Dict[int, Dict[str, List]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Fallback visualization without networkx."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count final states
    final_states = []
    for agent_id, data in agent_results.items():
        if data['action']:
            final_states.append(data['action'][-1])
    
    if final_states:
        state_counts = Counter(final_states)
        states = list(state_counts.keys())
        counts = list(state_counts.values())
        
        ax1.bar([f'Resource {s}' for s in states], counts)
        ax1.set_title('Final State Distribution')
        ax1.set_xlabel('Resource')
        ax1.set_ylabel('Number of Agents')
    else:
        ax1.text(0.5, 0.5, 'No state data available', ha='center', va='center')
        ax1.set_title('State Analysis')
    
    # Action frequency over time
    all_actions = []
    for agent_id, data in agent_results.items():
        all_actions.extend(data['action'])
    
    if all_actions:
        action_counts = Counter(all_actions)
        resources = list(action_counts.keys())
        frequencies = list(action_counts.values())
        
        ax2.pie(frequencies, labels=[f'Resource {r}' for r in resources], autopct='%1.1f%%')
        ax2.set_title('Overall Action Distribution')
    else:
        ax2.text(0.5, 0.5, 'No action data available', ha='center', va='center')
        ax2.set_title('Action Analysis')
    
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