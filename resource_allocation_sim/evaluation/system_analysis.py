"""System-wide analysis tools for simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from .metrics import calculate_entropy, calculate_gini_coefficient, calculate_system_metrics


def analyze_system_performance(
    results: Dict[str, Any],
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive analysis of system performance.
    
    Args:
        results: Simulation results dictionary
        detailed: Whether to include detailed analysis
        
    Returns:
        Dictionary containing system performance metrics
    """
    analysis = {}
    
    # Basic metrics
    final_consumption = np.array(results.get('final_consumption', []))
    total_cost = results.get('total_cost', 0.0)
    config = results.get('config', {})
    
    analysis['basic_metrics'] = {
        'total_cost': total_cost,
        'entropy': calculate_entropy(final_consumption),
        'gini_coefficient': calculate_gini_coefficient(final_consumption),
        'mean_consumption': np.mean(final_consumption),
        'std_consumption': np.std(final_consumption),
        'max_consumption': np.max(final_consumption) if len(final_consumption) > 0 else 0,
        'min_consumption': np.min(final_consumption) if len(final_consumption) > 0 else 0
    }
    
    if detailed and 'agent_results' in results:
        agent_results = results['agent_results']
        
        # Agent convergence analysis
        agent_entropies = {}
        convergence_times = {}
        
        for agent_id, data in agent_results.items():
            prob_history = data['prob']
            entropies = [calculate_entropy(probs) for probs in prob_history]
            agent_entropies[agent_id] = entropies
            
            # Find convergence time
            convergence_time = len(entropies)
            for i, entropy in enumerate(entropies):
                if entropy < 0.1:
                    convergence_time = i
                    break
            convergence_times[agent_id] = convergence_time
        
        analysis['agent_analysis'] = {
            'convergence_times': convergence_times,
            'mean_convergence_time': np.mean(list(convergence_times.values())),
            'std_convergence_time': np.std(list(convergence_times.values()))
        }
        
        # Resource utilization analysis
        if len(final_consumption) > 0:
            total_agents = config.get('num_agents', sum(final_consumption))
            utilization_rates = final_consumption / total_agents if total_agents > 0 else final_consumption
            
            analysis['utilization_analysis'] = {
                'utilization_rates': utilization_rates.tolist(),
                'utilization_variance': np.var(utilization_rates),
                'max_utilization_diff': np.max(utilization_rates) - np.min(utilization_rates)
            }
    
    return analysis


def plot_cost_evolution(
    multi_episode_results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cost evolution over episodes.
    
    Args:
        multi_episode_results: List of results from multiple episodes
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract costs
    costs = [result.get('total_cost', 0.0) for result in multi_episode_results]
    episodes = range(len(costs))
    
    # Plot cost evolution
    ax1.plot(episodes, costs, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Cost')
    ax1.set_title('Cost Evolution Over Episodes')
    ax1.grid(True, alpha=0.3)
    
    # Plot cost distribution
    ax2.hist(costs, bins=min(20, len(costs)//2), alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(costs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(costs):.2f}')
    ax2.set_xlabel('Total Cost')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Cost Distribution')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_entropy_evolution(
    multi_episode_results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot entropy evolution over episodes.
    
    Args:
        multi_episode_results: List of results from multiple episodes
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract entropies
    entropies = []
    for result in multi_episode_results:
        final_consumption = np.array(result.get('final_consumption', []))
        entropy = calculate_entropy(final_consumption)
        entropies.append(entropy)
    
    episodes = range(len(entropies))
    
    # Plot entropy evolution
    ax1.plot(episodes, entropies, 'o-', linewidth=2, markersize=6, color='green')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Entropy Evolution Over Episodes')
    ax1.grid(True, alpha=0.3)
    
    # Plot entropy distribution
    ax2.hist(entropies, bins=min(20, len(entropies)//2), alpha=0.7, 
             edgecolor='black', color='green')
    ax2.axvline(np.mean(entropies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(entropies):.3f}')
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Entropy Distribution')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_consumption_heatmap(
    multi_episode_results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot consumption patterns as heatmap.
    
    Args:
        multi_episode_results: List of results from multiple episodes
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    # Collect consumption data
    consumption_data = []
    for i, result in enumerate(multi_episode_results):
        consumption = result.get('final_consumption', [])
        if consumption:
            consumption_data.append([i] + consumption)
    
    if not consumption_data:
        # Create empty plot if no data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No consumption data available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Convert to DataFrame
    num_resources = len(consumption_data[0]) - 1
    columns = ['Episode'] + [f'Resource {i+1}' for i in range(num_resources)]
    df = pd.DataFrame(consumption_data, columns=columns)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    consumption_matrix = df.iloc[:, 1:].T  # Transpose for resources as rows
    
    sns.heatmap(
        consumption_matrix, 
        annot=True, 
        fmt='.1f',
        cmap='viridis',
        xticklabels=df['Episode'],
        yticklabels=[f'Resource {i+1}' for i in range(num_resources)],
        ax=ax
    )
    
    ax.set_title('Resource Consumption Across Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Resource')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_configurations(
    results_list: List[Tuple[str, List[Dict[str, Any]]]],
    metric: str = 'entropy',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare different configurations across multiple runs.
    
    Args:
        results_list: List of (config_name, results) tuples
        metric: Metric to compare ('entropy', 'cost', 'gini')
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    comparison_data = []
    
    for config_name, results in results_list:
        values = []
        for result in results:
            if metric == 'entropy':
                final_consumption = np.array(result.get('final_consumption', []))
                value = calculate_entropy(final_consumption)
            elif metric == 'cost':
                value = result.get('total_cost', 0.0)
            elif metric == 'gini':
                final_consumption = np.array(result.get('final_consumption', []))
                value = calculate_gini_coefficient(final_consumption)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            values.append(value)
        
        comparison_data.extend([(config_name, v) for v in values])
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data, columns=['Configuration', metric.title()])
    
    # Create box plot
    sns.boxplot(data=df, x='Configuration', y=metric.title(), ax=ax)
    ax.set_title(f'{metric.title()} Comparison Across Configurations')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_summary_report(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Generate a text summary report of simulation results.
    
    Args:
        results: Simulation results dictionary
        save_path: Optional path to save report
        
    Returns:
        Summary report as string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("SIMULATION SUMMARY REPORT")
    report_lines.append("=" * 60)
    
    # Configuration
    config = results.get('config', {})
    report_lines.append("\nCONFIGURATION:")
    report_lines.append(f"  Agents: {config.get('num_agents', 'N/A')}")
    report_lines.append(f"  Resources: {config.get('num_resources', 'N/A')}")
    report_lines.append(f"  Iterations: {config.get('num_iterations', 'N/A')}")
    report_lines.append(f"  Weight: {config.get('weight', 'N/A')}")
    report_lines.append(f"  Capacity: {config.get('capacity', 'N/A')}")
    
    # System metrics
    analysis = analyze_system_performance(results)
    metrics = analysis['basic_metrics']
    
    report_lines.append("\nSYSTEM METRICS:")
    report_lines.append(f"  Total Cost: {metrics['total_cost']:.4f}")
    report_lines.append(f"  Entropy: {metrics['entropy']:.4f}")
    report_lines.append(f"  Gini Coefficient: {metrics['gini_coefficient']:.4f}")
    report_lines.append(f"  Mean Consumption: {metrics['mean_consumption']:.2f}")
    report_lines.append(f"  Std Consumption: {metrics['std_consumption']:.2f}")
    
    # Agent analysis (if available)
    if 'agent_analysis' in analysis:
        agent_analysis = analysis['agent_analysis']
        report_lines.append("\nAGENT ANALYSIS:")
        report_lines.append(f"  Mean Convergence Time: {agent_analysis['mean_convergence_time']:.1f}")
        report_lines.append(f"  Std Convergence Time: {agent_analysis['std_convergence_time']:.1f}")
    
    # Resource utilization (if available)
    if 'utilization_analysis' in analysis:
        util_analysis = analysis['utilization_analysis']
        report_lines.append("\nRESOURCE UTILIZATION:")
        report_lines.append(f"  Utilization Variance: {util_analysis['utilization_variance']:.4f}")
        report_lines.append(f"  Max Utilization Difference: {util_analysis['max_utilization_diff']:.4f}")
    
    report_lines.append("\n" + "=" * 60)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def generate_analysis_report(results: Dict[str, Any]) -> str:
    """Generate analysis report - this function was referenced but missing."""
    return generate_summary_report(results)  # Use existing function 