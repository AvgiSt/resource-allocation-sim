"""System-wide analysis tools for simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from .metrics import calculate_entropy, calculate_gini_coefficient, calculate_system_metrics


def analyse_system_performance(
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
    analysis = analyse_system_performance(results)
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


def perform_convergence_statistical_tests(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform statistical tests for system-level convergence patterns.
    
    This function tests various hypotheses about system convergence behaviour,
    including temporal patterns, degeneracy achievement, and sequential ordering.
    
    Args:
        analysis: Analysis dictionary containing convergence data with keys:
            - 'convergence_times': dict with 'all_times' list
            - 'sequential_indices': dict with 'all_indices' list  
            - 'degeneracy_scores': dict with 'all_scores' list
    
    Returns:
        Dictionary containing statistical test results
    """
    import scipy.stats as stats
    
    tests = {}
    
    convergence_times = analysis.get('convergence_times', {}).get('all_times', [])
    sequential_indices = analysis.get('sequential_indices', {}).get('all_indices', [])
    degeneracy_scores = analysis.get('degeneracy_scores', {}).get('all_scores', [])
    
    # Test 1: Kolmogorov-Smirnov test for uniform distribution of convergence times
    if len(convergence_times) > 1:
        try:
            ks_stat, ks_pvalue = stats.kstest(
                np.array(convergence_times), 
                lambda x: stats.uniform.cdf(
                    x, 
                    loc=min(convergence_times), 
                    scale=max(convergence_times) - min(convergence_times)
                )
            )
            tests['convergence_uniformity'] = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'interpretation': 'H0: convergence times are uniformly distributed',
                'significant': ks_pvalue < 0.05
            }
        except Exception as e:
            tests['convergence_uniformity'] = {
                'test': 'Kolmogorov-Smirnov',
                'error': str(e),
                'statistic': None,
                'p_value': None
            }
    
    # Test 2: One-sample t-test for sequential index > 0.5 (more sequential than random)
    if len(sequential_indices) > 1:
        try:
            t_stat, t_pvalue = stats.ttest_1samp(sequential_indices, 0.5)
            tests['sequential_pattern'] = {
                'test': 'One-sample t-test',
                'statistic': t_stat,
                'p_value': t_pvalue,
                'interpretation': 'H0: sequential index = 0.5 (random), H1: sequential index > 0.5',
                'significant': t_pvalue < 0.05,
                'more_sequential': t_stat > 0
            }
        except Exception as e:
            tests['sequential_pattern'] = {
                'test': 'One-sample t-test',
                'error': str(e),
                'statistic': None,
                'p_value': None
            }
    
    return tests


def test_sequential_convergence_hypothesis(
    analysis: Dict[str, Any], 
    convergence_threshold_max_prob: float = 0.9,
    expected_degeneracy_proportion: float = 0.9
) -> Dict[str, Any]:
    """
    Test specific hypothesis about sequential convergence patterns.
    
    H2: Agents with uniform initial distribution sequentially converge 
    to degenerate distributions.
    
    Args:
        analysis: Analysis dictionary containing convergence data
        convergence_threshold_max_prob: Threshold for considering agent converged
        expected_degeneracy_proportion: Expected proportion achieving degeneracy
        
    Returns:
        Dictionary containing hypothesis test results
    """
    import scipy.stats as stats
    
    tests = {}
    degeneracy_scores = analysis.get('degeneracy_scores', {}).get('all_scores', [])
    
    # Test 3: Binomial test for proportion achieving degeneracy
    if len(degeneracy_scores) > 0:
        n_degenerate = sum(s > convergence_threshold_max_prob for s in degeneracy_scores)
        n_total = len(degeneracy_scores)
        
        try:
            # Use the newer binomtest function if available
            try:
                binom_result = stats.binomtest(n_degenerate, n_total, expected_degeneracy_proportion)
                binom_pvalue = binom_result.pvalue
            except AttributeError:
                # Fallback for older scipy versions
                binom_pvalue = stats.binom_test(n_degenerate, n_total, expected_degeneracy_proportion)
            
            tests['degeneracy_proportion'] = {
                'test': 'Binomial test',
                'observed': n_degenerate,
                'total': n_total,
                'proportion': n_degenerate / n_total if n_total > 0 else 0,
                'expected_proportion': expected_degeneracy_proportion,
                'p_value': binom_pvalue,
                'interpretation': f'H0: proportion degenerate = {expected_degeneracy_proportion}, H1: proportion ≠ {expected_degeneracy_proportion}',
                'significant': binom_pvalue < 0.05
            }
        except Exception as e:
            tests['degeneracy_proportion'] = {
                'test': 'Binomial test',
                'error': str(e),
                'observed': n_degenerate,
                'total': n_total,
                'proportion': n_degenerate / n_total if n_total > 0 else 0
            }
    
    return tests


def calculate_system_convergence_metrics(replication_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate system-level convergence pattern metrics.
    
    This function computes metrics that characterise how the system as a whole
    converges, including temporal patterns and coordination measures.
    
    Args:
        replication_data: List of dictionaries containing convergence data per replication
        
    Returns:
        Dictionary containing system convergence metrics
    """
    if not replication_data:
        return {}
    
    metrics = {
        'temporal_patterns': {},
        'coordination_measures': {},
        'system_efficiency': {}
    }
    
    # Extract all sequential indices
    sequential_indices = [rep.get('sequential_index', 0) for rep in replication_data]
    
    # Extract all convergence times across replications
    all_convergence_times = []
    for rep in replication_data:
        conv_times = rep.get('convergence_times', [])
        all_convergence_times.extend(conv_times)
    
    # Temporal pattern metrics
    if sequential_indices:
        metrics['temporal_patterns'] = {
            'mean_sequential_index': np.mean(sequential_indices),
            'std_sequential_index': np.std(sequential_indices),
            'sequential_consistency': 1.0 - np.std(sequential_indices),  # Higher = more consistent
            'proportion_sequential': np.mean([si > 0.5 for si in sequential_indices])
        }
    
    # Coordination measures
    if all_convergence_times:
        # Calculate coordination efficiency
        total_agents_converged = len(all_convergence_times)
        mean_convergence_time = np.mean(all_convergence_times)
        convergence_spread = np.max(all_convergence_times) - np.min(all_convergence_times) if len(all_convergence_times) > 1 else 0
        
        metrics['coordination_measures'] = {
            'mean_convergence_time': mean_convergence_time,
            'convergence_spread': convergence_spread,
            'convergence_efficiency': total_agents_converged / (mean_convergence_time + 1),  # Agents per time unit
            'temporal_coordination': 1.0 / (1.0 + convergence_spread / mean_convergence_time) if mean_convergence_time > 0 else 0
        }
    
    # System efficiency metrics
    convergence_success_rates = []
    for rep in replication_data:
        total_agents = rep.get('total_agents', 10)  # Default assumption
        converged_agents = rep.get('num_converged', len(rep.get('convergence_times', [])))
        success_rate = converged_agents / total_agents if total_agents > 0 else 0
        convergence_success_rates.append(success_rate)
    
    if convergence_success_rates:
        metrics['system_efficiency'] = {
            'mean_success_rate': np.mean(convergence_success_rates),
            'std_success_rate': np.std(convergence_success_rates),
            'min_success_rate': np.min(convergence_success_rates),
            'max_success_rate': np.max(convergence_success_rates)
        }
    
    return metrics


def evaluate_hypothesis_support(
    convergence_analysis: Dict[str, Any], 
    statistical_tests: Dict[str, Any],
    convergence_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate level of support for sequential convergence hypothesis.
    
    Combines statistical test results with convergence metrics to provide
    an overall assessment of hypothesis support.
    
    Args:
        convergence_analysis: Results from convergence analysis
        statistical_tests: Results from statistical tests
        convergence_metrics: System-level convergence metrics
        
    Returns:
        Dictionary containing hypothesis support evaluation
    """
    support = {
        'overall_support': 'undetermined',
        'criteria_met': {},
        'evidence_strength': {},
        'recommendation': ''
    }
    
    # Criterion 1: Sequential pattern (sequential index > 0.5)
    temporal_patterns = convergence_metrics.get('temporal_patterns', {})
    mean_seq_index = temporal_patterns.get('mean_sequential_index', 0)
    if mean_seq_index > 0:
        support['criteria_met']['sequential_pattern'] = mean_seq_index > 0.5
        support['evidence_strength']['sequential_index'] = mean_seq_index
    
    # Criterion 2: High degeneracy proportion
    deg_test = statistical_tests.get('degeneracy_proportion', {})
    deg_proportion = deg_test.get('proportion', 0)
    if deg_proportion > 0:
        support['criteria_met']['high_degeneracy'] = deg_proportion > 0.8
        support['evidence_strength']['degeneracy_proportion'] = deg_proportion
    
    # Criterion 3: Statistical significance
    seq_test = statistical_tests.get('sequential_pattern', {})
    if 'significant' in seq_test:
        support['criteria_met']['statistical_significance'] = seq_test['significant']
        support['evidence_strength']['sequential_p_value'] = seq_test.get('p_value', 1.0)
    
    # Overall assessment
    criteria_met = list(support['criteria_met'].values())
    if len(criteria_met) > 0:
        prop_criteria_met = sum(criteria_met) / len(criteria_met)
        
        if prop_criteria_met >= 0.8:
            support['overall_support'] = 'strong'
            support['recommendation'] = 'Strong evidence for sequential convergence hypothesis'
        elif prop_criteria_met >= 0.6:
            support['overall_support'] = 'moderate'
            support['recommendation'] = 'Moderate evidence for sequential convergence hypothesis'
        elif prop_criteria_met >= 0.4:
            support['overall_support'] = 'weak'
            support['recommendation'] = 'Weak evidence for sequential convergence hypothesis'
        else:
            support['overall_support'] = 'none'
            support['recommendation'] = 'Little evidence for sequential convergence hypothesis'
    
    return support 