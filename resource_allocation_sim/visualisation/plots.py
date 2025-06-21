"""General plotting utilities for simulation visualisation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union


def plot_resource_distribution(
    consumption: Union[List[float], np.ndarray],
    capacity: Union[List[float], np.ndarray],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot resource consumption vs capacity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    resources = range(len(consumption))
    x_pos = np.arange(len(resources))
    
    bars = ax.bar(x_pos, consumption, alpha=0.7, label='Consumption')
    ax.axhline(y=np.mean(capacity), color='red', linestyle='--', 
               label=f'Mean Capacity: {np.mean(capacity):.2f}')
    
    # Add capacity markers
    for i, cap in enumerate(capacity):
        ax.axhline(y=cap, xmin=(i-0.4)/len(resources), xmax=(i+0.4)/len(resources),
                   color='orange', linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Resource')
    ax.set_ylabel('Consumption')
    ax.set_title('Resource Consumption Distribution')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'R{i+1}' for i in resources])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence_comparison(
    results_dict: Dict[str, List[Dict[str, Any]]],
    metric: str = 'entropy',
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare convergence across different configurations.
    
    Args:
        results_dict: Dictionary mapping config names to result lists
        metric: Metric to plot ('entropy', 'cost', 'gini')
        ylabel: Custom y-axis label (defaults to metric.title())
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Determine y-axis label
    y_label = ylabel if ylabel is not None else metric.title()
    
    # Collect all data
    all_data = []
    
    for config_name, results_list in results_dict.items():
        values = []
        for result in results_list:
            if metric == 'entropy':
                from ..evaluation.metrics import calculate_entropy
                final_consumption = np.array(result.get('final_consumption', []))
                value = calculate_entropy(final_consumption)
            elif metric == 'cost':
                value = result.get('total_cost', 0.0)
            elif metric == 'gini':
                from ..evaluation.metrics import calculate_gini_coefficient
                final_consumption = np.array(result.get('final_consumption', []))
                value = calculate_gini_coefficient(final_consumption)
            
            values.append(value)
            all_data.append({'Config': config_name, y_label: value, 'Episode': len(values)})
        
        # Plot time series for this config
        axes[0].plot(values, label=config_name, alpha=0.7, linewidth=2)
    
    # Time series plot
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(y_label)
    axes[0].set_title(f'{y_label} Evolution by Configuration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    df = pd.DataFrame(all_data)
    sns.boxplot(data=df, x='Config', y=y_label, ax=axes[1])
    axes[1].set_title(f'{y_label} Distribution by Configuration')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Histogram
    for config_name, results_list in results_dict.items():
        values = [all_data[i][y_label] for i, d in enumerate(all_data) if d['Config'] == config_name]
        axes[2].hist(values, alpha=0.6, label=config_name, bins=15)
    
    axes[2].set_xlabel(y_label)
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'{y_label} Histogram')
    axes[2].legend()
    
    # Summary statistics
    summary_data = []
    for config_name, results_list in results_dict.items():
        values = [all_data[i][y_label] for i, d in enumerate(all_data) if d['Config'] == config_name]
        summary_data.append({
            'Config': config_name,
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values)
        })
    
    summary_df = pd.DataFrame(summary_data)
    axes[3].axis('tight')
    axes[3].axis('off')
    table = axes[3].table(cellText=summary_df.round(3).values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    axes[3].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_sensitivity(
    parameter_results: Dict[str, List[float]],
    parameter_name: str,
    metric_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot sensitivity of metric to parameter changes.
    
    Args:
        parameter_results: Dictionary mapping parameter values to metric values
        parameter_name: Name of the parameter being varied
        metric_name: Name of the metric being measured
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    param_values = []
    metric_means = []
    metric_stds = []
    
    for param_val, metric_vals in parameter_results.items():
        param_values.append(float(param_val))
        metric_means.append(np.mean(metric_vals))
        metric_stds.append(np.std(metric_vals))
    
    # Sort by parameter value
    sorted_indices = np.argsort(param_values)
    param_values = np.array(param_values)[sorted_indices]
    metric_means = np.array(metric_means)[sorted_indices]
    metric_stds = np.array(metric_stds)[sorted_indices]
    
    # Mean with error bars
    ax1.errorbar(param_values, metric_means, yerr=metric_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel(f'{metric_name} (Mean ± Std)')
    ax1.set_title(f'{metric_name} vs {parameter_name}')
    ax1.grid(True, alpha=0.3)
    
    # Box plot for each parameter value
    box_data = []
    box_labels = []
    for param_val, metric_vals in parameter_results.items():
        box_data.append(metric_vals)
        box_labels.append(str(param_val))
    
    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_xlabel(parameter_name)
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'{metric_name} Distribution by {parameter_name}')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cost_contour(
    capacity_combinations: List[Tuple[float, float, float]],
    costs: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cost contour for different capacity combinations (3D case).
    
    Args:
        capacity_combinations: List of (cap1, cap2, cap3) tuples
        costs: Corresponding costs
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    try:
        import plotly.figure_factory as ff
        import plotly.graph_objects as go
        
        # Convert to ternary coordinates
        capacities = np.array(capacity_combinations)
        total_cap = np.sum(capacities, axis=1)
        normalised_caps = capacities / total_cap[:, np.newaxis]
        
        # Create ternary contour plot
        fig = ff.create_ternary_contour(
            normalised_caps.T, np.array(costs),
            pole_labels=['Capacity A', 'Capacity B', 'Capacity C'],
            interp_mode='cartesian'
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            
        return fig
        
    except ImportError:
        # Fallback to matplotlib 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        capacities = np.array(capacity_combinations)
        x, y, z = capacities[:, 0], capacities[:, 1], capacities[:, 2]
        
        scatter = ax.scatter(x, y, z, c=costs, cmap='viridis', s=50)
        ax.set_xlabel('Capacity 1')
        ax.set_ylabel('Capacity 2')
        ax.set_zlabel('Capacity 3')
        ax.set_title('Cost by Capacity Configuration')
        
        plt.colorbar(scatter, ax=ax, label='Cost')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 


def create_sample_plots(
    num_agents: int = 30,
    num_resources: int = 3,
    capacity: List[float] = None,
    output_dir: str = "visualisation_output"
) -> None:
    """
    Generate sample visualisations for demonstration purposes.
    
    Args:
        num_agents: Number of agents in simulation
        num_resources: Number of resources
        capacity: Resource capacities
        output_dir: Directory to save plots
    """
    import os
    from ..core.simulation import SimulationRunner
    from ..utils.config import Config
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default capacity if not provided
    if capacity is None:
        capacity = [1.0] * num_resources
    elif len(capacity) != num_resources:
        # Adjust capacity to match resources
        capacity = capacity[:num_resources] + [1.0] * (num_resources - len(capacity))
    
    print(f"Generating sample plots with {num_agents} agents, {num_resources} resources")
    print(f"Capacity: {capacity}")
    
    # Create configuration
    config = Config()
    config.num_agents = num_agents
    config.num_resources = num_resources
    config.capacity = capacity
    config.num_iterations = 200
    
    # Run simulation
    runner = SimulationRunner(config)
    runner.setup()
    results = runner.run()
    
    # Generate plots
    
    # 1. Resource distribution plot
    fig1 = plot_resource_distribution(
        consumption=results['final_consumption'],
        capacity=capacity,
        save_path=os.path.join(output_dir, 'resource_distribution.png')
    )
    plt.close(fig1)
    print(f"Resource distribution plot saved")
    
    # 2. Generate multiple runs for comparison
    print("Running multiple simulations for comparison plots...")
    comparison_results = {}
    
    # Different weight values
    weights = [0.3, 0.5, 0.7]
    for weight in weights:
        config.weight = weight
        runner = SimulationRunner(config)
        runner.setup()
        run_results = []
        
        # Run multiple episodes
        for _ in range(5):
            result = runner.run()
            run_results.append(result)
        
        comparison_results[f'weight_{weight}'] = run_results
    
    # 3. Convergence comparison plot
    fig2 = plot_convergence_comparison(
        results_dict=comparison_results,
        metric='entropy',
        save_path=os.path.join(output_dir, 'convergence_comparison.png')
    )
    plt.close(fig2)
    print(f"Convergence comparison plot saved")
    
    # 4. Parameter sensitivity plot
    parameter_results = {}
    for weight in weights:
        metric_values = []
        for result in comparison_results[f'weight_{weight}']:
            # Calculate entropy for each result
            from ..evaluation.metrics import calculate_entropy
            entropy = calculate_entropy(np.array(result['final_consumption']))
            metric_values.append(entropy)
        parameter_results[str(weight)] = metric_values
    
    fig3 = plot_parameter_sensitivity(
        parameter_results=parameter_results,
        parameter_name='Learning Weight',
        metric_name='Entropy',
        save_path=os.path.join(output_dir, 'parameter_sensitivity.png')
    )
    plt.close(fig3)
    print(f"Parameter sensitivity plot saved")
    
    # 5. Try to generate ternary plot if available and num_resources == 3
    if num_resources == 3:
        try:
            from .ternary import plot_ternary_distribution
            fig4 = plot_ternary_distribution(
                consumption_history=[results['final_consumption']],
                save_path=os.path.join(output_dir, 'ternary_distribution.png')
            )
            plt.close(fig4)
            print(f"Ternary plot saved")
        except ImportError:
            print("  mpltern not available - skipping ternary plots")
        except Exception as e:
            print(f"Could not generate ternary plot: {e}")
    
    # 6. Try to generate network plot if available
    try:
        from .network import visualise_state_network
        fig5 = visualise_state_network(
            agent_history=results.get('agent_history', []),
            save_path=os.path.join(output_dir, 'state_network.png')
        )
        plt.close(fig5)
        print(f"Network plot saved")
    except ImportError:
        print("networkx not available - skipping network plots")
    except Exception as e:
        print(f"Could not generate network plot: {e}")
    
    print(f"\nSample plots generated successfully in: {output_dir}")
    print("Available plots:")
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            print(f"  - {filename}")


def generate_analysis_plots(
    results: Union[Dict, List[Dict]], 
    output_dir: str,
    include_network: bool = False,
    include_ternary: bool = False
) -> List[str]:
    """
    Generate comprehensive analysis plots from simulation results.
    
    Args:
        results: Simulation results (single dict or list of dicts)
        output_dir: Directory to save plots
        include_network: Whether to generate network plots
        include_ternary: Whether to generate ternary plots
        
    Returns:
        List of generated plot file paths
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert single result to list for consistency
    if isinstance(results, dict):
        results_list = [results]
    else:
        results_list = results
    
    generated_plots = []
    
    # Extract basic information from first result
    if not results_list:
        print("No results provided for plotting")
        return generated_plots
    
    first_result = results_list[0]
    num_resources = len(first_result.get('final_consumption', []))
    
    # 1. Resource distribution plot
    try:
        if first_result.get('final_consumption') and first_result.get('capacity'):
            fig_path = os.path.join(output_dir, 'resource_distribution.png')
            fig = plot_resource_distribution(
                consumption=first_result['final_consumption'],
                capacity=first_result.get('capacity', [1.0] * num_resources),
                save_path=fig_path
            )
            plt.close(fig)
            generated_plots.append(fig_path)
            print(f"Resource distribution plot saved: {fig_path}")
    except Exception as e:
        print(f"Could not generate resource distribution plot: {e}")
    
    print(f"\nGenerated {len(generated_plots)} analysis plots in: {output_dir}")
    
    return generated_plots 