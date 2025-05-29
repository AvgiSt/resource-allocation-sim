"""General plotting utilities for simulation visualization."""

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
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare convergence across different configurations.
    
    Args:
        results_dict: Dictionary mapping config names to result lists
        metric: Metric to plot ('entropy', 'cost', 'gini')
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
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
            all_data.append({'Config': config_name, metric.title(): value, 'Episode': len(values)})
        
        # Plot time series for this config
        axes[0].plot(values, label=config_name, alpha=0.7, linewidth=2)
    
    # Time series plot
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(metric.title())
    axes[0].set_title(f'{metric.title()} Evolution by Configuration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    df = pd.DataFrame(all_data)
    sns.boxplot(data=df, x='Config', y=metric.title(), ax=axes[1])
    axes[1].set_title(f'{metric.title()} Distribution by Configuration')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Histogram
    for config_name, results_list in results_dict.items():
        values = [all_data[i][metric.title()] for i, d in enumerate(all_data) if d['Config'] == config_name]
        axes[2].hist(values, alpha=0.6, label=config_name, bins=15)
    
    axes[2].set_xlabel(metric.title())
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'{metric.title()} Histogram')
    axes[2].legend()
    
    # Summary statistics
    summary_data = []
    for config_name, results_list in results_dict.items():
        values = [all_data[i][metric.title()] for i, d in enumerate(all_data) if d['Config'] == config_name]
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
        normalized_caps = capacities / total_cap[:, np.newaxis]
        
        # Create ternary contour plot
        fig = ff.create_ternary_contour(
            normalized_caps.T, np.array(costs),
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