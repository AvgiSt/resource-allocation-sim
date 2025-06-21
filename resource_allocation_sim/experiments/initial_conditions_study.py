"""Initial conditions study experiment testing Hypothesis 3."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .base_experiment import BaseExperiment
from ..evaluation.agent_analysis import analyse_agent_convergence, plot_probability_distribution
from ..evaluation.system_analysis import analyse_system_performance, plot_cost_evolution
from ..evaluation.metrics import calculate_entropy, calculate_probability_entropy, calculate_convergence_speed
from ..visualisation.plots import plot_parameter_sensitivity, plot_convergence_comparison
from ..visualisation.ternary import plot_ternary_distribution, plot_ternary_trajectory
from ..utils.config import Config


class InitialConditionsStudy(BaseExperiment):
    """
    Study testing hypothesis that system recovers performance when agents have 
    biased initial probability distributions across two of three resources.
    
    Hypothesis 3: The system recovers its performance when agents have an initial 
    probability distribution that is biased across two of the three available resources.
    """
    
    def __init__(self, **kwargs):
        """Initialise initial conditions study."""
        # Remove experiment_name from kwargs if present to avoid conflict
        kwargs.pop('experiment_name', None)
        
        super().__init__(
            experiment_name="initial_conditions_study",
            **kwargs
        )
        
        # Study-specific parameters
        self.analysis_results = {}
        
        # Set up base configuration
        self.base_config = self.setup_base_config()
        
        # Initial condition types to test
        self.initial_condition_types = [
            'uniform',                    # [0.333, 0.333, 0.333]
            'diagonal_point_1',           # [0.4, 0.289, 0.311]
            'diagonal_point_2',           # [0.444, 0.256, 0.3]
            'diagonal_point_3',           # [0.489, 0.222, 0.289]
            'diagonal_point_4',           # [0.533, 0.189, 0.278]
            'diagonal_point_5',           # [0.578, 0.156, 0.266]
            'diagonal_point_6',           # [0.622, 0.122, 0.256]
            'diagonal_point_7',           # [0.667, 0.089, 0.244]
            'diagonal_point_8',           # [0.711, 0.056, 0.233]
            'diagonal_point_9',           # [0.756, 0.022, 0.222]
            'diagonal_point_10',          # [0.8, 0.0, 0.2]
            'edge_bias_12'                # [0.45, 0.45, 0.1]
        ]
    
    def setup_base_config(self) -> Config:
        """Set up base configuration for the study."""
        config = Config()
        config.num_resources = 3  # Fixed at 3 for barycentric analysis
        config.num_agents = 10
        config.relative_capacity = [0.33, 0.33, 0.33]  # Balanced capacity
        config.num_iterations = 1000
        config.weight = 0.3  # Moderate learning rate
        config.agent_initialisation_method = "custom"  # Will be set by custom factory
        return config
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate configurations for different initial condition types."""
        configurations = []
        
        for init_type in self.initial_condition_types:
            config = {
                'num_resources': 3,
                'num_agents': 10,
                'relative_capacity': [0.33, 0.33, 0.33],
                'num_iterations': 1000,
                'weight': 0.3,
                'agent_initialisation_method': 'custom',
                'initial_condition_type': init_type
            }
            configurations.append(config)
        
        return configurations
    
    def generate_initial_probabilities(self, init_type: str, num_agents: int) -> List[List[float]]:
        """
        Generate initial probability distributions for agents based on condition type.
        
        Args:
            init_type: Type of initial condition
            num_agents: Number of agents
            
        Returns:
            List of probability distributions for each agent
        """
        probabilities = []
        
        for _ in range(num_agents):
            if init_type == 'uniform':
                # Standard uniform distribution
                probs = [1/3, 1/3, 1/3]
                
            elif init_type == 'edge_bias_12':
                # Bias towards resources 1 and 2
                probs = [0.45, 0.45, 0.10]
                
            elif init_type == 'edge_bias_13':
                # Bias towards resources 1 and 3
                probs = [0.45, 0.10, 0.45]
                
            elif init_type == 'edge_bias_23':
                # Bias towards resources 2 and 3
                probs = [0.10, 0.45, 0.45]
                
            elif init_type == 'vertex_bias_1':
                # Strong bias towards resource 1
                probs = [0.70, 0.15, 0.15]
                
            elif init_type == 'vertex_bias_2':
                # Strong bias towards resource 2
                probs = [0.15, 0.70, 0.15]
                
            elif init_type == 'vertex_bias_3':
                # Strong bias towards resource 3
                probs = [0.15, 0.15, 0.70]
                
            elif init_type == 'diagonal_point_1':
                # Fixed diagonal point 1: [0.4, 0.289, 0.311]
                probs = [0.4, 0.289, 0.311]
                
            elif init_type == 'diagonal_point_2':
                # Fixed diagonal point 2: [0.444, 0.256, 0.3]
                probs = [0.444, 0.256, 0.3]
                
            elif init_type == 'diagonal_point_3':
                # Fixed diagonal point 3: [0.489, 0.222, 0.289]
                probs = [0.489, 0.222, 0.289]
                
            elif init_type == 'diagonal_point_4':
                # Fixed diagonal point 4: [0.533, 0.189, 0.278]
                probs = [0.533, 0.189, 0.278]
                
            elif init_type == 'diagonal_point_5':
                # Fixed diagonal point 5: [0.578, 0.156, 0.266]
                probs = [0.578, 0.156, 0.266]
                
            elif init_type == 'diagonal_point_6':
                # Fixed diagonal point 6: [0.622, 0.122, 0.256]
                probs = [0.622, 0.122, 0.256]
                
            elif init_type == 'diagonal_point_7':
                # Fixed diagonal point 7: [0.667, 0.089, 0.244]
                probs = [0.667, 0.089, 0.244]
                
            elif init_type == 'diagonal_point_8':
                # Fixed diagonal point 8: [0.711, 0.056, 0.233]
                probs = [0.711, 0.056, 0.233]
                
            elif init_type == 'diagonal_point_9':
                # Fixed diagonal point 9: [0.756, 0.022, 0.222]
                probs = [0.756, 0.022, 0.222]
                
            elif init_type == 'diagonal_point_10':
                # Fixed diagonal point 10: [0.8, 0.0, 0.2]
                probs = [0.8, 0.0, 0.2]
            
            elif init_type == 'uniform_varied':
                # Slight random variations around uniform distribution
                base_uniform = 1/3
                variation = 0.1  # ±10% variation
                probs = [
                    np.random.uniform(base_uniform - variation, base_uniform + variation),
                    np.random.uniform(base_uniform - variation, base_uniform + variation),
                    np.random.uniform(base_uniform - variation, base_uniform + variation)
                ]
                
                # Ensure probabilities are non-negative and sum to 1
                probs = [max(0.01, p) for p in probs]  # Minimum 1% probability
                total = sum(probs)
                probs = [p/total for p in probs]
            
            else:
                # Default to uniform if unknown type
                probs = [1/3, 1/3, 1/3]
            
            probabilities.append(probs)
        
        return probabilities
    
    def create_custom_agent_factory(self, config_params: Dict[str, Any], config: Config):
        """Create custom agent factory for specific initial conditions."""
        init_type = config_params.get('initial_condition_type', 'uniform')
        initial_probabilities = self.generate_initial_probabilities(init_type, config.num_agents)
        
        def agent_factory(agent_id: int, config: Config):
            """Custom agent factory that sets specific initial probabilities."""
            from ..core.agent import Agent
            
            # Create agent with custom initialisation method
            agent = Agent(agent_id, config.num_resources, config.weight, "custom")
            
            # Override with custom initial probabilities
            if agent_id < len(initial_probabilities):
                agent.probabilities = np.array(initial_probabilities[agent_id])
                # Ensure probabilities sum to 1
                agent.probabilities = agent.probabilities / np.sum(agent.probabilities)
            
            return agent
        
        return agent_factory
    
    def analyse_results(self) -> Dict[str, Any]:
        """
        Analyse results comparing different initial conditions.
        
        Returns:
            Dictionary containing comprehensive analysis
        """
        if not hasattr(self, 'results') or not self.results:
            return {}
        
        # Group results by initial condition type
        condition_results = {}
        
        for config_result in self.results:
            for episode_result in config_result['episode_results']:
                init_type = config_result['config_params'].get('initial_condition_type', 'unknown')
                
                if init_type not in condition_results:
                    condition_results[init_type] = []
                
                condition_results[init_type].append(episode_result)
        
        # Analyse each condition
        analysis = {}
        for init_type, episodes in condition_results.items():
            analysis[init_type] = self.analyse_condition_performance(episodes)
        
        # Comparative analysis
        analysis['comparison'] = self.compare_conditions(analysis)
        analysis['hypothesis_support'] = self.evaluate_hypothesis_support(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    def analyse_condition_performance(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse performance metrics for a specific initial condition."""
        metrics = {
            'final_entropies': [],
            'final_costs': [],
            'convergence_times': [],
            'load_balances': [],
            'system_efficiencies': []
        }
        
        for episode in episodes:
            env_state = episode.get('environment_state', {})
            agent_results = episode.get('agent_results', {})
            
            # Final entropy calculation
            consumption_history = env_state.get('consumption_history', [])
            if consumption_history:
                final_consumption = consumption_history[-1]
                final_entropy = calculate_entropy(final_consumption)
                metrics['final_entropies'].append(final_entropy)
                
                # Load balance (standard deviation of final consumption)
                load_balance = np.std(final_consumption)
                metrics['load_balances'].append(load_balance)
            
            # Final cost
            cost_history = env_state.get('cost_history', [])
            if cost_history:
                final_cost = np.sum(cost_history[-1])
                metrics['final_costs'].append(final_cost)
            
            # Convergence analysis
            if agent_results:
                conv_analysis = analyse_agent_convergence(agent_results)
                conv_times = list(conv_analysis.get('convergence_times', {}).values())
                if conv_times:
                    avg_conv_time = np.mean(conv_times)
                    metrics['convergence_times'].append(avg_conv_time)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            else:
                summary[metric_name] = {
                    'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0, 'values': []
                }
        
        return summary
    
    def compare_conditions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different initial conditions."""
        comparison = {
            'best_entropy': None,
            'best_cost': None, 
            'fastest_convergence': None,
            'best_load_balance': None,
            'performance_ranking': []
        }
        
        # Find best performing conditions for each metric
        entropy_scores = {}
        cost_scores = {}
        convergence_scores = {}
        balance_scores = {}
        
        for init_type, results in analysis.items():
            if init_type == 'comparison':  # Skip recursive comparison
                continue
                
            if 'final_entropies' in results and results['final_entropies']['mean'] > 0:
                entropy_scores[init_type] = results['final_entropies']['mean']
            
            if 'final_costs' in results and results['final_costs']['mean'] > 0:
                cost_scores[init_type] = results['final_costs']['mean']
                
            if 'convergence_times' in results and results['convergence_times']['mean'] > 0:
                convergence_scores[init_type] = results['convergence_times']['mean']
                
            if 'load_balances' in results and results['load_balances']['mean'] >= 0:
                balance_scores[init_type] = results['load_balances']['mean']
        
        # Find best (lowest entropy, cost, convergence time, load balance)
        if entropy_scores:
            comparison['best_entropy'] = min(entropy_scores.keys(), key=lambda k: entropy_scores[k])
        if cost_scores:
            comparison['best_cost'] = min(cost_scores.keys(), key=lambda k: cost_scores[k])
        if convergence_scores:
            comparison['fastest_convergence'] = min(convergence_scores.keys(), key=lambda k: convergence_scores[k])
        if balance_scores:
            comparison['best_load_balance'] = min(balance_scores.keys(), key=lambda k: balance_scores[k])
        
        # Overall performance ranking (lower is better)
        overall_scores = {}
        for init_type in entropy_scores.keys():
            # Normalize scores and combine (equal weights)
            norm_entropy = entropy_scores.get(init_type, float('inf'))
            norm_cost = cost_scores.get(init_type, float('inf'))
            norm_conv = convergence_scores.get(init_type, float('inf'))
            norm_balance = balance_scores.get(init_type, float('inf'))
            
            # Combined score (lower is better)
            overall_scores[init_type] = norm_entropy + norm_cost + norm_conv + norm_balance
        
        if overall_scores:
            sorted_conditions = sorted(overall_scores.keys(), key=lambda k: overall_scores[k])
            comparison['performance_ranking'] = sorted_conditions
        
        return comparison
    
    def evaluate_hypothesis_support(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate support for Hypothesis 3."""
        support = {
            'overall_support': 'undetermined',
            'evidence_strength': {},
            'statistical_significance': {},
            'key_findings': []
        }
        
        # Check if biased conditions outperform uniform
        uniform_performance = analysis.get('uniform', {})
        biased_conditions = ['diagonal_point_1', 'diagonal_point_2', 'diagonal_point_3', 'diagonal_point_4', 'diagonal_point_5', 'diagonal_point_6', 'diagonal_point_7', 'diagonal_point_8', 'diagonal_point_9', 'diagonal_point_10', 'edge_bias_12']
        
        better_entropy_count = 0
        better_cost_count = 0
        better_convergence_count = 0
        total_comparisons = 0
        
        if uniform_performance and 'final_entropies' in uniform_performance:
            uniform_entropy = uniform_performance['final_entropies']['mean']
            uniform_cost = uniform_performance.get('final_costs', {}).get('mean', float('inf'))
            uniform_conv = uniform_performance.get('convergence_times', {}).get('mean', float('inf'))
            
            for condition in biased_conditions:
                if condition in analysis:
                    cond_results = analysis[condition]
                    total_comparisons += 1
                    
                    # Compare entropy (lower is better)
                    if 'final_entropies' in cond_results:
                        cond_entropy = cond_results['final_entropies']['mean']
                        if cond_entropy < uniform_entropy:
                            better_entropy_count += 1
                    
                    # Compare cost (lower is better)
                    if 'final_costs' in cond_results:
                        cond_cost = cond_results['final_costs']['mean']
                        if cond_cost < uniform_cost:
                            better_cost_count += 1
                    
                    # Compare convergence (lower is better)
                    if 'convergence_times' in cond_results:
                        cond_conv = cond_results['convergence_times']['mean']
                        if cond_conv < uniform_conv:
                            better_convergence_count += 1
        
        # Calculate evidence strength
        if total_comparisons > 0:
            entropy_ratio = better_entropy_count / total_comparisons
            cost_ratio = better_cost_count / total_comparisons
            conv_ratio = better_convergence_count / total_comparisons
            
            support['evidence_strength'] = {
                'entropy_improvement_ratio': entropy_ratio,
                'cost_improvement_ratio': cost_ratio,
                'convergence_improvement_ratio': conv_ratio,
                'overall_improvement_ratio': (entropy_ratio + cost_ratio + conv_ratio) / 3
            }
            
            # Overall support assessment
            overall_ratio = support['evidence_strength']['overall_improvement_ratio']
            if overall_ratio >= 0.75:
                support['overall_support'] = 'strong'
            elif overall_ratio >= 0.5:
                support['overall_support'] = 'moderate'
            elif overall_ratio >= 0.25:
                support['overall_support'] = 'weak'
            else:
                support['overall_support'] = 'none'
        
        return support
    
    def create_comprehensive_plots(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """Create comprehensive analysis plots for initial conditions study."""
        output_path = Path(output_dir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving plots to: {output_path}")  # Debug print to confirm path
        
        plot_files = []
        
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            self.analyse_results()
        
        # 1. Performance comparison across conditions
        plot_files.extend(self._create_performance_comparison_plots(output_path, show_plots))
        
        # 2. Entropy distribution analysis
        plot_files.extend(self._create_entropy_analysis_plots(output_path, show_plots))
        
        # 3. Barycentric coordinate visualizations
        plot_files.extend(self._create_barycentric_plots(output_path, show_plots))
        
        # 4. Statistical analysis
        plot_files.extend(self._create_statistical_plots(output_path, show_plots))
        
        return plot_files
    
    def _create_performance_comparison_plots(self, output_path: Path, show_plots: bool) -> List[str]:
        """Create performance comparison plots."""
        plot_files = []
        
        # Figure 1: Box plot comparison of final entropies
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))  # Increased figure size for longer labels
        
        # Get coordinate labels
        coordinate_labels = self.get_coordinate_labels()
        
        # Prepare data for box plots
        entropy_data = []
        cost_data = []
        convergence_data = []
        balance_data = []
        labels = []
        
        for init_type, results in self.analysis_results.items():
            if init_type in ['comparison', 'hypothesis_support']:
                continue
                
            if 'final_entropies' in results and results['final_entropies']['values']:
                entropy_data.append(results['final_entropies']['values'])
                # Use coordinate labels instead of condition names
                labels.append(coordinate_labels.get(init_type, init_type))
                
                if 'final_costs' in results and results['final_costs']['values']:
                    cost_data.append(results['final_costs']['values'])
                else:
                    cost_data.append([0])
                    
                if 'convergence_times' in results and results['convergence_times']['values']:
                    convergence_data.append(results['convergence_times']['values'])
                else:
                    convergence_data.append([0])
                    
                if 'load_balances' in results and results['load_balances']['values']:
                    balance_data.append(results['load_balances']['values'])
                else:
                    balance_data.append([0])
        
        # Plot entropy comparison
        ax = axes[0, 0]
        if entropy_data:
            bp = ax.boxplot(entropy_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
        ax.set_title('Final System Entropy Distribution')
        ax.set_ylabel('Entropy')
        ax.tick_params(axis='x', rotation=90)  # Increased rotation for coordinate labels
        
        # Plot cost comparison
        ax = axes[0, 1]
        if cost_data:
            bp = ax.boxplot(cost_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
        ax.set_title('Final System Cost Distribution')
        ax.set_ylabel('Total Cost')
        ax.tick_params(axis='x', rotation=90)  # Increased rotation for coordinate labels
        
        # Plot convergence time comparison
        ax = axes[1, 0]
        if convergence_data:
            bp = ax.boxplot(convergence_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
        ax.set_title('Convergence Time Distribution')
        ax.set_ylabel('Average Convergence Time')
        ax.tick_params(axis='x', rotation=90)  # Increased rotation for coordinate labels
        
        # Plot load balance comparison
        ax = axes[1, 1]
        if balance_data:
            bp = ax.boxplot(balance_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightyellow')
        ax.set_title('Load Balance Distribution')
        ax.set_ylabel('Load Balance (Std Dev)')
        ax.tick_params(axis='x', rotation=90)  # Increased rotation for coordinate labels
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'performance_comparison.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('performance_comparison.png')
        plt.close(fig)
        
        return plot_files
    
    def _create_entropy_analysis_plots(self, output_path: Path, show_plots: bool) -> List[str]:
        """Create entropy analysis plots matching the hypothesis figures."""
        plot_files = []
        
        # Figure 2: Combined agent entropy analysis (like figures in hypothesis description)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: All conditions entropy distribution
        ax = axes[0]
        entropy_data = []
        condition_labels = []
        
        # Get coordinate labels
        coordinate_labels = self.get_coordinate_labels()
        
        for init_type, results in self.analysis_results.items():
            if init_type in ['comparison', 'hypothesis_support']:
                continue
            if 'final_entropies' in results and results['final_entropies']['values']:
                entropy_data.append(results['final_entropies']['values'])
                # Use coordinate labels instead of condition names
                condition_labels.append(coordinate_labels.get(init_type, init_type))
        
        if entropy_data:
            bp = ax.boxplot(entropy_data, labels=condition_labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        ax.set_title('Combined Agent Entropy by Initial Condition')
        ax.set_ylabel('Final System Entropy')
        ax.tick_params(axis='x', rotation=90)  # Increased rotation for coordinate labels
        ax.grid(True, alpha=0.3)
        
        # Right plot: Hypothesis support visualization
        ax = axes[1]
        support_data = self.analysis_results.get('hypothesis_support', {})
        evidence = support_data.get('evidence_strength', {})
        
        if evidence:
            metrics = ['Entropy\nImprovement', 'Cost\nImprovement', 'Convergence\nImprovement', 'Overall\nImprovement']
            values = [
                evidence.get('entropy_improvement_ratio', 0),
                evidence.get('cost_improvement_ratio', 0), 
                evidence.get('convergence_improvement_ratio', 0),
                evidence.get('overall_improvement_ratio', 0)
            ]
            
            bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax.axhline(y=0.5, color='red', linestyle='--', label='Random Performance (50%)')
            ax.set_title('Hypothesis 3 Support Evidence')
            ax.set_ylabel('Proportion of Biased Conditions\nOutperforming Uniform')
            ax.set_ylim(0, 1)
            ax.legend()
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'entropy_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('entropy_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def _create_barycentric_plots(self, output_path: Path, show_plots: bool) -> List[str]:
        """Create barycentric coordinate visualisation plots."""
        from ..visualisation.ternary import plot_initial_conditions_barycentric
        
        plot_files = []
        
        # Create coordinate data dictionary for all initial condition types
        coordinate_data = {}
        for init_type in self.initial_condition_types:
            # Use the same method as the experiment to generate coordinates
            initial_probs = self.generate_initial_probabilities(init_type, 1)  # Just need one example
            if initial_probs:
                coordinate_data[init_type] = initial_probs[0]  # Take the first (and only) set of probabilities
        
        # Create barycentric plot using ternary module with coordinate data
        fig = plot_initial_conditions_barycentric(
            coordinate_data,
            save_path=str(output_path / 'barycentric_initial_conditions.png'),
            title='Initial Probability Distributions in Barycentric Coordinates'
        )
        
        if show_plots:
            plt.show()
        
        plot_files.append('barycentric_initial_conditions.png')
        plt.close(fig)
        
        return plot_files
    
    def _create_statistical_plots(self, output_path: Path, show_plots: bool) -> List[str]:
        """Create statistical analysis summary plots."""
        plot_files = []
        
        # Figure 4: Statistical summary
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance ranking
        ax = axes[0, 0]
        comparison = self.analysis_results.get('comparison', {})
        ranking = comparison.get('performance_ranking', [])
        
        # Get coordinate labels
        coordinate_labels = self.get_coordinate_labels()
        
        if ranking:
            y_pos = np.arange(len(ranking))
            ax.barh(y_pos, range(len(ranking), 0, -1), color='skyblue')
            ax.set_yticks(y_pos)
            # Use coordinate labels instead of condition names
            ranking_labels = [coordinate_labels.get(r, r) for r in ranking]
            ax.set_yticklabels(ranking_labels)
            ax.set_xlabel('Performance Rank (Higher = Better)')
            ax.set_title('Overall Performance Ranking')
        
        # Best condition by metric
        ax = axes[0, 1]
        if comparison:
            metrics = ['Entropy', 'Cost', 'Convergence', 'Load Balance']
            best_conditions = [
                comparison.get('best_entropy', 'N/A'),
                comparison.get('best_cost', 'N/A'),
                comparison.get('fastest_convergence', 'N/A'),
                comparison.get('best_load_balance', 'N/A')
            ]
            
            # Count frequency of each condition being best
            condition_counts = {}
            for condition in best_conditions:
                if condition != 'N/A':
                    condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            if condition_counts:
                conditions = list(condition_counts.keys())
                counts = list(condition_counts.values())
                # Use coordinate labels for condition names
                condition_labels = [coordinate_labels.get(c, c) for c in conditions]
                ax.bar(condition_labels, counts, color='lightcoral')
                ax.set_title('Frequency of Best Performance by Condition')
                ax.set_ylabel('Number of Metrics Where Best')
                ax.tick_params(axis='x', rotation=90)  # Increased rotation for coordinate labels
        
        # Hypothesis support summary
        ax = axes[1, 0]
        support_data = self.analysis_results.get('hypothesis_support', {})
        overall_support = support_data.get('overall_support', 'undetermined')
        
        support_levels = ['none', 'weak', 'moderate', 'strong']
        support_colors = ['red', 'orange', 'yellow', 'green']
        current_level = support_levels.index(overall_support) if overall_support in support_levels else 0
        
        bars = ax.bar(support_levels, [1 if i == current_level else 0 for i in range(len(support_levels))], 
                     color=support_colors)
        ax.set_title(f'Hypothesis 3 Support Level: {overall_support.upper()}')
        ax.set_ylabel('Support Indicator')
        ax.set_ylim(0, 1.2)
        
        # Summary table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        summary_data.append(['PERFORMANCE METRICS', '', ''])
        
        if comparison:
            # Get coordinate labels for summary table
            coordinate_labels = self.get_coordinate_labels()
            
            best_entropy = comparison.get('best_entropy', 'N/A')
            best_cost = comparison.get('best_cost', 'N/A')
            fastest_conv = comparison.get('fastest_convergence', 'N/A')
            
            summary_data.append(['Best Entropy', coordinate_labels.get(best_entropy, best_entropy), ''])
            summary_data.append(['Best Cost', coordinate_labels.get(best_cost, best_cost), ''])
            summary_data.append(['Fastest Convergence', coordinate_labels.get(fastest_conv, fastest_conv), ''])
        
        summary_data.append(['', '', ''])
        summary_data.append(['HYPOTHESIS SUPPORT', '', ''])
        summary_data.append(['Overall Assessment', overall_support.upper(), '✓' if overall_support in ['moderate', 'strong'] else '✗'])
        
        evidence = support_data.get('evidence_strength', {})
        if evidence:
            summary_data.append(['Entropy Improvement', f"{evidence.get('entropy_improvement_ratio', 0):.1%}", ''])
            summary_data.append(['Cost Improvement', f"{evidence.get('cost_improvement_ratio', 0):.1%}", ''])
            summary_data.append(['Overall Improvement', f"{evidence.get('overall_improvement_ratio', 0):.1%}", ''])
        
        if summary_data:
            table = ax.table(cellText=summary_data,
                           colLabels=['Metric', 'Value', 'Status'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        
        ax.set_title('Hypothesis 3 Analysis Summary')
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'statistical_summary.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('statistical_summary.png')
        plt.close(fig)
        
        return plot_files
    
    def save_analysis_results(self, output_dir: str) -> None:
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'analysis_results') and self.analysis_results:
            # Save analysis results as JSON
            with open(output_path / 'analysis_results.json', 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            
            # Generate and save hypothesis report
            report = self.generate_hypothesis_report()
            with open(output_path / 'hypothesis_3_report.txt', 'w') as f:
                f.write(report)
            
            print(f"Analysis results saved to: {output_path}")
    
    def generate_hypothesis_report(self) -> str:
        """Generate comprehensive hypothesis report."""
        if not hasattr(self, 'analysis_results'):
            return "No analysis results available."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INITIAL CONDITIONS STUDY - HYPOTHESIS 3 ANALYSIS REPORT")
        report_lines.append("=" * 80)
        
        # Executive Summary
        report_lines.append("\nEXECUTIVE SUMMARY:")
        report_lines.append("-" * 40)
        
        support_data = self.analysis_results.get('hypothesis_support', {})
        overall_support = support_data.get('overall_support', 'undetermined')
        report_lines.append(f"Hypothesis 3 Support Level: {overall_support.upper()}")
        
        evidence = support_data.get('evidence_strength', {})
        if evidence:
            report_lines.append(f"Overall Improvement Ratio: {evidence.get('overall_improvement_ratio', 0):.1%}")
            report_lines.append(f"Entropy Improvement: {evidence.get('entropy_improvement_ratio', 0):.1%}")
            report_lines.append(f"Cost Improvement: {evidence.get('cost_improvement_ratio', 0):.1%}")
            report_lines.append(f"Convergence Improvement: {evidence.get('convergence_improvement_ratio', 0):.1%}")
        
        # Best Performing Conditions
        comparison = self.analysis_results.get('comparison', {})
        if comparison:
            report_lines.append("\nBEST PERFORMING CONDITIONS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Best Entropy: {comparison.get('best_entropy', 'N/A')}")
            report_lines.append(f"Best Cost: {comparison.get('best_cost', 'N/A')}")
            report_lines.append(f"Fastest Convergence: {comparison.get('fastest_convergence', 'N/A')}")
            report_lines.append(f"Best Load Balance: {comparison.get('best_load_balance', 'N/A')}")
            
            ranking = comparison.get('performance_ranking', [])
            if ranking:
                report_lines.append(f"\nOverall Performance Ranking:")
                for i, condition in enumerate(ranking, 1):
                    report_lines.append(f"  {i}. {condition.replace('_', ' ').title()}")
        
        # Detailed Results by Condition
        report_lines.append("\nDETAILED RESULTS BY CONDITION:")
        report_lines.append("-" * 40)
        
        for condition, results in self.analysis_results.items():
            if condition in ['comparison', 'hypothesis_support']:
                continue
            
            report_lines.append(f"\n{condition.replace('_', ' ').title()}:")
            
            if 'final_entropies' in results:
                entropy_stats = results['final_entropies']
                report_lines.append(f"  Final Entropy: {entropy_stats['mean']:.3f} ± {entropy_stats['std']:.3f}")
            
            if 'final_costs' in results:
                cost_stats = results['final_costs']
                report_lines.append(f"  Final Cost: {cost_stats['mean']:.3f} ± {cost_stats['std']:.3f}")
            
            if 'convergence_times' in results:
                conv_stats = results['convergence_times']
                report_lines.append(f"  Convergence Time: {conv_stats['mean']:.1f} ± {conv_stats['std']:.1f}")
            
            if 'load_balances' in results:
                balance_stats = results['load_balances']
                report_lines.append(f"  Load Balance: {balance_stats['mean']:.3f} ± {balance_stats['std']:.3f}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)

    def get_coordinate_labels(self) -> Dict[str, str]:
        """Get coordinate labels for each condition type."""
        coordinate_mapping = {
            'uniform': '[0.333, 0.333, 0.333]',
            'diagonal_point_1': '[0.4, 0.289, 0.311]',
            'diagonal_point_2': '[0.444, 0.256, 0.3]',
            'diagonal_point_3': '[0.489, 0.222, 0.289]',
            'diagonal_point_4': '[0.533, 0.189, 0.278]',
            'diagonal_point_5': '[0.578, 0.156, 0.266]',
            'diagonal_point_6': '[0.622, 0.122, 0.256]',
            'diagonal_point_7': '[0.667, 0.089, 0.244]',
            'diagonal_point_8': '[0.711, 0.056, 0.233]',
            'diagonal_point_9': '[0.756, 0.022, 0.222]',
            'diagonal_point_10': '[0.8, 0.0, 0.2]',
            'edge_bias_12': '[0.45, 0.45, 0.1]'
        }
        return coordinate_mapping


def run_initial_conditions_study(
    num_replications: int = 100,
    num_iterations: int = 1000,
    output_dir: str = "results",
    show_plots: bool = False
) -> InitialConditionsStudy:
    """
    Run complete initial conditions study.
    
    Args:
        num_replications: Number of replications per initial condition
        num_iterations: Number of iterations per simulation
        output_dir: Output directory for results
        show_plots: Whether to display plots interactively
        
    Returns:
        Completed InitialConditionsStudy instance
    """
    # Create experiment with proper directory nesting
    study = InitialConditionsStudy(
        results_dir=f"{output_dir}/initial_conditions_study",
        experiment_name="initial_conditions_study"
    )
    
    # Update configuration
    study.base_config.num_iterations = num_iterations
    
    print(f"Running initial conditions study...")
    print(f"Replications per condition: {num_replications}")
    print(f"Iterations per simulation: {num_iterations}")
    print(f"Initial condition types: {len(study.initial_condition_types)}")
    
    # Run experiment using BaseExperiment interface
    full_results = study.run_experiment(num_episodes=num_replications)
    
    # Generate analysis and plots
    print("Generating analysis...")
    analysis = study.analyse_results()
    
    print("Creating visualisations...")
    actual_results_dir = study.get_results_dir()
    plot_files = study.create_comprehensive_plots(actual_results_dir, show_plots=show_plots)
    print(f"Generated {len(plot_files)} plots")
    
    # Save analysis results and report
    print("Saving analysis results...")
    study.save_analysis_results(actual_results_dir)
    print(f"Analysis saved to: {actual_results_dir}")
    
    # Print summary
    support_data = analysis.get('hypothesis_support', {})
    overall_support = support_data.get('overall_support', 'undetermined')
    print(f"\nHypothesis 3 Support: {overall_support.upper()}")
    
    if 'evidence_strength' in support_data:
        evidence = support_data['evidence_strength']
        print(f"Overall Improvement Ratio: {evidence.get('overall_improvement_ratio', 0):.1%}")
    
    comparison = analysis.get('comparison', {})
    if comparison:
        print(f"Best Entropy Condition: {comparison.get('best_entropy', 'N/A')}")
        print(f"Best Cost Condition: {comparison.get('best_cost', 'N/A')}")
    
    print(f"\nInitial conditions study completed!")
    print(f"Results available in: {study.get_results_dir()}/")
    
    return study


def test_directory_structure():
    """Test function to verify directory structure is correct."""
    print("Testing directory structure...")
    
    # Run minimal test
    study = run_initial_conditions_study(
        num_replications=100,
        num_iterations=1000,
        show_plots=False,
        output_dir="results"
    )
    
    # Verify directory structure
    results_dir = study.get_results_dir()
    
    expected_files = [
        'full_results.pickle',
        'metadata.json', 
        'summary_results.csv',
        'analysis_results.json',
        'hypothesis_3_report.txt'
    ]
    
    expected_plots = [
        'performance_comparison.png',
        'entropy_analysis.png',
        'barycentric_initial_conditions.png', 
        'statistical_summary.png'
    ]
    
    # Check main files
    for filename in expected_files:
        filepath = results_dir / filename
        if filepath.exists():
            print(f"{filename} created successfully")
        else:
            print(f"{filename} missing")
    
    # Check plots directory
    plots_dir = results_dir / 'plots'
    if plots_dir.exists():
        print(f"plots/ directory created")
        
        for plotname in expected_plots:
            plotpath = plots_dir / plotname
            if plotpath.exists():
                print(f" {plotname} created")
            else:
                print(f" {plotname} missing")
    else:
        print("plots/ directory missing")
    
    print(f"\nAll files saved to: {results_dir}")
    print(f"Directory structure: results/initial_conditions_study/initial_conditions_study_YYYYMMDD_HHMMSS/")
    return results_dir


if __name__ == "__main__":
    # Run test by default
    test_directory_structure() 