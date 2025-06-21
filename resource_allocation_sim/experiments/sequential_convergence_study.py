"""Sequential convergence study testing Hypothesis 2."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .base_experiment import BaseExperiment
from ..evaluation.agent_analysis import analyse_agent_convergence, plot_probability_distribution, plot_visited_probabilities
from ..evaluation.system_analysis import (
    analyse_system_performance, 
    perform_convergence_statistical_tests,
    test_sequential_convergence_hypothesis,
    calculate_system_convergence_metrics,
    evaluate_hypothesis_support as evaluate_system_hypothesis_support
)
from ..evaluation.metrics import calculate_entropy, calculate_probability_entropy, calculate_convergence_speed
from ..visualisation.plots import plot_parameter_sensitivity, plot_convergence_comparison
from ..visualisation.ternary import plot_ternary_distribution, plot_ternary_trajectory
from ..utils.config import Config


class SequentialConvergenceStudy(BaseExperiment):
    """
    Study testing hypothesis that agents sequentially converge to degenerate distributions.
    
    Hypothesis 2: Agents with uniform initial distribution across available choices, 
    sequentially converge to a degenerate distribution, with each agent becoming 
    certain of a single choice one after the other.
    """
    
    def __init__(self, **kwargs):
        """Initialise sequential convergence study."""
        # Extract convergence thresholds before passing kwargs to parent
        convergence_threshold_entropy = kwargs.pop('convergence_threshold_entropy', 0.5)
        convergence_threshold_max_prob = kwargs.pop('convergence_threshold_max_prob', 0.6)
        
        # Remove experiment_name from kwargs if present to avoid conflict
        kwargs.pop('experiment_name', None)
        
        super().__init__(
            experiment_name="sequential_convergence_study",
            **kwargs
        )
        
        # Study-specific parameters - relaxed for shorter runs
        self.convergence_threshold_entropy = convergence_threshold_entropy
        self.convergence_threshold_max_prob = convergence_threshold_max_prob
        self.analysis_results = {}
        
        # Set up base configuration
        self.base_config = self.setup_base_config()
    
    def setup_base_config(self) -> Config:
        """Set up base configuration ensuring uniform initial distributions."""
        config = Config()
        config.num_resources = 3  # Changed to 3 for ternary plots
        config.num_agents = 10
        config.capacity = [15, 15, 15]  # Updated for 3 resources
        config.num_iterations = 2000  # Longer to observe full convergence
        config.weight = 0.3  # Moderate learning rate
        config.agent_initialisation_method = "uniform"  # Critical for hypothesis
        return config
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate configurations for the sequential convergence study.
        
        Since we're testing a single hypothesis, we use one main configuration
        with uniform initial distributions.
        """
        # Single configuration that ensures uniform starting conditions
        config = {
            'num_resources': 3,  # Changed to 3 for ternary plots
            'num_agents': 10,
            'capacity': [0.3, 0.3, 0.3],  # Updated for 3 resources
            'num_iterations': 2000,
            'weight': 0.3,
            'agent_initialisation_method': 'uniform'
        }
        
        return [config]
    
    def analyse_results(self) -> Dict[str, Any]:
        """
        Analyse results for sequential convergence patterns.
        
        Returns:
            Dictionary containing comprehensive analysis of sequential convergence
        """
        if not hasattr(self, 'results') or not self.results:
            return {}
        
        # Extract convergence data from all replications
        all_convergence_data = []
        
        for config_result in self.results:
            for episode_result in config_result['episode_results']:
                # analyse convergence for this episode
                agent_results = episode_result['agent_results']
                convergence_data = self.analyse_individual_convergence(agent_results)
                convergence_data['episode'] = episode_result['episode']
                all_convergence_data.append(convergence_data)
        
        # Store convergence data for detailed analysis
        self.convergence_episodes = all_convergence_data
        
        # Perform comprehensive analysis
        analysis = self.analyse_convergence_sequence_from_episodes(all_convergence_data)
        
        return analysis
    
    def analyse_convergence_sequence_from_episodes(self, convergence_episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """analyse convergence patterns from episode data."""
        all_convergence_times = []
        all_sequential_indices = []
        all_degeneracy_scores = []
        replication_convergence_data = []
        
        for conv_data in convergence_episodes:
            # Extract convergence times
            conv_times = list(conv_data.get('agent_convergence_times', {}).values())
            if conv_times:
                all_convergence_times.extend(conv_times)
                
                # Calculate sequential index for this episode
                seq_index = self.calculate_sequential_index(conv_times)
                all_sequential_indices.append(seq_index)
                
                # Extract degeneracy scores
                deg_scores = list(conv_data.get('agent_degeneracy_scores', {}).values())
                all_degeneracy_scores.extend(deg_scores)
                
                # Store detailed data for this episode
                conv_order = conv_data.get('convergence_order', [])
                replication_convergence_data.append({
                    'convergence_times': conv_times,
                    'sequential_index': seq_index,
                    'convergence_order': [agent_id for _, agent_id in conv_order],
                    'num_converged': len(conv_times),
                    'degeneracy_scores': deg_scores,
                    'episode': conv_data.get('episode', 0)
                })
        
        # Calculate summary statistics
        analysis = {
            'convergence_times': {
                'all_times': all_convergence_times,
                'mean_time': np.mean(all_convergence_times) if all_convergence_times else 0,
                'std_time': np.std(all_convergence_times) if all_convergence_times else 0,
                'median_time': np.median(all_convergence_times) if all_convergence_times else 0
            },
            'sequential_indices': {
                'all_indices': all_sequential_indices,
                'mean_index': np.mean(all_sequential_indices) if all_sequential_indices else 0,
                'std_index': np.std(all_sequential_indices) if all_sequential_indices else 0
            },
            'degeneracy_scores': {
                'all_scores': all_degeneracy_scores,
                'mean_score': np.mean(all_degeneracy_scores) if all_degeneracy_scores else 0,
                'proportion_degenerate': np.mean([s > self.convergence_threshold_max_prob for s in all_degeneracy_scores]) if all_degeneracy_scores else 0
            },
            'replication_data': replication_convergence_data
        }
        
        # Statistical tests
        analysis['statistical_tests'] = self.perform_statistical_tests(analysis)
        
        # Hypothesis evaluation
        analysis['hypothesis_support'] = self.evaluate_hypothesis_support(analysis)
        
        return analysis
    
    def run_single_simulation(self) -> Dict[str, Any]:
        """Run single simulation and extract convergence data."""
        from ..core.simulation import SimulationRunner
        
        # Create simulation with base config
        runner = SimulationRunner(self.base_config)
        runner.setup()
        results = runner.run()
        
        # Extract agent convergence information
        agent_results = results['agent_results']
        convergence_data = self.analyse_individual_convergence(agent_results)
        
        # Add convergence data to results
        results['convergence_analysis'] = convergence_data
        
        return results
    
    def analyse_individual_convergence(self, agent_results: Dict[int, Dict[str, List]]) -> Dict[str, Any]:
        """Analyse individual agent convergence patterns."""
        convergence_data = {
            'agent_convergence_times': {},
            'agent_degeneracy_scores': {},
            'agent_preferred_resources': {},
            'entropy_evolution': {},
            'max_prob_evolution': {},
            'convergence_order': [],
            'converged_agents': set()
        }
        
        # Process each agent
        for agent_id, data in agent_results.items():
            prob_history = np.array(data['prob'])
            
            # Calculate entropy and max probability evolution
            # Use probability entropy for agent probability distributions
            entropies = [calculate_probability_entropy(probs) for probs in prob_history]
            max_probs = [np.max(probs) for probs in prob_history]
            
            convergence_data['entropy_evolution'][agent_id] = entropies
            convergence_data['max_prob_evolution'][agent_id] = max_probs
            
            # Find convergence time
            convergence_time = self.find_convergence_time(entropies, max_probs)
            
            if convergence_time is not None:
                convergence_data['agent_convergence_times'][agent_id] = convergence_time
                convergence_data['agent_degeneracy_scores'][agent_id] = max_probs[convergence_time]
                convergence_data['agent_preferred_resources'][agent_id] = np.argmax(prob_history[convergence_time])
                convergence_data['converged_agents'].add(agent_id)
                
                # Add to convergence order
                convergence_data['convergence_order'].append((convergence_time, agent_id))
        
        # Sort convergence order by time
        convergence_data['convergence_order'].sort()
        
        return convergence_data
    
    def find_convergence_time(self, entropies: List[float], max_probs: List[float]) -> Optional[int]:
        """Find when agent converged based on entropy and max probability criteria."""
        for i, (entropy, max_prob) in enumerate(zip(entropies, max_probs)):
            if entropy < self.convergence_threshold_entropy and max_prob > self.convergence_threshold_max_prob:
                return i
        return None
    
    def calculate_sequential_index(self, convergence_times: List[int]) -> float:
        """
        Calculate sequential index - measure of how sequential vs simultaneous convergence is.
        
        Returns value between 0 (perfectly simultaneous) and 1 (perfectly sequential).
        """
        if len(convergence_times) <= 1:
            return 1.0
        
        # Sort convergence times
        sorted_times = sorted(convergence_times)
        
        # Calculate actual spread
        actual_spread = sorted_times[-1] - sorted_times[0]
        
        # Calculate maximum possible spread (if perfectly sequential)
        max_possible_spread = len(convergence_times) - 1
        
        # Sequential index
        if max_possible_spread == 0:
            return 1.0
        
        return min(actual_spread / max_possible_spread, 1.0)
    
    def analyse_convergence_sequence(self) -> Dict[str, Any]:
        """Analyse convergence sequence patterns across all replications."""
        # Use the convergence episodes if available, otherwise extract from results
        if hasattr(self, 'convergence_episodes'):
            return self.analyse_convergence_sequence_from_episodes(self.convergence_episodes)
        
        # Fallback: extract from results if convergence_episodes not available
        all_convergence_data = []
        if hasattr(self, 'results') and self.results:
            for result in self.results:
                if 'convergence_analysis' in result:
                    all_convergence_data.append(result['convergence_analysis'])
        
        return self.analyse_convergence_sequence_from_episodes(all_convergence_data)
    
    def perform_statistical_tests(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests for sequential convergence hypothesis using system analysis functions."""
        # Use system analysis functions for statistical testing
        basic_tests = perform_convergence_statistical_tests(analysis)
        hypothesis_tests = test_sequential_convergence_hypothesis(
            analysis, 
            convergence_threshold_max_prob=self.convergence_threshold_max_prob,
            expected_degeneracy_proportion=0.9
        )
        
        # Combine all tests
        tests = {**basic_tests, **hypothesis_tests}
        return tests
    
    def create_convergence_timeline_plots(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """Create convergence timeline analysis plots."""
        output_path = Path(output_dir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        # Get convergence analysis using the proper method
        conv_analysis = self.analyse_convergence_sequence()
        
        # Figure 1: Convergence Timeline Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Agent Convergence Timeline for first few replications
        ax = axes[0, 0]
        replication_data = conv_analysis.get('replication_data', [])
        for i, rep_data in enumerate(replication_data[:5]):  # Show first 5 replications
            conv_order = rep_data.get('convergence_order', [])
            conv_times = rep_data.get('convergence_times', [])
            
            for j, (agent_id, time) in enumerate(zip(conv_order, sorted(conv_times))):
                ax.barh(i * len(conv_order) + j, 50, left=time, height=0.8, 
                       label=f'Agent {agent_id}' if i == 0 else "", alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Replication * Agent')
        ax.set_title('Agent Convergence Timeline')
        
        # (b) Cumulative Convergence Plot
        ax = axes[0, 1]
        if replication_data:
            # Average cumulative convergence across replications
            all_times = conv_analysis.get('convergence_times', {}).get('all_times', [])
            max_time = max(all_times) if all_times else 2000
            time_points = np.arange(0, max_time + 1)
            
            cumulative_curves = []
            for rep_data in replication_data:
                conv_times = sorted(rep_data.get('convergence_times', []))
                cumulative = np.zeros(len(time_points))
                
                conv_count = 0
                for i, t in enumerate(time_points):
                    while conv_count < len(conv_times) and conv_times[conv_count] <= t:
                        conv_count += 1
                    cumulative[i] = conv_count
                
                cumulative_curves.append(cumulative)
            
            if cumulative_curves:
                # Plot mean and confidence interval
                mean_cumulative = np.mean(cumulative_curves, axis=0)
                std_cumulative = np.std(cumulative_curves, axis=0)
                
                ax.plot(time_points, mean_cumulative, 'b-', linewidth=2, label='Mean')
                ax.fill_between(time_points, 
                               mean_cumulative - std_cumulative,
                               mean_cumulative + std_cumulative,
                               alpha=0.3, label='±1 SD')
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Number of Converged Agents')
                ax.set_title('Cumulative Convergence Pattern')
                ax.legend()
        
        # (c) Entropy Evolution Heatmap for one replication
        ax = axes[1, 0]
        if hasattr(self, 'convergence_episodes') and self.convergence_episodes:
            entropy_data = self.convergence_episodes[0].get('entropy_evolution', {})
            
            if entropy_data:
                # Create entropy matrix
                max_iter = min(500, max(len(entropies) for entropies in entropy_data.values()) if entropy_data else 500)
                entropy_matrix = np.full((len(entropy_data), max_iter), np.nan)
                
                for i, (agent_id, entropies) in enumerate(entropy_data.items()):
                    n_vals = min(max_iter, len(entropies))
                    entropy_matrix[i, :n_vals] = entropies[:n_vals]
                
                im = ax.imshow(entropy_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Agent ID')
                ax.set_title('Entropy Evolution Heatmap')
                plt.colorbar(im, ax=ax, label='Entropy')
        
        # (d) Sequential Index Distribution
        ax = axes[1, 1]
        seq_indices = conv_analysis.get('sequential_indices', {}).get('all_indices', [])
        if seq_indices:
            ax.hist(seq_indices, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(seq_indices), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(seq_indices):.3f}')
            ax.axvline(0.5, color='orange', linestyle='--', 
                      label='Random (0.5)')
            ax.set_xlabel('Sequential Index')
            ax.set_ylabel('Frequency')
            ax.set_title('Sequential Index Distribution')
            ax.legend()
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'convergence_timeline_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('convergence_timeline_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def create_probability_evolution_plots(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """Create probability evolution analysis plots."""
        output_path = Path(output_dir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        conv_analysis = self.analyse_convergence_sequence()
        
        # Figure 2: Probability Evolution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Individual Probability Trajectories for one replication
        ax = axes[0, 0]
        if hasattr(self, 'results') and self.results:
            # Get first episode data
            episode_results = self.results[0]['episode_results']
            if episode_results:
                agent_results = episode_results[0]['agent_results']
                colors = plt.cm.tab10(np.linspace(0, 1, self.base_config.num_resources))
                
                # Plot first agent's trajectory
                first_agent = list(agent_results.keys())[0]
                prob_history = np.array(agent_results[first_agent]['prob'])
                
                for resource in range(self.base_config.num_resources):
                    ax.plot(prob_history[:500, resource], color=colors[resource], 
                           label=f'Resource {resource}', linewidth=2)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Probability')
                ax.set_title(f'Agent {first_agent} Probability Evolution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # (b) Resource Preference Heatmap
        ax = axes[0, 1]
        replication_data = conv_analysis.get('replication_data', [])
        if replication_data and hasattr(self, 'results'):
            # Collect final preferences across all episodes
            preference_matrix = []
            for config_result in self.results:
                for episode_result in config_result['episode_results']:
                    agent_results = episode_result['agent_results']
                    rep_prefs = []
                    for agent_id in sorted(agent_results.keys()):
                        final_probs = agent_results[agent_id]['prob'][-1]
                        rep_prefs.append(final_probs)
                    if rep_prefs:
                        preference_matrix.append(rep_prefs)
            
            if preference_matrix:
                # Average preferences across replications
                avg_preferences = np.mean(preference_matrix, axis=0)
                
                im = ax.imshow(avg_preferences, aspect='auto', cmap='Blues')
                ax.set_xlabel('Resource')
                ax.set_ylabel('Agent')
                ax.set_title('Average Final Probability Preferences')
                plt.colorbar(im, ax=ax, label='Probability')
        
        # (c) Convergence Order Matrix
        ax = axes[1, 0]
        convergence_orders = [rep_data.get('convergence_order', []) for rep_data in replication_data]
        if len(convergence_orders) > 1:
            # Create matrix showing convergence position for each agent across replications
            order_matrix = np.full((len(convergence_orders), self.base_config.num_agents), np.nan)
            
            for rep_idx, order in enumerate(convergence_orders):
                for pos, agent_id in enumerate(order):
                    if agent_id < self.base_config.num_agents:
                        order_matrix[rep_idx, agent_id] = pos
            
            im = ax.imshow(order_matrix, aspect='auto', cmap='viridis')
            ax.set_xlabel('Agent ID')
            ax.set_ylabel('Replication')
            ax.set_title('Convergence Order Matrix')
            plt.colorbar(im, ax=ax, label='Convergence Position')
        
        # (d) Degeneracy Score Distribution
        ax = axes[1, 1]
        deg_scores = conv_analysis.get('degeneracy_scores', {}).get('all_scores', [])
        if deg_scores:
            ax.hist(deg_scores, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(self.convergence_threshold_max_prob, color='red', linestyle='--',
                      label=f'Threshold: {self.convergence_threshold_max_prob}')
            ax.axvline(np.mean(deg_scores), color='orange', linestyle='--',
                      label=f'Mean: {np.mean(deg_scores):.3f}')
            ax.set_xlabel('Maximum Probability at Convergence')
            ax.set_ylabel('Frequency')
            ax.set_title('Degeneracy Score Distribution')
            ax.legend()
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'probability_evolution_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('probability_evolution_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def create_barycentric_trajectory_plots(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """Create barycentric coordinate trajectory plots for 3-resource case."""
        output_path = Path(output_dir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        if self.base_config.num_resources != 3:
            print("Barycentric plots only available for 3 resources")
            return plot_files
        
        if not hasattr(self, 'results') or not self.results:
            print("No results available for barycentric plots")
            return plot_files
        
        try:
            # Get agent results from first episode for detailed trajectory analysis
            episode_results = self.results[0]['episode_results']
            if episode_results:
                agent_results = episode_results[0]['agent_results']
                
                # Figure 5a: All agent trajectories on single ternary plot
                fig = plot_visited_probabilities(
                    agent_results,
                    save_path=output_path / 'barycentric_all_trajectories.png'
                )
                if show_plots:
                    plt.show()
                plt.close(fig)
                plot_files.append('barycentric_all_trajectories.png')
                
                # Figure 5b: Final distribution positions
                fig = plot_ternary_distribution(
                    agent_results,
                    save_path=output_path / 'barycentric_final_distributions.png',
                    title="Sequential Convergence: Final Agent Distributions"
                )
                if show_plots:
                    plt.show()
                plt.close(fig)
                plot_files.append('barycentric_final_distributions.png')
                
                # Figure 5c: Individual agent trajectories (first 4 agents)
                fig, axes = plt.subplots(2, 2, figsize=(16, 16))
                axes = axes.flatten()
                
                agent_ids = list(agent_results.keys())[:4]
                for i, agent_id in enumerate(agent_ids):
                    try:
                        # Create individual trajectory plot
                        temp_fig = plot_ternary_trajectory(
                            agent_results,
                            agent_id,
                            save_path=None
                        )
                        # Copy the plot to subplot (this is a bit hacky but works)
                        plt.close(temp_fig)
                        
                        # Manual ternary trajectory for subplot
                        ax = axes[i]
                        prob_history = np.array(agent_results[agent_id]['prob'])
                        
                        # Define triangle vertices for manual projection
                        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
                        projected = np.dot(prob_history, triangle)
                        
                        # Draw triangle
                        triangle_closed = np.vstack([triangle, triangle[0]])
                        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', linewidth=2)
                        
                        # Plot trajectory with time coloring
                        colors = plt.cm.viridis(np.linspace(0, 1, len(projected)))
                        ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=20, alpha=0.7)
                        ax.plot(projected[:, 0], projected[:, 1], '-', color='gray', alpha=0.3)
                        
                        # Mark start and end
                        ax.scatter(projected[0, 0], projected[0, 1], c='green', s=100, 
                                  marker='s', label='Start', zorder=5)
                        ax.scatter(projected[-1, 0], projected[-1, 1], c='red', s=100,
                                  marker='*', label='End', zorder=5)
                        
                        # Labels
                        label_offset = 0.05
                        ax.text(triangle[0, 0] - label_offset, triangle[0, 1] - label_offset, 
                                'Resource 2', ha='center', va='top', fontsize=10)
                        ax.text(triangle[1, 0] + label_offset, triangle[1, 1] - label_offset, 
                                'Resource 3', ha='center', va='top', fontsize=10)
                        ax.text(triangle[2, 0], triangle[2, 1] + label_offset, 
                                'Resource 1', ha='center', va='bottom', fontsize=10)
                        
                        ax.set_title(f'Agent {agent_id} Trajectory')
                        ax.set_aspect('equal')
                        ax.axis('off')
                        if i == 0:  # Only show legend on first subplot
                            ax.legend()
                        
                    except Exception as e:
                        print(f"Could not create trajectory for Agent {agent_id}: {e}")
                        ax.text(0.5, 0.5, f'Agent {agent_id}\nTrajectory Unavailable', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                
                plt.suptitle('Individual Agent Trajectories in Barycentric Coordinates', fontsize=16)
                plt.tight_layout()
                
                if show_plots:
                    plt.show()
                    
                plot_file = output_path / 'barycentric_individual_trajectories.png'
                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                plot_files.append('barycentric_individual_trajectories.png')
                plt.close(fig)
                
                print(f"Generated {len(plot_files)} barycentric coordinate plots")
        
        except Exception as e:
            print(f"Error creating barycentric plots: {e}")
        
        return plot_files
    
    def create_system_dynamics_plots(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """Create system dynamics analysis plots."""
        output_path = Path(output_dir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        conv_analysis = self.analyse_convergence_sequence()
        
        # Figure 3: System Dynamics Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Resource Competition Timeline
        ax = axes[0, 0]
        if hasattr(self, 'results') and self.results:
            # Get resource consumption over time from first episode
            episode_results = self.results[0]['episode_results']
            if episode_results:
                env_state = episode_results[0].get('environment_state', {})
                consumption_history = env_state.get('consumption_history', [])
                
                if consumption_history:
                    # Convert to array for easier manipulation
                    consumption_array = np.array(consumption_history)
                    time_points = np.arange(len(consumption_array))
                    
                    # Create stacked area plot
                    resource_labels = [f'Resource {i}' for i in range(self.base_config.num_resources)]
                    ax.stackplot(time_points, consumption_array.T, labels=resource_labels, alpha=0.7)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Resource Consumption')
                    ax.set_title('Resource Competition Timeline')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
        
        # (b) Performance During Convergence
        ax = axes[0, 1]
        if hasattr(self, 'results') and self.results:
            episode_results = self.results[0]['episode_results']
            if episode_results:
                env_state = episode_results[0].get('environment_state', {})
                cost_history = env_state.get('cost_history', [])
                
                if cost_history:
                    # Calculate system metrics over time
                    total_costs = [sum(costs) for costs in cost_history]
                    entropies = []
                    
                    consumption_history = env_state.get('consumption_history', [])
                    for consumption in consumption_history:
                        entropy = calculate_entropy(consumption) if consumption else 0
                        entropies.append(entropy)
                    
                    time_points = np.arange(len(total_costs))
                    
                    # Plot total cost
                    ax2 = ax.twinx()
                    line1 = ax.plot(time_points, total_costs, 'b-', label='Total Cost', linewidth=2)
                    line2 = ax2.plot(time_points, entropies, 'r-', label='System Entropy', linewidth=2)
                    
                    # Mark convergence events
                    replication_data = conv_analysis.get('replication_data', [])
                    if replication_data:
                        conv_times = replication_data[0].get('convergence_times', [])
                        for i, conv_time in enumerate(sorted(conv_times)):
                            if conv_time < len(time_points):
                                ax.axvline(conv_time, color='green', linestyle='--', alpha=0.7)
                                if i == 0:  # Only label first one
                                    ax.axvline(conv_time, color='green', linestyle='--', alpha=0.7, label='Agent Convergence')
                    
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Total Cost', color='b')
                    ax2.set_ylabel('System Entropy', color='r')
                    ax.set_title('Performance During Convergence')
                    
                    # Combine legends
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper right')
                    ax.grid(True, alpha=0.3)
        
        # (c) Agent Interaction Effects  
        ax = axes[1, 0]
        if hasattr(self, 'results') and self.results:
            episode_results = self.results[0]['episode_results']
            if episode_results:
                agent_results = episode_results[0]['agent_results']
                
                # Calculate correlation matrix between agent final preferences
                final_probs = []
                agent_ids = []
                for agent_id, data in agent_results.items():
                    if data['prob']:
                        final_probs.append(data['prob'][-1])
                        agent_ids.append(agent_id)
                
                if len(final_probs) > 1:
                    try:
                        with np.errstate(invalid='ignore'):
                            correlation_matrix = np.corrcoef(final_probs)
                            # Handle NaN values
                            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
                    except (ValueError, np.linalg.LinAlgError):
                        # Fallback to identity matrix if correlation fails
                        correlation_matrix = np.eye(len(final_probs))
                    
                    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_xticks(range(len(agent_ids)))
                    ax.set_yticks(range(len(agent_ids)))
                    ax.set_xticklabels([f'A{aid}' for aid in agent_ids])
                    ax.set_yticklabels([f'A{aid}' for aid in agent_ids])
                    ax.set_title('Agent Final Preference Correlations')
                    plt.colorbar(im, ax=ax, label='Correlation')
                    
                    # Add correlation values as text
                    for i in range(len(agent_ids)):
                        for j in range(len(agent_ids)):
                            ax.text(j, i, f'{correlation_matrix[i,j]:.2f}', 
                                   ha='center', va='center', 
                                   color='white' if abs(correlation_matrix[i,j]) > 0.5 else 'black')
        
        # (d) Convergence Speed vs Order
        ax = axes[1, 1]
        replication_data = conv_analysis.get('replication_data', [])
        if replication_data:
            # Collect convergence order and speed data across replications
            orders = []
            speeds = []
            
            for rep_data in replication_data:
                conv_order = rep_data.get('convergence_order', [])
                conv_times = rep_data.get('convergence_times', [])
                
                for position, (agent_id, conv_time) in enumerate(zip(conv_order, sorted(conv_times))):
                    orders.append(position + 1)  # Position in convergence order
                    speeds.append(1.0 / (conv_time + 1))  # Convergence speed (inverse of time)
            
            if orders and speeds:
                ax.scatter(orders, speeds, alpha=0.6, s=50)
                
                # Add trend line
                if len(orders) > 1:
                    z = np.polyfit(orders, speeds, 1)
                    p = np.poly1d(z)
                    ax.plot(orders, p(orders), 'r--', alpha=0.8, label=f'Trend: slope={z[0]:.4f}')
                    ax.legend()
                
                ax.set_xlabel('Convergence Order')
                ax.set_ylabel('Convergence Speed (1/time)')
                ax.set_title('Convergence Speed vs Order')
                ax.grid(True, alpha=0.3)
                
                # Calculate correlation
                if len(orders) > 1:
                    try:
                        with np.errstate(invalid='ignore'):
                            corr_matrix = np.corrcoef(orders, speeds)
                            correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    except (ValueError, np.linalg.LinAlgError):
                        correlation = 0.0
                    
                    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'system_dynamics_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('system_dynamics_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def create_statistical_analysis_plots(self, output_dir: str, show_plots: bool = False) -> List[str]:
        """Create statistical analysis plots."""
        output_path = Path(output_dir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        conv_analysis = self.analyse_convergence_sequence()
        
        # Figure 4: Statistical Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Convergence Time Distributions
        ax = axes[0, 0]
        conv_times = conv_analysis.get('convergence_times', {}).get('all_times', [])
        if conv_times:
            ax.hist(conv_times, bins=30, alpha=0.7, edgecolor='black', density=True)
            ax.axvline(np.mean(conv_times), color='red', linestyle='--',
                      label=f'Mean: {np.mean(conv_times):.1f}')
            ax.axvline(np.median(conv_times), color='orange', linestyle='--',
                      label=f'Median: {np.median(conv_times):.1f}')
            ax.set_xlabel('Convergence Time (iterations)')
            ax.set_ylabel('Density')
            ax.set_title('Convergence Time Distribution')
            ax.legend()
        
        # (b) Sequential Pattern Validation
        ax = axes[0, 1]
        seq_indices = conv_analysis.get('sequential_indices', {}).get('all_indices', [])
        if seq_indices:
            # Box plot of sequential indices
            ax.boxplot([seq_indices], tick_labels=['Sequential Index'])
            ax.axhline(0.5, color='red', linestyle='--', label='Random (0.5)')
            ax.axhline(np.mean(seq_indices), color='blue', linestyle='-', 
                      label=f'Observed Mean: {np.mean(seq_indices):.3f}')
            ax.set_ylabel('Sequential Index')
            ax.set_title('Sequential Pattern Validation')
            ax.legend()
        
        # (c) Replication Consistency
        ax = axes[1, 0]
        replication_data = conv_analysis.get('replication_data', [])
        if len(replication_data) > 1:
            # Plot sequential index vs replication number
            rep_numbers = range(len(replication_data))
            seq_indices_by_rep = [rep.get('sequential_index', 0) for rep in replication_data]
            
            ax.scatter(rep_numbers, seq_indices_by_rep, alpha=0.7)
            ax.axhline(np.mean(seq_indices_by_rep), color='red', linestyle='--',
                      label=f'Mean: {np.mean(seq_indices_by_rep):.3f}')
            ax.set_xlabel('Replication Number')
            ax.set_ylabel('Sequential Index')
            ax.set_title('Sequential Index Consistency Across Replications')
            ax.legend()
        
        # (d) Comprehensive Hypothesis Test Results
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Create comprehensive statistical summary
        summary_data = []
        
        # Get key data
        statistical_tests = conv_analysis.get('statistical_tests', {})
        seq_indices = conv_analysis.get('sequential_indices', {}).get('all_indices', [])
        degeneracy_scores = conv_analysis.get('degeneracy_scores', {}).get('all_scores', [])
        convergence_times = conv_analysis.get('convergence_times', {}).get('all_times', [])
        
        # Section 1: Key Metrics
        summary_data.append(['HYPOTHESIS METRICS', '', '', ''])
        
        if seq_indices:
            mean_seq = np.mean(seq_indices)
            summary_data.append([
                'Sequential Index Mean', 
                f'{mean_seq:.3f}', 
                f'σ = {np.std(seq_indices):.3f}',
                'Sequential' if mean_seq > 0.5 else 'Random'
            ])
        
        if degeneracy_scores:
            deg_prop = np.mean([s > self.convergence_threshold_max_prob for s in degeneracy_scores])
            summary_data.append([
                'Degeneracy Proportion', 
                f'{deg_prop:.3f}', 
                f'({int(deg_prop * len(degeneracy_scores))}/{len(degeneracy_scores)})',
                'High' if deg_prop > 0.8 else 'Low'
            ])
        
        if convergence_times:
            summary_data.append([
                'Mean Convergence Time', 
                f'{np.mean(convergence_times):.1f}', 
                f'σ = {np.std(convergence_times):.1f}',
                'Fast' if np.mean(convergence_times) < 100 else 'Slow'
            ])
        
        # Section 2: Statistical Tests
        summary_data.append(['', '', '', ''])
        summary_data.append(['STATISTICAL TESTS', '', '', ''])
        
        for test_name, test_results in statistical_tests.items():
            if isinstance(test_results, dict) and 'p_value' in test_results:
                # Handle inf and nan values properly
                statistic_val = test_results.get('statistic', 'N/A')
                if isinstance(statistic_val, (int, float)):
                    if np.isinf(statistic_val):
                        stat_str = "∞" if statistic_val > 0 else "-∞"
                    elif np.isnan(statistic_val):
                        stat_str = "N/A"
                    else:
                        stat_str = f"{statistic_val:.3f}"
                else:
                    stat_str = str(statistic_val)
                
                # Get additional test-specific info
                extra_info = ""
                if test_name == 'degeneracy_proportion' and 'proportion' in test_results:
                    extra_info = f"({test_results['proportion']:.3f})"
                elif test_name == 'sequential_pattern' and 'more_sequential' in test_results:
                    extra_info = "✓" if test_results['more_sequential'] else "✗"
                
                summary_data.append([
                    test_name.replace('_', ' ').title(),
                    stat_str,
                    f"p = {test_results['p_value']:.4f}",
                    f"{'✓' if test_results['p_value'] < 0.05 else '✗'} {extra_info}"
                ])
        
        # Section 3: Overall Assessment
        support = conv_analysis.get('hypothesis_support', {})
        if support:
            summary_data.append(['', '', '', ''])
            summary_data.append(['HYPOTHESIS SUPPORT', '', '', ''])
            overall_support = support.get('overall_support', 'undetermined')
            summary_data.append([
                'Overall Assessment',
                overall_support.upper(),
                '',
                '✓' if overall_support in ['strong', 'moderate'] else '✗'
            ])
        
        if summary_data:
            table = ax.table(cellText=summary_data,
                           colLabels=['Metric/Test', 'Value/Statistic', 'Additional Info', 'Status'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Color-code the header rows
            for i, row in enumerate(summary_data):
                if row[0] in ['HYPOTHESIS METRICS', 'STATISTICAL TESTS', 'HYPOTHESIS SUPPORT']:
                    for j in range(4):
                        table[(i+1, j)].set_facecolor('#E8E8E8')
                        table[(i+1, j)].set_text_props(weight='bold')
        
        ax.set_title('Sequential Convergence Hypothesis Analysis')
        
        plt.tight_layout()
        
        if show_plots:
            plt.show()
        
        plot_file = output_path / 'statistical_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('statistical_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def evaluate_hypothesis_support(self, conv_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate level of support for the sequential convergence hypothesis using system analysis."""
        # Get statistical tests and convergence metrics
        statistical_tests = conv_analysis.get('statistical_tests', {})
        replication_data = conv_analysis.get('replication_data', [])
        
        # Calculate system convergence metrics
        convergence_metrics = calculate_system_convergence_metrics(replication_data)
        
        # Use system analysis function for hypothesis evaluation
        support = evaluate_system_hypothesis_support(
            conv_analysis, 
            statistical_tests, 
            convergence_metrics
        )
        
        return support


def run_sequential_convergence_study(
    num_replications: int = 30,
    num_iterations: int = 2000,
    output_dir: str = "results/sequential_convergence_study",
    show_plots: bool = False,
    convergence_threshold_entropy: float = None,
    convergence_threshold_max_prob: float = None
) -> SequentialConvergenceStudy:
    """
    Run complete sequential convergence study.
    
    Args:
        num_replications: Number of replications to run
        num_iterations: Number of iterations per simulation
        output_dir: Output directory for results
        show_plots: Whether to display plots interactively during creation
        convergence_threshold_entropy: Entropy threshold for convergence (default: 0.5 for short runs, 0.1 for long runs)
        convergence_threshold_max_prob: Max probability threshold for convergence (default: 0.6 for short runs, 0.9 for long runs)
        
    Returns:
        Completed SequentialConvergenceStudy instance
    """
    # Set adaptive thresholds based on run length
    if convergence_threshold_entropy is None:
        convergence_threshold_entropy = 0.1 if num_iterations >= 100 else 0.5
    
    if convergence_threshold_max_prob is None:
        convergence_threshold_max_prob = 0.9 if num_iterations >= 100 else 0.6
    
    # Create experiment
    study = SequentialConvergenceStudy(
        results_dir=output_dir,
        experiment_name="sequential_convergence_study",
        convergence_threshold_entropy=convergence_threshold_entropy,
        convergence_threshold_max_prob=convergence_threshold_max_prob
    )
    
    # Update configuration
    study.base_config.num_iterations = num_iterations
    
    print(f"Running sequential convergence study...")
    print(f"Replications: {num_replications}")
    print(f"Iterations per simulation: {num_iterations}")
    print(f"Agents: {study.base_config.num_agents}")
    print(f"Resources: {study.base_config.num_resources}")
    print(f"Show plots: {show_plots}")
    print(f"Convergence thresholds: entropy < {convergence_threshold_entropy}, max_prob > {convergence_threshold_max_prob}")
    
    # Run experiment using BaseExperiment interface
    full_results = study.run_experiment(num_episodes=num_replications)
    
    # Generate additional visualisations
    print("Creating visualisations...")
    # Use the actual results directory from the study, not the base output_dir
    actual_results_dir = study.get_results_dir()
    timeline_plots = study.create_convergence_timeline_plots(actual_results_dir, show_plots=show_plots)
    evolution_plots = study.create_probability_evolution_plots(actual_results_dir, show_plots=show_plots)
    system_plots = study.create_system_dynamics_plots(actual_results_dir, show_plots=show_plots)
    barycentric_plots = study.create_barycentric_trajectory_plots(actual_results_dir, show_plots=show_plots)
    stats_plots = study.create_statistical_analysis_plots(actual_results_dir, show_plots=show_plots)
    
    total_plots = len(timeline_plots) + len(evolution_plots) + len(system_plots) + len(barycentric_plots) + len(stats_plots)
    print(f"Generated {total_plots} plots")
    
    # Print summary
    analysis = full_results.get('analysis', {})
    support = analysis.get('hypothesis_support', {})
    overall_support = support.get('overall_support', 'undetermined')
    print(f"\nHypothesis Support: {overall_support.upper()}")
    
    if 'evidence_strength' in support:
        evidence = support['evidence_strength']
        
        # Handle sequential index
        seq_index = evidence.get('sequential_index', 'N/A')
        if isinstance(seq_index, (int, float)):
            print(f"Sequential Index: {seq_index:.3f}")
        else:
            print(f"Sequential Index: {seq_index}")
        
        # Handle degeneracy proportion
        deg_prop = evidence.get('degeneracy_proportion', 'N/A')
        if isinstance(deg_prop, (int, float)):
            print(f"Degeneracy Proportion: {deg_prop:.3f}")
        else:
            print(f"Degeneracy Proportion: {deg_prop}")
    
    print(f"\nSequential convergence study completed!")
    print(f"Results available in: {study.get_results_dir()}/")
    
    return study


if __name__ == "__main__":
    study = run_sequential_convergence_study(num_replications=100, num_iterations=1000, show_plots=False) 