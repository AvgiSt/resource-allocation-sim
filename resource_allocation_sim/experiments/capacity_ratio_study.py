"""Capacity ratio study experiment testing Hypothesis 4."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .parameter_sweep import ParameterSweepExperiment
from ..evaluation.agent_analysis import analyse_agent_convergence, plot_probability_distribution
from ..evaluation.system_analysis import analyse_system_performance, plot_cost_evolution
from ..evaluation.metrics import calculate_entropy, calculate_convergence_speed
from ..visualisation.plots import plot_parameter_sensitivity, plot_convergence_comparison
from ..visualisation.ternary import plot_ternary_distribution, plot_ternary_trajectory
from ..utils.config import Config


class CapacityRatioStudy(ParameterSweepExperiment):
    """
    Comprehensive study of capacity ratio effects on agent specialisation patterns.
    
    Tests Hypothesis 4: Asymmetric capacity configurations create predictable agent 
    specialisation patterns where high-capacity resources attract early sequential 
    convergers whilst low-capacity resources either remain underutilised or attract 
    late specialised convergers, resulting in hierarchical resource utilisation 
    that reflects the capacity hierarchy.
    
    This class provides a complete analysis pipeline including:
    - Data generation across capacity ratio configurations
    - Hierarchical specialisation analysis
    - Resource utilisation correlation analysis
    - Convergence timing analysis
    - Performance optimisation analysis
    - Comprehensive visualisations and statistical tests
    """
    
    def __init__(
        self,
        capacity_configurations: Optional[List[List[float]]] = None,
        **kwargs
    ):
        """
        Initialise capacity ratio study.
        
        Args:
            capacity_configurations: List of capacity ratio configurations to test
            **kwargs: Arguments passed to ParameterSweepExperiment
        """
        if capacity_configurations is None:
            capacity_configurations = [
                [0.33, 0.33, 0.33],  # Symmetric baseline
                [0.5, 0.3, 0.2],     # Moderate asymmetry 1
                [0.4, 0.4, 0.2],     # Moderate asymmetry 2
                [0.6, 0.3, 0.1],     # High asymmetry 1
                [0.7, 0.2, 0.1],     # High asymmetry 2
                [0.5, 0.25, 0.25],   # Single dominant
                [0.8, 0.15, 0.05],   # Extreme asymmetry
                [0.45, 0.45, 0.1],   # Two dominant
                [0.6, 0.25, 0.15],   # Graduated hierarchy
                [0.55, 0.35, 0.1]    # Strong binary
            ]
        
        # Convert configurations to parameter values for base class
        parameter_values = [f"{config[0]:.2f}_{config[1]:.2f}_{config[2]:.2f}" 
                          for config in capacity_configurations]
        
        super().__init__(
            parameter_name='capacity_ratio',
            parameter_values=parameter_values,
            **kwargs
        )
        
        # Study-specific configuration
        self.capacity_configurations = capacity_configurations
        self.capacity_labels = parameter_values
        self.analysis_results = {}
        self.detailed_data = {}
        
        # Set up base configuration
        self.base_config = self.setup_base_config()
    
    def setup_base_config(self) -> Config:
        """Set up base configuration for the capacity ratio study."""
        config = Config()
        config.num_resources = 3
        config.num_agents = 10
        config.num_iterations = 1000
        config.weight = 0.3  # Standard learning rate
        config.agent_initialisation_method = "uniform"
        return config
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate configurations for different capacity ratios."""
        configurations = []
        
        for capacity_config in self.capacity_configurations:
            config = {
                'num_resources': 3,
                'num_agents': 10,
                'relative_capacity': capacity_config,
                'num_iterations': 1000,
                'weight': 0.3,
                'agent_initialisation_method': 'uniform',
                'capacity_ratio': f"{capacity_config[0]:.2f}_{capacity_config[1]:.2f}_{capacity_config[2]:.2f}"
            }
            configurations.append(config)
        
        return configurations
    
    def analyse_results(self) -> Dict[str, Any]:
        """
        Analyse results for capacity-driven specialisation patterns.
        
        Returns:
            Dictionary containing comprehensive analysis of capacity effects
        """
        # Check if this is being called by BaseExperiment before conversion
        if not hasattr(self, 'converted_results'):
            # Return empty analysis - real analysis will be done after conversion
            return {}
        
        # Results are already converted in run_capacity_ratio_study function
        # Use self.converted_results directly
        
        # Perform comprehensive analysis
        analysis = {
            'hierarchical_patterns': self.analyse_hierarchical_specialisation(),
            'utilisation_correlation': self.analyse_capacity_utilisation_correlation(),
            'convergence_timing': self.analyse_convergence_timing_patterns(),
            'performance_analysis': self.analyse_performance_across_configurations(),
            'specialisation_metrics': self.analyse_agent_specialisation_patterns()
        }
        
        # Statistical tests and hypothesis evaluation
        analysis['statistical_tests'] = self.perform_capacity_statistical_tests(analysis)
        analysis['hypothesis_support'] = self.evaluate_capacity_hypothesis_support(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    def perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis after results conversion.
        
        Returns:
            Dictionary containing comprehensive analysis of capacity effects
        """
        if not hasattr(self, 'converted_results') or not self.converted_results:
            return {}
        
        # Perform comprehensive analysis
        analysis = {
            'hierarchical_patterns': self.analyse_hierarchical_specialisation(),
            'utilisation_correlation': self.analyse_capacity_utilisation_correlation(),
            'convergence_timing': self.analyse_convergence_timing_patterns(),
            'performance_analysis': self.analyse_performance_across_configurations(),
            'specialisation_metrics': self.analyse_agent_specialisation_patterns()
        }
        
        # Statistical tests and hypothesis evaluation
        analysis['statistical_tests'] = self.perform_capacity_statistical_tests(analysis)
        analysis['hypothesis_support'] = self.evaluate_capacity_hypothesis_support(analysis)
        
        self.analysis_results = analysis
        return analysis
    
    def analyse_hierarchical_specialisation(self) -> Dict[str, Any]:
        """Analyse whether agents specialise according to capacity hierarchy."""
        hierarchical_analysis = {
            'capacity_preference_correlation': {},
            'resource_popularity_ranking': {},
            'hierarchy_consistency': {}
        }
        
        for result in self.converted_results:
            capacity_config = result['config_params']['relative_capacity']
            capacity_label = result['config_params']['capacity_ratio']
            agent_results = result['simulation_results']['agent_results']
            
            if capacity_label not in hierarchical_analysis['capacity_preference_correlation']:
                hierarchical_analysis['capacity_preference_correlation'][capacity_label] = []
                hierarchical_analysis['resource_popularity_ranking'][capacity_label] = []
                hierarchical_analysis['hierarchy_consistency'][capacity_label] = []
            
            # Analyse final agent preferences
            final_preferences = []
            resource_utilisation = np.zeros(3)
            
            for agent_id, data in agent_results.items():
                final_probs = data['prob'][-1]
                preferred_resource = np.argmax(final_probs)
                resource_utilisation[preferred_resource] += 1
                final_preferences.append((agent_id, preferred_resource, np.max(final_probs)))
            
            # Calculate correlation between capacity and utilisation
            capacity_order = np.argsort(capacity_config)[::-1]  # High to low capacity
            utilisation_order = np.argsort(resource_utilisation)[::-1]  # High to low utilisation
            
            # Spearman rank correlation
            correlation = 1.0 - (6 * sum((capacity_order[i] - utilisation_order[i])**2 for i in range(3))) / (3 * (3**2 - 1))
            hierarchical_analysis['capacity_preference_correlation'][capacity_label].append(correlation)
            
            # Resource popularity ranking
            popularity_ranking = list(utilisation_order)
            hierarchical_analysis['resource_popularity_ranking'][capacity_label].append(popularity_ranking)
            
            # Hierarchy consistency (do high-capacity resources get more agents?)
            expected_hierarchy = capacity_order.tolist()
            actual_hierarchy = utilisation_order.tolist()
            consistency = sum(1 for i in range(3) if expected_hierarchy[i] == actual_hierarchy[i]) / 3.0
            hierarchical_analysis['hierarchy_consistency'][capacity_label].append(consistency)
        
        return hierarchical_analysis
    
    def analyse_capacity_utilisation_correlation(self) -> Dict[str, Any]:
        """Analyse correlation between capacity values and final utilisation."""
        correlation_analysis = {
            'pearson_correlations': {},
            'spearman_correlations': {},
            'utilisation_distributions': {},
            'asymmetry_effects': {}
        }
        
        for result in self.converted_results:
            capacity_config = result['config_params']['relative_capacity']
            capacity_label = result['config_params']['capacity_ratio']
            agent_results = result['simulation_results']['agent_results']
            env_state = result['simulation_results']['environment_state']
            
            if capacity_label not in correlation_analysis['pearson_correlations']:
                correlation_analysis['pearson_correlations'][capacity_label] = []
                correlation_analysis['spearman_correlations'][capacity_label] = []
                correlation_analysis['utilisation_distributions'][capacity_label] = []
                correlation_analysis['asymmetry_effects'][capacity_label] = []
            
            # Calculate final utilisation from consumption
            final_consumption = env_state['consumption_history'][-1]
            final_utilisation = np.array(final_consumption) / sum(final_consumption) if sum(final_consumption) > 0 else np.zeros(3)
            
            # Calculate correlations
            pearson_corr = np.corrcoef(capacity_config, final_utilisation)[0, 1] if len(set(capacity_config)) > 1 else 0
            
            # Spearman rank correlation (only if not constant arrays)
            from scipy.stats import spearmanr
            if len(set(capacity_config)) > 1 and len(set(final_utilisation)) > 1:
                spearman_corr, _ = spearmanr(capacity_config, final_utilisation)
            else:
                spearman_corr = 0.0  # No correlation possible with constant arrays
            
            correlation_analysis['pearson_correlations'][capacity_label].append(pearson_corr if not np.isnan(pearson_corr) else 0)
            correlation_analysis['spearman_correlations'][capacity_label].append(spearman_corr if not np.isnan(spearman_corr) else 0)
            correlation_analysis['utilisation_distributions'][capacity_label].append(final_utilisation.tolist())
            
            # Asymmetry effect (standard deviation of capacities vs utilisation asymmetry)
            capacity_asymmetry = np.std(capacity_config)
            utilisation_asymmetry = np.std(final_utilisation)
            correlation_analysis['asymmetry_effects'][capacity_label].append((capacity_asymmetry, utilisation_asymmetry))
        
        return correlation_analysis
    
    def analyse_convergence_timing_patterns(self) -> Dict[str, Any]:
        """Analyse whether high-capacity resources attract earlier convergers."""
        timing_analysis = {
            'convergence_by_capacity': {},
            'resource_convergence_order': {},
            'early_vs_late_preferences': {}
        }
        
        for result in self.converted_results:
            capacity_config = result['config_params']['relative_capacity']
            capacity_label = result['config_params']['capacity_ratio']
            agent_results = result['simulation_results']['agent_results']
            
            if capacity_label not in timing_analysis['convergence_by_capacity']:
                timing_analysis['convergence_by_capacity'][capacity_label] = []
                timing_analysis['resource_convergence_order'][capacity_label] = []
                timing_analysis['early_vs_late_preferences'][capacity_label] = []
            
            # Find convergence times and preferred resources
            agent_convergence_data = []
            for agent_id, data in agent_results.items():
                prob_history = np.array(data['prob'])
                entropies = [calculate_entropy(probs) for probs in prob_history]
                
                # Find convergence time (when entropy < 0.1)
                convergence_time = None
                for i, entropy in enumerate(entropies):
                    if entropy < 0.1:
                        convergence_time = i
                        break
                
                if convergence_time is not None:
                    preferred_resource = np.argmax(prob_history[convergence_time])
                    agent_convergence_data.append((agent_id, convergence_time, preferred_resource))
            
            if agent_convergence_data:
                # Sort by convergence time
                agent_convergence_data.sort(key=lambda x: x[1])
                
                # Analyse resource preferences by convergence order
                early_convergers = agent_convergence_data[:len(agent_convergence_data)//2]
                late_convergers = agent_convergence_data[len(agent_convergence_data)//2:]
                
                early_preferences = [x[2] for x in early_convergers]
                late_preferences = [x[2] for x in late_convergers]
                
                timing_analysis['convergence_by_capacity'][capacity_label].append(agent_convergence_data)
                timing_analysis['early_vs_late_preferences'][capacity_label].append({
                    'early': early_preferences,
                    'late': late_preferences
                })
        
        return timing_analysis
    
    def analyse_performance_across_configurations(self) -> Dict[str, Any]:
        """Analyse system performance across different capacity configurations."""
        performance_analysis = {
            'total_costs': {},
            'load_balance_quality': {},
            'system_efficiency': {},
            'convergence_rates': {}
        }
        
        for result in self.converted_results:
            capacity_label = result['config_params']['capacity_ratio']
            env_state = result['simulation_results']['environment_state']
            agent_results = result['simulation_results']['agent_results']
            
            if capacity_label not in performance_analysis['total_costs']:
                performance_analysis['total_costs'][capacity_label] = []
                performance_analysis['load_balance_quality'][capacity_label] = []
                performance_analysis['system_efficiency'][capacity_label] = []
                performance_analysis['convergence_rates'][capacity_label] = []
            
            # Calculate performance metrics
            cost_history = np.array(env_state['cost_history'])
            final_total_cost = np.sum(cost_history[-1])
            steady_state_cost = np.mean(np.sum(cost_history[-100:], axis=1))
            
            performance_analysis['total_costs'][capacity_label].append(steady_state_cost)
            
            # Load balance quality (lower std = better balance)
            consumption_history = np.array(env_state['consumption_history'])
            final_consumption = consumption_history[-1]
            load_balance = np.std(final_consumption) if len(final_consumption) > 0 else 0
            performance_analysis['load_balance_quality'][capacity_label].append(load_balance)
            
            # System efficiency
            total_consumption = np.sum(final_consumption)
            efficiency = steady_state_cost / total_consumption if total_consumption > 0 else float('inf')
            performance_analysis['system_efficiency'][capacity_label].append(efficiency)
            
            # Convergence rate (fraction of agents that converged)
            converged_agents = 0
            for agent_id, data in agent_results.items():
                prob_history = np.array(data['prob'])
                final_entropy = calculate_entropy(prob_history[-1])
                if final_entropy < 0.1:
                    converged_agents += 1
            
            convergence_rate = converged_agents / len(agent_results)
            performance_analysis['convergence_rates'][capacity_label].append(convergence_rate)
        
        return performance_analysis
    
    def analyse_agent_specialisation_patterns(self) -> Dict[str, Any]:
        """Analyse detailed agent specialisation patterns in ternary space."""
        specialisation_analysis = {
            'specialisation_indices': {},
            'ternary_positions': {},
            'vertex_attraction': {},
            'trajectory_directness': {}
        }
        
        for result in self.converted_results:
            capacity_label = result['config_params']['capacity_ratio']
            agent_results = result['simulation_results']['agent_results']
            
            if capacity_label not in specialisation_analysis['specialisation_indices']:
                specialisation_analysis['specialisation_indices'][capacity_label] = []
                specialisation_analysis['ternary_positions'][capacity_label] = []
                specialisation_analysis['vertex_attraction'][capacity_label] = []
                specialisation_analysis['trajectory_directness'][capacity_label] = []
            
            # Calculate specialisation metrics for each agent
            replication_specialisation = []
            replication_positions = []
            vertex_attractions = [0, 0, 0]  # Count agents near each vertex
            trajectory_directness_scores = []
            
            for agent_id, data in agent_results.items():
                prob_history = np.array(data['prob'])
                
                # Specialisation index (1 - entropy) of final distribution
                final_entropy = calculate_entropy(prob_history[-1])
                specialisation_index = 1.0 - (final_entropy / np.log(3))  # Normalised to [0,1]
                replication_specialisation.append(specialisation_index)
                
                # Final ternary position
                final_position = prob_history[-1].tolist()
                replication_positions.append(final_position)
                
                # Vertex attraction (which vertex is agent closest to?)
                vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                distances = [np.linalg.norm(np.array(final_position) - np.array(vertex)) for vertex in vertices]
                closest_vertex = np.argmin(distances)
                vertex_attractions[closest_vertex] += 1
                
                # Trajectory directness (distance from initial to final / path length)
                if len(prob_history) > 1:
                    initial_pos = prob_history[0]
                    final_pos = prob_history[-1]
                    direct_distance = np.linalg.norm(final_pos - initial_pos)
                    
                    path_length = sum(np.linalg.norm(prob_history[i+1] - prob_history[i]) 
                                    for i in range(len(prob_history)-1))
                    
                    directness = direct_distance / path_length if path_length > 0 else 0
                    trajectory_directness_scores.append(directness)
            
            specialisation_analysis['specialisation_indices'][capacity_label].append(replication_specialisation)
            specialisation_analysis['ternary_positions'][capacity_label].append(replication_positions)
            specialisation_analysis['vertex_attraction'][capacity_label].append(vertex_attractions)
            specialisation_analysis['trajectory_directness'][capacity_label].append(trajectory_directness_scores)
        
        return specialisation_analysis
    
    def perform_capacity_statistical_tests(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests for capacity ratio hypothesis."""
        from scipy import stats
        
        tests = {}
        
        # Test 1: Capacity-utilisation correlation significance
        hierarchical_data = analysis['hierarchical_patterns']['capacity_preference_correlation']
        all_correlations = []
        for config_correlations in hierarchical_data.values():
            all_correlations.extend(config_correlations)
        
        if all_correlations:
            # Test if correlations are significantly positive
            t_stat, p_value = stats.ttest_1samp(all_correlations, 0)
            tests['capacity_utilisation_correlation'] = {
                'statistic': t_stat,
                'p_value': p_value,
                'mean_correlation': np.mean(all_correlations),
                'significant_positive': p_value < 0.05 and np.mean(all_correlations) > 0
            }
        
        # Test 2: Asymmetry effect on performance
        performance_data = analysis['performance_analysis']['total_costs']
        asymmetry_scores = []
        costs = []
        
        for config_label, config_costs in performance_data.items():
            # Calculate asymmetry from label
            capacity_values = [float(x) for x in config_label.split('_')]
            asymmetry = np.std(capacity_values)
            
            for cost in config_costs:
                asymmetry_scores.append(asymmetry)
                costs.append(cost)
        
        if len(asymmetry_scores) > 1:
            correlation, p_value = stats.pearsonr(asymmetry_scores, costs)
            tests['asymmetry_performance_correlation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Test 3: Hierarchy consistency
        hierarchy_data = analysis['hierarchical_patterns']['hierarchy_consistency']
        all_consistency = []
        for config_consistency in hierarchy_data.values():
            all_consistency.extend(config_consistency)
        
        if all_consistency:
            # Test if consistency is better than random (0.33 for 3 resources)
            t_stat, p_value = stats.ttest_1samp(all_consistency, 1/3)
            tests['hierarchy_consistency'] = {
                'statistic': t_stat,
                'p_value': p_value,
                'mean_consistency': np.mean(all_consistency),
                'better_than_random': p_value < 0.05 and np.mean(all_consistency) > 1/3
            }
        
        return tests
    
    def evaluate_capacity_hypothesis_support(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate level of support for capacity ratio hypothesis."""
        statistical_tests = analysis['statistical_tests']
        
        # Collect evidence
        evidence = {}
        
        # Evidence 1: Capacity-utilisation correlation
        corr_test = statistical_tests.get('capacity_utilisation_correlation', {})
        if corr_test:
            evidence['capacity_correlation'] = {
                'value': corr_test.get('mean_correlation', 0),
                'significant': corr_test.get('significant_positive', False),
                'strength': 'strong' if corr_test.get('mean_correlation', 0) > 0.5 else 'moderate' if corr_test.get('mean_correlation', 0) > 0.3 else 'weak'
            }
        
        # Evidence 2: Hierarchy consistency
        hierarchy_test = statistical_tests.get('hierarchy_consistency', {})
        if hierarchy_test:
            evidence['hierarchy_consistency'] = {
                'value': hierarchy_test.get('mean_consistency', 0),
                'better_than_random': hierarchy_test.get('better_than_random', False),
                'strength': 'strong' if hierarchy_test.get('mean_consistency', 0) > 0.8 else 'moderate' if hierarchy_test.get('mean_consistency', 0) > 0.6 else 'weak'
            }
        
        # Evidence 3: Specialisation patterns
        specialisation_data = analysis['specialisation_metrics']['specialisation_indices']
        mean_specialisation = np.mean([np.mean(config_data) for config_data in specialisation_data.values()]) if specialisation_data else 0
        evidence['specialisation_strength'] = {
            'value': mean_specialisation,
            'strength': 'strong' if mean_specialisation > 0.8 else 'moderate' if mean_specialisation > 0.6 else 'weak'
        }
        
        # Overall assessment
        strong_evidence = sum(1 for e in evidence.values() if e.get('strength') == 'strong')
        moderate_evidence = sum(1 for e in evidence.values() if e.get('strength') == 'moderate')
        total_evidence = len(evidence)
        
        if strong_evidence >= 2 or (strong_evidence >= 1 and moderate_evidence >= 1):
            overall_support = 'strong'
        elif strong_evidence >= 1 or moderate_evidence >= 2:
            overall_support = 'moderate'
        elif moderate_evidence >= 1:
            overall_support = 'weak'
        else:
            overall_support = 'none'
        
        return {
            'overall_support': overall_support,
            'evidence_strength': evidence,
            'criteria_met': f"{strong_evidence + moderate_evidence}/{total_evidence}",
            'statistical_significance': any(e.get('significant', False) for e in [corr_test, hierarchy_test])
        }
    
    def create_comprehensive_plots(self, output_dir: str) -> List[str]:
        """Create comprehensive analysis plots for capacity ratio study."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            self.perform_comprehensive_analysis()
        
        # 1. Capacity-utilisation correlation plots
        plot_files.extend(self._create_capacity_correlation_plots(output_path))
        
        # 2. Hierarchical specialisation analysis
        plot_files.extend(self._create_hierarchy_analysis_plots(output_path))
        
        # 3. Performance across configurations
        plot_files.extend(self._create_performance_analysis_plots(output_path))
        
        # 4. Ternary specialisation plots
        plot_files.extend(self._create_ternary_specialisation_plots(output_path))
        
        # 5. Statistical analysis summary
        plot_files.extend(self._create_statistical_summary_plots(output_path))
        
        return plot_files
    
    def _create_capacity_correlation_plots(self, output_path: Path) -> List[str]:
        """Create capacity-utilisation correlation analysis plots."""
        plot_files = []
        
        # Figure 1: Capacity vs Utilisation Correlation
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Configuration comparison
        ax = axes[0, 0]
        correlation_data = self.analysis_results['utilisation_correlation']['pearson_correlations']
        
        config_labels = []
        mean_correlations = []
        correlation_errors = []
        
        for config_label, correlations in correlation_data.items():
            config_labels.append(config_label.replace('_', ', '))
            mean_correlations.append(np.mean(correlations))
            correlation_errors.append(np.std(correlations))
        
        bars = ax.bar(range(len(config_labels)), mean_correlations, yerr=correlation_errors, 
                     capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No correlation')
        ax.set_xlabel('Capacity Configuration')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Capacity-Utilisation Correlation by Configuration')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (b) Asymmetry effect
        ax = axes[0, 1]
        asymmetry_data = self.analysis_results['utilisation_correlation']['asymmetry_effects']
        
        all_cap_asymmetry = []
        all_util_asymmetry = []
        
        for config_data in asymmetry_data.values():
            for cap_asym, util_asym in config_data:
                all_cap_asymmetry.append(cap_asym)
                all_util_asymmetry.append(util_asym)
        
        if all_cap_asymmetry and all_util_asymmetry:
            ax.scatter(all_cap_asymmetry, all_util_asymmetry, alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(all_cap_asymmetry, all_util_asymmetry, 1)
            p = np.poly1d(z)
            ax.plot(all_cap_asymmetry, p(all_cap_asymmetry), 'r--', alpha=0.8, 
                   label=f'Trend: slope={z[0]:.3f}')
            ax.legend()
        
        ax.set_xlabel('Capacity Asymmetry (Std Dev)')
        ax.set_ylabel('Utilisation Asymmetry (Std Dev)')
        ax.set_title('Capacity vs Utilisation Asymmetry')
        ax.grid(True, alpha=0.3)
        
        # (c) Correlation strength distribution
        ax = axes[1, 0]
        all_correlations = []
        for correlations in correlation_data.values():
            all_correlations.extend(correlations)
        
        if all_correlations:
            ax.hist(all_correlations, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_correlations), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_correlations):.3f}')
            ax.axvline(0, color='orange', linestyle='--', alpha=0.7, label='No correlation')
            ax.set_xlabel('Correlation Coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Capacity-Utilisation Correlations')
            ax.legend()
        
        # (d) Spearman vs Pearson comparison
        ax = axes[1, 1]
        spearman_data = self.analysis_results['utilisation_correlation']['spearman_correlations']
        
        pearson_values = []
        spearman_values = []
        
        for config_label in correlation_data.keys():
            if config_label in spearman_data:
                pearson_values.extend(correlation_data[config_label])
                spearman_values.extend(spearman_data[config_label])
        
        if pearson_values and spearman_values:
            ax.scatter(pearson_values, spearman_values, alpha=0.6)
            
            # Add diagonal line
            min_val = min(min(pearson_values), min(spearman_values))
            max_val = max(max(pearson_values), max(spearman_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
            ax.legend()
        
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Pearson vs Spearman Correlations')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = output_path / 'capacity_correlation_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('capacity_correlation_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def _create_hierarchy_analysis_plots(self, output_path: Path) -> List[str]:
        """Create hierarchical specialisation analysis plots."""
        plot_files = []
        
        # Figure 2: Hierarchy Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Hierarchy consistency by configuration
        ax = axes[0, 0]
        hierarchy_data = self.analysis_results['hierarchical_patterns']['hierarchy_consistency']
        
        config_labels = []
        mean_consistency = []
        consistency_errors = []
        
        for config_label, consistency_values in hierarchy_data.items():
            config_labels.append(config_label.replace('_', ', '))
            mean_consistency.append(np.mean(consistency_values))
            consistency_errors.append(np.std(consistency_values))
        
        bars = ax.bar(range(len(config_labels)), mean_consistency, yerr=consistency_errors,
                     capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax.axhline(y=1/3, color='red', linestyle='--', alpha=0.7, label='Random (0.33)')
        ax.set_xlabel('Capacity Configuration')
        ax.set_ylabel('Hierarchy Consistency')
        ax.set_title('Hierarchy Consistency by Configuration')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # (b) Resource popularity ranking patterns
        ax = axes[0, 1]
        popularity_data = self.analysis_results['hierarchical_patterns']['resource_popularity_ranking']
        
        # Create resource preference heatmap
        config_resource_counts = {}
        for config_label, rankings in popularity_data.items():
            resource_position_counts = np.zeros((3, 3))  # [resource][position]
            
            for ranking in rankings:
                for position, resource in enumerate(ranking):
                    if 0 <= resource < 3:
                        resource_position_counts[resource][position] += 1
            
            config_resource_counts[config_label] = resource_position_counts
        
        # Show heatmap for most asymmetric configuration
        most_asymmetric_config = max(config_resource_counts.keys(), 
                                   key=lambda x: np.std([float(v) for v in x.split('_')]))
        
        if most_asymmetric_config in config_resource_counts:
            heatmap_data = config_resource_counts[most_asymmetric_config]
            normalised_data = heatmap_data / (heatmap_data.sum(axis=1, keepdims=True) + 1e-10)
            
            im = ax.imshow(normalised_data, cmap='Blues', aspect='auto')
            ax.set_xlabel('Popularity Position (0=Most Popular)')
            ax.set_ylabel('Resource ID')
            ax.set_title(f'Resource Popularity Pattern\n({most_asymmetric_config.replace("_", ", ")})')
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['1st', '2nd', '3rd'])
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Resource 1', 'Resource 2', 'Resource 3'])
            plt.colorbar(im, ax=ax, label='Proportion')
        
        # (c) Capacity vs utilisation scatter
        ax = axes[1, 0]
        
        # Collect capacity and utilisation data
        capacity_values = []
        utilisation_values = []
        config_colours = []
        colour_map = plt.cm.Set3(np.linspace(0, 1, len(self.capacity_configurations)))
        
        for i, (config_label, util_distributions) in enumerate(
            self.analysis_results['utilisation_correlation']['utilisation_distributions'].items()
        ):
            capacity_config = [float(x) for x in config_label.split('_')]
            
            for util_dist in util_distributions:
                for resource_idx in range(3):
                    capacity_values.append(capacity_config[resource_idx])
                    utilisation_values.append(util_dist[resource_idx])
                    config_colours.append(colour_map[i])
        
        if capacity_values and utilisation_values:
            scatter = ax.scatter(capacity_values, utilisation_values, c=config_colours, alpha=0.6, s=30)
            
            # Add trend line
            z = np.polyfit(capacity_values, utilisation_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(capacity_values), max(capacity_values), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2,
                   label=f'Trend: R² = {np.corrcoef(capacity_values, utilisation_values)[0,1]**2:.3f}')
            ax.legend()
        
        ax.set_xlabel('Capacity Value')
        ax.set_ylabel('Final Utilisation')
        ax.set_title('Capacity vs Utilisation (All Resources)')
        ax.grid(True, alpha=0.3)
        
        # (d) Configuration asymmetry analysis
        ax = axes[1, 1]
        
        asymmetry_scores = []
        performance_scores = []
        config_names = []
        
        for config_label in hierarchy_data.keys():
            capacity_config = [float(x) for x in config_label.split('_')]
            asymmetry = np.std(capacity_config)
            
            # Get mean hierarchy consistency as performance measure
            consistency = np.mean(hierarchy_data[config_label])
            
            asymmetry_scores.append(asymmetry)
            performance_scores.append(consistency)
            config_names.append(config_label.replace('_', ', '))
        
        if asymmetry_scores and performance_scores:
            scatter = ax.scatter(asymmetry_scores, performance_scores, s=100, alpha=0.7, 
                               c=range(len(asymmetry_scores)), cmap='viridis')
            
            # Add trend line
            if len(asymmetry_scores) > 1:
                z = np.polyfit(asymmetry_scores, performance_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(asymmetry_scores), max(asymmetry_scores), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2,
                       label=f'Trend: slope={z[0]:.3f}')
                ax.legend()
            
            # Add configuration labels
            for i, name in enumerate(config_names):
                ax.annotate(name, (asymmetry_scores[i], performance_scores[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Configuration Asymmetry (Std Dev)')
        ax.set_ylabel('Hierarchy Consistency')
        ax.set_title('Asymmetry vs Hierarchy Performance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = output_path / 'hierarchy_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('hierarchy_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def _create_performance_analysis_plots(self, output_path: Path) -> List[str]:
        """Create performance analysis plots."""
        plot_files = []
        
        # Figure 3: Performance Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Total costs by configuration
        ax = axes[0, 0]
        cost_data = self.analysis_results['performance_analysis']['total_costs']
        
        config_labels = []
        mean_costs = []
        cost_errors = []
        
        for config_label, costs in cost_data.items():
            config_labels.append(config_label.replace('_', ', '))
            mean_costs.append(np.mean(costs))
            cost_errors.append(np.std(costs))
        
        bars = ax.bar(range(len(config_labels)), mean_costs, yerr=cost_errors,
                     capsize=5, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax.set_xlabel('Capacity Configuration')
        ax.set_ylabel('Mean Total Cost')
        ax.set_title('System Cost by Configuration')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # (b) Load balance quality
        ax = axes[0, 1]
        balance_data = self.analysis_results['performance_analysis']['load_balance_quality']
        
        mean_balance = []
        balance_errors = []
        
        for config_label in config_labels:
            original_label = config_label.replace(', ', '_')
            if original_label in balance_data:
                balance_values = balance_data[original_label]
                mean_balance.append(np.mean(balance_values))
                balance_errors.append(np.std(balance_values))
            else:
                mean_balance.append(0)
                balance_errors.append(0)
        
        bars = ax.bar(range(len(config_labels)), mean_balance, yerr=balance_errors,
                     capsize=5, alpha=0.7, color='lightblue', edgecolor='darkblue')
        ax.set_xlabel('Capacity Configuration')
        ax.set_ylabel('Load Balance (Std Dev)')
        ax.set_title('Load Balance Quality by Configuration')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # (c) Convergence rates
        ax = axes[1, 0]
        convergence_data = self.analysis_results['performance_analysis']['convergence_rates']
        
        mean_convergence = []
        convergence_errors = []
        
        for config_label in config_labels:
            original_label = config_label.replace(', ', '_')
            if original_label in convergence_data:
                conv_values = convergence_data[original_label]
                mean_convergence.append(np.mean(conv_values))
                convergence_errors.append(np.std(conv_values))
            else:
                mean_convergence.append(0)
                convergence_errors.append(0)
        
        bars = ax.bar(range(len(config_labels)), mean_convergence, yerr=convergence_errors,
                     capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax.set_xlabel('Capacity Configuration')
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Agent Convergence Rate by Configuration')
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # (d) Performance trade-offs
        ax = axes[1, 1]
        
        # Cost vs convergence rate trade-off
        if mean_costs and mean_convergence:
            scatter = ax.scatter(mean_costs, mean_convergence, s=100, alpha=0.7,
                               c=range(len(mean_costs)), cmap='plasma')
            
            # Add configuration labels
            for i, label in enumerate(config_labels):
                ax.annotate(label, (mean_costs[i], mean_convergence[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add trend line if enough points
            if len(mean_costs) > 2:
                z = np.polyfit(mean_costs, mean_convergence, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(mean_costs), max(mean_costs), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2,
                       label=f'Trend: slope={z[0]:.3f}')
                ax.legend()
        
        ax.set_xlabel('Mean Total Cost')
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Cost vs Convergence Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = output_path / 'performance_analysis.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('performance_analysis.png')
        plt.close(fig)
        
        return plot_files
    
    def _create_ternary_specialisation_plots(self, output_path: Path) -> List[str]:
        """Create ternary plots showing agent specialisation patterns."""
        plot_files = []
        
        if not hasattr(self, 'converted_results'):
            return plot_files
        
        # Select representative configurations for ternary analysis
        symmetric_config = '0.33_0.33_0.33'
        asymmetric_configs = ['0.50_0.30_0.20', '0.70_0.20_0.10', '0.80_0.15_0.05']
        
        representative_configs = [symmetric_config] + asymmetric_configs
        available_configs = [config for config in representative_configs 
                           if any(result['config_params']['capacity_ratio'] == config 
                                for result in self.converted_results)]
        
        # Figure 4: Ternary Specialisation Comparison
        if available_configs:
            n_configs = len(available_configs)
            fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 6))
            if n_configs == 1:
                axes = [axes]
            
            for i, config_label in enumerate(available_configs):
                # Get agent results for this configuration (first replication)
                config_results = [result for result in self.converted_results 
                                if result['config_params']['capacity_ratio'] == config_label]
                
                if config_results:
                    agent_results = config_results[0]['simulation_results']['agent_results']
                    capacity_config = config_results[0]['config_params']['relative_capacity']
                    
                    # Create ternary plot
                    try:
                        # Use ternary plotting function
                        temp_fig = plot_ternary_distribution(
                            agent_results,
                            save_path=None,
                            title=f"Capacity: {config_label.replace('_', ', ')}"
                        )
                        plt.close(temp_fig)
                        
                        # Manual ternary plot for subplot
                        ax = axes[i]
                        
                        # Draw triangle
                        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
                        triangle_closed = np.vstack([triangle, triangle[0]])
                        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'k-', linewidth=2)
                        
                        # Plot agent final positions
                        final_positions = []
                        for agent_id, data in agent_results.items():
                            if data['prob']:
                                final_probs = data['prob'][-1]
                                final_positions.append(final_probs)
                        
                        if final_positions:
                            # Convert to ternary coordinates
                            projected_positions = np.dot(final_positions, triangle)
                            
                            # Colour by capacity preference
                            colours = []
                            for pos in final_positions:
                                preferred_resource = np.argmax(pos)
                                capacity_value = capacity_config[preferred_resource]
                                colours.append(capacity_value)
                            
                            scatter = ax.scatter(projected_positions[:, 0], projected_positions[:, 1], 
                                               c=colours, s=100, cmap='viridis', alpha=0.8, 
                                               edgecolors='black', linewidth=1)
                            
                            # Add capacity values as text at vertices
                            vertex_labels = [f'R1: {capacity_config[0]:.2f}', 
                                           f'R2: {capacity_config[1]:.2f}', 
                                           f'R3: {capacity_config[2]:.2f}']
                            
                            ax.text(triangle[2, 0], triangle[2, 1] + 0.05, vertex_labels[0], 
                                   ha='center', va='bottom', fontweight='bold')
                            ax.text(triangle[0, 0] - 0.05, triangle[0, 1] - 0.05, vertex_labels[1], 
                                   ha='center', va='top', fontweight='bold')
                            ax.text(triangle[1, 0] + 0.05, triangle[1, 1] - 0.05, vertex_labels[2], 
                                   ha='center', va='top', fontweight='bold')
                        
                        ax.set_title(f'Capacity: [{config_label.replace("_", ", ")}]')
                        ax.set_aspect('equal')
                        ax.axis('off')
                        
                    except Exception as e:
                        print(f"Could not create ternary plot for {config_label}: {e}")
                        ax.text(0.5, 0.5, f'Configuration:\n{config_label.replace("_", ", ")}\nPlot Unavailable', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
            
            plt.suptitle('Agent Specialisation Patterns by Capacity Configuration', fontsize=16)
            plt.tight_layout()
            
            plot_file = output_path / 'ternary_specialisation_comparison.png'
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append('ternary_specialisation_comparison.png')
            plt.close(fig)
        
        return plot_files
    
    def _create_statistical_summary_plots(self, output_path: Path) -> List[str]:
        """Create statistical analysis summary plots."""
        plot_files = []
        
        # Figure 5: Statistical Summary
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (a) Hypothesis support summary
        ax = axes[0, 0]
        ax.axis('tight')
        ax.axis('off')
        
        # Create hypothesis summary table
        summary_data = []
        
        if 'hypothesis_support' in self.analysis_results:
            support = self.analysis_results['hypothesis_support']
            evidence = support.get('evidence_strength', {})
            
            summary_data.append(['HYPOTHESIS 4 METRICS', '', '', ''])
            
            # Capacity correlation
            if 'capacity_correlation' in evidence:
                corr_info = evidence['capacity_correlation']
                summary_data.append([
                    'Capacity-Utilisation Correlation',
                    f"{corr_info['value']:.3f}",
                    corr_info['strength'].title(),
                    '✓' if corr_info.get('significant', False) else '✗'
                ])
            
            # Hierarchy consistency
            if 'hierarchy_consistency' in evidence:
                hier_info = evidence['hierarchy_consistency']
                summary_data.append([
                    'Hierarchy Consistency',
                    f"{hier_info['value']:.3f}",
                    hier_info['strength'].title(),
                    '✓' if hier_info.get('better_than_random', False) else '✗'
                ])
            
            # Specialisation strength
            if 'specialisation_strength' in evidence:
                spec_info = evidence['specialisation_strength']
                summary_data.append([
                    'Specialisation Index',
                    f"{spec_info['value']:.3f}",
                    spec_info['strength'].title(),
                    '✓' if spec_info['value'] > 0.6 else '✗'
                ])
            
            summary_data.append(['', '', '', ''])
            summary_data.append(['OVERALL ASSESSMENT', '', '', ''])
            summary_data.append([
                'Hypothesis Support',
                support.get('overall_support', 'unknown').upper(),
                support.get('criteria_met', 'N/A'),
                '✓' if support.get('overall_support') in ['strong', 'moderate'] else '✗'
            ])
        
        if summary_data:
            table = ax.table(cellText=summary_data,
                           colLabels=['Metric', 'Value', 'Strength', 'Status'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Format header rows
            for i, row in enumerate(summary_data):
                if row[0] in ['HYPOTHESIS 4 METRICS', 'OVERALL ASSESSMENT']:
                    for j in range(4):
                        table[(i+1, j)].set_facecolor('#E8E8E8')
                        table[(i+1, j)].set_text_props(weight='bold')
        
        ax.set_title('Capacity Ratio Hypothesis Analysis')
        
        # (b) Statistical test results
        ax = axes[0, 1]
        
        if 'statistical_tests' in self.analysis_results:
            tests = self.analysis_results['statistical_tests']
            
            test_names = []
            p_values = []
            test_statistics = []
            
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    test_names.append(test_name.replace('_', ' ').title())
                    p_values.append(test_result['p_value'])
                    test_statistics.append(test_result.get('statistic', 0))
            
            if test_names:
                # Create bar plot of p-values
                bars = ax.bar(range(len(test_names)), p_values, alpha=0.7, 
                             color=['green' if p < 0.05 else 'red' for p in p_values])
                ax.axhline(y=0.05, color='blue', linestyle='--', alpha=0.7, 
                          label='Significance threshold (0.05)')
                ax.set_xlabel('Statistical Test')
                ax.set_ylabel('P-value')
                ax.set_title('Statistical Test Results')
                ax.set_xticks(range(len(test_names)))
                ax.set_xticklabels(test_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
        
        # (c) Configuration performance ranking
        ax = axes[1, 0]
        
        # Rank configurations by multiple criteria
        if hasattr(self, 'analysis_results'):
            config_rankings = {}
            
            # Get hierarchy consistency scores
            hierarchy_data = self.analysis_results['hierarchical_patterns']['hierarchy_consistency']
            for config_label, consistency_values in hierarchy_data.items():
                config_rankings[config_label] = {
                    'hierarchy_score': np.mean(consistency_values),
                    'correlation_score': 0,
                    'performance_score': 0
                }
            
            # Add correlation scores
            correlation_data = self.analysis_results['hierarchical_patterns']['capacity_preference_correlation']
            for config_label, correlation_values in correlation_data.items():
                if config_label in config_rankings:
                    config_rankings[config_label]['correlation_score'] = np.mean(correlation_values)
            
            # Add performance scores (inverse of cost)
            cost_data = self.analysis_results['performance_analysis']['total_costs']
            max_cost = max(np.mean(costs) for costs in cost_data.values()) if cost_data else 1
            for config_label, costs in cost_data.items():
                if config_label in config_rankings:
                    config_rankings[config_label]['performance_score'] = 1 - (np.mean(costs) / max_cost)
            
            # Create ranking visualization
            if config_rankings:
                config_names = list(config_rankings.keys())
                hierarchy_scores = [config_rankings[c]['hierarchy_score'] for c in config_names]
                correlation_scores = [config_rankings[c]['correlation_score'] for c in config_names]
                performance_scores = [config_rankings[c]['performance_score'] for c in config_names]
                
                x = np.arange(len(config_names))
                width = 0.25
                
                ax.bar(x - width, hierarchy_scores, width, label='Hierarchy Consistency', alpha=0.7)
                ax.bar(x, correlation_scores, width, label='Capacity Correlation', alpha=0.7)
                ax.bar(x + width, performance_scores, width, label='Performance Score', alpha=0.7)
                
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Score')
                ax.set_title('Configuration Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels([c.replace('_', ',') for c in config_names], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # (d) Capacity asymmetry effects
        ax = axes[1, 1]
        
        # Plot relationship between capacity asymmetry and various outcomes
        asymmetry_scores = []
        hierarchy_scores = []
        config_labels = []
        
        for config_label in hierarchy_data.keys():
            capacity_config = [float(x) for x in config_label.split('_')]
            asymmetry = np.std(capacity_config)
            hierarchy_score = np.mean(hierarchy_data[config_label])
            
            asymmetry_scores.append(asymmetry)
            hierarchy_scores.append(hierarchy_score)
            config_labels.append(config_label.replace('_', ', '))
        
        if asymmetry_scores and hierarchy_scores:
            scatter = ax.scatter(asymmetry_scores, hierarchy_scores, s=100, alpha=0.7,
                               c=range(len(asymmetry_scores)), cmap='coolwarm')
            
            # Add trend line
            if len(asymmetry_scores) > 2:
                z = np.polyfit(asymmetry_scores, hierarchy_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(asymmetry_scores), max(asymmetry_scores), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2,
                       label=f'R² = {np.corrcoef(asymmetry_scores, hierarchy_scores)[0,1]**2:.3f}')
                ax.legend()
            
            # Label points
            for i, label in enumerate(config_labels):
                ax.annotate(label, (asymmetry_scores[i], hierarchy_scores[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Capacity Asymmetry (Std Dev)')
        ax.set_ylabel('Hierarchy Consistency')
        ax.set_title('Asymmetry Effect on Hierarchical Organisation')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = output_path / 'statistical_summary.png'
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append('statistical_summary.png')
        plt.close(fig)
        
        return plot_files


def run_capacity_ratio_study(
    num_replications: int = 100,
    output_dir: str = "results/capacity_ratio_study",
    show_plots: bool = False
) -> CapacityRatioStudy:
    """
    Run complete capacity ratio study with comprehensive analysis.
    
    Args:
        num_replications: Number of replications per capacity configuration
        output_dir: Output directory for results
        show_plots: Whether to display plots interactively
        
    Returns:
        Completed CapacityRatioStudy instance with full analysis
    """
    # Create experiment
    study = CapacityRatioStudy(
        results_dir=output_dir,
        experiment_name="capacity_ratio_study"
    )
    
    print(f"Running capacity ratio study...")
    print(f"Capacity configurations: {len(study.capacity_configurations)}")
    for i, config in enumerate(study.capacity_configurations):
        print(f"  {i+1}. [{config[0]:.2f}, {config[1]:.2f}, {config[2]:.2f}]")
    print(f"Replications per configuration: {num_replications}")
    
    # Run experiment using BaseExperiment interface
    full_results = study.run_experiment(num_episodes=num_replications)
    
    # Convert results to expected format for analysis
    study.results = []
    study.converted_results = []
    for config_result in full_results['results']:
        config_params = config_result['config_params']
        for episode_result in config_result['episode_results']:
            converted_result = {
                'config_params': config_params,
                'simulation_results': episode_result,
                'replication_id': episode_result['episode']
            }
            study.results.append(converted_result)
            study.converted_results.append(converted_result)
    
    # Generate comprehensive analysis
    print("Generating detailed analysis...")
    analysis = study.perform_comprehensive_analysis()
    
    # Create comprehensive plots
    print("Creating visualisations...")
    actual_results_dir = study.get_results_dir()
    plot_files = study.create_comprehensive_plots(f"{actual_results_dir}/plots")
    print(f"Generated {len(plot_files)} plots")
    
    # Save analysis results
    study.save_analysis_results(actual_results_dir)
    
    # Print summary
    print(f"\nCapacity ratio study completed!")
    print(f"Results available in: {actual_results_dir}/")
    
    if 'hypothesis_support' in analysis:
        support = analysis['hypothesis_support']
        print(f"\nKey Findings:")
        print(f"Hypothesis Support: {support.get('overall_support', 'unknown').upper()}")
        
        evidence = support.get('evidence_strength', {})
        if 'capacity_correlation' in evidence:
            corr_strength = evidence['capacity_correlation']
            print(f"Capacity-Utilisation Correlation: {corr_strength['value']:.3f} ({corr_strength['strength']})")
        
        if 'hierarchy_consistency' in evidence:
            hier_strength = evidence['hierarchy_consistency']
            print(f"Hierarchy Consistency: {hier_strength['value']:.3f} ({hier_strength['strength']})")
        
        if 'specialisation_strength' in evidence:
            spec_strength = evidence['specialisation_strength']
            print(f"Specialisation Index: {spec_strength['value']:.3f} ({spec_strength['strength']})")
        
        print(f"Statistical Significance: {'Yes' if support.get('statistical_significance', False) else 'No'}")
    
    return study

def save_analysis_results(self, output_dir: str) -> None:
    """Save analysis results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if hasattr(self, 'analysis_results') and self.analysis_results:
        # Save analysis results as JSON
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Generate and save detailed report
        report = self.generate_capacity_report()
        with open(output_path / 'capacity_ratio_hypothesis_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Analysis results saved to: {output_path}")

def generate_capacity_report(self) -> str:
    """Generate comprehensive capacity ratio hypothesis report."""
    if not hasattr(self, 'analysis_results') or not self.analysis_results:
        return "No analysis results available"
    
    analysis = self.analysis_results
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("CAPACITY RATIO STUDY - HYPOTHESIS 4 ANALYSIS REPORT")
    report_lines.append("=" * 80)
    
    # Hypothesis statement
    report_lines.append("\nHYPOTHESIS 4:")
    report_lines.append("Asymmetric capacity configurations create predictable agent specialisation")
    report_lines.append("patterns where high-capacity resources attract early sequential convergers")
    report_lines.append("whilst low-capacity resources either remain underutilised or attract late")
    report_lines.append("specialised convergers, resulting in hierarchical resource utilisation")
    report_lines.append("that reflects the capacity hierarchy.")
    
    # Executive summary
    report_lines.append("\nEXECUTIVE SUMMARY:")
    report_lines.append("-" * 40)
    
    if 'hypothesis_support' in analysis:
        support = analysis['hypothesis_support']
        evidence = support.get('evidence_strength', {})
        
        report_lines.append(f"Overall Support: {support.get('overall_support', 'unknown').upper()}")
        report_lines.append(f"Criteria Met: {support.get('criteria_met', 'N/A')}")
        report_lines.append(f"Statistical Significance: {'Yes' if support.get('statistical_significance', False) else 'No'}")
        
        if 'capacity_correlation' in evidence:
            corr_info = evidence['capacity_correlation']
            report_lines.append(f"Capacity-Utilisation Correlation: {corr_info['value']:.3f} ({corr_info['strength']})")
        
        if 'hierarchy_consistency' in evidence:
            hier_info = evidence['hierarchy_consistency']
            report_lines.append(f"Hierarchy Consistency: {hier_info['value']:.3f} ({hier_info['strength']})")
    
    # Detailed findings
    report_lines.append("\nDETAILED FINDINGS:")
    report_lines.append("-" * 40)
    
    # 1. Hierarchical specialisation patterns
    if 'hierarchical_patterns' in analysis:
        hier_data = analysis['hierarchical_patterns']
        report_lines.append("\n1. HIERARCHICAL SPECIALISATION PATTERNS:")
        
        # Capacity-preference correlations
        if 'capacity_preference_correlation' in hier_data:
            corr_data = hier_data['capacity_preference_correlation']
            all_correlations = []
            for correlations in corr_data.values():
                all_correlations.extend(correlations)
            
            if all_correlations:
                mean_corr = np.mean(all_correlations)
                std_corr = np.std(all_correlations)
                report_lines.append(f"   Mean Capacity-Utilisation Correlation: {mean_corr:.3f} ± {std_corr:.3f}")
                
                # Configuration breakdown
                report_lines.append("   Configuration Analysis:")
                for config_label, correlations in corr_data.items():
                    config_mean = np.mean(correlations)
                    report_lines.append(f"     [{config_label.replace('_', ', ')}]: {config_mean:.3f}")
        
        # Hierarchy consistency
        if 'hierarchy_consistency' in hier_data:
            consistency_data = hier_data['hierarchy_consistency']
            all_consistency = []
            for consistency_values in consistency_data.values():
                all_consistency.extend(consistency_values)
            
            if all_consistency:
                mean_consistency = np.mean(all_consistency)
                report_lines.append(f"   Mean Hierarchy Consistency: {mean_consistency:.3f}")
                report_lines.append(f"   Better than Random (0.33): {'Yes' if mean_consistency > 1/3 else 'No'}")
    
    # 2. Performance analysis
    if 'performance_analysis' in analysis:
        perf_data = analysis['performance_analysis']
        report_lines.append("\n2. PERFORMANCE ANALYSIS:")
        
        # Cost analysis
        if 'total_costs' in perf_data:
            cost_data = perf_data['total_costs']
            best_config = min(cost_data.keys(), key=lambda k: np.mean(cost_data[k]))
            worst_config = max(cost_data.keys(), key=lambda k: np.mean(cost_data[k]))
            
            best_cost = np.mean(cost_data[best_config])
            worst_cost = np.mean(cost_data[worst_config])
            
            report_lines.append(f"   Best Configuration: [{best_config.replace('_', ', ')}] - Cost: {best_cost:.3f}")
            report_lines.append(f"   Worst Configuration: [{worst_config.replace('_', ', ')}] - Cost: {worst_cost:.3f}")
            report_lines.append(f"   Performance Range: {((worst_cost - best_cost) / best_cost * 100):.1f}%")
        
        # Convergence rates
        if 'convergence_rates' in perf_data:
            conv_data = perf_data['convergence_rates']
            all_rates = []
            for rates in conv_data.values():
                all_rates.extend(rates)
            
            if all_rates:
                mean_rate = np.mean(all_rates)
                report_lines.append(f"   Mean Convergence Rate: {mean_rate:.3f}")
    
    # 3. Statistical test results
    if 'statistical_tests' in analysis:
        tests = analysis['statistical_tests']
        report_lines.append("\n3. STATISTICAL TEST RESULTS:")
        
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict):
                test_display_name = test_name.replace('_', ' ').title()
                report_lines.append(f"   {test_display_name}:")
                
                if 'p_value' in test_result:
                    p_val = test_result['p_value']
                    significance = "Significant" if p_val < 0.05 else "Not Significant"
                    report_lines.append(f"     P-value: {p_val:.4f} ({significance})")
                
                if 'statistic' in test_result:
                    stat = test_result['statistic']
                    if not np.isnan(stat) and not np.isinf(stat):
                        report_lines.append(f"     Test Statistic: {stat:.3f}")
    
    # Conclusions
    report_lines.append("\nCONCLUSIONS:")
    report_lines.append("-" * 40)
    
    if 'hypothesis_support' in analysis:
        support_level = analysis['hypothesis_support'].get('overall_support', 'unknown')
        
        if support_level == 'strong':
            report_lines.append("STRONG SUPPORT for Capacity-Driven Specialisation Hypothesis")
            report_lines.append("  - Clear correlation between capacity and utilisation patterns")
            report_lines.append("  - Consistent hierarchical organisation across configurations")
            report_lines.append("  - Statistical significance achieved")
        
        elif support_level == 'moderate':
            report_lines.append("MODERATE SUPPORT for Capacity-Driven Specialisation Hypothesis")
            report_lines.append("  - Some evidence of capacity effects")
            report_lines.append("  - Hierarchical patterns partially observed")
            report_lines.append("  - Mixed statistical significance")
        
        elif support_level == 'weak':
            report_lines.append("WEAK SUPPORT for Capacity-Driven Specialisation Hypothesis")
            report_lines.append("  - Limited evidence of capacity effects")
            report_lines.append("  - Inconsistent hierarchical patterns")
            report_lines.append("  - Low statistical significance")
        
        else:
            report_lines.append("NO SUPPORT for Capacity-Driven Specialisation Hypothesis")
            report_lines.append("  - No clear capacity-utilisation relationship")
            report_lines.append("  - Random or inconsistent patterns")
            report_lines.append("  - No statistical significance")
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)

# Add methods to CapacityRatioStudy class
CapacityRatioStudy.save_analysis_results = save_analysis_results
CapacityRatioStudy.generate_capacity_report = generate_capacity_report


if __name__ == "__main__":
    # Run with standard parameters
    study = run_capacity_ratio_study(num_replications=100) 
