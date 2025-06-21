"""Capacity analysis experiment for studying resource capacity effects."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from itertools import product

from .base_experiment import BaseExperiment
from ..evaluation.metrics import calculate_entropy, calculate_gini_coefficient


class CapacityAnalysisExperiment(BaseExperiment):
    """
    Experiment for analyzing effects of different capacity configurations.
    
    Tests various combinations of resource capacities and their effects
    on system performance.
    """
    
    def __init__(
        self,
        capacity_ranges: Dict[str, List[float]],
        symmetric_configs: bool = True,
        **kwargs
    ):
        """
        Initialise capacity analysis experiment.
        
        Args:
            capacity_ranges: Dictionary mapping resource names to capacity ranges
            symmetric_configs: Whether to include symmetric configurations
            **kwargs: Arguments passed to BaseExperiment
        """
        super().__init__(**kwargs)
        self.capacity_ranges = capacity_ranges
        self.symmetric_configs = symmetric_configs
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate capacity configurations."""
        configurations = []
        
        # Get number of resources
        num_resources = self.base_config.num_resources
        
        if num_resources == 3 and len(self.capacity_ranges) == 1:
            # Special case for 3 resources with single range
            capacity_values = list(self.capacity_ranges.values())[0]
            
            # Generate all combinations
            for c1, c2, c3 in product(capacity_values, repeat=3):
                configurations.append({'capacity': [c1, c2, c3]})
            
            # Add symmetric configurations if requested
            if self.symmetric_configs:
                for cap in capacity_values:
                    configurations.append({'capacity': [cap, cap, cap]})
                    
        else:
            # General case: direct capacity specification
            if 'capacity' in self.capacity_ranges:
                for capacity_config in self.capacity_ranges['capacity']:
                    configurations.append({'capacity': capacity_config})
            else:
                # Generate from individual resource ranges
                resource_names = list(self.capacity_ranges.keys())
                resource_ranges = [self.capacity_ranges[name] for name in resource_names]
                
                for combination in product(*resource_ranges):
                    capacity_list = list(combination)
                    configurations.append({'capacity': capacity_list})
        
        return configurations
    
    def analyse_results(self) -> Dict[str, Any]:
        """Analyse capacity analysis results."""
        analysis = {
            'capacity_effects': {},
            'optimal_capacities': {},
            'balance_analysis': {},
            'capacity_clusters': {}
        }
        
        # Extract capacity configurations and their results
        capacity_results = []
        for result in self.results:
            capacity = result['config_params']['capacity']
            summary_metrics = result['summary_metrics']
            
            capacity_results.append({
                'capacity': capacity,
                'total_capacity': sum(capacity),
                'capacity_balance': self._calculate_balance_metric(capacity),
                'entropy_mean': summary_metrics.get('entropy_mean', 0),
                'total_cost_mean': summary_metrics.get('total_cost_mean', 0),
                'gini_coefficient_mean': summary_metrics.get('gini_coefficient_mean', 0)
            })
        
        # Analyse capacity effects
        analysis['capacity_effects'] = self._analyse_capacity_effects(capacity_results)
        
        # Find optimal capacities
        analysis['optimal_capacities'] = self._find_optimal_capacities(capacity_results)
        
        # Balance analysis
        analysis['balance_analysis'] = self._analyse_balance_effects(capacity_results)
        
        # Capacity clustering
        analysis['capacity_clusters'] = self._cluster_capacities(capacity_results)
        
        return analysis
    
    def _calculate_balance_metric(self, capacity: List[float]) -> float:
        """Calculate capacity balance metric (lower = more balanced)."""
        if len(capacity) <= 1:
            return 0.0
        
        mean_cap = np.mean(capacity)
        if mean_cap == 0:
            return 0.0
        
        # Coefficient of variation
        return np.std(capacity) / mean_cap
    
    def _analyse_capacity_effects(self, capacity_results: List[Dict]) -> Dict[str, Any]:
        """Analyse effects of total capacity and individual resource capacities."""
        effects = {}
        
        # Total capacity effects
        total_caps = [r['total_capacity'] for r in capacity_results]
        costs = [r['total_cost_mean'] for r in capacity_results]
        entropies = [r['entropy_mean'] for r in capacity_results]
        
        if len(set(total_caps)) > 1:  # Only if there's variation
            effects['total_capacity_vs_cost'] = {
                'correlation': self._safe_correlation(total_caps, costs),
                'trend': self._determine_trend(total_caps, costs)
            }
            
            effects['total_capacity_vs_entropy'] = {
                'correlation': self._safe_correlation(total_caps, entropies),
                'trend': self._determine_trend(total_caps, entropies)
            }
        
        # Individual resource effects
        if capacity_results:
            num_resources = len(capacity_results[0]['capacity'])
            
            for i in range(num_resources):
                resource_caps = [r['capacity'][i] for r in capacity_results]
                
                effects[f'resource_{i+1}_vs_cost'] = {
                    'correlation': self._safe_correlation(resource_caps, costs),
                    'trend': self._determine_trend(resource_caps, costs)
                }
        
        return effects
    
    def _find_optimal_capacities(self, capacity_results: List[Dict]) -> Dict[str, Any]:
        """Find optimal capacity configurations for different metrics."""
        optimal = {}
        
        # Find configurations that minimize cost
        min_cost_result = min(capacity_results, key=lambda r: r['total_cost_mean'])
        optimal['min_cost'] = {
            'capacity': min_cost_result['capacity'],
            'cost': min_cost_result['total_cost_mean'],
            'balance': min_cost_result['capacity_balance']
        }
        
        # Find configurations that maximize entropy
        max_entropy_result = max(capacity_results, key=lambda r: r['entropy_mean'])
        optimal['max_entropy'] = {
            'capacity': max_entropy_result['capacity'],
            'entropy': max_entropy_result['entropy_mean'],
            'balance': max_entropy_result['capacity_balance']
        }
        
        # Find most balanced configuration with good performance
        balanced_results = sorted(capacity_results, key=lambda r: r['capacity_balance'])
        if balanced_results:
            optimal['most_balanced'] = {
                'capacity': balanced_results[0]['capacity'],
                'balance': balanced_results[0]['capacity_balance'],
                'cost': balanced_results[0]['total_cost_mean']
            }
        
        return optimal
    
    def _analyse_balance_effects(self, capacity_results: List[Dict]) -> Dict[str, Any]:
        """Analyse effects of capacity balance on performance."""
        balance_analysis = {}
        
        balances = [r['capacity_balance'] for r in capacity_results]
        costs = [r['total_cost_mean'] for r in capacity_results]
        entropies = [r['entropy_mean'] for r in capacity_results]
        
        # Balance vs performance correlations
        balance_analysis['balance_vs_cost'] = {
            'correlation': self._safe_correlation(balances, costs),
            'trend': self._determine_trend(balances, costs)
        }
        
        balance_analysis['balance_vs_entropy'] = {
            'correlation': self._safe_correlation(balances, entropies),
            'trend': self._determine_trend(balances, entropies)
        }
        
        # Compare balanced vs unbalanced configurations
        median_balance = np.median(balances)
        balanced = [r for r in capacity_results if r['capacity_balance'] <= median_balance]
        unbalanced = [r for r in capacity_results if r['capacity_balance'] > median_balance]
        
        if balanced and unbalanced:
            balance_analysis['balanced_vs_unbalanced'] = {
                'balanced_avg_cost': np.mean([r['total_cost_mean'] for r in balanced]),
                'unbalanced_avg_cost': np.mean([r['total_cost_mean'] for r in unbalanced]),
                'balanced_avg_entropy': np.mean([r['entropy_mean'] for r in balanced]),
                'unbalanced_avg_entropy': np.mean([r['entropy_mean'] for r in unbalanced])
            }
        
        return balance_analysis
    
    def _cluster_capacities(self, capacity_results: List[Dict]) -> Dict[str, Any]:
        """Cluster capacity configurations by performance."""
        clustering = {}
        
        # Simple performance-based clustering
        costs = [r['total_cost_mean'] for r in capacity_results]
        entropies = [r['entropy_mean'] for r in capacity_results]
        
        # Divide into performance quartiles
        cost_quartiles = np.percentile(costs, [25, 50, 75])
        entropy_quartiles = np.percentile(entropies, [25, 50, 75])
        
        clusters = {
            'low_cost': [],
            'medium_cost': [],
            'high_cost': [],
            'low_entropy': [],
            'medium_entropy': [],
            'high_entropy': []
        }
        
        for result in capacity_results:
            cost = result['total_cost_mean']
            entropy = result['entropy_mean']
            
            # Cost clustering
            if cost <= cost_quartiles[0]:
                clusters['low_cost'].append(result['capacity'])
            elif cost <= cost_quartiles[2]:
                clusters['medium_cost'].append(result['capacity'])
            else:
                clusters['high_cost'].append(result['capacity'])
            
            # Entropy clustering
            if entropy <= entropy_quartiles[0]:
                clusters['low_entropy'].append(result['capacity'])
            elif entropy <= entropy_quartiles[2]:
                clusters['medium_entropy'].append(result['capacity'])
            else:
                clusters['high_entropy'].append(result['capacity'])
        
        clustering['performance_clusters'] = clusters
        
        return clustering
    
    def _safe_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Safely calculate correlation coefficient."""
        if len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
            return 0.0
        
        try:
            with np.errstate(invalid='ignore'):
                corr_matrix = np.corrcoef(x_values, y_values)
                correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                return correlation
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return 0.0
    
    def _determine_trend(self, x_values: List[float], y_values: List[float]) -> str:
        """Determine trend direction between two variables."""
        if len(set(x_values)) <= 1:
            return 'no_variation'
        
        correlation = self._safe_correlation(x_values, y_values)
        
        if abs(correlation) > 0.7:
            return 'strong_positive' if correlation > 0 else 'strong_negative'
        elif abs(correlation) > 0.3:
            return 'weak_positive' if correlation > 0 else 'weak_negative'
        else:
            return 'no_clear_trend' 