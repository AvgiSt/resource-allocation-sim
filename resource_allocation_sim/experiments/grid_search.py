"""Grid search experiment for parameter exploration."""

import numpy as np
from typing import Dict, List, Any, Optional
from itertools import product

from .base_experiment import BaseExperiment
from ..utils.helpers import generate_parameter_grid
from ..evaluation.metrics import calculate_entropy, calculate_gini_coefficient


class GridSearchExperiment(BaseExperiment):
    """
    Grid search experiment for systematic parameter exploration.
    
    Tests all combinations of specified parameter values.
    """
    
    def __init__(
        self,
        parameter_grid: Dict[str, List[Any]],
        **kwargs
    ):
        """
        Initialise grid search experiment.
        
        Args:
            parameter_grid: Dictionary of parameter names to lists of values
            base_config: Base configuration
            results_dir: Directory to save results
            experiment_name: Optional experiment name
        """
        super().__init__(**kwargs)
        self.parameter_grid = parameter_grid
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        return generate_parameter_grid(**self.parameter_grid)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze grid search results."""
        analysis = {
            'parameter_effects': {},
            'best_configurations': {},
            'parameter_correlations': {}
        }
        
        # Extract data for analysis
        all_data = []
        for result in self.results:
            config_params = result['config_params']
            summary_metrics = result['summary_metrics']
            
            row = {}
            row.update(config_params)
            row.update(summary_metrics)
            all_data.append(row)
        
        if not all_data:
            return analysis
        
        import pandas as pd
        df = pd.DataFrame(all_data)
        
        # Analyze parameter effects
        for param_name in self.parameter_grid.keys():
            if param_name in df.columns:
                analysis['parameter_effects'][param_name] = self._analyze_parameter_effect(
                    df, param_name
                )
        
        # Find best configurations
        metric_names = ['entropy_mean', 'total_cost_mean', 'gini_coefficient_mean']
        for metric in metric_names:
            if metric in df.columns:
                best_idx = df[metric].idxmin() if 'cost' in metric else df[metric].idxmax()
                best_config = df.loc[best_idx, list(self.parameter_grid.keys())].to_dict()
                analysis['best_configurations'][metric] = {
                    'config': best_config,
                    'value': df.loc[best_idx, metric]
                }
        
        # Parameter correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            analysis['parameter_correlations'] = correlation_matrix.to_dict()
        
        return analysis
    
    def _analyze_parameter_effect(
        self, 
        df: 'pd.DataFrame', 
        param_name: str
    ) -> Dict[str, Any]:
        """Analyze the effect of a single parameter."""
        effect_analysis = {}
        
        # Group by parameter value
        grouped = df.groupby(param_name)
        
        metric_names = [col for col in df.columns if col.endswith('_mean')]
        
        for metric in metric_names:
            if metric in df.columns:
                effect_analysis[metric] = {
                    'values': grouped[metric].mean().to_dict(),
                    'std': grouped[metric].std().to_dict(),
                    'variance_explained': self._calculate_variance_explained(df, param_name, metric)
                }
        
        return effect_analysis
    
    def _calculate_variance_explained(
        self, 
        df: 'pd.DataFrame', 
        param_name: str, 
        metric_name: str
    ) -> float:
        """Calculate variance explained by parameter."""
        try:
            total_variance = df[metric_name].var()
            if total_variance == 0:
                return 0.0
            
            grouped = df.groupby(param_name)[metric_name]
            between_group_variance = grouped.mean().var()
            
            return between_group_variance / total_variance
        except Exception:
            return 0.0 