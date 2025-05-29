"""Parameter sweep experiment for single-parameter analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from .base_experiment import BaseExperiment


class ParameterSweepExperiment(BaseExperiment):
    """
    Parameter sweep experiment for analyzing single parameter effects.
    
    Varies one parameter while keeping others fixed.
    """
    
    def __init__(
        self,
        parameter_name: str,
        parameter_values: List[Any],
        **kwargs
    ):
        """
        Initialize parameter sweep experiment.
        
        Args:
            parameter_name: Name of parameter to sweep
            parameter_values: List of values to test
            **kwargs: Arguments passed to BaseExperiment
        """
        super().__init__(**kwargs)
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate configurations with swept parameter."""
        configurations = []
        for value in self.parameter_values:
            config = {self.parameter_name: value}
            configurations.append(config)
        return configurations
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze parameter sweep results."""
        analysis = {
            'parameter_name': self.parameter_name,
            'parameter_values': self.parameter_values,
            'metrics_vs_parameter': {},
            'trends': {},
            'optimal_values': {}
        }
        
        # Extract metrics for each parameter value
        param_to_metrics = {}
        
        for result in self.results:
            param_value = result['config_params'][self.parameter_name]
            summary_metrics = result['summary_metrics']
            
            if param_value not in param_to_metrics:
                param_to_metrics[param_value] = []
            param_to_metrics[param_value].append(summary_metrics)
        
        # Calculate statistics for each metric
        metric_names = set()
        for metrics_list in param_to_metrics.values():
            for metrics in metrics_list:
                metric_names.update(metrics.keys())
        
        for metric in metric_names:
            metric_data = {}
            
            for param_value in self.parameter_values:
                if param_value in param_to_metrics:
                    values = [m.get(metric, 0) for m in param_to_metrics[param_value]]
                    metric_data[param_value] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }
                else:
                    metric_data[param_value] = {
                        'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'values': []
                    }
            
            analysis['metrics_vs_parameter'][metric] = metric_data
            
            # Analyze trends
            means = [metric_data[pv]['mean'] for pv in self.parameter_values]
            analysis['trends'][metric] = self._analyze_trend(self.parameter_values, means)
            
            # Find optimal values
            if 'cost' in metric.lower():
                # For cost metrics, find minimum
                optimal_param = min(metric_data.keys(), key=lambda k: metric_data[k]['mean'])
            else:
                # For other metrics, find maximum (assuming higher is better)
                optimal_param = max(metric_data.keys(), key=lambda k: metric_data[k]['mean'])
            
            analysis['optimal_values'][metric] = {
                'parameter_value': optimal_param,
                'metric_value': metric_data[optimal_param]['mean']
            }
        
        return analysis
    
    def _analyze_trend(
        self, 
        param_values: List[Any], 
        metric_values: List[float]
    ) -> Dict[str, Any]:
        """Analyze trend in metric vs parameter."""
        trend_analysis = {}
        
        try:
            # Convert to numeric if possible
            numeric_params = [float(p) for p in param_values]
            numeric_metrics = [float(m) for m in metric_values]
            
            # Calculate correlation
            correlation = np.corrcoef(numeric_params, numeric_metrics)[0, 1]
            trend_analysis['correlation'] = correlation
            
            # Fit linear trend
            coeffs = np.polyfit(numeric_params, numeric_metrics, 1)
            trend_analysis['linear_slope'] = coeffs[0]
            trend_analysis['linear_intercept'] = coeffs[1]
            
            # Determine trend direction
            if abs(correlation) > 0.7:
                if correlation > 0:
                    trend_analysis['trend'] = 'increasing'
                else:
                    trend_analysis['trend'] = 'decreasing'
            elif abs(correlation) > 0.3:
                if correlation > 0:
                    trend_analysis['trend'] = 'weakly_increasing'
                else:
                    trend_analysis['trend'] = 'weakly_decreasing'
            else:
                trend_analysis['trend'] = 'no_clear_trend'
                
        except (ValueError, TypeError):
            trend_analysis['trend'] = 'non_numeric_parameter'
            trend_analysis['correlation'] = None
        
        return trend_analysis 