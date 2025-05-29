"""Comprehensive study combining multiple experiment types."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import logging

from .base_experiment import BaseExperiment
from .parameter_sweep import ParameterSweepExperiment
from .capacity_analysis import CapacityAnalysisExperiment
from .grid_search import GridSearchExperiment
from ..utils.config import Config
from ..utils.io import ensure_directory, save_results


class ComprehensiveStudy:
    """
    Comprehensive study that combines multiple experiment types.
    
    Runs a full characterization of the resource allocation system
    including parameter sensitivity, capacity analysis, scaling studies, etc.
    """
    
    def __init__(
        self,
        study_config: Dict[str, Any],
        base_config: Config,
        results_dir: str = "results/comprehensive_studies"
    ):
        """Initialize comprehensive study."""
        self.study_config = study_config
        self.base_config = base_config
        
        # Create study directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = study_config.get('study_name', 'comprehensive_study')
        self.experiment_dir = Path(results_dir) / f"{study_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.study_results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            'study_config': study_config,
            'base_config': base_config.to_dict()
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the study."""
        log_file = self.experiment_dir / 'study.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Run the comprehensive study."""
        self.metadata['start_time'] = datetime.now().isoformat()
        
        logging.info("Starting comprehensive study")
        
        try:
            # 1. Weight Sensitivity Analysis
            if self.study_config.get('analyses', {}).get('weight_analysis', True):
                logging.info("Running weight sensitivity analysis")
                self.study_results['weight_analysis'] = self._run_weight_analysis(num_episodes)
            
            # 2. Capacity Analysis
            if self.study_config.get('analyses', {}).get('capacity_analysis', True):
                logging.info("Running capacity analysis")
                self.study_results['capacity_analysis'] = self._run_capacity_analysis(num_episodes)
            
            # 3. Scaling Analysis
            if self.study_config.get('analyses', {}).get('scaling_analysis', False):
                logging.info("Running scaling analysis")
                self.study_results['scaling_analysis'] = self._run_scaling_analysis(num_episodes)
            
            # 4. Initial Condition Analysis
            if self.study_config.get('analyses', {}).get('initial_condition_analysis', False):
                logging.info("Running initial condition analysis")
                self.study_results['initial_condition_analysis'] = self._run_initial_condition_analysis(num_episodes)
            
            # 5. Convergence Analysis
            if self.study_config.get('analyses', {}).get('convergence_analysis', False):
                logging.info("Running convergence analysis")
                self.study_results['convergence_analysis'] = self._run_convergence_analysis(num_episodes)
            
            # Generate comprehensive report
            self._generate_comprehensive_report()
            
            # Save results
            self._save_results()
            
        except Exception as e:
            logging.error(f"Error during comprehensive study: {e}")
            raise
        finally:
            self.metadata['end_time'] = datetime.now().isoformat()
        
        logging.info("Comprehensive study completed")
        return self.study_results
    
    def _run_weight_analysis(self, num_episodes: int) -> Dict[str, Any]:
        """Run weight sensitivity analysis."""
        weight_config = self.study_config.get('weight_analysis', {})
        weight_range = weight_config.get('weight_range', [0.1, 0.9])
        num_steps = weight_config.get('num_steps', 9)
        
        weight_values = np.linspace(weight_range[0], weight_range[1], num_steps)
        
        experiment = ParameterSweepExperiment(
            parameter_name='weight',
            parameter_values=weight_values.tolist(),
            base_config=self.base_config,
            results_dir=self.experiment_dir / 'weight_analysis',
            experiment_name='weight_sweep'
        )
        
        results = experiment.run_experiment(num_episodes)
        analysis = experiment.analyze_results()
        
        return {
            'results': results,
            'analysis': analysis,
            'best_weight': self._find_optimal_weight(analysis)
        }
    
    def _run_capacity_analysis(self, num_episodes: int) -> Dict[str, Any]:
        """Run capacity configuration analysis."""
        capacity_config = self.study_config.get('capacity_analysis', {})
        capacity_ranges = capacity_config.get('capacity_ranges', {
            'symmetric': [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            'asymmetric': [[0.5, 1.0, 1.5], [0.2, 0.8, 1.0], [1.0, 2.0, 0.5]]
        })
        
        # Flatten capacity ranges
        all_capacities = []
        for category, configs in capacity_ranges.items():
            all_capacities.extend(configs)
        
        experiment = CapacityAnalysisExperiment(
            capacity_ranges={'capacity': all_capacities},
            base_config=self.base_config,
            results_dir=self.experiment_dir / 'capacity_analysis',
            experiment_name='capacity_study'
        )
        
        results = experiment.run_experiment(num_episodes)
        analysis = experiment.analyze_results()
        
        return {
            'results': results,
            'analysis': analysis,
            'optimal_configurations': self._find_optimal_capacities(analysis)
        }
    
    def _run_scaling_analysis(self, num_episodes: int) -> Dict[str, Any]:
        """Run scaling analysis."""
        scaling_config = self.study_config.get('scaling_analysis', {})
        agent_counts = scaling_config.get('agent_counts', [5, 10, 20, 50])
        resource_counts = scaling_config.get('resource_counts', [3, 5, 7])
        
        scaling_results = {
            'agent_scaling': {},
            'resource_scaling': {}
        }
        
        # Agent scaling
        for num_agents in agent_counts:
            experiment = ParameterSweepExperiment(
                parameter_name='num_agents',
                parameter_values=[num_agents],
                base_config=self.base_config,
                results_dir=self.experiment_dir / 'scaling_analysis' / f'agents_{num_agents}',
                experiment_name=f'agent_scaling_{num_agents}'
            )
            
            results = experiment.run_experiment(num_episodes)
            scaling_results['agent_scaling'][num_agents] = results
        
        # Resource scaling
        for num_resources in resource_counts:
            # Adjust base config for different resource counts
            modified_config = Config()
            modified_config.__dict__.update(self.base_config.__dict__)
            modified_config.num_resources = num_resources
            modified_config.capacity = [1.0] * num_resources
            
            experiment = ParameterSweepExperiment(
                parameter_name='num_resources',
                parameter_values=[num_resources],
                base_config=modified_config,
                results_dir=self.experiment_dir / 'scaling_analysis' / f'resources_{num_resources}',
                experiment_name=f'resource_scaling_{num_resources}'
            )
            
            results = experiment.run_experiment(num_episodes)
            scaling_results['resource_scaling'][num_resources] = results
        
        return scaling_results
    
    def _run_initial_condition_analysis(self, num_episodes: int) -> Dict[str, Any]:
        """Run initial condition analysis."""
        # This would require modifications to the Agent class to support different
        # initial probability distributions. For now, return placeholder.
        return {
            'conditions_tested': ['uniform', 'concentrated', 'bimodal'],
            'results': {},
            'analysis': 'Initial condition analysis requires extended Agent implementation'
        }
    
    def _run_convergence_analysis(self, num_episodes: int) -> Dict[str, Any]:
        """Run convergence criteria analysis."""
        # Test different convergence criteria
        convergence_results = {}
        
        # Run with different iteration counts to study convergence
        iteration_counts = [100, 500, 1000, 2000]
        
        for iterations in iteration_counts:
            modified_config = Config()
            modified_config.__dict__.update(self.base_config.__dict__)
            modified_config.num_iterations = iterations
            
            experiment = ParameterSweepExperiment(
                parameter_name='num_iterations',
                parameter_values=[iterations],
                base_config=modified_config,
                results_dir=self.experiment_dir / 'convergence_analysis' / f'iter_{iterations}',
                experiment_name=f'convergence_{iterations}'
            )
            
            results = experiment.run_experiment(num_episodes)
            convergence_results[iterations] = results
        
        return convergence_results
    
    def _find_optimal_weight(self, weight_analysis: Dict[str, Any]) -> float:
        """Find optimal weight parameter."""
        if 'metrics_vs_parameter' in weight_analysis:
            metrics = weight_analysis['metrics_vs_parameter']
            
            # Look for cost metrics to minimize
            best_weight = 0.5  # default
            best_score = float('inf')
            
            for metric_name, metric_data in metrics.items():
                if 'total_cost_mean' in metric_name:
                    for weight, cost_data in metric_data.items():
                        score = cost_data['mean']
                        if score < best_score:
                            best_score = score
                            best_weight = weight
            
            return best_weight
        
        return 0.5  # default fallback
    
    def _find_optimal_capacities(self, capacity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find optimal capacity configurations."""
        optimal_configs = []
        
        if 'capacity_effects' in capacity_analysis:
            effects = capacity_analysis['capacity_effects']
            
            for config, metrics in effects.items():
                efficiency_score = 1.0 / (metrics.get('total_cost_mean', 1.0) + 1e-6)
                fairness_score = 1.0 / (1.0 + metrics.get('gini_coefficient_mean', 0.0))
                combined_score = efficiency_score * fairness_score
                
                optimal_configs.append({
                    'configuration': config,
                    'efficiency_score': efficiency_score,
                    'fairness_score': fairness_score,
                    'combined_score': combined_score
                })
        
        optimal_configs.sort(key=lambda x: x['combined_score'], reverse=True)
        return optimal_configs[:5]
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        report_sections = []
        
        report_sections.append("=" * 80)
        report_sections.append("COMPREHENSIVE RESOURCE ALLOCATION STUDY REPORT")
        report_sections.append("=" * 80)
        report_sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("")
        
        # Executive Summary
        report_sections.append("EXECUTIVE SUMMARY")
        report_sections.append("-" * 40)
        
        # Weight Analysis Summary
        if 'weight_analysis' in self.study_results:
            weight_results = self.study_results['weight_analysis']
            best_weight = weight_results.get('best_weight', 'N/A')
            report_sections.append(f"Optimal Weight Parameter: {best_weight}")
        
        # Capacity Analysis Summary
        if 'capacity_analysis' in self.study_results:
            capacity_results = self.study_results['capacity_analysis']
            optimal_configs = capacity_results.get('optimal_configurations', [])
            if optimal_configs:
                best_config = optimal_configs[0]
                report_sections.append(f"Best Capacity Configuration: {best_config['configuration']}")
                report_sections.append(f"Combined Score: {best_config['combined_score']:.4f}")
        
        # Scaling Analysis Summary
        if 'scaling_analysis' in self.study_results:
            scaling_results = self.study_results['scaling_analysis']
            report_sections.append("Scaling behavior analyzed for both agents and resources")
        
        # Save report
        report_content = "\n".join(report_sections)
        report_path = self.experiment_dir / 'comprehensive_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.study_results['report_path'] = str(report_path)
    
    def _save_results(self):
        """Save all study results."""
        # Save complete results
        save_results(
            {
                'metadata': self.metadata,
                'study_config': self.study_config,
                'results': self.study_results
            },
            'comprehensive_study_results',
            'pickle',
            self.experiment_dir
        )
        
        # Save metadata
        with open(self.experiment_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save study config
        with open(self.experiment_dir / 'study_config.json', 'w') as f:
            json.dump(self.study_config, f, indent=2)
    
    def get_results_dir(self) -> Path:
        """Get study results directory."""
        return self.experiment_dir
    
    def analyze_cross_experiment_correlations(self) -> Dict[str, Any]:
        """Analyze correlations across different experiments."""
        correlations = {}
        
        # Extract key metrics from each experiment type
        weight_metrics = self._extract_metrics_from_weight_analysis()
        capacity_metrics = self._extract_metrics_from_capacity_analysis()
        
        # Calculate correlations between different analysis types
        if weight_metrics and capacity_metrics:
            # This would require more sophisticated analysis
            correlations['weight_vs_capacity'] = self._calculate_correlation_matrix(
                weight_metrics, capacity_metrics
            )
        
        return correlations
    
    def _extract_metrics_from_weight_analysis(self) -> Optional[Dict[str, List[float]]]:
        """Extract metrics from weight analysis results."""
        if 'weight_analysis' not in self.study_results:
            return None
        
        analysis = self.study_results['weight_analysis']['analysis']
        if 'metrics_vs_parameter' not in analysis:
            return None
        
        metrics = {}
        for metric_name, metric_data in analysis['metrics_vs_parameter'].items():
            values = [data['mean'] for data in metric_data.values()]
            metrics[metric_name] = values
        
        return metrics
    
    def _extract_metrics_from_capacity_analysis(self) -> Optional[Dict[str, List[float]]]:
        """Extract metrics from capacity analysis results."""
        if 'capacity_analysis' not in self.study_results:
            return None
        
        # This would extract metrics from capacity analysis
        # Implementation depends on capacity analysis structure
        return None
    
    def _calculate_correlation_matrix(
        self, 
        metrics1: Dict[str, List[float]], 
        metrics2: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate correlation matrix between two sets of metrics."""
        correlations = {}
        
        for metric1_name, values1 in metrics1.items():
            for metric2_name, values2 in metrics2.items():
                if len(values1) == len(values2):
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[f"{metric1_name}_vs_{metric2_name}"] = correlation
        
        return correlations 