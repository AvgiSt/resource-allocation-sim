"""Command-line script for running experiments."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..utils.config import Config
from ..experiments.grid_search import GridSearchExperiment
from ..experiments.parameter_sweep import ParameterSweepExperiment
from ..experiments.capacity_analysis import CapacityAnalysisExperiment


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description='Run resource allocation simulation experiments')
    parser.add_argument('experiment_type', choices=['grid', 'sweep', 'capacity'],
                       help='Type of experiment to run')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/experiments',
                       help='Output directory for results')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes per configuration')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    # Grid search specific arguments
    parser.add_argument('--grid-params', type=str, 
                       help='JSON string with parameter grid (for grid search)')
    
    # Parameter sweep specific arguments  
    parser.add_argument('--sweep-param', type=str,
                       help='Parameter name to sweep (for parameter sweep)')
    parser.add_argument('--sweep-values', type=str,
                       help='Comma-separated values to sweep (for parameter sweep)')
    
    # Capacity analysis specific arguments
    parser.add_argument('--capacity-ranges', type=str,
                       help='JSON string with capacity ranges (for capacity analysis)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config()
        if args.config:
            config = Config(args.config)
        
        # Create experiment based on type
        if args.experiment_type == 'grid':
            experiment = create_grid_experiment(args, config)
        elif args.experiment_type == 'sweep':
            experiment = create_sweep_experiment(args, config)
        elif args.experiment_type == 'capacity':
            experiment = create_capacity_experiment(args, config)
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment_type}")
        
        # Run experiment
        print(f"Starting {args.experiment_type} experiment...")
        print(f"Output directory: {experiment.get_results_dir()}")
        
        def progress_callback(current, total, config_params):
            print(f"Progress: {current+1}/{total} - Config: {config_params}")
        
        results = experiment.run_experiment(
            num_episodes=args.episodes,
            progress_callback=progress_callback
        )
        
        print("Experiment completed successfully!")
        print(f"Results saved to: {experiment.get_results_dir()}")
        
        # Print summary
        print("\nExperiment Summary:")
        print(f"- Configurations tested: {len(results['results'])}")
        print(f"- Episodes per config: {args.episodes}")
        print(f"- Total simulations: {len(results['results']) * args.episodes}")
        
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        sys.exit(1)


def create_grid_experiment(args, config: Config) -> GridSearchExperiment:
    """Create grid search experiment."""
    import json
    
    if not args.grid_params:
        # Default grid
        parameter_grid = {
            'weight': [0.3, 0.5, 0.7],
            'num_agents': [3, 5, 7]
        }
    else:
        parameter_grid = json.loads(args.grid_params)
    
    return GridSearchExperiment(
        parameter_grid=parameter_grid,
        base_config=config,
        results_dir=args.output,
        experiment_name=args.name
    )


def create_sweep_experiment(args, config: Config) -> ParameterSweepExperiment:
    """Create parameter sweep experiment."""
    if not args.sweep_param or not args.sweep_values:
        raise ValueError("Parameter sweep requires --sweep-param and --sweep-values")
    
    # Parse sweep values
    values_str = args.sweep_values.split(',')
    try:
        # Try to convert to float
        sweep_values = [float(v.strip()) for v in values_str]
    except ValueError:
        # Keep as strings
        sweep_values = [v.strip() for v in values_str]
    
    return ParameterSweepExperiment(
        parameter_name=args.sweep_param,
        parameter_values=sweep_values,
        base_config=config,
        results_dir=args.output,
        experiment_name=args.name
    )


def create_capacity_experiment(args, config: Config) -> CapacityAnalysisExperiment:
    """Create capacity analysis experiment."""
    import json
    
    if not args.capacity_ranges:
        # Default capacity ranges
        capacity_ranges = {'capacity_values': [0.5, 1.0, 1.5, 2.0]}
    else:
        capacity_ranges = json.loads(args.capacity_ranges)
    
    return CapacityAnalysisExperiment(
        capacity_ranges=capacity_ranges,
        base_config=config,
        results_dir=args.output,
        experiment_name=args.name
    )


if __name__ == '__main__':
    main() 