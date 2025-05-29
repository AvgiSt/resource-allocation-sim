"""Command-line interface for resource allocation simulation."""

import click
import sys
import json
import yaml
from pathlib import Path
from typing import Optional

from .utils.config import Config
from .experiments.comprehensive_study import ComprehensiveStudy
from .experiments.grid_search import GridSearchExperiment
from .experiments.parameter_sweep import ParameterSweepExperiment
from .experiments.capacity_analysis import CapacityAnalysisExperiment
from .core.simulation import SimulationRunner
from .utils.io import save_results, load_results, ensure_directory


@click.group()
@click.version_option()
def cli():
    """Resource Allocation Simulation Framework."""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--iterations', '-i', default=1000, 
              help='Number of iterations per episode')
@click.option('--episodes', '-e', default=10, 
              help='Number of episodes to run')
@click.option('--agents', '-a', default=20, 
              help='Number of agents')
@click.option('--resources', '-r', default=3, 
              help='Number of resources')
@click.option('--weight', '-w', default=0.6, 
              help='Learning weight parameter')
@click.option('--capacity', '-cap', multiple=True, type=float,
              help='Resource capacities (repeat for each resource)')
@click.option('--output', '-o', default='results/single_run',
              help='Output directory')
@click.option('--verbose', '-v', is_flag=True, 
              help='Verbose output')
def run(config, iterations, episodes, agents, resources, weight, capacity, output, verbose):
    """Run a single simulation."""
    try:
        # Load configuration
        if config:
            sim_config = Config(config)
        else:
            sim_config = Config()
            sim_config.num_iterations = iterations
            sim_config.num_agents = agents
            sim_config.num_resources = resources
            sim_config.weight = weight
            if capacity:
                sim_config.capacity = list(capacity)
            else:
                sim_config.capacity = [1.0] * resources
        
        # Validate configuration
        sim_config.validate()
        
        # Create output directory
        output_dir = ensure_directory(output)
        
        # Run simulation
        runner = SimulationRunner(sim_config)
        runner.setup()
        
        if verbose:
            click.echo(f"Running simulation with {agents} agents, {resources} resources")
            click.echo(f"Weight: {weight}, Capacity: {sim_config.capacity}")
        
        # Run multiple episodes
        all_results = runner.run_multiple_episodes(episodes)
        
        # Save results
        results_file = save_results(all_results, 'simulation_results', 'pickle', output_dir)
        
        # Calculate summary statistics
        final_costs = [r['total_cost'] for r in all_results]
        mean_cost = sum(final_costs) / len(final_costs)
        
        click.echo(f"Simulation completed!")
        click.echo(f"Mean final cost: {mean_cost:.4f}")
        click.echo(f"Results saved to: {results_file}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Study configuration file')
@click.option('--episodes', '-e', default=10, 
              help='Number of episodes per configuration')
@click.option('--name', help='Study name override')
def study(config, episodes, name):
    """Run comprehensive study from configuration file."""
    try:
        # Load study configuration
        config_path = Path(config)
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                study_config = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                study_config = json.load(f)
        
        # Override study name if provided
        if name:
            study_config['study_name'] = name
        
        # Create base config from study config
        base_sim_config = study_config.get('base_simulation', {})
        base_config = Config(**base_sim_config)
        
        # Run comprehensive study
        study = ComprehensiveStudy(
            study_config=study_config,
            base_config=base_config
        )
        
        click.echo(f"Starting comprehensive study: {study_config.get('study_name', 'Unnamed')}")
        results = study.run(episodes)
        
        click.echo(f"Study completed! Results saved to: {study.get_results_dir()}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def experiment():
    """Run specific experiment types."""
    pass


@experiment.command()
@click.option('--grid-params', required=True, 
              help='JSON string with parameter grid')
@click.option('--episodes', '-e', default=10, help='Number of episodes')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Base configuration file')
@click.option('--output', '-o', default='results/experiments',
              help='Output directory')
@click.option('--name', help='Experiment name')
def grid(grid_params, episodes, config, output, name):
    """Run grid search experiment."""
    try:
        parameter_grid = json.loads(grid_params)
        
        # Load base config
        if config:
            base_config = Config(config)
        else:
            base_config = Config()
        
        # Create experiment
        experiment = GridSearchExperiment(
            parameter_grid=parameter_grid,
            base_config=base_config,
            results_dir=output,
            experiment_name=name
        )
        
        # Run experiment
        results = experiment.run_experiment(episodes)
        
        click.echo(f"Grid search completed! Results saved to: {experiment.get_results_dir()}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@experiment.command()
@click.option('--param-name', required=True, help='Parameter to sweep')
@click.option('--param-values', required=True, 
              help='Comma-separated values to test')
@click.option('--episodes', '-e', default=10, help='Number of episodes')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Base configuration file')
@click.option('--output', '-o', default='results/experiments',
              help='Output directory')
@click.option('--name', help='Experiment name')
def sweep(param_name, param_values, episodes, config, output, name):
    """Run parameter sweep experiment."""
    try:
        # Convert parameter values
        if ',' in param_values:
            values = [float(x.strip()) for x in param_values.split(',')]
        else:
            values = [float(param_values)]
        
        # Load base config
        if config:
            base_config = Config(config)
        else:
            base_config = Config()
        
        # Create experiment
        experiment = ParameterSweepExperiment(
            parameter_name=param_name,
            parameter_values=values,
            base_config=base_config,
            results_dir=output,
            experiment_name=name
        )
        
        # Run experiment
        results = experiment.run_experiment(episodes)
        
        click.echo(f"Parameter sweep completed! Results saved to: {experiment.get_results_dir()}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@experiment.command()
@click.option('--capacity-configs', 
              help='JSON string with capacity ranges')
@click.option('--episodes', '-e', default=10, help='Number of episodes')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Base configuration file')
@click.option('--output', '-o', default='results/experiments',
              help='Output directory')
@click.option('--name', help='Experiment name')
def capacity(capacity_configs, episodes, config, output, name):
    """Run capacity analysis experiment."""
    try:
        import json
        capacity_ranges = json.loads(capacity_configs)
        
        # Load base config
        if config:
            base_config = Config(config)
        else:
            base_config = Config()
        
        # Create experiment
        experiment = CapacityAnalysisExperiment(
            capacity_ranges=capacity_ranges,
            base_config=base_config,
            results_dir=output,
            experiment_name=name
        )
        
        # Run experiment
        results = experiment.run_experiment(episodes)
        
        click.echo(f"Capacity analysis completed! Results saved to: {experiment.get_results_dir()}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='analysis_output',
              help='Output directory for analysis')
@click.option('--agents', is_flag=True, help='Include agent analysis')
@click.option('--network', is_flag=True, help='Generate network plots')
@click.option('--ternary', is_flag=True, help='Generate ternary plots')
@click.option('--report', is_flag=True, help='Generate comprehensive report')
def analyze(results_path, output, agents, network, ternary, report):
    """Analyze simulation results."""
    try:
        from .scripts.analyze_results import analyze_results_file
        
        results_path = Path(results_path)
        output_dir = ensure_directory(output)
        
        # Load and analyze results
        analyze_results_file(
            results_path=results_path,
            output_dir=output_dir,
            include_agents=agents,
            include_network=network,
            include_ternary=ternary,
            generate_report=report
        )
        
        click.echo(f"Analysis completed! Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--agents', '-a', default=30, help='Number of agents')
@click.option('--resources', '-r', default=3, help='Number of resources')
@click.option('--capacity', '-cap', multiple=True, type=float,
              default=[1.0, 1.0, 1.0], help='Resource capacities')
@click.option('--output', '-o', default='visualization_output',
              help='Output directory')
def visualize(agents, resources, capacity, output):
    """Generate visualization examples."""
    try:
        from .visualization.plots import plot_resource_distribution
        from .visualization.ternary import plot_ternary_distribution
        from .visualization.network import visualize_state_network
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create sample data
        consumption = np.random.multinomial(agents, [1/resources]*resources, size=1)[0]
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resource distribution plot
        fig = plot_resource_distribution(consumption, capacity)
        fig.savefig(output_dir / 'resource_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        click.echo(f"Visualizations saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system information and available commands."""
    click.echo("Resource Allocation Simulation Framework")
    click.echo("=" * 50)
    click.echo()
    click.echo("Available commands:")
    click.echo("  run        - Run single simulation")
    click.echo("  study      - Run comprehensive study")
    click.echo("  experiment - Run specific experiment type")
    click.echo("  analyze    - Analyze results")
    click.echo("  visualize  - Generate visualizations")
    click.echo("  info       - Show this information")
    click.echo()
    click.echo("For help on specific commands, use: command --help")
    click.echo("Example: resource-sim run --help")


if __name__ == '__main__':
    cli() 