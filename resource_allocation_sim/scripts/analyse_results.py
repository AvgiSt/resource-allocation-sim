"""Results analysis script for resource allocation simulations."""

import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ..evaluation.metrics import calculate_entropy, calculate_gini_coefficient
from ..evaluation.system_analysis import SystemAnalysis
from ..visualisation.plots import plot_convergence_comparison, plot_parameter_sensitivity
from ..visualisation.ternary import plot_ternary_distribution
from ..visualisation.network import visualise_state_network


def analyse_single_run(
    results_path: str,
    output_dir: Optional[str] = None,
    include_plots: bool = True
) -> Dict[str, Any]:
    """
    Analyse results from a single simulation run.
    
    Args:
        results_path: Path to results file
        output_dir: Directory to save analysis outputs
        include_plots: Whether to generate plots
        
    Returns:
        Dictionary of analysis results
    """
    # Load results
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Perform system analysis
    analysis = SystemAnalysis(results)
    metrics = analysis.calculate_all_metrics()
    
    print(f"Analysis for: {results_path}")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Generate plots if requested
    if include_plots and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Basic plots
        try:
            from ..visualisation.plots import plot_resource_distribution
            fig = plot_resource_distribution(
                results['final_consumption'],
                results.get('capacity', [1.0] * len(results['final_consumption'])),
                save_path=str(output_path / 'resource_distribution.png')
            )
            print(f"Saved resource distribution plot to {output_path}")
        except Exception as e:
            print(f"Warning: Could not generate resource distribution plot: {e}")
        
        # Network visualisation if possible
        try:
            if 'agent_results' in results:
                primary_result = results['agent_results'] if isinstance(results['agent_results'], dict) else results['agent_results'][0]
                fig = visualise_state_network(primary_result['agent_results'])
                fig.savefig(str(output_path / 'state_network.png'), dpi=300, bbox_inches='tight')
                print(f"Saved network visualisation to {output_path}")
        except Exception as e:
            print(f"Warning: Could not generate network visualisation: {e}")
    
    return metrics


def analyse_parameter_sweep(
    results_dir: str,
    parameter_name: str,
    output_dir: Optional[str] = None
) -> None:
    """Analyse results from parameter sweep experiment."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Load all results
    parameter_results = {}
    
    for result_file in results_path.glob("*.pkl"):
        try:
            with open(result_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract parameter value from filename or data
            param_value = data.get(parameter_name, result_file.stem)
            parameter_results[param_value] = data
            
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
    
    if not parameter_results:
        print("No valid results found")
        return
    
    print(f"Analysing parameter sweep for: {parameter_name}")
    print(f"Found {len(parameter_results)} parameter values")
    
    # Generate comparison plots
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert to format expected by plotting function
            plot_data = {}
            for param_val, results in parameter_results.items():
                plot_data[str(param_val)] = [results]  # Wrap in list for consistency
            
            fig = plot_convergence_comparison(
                plot_data,
                metric='entropy',
                save_path=str(output_path / 'parameter_comparison.png')
            )
            print(f"Saved parameter comparison plot to {output_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate comparison plot: {e}")


def create_basic_visualisations(
    results_path: str,
    output_dir: str
) -> None:
    """Create basic visualisation plots."""
    results_path = Path(results_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from ..visualisation.plots import plot_resource_distribution
        
        # Load results
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Resource distribution
        if 'final_consumption' in results:
            fig = plot_resource_distribution(
                results['final_consumption'],
                results.get('capacity', [1.0] * len(results['final_consumption'])),
                save_path=str(output_path / 'resource_distribution.png')
            )
            print(f"Created resource distribution plot")
        
        # Consumption over time
        if 'consumption_history' in results:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            history = results['consumption_history']
            
            for i, resource_history in enumerate(history.T):
                ax.plot(resource_history, label=f'Resource {i+1}')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Consumption')
            ax.set_title('Consumption Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.savefig(str(output_path / 'consumption_timeline.png'), dpi=300, bbox_inches='tight')
            print(f"Created consumption timeline plot")
        
    except Exception as e:
        print(f"Error creating visualisations: {e}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyse resource allocation simulation results')
    parser.add_argument('results_path', help='Path to results file or directory')
    parser.add_argument('--output', '-o', default='analysis_output', 
                       help='Output directory for analysis')
    parser.add_argument('--parameter', '-p', help='Parameter name for sweep analysis')
    parser.add_argument('--basic-plots', action='store_true', 
                       help='Generate basic visualisation plots')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip plot generation')
    parser.add_argument('--network', action='store_true', 
                       help='Generate network visualisations')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    
    if not results_path.exists():
        print(f"Error: Path does not exist: {results_path}")
        sys.exit(1)
    
    try:
        if results_path.is_file():
            # Single file analysis
            metrics = analyse_single_run(
                str(results_path),
                args.output if not args.no_plots else None,
                include_plots=not args.no_plots
            )
            
            if args.basic_plots:
                create_basic_visualisations(str(results_path), args.output)
                
        elif results_path.is_dir():
            # Directory analysis
            if args.parameter:
                analyse_parameter_sweep(str(results_path), args.parameter, args.output)
            else:
                print("For directory analysis, please specify --parameter")
                sys.exit(1)
        
        print(f"\nAnalysis completed! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 