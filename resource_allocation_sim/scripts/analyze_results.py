"""Command-line script for analyzing simulation results."""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.io import load_results
from ..evaluation.metrics import calculate_system_metrics
from ..evaluation.agent_analysis import analyze_agent_convergence
from ..evaluation.system_analysis import analyze_system_performance, generate_analysis_report
from ..visualization.plots import plot_convergence_comparison, plot_parameter_sensitivity
from ..visualization.ternary import plot_ternary_distribution
from ..visualization.network import visualize_state_network


def analyze_results_file(
    results_path: Path,
    output_dir: Path,
    include_agents: bool = False,
    include_network: bool = False,
    include_ternary: bool = False,
    generate_report: bool = False
):
    """
    Analyze a single results file.
    
    Args:
        results_path: Path to results file
        output_dir: Output directory for analysis
        include_agents: Whether to include agent-specific analysis
        include_network: Whether to generate network visualizations
        include_ternary: Whether to generate ternary plots
        generate_report: Whether to generate a comprehensive report
    """
    # Load results
    if results_path.is_file():
        results = load_results(results_path)
    else:
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Handle both single episode and multi-episode results
    if isinstance(results, list):
        # Multi-episode results
        primary_result = results[-1]  # Use last episode
        all_episodes = results
    else:
        # Single episode result
        primary_result = results
        all_episodes = [results]
    
    # Basic system analysis
    system_analysis = analyze_system_performance(primary_result)
    
    # Save system analysis
    with open(output_dir / 'system_analysis.json', 'w') as f:
        json.dump(system_analysis, f, indent=2)
    
    # Generate basic plots
    _create_basic_plots(primary_result, output_dir)
    
    # Agent analysis
    if include_agents and 'agent_results' in primary_result:
        agent_analysis = analyze_agent_convergence(primary_result['agent_results'])
        
        with open(output_dir / 'agent_analysis.json', 'w') as f:
            json.dump(agent_analysis, f, indent=2)
        
        _create_agent_plots(agent_analysis, output_dir)
    
    # Network analysis
    if include_network and 'agent_results' in primary_result:
        try:
            fig = visualize_state_network(primary_result['agent_results'])
            fig.savefig(output_dir / 'state_network.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Could not generate network visualization: {e}")
    
    # Ternary plots
    if include_ternary and 'agent_results' in primary_result:
        try:
            # Check if we have 3 resources
            first_agent_data = next(iter(primary_result['agent_results'].values()))
            if first_agent_data['prob'] and len(first_agent_data['prob'][0]) == 3:
                fig = plot_ternary_distribution(primary_result['agent_results'])
                fig.savefig(output_dir / 'ternary_distribution.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Could not generate ternary plot: {e}")
    
    # Comprehensive report
    if generate_report:
        report = generate_analysis_report(primary_result)
        with open(output_dir / 'analysis_report.txt', 'w') as f:
            f.write(report)


def _create_basic_plots(results: Dict[str, Any], output_dir: Path):
    """Create basic visualization plots."""
    import matplotlib.pyplot as plt
    
    # Resource distribution plot
    if 'final_consumption' in results:
        from ..visualization.plots import plot_resource_distribution
        
        capacity = results.get('config', {}).get('capacity', [1.0, 1.0, 1.0])
        fig = plot_resource_distribution(results['final_consumption'], capacity)
        fig.savefig(output_dir / 'resource_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Cost evolution (if available)
    if 'agent_results' in results:
        _plot_system_evolution(results, output_dir)


def _plot_system_evolution(results: Dict[str, Any], output_dir: Path):
    """Plot system evolution over time."""
    import matplotlib.pyplot as plt
    from ..evaluation.metrics import calculate_entropy
    
    agent_results = results['agent_results']
    
    # Extract evolution data
    all_entropies = []
    iterations = None
    
    for agent_id, data in agent_results.items():
        prob_history = data['prob']
        if prob_history:
            entropies = [calculate_entropy(probs) for probs in prob_history]
            all_entropies.append(entropies)
            if iterations is None:
                iterations = len(entropies)
    
    if all_entropies:
        # Plot entropy evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, entropies in enumerate(all_entropies):
            ax.plot(entropies, alpha=0.3, color='blue')
        
        # Plot average
        mean_entropies = [sum(entropies[i] for entropies in all_entropies) / len(all_entropies) 
                         for i in range(iterations)]
        ax.plot(mean_entropies, 'r-', linewidth=2, label='Mean Entropy')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy')
        ax.set_title('System Entropy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.savefig(output_dir / 'entropy_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()


def _create_agent_plots(agent_analysis: Dict[str, Any], output_dir: Path):
    """Create agent-specific plots."""
    try:
        import matplotlib.pyplot as plt
        from ..evaluation.agent_analysis import plot_probability_distribution, plot_agent_entropy_evolution
        
        # This function was referenced but not implemented
        # Add basic implementation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot convergence times if available
        if 'convergence_times' in agent_analysis:
            conv_times = list(agent_analysis['convergence_times'].values())
            ax.hist(conv_times, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Convergence Time')
            ax.set_ylabel('Number of Agents')
            ax.set_title('Agent Convergence Time Distribution')
            
            fig.savefig(output_dir / 'agent_convergence_times.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Warning: Could not generate agent plots: {e}")


def main():
    """Main entry point for results analysis."""
    parser = argparse.ArgumentParser(description='Analyze resource allocation simulation results')
    parser.add_argument('results_path', type=str, help='Path to results file or directory')
    parser.add_argument('--output', type=str, default='analysis_output',
                       help='Output directory for analysis')
    parser.add_argument('--agents', action='store_true',
                       help='Include individual agent analysis')
    parser.add_argument('--network', action='store_true',
                       help='Generate network visualizations')
    parser.add_argument('--ternary', action='store_true',
                       help='Generate ternary plots (for 3-resource systems)')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive analysis report')
    
    args = parser.parse_args()
    
    try:
        results_path = Path(args.results_path)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if results_path.is_file():
            analyze_results_file(
                results_path=results_path,
                output_dir=output_dir,
                include_agents=args.agents,
                include_network=args.network,
                include_ternary=args.ternary,
                generate_report=args.report
            )
        elif results_path.is_dir():
            # Analyze all pickle files in directory
            pickle_files = list(results_path.glob('*.pkl'))
            
            if not pickle_files:
                print(f"No .pkl files found in {results_path}")
                return
            
            for pkl_file in pickle_files:
                file_output_dir = output_dir / pkl_file.stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                analyze_results_file(
                    results_path=pkl_file,
                    output_dir=file_output_dir,
                    include_agents=args.agents,
                    include_network=args.network,
                    include_ternary=args.ternary,
                    generate_report=args.report
                )
        else:
            raise FileNotFoundError(f"Results path not found: {results_path}")
        
        print(f"Analysis completed! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main() 