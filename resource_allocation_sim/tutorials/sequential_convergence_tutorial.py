#!/usr/bin/env python3
"""
Sequential Convergence Study Tutorial

This tutorial demonstrates how to run the Sequential Convergence Study experiment,
which investigates whether agents starting from uniform initial probability
distributions exhibit sequential convergence patterns rather than simultaneous
convergence.

CONCEPT:
In multi-agent learning systems, agents may converge to their final resource
preferences at different times rather than simultaneously. This sequential
convergence pattern emerges from the partial observability of the system,
where early convergers influence the cost landscape for subsequent agents
through environmental feedback.

This experiment tests the hypothesis that agents will exhibit sequential
convergence to degenerate distributions, with individual agents converging
at distinct time points rather than simultaneously, and all agents achieving
specialised resource preferences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import sys
import os
from typing import List, Tuple

# Add the parent directory to the path to import the experiment modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.sequential_convergence_study import SequentialConvergenceStudy, run_sequential_convergence_study
from utils.config import Config
from evaluation.metrics import calculate_entropy, calculate_convergence_speed
from visualisation.plots import plot_convergence_comparison


def explain_sequential_convergence_concept():
    """
    Explain the concept of sequential convergence in multi-agent learning.
    """
    print("=" * 80)
    print("SEQUENTIAL CONVERGENCE STUDY - CONCEPT EXPLANATION")
    print("=" * 80)
    
    print("\n1. WHAT IS SEQUENTIAL CONVERGENCE?")
    print("-" * 40)
    print("Sequential convergence occurs when agents in a multi-agent system")
    print("reach their final resource preferences at different times rather")
    print("than simultaneously. This creates a temporal ordering of convergence")
    print("events that reflects the emergent coordination dynamics.")
    print()
    print("Key characteristics:")
    print("- Agents converge one after another, not all at once")
    print("- Early convergers influence the environment for later agents")
    print("- Convergence timing reflects resource preference hierarchies")
    print("- Sequential patterns emerge from partial observability")
    
    print("\n2. WHY STUDY SEQUENTIAL CONVERGENCE?")
    print("-" * 40)
    print("- Understanding emergent coordination patterns in multi-agent systems")
    print("- Validating theoretical predictions about convergence dynamics")
    print("- Identifying the role of partial observability in coordination")
    print("- Characterising the temporal structure of learning processes")
    print("- Providing insights into system-level coordination mechanisms")
    
    print("\n3. THEORETICAL FOUNDATION:")
    print("-" * 40)
    print("Sequential convergence emerges from several theoretical principles:")
    print()
    print("Partial Observability:")
    print("- Agents cannot observe other agents' probability distributions")
    print("- Agents only see environmental feedback (resource costs)")
    print("- This creates information asymmetry between agents")
    print()
    print("Environmental Feedback:")
    print("- Early convergers create cost gradients in the environment")
    print("- These gradients influence subsequent agent decisions")
    print("- Later agents adapt to the changed cost landscape")
    print()
    print("Information Cascades:")
    print("- Early decisions create signals for later agents")
    print("- These signals guide subsequent convergence events")
    print("- The process creates a natural ordering of specialisation")
    
    print("\n4. KEY RESEARCH QUESTIONS:")
    print("-" * 40)
    print("- Do agents exhibit sequential convergence patterns?")
    print("- What determines the order of convergence events?")
    print("- How does partial observability influence convergence timing?")
    print("- Are sequential patterns consistent across different scenarios?")
    print("- What are the implications for system-level coordination?")
    
    print("\n5. EXPECTED OUTCOMES:")
    print("-" * 40)
    print("- Agents should converge sequentially rather than simultaneously")
    print("- Convergence timing should show clear temporal separation")
    print("- Early convergers should influence later convergence events")
    print("- All agents should achieve degenerate distributions")
    print("- Sequential patterns should be consistent across replications")


def get_user_parameters():
    """
    Get user input for experiment parameters.
    """
    print("\n" + "=" * 80)
    print("PARAMETER CUSTOMISATION")
    print("=" * 80)
    
    # System parameters
    print("\n1. SYSTEM CONFIGURATION:")
    print("-" * 30)
    
    num_agents = int(input("Number of agents (default: 10): ") or "10")
    num_resources = int(input("Number of resources (default: 3): ") or "3")
    num_iterations = int(input("Number of iterations (default: 2000): ") or "2000")
    num_replications = int(input("Number of replications (default: 100): ") or "100")
    
    # Learning parameters
    print("\n2. LEARNING PARAMETERS:")
    print("-" * 30)
    
    weight = float(input("Learning rate (weight) (default: 0.3): ") or "0.3")
    
    # Resource capacity configuration
    print("\n3. RESOURCE CAPACITY CONFIGURATION:")
    print("-" * 40)
    print("Choose capacity configuration:")
    print("1. Balanced capacities (all equal)")
    print("2. Custom capacity distribution")
    
    capacity_choice = input("Enter choice (1-2, default: 1): ") or "1"
    
    if capacity_choice == "1":
        relative_capacity = [1.0/num_resources] * num_resources
    else:
        print(f"Enter capacity for each of {num_resources} resources:")
        relative_capacity = []
        for i in range(num_resources):
            cap = float(input(f"Capacity for resource {i+1} (default: {1.0/num_resources:.3f}): ") or str(1.0/num_resources))
            relative_capacity.append(cap)
        # Normalise to sum to 1
        total_cap = sum(relative_capacity)
        relative_capacity = [cap/total_cap for cap in relative_capacity]
    
    # Initial conditions configuration
    print("\n4. INITIAL CONDITIONS CONFIGURATION:")
    print("-" * 40)
    print("Choose initial condition type:")
    print("1. Uniform initialisation (all agents start with equal probabilities)")
    print("2. Random initialisation (random probabilities for each agent)")
    print("3. Custom initial distribution")
    
    init_choice = input("Enter choice (1-3, default: 1): ") or "1"
    
    if init_choice == "1":
        initial_condition_type = "uniform"
    elif init_choice == "2":
        initial_condition_type = "random"
    else:  # init_choice == "3"
        print(f"Enter custom probability distribution (must sum to 1.0)")
        print(f"Example for {num_resources} resources: 0.333,0.333,0.334")
        probs_str = input("Probabilities (comma-separated): ")
        probs = [float(x.strip()) for x in probs_str.split(',')]
        # Normalise to sum to 1
        total = sum(probs)
        probs = [p/total for p in probs]
        initial_condition_type = ("custom", probs)
    
    # Convergence parameters
    print("\n5. CONVERGENCE PARAMETERS:")
    print("-" * 30)
    
    convergence_entropy_threshold = float(input("Convergence entropy threshold (default: 0.1): ") or "0.1")
    convergence_max_prob_threshold = float(input("Convergence max probability threshold (default: 0.9): ") or "0.9")
    
    # Analysis parameters
    print("\n6. ANALYSIS PARAMETERS:")
    print("-" * 30)
    
    enable_barycentric_analysis = input("Enable barycentric coordinate analysis? (y/n, default: y): ").lower() != "n"
    enable_trajectory_analysis = input("Enable individual trajectory analysis? (y/n, default: y): ").lower() != "n"
    
    # Output configuration
    print("\n7. OUTPUT CONFIGURATION:")
    print("-" * 30)
    
    output_dir = input("Output directory (default: results/sequential_convergence_tutorial): ") or "results/sequential_convergence_tutorial"
    show_plots = input("Show plots interactively? (y/n, default: y): ").lower() != "n"
    
    # Create parameters dictionary
    params = {
        'num_agents': num_agents,
        'num_resources': num_resources,
        'num_iterations': num_iterations,
        'num_replications': num_replications,
        'weight': weight,
        'relative_capacity': relative_capacity,
        'initial_condition_type': initial_condition_type,
        'convergence_entropy_threshold': convergence_entropy_threshold,
        'convergence_max_prob_threshold': convergence_max_prob_threshold,
        'enable_barycentric_analysis': enable_barycentric_analysis,
        'enable_trajectory_analysis': enable_trajectory_analysis,
        'output_dir': output_dir,
        'show_plots': show_plots
    }
    
    return params


def run_sequential_convergence_experiment(params):
    """
    Run the sequential convergence study experiment with the given parameters.
    """
    print("\n" + "=" * 80)
    print("RUNNING SEQUENTIAL CONVERGENCE STUDY EXPERIMENT")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(params['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = Config()
    config.num_agents = params['num_agents']
    config.num_resources = params['num_resources']
    config.num_iterations = params['num_iterations']
    config.relative_capacity = params['relative_capacity']
    config.convergence_entropy_threshold = params['convergence_entropy_threshold']
    config.convergence_max_prob_threshold = params['convergence_max_prob_threshold']
    config.weight = params['weight']
    
    # Set initial condition type
    if isinstance(params['initial_condition_type'], tuple):
        # Custom initial condition
        config.agent_initialisation_method = "custom"
        config.custom_initial_probabilities = params['initial_condition_type'][1]
    else:
        config.agent_initialisation_method = params['initial_condition_type']
    
    # Run the experiment
    print(f"\nRunning sequential convergence experiment...")
    print(f"Number of agents: {params['num_agents']}")
    print(f"Number of resources: {params['num_resources']}")
    print(f"Number of iterations: {params['num_iterations']}")
    print(f"Number of replications: {params['num_replications']}")
    print(f"Learning rate: {params['weight']}")
    print(f"Initial condition type: {params['initial_condition_type']}")
    
    # Create and run the study
    study = SequentialConvergenceStudy(
        base_config=config,
        results_dir=params['output_dir'],
        experiment_name="sequential_convergence_tutorial",
        convergence_threshold_entropy=params['convergence_entropy_threshold'],
        convergence_threshold_max_prob=params['convergence_max_prob_threshold']
    )
    
    # Run the experiment
    results = study.run_experiment(num_episodes=params['num_replications'])
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {output_path}")
    
    return study, results


def generate_analysis_and_plots(study, params):
    """
    Generate comprehensive analysis and visualisations.
    """
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS AND VISUALISATIONS")
    print("=" * 80)
    
    output_path = Path(params['output_dir'])
    plots_path = output_path / "plots"
    plots_path.mkdir(exist_ok=True)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # 1. Statistical analysis
    print("1. Statistical analysis of sequential patterns...")
    stat_plots = study.create_statistical_analysis_plots(str(plots_path), params['show_plots'])
    
    # 2. Convergence timeline analysis
    print("2. Convergence timeline analysis...")
    timeline_plots = study.create_convergence_timeline_plots(str(plots_path), params['show_plots'])
    
    # 3. Probability evolution analysis
    print("3. Probability evolution analysis...")
    prob_plots = study.create_probability_evolution_plots(str(plots_path), params['show_plots'])
    
    # 4. System dynamics analysis
    print("4. System dynamics analysis...")
    dynamics_plots = study.create_system_dynamics_plots(str(plots_path), params['show_plots'])
    
    # 5. Barycentric coordinate analysis (if enabled)
    if params['enable_barycentric_analysis']:
        print("5. Barycentric coordinate analysis...")
        barycentric_plots = study.create_barycentric_trajectory_plots(str(plots_path), params['show_plots'])
    else:
        barycentric_plots = []
    
    # Combine all plot files
    all_plots = stat_plots + timeline_plots + prob_plots + dynamics_plots + barycentric_plots
    
    print(f"Generated {len(all_plots)} plot files:")
    for plot_file in all_plots:
        print(f"  - {plot_file}")
    
    # Generate analysis
    print("\nGenerating analysis...")
    analysis_results = study.analyse_results()
    
    # Save analysis results
    analysis_path = output_path / "sequential_convergence_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Generate hypothesis evaluation
    print("Generating hypothesis evaluation...")
    if 'hypothesis_support' in analysis_results:
        hypothesis_support = analysis_results['hypothesis_support']
        report = f"Sequential Convergence Study - Hypothesis Evaluation Report\n"
        report += f"=" * 60 + "\n\n"
        
        if 'overall_support' in hypothesis_support:
            report += f"Overall Hypothesis Support: {hypothesis_support['overall_support']}\n\n"
        
        if 'sequential_convergence_support' in hypothesis_support:
            seq_support = hypothesis_support['sequential_convergence_support']
            report += f"Sequential Convergence Support: {seq_support}\n"
        
        if 'degeneracy_achievement_support' in hypothesis_support:
            deg_support = hypothesis_support['degeneracy_achievement_support']
            report += f"Degeneracy Achievement Support: {deg_support}\n"
        
        if 'statistical_significance' in hypothesis_support:
            stat_sig = hypothesis_support['statistical_significance']
            report += f"Statistical Significance: {stat_sig}\n"
    else:
        report = "Analysis completed but hypothesis evaluation not available.\n"
    
    report_path = output_path / "hypothesis_evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save results
    print("Saving results...")
    study.save_results(str(output_path))
    
    print(f"\nAll analysis and plots generated successfully!")
    print(f"Files saved to: {output_path}")
    
    # Display key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    if 'sequential_indices' in analysis_results:
        seq_indices = analysis_results['sequential_indices']
        mean_index = seq_indices.get('mean_index', 0)
        print(f"\nSequential index: {mean_index:.3f}")
        if mean_index > 0.8:
            print("  → Strong sequential convergence patterns detected")
        elif mean_index > 0.5:
            print("  → Moderate sequential convergence patterns detected")
        else:
            print("  → Weak or no sequential convergence patterns")
    
    if 'degeneracy_scores' in analysis_results:
        deg_scores = analysis_results['degeneracy_scores']
        proportion_degenerate = deg_scores.get('proportion_degenerate', 0)
        print(f"\nDegeneracy achievement: {proportion_degenerate:.1%}")
        if proportion_degenerate > 0.9:
            print("  → Excellent degeneracy achievement")
        elif proportion_degenerate > 0.7:
            print("  → Good degeneracy achievement")
        else:
            print("  → Poor degeneracy achievement")
    
    if 'convergence_times' in analysis_results:
        conv_times = analysis_results['convergence_times']
        mean_time = conv_times.get('mean_time', 0)
        print(f"\nAverage convergence time: {mean_time:.1f} iterations")
    
    if params['show_plots']:
        print(f"\nDisplaying plots...")
        # Show plots if requested
        plt.show()


def display_output_structure(params):
    """
    Display the structure of generated outputs.
    """
    print("\n" + "=" * 80)
    print("OUTPUT STRUCTURE")
    print("=" * 80)
    
    output_path = Path(params['output_dir'])
    
    print(f"\nResults will be saved to: {output_path}")
    print("\nFile structure:")
    print(f"{output_path}/")
    print("├── plots/")
    print("│   ├── statistical_analysis.png")
    print("│   ├── convergence_timeline_analysis.png")
    print("│   ├── probability_evolution_analysis.png")
    print("│   ├── system_dynamics_analysis.png")
    
    if params['enable_barycentric_analysis']:
        print("│   ├── barycentric_all_trajectories.png")
        print("│   ├── barycentric_individual_trajectories.png")
        print("│   └── barycentric_final_distributions.png")
    
    if params['enable_trajectory_analysis']:
        print("│   └── individual_trajectories.png")
    
    print("├── sequential_convergence_raw_data.csv")
    print("├── sequential_convergence_analysis.json")
    print("└── hypothesis_evaluation_report.txt")
    
    print("\nKey outputs explained:")
    print("- statistical_analysis.png: Comprehensive analysis of sequential patterns")
    print("- convergence_timeline_analysis.png: Temporal evolution of convergence events")
    print("- probability_evolution_analysis.png: How agent probabilities change over time")
    print("- system_dynamics_analysis.png: System-wide dynamics during convergence")
    if params['enable_barycentric_analysis']:
        print("- barycentric_*.png: Geometric analysis in probability space")
    print("- sequential_convergence_analysis.json: Statistical analysis results")
    print("- hypothesis_evaluation_report.txt: Detailed hypothesis testing results")


def main():
    """
    Main function to run the sequential convergence tutorial.
    """
    print("SEQUENTIAL CONVERGENCE STUDY TUTORIAL")
    print("=" * 80)
    print("This tutorial demonstrates how agents exhibit sequential convergence")
    print("patterns in multi-agent learning systems.")
    print("=" * 80)
    
    # Explain the concept
    explain_sequential_convergence_concept()
    
    # Get user parameters
    params = get_user_parameters()
    
    # Display output structure
    display_output_structure(params)
    
    # Confirm execution
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIRMATION")
    print("=" * 80)
    
    print(f"\nExperiment configuration:")
    print(f"- Number of agents: {params['num_agents']}")
    print(f"- Number of resources: {params['num_resources']}")
    print(f"- Number of iterations: {params['num_iterations']}")
    print(f"- Number of replications: {params['num_replications']}")
    print(f"- Learning rate: {params['weight']}")
    print(f"- Initial condition type: {params['initial_condition_type']}")
    print(f"- Barycentric analysis: {'Enabled' if params['enable_barycentric_analysis'] else 'Disabled'}")
    print(f"- Trajectory analysis: {'Enabled' if params['enable_trajectory_analysis'] else 'Disabled'}")
    print(f"- Output directory: {params['output_dir']}")
    
    confirm = input("\nProceed with experiment? (y/n, default: y): ").lower()
    if confirm == "n":
        print("Experiment cancelled.")
        return
    
    try:
        # Run the experiment
        study, results = run_sequential_convergence_experiment(params)
        
        # Generate analysis and plots
        generate_analysis_and_plots(study, params)
        
        print("\n" + "=" * 80)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults saved to: {params['output_dir']}")
        print("\nNext steps:")
        print("1. Examine the generated plots to understand sequential convergence patterns")
        print("2. Review the hypothesis evaluation report for statistical findings")
        print("3. Analyse the temporal structure of convergence events")
        print("4. Consider running additional experiments with different parameters")
        
    except Exception as e:
        print(f"\nError during experiment execution: {e}")
        print("Please check your parameters and try again.")


if __name__ == "__main__":
    main() 