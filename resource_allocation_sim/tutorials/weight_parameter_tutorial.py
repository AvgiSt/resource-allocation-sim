#!/usr/bin/env python3
"""
Weight Parameter Study Tutorial

This tutorial demonstrates how to run the Weight Parameter Study experiment,
which investigates how the learning rate parameter affects agent learning
dynamics, convergence speed, and system performance.

CONCEPT:
The weight parameter (w) controls the learning intensity in the stochastic
learning algorithm. Higher weights lead to faster convergence but may result
in suboptimal final performance due to reduced exploration. Lower weights
provide more thorough exploration but slower convergence.

This experiment tests the hypothesis that there exists an optimal weight
range that balances convergence speed with system performance, and that
the weight parameter significantly influences the exploration-exploitation
trade-off in multi-agent learning.
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

from experiments.weight_parameter_study import WeightParameterStudy, run_weight_parameter_study
from utils.config import Config
from evaluation.metrics import calculate_entropy, calculate_convergence_speed
from visualisation.plots import plot_convergence_comparison


def explain_weight_parameter_concept():
    """
    Explain the concept of weight parameter effects in multi-agent learning.
    """
    print("=" * 80)
    print("WEIGHT PARAMETER STUDY - CONCEPT EXPLANATION")
    print("=" * 80)
    
    print("\n1. WHAT IS THE WEIGHT PARAMETER?")
    print("-" * 40)
    print("The weight parameter (w) controls the learning intensity in the")
    print("stochastic learning algorithm. It determines how strongly agents")
    print("update their probability distributions based on environmental feedback.")
    print()
    print("Learning intensity: λ(t) = w × L(t)")
    print("where L(t) is the cost of the selected resource at time t")
    print()
    print("Weight range: w ∈ (0, 1)")
    print("- Low weights (w < 0.1): Conservative learning, slow convergence")
    print("- Moderate weights (0.1 ≤ w ≤ 0.5): Balanced learning")
    print("- High weights (w > 0.5): Aggressive learning, fast convergence")
    
    print("\n2. WHY STUDY THE WEIGHT PARAMETER?")
    print("-" * 40)
    print("- Understanding the exploration-exploitation trade-off")
    print("- Identifying optimal learning rates for different scenarios")
    print("- Quantifying the speed-performance relationship")
    print("- Validating theoretical predictions about convergence rates")
    print("- Providing practical guidelines for parameter selection")
    
    print("\n3. THEORETICAL FOUNDATION:")
    print("-" * 40)
    print("The weight parameter directly influences the learning dynamics:")
    print("- Higher weights increase learning intensity λ(t)")
    print("- Increased intensity leads to faster probability updates")
    print("- Faster updates result in quicker convergence")
    print("- However, excessive intensity may cause overshooting")
    print("- Overshooting can lead to suboptimal final performance")
    print()
    print("The exploration-exploitation trade-off:")
    print("- Low weights: More exploration, slower convergence, better final performance")
    print("- High weights: Less exploration, faster convergence, potentially worse performance")
    print("- Optimal weights: Balance between speed and quality")
    
    print("\n4. KEY RESEARCH QUESTIONS:")
    print("-" * 40)
    print("- How does the weight parameter affect convergence speed?")
    print("- What is the relationship between weight and final system performance?")
    print("- Is there an optimal weight range for different scenarios?")
    print("- How does weight influence the exploration-exploitation balance?")
    print("- Can weight selection compensate for poor initial conditions?")
    
    print("\n5. EXPECTED OUTCOMES:")
    print("-" * 40)
    print("- Higher weights should lead to faster convergence")
    print("- Lower weights should achieve better final performance")
    print("- There should be an optimal weight range for balanced performance")
    print("- Weight effects should be consistent across different scenarios")
    print("- The exploration-exploitation trade-off should be clearly observable")


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
    num_resources = int(input("Number of resources (default: 5): ") or "5")
    num_iterations = int(input("Number of iterations (default: 1000): ") or "1000")
    num_replications = int(input("Number of replications (default: 30): ") or "30")
    
    # Weight parameter configuration
    print("\n2. WEIGHT PARAMETER CONFIGURATION:")
    print("-" * 40)
    print("Choose weight parameter selection method:")
    print("1. Use predefined weight range (0.01 to 0.95)")
    print("2. Custom weight values")
    print("3. Specific weight range")
    print("4. Focus on optimal range (0.1 to 0.5)")
    
    weight_choice = input("Enter choice (1-4, default: 1): ") or "1"
    
    if weight_choice == "1":
        weight_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    elif weight_choice == "2":
        print("Enter custom weight values (comma-separated, e.g., 0.1,0.2,0.3):")
        weight_str = input("Weight values: ")
        weight_values = [float(x.strip()) for x in weight_str.split(',')]
    elif weight_choice == "3":
        min_weight = float(input("Minimum weight (default: 0.01): ") or "0.01")
        max_weight = float(input("Maximum weight (default: 0.95): ") or "0.95")
        num_weights = int(input("Number of weight values (default: 10): ") or "10")
        weight_values = np.linspace(min_weight, max_weight, num_weights).tolist()
    else:  # weight_choice == "4"
        weight_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
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
        print(f"Example for {num_resources} resources: 0.2,0.2,0.2,0.2,0.2")
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
    
    # Output configuration
    print("\n6. OUTPUT CONFIGURATION:")
    print("-" * 30)
    
    output_dir = input("Output directory (default: results/weight_parameter_tutorial): ") or "results/weight_parameter_tutorial"
    show_plots = input("Show plots interactively? (y/n, default: y): ").lower() != "n"
    
    # Create parameters dictionary
    params = {
        'num_agents': num_agents,
        'num_resources': num_resources,
        'num_iterations': num_iterations,
        'num_replications': num_replications,
        'weight_values': weight_values,
        'relative_capacity': relative_capacity,
        'initial_condition_type': initial_condition_type,
        'convergence_entropy_threshold': convergence_entropy_threshold,
        'convergence_max_prob_threshold': convergence_max_prob_threshold,
        'output_dir': output_dir,
        'show_plots': show_plots
    }
    
    return params


def run_weight_parameter_experiment(params):
    """
    Run the weight parameter study experiment with the given parameters.
    """
    print("\n" + "=" * 80)
    print("RUNNING WEIGHT PARAMETER STUDY EXPERIMENT")
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
    
    # Run the experiment
    print(f"\nRunning experiment with {len(params['weight_values'])} weight values...")
    print(f"Weight values: {params['weight_values']}")
    print(f"Number of replications per weight: {params['num_replications']}")
    print(f"Total experiments: {len(params['weight_values']) * params['num_replications']}")
    
    # Create and run the study
    study = WeightParameterStudy(
        weight_values=params['weight_values'],
        base_config=config,
        results_dir=params['output_dir'],
        experiment_name="weight_parameter_tutorial"
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
    
    # Create comprehensive plots using the study's method
    print("Creating comprehensive visualisations...")
    plot_files = study.create_comprehensive_plots(str(plots_path))
    
    print(f"Generated {len(plot_files)} plot files:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    
    # Generate detailed analysis
    print("\nGenerating detailed analysis...")
    analysis_results = study.generate_detailed_analysis()
    
    # Save analysis results
    analysis_path = output_path / "weight_parameter_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    report = study.generate_comprehensive_report()
    
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
    
    # Extract key findings from analysis
    if 'optimal_values' in analysis_results:
        optimal_values = analysis_results['optimal_values']
        print(f"\nOptimal weight values by metric:")
        for metric, optimal in optimal_values.items():
            print(f"  {metric}: {optimal['parameter_value']} (value: {optimal['metric_value']:.3f})")
    
    if 'trends' in analysis_results:
        trends = analysis_results['trends']
        print(f"\nKey trends:")
        for metric, trend in trends.items():
            if 'trend' in trend:
                print(f"  {metric}: {trend['trend']}")
    
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
    print("│   ├── convergence_times_vs_weight.png")
    print("│   ├── costs_vs_weight.png")
    print("│   ├── final_entropy_vs_weight.png")
    print("│   ├── cost_convergence_tradeoff.png")
    print("│   ├── stability_performance_tradeoff.png")
    print("│   ├── performance_distributions.png")
    print("│   ├── performance_heatmap.png")
    print("│   └── system_behaviour_radar.png")
    print("├── weight_parameter_raw_data.csv")
    print("├── weight_parameter_analysis.json")
    print("└── hypothesis_evaluation_report.txt")
    
    print("\nKey outputs explained:")
    print("- convergence_times_vs_weight.png: Shows how weight affects convergence speed")
    print("- costs_vs_weight.png: Shows how weight affects final system performance")
    print("- cost_convergence_tradeoff.png: Visualises the fundamental trade-off")
    print("- performance_heatmap.png: Comprehensive view of all performance metrics")
    print("- weight_parameter_analysis.json: Statistical analysis results")
    print("- hypothesis_evaluation_report.txt: Detailed hypothesis testing results")


def main():
    """
    Main function to run the weight parameter tutorial.
    """
    print("WEIGHT PARAMETER STUDY TUTORIAL")
    print("=" * 80)
    print("This tutorial demonstrates how the weight parameter affects")
    print("multi-agent learning dynamics and system performance.")
    print("=" * 80)
    
    # Explain the concept
    explain_weight_parameter_concept()
    
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
    print(f"- Weight values: {params['weight_values']}")
    print(f"- Total experiments: {len(params['weight_values']) * params['num_replications']}")
    print(f"- Output directory: {params['output_dir']}")
    
    confirm = input("\nProceed with experiment? (y/n, default: y): ").lower()
    if confirm == "n":
        print("Experiment cancelled.")
        return
    
    try:
        # Run the experiment
        study, results = run_weight_parameter_experiment(params)
        
        # Generate analysis and plots
        generate_analysis_and_plots(study, params)
        
        print("\n" + "=" * 80)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults saved to: {params['output_dir']}")
        print("\nNext steps:")
        print("1. Examine the generated plots to understand weight parameter effects")
        print("2. Review the hypothesis evaluation report for statistical findings")
        print("3. Use the analysis results to select optimal weight parameters")
        print("4. Consider running additional experiments with different configurations")
        
    except Exception as e:
        print(f"\nError during experiment execution: {e}")
        print("Please check your parameters and try again.")


if __name__ == "__main__":
    main() 