#!/usr/bin/env python3
"""
Initial Conditions Study Tutorial

This tutorial demonstrates how to run the Initial Conditions Study experiment,
which investigates how biased initial probability distributions affect agent
learning dynamics and system performance compared to uniform initialisation.

CONCEPT:
The initial probability distribution determines how agents start their learning
process. Uniform initialisation (equal probabilities for all resources) represents
maximum uncertainty, whilst biased distributions provide agents with initial
preferences that may accelerate convergence and improve system performance.

This experiment tests the hypothesis that strategic initial positioning can
enhance coordination and reduce the time required to achieve optimal resource
allocation through basin of attraction effects in the learning dynamics.
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

from experiments.initial_conditions_study import InitialConditionsStudy, run_initial_conditions_study
from utils.config import Config
from evaluation.metrics import calculate_entropy, calculate_convergence_speed
from visualisation.plots import plot_convergence_comparison


def explain_initial_conditions_concept():
    """
    Explain the concept of initial conditions effects in multi-agent learning.
    """
    print("=" * 80)
    print("INITIAL CONDITIONS STUDY - CONCEPT EXPLANATION")
    print("=" * 80)
    
    print("\n1. WHAT ARE INITIAL CONDITIONS?")
    print("-" * 40)
    print("Initial conditions refer to the starting probability distributions")
    print("that agents use when beginning their learning process. These")
    print("distributions determine how uncertain or certain agents are about")
    print("their resource preferences at the start of the experiment.")
    print()
    print("Uniform initialisation: [0.333, 0.333, 0.333] - Maximum uncertainty")
    print("Biased initialisation: [0.6, 0.3, 0.1] - Initial preference for resource 1")
    print("Vertex initialisation: [0.9, 0.05, 0.05] - Strong preference for resource 1")
    
    print("\n2. WHY STUDY INITIAL CONDITIONS?")
    print("-" * 40)
    print("- Understanding basin of attraction effects in learning dynamics")
    print("- Identifying strategic initial positioning for improved performance")
    print("- Reducing convergence time through intelligent initialisation")
    print("- Exploring the relationship between initial uncertainty and final outcomes")
    print("- Validating theoretical predictions about learning pathway influence")
    
    print("\n3. THEORETICAL FOUNDATION:")
    print("-" * 40)
    print("The learning dynamics create basins of attraction around different")
    print("equilibria in the probability space. Initial positioning determines")
    print("which basin an agent will converge to, influencing both the speed")
    print("and quality of convergence.")
    print()
    print("Strategic initial conditions can:")
    print("- Guide agents toward favourable equilibria")
    print("- Reduce exploration overhead in the learning process")
    print("- Accelerate coordination through reduced initial uncertainty")
    print("- Improve system-wide performance through better resource allocation")
    
    print("\n4. KEY RESEARCH QUESTIONS:")
    print("-" * 40)
    print("- Do biased initial conditions improve system performance?")
    print("- How do different initial conditions affect convergence speed?")
    print("- What is the optimal initial positioning for different scenarios?")
    print("- How do basin of attraction effects influence learning outcomes?")
    print("- Can strategic initialisation compensate for poor learning parameters?")
    
    print("\n5. EXPECTED OUTCOMES:")
    print("-" * 40)
    print("- Biased initial conditions should improve performance over uniform")
    print("- Vertex-proximate conditions should achieve fastest convergence")
    print("- Strategic positioning should reduce convergence time")
    print("- Initial conditions should influence final resource allocation patterns")
    print("- Basin of attraction effects should be observable in convergence pathways")


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
    num_iterations = int(input("Number of iterations (default: 1000): ") or "1000")
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
    print("Choose initial condition selection method:")
    print("1. Use all predefined conditions")
    print("2. Select specific predefined conditions")
    print("3. Create custom initial conditions")
    print("4. Combination of predefined and custom")
    
    condition_choice = input("Enter choice (1-4, default: 1): ") or "1"
    
    if condition_choice == "1":
        initial_condition_types = "all_predefined"
    elif condition_choice == "2":
        print("\nAvailable predefined conditions:")
        predefined_conditions = [
            'uniform', 'diagonal_point_1', 'diagonal_point_2', 'diagonal_point_3',
            'diagonal_point_4', 'diagonal_point_5', 'diagonal_point_6', 'diagonal_point_7',
            'diagonal_point_8', 'diagonal_point_9', 'diagonal_point_10', 'edge_bias_12'
        ]
        for i, condition in enumerate(predefined_conditions, 1):
            print(f"{i:2d}. {condition}")
        
        selection = input("Enter condition numbers (comma-separated, e.g., 1,3,5): ")
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        initial_condition_types = [predefined_conditions[i] for i in indices if 0 <= i < len(predefined_conditions)]
    elif condition_choice == "3":
        initial_condition_types = "custom_only"
    else:  # condition_choice == "4"
        initial_condition_types = "mixed"
    
    # Custom condition configuration (if needed)
    custom_conditions = []
    if condition_choice in ["3", "4"]:
        print("\n5. CUSTOM INITIAL CONDITIONS:")
        print("-" * 40)
        print("You can create custom initial conditions in several ways:")
        print("1. Fixed distribution for all agents")
        print("2. Different distributions for each agent")
        print("3. Random distributions within specified ranges")
        
        custom_choice = input("Choose custom condition type (1-3): ")
        
        if custom_choice == "1":
            print(f"Enter probability distribution for all agents (must sum to 1.0)")
            print(f"Example for {num_resources} resources: 0.5,0.3,0.2")
            probs_str = input("Probabilities (comma-separated): ")
            probs = [float(x.strip()) for x in probs_str.split(',')]
            # Normalise to sum to 1
            total = sum(probs)
            probs = [p/total for p in probs]
            custom_conditions.append(("custom_fixed", probs))
            
        elif custom_choice == "2":
            print(f"Enter distributions for each of {num_agents} agents")
            for i in range(num_agents):
                probs_str = input(f"Agent {i} probabilities (comma-separated): ")
                probs = [float(x.strip()) for x in probs_str.split(',')]
                total = sum(probs)
                probs = [p/total for p in probs]
                custom_conditions.append((f"custom_agent_{i}", probs))
                
        else:  # custom_choice == "3"
            print("Specify random range for each resource")
            min_vals_str = input("Minimum values (comma-separated): ")
            max_vals_str = input("Maximum values (comma-separated): ")
            min_vals = [float(x.strip()) for x in min_vals_str.split(',')]
            max_vals = [float(x.strip()) for x in max_vals_str.split(',')]
            custom_conditions.append(("custom_random", (min_vals, max_vals)))
    
    # Convergence parameters
    print("\n6. CONVERGENCE PARAMETERS:")
    print("-" * 30)
    convergence_threshold_entropy = float(input("Convergence entropy threshold (default: 0.1): ") or "0.1")
    convergence_threshold_max_prob = float(input("Convergence max probability threshold (default: 0.9): ") or "0.9")
    
    # Output configuration
    print("\n7. OUTPUT CONFIGURATION:")
    print("-" * 30)
    output_dir = input("Output directory (default: results/initial_conditions_tutorial): ") or "results/initial_conditions_tutorial"
    show_plots = input("Show plots interactively? (y/n, default: n): ").lower() == 'y'
    
    # Create parameter dictionary
    params = {
        'num_agents': num_agents,
        'num_resources': num_resources,
        'num_iterations': num_iterations,
        'num_replications': num_replications,
        'weight': weight,
        'relative_capacity': relative_capacity,
        'initial_condition_types': initial_condition_types,
        'custom_conditions': custom_conditions,
        'convergence_threshold_entropy': convergence_threshold_entropy,
        'convergence_threshold_max_prob': convergence_threshold_max_prob,
        'output_dir': output_dir,
        'show_plots': show_plots
    }
    
    # Display configuration
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Number of agents: {num_agents}")
    print(f"Number of resources: {num_resources}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Number of replications: {num_replications}")
    print(f"Learning rate (weight): {weight}")
    print(f"Relative capacity: {[f'{c:.3f}' for c in relative_capacity]}")
    print(f"Initial condition selection: {initial_condition_types}")
    if custom_conditions:
        print(f"Custom conditions: {len(custom_conditions)} defined")
    print(f"Convergence entropy threshold: {convergence_threshold_entropy}")
    print(f"Convergence max probability threshold: {convergence_threshold_max_prob}")
    print(f"Output directory: {output_dir}")
    print(f"Show plots: {show_plots}")
    
    confirm = input("\nProceed with this configuration? (y/n, default: y): ").lower()
    if confirm == 'n':
        print("Configuration cancelled. Please run the tutorial again.")
        return None
    
    return params


def run_initial_conditions_experiment(params):
    """
    Run the initial conditions study experiment with the given parameters.
    """
    print("\n" + "=" * 80)
    print("RUNNING INITIAL CONDITIONS STUDY EXPERIMENT")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(params['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create custom configuration
    config = Config()
    config.num_agents = params['num_agents']
    config.num_resources = params['num_resources']
    config.relative_capacity = params['relative_capacity']
    config.num_iterations = params['num_iterations']
    config.weight = params['weight']
    
    # Create and run the experiment
    print(f"Creating InitialConditionsStudy...")
    study = InitialConditionsStudy(
        results_dir=params['output_dir'],
        experiment_name="initial_conditions_tutorial"
    )
    
    # Override base config with user parameters
    study.base_config = config
    
    # Set up initial conditions based on user selection
    if params['initial_condition_types'] == "all_predefined":
        # Use all predefined conditions
        study.initial_condition_types = [
            'uniform', 'diagonal_point_1', 'diagonal_point_2', 'diagonal_point_3',
            'diagonal_point_4', 'diagonal_point_5', 'diagonal_point_6', 'diagonal_point_7',
            'diagonal_point_8', 'diagonal_point_9', 'diagonal_point_10', 'edge_bias_12'
        ]
    elif params['initial_condition_types'] == "custom_only":
        # Use only custom conditions
        study.initial_condition_types = []
        for condition_name, condition_data in params['custom_conditions']:
            study.initial_condition_types.append(condition_name)
    elif params['initial_condition_types'] == "mixed":
        # Use predefined + custom conditions
        study.initial_condition_types = [
            'uniform', 'diagonal_point_1', 'diagonal_point_2', 'diagonal_point_3',
            'diagonal_point_4', 'diagonal_point_5', 'diagonal_point_6', 'diagonal_point_7',
            'diagonal_point_8', 'diagonal_point_9', 'diagonal_point_10', 'edge_bias_12'
        ]
        for condition_name, condition_data in params['custom_conditions']:
            study.initial_condition_types.append(condition_name)
    else:
        # Use specific predefined conditions
        study.initial_condition_types = params['initial_condition_types']
    
    # Override generate_initial_probabilities method to handle custom conditions
    if params['custom_conditions']:
        original_method = study.generate_initial_probabilities
        
        # Store custom condition data in a closure
        custom_condition_data = {}
        for condition_name, condition_data in params['custom_conditions']:
            custom_condition_data[condition_name] = condition_data
        
        def enhanced_method(init_type: str, num_agents: int) -> List[List[float]]:
            if init_type in custom_condition_data:
                condition_data = custom_condition_data[init_type]
                if init_type == "custom_fixed":
                    return [condition_data.copy() for _ in range(num_agents)]
                elif init_type.startswith("custom_agent_"):
                    # For individual agent conditions, return the specific agent's distribution
                    agent_id = int(init_type.split("_")[-1])
                    if agent_id < num_agents:
                        return [condition_data if i == agent_id else [1.0/params['num_resources']] * params['num_resources'] for i in range(num_agents)]
                    else:
                        return [[1.0/params['num_resources']] * params['num_resources'] for _ in range(num_agents)]
                elif init_type == "custom_random":
                    min_vals, max_vals = condition_data
                    probs_list = []
                    for _ in range(num_agents):
                        probs = [np.random.uniform(min_vals[i], max_vals[i]) for i in range(params['num_resources'])]
                        total = sum(probs)
                        probs = [p/total for p in probs]
                        probs_list.append(probs)
                    return probs_list
            # Always return a valid result
            return original_method(init_type, num_agents)
        
        # Override the method
        study.generate_initial_probabilities = enhanced_method
    
    print(f"Running experiment with {len(study.initial_condition_types)} initial conditions...")
    print(f"Conditions: {study.initial_condition_types}")
    print(f"Replications per condition: {params['num_replications']}")
    print("This may take several minutes depending on the number of conditions and replications...")
    
    # Run experiment
    full_results = study.run_experiment(num_episodes=params['num_replications'])
    
    # Convert results to expected format
    study.results = []
    for config_result in full_results['results']:
        config_params = config_result['config_params']
        for episode_result in config_result['episode_results']:
            study.results.append({
                'config_params': config_params,
                'simulation_results': episode_result,
                'replication_id': episode_result['episode']
            })
    
    print("Experiment completed successfully!")
    
    return study


def generate_analysis_and_plots(study, params):
    """
    Generate comprehensive analysis and create all figures.
    """
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS AND PLOTS")
    print("=" * 80)
    
    # Generate detailed analysis
    print("Generating detailed statistical analysis...")
    analysis_results = study.analyse_results()
    
    # Create comprehensive plots
    print("Creating comprehensive visualisations...")
    plots_dir = f"{params['output_dir']}/plots"
    plot_files = study.create_comprehensive_plots(plots_dir, params['show_plots'])
    
    print(f"Generated {len(plot_files)} plot files:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    
    # Save analysis results
    print("Saving analysis results...")
    study.save_analysis_results(params['output_dir'])
    
    # Generate hypothesis report
    print("Generating hypothesis evaluation report...")
    report = study.generate_hypothesis_report()
    
    # Save report
    report_path = f"{params['output_dir']}/hypothesis_evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Hypothesis evaluation report saved to: {report_path}")
    
    # Display key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    if 'hypothesis_support' in analysis_results:
        hyp_support = analysis_results['hypothesis_support']
        overall_support = hyp_support.get('overall_support', 'unknown')
        print(f"Overall Hypothesis Support: {overall_support.upper()}")
        
        if 'condition_comparisons' in hyp_support:
            print("\nCondition Performance Comparison:")
            comparisons = hyp_support['condition_comparisons']
            for comparison in comparisons:
                print(f"  {comparison['condition']}: {comparison['performance']}")
    
    if 'performance_analysis' in analysis_results:
        perf_analysis = analysis_results['performance_analysis']
        if 'best_condition' in perf_analysis:
            best_condition = perf_analysis['best_condition']
            print(f"\nBest performing condition: {best_condition['name']}")
            print(f"  - Entropy: {best_condition['entropy']:.3f}")
            print(f"  - Cost: {best_condition['cost']:.3f}")
            print(f"  - Performance improvement: {best_condition['improvement']:.1f}%")
    
    return analysis_results, plot_files


def display_output_structure(params):
    """
    Display the output file structure and explain what each file contains.
    """
    print("\n" + "=" * 80)
    print("OUTPUT FILES AND STRUCTURE")
    print("=" * 80)
    
    output_path = Path(params['output_dir'])
    
    print(f"\nAll results are saved in: {output_path}")
    print("\nFile structure:")
    print(f"{output_path}/")
    print("├── plots/")
    print("│   ├── performance_comparison.png")
    print("│   ├── entropy_analysis.png")
    print("│   ├── barycentric_initial_positions.png")
    print("│   └── statistical_analysis.png")
    print("├── initial_conditions_raw_data.csv")
    print("├── initial_conditions_analysis.json")
    print("└── hypothesis_evaluation_report.txt")
    
    print("\nFile descriptions:")
    print("- plots/: All generated visualisations")
    print("- initial_conditions_raw_data.csv: Raw experimental data")
    print("- initial_conditions_analysis.json: Statistical analysis results")
    print("- hypothesis_evaluation_report.txt: Detailed hypothesis evaluation")
    
    print("\nKey plots explained:")
    print("- performance_comparison.png: Compares final entropy and cost across conditions")
    print("- entropy_analysis.png: Shows entropy evolution over time for each condition")
    print("- barycentric_initial_positions.png: Shows starting positions in probability space")
    print("- statistical_analysis.png: Statistical tests and hypothesis evaluation")


def main():
    """
    Main tutorial function.
    """
    print("INITIAL CONDITIONS STUDY TUTORIAL")
    print("=" * 80)
    print("This tutorial demonstrates the Initial Conditions Study experiment.")
    print("You will learn about initial condition effects and generate")
    print("comprehensive analysis with customisable parameters.")
    
    # Step 1: Explain the concept
    explain_initial_conditions_concept()
    
    # Step 2: Get user parameters
    params = get_user_parameters()
    if params is None:
        return
    
    # Step 3: Run the experiment
    study = run_initial_conditions_experiment(params)
    
    # Step 4: Generate analysis and plots
    analysis_results, plot_files = generate_analysis_and_plots(study, params)
    
    # Step 5: Display output structure
    display_output_structure(params)
    
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"All results saved in: {params['output_dir']}")
    print(f"Generated {len(plot_files)} plot files")
    print("\nYou can now:")
    print("- Examine the generated plots in the plots/ directory")
    print("- Review the hypothesis evaluation report for detailed findings")
    print("- Modify parameters and run again to explore different scenarios")
    print("- Use the raw data for further custom analysis")
    print("- Compare results with other experiments to understand system dynamics")


if __name__ == "__main__":
    main() 