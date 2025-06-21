"""
Capacity Ratio Study Tutorial

This tutorial demonstrates how to run the Capacity Ratio Study experiment,
which investigates how asymmetric resource capacities influence agent
specialisation patterns and system performance.

CONCEPT:
The capacity ratio study examines how different resource capacity configurations
affect agent learning dynamics and specialisation patterns. Asymmetric capacities
create hierarchical resource environments where high-capacity resources may
attract more agents, whilst low-capacity resources remain underutilised or
attract late convergers.

This experiment tests the hypothesis that capacity asymmetry drives predictable
agent specialisation patterns, with agents naturally organising themselves
according to resource capacity hierarchies through environmental feedback.
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

from experiments.capacity_ratio_study import CapacityRatioStudy, run_capacity_ratio_study
from utils.config import Config
from evaluation.metrics import calculate_entropy, calculate_convergence_speed
from visualisation.plots import plot_convergence_comparison


def explain_capacity_ratio_concept():
    """
    Explain the concept of capacity ratio effects in multi-agent learning.
    """
    print("=" * 80)
    print("CAPACITY RATIO STUDY - CONCEPT EXPLANATION")
    print("=" * 80)
    
    print("\n1. WHAT ARE CAPACITY RATIOS?")
    print("-" * 40)
    print("Capacity ratios refer to the relative capacity distribution across")
    print("available resources in the system. These ratios determine how much")
    print("load each resource can handle before experiencing congestion.")
    print()
    print("Symmetric capacities: [0.33, 0.33, 0.33] - Equal resource availability")
    print("Asymmetric capacities: [0.6, 0.3, 0.1] - Unequal resource availability")
    print("Extreme asymmetry: [0.8, 0.15, 0.05] - Highly hierarchical resources")
    
    print("\n2. WHY STUDY CAPACITY RATIOS?")
    print("-" * 40)
    print("- Understanding how resource asymmetry influences agent specialisation")
    print("- Identifying capacity-driven coordination patterns")
    print("- Exploring hierarchical resource utilisation dynamics")
    print("- Validating theoretical predictions about capacity effects")
    print("- Optimising system performance through capacity design")
    
    print("\n3. THEORETICAL FOUNDATION:")
    print("-" * 40)
    print("Asymmetric capacity configurations create natural hierarchies in")
    print("the resource environment. High-capacity resources provide better")
    print("performance and attract more agents, whilst low-capacity resources")
    print("may remain underutilised or serve as fallback options.")
    print()
    print("Capacity-driven specialisation can:")
    print("- Create predictable agent organisation patterns")
    print("- Improve system efficiency through strategic resource allocation")
    print("- Enable hierarchical coordination without explicit communication")
    print("- Optimise resource utilisation based on capacity constraints")
    
    print("\n4. KEY RESEARCH QUESTIONS:")
    print("-" * 40)
    print("- Do asymmetric capacities drive predictable specialisation patterns?")
    print("- How do capacity hierarchies influence agent convergence timing?")
    print("- What is the relationship between capacity asymmetry and performance?")
    print("- Can capacity design be used to engineer desired coordination?")
    print("- How do agents adapt to different capacity configurations?")
    
    print("\n5. EXPECTED OUTCOMES:")
    print("-" * 40)
    print("- High-capacity resources should attract more agents")
    print("- Capacity-utilisation correlations should be positive")
    print("- Asymmetric configurations should improve system performance")
    print("- Hierarchical specialisation patterns should emerge")
    print("- Convergence timing should reflect capacity preferences")


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
    num_replications = int(input("Number of replications (default: 50): ") or "50")
    
    # Learning parameters
    print("\n2. LEARNING PARAMETERS:")
    print("-" * 30)
    
    weight = float(input("Learning rate (weight) (default: 0.3): ") or "0.3")
    
    # Capacity configuration
    print("\n3. CAPACITY CONFIGURATION SELECTION:")
    print("-" * 40)
    print("Choose capacity configuration method:")
    print("1. Use all predefined configurations")
    print("2. Select specific predefined configurations")
    print("3. Create custom capacity configurations")
    print("4. Mix predefined and custom configurations")
    
    capacity_choice = input("Enter choice (1-4, default: 1): ") or "1"
    
    if capacity_choice == "1":
        capacity_configurations = "all_predefined"
    elif capacity_choice == "2":
        print("\nAvailable predefined configurations:")
        predefined_configs = [
            [0.33, 0.33, 0.33],  # Symmetric baseline
            [0.5, 0.3, 0.2],     # Moderate asymmetry 1
            [0.4, 0.4, 0.2],     # Moderate asymmetry 2
            [0.6, 0.3, 0.1],     # High asymmetry 1
            [0.7, 0.2, 0.1],     # High asymmetry 2
            [0.5, 0.25, 0.25],   # Single dominant
            [0.8, 0.15, 0.05],   # Extreme asymmetry
            [0.45, 0.45, 0.1],   # Two dominant
            [0.6, 0.25, 0.15],   # Graduated hierarchy
            [0.55, 0.35, 0.1]    # Strong binary
        ]
        for i, config in enumerate(predefined_configs, 1):
            print(f"{i:2d}. [{config[0]:.2f}, {config[1]:.2f}, {config[2]:.2f}]")
        
        selection = input("Enter configuration numbers (comma-separated, e.g., 1,3,5): ")
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        capacity_configurations = [predefined_configs[i] for i in indices if 0 <= i < len(predefined_configs)]
    elif capacity_choice == "3":
        capacity_configurations = "custom_only"
    else:  # capacity_choice == "4"
        capacity_configurations = "mixed"
    
    # Custom capacity configuration (if needed)
    custom_capacities = []
    if capacity_choice in ["3", "4"]:
        print("\n4. CUSTOM CAPACITY CONFIGURATIONS:")
        print("-" * 40)
        print("You can create custom capacity configurations in several ways:")
        print("1. Manual entry of capacity values")
        print("2. Generate configurations with specific asymmetry levels")
        print("3. Create configurations based on resource types")
        
        custom_choice = input("Choose custom configuration type (1-3): ")
        
        if custom_choice == "1":
            print(f"Enter capacity values for each configuration")
            print(f"Example for {num_resources} resources: 0.6,0.3,0.1")
            while True:
                probs_str = input("Capacity values (comma-separated, or 'done' to finish): ")
                if probs_str.lower() == 'done':
                    break
                capacities = [float(x.strip()) for x in probs_str.split(',')]
                # Normalise to sum to 1
                total = sum(capacities)
                capacities = [cap/total for cap in capacities]
                custom_capacities.append(capacities)
                
        elif custom_choice == "2":
            print("Generate configurations with specific asymmetry levels")
            num_configs = int(input("Number of configurations to generate (default: 5): ") or "5")
            asymmetry_levels = input("Asymmetry levels (comma-separated, e.g., 0.1,0.2,0.3): ")
            levels = [float(x.strip()) for x in asymmetry_levels.split(',')]
            
            for level in levels:
                # Create configuration with specified asymmetry
                base = (1.0 - level) / num_resources
                dominant = base + level
                config = [dominant] + [base] * (num_resources - 1)
                # Shuffle to create different patterns
                np.random.shuffle(config)
                custom_capacities.append(config)
                
        else:  # custom_choice == "3"
            print("Create configurations based on resource types")
            print("1. High-capacity primary resource")
            print("2. Balanced secondary resources")
            print("3. Low-capacity backup resources")
            
            primary_ratio = float(input("Primary resource capacity ratio (default: 0.6): ") or "0.6")
            secondary_ratio = float(input("Secondary resource capacity ratio (default: 0.3): ") or "0.3")
            backup_ratio = 1.0 - primary_ratio - secondary_ratio
            
            if backup_ratio >= 0:
                custom_capacities.append([primary_ratio, secondary_ratio, backup_ratio])
            else:
                print("Invalid ratios - using default")
                custom_capacities.append([0.6, 0.3, 0.1])
    
    # Convergence parameters
    print("\n5. CONVERGENCE PARAMETERS:")
    print("-" * 30)
    convergence_threshold_entropy = float(input("Convergence entropy threshold (default: 0.1): ") or "0.1")
    convergence_threshold_max_prob = float(input("Convergence max probability threshold (default: 0.9): ") or "0.9")
    
    # Output configuration
    print("\n6. OUTPUT CONFIGURATION:")
    print("-" * 30)
    output_dir = input("Output directory (default: results/capacity_ratio_tutorial): ") or "results/capacity_ratio_tutorial"
    show_plots = input("Show plots interactively? (y/n, default: n): ").lower() == 'y'
    
    # Create parameter dictionary
    params = {
        'num_agents': num_agents,
        'num_resources': num_resources,
        'num_iterations': num_iterations,
        'num_replications': num_replications,
        'weight': weight,
        'capacity_configurations': capacity_configurations,
        'custom_capacities': custom_capacities,
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
    print(f"Capacity configuration selection: {capacity_configurations}")
    if custom_capacities:
        print(f"Custom capacities: {len(custom_capacities)} defined")
        for i, config in enumerate(custom_capacities):
            print(f"  Configuration {i+1}: {[f'{c:.3f}' for c in config]}")
    print(f"Convergence entropy threshold: {convergence_threshold_entropy}")
    print(f"Convergence max probability threshold: {convergence_threshold_max_prob}")
    print(f"Output directory: {output_dir}")
    print(f"Show plots: {show_plots}")
    
    confirm = input("\nProceed with this configuration? (y/n, default: y): ").lower()
    if confirm == 'n':
        print("Configuration cancelled. Please run the tutorial again.")
        return None
    
    return params


def run_capacity_ratio_experiment(params):
    """
    Run the capacity ratio study experiment with the given parameters.
    """
    print("\n" + "=" * 80)
    print("RUNNING CAPACITY RATIO STUDY EXPERIMENT")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(params['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up capacity configurations based on user selection
    if params['capacity_configurations'] == "all_predefined":
        # Use all predefined configurations
        capacity_configs = [
            [0.33, 0.33, 0.33],  # Symmetric baseline
            [0.5, 0.3, 0.2],     # Moderate asymmetry 1
            [0.4, 0.4, 0.2],     # Moderate asymmetry 2
            [0.6, 0.3, 0.1],     # High asymmetry 1
            [0.7, 0.2, 0.1],     # High asymmetry 2
            [0.5, 0.25, 0.25],   # Single dominant
            [0.8, 0.15, 0.05],   # Extreme asymmetry
            [0.45, 0.45, 0.1],   # Two dominant
            [0.6, 0.25, 0.15],   # Graduated hierarchy
            [0.55, 0.35, 0.1]    # Strong binary
        ]
    elif params['capacity_configurations'] == "custom_only":
        # Use only custom configurations
        capacity_configs = params['custom_capacities']
    elif params['capacity_configurations'] == "mixed":
        # Use predefined + custom configurations
        capacity_configs = [
            [0.33, 0.33, 0.33],  # Symmetric baseline
            [0.5, 0.3, 0.2],     # Moderate asymmetry 1
            [0.4, 0.4, 0.2],     # Moderate asymmetry 2
            [0.6, 0.3, 0.1],     # High asymmetry 1
            [0.7, 0.2, 0.1],     # High asymmetry 2
            [0.5, 0.25, 0.25],   # Single dominant
            [0.8, 0.15, 0.05],   # Extreme asymmetry
            [0.45, 0.45, 0.1],   # Two dominant
            [0.6, 0.25, 0.15],   # Graduated hierarchy
            [0.55, 0.35, 0.1]    # Strong binary
        ]
        capacity_configs.extend(params['custom_capacities'])
    else:
        # Use specific predefined configurations
        capacity_configs = params['capacity_configurations']
    
    # Create and run the experiment
    print(f"Creating CapacityRatioStudy...")
    study = CapacityRatioStudy(
        capacity_configurations=capacity_configs,
        results_dir=params['output_dir'],
        experiment_name="capacity_ratio_tutorial"
    )
    
    # Override base config with user parameters
    study.base_config.num_agents = params['num_agents']
    study.base_config.num_resources = params['num_resources']
    study.base_config.num_iterations = params['num_iterations']
    study.base_config.weight = params['weight']
    
    print(f"Running experiment with {len(capacity_configs)} capacity configurations...")
    print("Capacity configurations:")
    for i, config in enumerate(capacity_configs):
        print(f"  {i+1}. [{config[0]:.2f}, {config[1]:.2f}, {config[2]:.2f}]")
    print(f"Replications per configuration: {params['num_replications']}")
    print("This may take several minutes depending on the number of configurations and replications...")
    
    # Run experiment
    full_results = study.run_experiment(num_episodes=params['num_replications'])
    
    # Convert results to expected format
    study.results = []
    converted_results = []
    for config_result in full_results['results']:
        config_params = config_result['config_params']
        for episode_result in config_result['episode_results']:
            converted_result = {
                'config_params': config_params,
                'simulation_results': episode_result,
                'replication_id': episode_result['episode']
            }
            study.results.append(converted_result)
            converted_results.append(converted_result)
    
    print("Experiment completed successfully!")
    
    return study, converted_results


def generate_analysis_and_plots(study, converted_results, params):
    """
    Generate comprehensive analysis and create all figures.
    """
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS AND PLOTS")
    print("=" * 80)
    
    # Temporarily set converted_results for analysis
    study.converted_results = converted_results
    
    # Generate detailed analysis
    print("Generating detailed statistical analysis...")
    analysis_results = study.perform_comprehensive_analysis()
    
    # Create comprehensive plots
    print("Creating comprehensive visualisations...")
    plots_dir = f"{params['output_dir']}/plots"
    plot_files = study.create_comprehensive_plots(plots_dir)
    
    print(f"Generated {len(plot_files)} plot files:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    
    # Save analysis results
    print("Saving analysis results...")
    study.save_analysis_results(params['output_dir'])
    
    # Generate hypothesis report
    print("Generating hypothesis evaluation report...")
    report = study.generate_capacity_report()
    
    # Save report
    report_path = f"{params['output_dir']}/capacity_ratio_hypothesis_report.txt"
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
        
        evidence = hyp_support.get('evidence_strength', {})
        if 'capacity_correlation' in evidence:
            corr_info = evidence['capacity_correlation']
            print(f"Capacity-Utilisation Correlation: {corr_info['value']:.3f} ({corr_info['strength']})")
        
        if 'hierarchy_consistency' in evidence:
            hier_info = evidence['hierarchy_consistency']
            print(f"Hierarchy Consistency: {hier_info['value']:.3f} ({hier_info['strength']})")
        
        if 'specialisation_strength' in evidence:
            spec_info = evidence['specialisation_strength']
            print(f"Specialisation Index: {spec_info['value']:.3f} ({spec_info['strength']})")
    
    if 'performance_analysis' in analysis_results:
        perf_analysis = analysis_results['performance_analysis']
        if 'total_costs' in perf_analysis:
            cost_data = perf_analysis['total_costs']
            best_config = min(cost_data.keys(), key=lambda k: np.mean(cost_data[k]))
            best_cost = np.mean(cost_data[best_config])
            print(f"\nBest performing configuration: {best_config.replace('_', ', ')}")
            print(f"  - Mean cost: {best_cost:.3f}")
    
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
    print("│   ├── capacity_correlation_analysis.png")
    print("│   ├── hierarchy_analysis.png")
    print("│   ├── performance_analysis.png")
    print("│   ├── ternary_specialisation_comparison.png")
    print("│   └── statistical_summary.png")
    print("├── capacity_ratio_raw_data.csv")
    print("├── analysis_results.json")
    print("└── capacity_ratio_hypothesis_report.txt")
    
    print("\nFile descriptions:")
    print("- plots/: All generated visualisations")
    print("- capacity_ratio_raw_data.csv: Raw experimental data")
    print("- analysis_results.json: Statistical analysis results")
    print("- capacity_ratio_hypothesis_report.txt: Detailed hypothesis evaluation")
    
    print("\nKey plots explained:")
    print("- capacity_correlation_analysis.png: Capacity-utilisation correlation analysis")
    print("- hierarchy_analysis.png: Hierarchical specialisation patterns")
    print("- performance_analysis.png: System performance across configurations")
    print("- ternary_specialisation_comparison.png: Agent specialisation in ternary space")
    print("- statistical_summary.png: Statistical tests and hypothesis evaluation")


def main():
    """
    Main tutorial function.
    """
    print("CAPACITY RATIO STUDY TUTORIAL")
    print("=" * 80)
    print("This tutorial demonstrates the Capacity Ratio Study experiment.")
    print("You will learn about capacity-driven specialisation and generate")
    print("comprehensive analysis with customisable parameters.")
    
    # Step 1: Explain the concept
    explain_capacity_ratio_concept()
    
    # Step 2: Get user parameters
    params = get_user_parameters()
    if params is None:
        return
    
    # Step 3: Run the experiment
    study, converted_results = run_capacity_ratio_experiment(params)
    
    # Step 4: Generate analysis and plots
    analysis_results, plot_files = generate_analysis_and_plots(study, converted_results, params)
    
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