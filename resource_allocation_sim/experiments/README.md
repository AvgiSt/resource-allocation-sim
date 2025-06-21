# Experiments Module

This module provides a unified interface for running hypothesis studies with consistent calling patterns.

## Overview

The experiments module has been designed with a **consistent single-file interface** for all hypothesis studies:

- **Hypothesis 1**: Weight Parameter Study → `weight_parameter_study.py`
- **Hypothesis 2**: Sequential Convergence Study → `sequential_convergence_study.py`

Each hypothesis study provides:
1. **Data generation** across parameter values
2. **Comprehensive analysis** with statistical tests
3. **Detailed visualisations** and plots  
4. **Performance evaluation** and recommendations
5. **Unified calling pattern** for ease of use

## Quick Start

### Option 1: Unified Interface (Recommended)

```python
from resource_allocation_sim.experiments import run_hypothesis_1_study, run_hypothesis_2_study

# Run Hypothesis 1: Weight Parameter Study
study1 = run_hypothesis_1_study(
    num_replications=50,
    output_dir="results/hypothesis1"
)

# Run Hypothesis 2: Sequential Convergence Study  
study2 = run_hypothesis_2_study(
    num_replications=30,
    num_iterations=1000,
    output_dir="results/hypothesis2"
)
```

### Option 2: Direct Module Import

```python
from resource_allocation_sim.experiments.weight_parameter_study import run_weight_parameter_study
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study

# Equivalent to Option 1
study1 = run_weight_parameter_study(num_replications=50)
study2 = run_sequential_convergence_study(num_replications=30)
```

### Option 3: Advanced Custom Configuration

```python
from resource_allocation_sim.experiments import WeightParameterStudy, SequentialConvergenceStudy

# Custom weight parameter study
weight_study = WeightParameterStudy(
    weight_values=[0.1, 0.3, 0.5, 0.7, 0.9],  # Custom weight values
    results_dir="custom_results"
)
weight_results = weight_study.run_experiment(num_episodes=100)

# Custom sequential convergence study
seq_study = SequentialConvergenceStudy(
    convergence_threshold_entropy=0.05,  # Stricter convergence criteria
    convergence_threshold_max_prob=0.95
)
seq_results = seq_study.run_experiment(num_episodes=50)
```

## Hypothesis Studies

### Hypothesis 1: Weight Parameter Study

**Research Question**: Does the weight parameter significantly affect convergence speed, system performance, and learning stability?

**Key Features**:
- Tests 10 weight values: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
- Comprehensive performance analysis with tradeoff visualisations
- Statistical evaluation of parameter effects
- Optimal parameter recommendations

**Usage**:
```python
from resource_allocation_sim.experiments import run_hypothesis_1_study

study = run_hypothesis_1_study(
    num_replications=50,        # Replications per weight value
    output_dir="results/h1",    # Output directory
    show_plots=False            # Save plots without display
)

# Access results
optimal_weight = study.analysis_results['recommendations']['optimal_for_cost']
print(f"Optimal weight for cost: {optimal_weight}")
```

**Output Structure**:
```
results/hypothesis1_weight_study_YYYYMMDD_HHMMSS/
├── plots/
│   ├── convergence_times_vs_weight.png
│   ├── costs_vs_weight.png
│   ├── performance_heatmap.png
│   ├── cost_convergence_tradeoff.png
│   └── [8 more visualisation files]
├── weight_study_raw_data.csv
├── weight_study_analysis.json
├── comprehensive_analysis_report.txt
└── [BaseExperiment standard files]
```

### Hypothesis 2: Sequential Convergence Study

**Research Question**: Do agents with uniform initial distribution sequentially converge to degenerate distributions?

**Key Features**:
- Tests sequential vs simultaneous convergence patterns
- Barycentric coordinate trajectory analysis (3 resources)
- Statistical tests for sequential ordering
- Degeneracy proportion measurement

**Usage**:
```python
from resource_allocation_sim.experiments import run_hypothesis_2_study

study = run_hypothesis_2_study(
    num_replications=30,                        # Independent simulation runs
    num_iterations=1000,                        # Iterations per simulation
    output_dir="results/h2",                    # Output directory
    convergence_threshold_entropy=0.1,          # Convergence detection threshold
    convergence_threshold_max_prob=0.9          # Degeneracy threshold
)

# Access results
support_level = study.analysis['hypothesis_support']['overall_support']
sequential_index = study.analysis['hypothesis_support']['evidence_strength']['sequential_index']
print(f"Hypothesis support: {support_level}")
print(f"Sequential index: {sequential_index:.3f}")
```

**Output Structure**:
```
results/hypothesis2_sequential_study_YYYYMMDD_HHMMSS/
├── plots/
│   ├── convergence_timeline.png
│   ├── probability_evolution.png
│   ├── system_dynamics.png
│   ├── statistical_analysis.png
│   ├── barycentric_trajectories.png
│   ├── barycentric_final_positions.png
│   └── barycentric_individual.png
├── experiment_results.json
├── analysis_results.json
└── [BaseExperiment standard files]
```

## Consistent Interface Design

### Common Parameters

All hypothesis studies support these common parameters:

```python
def run_hypothesis_X_study(
    num_replications: int,      # Number of independent runs
    output_dir: str,            # Results output directory  
    show_plots: bool = False    # Interactive vs batch plotting
) -> StudyInstance:
    """Returns completed study with full analysis."""
```

### Common Return Pattern

All studies return a completed instance with:

```python
study = run_hypothesis_X_study(...)

# Access experimental data
study.results                   # Raw simulation results
study.get_results_dir()        # Path to output directory

# Access analysis results  
study.analysis_results         # Comprehensive analysis (H1)
study.analysis                 # Analysis results (H2)

# Generate additional reports
study.generate_comprehensive_report()  # Text report (H1)
study.create_comprehensive_plots()     # Additional plots
```

### Common Output Structure

All studies create timestamped directories with:

```
results/
└── hypothesis_X_study_YYYYMMDD_HHMMSS/
    ├── plots/                     # Visualisation files (.png)
    ├── experiment_results.json    # Raw numerical results  
    ├── analysis_results.json      # Statistical analysis
    ├── summary_results.csv        # Summary data
    ├── metadata.json             # Experiment metadata
    └── [study-specific files]    # Additional reports/data
```

## Advanced Usage

### Custom Parameter Ranges

```python
# Hypothesis 1: Custom weight values
from resource_allocation_sim.experiments import WeightParameterStudy

study = WeightParameterStudy(
    weight_values=[0.1, 0.2, 0.3, 0.4, 0.5],  # Custom range
    results_dir="custom_weight_study"
)
results = study.run_experiment(num_episodes=30)

# Hypothesis 2: Custom convergence criteria
from resource_allocation_sim.experiments import SequentialConvergenceStudy

study = SequentialConvergenceStudy(
    convergence_threshold_entropy=0.05,     # Stricter entropy threshold
    convergence_threshold_max_prob=0.95,    # Higher degeneracy requirement
    results_dir="strict_convergence_study"
)
results = study.run_experiment(num_episodes=50)
```

### Batch Processing Multiple Configurations

```python
from resource_allocation_sim.experiments import run_hypothesis_1_study, run_hypothesis_2_study

# Run multiple parameter configurations
configurations = [
    {"num_replications": 30, "output_dir": "results/h1_quick"},
    {"num_replications": 100, "output_dir": "results/h1_detailed"}
]

h1_results = []
for config in configurations:
    study = run_hypothesis_1_study(**config)
    h1_results.append(study)

# Compare results across configurations
for i, study in enumerate(h1_results):
    optimal = study.analysis_results['recommendations']['optimal_for_cost']
    print(f"Config {i}: Optimal weight = {optimal}")
```

### Integration with Other Experiments

```python
from resource_allocation_sim.experiments import GridSearchExperiment, ParameterSweepExperiment

# Use hypothesis studies alongside other experiment types
grid_study = GridSearchExperiment(
    parameter_grid={
        'num_agents': [5, 10, 15],
        'weight': [0.3, 0.5, 0.7]
    }
)

# Or build custom studies using the base classes
class CustomHypothesisStudy(ParameterSweepExperiment):
    def __init__(self):
        super().__init__(
            parameter_name='custom_param',
            parameter_values=[1, 2, 3]
        )
    
    def analyse_results(self):
        # Custom analysis logic
        return {'custom_metric': 'custom_value'}
```

## Migration from Old Interface

### Before (Inconsistent)

```python
# Old Hypothesis 1 (required 2 steps)
from resource_allocation_sim.experiments.weight_parameter_study import run_weight_parameter_study
from resource_allocation_sim.experiments.hypothesis1_comprehensive_analysis import WeightParameterAnalyser

# Step 1: Generate data
study = run_weight_parameter_study(num_replications=50)

# Step 2: Analyse data  
analyser = WeightParameterAnalyser()
analyser.load_all_results()
analysis = analyser.run_comprehensive_analysis()

# Old Hypothesis 2 (single step)  
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study
study = run_sequential_convergence_study(num_replications=30)
```

### After (Consistent)

```python
# New unified interface (both single step)
from resource_allocation_sim.experiments import run_hypothesis_1_study, run_hypothesis_2_study

# Both hypotheses now have identical calling patterns
study1 = run_hypothesis_1_study(num_replications=50)    # Complete analysis included
study2 = run_hypothesis_2_study(num_replications=30)    # Complete analysis included
```

## Best Practices

### 1. Use Appropriate Replication Counts

```python
# Quick testing (development)
study = run_hypothesis_1_study(num_replications=10)

# Standard analysis (research)
study = run_hypothesis_1_study(num_replications=50)  

# High-precision (publication)
study = run_hypothesis_1_study(num_replications=100)
```

### 2. Organize Output Directories

```python
# Organized directory structure
study1 = run_hypothesis_1_study(output_dir="results/2024_study/hypothesis1")
study2 = run_hypothesis_2_study(output_dir="results/2024_study/hypothesis2")
```

### 3. Batch Processing for Reproducibility

```python
# Reproducible batch runs
import time

timestamp = int(time.time())
for i in range(3):  # Multiple runs for robustness
    study = run_hypothesis_1_study(
        num_replications=50,
        output_dir=f"results/batch_{timestamp}/run_{i+1}"
    )
```

### 4. Access Results Programmatically

```python
# Hypothesis 1: Performance metrics
study1 = run_hypothesis_1_study(num_replications=30)
recommendations = study1.analysis_results['recommendations']
print(f"Best weight: {recommendations['optimal_for_cost']}")
print(f"Cost range: {recommendations['suggested_range']}")

# Hypothesis 2: Statistical evidence
study2 = run_hypothesis_2_study(num_replications=30)  
support = study2.analysis['hypothesis_support']
print(f"Sequential support: {support['overall_support']}")
print(f"Evidence strength: {support['evidence_strength']}")
```

This unified interface ensures both hypotheses follow the same pattern: **one function call → complete analysis**. 