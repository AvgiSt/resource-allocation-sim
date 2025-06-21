# Capacity Ratio Study Tutorial

## Overview

This tutorial demonstrates how to run the **Capacity Ratio Study** experiment, which investigates how asymmetric resource capacities influence agent specialisation patterns and system performance.

## Concept Explanation

### What are Capacity Ratios?

Capacity ratios refer to the relative capacity distribution across available resources in the system. These ratios determine how much load each resource can handle before experiencing congestion.

**Examples:**
- **Symmetric capacities**: [0.33, 0.33, 0.33] - Equal resource availability
- **Asymmetric capacities**: [0.6, 0.3, 0.1] - Unequal resource availability
- **Extreme asymmetry**: [0.8, 0.15, 0.05] - Highly hierarchical resources

### Why Study Capacity Ratios?

1. **Understanding how resource asymmetry influences agent specialisation**
2. **Identifying capacity-driven coordination patterns**
3. **Exploring hierarchical resource utilisation dynamics**
4. **Validating theoretical predictions about capacity effects**
5. **Optimising system performance through capacity design**

### Theoretical Foundation

Asymmetric capacity configurations create natural hierarchies in the resource environment. High-capacity resources provide better performance and attract more agents, whilst low-capacity resources may remain underutilised or serve as fallback options.

Capacity-driven specialisation can:
- **Create predictable agent organisation patterns**
- **Improve system efficiency through strategic resource allocation**
- **Enable hierarchical coordination without explicit communication**
- **Optimise resource utilisation based on capacity constraints**

### Key Research Questions

1. Do asymmetric capacities drive predictable specialisation patterns?
2. How do capacity hierarchies influence agent convergence timing?
3. What is the relationship between capacity asymmetry and performance?
4. Can capacity design be used to engineer desired coordination?
5. How do agents adapt to different capacity configurations?

### Expected Outcomes

- High-capacity resources should attract more agents
- Capacity-utilisation correlations should be positive
- Asymmetric configurations should improve system performance
- Hierarchical specialisation patterns should emerge
- Convergence timing should reflect capacity preferences

## Usage Instructions

### Quick Start

```bash
# Run the tutorial with default parameters
python -m resource_allocation_sim.tutorials.capacity_ratio_tutorial
```

### Interactive Parameter Customisation

The tutorial allows you to customise all parameters:

#### 1. System Configuration
- **Number of agents**: Default 10
- **Number of resources**: Default 3 (optimal for ternary analysis)
- **Number of iterations**: Default 1000
- **Number of replications**: Default 50

#### 2. Learning Parameters
- **Learning rate (weight)**: Default 0.3

#### 3. Capacity Configuration Selection
Choose from four options:
- **All predefined configurations**: Test all 10 predefined configurations
- **Specific predefined configurations**: Select from available configurations
- **Custom configurations only**: Create your own capacity configurations
- **Mixed approach**: Combine predefined and custom configurations

#### 4. Custom Capacity Configurations
You can create custom capacity configurations in three ways:
- **Manual entry**: Specify capacity values directly
- **Asymmetry-based generation**: Generate configurations with specific asymmetry levels
- **Resource type-based**: Create configurations based on resource types (primary, secondary, backup)

#### 5. Convergence Parameters
- **Convergence entropy threshold**: Default 0.1
- **Convergence max probability threshold**: Default 0.9

#### 6. Output Configuration
- **Output directory**: Where to save results
- **Show plots**: Whether to display plots interactively

## Generated Outputs

### File Structure

```
results/capacity_ratio_tutorial/
├── plots/
│   ├── capacity_correlation_analysis.png
│   ├── hierarchy_analysis.png
│   ├── performance_analysis.png
│   ├── ternary_specialisation_comparison.png
│   └── statistical_summary.png
├── capacity_ratio_raw_data.csv
├── analysis_results.json
└── capacity_ratio_hypothesis_report.txt
```

### Key Plots Explained

#### 1. Capacity Correlation Analysis
- **capacity_correlation_analysis.png**: Capacity-utilisation correlation analysis
- Shows how resource capacity relates to agent utilisation patterns

#### 2. Hierarchy Analysis
- **hierarchy_analysis.png**: Hierarchical specialisation patterns
- Demonstrates how agents organise themselves according to capacity hierarchies

#### 3. Performance Analysis
- **performance_analysis.png**: System performance across configurations
- Compares cost, load balance, and convergence rates across different capacity setups

#### 4. Ternary Specialisation Comparison
- **ternary_specialisation_comparison.png**: Agent specialisation in ternary space
- Visualises agent positions in probability space for different capacity configurations

#### 5. Statistical Summary
- **statistical_summary.png**: Statistical tests and hypothesis evaluation
- Provides quantitative evidence for capacity-driven specialisation

### Data Files

- **capacity_ratio_raw_data.csv**: Raw experimental data for further analysis
- **analysis_results.json**: Statistical analysis results in JSON format
- **capacity_ratio_hypothesis_report.txt**: Detailed hypothesis evaluation with findings

## Expected Results

### Typical Findings

1. **Capacity-Utilisation Correlation**: Should be positive and significant
2. **Hierarchical Organisation**: Agents should prefer high-capacity resources
3. **Performance Improvement**: Asymmetric configurations should outperform symmetric
4. **Specialisation Patterns**: Clear specialisation according to capacity hierarchy

### Key Metrics

- **Capacity-Utilisation Correlation**: Strength of relationship between capacity and usage
- **Hierarchy Consistency**: How well utilisation matches capacity ranking
- **System Performance**: Total cost and efficiency measures
- **Specialisation Index**: Degree of agent specialisation achieved

### Hypothesis Evaluation

The experiment evaluates several key hypotheses:

1. **Capacity-Driven Specialisation**: Asymmetric capacities create predictable patterns
2. **Hierarchical Organisation**: Agents organise according to capacity rankings
3. **Performance Optimisation**: Asymmetric configurations improve system performance
4. **Correlation Strength**: Strong positive capacity-utilisation correlations
5. **Statistical Significance**: Performance differences are statistically significant

## Customisation Examples

### Example 1: Quick Testing
```python
# Test with minimal parameters
num_agents = 5
num_resources = 3
num_iterations = 500
num_replications = 10
capacity_configurations = [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]]
```

### Example 2: Comprehensive Analysis
```python
# Test all predefined configurations
capacity_configurations = "all_predefined"
num_replications = 100
num_iterations = 2000
```

### Example 3: Custom Configurations
```python
# Create custom capacity configurations
custom_capacities = [
    [0.8, 0.15, 0.05],  # Extreme asymmetry
    [0.6, 0.25, 0.15],  # Moderate asymmetry
    [0.4, 0.35, 0.25]   # Mild asymmetry
]
capacity_configurations = "custom_only"
```

### Example 4: Mixed Approach
```python
# Combine predefined and custom
capacity_configurations = "mixed"
custom_capacities = [[0.9, 0.08, 0.02]]  # Very extreme asymmetry
```

## Available Capacity Configurations

### Predefined Configurations

| Configuration | Description | Capacity Values |
|---------------|-------------|-----------------|
| `[0.33, 0.33, 0.33]` | Symmetric baseline | Equal resource availability |
| `[0.5, 0.3, 0.2]` | Moderate asymmetry 1 | Single dominant resource |
| `[0.4, 0.4, 0.2]` | Moderate asymmetry 2 | Two dominant resources |
| `[0.6, 0.3, 0.1]` | High asymmetry 1 | Strong hierarchy |
| `[0.7, 0.2, 0.1]` | High asymmetry 2 | Very strong hierarchy |
| `[0.5, 0.25, 0.25]` | Single dominant | Balanced secondary resources |
| `[0.8, 0.15, 0.05]` | Extreme asymmetry | Maximum hierarchy |
| `[0.45, 0.45, 0.1]` | Two dominant | Strong binary structure |
| `[0.6, 0.25, 0.15]` | Graduated hierarchy | Three-tier structure |
| `[0.55, 0.35, 0.1]` | Strong binary | Clear primary-secondary |

### Custom Configuration Types

1. **Manual Entry**: Specify exact capacity values
2. **Asymmetry-Based**: Generate based on desired asymmetry levels
3. **Resource Type-Based**: Create based on resource roles (primary, secondary, backup)

## Troubleshooting

### Common Issues

1. **Poor Performance**: Try different capacity configurations or adjust learning rate
2. **No Convergence**: Increase number of iterations or adjust convergence thresholds
3. **Memory Issues**: Reduce number of agents or replications
4. **Slow Execution**: Use fewer configurations or replications for testing

### Performance Tips

1. Start with 3 resources for ternary analysis
2. Use moderate learning rates (0.2-0.4) for clear effects
3. Ensure sufficient iterations (1000+) for convergence
4. Use 50+ replications for robust statistical analysis
5. Test asymmetric configurations for maximum effect

### Parameter Guidelines

- **Learning Rate**: 0.2-0.4 for clear capacity effects
- **Iterations**: 1000+ for reliable convergence
- **Replications**: 50+ for statistical significance
- **Resources**: 3 for ternary analysis, 3-5 for general study
- **Agents**: 5-20 for manageable computation

## Advanced Usage

### Custom Analysis

After running the tutorial, you can perform custom analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load raw data
df = pd.read_csv('results/capacity_ratio_tutorial/capacity_ratio_raw_data.csv')

# Analyze performance by configuration
config_performance = df.groupby('capacity_ratio')['total_cost'].agg(['mean', 'std'])
print("Performance by capacity configuration:")
print(config_performance)

# Compare with symmetric baseline
symmetric_cost = config_performance.loc['0.33_0.33_0.33', 'mean']
improvements = ((symmetric_cost - config_performance['mean']) / symmetric_cost) * 100
print("\nImprovement over symmetric baseline:")
print(improvements)
```

### Integration with Other Experiments

The capacity ratio study can be combined with other experiments:

1. **Weight Parameter Study**: Test different weights with asymmetric capacities
2. **Sequential Convergence Study**: Examine how capacities affect convergence patterns
3. **Initial Conditions Study**: Investigate capacity effects under biased initialisation

### Custom Configuration Creation

```python
# Example: Create graduated hierarchy
def create_graduated_hierarchy(num_resources, asymmetry=0.5):
    """Create graduated hierarchy with specified asymmetry."""
    base = (1.0 - asymmetry) / num_resources
    dominant = base + asymmetry
    config = [dominant] + [base] * (num_resources - 1)
    return config

# Use in experiment
custom_capacities = [create_graduated_hierarchy(3, 0.6)]
```

## Theoretical Background

### Capacity-Driven Specialisation Theory

Capacity ratios influence agent behaviour through several mechanisms:

1. **Resource Attraction**: High-capacity resources provide better performance
2. **Congestion Avoidance**: Agents avoid overloading low-capacity resources
3. **Hierarchical Organisation**: Natural ordering emerges based on capacity
4. **Performance Optimisation**: System efficiency improves with capacity design

### Mathematical Foundation

The capacity effects can be explained through:

- **Congestion game theory**
- **Resource allocation optimisation**
- **Multi-agent learning dynamics**
- **Hierarchical organisation theory**
- **Performance analysis frameworks**

## References

- Original experiment: `resource_allocation_sim/experiments/capacity_ratio_study.py`
- Experiment report: `reports/experiments/capacity_ratio_study.md`
- Base experiment class: `resource_allocation_sim/experiments/base_experiment.py`
- Ternary coordinate analysis: `resource_allocation_sim/visualisation/ternary.py` 