# Initial Conditions Study Tutorial

## Overview

This tutorial demonstrates how to run the **Initial Conditions Study** experiment, which investigates how biased initial probability distributions affect agent learning dynamics and system performance compared to uniform initialisation.

## Concept Explanation

### What are Initial Conditions?

Initial conditions refer to the starting probability distributions that agents use when beginning their learning process. These distributions determine how uncertain or certain agents are about their resource preferences at the start of the experiment.

**Examples:**
- **Uniform initialisation**: [0.333, 0.333, 0.333] - Maximum uncertainty
- **Biased initialisation**: [0.6, 0.3, 0.1] - Initial preference for resource 1
- **Vertex initialisation**: [0.9, 0.05, 0.05] - Strong preference for resource 1

### Why Study Initial Conditions?

1. **Understanding basin of attraction effects in learning dynamics**
2. **Identifying strategic initial positioning for improved performance**
3. **Reducing convergence time through intelligent initialisation**
4. **Exploring the relationship between initial uncertainty and final outcomes**
5. **Validating theoretical predictions about learning pathway influence**

### Theoretical Foundation

The learning dynamics create basins of attraction around different equilibria in the probability space. Initial positioning determines which basin an agent will converge to, influencing both the speed and quality of convergence.

Strategic initial conditions can:
- **Guide agents toward favourable equilibria**
- **Reduce exploration overhead in the learning process**
- **Accelerate coordination through reduced initial uncertainty**
- **Improve system-wide performance through better resource allocation**

### Key Research Questions

1. Do biased initial conditions improve system performance?
2. How do different initial conditions affect convergence speed?
3. What is the optimal initial positioning for different scenarios?
4. How do basin of attraction effects influence learning outcomes?
5. Can strategic initialisation compensate for poor learning parameters?

### Expected Outcomes

- Biased initial conditions should improve performance over uniform
- Vertex-proximate conditions should achieve fastest convergence
- Strategic positioning should reduce convergence time
- Initial conditions should influence final resource allocation patterns
- Basin of attraction effects should be observable in convergence pathways

## Usage Instructions

### Quick Start

```bash
# Run the tutorial with default parameters
python -m resource_allocation_sim.tutorials.initial_conditions_tutorial
```

### Interactive Parameter Customisation

The tutorial allows you to customise all parameters:

#### 1. System Configuration
- **Number of agents**: Default 10
- **Number of resources**: Default 3 (optimal for barycentric analysis)
- **Number of iterations**: Default 1000
- **Number of replications**: Default 100

#### 2. Learning Parameters
- **Learning rate (weight)**: Default 0.3

#### 3. Resource Capacity Configuration
- **Balanced capacities**: All resources have equal capacity
- **Custom distribution**: Specify capacity for each resource

#### 4. Initial Conditions Configuration
Choose from four options:
- **All predefined conditions**: Test all 12 predefined conditions
- **Specific predefined conditions**: Select from available conditions
- **Custom conditions only**: Create your own initial conditions
- **Mixed approach**: Combine predefined and custom conditions

#### 5. Custom Initial Conditions
You can create custom initial conditions in three ways:
- **Fixed distribution**: Same probabilities for all agents
- **Individual distributions**: Different probabilities per agent
- **Random distributions**: Random values within specified ranges

#### 6. Convergence Parameters
- **Convergence entropy threshold**: Default 0.1
- **Convergence max probability threshold**: Default 0.9

#### 7. Output Configuration
- **Output directory**: Where to save results
- **Show plots**: Whether to display plots interactively

## Generated Outputs

### File Structure

```
results/initial_conditions_tutorial/
├── plots/
│   ├── performance_comparison.png
│   ├── entropy_analysis.png
│   ├── barycentric_initial_positions.png
│   └── statistical_analysis.png
├── initial_conditions_raw_data.csv
├── initial_conditions_analysis.json
└── hypothesis_evaluation_report.txt
```

### Key Plots Explained

#### 1. Performance Comparison
- **performance_comparison.png**: Compares final entropy and cost across conditions
- Shows how different initial conditions affect system performance

#### 2. Entropy Analysis
- **entropy_analysis.png**: Shows entropy evolution over time for each condition
- Demonstrates how quickly agents converge to specialisation

#### 3. Barycentric Coordinate Analysis
- **barycentric_initial_positions.png**: Shows starting positions in probability space
- Visualises the geometric relationships between initial conditions

#### 4. Statistical Analysis
- **statistical_analysis.png**: Statistical tests and hypothesis evaluation
- Provides quantitative evidence for performance differences

### Data Files

- **initial_conditions_raw_data.csv**: Raw experimental data for further analysis
- **initial_conditions_analysis.json**: Statistical analysis results in JSON format
- **hypothesis_evaluation_report.txt**: Detailed hypothesis evaluation with findings

## Expected Results

### Typical Findings

1. **Biased Conditions**: Should outperform uniform initialisation
2. **Vertex Conditions**: Should achieve fastest convergence and lowest entropy
3. **Edge Bias Conditions**: Should show moderate improvement over uniform
4. **Performance Hierarchy**: Vertex > Edge Bias > Uniform

### Key Metrics

- **Final Entropy**: Uncertainty in agent resource preferences
- **System Cost**: Total system efficiency measure
- **Convergence Time**: Time required to reach stable distributions
- **Performance Improvement**: Percentage improvement over uniform baseline

### Hypothesis Evaluation

The experiment evaluates several key hypotheses:

1. **Performance Recovery**: Biased conditions improve system performance
2. **Convergence Speed**: Strategic initialisation accelerates convergence
3. **Basin Effects**: Initial positioning influences final equilibria
4. **Coordination Enhancement**: Biased conditions improve coordination
5. **Statistical Significance**: Performance differences are statistically significant

## Customisation Examples

### Example 1: Quick Testing
```python
# Test with minimal parameters
num_agents = 5
num_resources = 3
num_iterations = 500
num_replications = 10
initial_condition_types = ['uniform', 'edge_bias_12']
```

### Example 2: Comprehensive Analysis
```python
# Test all predefined conditions
initial_condition_types = "all_predefined"
num_replications = 100
num_iterations = 2000
```

### Example 3: Custom Conditions
```python
# Create custom vertex bias
custom_conditions = [("custom_vertex", [0.8, 0.1, 0.1])]
initial_condition_types = "custom_only"
```

### Example 4: Mixed Approach
```python
# Combine predefined and custom
initial_condition_types = ['uniform', 'edge_bias_12']
custom_conditions = [("custom_strong_bias", [0.9, 0.05, 0.05])]
```

## Available Initial Conditions

### Predefined Conditions

| Condition | Description | Probabilities |
|-----------|-------------|---------------|
| `uniform` | Uniform distribution | [0.333, 0.333, 0.333] |
| `diagonal_point_1` | Diagonal point 1 | [0.4, 0.289, 0.311] |
| `diagonal_point_2` | Diagonal point 2 | [0.444, 0.256, 0.3] |
| `diagonal_point_3` | Diagonal point 3 | [0.489, 0.222, 0.289] |
| `diagonal_point_4` | Diagonal point 4 | [0.533, 0.189, 0.278] |
| `diagonal_point_5` | Diagonal point 5 | [0.578, 0.156, 0.266] |
| `diagonal_point_6` | Diagonal point 6 | [0.622, 0.122, 0.256] |
| `diagonal_point_7` | Diagonal point 7 | [0.667, 0.089, 0.244] |
| `diagonal_point_8` | Diagonal point 8 | [0.711, 0.056, 0.233] |
| `diagonal_point_9` | Diagonal point 9 | [0.756, 0.022, 0.222] |
| `diagonal_point_10` | Diagonal point 10 | [0.8, 0.0, 0.2] |
| `edge_bias_12` | Bias towards resources 1 and 2 | [0.45, 0.45, 0.10] |

### Custom Condition Types

1. **Fixed Distribution**: Same probabilities for all agents
2. **Individual Distributions**: Different probabilities per agent
3. **Random Distributions**: Random values within specified ranges

## Troubleshooting

### Common Issues

1. **Poor Performance**: Try different initial conditions or adjust learning rate
2. **No Convergence**: Increase number of iterations or adjust convergence thresholds
3. **Memory Issues**: Reduce number of agents or replications
4. **Slow Execution**: Use fewer conditions or replications for testing

### Performance Tips

1. Start with 3 resources for barycentric analysis
2. Use moderate learning rates (0.2-0.4) for clear effects
3. Ensure sufficient iterations (1000+) for convergence
4. Use 50+ replications for robust statistical analysis
5. Test vertex conditions for maximum performance improvement

### Parameter Guidelines

- **Learning Rate**: 0.2-0.4 for clear initial condition effects
- **Iterations**: 1000+ for reliable convergence
- **Replications**: 50+ for statistical significance
- **Resources**: 3 for barycentric analysis, 3-5 for general study
- **Agents**: 5-20 for manageable computation

## Advanced Usage

### Custom Analysis

After running the tutorial, you can perform custom analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load raw data
df = pd.read_csv('results/initial_conditions_tutorial/initial_conditions_raw_data.csv')

# Analyze performance by condition
condition_performance = df.groupby('initial_condition')['final_entropy'].agg(['mean', 'std'])
print("Performance by initial condition:")
print(condition_performance)

# Compare with uniform baseline
uniform_entropy = condition_performance.loc['uniform', 'mean']
improvements = ((uniform_entropy - condition_performance['mean']) / uniform_entropy) * 100
print("\nImprovement over uniform:")
print(improvements)
```

### Integration with Other Experiments

The initial conditions study can be combined with other experiments:

1. **Weight Parameter Study**: Test different weights with biased initial conditions
2. **Sequential Convergence Study**: Examine how initial conditions affect convergence patterns
3. **Capacity Ratio Study**: Investigate initial condition effects under asymmetric capacities

### Custom Condition Creation

```python
# Example: Create strong vertex bias
def create_vertex_bias(vertex_id, strength=0.8):
    """Create strong bias towards specific vertex."""
    probs = [0.1] * 3
    probs[vertex_id] = strength
    remaining = 1.0 - strength
    for i in range(3):
        if i != vertex_id:
            probs[i] = remaining / 2
    return probs

# Use in experiment
custom_conditions = [("strong_vertex_0", create_vertex_bias(0, 0.9))]
```

## Theoretical Background

### Basin of Attraction Theory

Initial conditions influence convergence through basin of attraction effects:

1. **Attraction Basins**: Different equilibria have different attraction regions
2. **Initial Positioning**: Determines which basin an agent converges to
3. **Pathway Effects**: Initial conditions influence the convergence pathway
4. **Performance Impact**: Basin selection affects final performance

### Mathematical Foundation

The initial condition effects can be explained through:

- **Potential game dynamics**
- **Lyapunov stability analysis**
- **Basin of attraction mapping**
- **Information-theoretic analysis**
- **Multi-agent learning theory**

## References

- Original experiment: `resource_allocation_sim/experiments/initial_conditions_study.py`
- Experiment report: `reports/experiments/initial_conditions_study.md`
- Base experiment class: `resource_allocation_sim/experiments/base_experiment.py`
- Barycentric coordinate analysis: `resource_allocation_sim/visualisation/ternary.py` 