# Weight Parameter Study Tutorial

## Overview

This tutorial demonstrates how to run the **Weight Parameter Study** experiment, which investigates how the learning rate parameter affects agent learning dynamics, convergence speed, and system performance.

## Concept Explanation

### What is the Weight Parameter?

The weight parameter (w) controls the learning intensity in the stochastic learning algorithm. It determines how strongly agents update their probability distributions based on environmental feedback.

**Learning intensity formula:**
```
λ(t) = w × L(t)
```
where L(t) is the cost of the selected resource at time t.

**Weight range:** w ∈ (0, 1)

- **Low weights (w < 0.1)**: Conservative learning, slow convergence
- **Moderate weights (0.1 ≤ w ≤ 0.5)**: Balanced learning
- **High weights (w > 0.5)**: Aggressive learning, fast convergence

### Why Study the Weight Parameter?

1. **Understanding the exploration-exploitation trade-off**
2. **Identifying optimal learning rates for different scenarios**
3. **Quantifying the speed-performance relationship**
4. **Validating theoretical predictions about convergence rates**
5. **Providing practical guidelines for parameter selection**

### Theoretical Foundation

The weight parameter directly influences the learning dynamics:

- **Higher weights** increase learning intensity λ(t)
- **Increased intensity** leads to faster probability updates
- **Faster updates** result in quicker convergence
- **However, excessive intensity** may cause overshooting
- **Overshooting** can lead to suboptimal final performance

**The exploration-exploitation trade-off:**
- **Low weights**: More exploration, slower convergence, better final performance
- **High weights**: Less exploration, faster convergence, potentially worse performance
- **Optimal weights**: Balance between speed and quality

### Key Research Questions

1. How does the weight parameter affect convergence speed?
2. What is the relationship between weight and final system performance?
3. Is there an optimal weight range for different scenarios?
4. How does weight influence the exploration-exploitation balance?
5. Can weight selection compensate for poor initial conditions?

### Expected Outcomes

- Higher weights should lead to faster convergence
- Lower weights should achieve better final performance
- There should be an optimal weight range for balanced performance
- Weight effects should be consistent across different scenarios
- The exploration-exploitation trade-off should be clearly observable

## Usage Instructions

### Quick Start

```bash
# Run the tutorial with default parameters
python -m resource_allocation_sim.tutorials.weight_parameter_tutorial
```

### Interactive Parameter Customisation

The tutorial allows you to customise all parameters:

#### 1. System Configuration
- **Number of agents**: Default 10
- **Number of resources**: Default 5
- **Number of iterations**: Default 1000
- **Number of replications**: Default 30

#### 2. Weight Parameter Configuration
Choose from four options:
- **Predefined weight range**: Test weights from 0.01 to 0.95
- **Custom weight values**: Specify your own weight values
- **Specific weight range**: Define min/max with number of values
- **Optimal range focus**: Focus on weights 0.1 to 0.5

#### 3. Resource Capacity Configuration
- **Balanced capacities**: All resources have equal capacity
- **Custom distribution**: Specify capacity for each resource

#### 4. Initial Conditions Configuration
- **Uniform initialisation**: All agents start with equal probabilities
- **Random initialisation**: Random probabilities for each agent
- **Custom distribution**: Specify your own initial probabilities

#### 5. Convergence Parameters
- **Convergence entropy threshold**: Default 0.1
- **Convergence max probability threshold**: Default 0.9

#### 6. Output Configuration
- **Output directory**: Where to save results
- **Show plots**: Whether to display plots interactively

## Generated Outputs

### File Structure

```
results/weight_parameter_tutorial/
├── plots/
│   ├── convergence_times_vs_weight.png
│   ├── costs_vs_weight.png
│   ├── final_entropy_vs_weight.png
│   ├── cost_convergence_tradeoff.png
│   ├── stability_performance_tradeoff.png
│   ├── performance_distributions.png
│   ├── performance_heatmap.png
│   └── system_behaviour_radar.png
├── weight_parameter_raw_data.csv
├── weight_parameter_analysis.json
└── hypothesis_evaluation_report.txt
```

### Key Plots Explained

#### 1. Convergence Analysis
- **convergence_times_vs_weight.png**: Shows how weight affects convergence speed
- **final_entropy_vs_weight.png**: Shows how weight affects final uncertainty

#### 2. Performance Analysis
- **costs_vs_weight.png**: Shows how weight affects final system performance
- **performance_distributions.png**: Statistical distribution of performance across weights

#### 3. Trade-off Analysis
- **cost_convergence_tradeoff.png**: Visualises the fundamental speed-performance trade-off
- **stability_performance_tradeoff.png**: Shows stability vs performance relationships

#### 4. Comprehensive Analysis
- **performance_heatmap.png**: Multi-dimensional performance analysis across all weights
- **system_behaviour_radar.png**: Radar plots showing behavioural profiles for different weights

### Data Files

- **weight_parameter_raw_data.csv**: Raw experimental data for further analysis
- **weight_parameter_analysis.json**: Statistical analysis results in JSON format
- **hypothesis_evaluation_report.txt**: Detailed hypothesis evaluation with findings

## Expected Results

### Typical Findings

1. **Convergence Speed**: Higher weights lead to faster convergence
2. **System Performance**: Lower weights often achieve better final performance
3. **Optimal Range**: Weights between 0.1 and 0.5 typically provide good balance
4. **Trade-off**: Clear speed-performance trade-off across weight spectrum

### Key Metrics

- **Convergence Time**: Time required to reach stable distributions
- **Final System Cost**: Total system efficiency measure
- **Final Entropy**: Uncertainty in agent resource preferences
- **Performance Variance**: Consistency of outcomes across replications

### Hypothesis Evaluation

The experiment evaluates several key hypotheses:

1. **Weight Effects**: Weight parameter significantly affects learning dynamics
2. **Convergence Speed**: Higher weights produce faster convergence
3. **Performance Trade-off**: Speed improvements come at cost of final performance
4. **Optimal Range**: There exists an optimal weight range for balanced performance
5. **Exploration-Exploitation**: Weight controls the exploration-exploitation balance

## Customisation Examples

### Example 1: Quick Testing
```python
# Test with minimal parameters
num_agents = 5
num_resources = 3
num_iterations = 500
num_replications = 10
weight_values = [0.1, 0.3, 0.5]
```

### Example 2: Comprehensive Analysis
```python
# Test full weight range
weight_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
num_replications = 50
num_iterations = 2000
```

### Example 3: Optimal Range Focus
```python
# Focus on optimal range
weight_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
num_replications = 100
```

### Example 4: Custom Weights
```python
# Test specific weight values
weight_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

## Available Weight Configurations

### Predefined Weight Range
```python
weight_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
```

### Conservative Learning
```python
weight_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
```

### Aggressive Learning
```python
weight_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
```

### Balanced Learning
```python
weight_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
```

## Interpretation Guidelines

### Convergence Speed Analysis
- **Fast convergence**: Weights > 0.5 typically achieve convergence in < 20 iterations
- **Moderate convergence**: Weights 0.1-0.5 typically achieve convergence in 20-100 iterations
- **Slow convergence**: Weights < 0.1 may require > 100 iterations

### Performance Analysis
- **Best performance**: Often achieved with weights 0.01-0.1
- **Balanced performance**: Weights 0.1-0.5 provide good balance
- **Fast but suboptimal**: Weights > 0.5 may sacrifice performance for speed

### Trade-off Analysis
- **Exploration regime**: Weights < 0.1 prioritise thorough exploration
- **Balanced regime**: Weights 0.1-0.5 balance exploration and exploitation
- **Exploitation regime**: Weights > 0.5 prioritise rapid exploitation

## Troubleshooting

### Common Issues

1. **Slow execution**: Reduce number of replications or iterations
2. **Memory issues**: Reduce number of agents or resources
3. **Poor convergence**: Check that weight values are in valid range (0, 1)
4. **Inconsistent results**: Increase number of replications for better statistics

### Performance Tips

1. **Start with predefined range**: Use the default weight range for initial exploration
2. **Focus on optimal range**: Concentrate on weights 0.1-0.5 for most applications
3. **Use sufficient replications**: At least 30 replications for reliable statistics
4. **Monitor convergence**: Check that agents actually converge within iteration limit

## Related Tutorials

- **Initial Conditions Tutorial**: Study how initial conditions affect learning
- **Sequential Convergence Tutorial**: Investigate temporal convergence patterns
- **Capacity Ratio Tutorial**: Explore asymmetric resource capacity effects

## References

For more information about weight parameter effects in multi-agent learning:

1. **Exploration-Exploitation Trade-off**: Sutton & Barto (2018)
2. **Learning Rate Selection**: Lattimore & Szepesvári (2020)
3. **Multi-Agent Learning**: Busoniu et al. (2010)
4. **Stochastic Learning**: Narendra & Thathachar (2012) 