# Sequential Convergence Study Tutorial

## Overview

This tutorial demonstrates how to run the **Sequential Convergence Study** experiment, which investigates whether agents starting from uniform initial probability distributions exhibit sequential convergence patterns rather than simultaneous convergence.

## Concept Explanation

### What is Sequential Convergence?

Sequential convergence occurs when agents in a multi-agent system reach their final resource preferences at different times rather than simultaneously. This creates a temporal ordering of convergence events that reflects the emergent coordination dynamics.

**Key characteristics:**
- Agents converge one after another, not all at once
- Early convergers influence the environment for later agents
- Convergence timing reflects resource preference hierarchies
- Sequential patterns emerge from partial observability

### Why Study Sequential Convergence?

1. **Understanding emergent coordination patterns in multi-agent systems**
2. **Validating theoretical predictions about convergence dynamics**
3. **Identifying the role of partial observability in coordination**
4. **Characterising the temporal structure of learning processes**
5. **Providing insights into system-level coordination mechanisms**

### Theoretical Foundation

Sequential convergence emerges from several theoretical principles:

**Partial Observability:**
- Agents cannot observe other agents' probability distributions
- Agents only see environmental feedback (resource costs)
- This creates information asymmetry between agents

**Environmental Feedback:**
- Early convergers create cost gradients in the environment
- These gradients influence subsequent agent decisions
- Later agents adapt to the changed cost landscape

**Information Cascades:**
- Early decisions create signals for later agents
- These signals guide subsequent convergence events
- The process creates a natural ordering of specialisation

### Key Research Questions

1. Do agents exhibit sequential convergence patterns?
2. What determines the order of convergence events?
3. How does partial observability influence convergence timing?
4. Are sequential patterns consistent across different scenarios?
5. What are the implications for system-level coordination?

### Expected Outcomes

- Agents should converge sequentially rather than simultaneously
- Convergence timing should show clear temporal separation
- Early convergers should influence later convergence events
- All agents should achieve degenerate distributions
- Sequential patterns should be consistent across replications

## Usage Instructions

### Quick Start

```bash
# Run the tutorial with default parameters
python -m resource_allocation_sim.tutorials.sequential_convergence_tutorial
```

### Interactive Parameter Customisation

The tutorial allows you to customise all parameters:

#### 1. System Configuration
- **Number of agents**: Default 10
- **Number of resources**: Default 3 (optimal for barycentric analysis)
- **Number of iterations**: Default 2000
- **Number of replications**: Default 100

#### 2. Learning Parameters
- **Learning rate (weight)**: Default 0.3

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

#### 6. Analysis Parameters
- **Barycentric coordinate analysis**: Enable geometric analysis in probability space
- **Individual trajectory analysis**: Enable detailed agent trajectory analysis

#### 7. Output Configuration
- **Output directory**: Where to save results
- **Show plots**: Whether to display plots interactively

## Generated Outputs

### File Structure

```
results/sequential_convergence_tutorial/
├── plots/
│   ├── statistical_analysis.png
│   ├── convergence_timeline_analysis.png
│   ├── probability_evolution_analysis.png
│   ├── system_dynamics_analysis.png
│   ├── barycentric_all_trajectories.png
│   ├── barycentric_individual_trajectories.png
│   ├── barycentric_final_distributions.png
│   └── individual_trajectories.png
├── sequential_convergence_raw_data.csv
├── sequential_convergence_analysis.json
└── hypothesis_evaluation_report.txt
```

### Key Plots Explained

#### 1. Statistical Analysis
- **statistical_analysis.png**: Comprehensive analysis of sequential patterns
- Shows convergence intervals, cumulative patterns, and sequential indices

#### 2. Temporal Analysis
- **convergence_timeline_analysis.png**: Temporal evolution of convergence events
- Shows when each agent converges and the ordering of events

#### 3. Learning Dynamics
- **probability_evolution_analysis.png**: How agent probabilities change over time
- Shows the learning process and preference formation

#### 4. System Dynamics
- **system_dynamics_analysis.png**: System-wide dynamics during convergence
- Shows how early convergers influence later agents

#### 5. Geometric Analysis (Barycentric)
- **barycentric_all_trajectories.png**: All agent trajectories in probability space
- **barycentric_individual_trajectories.png**: Individual agent pathways
- **barycentric_final_distributions.png**: Final specialisation patterns

#### 6. Individual Analysis
- **individual_trajectories.png**: Detailed analysis of individual agent behaviour

### Data Files

- **sequential_convergence_raw_data.csv**: Raw experimental data for further analysis
- **sequential_convergence_analysis.json**: Statistical analysis results in JSON format
- **hypothesis_evaluation_report.txt**: Detailed hypothesis evaluation with findings

## Expected Results

### Typical Findings

1. **Sequential Patterns**: Agents converge one after another, not simultaneously
2. **Temporal Separation**: Clear time gaps between convergence events
3. **Consistent Ordering**: Similar convergence order across replications
4. **Complete Specialisation**: All agents achieve degenerate distributions
5. **Environmental Influence**: Early convergers affect later convergence

### Key Metrics

- **Sequential Index**: Measure of temporal separation (1.0 = perfect sequential)
- **Convergence Timing**: When each agent reaches specialisation
- **Degeneracy Achievement**: Percentage of agents achieving complete specialisation
- **Convergence Consistency**: Similarity of patterns across replications

### Hypothesis Evaluation

The experiment evaluates several key hypotheses:

1. **Sequential Convergence**: Agents converge sequentially rather than simultaneously
2. **Temporal Separation**: Convergence events show clear temporal gaps
3. **Environmental Feedback**: Early convergers influence later agents
4. **Complete Specialisation**: All agents achieve degenerate distributions
5. **Pattern Consistency**: Sequential patterns are consistent across replications

## Customisation Examples

### Example 1: Quick Testing
```python
# Test with minimal parameters
num_agents = 5
num_resources = 3
num_iterations = 1000
num_replications = 10
weight = 0.3
```

### Example 2: Comprehensive Analysis
```python
# Full analysis with all features
num_agents = 10
num_resources = 3
num_iterations = 2000
num_replications = 100
enable_barycentric_analysis = True
enable_trajectory_analysis = True
```

### Example 3: Large System
```python
# Test with larger system
num_agents = 20
num_resources = 5
num_iterations = 3000
num_replications = 50
```

### Example 4: Different Learning Rates
```python
# Test different learning rates
weight = 0.1  # Conservative learning
# or
weight = 0.5  # Aggressive learning
```

## Available Configurations

### Standard Configuration
```python
num_agents = 10
num_resources = 3
num_iterations = 2000
num_replications = 100
weight = 0.3
initial_condition_type = "uniform"
```

### Conservative Learning
```python
weight = 0.1
num_iterations = 3000  # More iterations for slower learning
```

### Aggressive Learning
```python
weight = 0.5
num_iterations = 1000  # Fewer iterations for faster learning
```

### Large System
```python
num_agents = 20
num_resources = 5
num_iterations = 2500
```

## Interpretation Guidelines

### Sequential Pattern Analysis
- **Sequential Index = 1.0**: Perfect sequential convergence
- **Sequential Index > 0.8**: Strong sequential patterns
- **Sequential Index < 0.5**: Weak or random patterns

### Convergence Timing Analysis
- **Early convergers**: First 30% of agents (typically iterations 50-100)
- **Middle convergers**: 30-70% of agents (typically iterations 100-200)
- **Late convergers**: Last 30% of agents (typically iterations 200-300)

### Specialisation Analysis
- **Complete specialisation**: All agents achieve entropy < 0.1
- **Resource preference**: Clear preference for specific resources
- **Load balancing**: Even distribution across available resources

### Environmental Feedback Analysis
- **Cost gradients**: Early convergers create predictable cost patterns
- **Adaptation**: Later agents adapt to remaining opportunities
- **Coordination**: Emergent coordination without explicit communication

## Troubleshooting

### Common Issues

1. **No sequential patterns**: Check that weight is not too high (causes simultaneous convergence)
2. **Slow execution**: Reduce number of replications or iterations
3. **Memory issues**: Reduce number of agents or disable trajectory analysis
4. **Poor convergence**: Increase number of iterations or adjust convergence thresholds

### Performance Tips

1. **Start with standard configuration**: Use default parameters for initial exploration
2. **Use sufficient iterations**: At least 1000 iterations for reliable convergence
3. **Enable barycentric analysis**: Provides valuable geometric insights
4. **Monitor convergence**: Check that agents actually converge within iteration limit

## Related Tutorials

- **Weight Parameter Tutorial**: Study how learning rate affects convergence
- **Initial Conditions Tutorial**: Investigate initial condition effects
- **Capacity Ratio Tutorial**: Explore asymmetric resource capacity effects

## References

For more information about sequential convergence in multi-agent learning:

1. **Partial Observability**: Harsanyi (1967)
2. **Information Cascades**: Bikhchandani et al. (1992)
3. **Multi-Agent Learning**: Busoniu et al. (2010)
4. **Convergence Dynamics**: Sandholm (2001)
5. **Environmental Feedback**: Tumer & Agogino (2008) 