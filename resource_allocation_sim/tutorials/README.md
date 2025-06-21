# Resource Allocation Simulation Tutorials

## Overview

This directory contains comprehensive tutorials for running the four main experiments in the resource allocation simulation framework. Each tutorial provides interactive parameter customisation, detailed concept explanations, and comprehensive analysis capabilities.

## Available Tutorials

### 1. Weight Parameter Study Tutorial
**File:** `weight_parameter_tutorial.py`  
**README:** `README_weight_parameter.md`

Investigates how the learning rate parameter affects agent learning dynamics, convergence speed, and system performance. Explores the fundamental exploration-exploitation trade-off in multi-agent learning.

**Key Features:**
- Test weight values from 0.01 to 0.95
- Comprehensive performance analysis
- Trade-off visualisation
- Statistical hypothesis testing

**Quick Start:**
```bash
python -m resource_allocation_sim.tutorials.weight_parameter_tutorial
```

### 2. Sequential Convergence Study Tutorial
**File:** `sequential_convergence_tutorial.py`  
**README:** `README_sequential_convergence.md`

Examines whether agents exhibit sequential convergence patterns rather than simultaneous convergence. Investigates the temporal structure of learning processes and emergent coordination mechanisms.

**Key Features:**
- Temporal convergence analysis
- Barycentric coordinate visualisation
- Environmental feedback analysis
- Sequential pattern validation

**Quick Start:**
```bash
python -m resource_allocation_sim.tutorials.sequential_convergence_tutorial
```

### 3. Initial Conditions Study Tutorial
**File:** `initial_conditions_tutorial.py`  
**README:** `README_initial_conditions.md`

Studies how biased initial probability distributions affect agent learning dynamics and system performance compared to uniform initialisation. Explores basin of attraction effects in learning dynamics.

**Key Features:**
- Multiple initial condition types
- Basin of attraction analysis
- Performance recovery assessment
- Strategic initialisation guidance

**Quick Start:**
```bash
python -m resource_allocation_sim.tutorials.initial_conditions_tutorial
```

### 4. Capacity Ratio Study Tutorial
**File:** `capacity_ratio_tutorial.py`  
**README:** `README_capacity_ratio.md`

Investigates how asymmetric resource capacities drive predictable agent specialisation patterns. Examines capacity-driven coordination and hierarchical specialisation.

**Key Features:**
- Asymmetric capacity configurations
- Hierarchical specialisation analysis
- Ternary coordinate visualisation
- Capacity-utilisation correlation

**Quick Start:**
```bash
python -m resource_allocation_sim.tutorials.capacity_ratio_tutorial
```

## Tutorial Structure

Each tutorial follows a consistent structure:

### 1. Concept Explanation
- **What** the experiment investigates
- **Why** it's important to study
- **Theoretical foundation** and background
- **Key research questions**
- **Expected outcomes**

### 2. Interactive Parameter Customisation
- **System configuration** (agents, resources, iterations)
- **Learning parameters** (weight, convergence thresholds)
- **Resource capacity** configuration
- **Initial conditions** setup
- **Analysis options** and output configuration

### 3. Experiment Execution
- **Automatic experiment** running
- **Progress monitoring** and status updates
- **Error handling** and validation
- **Results collection** and storage

### 4. Analysis and Visualisation
- **Comprehensive plotting** with multiple visualisations
- **Statistical analysis** and hypothesis testing
- **Data export** in multiple formats
- **Interactive plot** display (optional)

### 5. Results Interpretation
- **Key findings** summary
- **Statistical significance** assessment
- **Practical implications** and recommendations
- **Next steps** and further analysis suggestions

## Common Parameters

All tutorials share common parameter categories:

### System Configuration
- **Number of agents**: 5-20 (default: 10)
- **Number of resources**: 3-8 (default: 3-5)
- **Number of iterations**: 500-3000 (default: 1000-2000)
- **Number of replications**: 10-100 (default: 30-100)

### Learning Parameters
- **Learning rate (weight)**: 0.01-0.95 (default: 0.3)
- **Convergence entropy threshold**: 0.05-0.2 (default: 0.1)
- **Convergence max probability threshold**: 0.8-0.95 (default: 0.9)

### Resource Configuration
- **Balanced capacities**: All resources equal
- **Custom distribution**: User-specified capacity ratios
- **Asymmetric configurations**: Predefined asymmetric setups

### Initial Conditions
- **Uniform initialisation**: Equal probabilities for all resources
- **Random initialisation**: Random probabilities per agent
- **Custom distribution**: User-specified initial probabilities
- **Biased conditions**: Strategic initial positioning

## Output Structure

Each tutorial generates a consistent output structure:

```
results/[tutorial_name]/
├── plots/
│   ├── [experiment_specific_plots].png
│   └── [analysis_plots].png
├── [tutorial_name]_raw_data.csv
├── [tutorial_name]_analysis.json
└── hypothesis_evaluation_report.txt
```

### Common Output Files
- **Raw data CSV**: Complete experimental data for further analysis
- **Analysis JSON**: Statistical analysis results in structured format
- **Hypothesis report**: Detailed hypothesis testing and evaluation
- **Multiple plots**: Comprehensive visualisations of results

## Usage Guidelines

### Getting Started
1. **Choose a tutorial** based on your research question
2. **Read the concept explanation** to understand the experiment
3. **Run with default parameters** for initial exploration
4. **Customise parameters** based on your specific needs
5. **Analyse results** using the generated plots and reports

### Parameter Selection Tips
- **Start conservative**: Use default parameters for initial runs
- **Scale gradually**: Increase complexity (agents, iterations, replications) as needed
- **Focus on key parameters**: Each experiment has specific parameters that matter most
- **Consider computational cost**: Balance detail with execution time

### Best Practices
- **Use sufficient replications**: At least 30 for reliable statistics
- **Monitor convergence**: Ensure agents actually converge within iteration limits
- **Validate results**: Check that outcomes make theoretical sense
- **Document parameters**: Keep track of configurations for reproducibility

## Integration and Comparison

### Cross-Experiment Analysis
The tutorials can be used together for comprehensive analysis:

1. **Parameter sensitivity**: Use weight parameter study to find optimal learning rates
2. **Convergence patterns**: Use sequential convergence study to understand temporal dynamics
3. **Initial conditions**: Use initial conditions study to optimise starting configurations
4. **System design**: Use capacity ratio study to design resource hierarchies

### Comparative Studies
- **Weight vs Initial Conditions**: How do optimal weights vary with different initial conditions?
- **Capacity vs Convergence**: How do asymmetric capacities affect convergence patterns?
- **Parameter Interactions**: How do multiple parameters interact to affect performance?

## Troubleshooting

### Common Issues
1. **Slow execution**: Reduce replications or iterations
2. **Memory problems**: Reduce number of agents or resources
3. **Poor convergence**: Increase iterations or adjust convergence thresholds
4. **Inconsistent results**: Increase number of replications

### Performance Optimisation
1. **Use appropriate system sizes**: Start small and scale up
2. **Choose efficient parameters**: Use recommended parameter ranges
3. **Disable unnecessary analysis**: Turn off optional analysis features for speed
4. **Monitor resource usage**: Check memory and CPU usage during execution

## Advanced Usage

### Custom Analysis
After running tutorials, you can perform custom analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load tutorial results
df = pd.read_csv('results/[tutorial_name]/[tutorial_name]_raw_data.csv')

# Custom analysis
# ... your analysis code here
```

### Integration with Experiments
Tutorials can be integrated with the main experiment framework:

```python
from resource_allocation_sim.experiments import *

# Use tutorial parameters with experiment classes
study = WeightParameterStudy(config)
results = study.run_experiment(weight_values=[0.1, 0.3, 0.5])
```

### Batch Processing
Run multiple tutorials in sequence:

```bash
# Run all tutorials with default parameters
python -m resource_allocation_sim.tutorials.weight_parameter_tutorial
python -m resource_allocation_sim.tutorials.sequential_convergence_tutorial
python -m resource_allocation_sim.tutorials.initial_conditions_tutorial
python -m resource_allocation_sim.tutorials.capacity_ratio_tutorial
```

## References

### Core Literature
1. **Multi-Agent Learning**: Busoniu et al. (2010)
2. **Stochastic Learning**: Narendra & Thathachar (2012)
3. **Congestion Games**: Rosenthal (1973)
4. **Potential Games**: Monderer & Shapley (1996)

### Experiment-Specific References
- **Weight Parameter**: Sutton & Barto (2018), Lattimore & Szepesvári (2020)
- **Sequential Convergence**: Harsanyi (1967), Sandholm (2001)
- **Initial Conditions**: Basin of attraction theory, dynamical systems
- **Capacity Ratios**: Resource allocation theory, hierarchical systems

## Support and Development

### Getting Help
- **Read the individual README files** for detailed tutorial-specific information
- **Check the experiment reports** in `reports/experiments/` for detailed findings
- **Examine the source code** in `experiments/` for implementation details

### Contributing
- **Add new tutorials** for additional experiments
- **Improve existing tutorials** with better explanations or features
- **Create comparison tutorials** that combine multiple experiments
- **Add new analysis methods** to existing tutorials

### Future Development
- **Interactive web interface** for parameter customisation
- **Real-time visualisation** during experiment execution
- **Automated parameter optimisation** based on objectives
- **Integration with external analysis tools** 