# Initial Conditions Study - Hypothesis 3

## Hypothesis
**H3**: The system recovers its performance when agents have an initial probability distribution that is biased across two of the three available resources.

## Research Questions
1. **Performance Recovery**: Do biased initial conditions improve system performance compared to uniform initialisation?
2. **Bias Direction**: Does the specific pair of biased resources matter for performance?
3. **Entropy Evolution**: How do different initial conditions affect final system entropy?
4. **Convergence Speed**: Do biased conditions lead to faster convergence?
5. **Load Balancing**: How do initial biases affect final resource allocation patterns?
6. **Stability**: Are systems with biased initial conditions more stable?

## Key Concepts

### Biased Initial Distribution
Agents start with probability distributions that favour **two of the three** available resources:
- **Edge Bias**: High probability on two resources, low on third (e.g., [0.45, 0.45, 0.10])
- **Diagonal Sampling**: Probabilities sampled from diagonal line in barycentric coordinates
- **Vertex Bias**: Strong preference for one resource with moderate second choice

### Barycentric Coordinate System
With 3 resources, probability distributions are visualised in barycentric coordinates:
- **Vertices**: Complete specialisation (1,0,0), (0,1,0), (0,0,1)
- **Edges**: Two-resource bias (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
- **Centre**: Uniform distribution (0.333,0.333,0.333)
- **Diagonal**: Line y = √3/3(5x-2) representing systematic bias

### Performance Recovery
System performance is measured by:
- **Final Entropy**: Lower entropy indicates better coordination
- **Cost Efficiency**: Total system cost after convergence
- **Load Balance**: Even distribution across resources
- **Convergence Speed**: Time to reach stable state

## Experimental Design

### Core Setup
- **Resources**: 3 (enables barycentric coordinate analysis)
- **Agents**: 10 (5 and 23 agent variants for comparison)
- **Iterations**: 1000 (sufficient for convergence)
- **Replications**: 100 per initial condition type
- **Weight Parameter**: 0.3 (moderate learning rate)
- **Capacity**: [0.33, 0.33, 0.33] (balanced across resources)

### Initial Condition Types

#### 1. Uniform (Baseline)
```python
# All agents start with equal probabilities
probabilities = [0.333, 0.333, 0.333]
```

#### 2. Edge Bias (Three Variants)
```python
# Resources 1&2 bias
edge_bias_12 = [0.45, 0.45, 0.10]

# Resources 1&3 bias  
edge_bias_13 = [0.45, 0.10, 0.45]

# Resources 2&3 bias
edge_bias_23 = [0.10, 0.45, 0.45]
```

#### 3. Diagonal Sampling
```python
# Sample from diagonal y = √3/3(5x-2)
# Represents systematic bias across probability simplex
```

#### 4. Vertex Bias (Three Variants)
```python
# Strong preference for Resource 1
vertex_bias_1 = [0.70, 0.15, 0.15]

# Strong preference for Resource 2
vertex_bias_2 = [0.15, 0.70, 0.15]

# Strong preference for Resource 3  
vertex_bias_3 = [0.15, 0.15, 0.70]
```

## Implementation Command

### Basic Usage
```python
from resource_allocation_sim.experiments.initial_conditions_study import run_initial_conditions_study

# Standard research run
study = run_initial_conditions_study(
    num_replications=100,       # Robust statistical sample
    num_iterations=1000,        # Sufficient convergence time
    show_plots=False,           # Save plots without display
    output_dir="results"        # Creates results/initial_conditions_study/
)

# Console output shows exact paths:
# Running initial conditions study...
# Generating analysis...
# Creating visualisations...
# Saving plots to: results/initial_conditions_study/initial_conditions_study_20241201_143052/plots
# Generated 4 plots
# Saving analysis results...
# Analysis saved to: results/initial_conditions_study/initial_conditions_study_20241201_143052
# Initial conditions study completed!
# Results available in: results/initial_conditions_study/initial_conditions_study_20241201_143052/
```

### Quick Testing
```python
# Fast test with minimal parameters
study = run_initial_conditions_study(
    num_replications=10,        # Quick testing
    num_iterations=100,         # Short runs
    show_plots=True,            # Interactive display
    output_dir="test_results"
)
```

## Expected Outcomes
- **Strong Support**: Biased conditions show significantly better performance than uniform
- **Partial Support**: Some biased conditions perform better, others similar to uniform
- **Weak Support**: Minimal performance differences between conditions
- **No Support**: Uniform initialisation performs as well or better than biased conditions

## Success Criteria
1. **Performance Improvement**: Biased conditions show lower final entropy than uniform
2. **Faster Convergence**: Biased conditions converge more quickly
3. **Better Load Balance**: More even resource utilisation
4. **Consistent Results**: Patterns hold across multiple replications
5. **Statistical Significance**: Differences are statistically meaningful

## Detailed Experimental Guide

### Step-by-Step Execution

#### Step 1: Quick Test Run
```python
# Fast test with minimal parameters
from resource_allocation_sim.experiments.initial_conditions_study import run_initial_conditions_study

study = run_initial_conditions_study(
    num_replications=5,         # Few replications for speed
    num_iterations=100,         # Short runs for quick testing
    show_plots=True,            # Interactive display
    output_dir="test_results"
)
```

#### Step 2: Standard Research Run
```python
# Full research-quality analysis
study = run_initial_conditions_study(
    num_replications=100,       # Robust statistical sample (matches hypothesis description)
    num_iterations=1000,        # Sufficient convergence time
    show_plots=False,           # Save plots without display
    output_dir="results"  # Creates results/initial_conditions_study/
)
```

#### Step 3: Comparison with Different Agent Counts
```python
# Test with 5 agents (as mentioned in hypothesis)
study_5agents = run_initial_conditions_study(
    num_replications=100,
    num_iterations=800,         # Faster convergence with fewer agents
    output_dir="results"  # Will create nested structure
)
study_5agents.base_config.num_agents = 5
study_5agents.base_config.relative_capacity = [0.6, 0.6, 0.6]  # Higher capacity per agent

# Test with 23 agents (as mentioned in hypothesis)
study_23agents = run_initial_conditions_study(
    num_replications=100,
    num_iterations=1200,        # Slower convergence with more agents
    output_dir="results"  # Will create nested structure
)
study_23agents.base_config.num_agents = 23
study_23agents.base_config.relative_capacity = [0.3, 0.3, 0.3]  # Lower capacity per agent
```

### Multi-Agent Comparison Study

```python
# Systematic comparison across agent counts
from resource_allocation_sim.experiments.initial_conditions_study import InitialConditionsStudy

agent_counts = [5, 10, 15, 23]
results_by_agents = {}

for num_agents in agent_counts:
    print(f"Running study with {num_agents} agents...")
    
    study = InitialConditionsStudy(
        results_dir=f"results/initial_conditions_study_agents_{num_agents}",
        experiment_name=f"initial_conditions_{num_agents}_agents"
    )
    
    # Configure for agent count
    study.base_config.num_agents = num_agents
    if num_agents <= 10:
        study.base_config.relative_capacity = [0.5, 0.5, 0.5]  # Higher capacity for smaller groups
        study.base_config.num_iterations = 800
    else:
        study.base_config.relative_capacity = [0.3, 0.3, 0.3]  # Lower capacity for larger groups
        study.base_config.num_iterations = 1200
    
    # Run experiment
    results = study.run_experiment(num_episodes=100)
    analysis = study.analyse_results()
    
    results_by_agents[num_agents] = {
        'study': study,
        'analysis': analysis
    }

# Generate comparison report
print("\n" + "="*80)
print("AGENT COUNT COMPARISON RESULTS")
print("="*80)

for num_agents, data in results_by_agents.items():
    support = data['analysis']['hypothesis_support']
    overall_support = support.get('overall_support', 'undetermined')
    
    if 'evidence_strength' in support:
        improvement_ratio = support['evidence_strength'].get('overall_improvement_ratio', 0)
        print(f"{num_agents:2d} Agents | Support: {overall_support.upper():<10} | "
              f"Improvement: {improvement_ratio:.1%}")
    else:
        print(f"{num_agents:2d} Agents | Support: {overall_support.upper()}")
```

## Metrics and Analysis

### Performance Metrics

#### 1. Final System Entropy
- **Formula**: H = -Σ(pi * log(pi)) where pi is proportion consuming resource i
- **Range**: 0 (perfect coordination) to log(3) ≈ 1.099 (uniform distribution)
- **Interpretation**: Lower entropy indicates better performance recovery
- **Hypothesis Prediction**: Biased conditions should achieve lower entropy than uniform

#### 2. Final System Cost
- **Calculation**: Sum of individual resource costs at final iteration
- **Components**: Based on congestion function L(r,t)
- **Range**: Minimum when load is balanced within capacity
- **Hypothesis Prediction**: Biased conditions should achieve lower costs

#### 3. Convergence Speed
- **Measurement**: Average time for agents to reach stable probability distributions
- **Method**: Track entropy changes over time, identify convergence point
- **Threshold**: When agent entropy < 0.1 and max probability > 0.9
- **Hypothesis Prediction**: Biased conditions should converge faster

#### 4. Load Balance Quality
- **Metric**: Standard deviation of final resource consumption
- **Formula**: σ = √(Σ(xi - μ)² / N) where xi is consumption of resource i
- **Range**: 0 (perfect balance) to high values (poor balance)
- **Hypothesis Prediction**: Biased conditions should achieve better balance

### Comparative Analysis Framework

#### 1. Relative Performance Assessment
```python
def compare_to_uniform(condition_results, uniform_results):
    """Compare biased condition performance to uniform baseline."""
    improvements = {
        'entropy': condition_results['final_entropy'] < uniform_results['final_entropy'],
        'cost': condition_results['final_cost'] < uniform_results['final_cost'],
        'convergence': condition_results['convergence_time'] < uniform_results['convergence_time'],
        'balance': condition_results['load_balance'] < uniform_results['load_balance']
    }
    return improvements
```

#### 2. Statistical Significance Testing
- **Test**: Welch's t-test for unequal variances
- **Null Hypothesis**: No difference between biased and uniform conditions
- **Alternative**: Biased conditions perform better
- **Significance Level**: α = 0.05

#### 3. Effect Size Calculation
- **Cohen's d**: Standardized difference between means
- **Interpretation**: 
  - Small effect: d = 0.2
  - Medium effect: d = 0.5  
  - Large effect: d = 0.8

### Hypothesis Support Evaluation

#### Support Levels
1. **Strong Support** (≥75% of biased conditions outperform uniform):
   - Clear performance improvement across multiple metrics
   - Statistically significant differences
   - Consistent patterns across replications

2. **Moderate Support** (50-74% improvement):
   - Some biased conditions perform better
   - Mixed results across different bias types
   - Statistical significance in some metrics

3. **Weak Support** (25-49% improvement):
   - Minimal performance differences
   - Inconsistent patterns
   - Limited statistical significance

4. **No Support** (<25% improvement):
   - Uniform performs as well or better
   - No systematic advantage to biased initialisation

## Required Visualisations

### Figure 1: Performance Comparison Matrix
```
2×2 subplot layout:
(a) Final System Entropy Distribution - Box plots by initial condition
(b) Final System Cost Distribution - Box plots by initial condition  
(c) Convergence Time Distribution - Box plots by initial condition
(d) Load Balance Distribution - Box plots by initial condition
```

### Figure 2: Entropy Analysis (Matches Hypothesis Figures)
```
1×2 subplot layout:
(a) Combined Agent Entropy by Initial Condition - Box plot analysis matching figures from hypothesis
(b) Hypothesis Support Evidence - Bar chart showing improvement ratios
```

### Figure 3: Barycentric Coordinate Analysis
```
Single plot showing:
- Triangle representing probability simplex
- Points marking different initial condition types
- Visual representation of bias directions
- Labels for uniform (centre) vs biased (edge/vertex) positions
```

### Figure 4: Statistical Summary
```
2×2 subplot layout:
(a) Overall Performance Ranking - Horizontal bar chart
(b) Frequency of Best Performance - Count of metrics where each condition excels
(c) Hypothesis Support Level - Visual indicator of support strength
(d) Analysis Summary Table - Key metrics and findings
```

## Implementation Features

### Custom Initial Probability Generation
```python
def generate_initial_probabilities(self, init_type: str, num_agents: int):
    """Generate probability distributions based on condition type."""
    # Supports 8 different initial condition types:
    # - uniform: [0.333, 0.333, 0.333]
    # - edge_bias_12: [0.45, 0.45, 0.10] 
    # - edge_bias_13: [0.45, 0.10, 0.45]
    # - edge_bias_23: [0.10, 0.45, 0.45]
    # - vertex_bias_1: [0.70, 0.15, 0.15]
    # - vertex_bias_2: [0.15, 0.70, 0.15] 
    # - vertex_bias_3: [0.15, 0.15, 0.70]
    # - diagonal_sample: Sample from y = √3/3(5x-2)
```

### Diagonal Sampling Implementation
The diagonal sampling implements the specific mathematical relationship from the hypothesis:
```python
# Sample from diagonal line y = √3/3(5x-2)
x = np.random.uniform(0.4, 1.0)  # Focus on upper part of triangle
y = np.sqrt(3)/3 * (5*x - 2)

# Convert to probability distribution
p1, p2, p3 = x, y, 1-x-y
# Ensure valid probability distribution
```

### Comprehensive Analysis Pipeline
1. **Data Collection**: Extract performance metrics from simulation results
2. **Condition Comparison**: Compare each biased condition against uniform baseline
3. **Statistical Testing**: Perform significance tests and effect size calculations
4. **Hypothesis Evaluation**: Assess overall support strength
5. **Visualization**: Generate publication-quality plots
6. **Report Generation**: Summarize findings and recommendations

## Expected Research Outcomes

### Strong Hypothesis Support Scenario
- **Entropy**: 80%+ of biased conditions achieve lower entropy than uniform
- **Cost**: Consistent cost improvements across biased conditions
- **Convergence**: Faster convergence for most biased initializations
- **Statistical**: p < 0.05 for multiple metrics with large effect sizes

### Partial Support Scenario  
- **Selective Improvement**: Some bias types (e.g., edge bias) perform better than others
- **Metric Dependency**: Strong performance in entropy but not necessarily cost
- **Agent Count Sensitivity**: Effects vary with number of agents

### Null Result Scenario
- **No Systematic Advantage**: Uniform performs competitively across all metrics
- **High Variability**: Large variance obscures any potential benefits
- **Context Dependency**: Benefits only appear under specific parameter combinations

## File Structure and Output

### Generated Files
The experiment creates a timestamped directory structure under `results/`:

```
results/
└── initial_conditions_study/
    └── initial_conditions_study_YYYYMMDD_HHMMSS/
        ├── full_results.pickle            # Complete experimental data (BaseExperiment)
        ├── metadata.json                  # Experiment metadata
        ├── summary_results.csv            # Summary metrics CSV
        ├── analysis_results.json          # Statistical analysis results
        ├── hypothesis_3_report.txt        # Comprehensive text report
        └── plots/
            ├── performance_comparison.png    # Figure 1: Performance metrics box plots
            ├── entropy_analysis.png          # Figure 2: Entropy analysis matching hypothesis
            ├── barycentric_initial_conditions.png  # Figure 3: Probability space visualization
            └── statistical_summary.png       # Figure 4: Summary analysis and rankings
```

#### Directory Structure Details
- **Automatic timestamping**: Each run creates a unique directory with timestamp
- **Plots subdirectory**: All visualizations are organized in `plots/` subfolder
- **Multiple formats**: Results saved as JSON, CSV, pickle, and text for different uses
- **Debug output**: Console shows exact paths where files are saved

### Programmatic Access
```python
# Access results after experiment
study = run_initial_conditions_study(num_replications=100)

# Get the results directory path
results_dir = study.get_results_dir()
print(f"Results saved to: {results_dir}")
# Example output: results/initial_conditions_study/initial_conditions_study_20241201_143052/

# Get analysis results
analysis = study.analysis_results
support_level = analysis['hypothesis_support']['overall_support']
best_condition = analysis['comparison']['best_entropy']

# Print detailed findings
for condition, results in analysis.items():
    if condition not in ['comparison', 'hypothesis_support']:
        entropy_mean = results['final_entropies']['mean']
        print(f"{condition}: Mean Entropy = {entropy_mean:.3f}")

# Access saved files directly
import json
with open(results_dir / 'analysis_results.json', 'r') as f:
    saved_analysis = json.load(f)

# Read the hypothesis report
with open(results_dir / 'hypothesis_3_report.txt', 'r') as f:
    report_text = f.read()
    print(report_text)
```

### Directory Organization Examples

#### Example 1: Quick Test Run
```bash
# After running a quick test
results/
└── initial_conditions_study/
    └── initial_conditions_study_20241201_143052/
        ├── full_results.pickle      # 5 replications × 8 conditions data
        ├── metadata.json            # Start/end times, configurations
        ├── analysis_results.json    # Hypothesis support analysis
        └── plots/
            ├── performance_comparison.png
            ├── entropy_analysis.png
            ├── barycentric_initial_conditions.png
            └── statistical_summary.png
```

#### Example 2: Full Research Run  
```bash
# After running full study
results/
└── initial_conditions_study/
    └── initial_conditions_study_20241201_151234/
        ├── full_results.pickle      # 100 replications × 8 conditions data (~large file)
        ├── metadata.json            # Complete experiment metadata
        ├── summary_results.csv      # Easy-to-analyze summary data
        ├── analysis_results.json    # Detailed statistical analysis
        ├── hypothesis_3_report.txt  # Publication-ready report
        └── plots/
            ├── performance_comparison.png    # High-quality figures
            ├── entropy_analysis.png
            ├── barycentric_initial_conditions.png
            └── statistical_summary.png
```

#### Example 3: Multi-Agent Comparison
```bash
# After running agent count comparison
results/
├── initial_conditions_study_agents_5/
│   └── initial_conditions_5_agents_20241201_152101/
├── initial_conditions_study_agents_10/
│   └── initial_conditions_10_agents_20241201_152345/
├── initial_conditions_study_agents_15/
│   └── initial_conditions_15_agents_20241201_152612/
└── initial_conditions_study_agents_23/
    └── initial_conditions_23_agents_20241201_152834/
``` 