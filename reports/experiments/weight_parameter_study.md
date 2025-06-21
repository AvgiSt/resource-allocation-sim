# Weight Parameter (Learning Rate) Study

## Status: COMPLETED ✓

## Hypothesis
**H1**: The weight parameter `w` significantly affects agent learning dynamics, convergence speed, and final system performance in the resource allocation simulation.

**RESULT**: ✓ **CONFIRMED** - Significant effects observed with optimal weight = 0.9 for both cost and convergence performance.

## Research Questions
1. **Convergence Speed**: How does `w` affect the time required for agents to reach stable probability distributions?
   - **Finding**: Higher weights (0.7-0.9) achieve significantly faster convergence
2. **System Performance**: What is the relationship between `w` and final system cost/efficiency?
   - **Finding**: Weight = 0.9 provides optimal cost performance across replications
3. **Learning Stability**: How does `w` influence the variance in final outcomes?
   - **Finding**: Moderate weights (0.3-0.5) show better stability; very high weights can be volatile
4. **Exploration vs Exploitation**: How does `w` balance exploration of new resources vs exploitation of known good resources?
   - **Finding**: Clear tradeoff observed; optimal balance at w = 0.9 for this configuration
5. **Optimal Range**: What range of `w` values provides the best trade-off between learning speed and stability?
   - **Finding**: Recommended range 0.7-0.9, with 0.9 as optimal for most scenarios

## Experimental Design

### Parameter Values Tested
```python
weight_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
```

**Rationale**:
- **Low values (0.01-0.1)**: Slow, conservative learning
- **Medium values (0.2-0.5)**: Balanced learning  
- **High values (0.7-0.95)**: Fast, aggressive learning

### Fixed Parameters (As Implemented)
- **Resources**: 5
- **Agents**: 10
- **Relative Capacity**: [0.2, 0.2, 0.2, 0.2, 0.2] (1.5x agent capacity per resource)
- **Iterations**: 100 (sufficient for convergence analysis)
- **Replications**: Configurable (default 5 for testing, 30+ for full analysis)
- **Initialisation**: "uniform" (for consistency)
- **Environment**: Standard exponential cost function

### Metrics Collected (Implemented Analysis)

#### 1. Convergence Metrics
- **Convergence times**: Per-agent convergence detection using entropy thresholds
- **Convergence speeds**: System-level rate of cost stabilisation  
- **Final entropies**: Agent probability distribution uncertainty
- **Learning stability**: Variance in final probability states

#### 2. Performance Metrics
- **Final costs**: Total system cost at convergence
- **Steady-state costs**: Mean cost over final 100 iterations
- **Load balancing quality**: Standard deviation of resource utilisation
- **System efficiency**: Cost per unit of total demand

#### 3. Learning Dynamics
- **Decision certainty**: Maximum probability achieved by agents
- **Exploration ratios**: Proportion of different resources explored
- **Trajectory variance**: Fluctuation in agent probability paths
- **Agent diversity**: Inter-agent differences in final states

#### 4. Advanced Analysis
- **Cost-convergence tradeoffs**: Performance vs speed relationships  
- **Stability-performance tradeoffs**: Variance vs mean cost analysis
- **Statistical significance testing**: ANOVA and regression analysis
- **Comprehensive visualisation**: 6 detailed plot categories

## Key Metrics Explained ✓

### 1. Cost (Final Cost) 
**Definition**: Average cost per resource at the final iteration (iteration 99)

**Calculation**:
```python
# For each replication:
cost_history = environment_state['cost_history']  # Shape: [100 iterations, 5 resources]
final_cost = np.mean(cost_history[-1])           # Mean of costs at iteration 99
```

**Per-Resource Cost Formula**:
```python
# For each resource r:
if consumption[r] == 0:
    cost[r] = 0.0                                    # No agents = no cost
elif consumption[r] <= capacity[r]:  
    cost[r] = 1.0                                    # No congestion = base cost
else:
    cost[r] = exp(1 - consumption[r]/capacity[r])    # Exponential congestion penalty
```

**Interpretation**:
- **Range**: 0.0 to ~2.5 (theoretical max when all agents on one resource)
- **Lower is better**: Indicates better resource allocation
- **Typical values**: 1.0-1.5 for balanced allocation, >2.0 for poor allocation
- **Weight effect**: Higher learning rates typically achieve lower final costs

### 2. Load Balance (Standard Deviation)
**Definition**: Variability in resource consumption at the final iteration

**Calculation**:
```python
# For each replication:
consumption_history = environment_state['consumption_history']  # [100 iterations, 5 resources]
final_consumption = consumption_history[-1]                     # [3, 2, 1, 2, 2] (example)
load_balance = np.std(final_consumption)                        # Standard deviation
```

**Interpretation**:
- **Perfect balance**: std = 0 (all resources have equal agents: [2, 2, 2, 2, 2])
- **Poor balance**: std > 2 (agents clustered: [8, 1, 0, 1, 0])
- **Typical values**: 0.5-1.5 for reasonable balance
- **Lower is better**: More even distribution of agents across resources
- **Weight effect**: Moderate weights often achieve better balance

### 3. Consumption Entropy 
**Definition**: Information-theoretic measure of distribution evenness

**Calculation**:
```python
def calculate_entropy(consumption):
    # Normalise to probabilities
    total = np.sum(consumption)
    if total == 0:
        return 0.0
    
    probabilities = consumption / total
    
    # Calculate Shannon entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log(p)
    return entropy
```

**Example**:
```python
# Perfect balance: [2, 2, 2, 2, 2] → probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
# entropy = -5 * (0.2 * log(0.2)) = 1.609 (maximum for 5 resources)

# Poor balance: [8, 1, 0, 1, 0] → probabilities = [0.8, 0.1, 0.0, 0.1, 0.0]  
# entropy = -(0.8*log(0.8) + 0.1*log(0.1) + 0.1*log(0.1)) = 0.639
```

**Interpretation**:
- **Maximum entropy**: ln(5) = 1.609 for 5 resources (perfect balance)
- **Minimum entropy**: 0.0 (all agents on one resource)
- **Higher is better**: More even distribution
- **Complementary to std**: Entropy considers relative proportions, std considers absolute differences

### 4. Final Entropy (Agent Probabilities)
**Definition**: Shannon entropy of each agent's final probability distribution over resources

**Calculation**:
```python
# For each agent at final iteration:
final_probabilities = agent_prob_history[-1]        # [0.1, 0.05, 0.8, 0.03, 0.02]

# Calculate Shannon entropy
entropy = 0.0
for p in final_probabilities:
    if p > 0:
        entropy -= p * np.log(p)                     # Shannon entropy formula
# Result: entropy = 0.639 (agent is relatively certain about resource 2)
```

**Example**:
```python
# High uncertainty (still exploring): [0.25, 0.2, 0.3, 0.15, 0.1]
# entropy = -(0.25*ln(0.25) + 0.2*ln(0.2) + 0.3*ln(0.3) + 0.15*ln(0.15) + 0.1*ln(0.1)) = 1.485

# Moderate certainty: [0.1, 0.05, 0.8, 0.03, 0.02]  
# entropy = -(0.1*ln(0.1) + 0.05*ln(0.05) + 0.8*ln(0.8) + 0.03*ln(0.03) + 0.02*ln(0.02)) = 0.639

# High certainty (converged): [0.02, 0.01, 0.95, 0.01, 0.01]
# entropy = -(0.02*ln(0.02) + 0.01*ln(0.01) + 0.95*ln(0.95) + 0.01*ln(0.01) + 0.01*ln(0.01)) = 0.242

# Perfect certainty: [0.0, 0.0, 1.0, 0.0, 0.0] → entropy = 0.0
```

**Interpretation**:
- **Maximum entropy**: ln(5) = 1.609 for 5 resources (uniform distribution: [0.2, 0.2, 0.2, 0.2, 0.2])
- **Minimum entropy**: 0.0 (perfect specialisation: [0, 0, 1, 0, 0])
- **Lower is better**: Indicates agent has converged to preferred resource(s)
- **Weight effect**: Higher weights lead to lower final entropy (faster convergence)
- **What's plotted**: Average final entropy across all agents for each weight value

**Note**: This is different from Consumption Entropy (section 3), which measures the distribution of actual resource usage across the system.

### 5. Convergence Speed
**Definition**: Rate of cost reduction in the final 10% of iterations

**Calculation**:
```python
def _calculate_convergence_speed(cost_history):
    # Step 1: Calculate total system cost per iteration
    total_costs = np.sum(cost_history, axis=1)  # [100 values]
    
    # Step 2: Focus on final 10% of iterations  
    final_10_percent = int(0.1 * len(total_costs))  # Last 10 iterations
    final_costs = total_costs[-final_10_percent:]   # [10 values]
    
    # Step 3: Calculate rate of change (differences between consecutive iterations)
    cost_changes = np.diff(final_costs)  # [9 values]
    
    # Step 4: Return negative mean (positive = improving, negative = worsening)
    return -np.mean(cost_changes)
```

**Example**:
```python
# Good convergence: final costs = [5.2, 5.1, 5.0, 5.0, 4.9, 4.9, 4.8, 4.8, 4.7, 4.7]
# cost_changes = [-0.1, -0.1, 0.0, -0.1, 0.0, -0.1, 0.0, -0.1, 0.0]
# convergence_speed = -(-0.044) = 0.044 (positive = good)

# Poor convergence: final costs = [5.1, 5.3, 5.0, 5.2, 5.1, 5.4, 5.2, 5.1, 5.3, 5.2]
# cost_changes = [0.2, -0.3, 0.2, -0.1, 0.3, -0.2, -0.1, 0.2, -0.1]  
# convergence_speed = -(-0.011) = 0.011 (lower = oscillating/unstable)
```

**Interpretation**:
- **Positive values**: System is still improving (costs decreasing)
- **Values near zero**: System has converged (costs stable)
- **Negative values**: System is worsening (costs increasing)
- **Typical values**: 0.001-0.1 for good convergence
- **Weight effect**: Higher weights often show faster initial convergence but may plateau

### 6. Decision Certainty
**Definition**: Maximum probability an agent assigns to any single resource at convergence

**Calculation**:
```python
# For each agent at final iteration:
final_probabilities = agent_prob_history[-1]        # [0.1, 0.05, 0.8, 0.03, 0.02]
decision_certainty = np.max(final_probabilities)    # 0.8 (agent is 80% certain about resource 2)
```

**Example**:
```python
# High certainty (converged): [0.05, 0.02, 0.9, 0.02, 0.01] → certainty = 0.9
# Low certainty (exploring): [0.25, 0.2, 0.3, 0.15, 0.1] → certainty = 0.3
# Perfect certainty: [0.0, 0.0, 1.0, 0.0, 0.0] → certainty = 1.0
```

**Interpretation**:
- **Range**: 0.2 to 1.0 (starts at 0.2 for uniform distribution over 5 resources)
- **Higher is better**: Indicates stronger preference/specialisation
- **Weight effect**: Higher weights lead to higher decision certainty (faster convergence)
- **Relationship to entropy**: High certainty = low entropy

### 7. System Coverage Ratio (Multi-Agent)
**Definition**: Fraction of all possible (agent, resource) pairs that were explored by the system

**Calculation**:
```python
def calculate_system_coverage_ratio(all_action_histories, num_resources, num_agents):
    # Find all unique (agent_id, resource) combinations that were tried
    explored_pairs = set()
    for agent_id, action_history in enumerate(all_action_histories):
        for resource in set(action_history):  # Unique resources this agent tried
            explored_pairs.add((agent_id, resource))
    
    total_possible_pairs = num_agents * num_resources  # 10 agents × 5 resources = 50
    return len(explored_pairs) / total_possible_pairs
```

**Example**:
```python
# High exploration: All 10 agents try all 5 resources
# explored_pairs = {(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), ...} = 50 pairs
# coverage_ratio = 50/50 = 1.0

# Low exploration: Each agent tries only 2 resources  
# explored_pairs = {(0,0), (0,1), (1,1), (1,2), ...} = 20 pairs
# coverage_ratio = 20/50 = 0.4
```

**Interpretation**:
- **Range**: 0.2 to 1.0 (minimum 0.2 since each agent must try at least 1 resource)
- **Higher is better**: More comprehensive system-wide exploration
- **Weight effect**: Lower weights → higher coverage (more exploration before convergence)
- **Scalability**: Automatically accounts for different numbers of agents

### 8. Exploration Diversity (Multi-Agent)
**Definition**: Variation in individual exploration strategies across agents (coefficient of variation)

**Calculation**:
```python
def calculate_exploration_diversity(all_action_histories):
    # Calculate individual exploration ratio for each agent
    individual_ratios = []
    for action_history in all_action_histories:
        unique_resources = len(set(action_history))  # Resources this agent tried
        total_actions = len(action_history)          # Total decisions made
        exploration_ratio = unique_resources / total_actions
        individual_ratios.append(exploration_ratio)
    
    # Calculate coefficient of variation (std/mean)
    mean_ratio = np.mean(individual_ratios)
    std_ratio = np.std(individual_ratios)
    return std_ratio / (mean_ratio + 1e-10)  # Avoid division by zero
```

**Example**:
```python
# Low diversity (all agents similar): [0.05, 0.05, 0.05, 0.05, 0.05]
# mean = 0.05, std = 0.0 → diversity = 0.0/0.05 = 0.0

# High diversity (mixed strategies): [0.05, 0.03, 0.08, 0.02, 0.04]  
# mean = 0.044, std = 0.021 → diversity = 0.021/0.044 = 0.48

# Perfect diversity: Some explore (0.05), others specialise (0.01)
```

**Interpretation**:
- **Range**: 0.0 to ~0.6 (higher values indicate more strategy variation)
- **Higher values**: Some agents explore broadly while others specialise quickly
- **Lower values**: All agents follow similar exploration patterns
- **Weight effect**: Higher weights often create more diversity (some fast convergers, some explorers)

### 9. Collective Discovery Rate (Multi-Agent)
**Definition**: How quickly the system as a whole discovers available resources (normalised)

**Calculation**:
```python
def calculate_collective_discovery_rate(all_action_histories, num_resources):
    # Track when each resource is first discovered by ANY agent
    discovered_resources = set()
    max_iterations = max(len(history) for history in all_action_histories)
    
    # Find iteration when 90% of resources were discovered
    for iteration in range(max_iterations):
        for action_history in all_action_histories:
            if iteration < len(action_history):
                discovered_resources.add(action_history[iteration])
        
        if len(discovered_resources) >= 0.9 * num_resources:
            discovery_90_time = iteration
            break
    
    # Normalise: 1.0 = immediate discovery, 0.0 = never discovered 90%
    return 1 - (discovery_90_time / max_iterations)
```

**Example**:
```python
# Fast discovery: 90% of resources found by iteration 5/100
# discovery_rate = 1 - (5/100) = 0.95

# Slow discovery: 90% of resources found by iteration 50/100  
# discovery_rate = 1 - (50/100) = 0.5

# Very slow: 90% discovered by iteration 95/100
# discovery_rate = 1 - (95/100) = 0.05
```

**Interpretation**:
- **Range**: 0.0 to 1.0 
- **Higher is better**: System discovers available options more quickly
- **Weight effect**: Lower weights may lead to faster discovery (more exploration)
- **System efficiency**: Measures collective learning speed vs individual convergence

### Metric Relationships for Radar Charts
**Decision Certainty vs Convergence Speed**: Both increase with higher weights (faster, more decisive learning)
**System Coverage vs Exploration Diversity**: High coverage with low diversity = systematic exploration; high coverage with high diversity = mixed strategies
**Collective Discovery vs Decision Certainty**: Trade-off between system-wide exploration and individual specialisation
**Weight Parameter Effects**: 
- **Low weights (w=0.1)**: High coverage, low diversity, high discovery, low certainty
- **High weights (w=0.9)**: Low coverage, high diversity, variable discovery, high certainty

## Generated Plots and Figures ✓

### 1. Basic Parameter Sensitivity Plots
```
convergence_times_vs_weight.png
- Line plot with error bars showing convergence speed vs weight
- Clear demonstration of faster convergence at higher weights

final_entropy_vs_weight.png  
- Average final entropy of agent probability distributions vs weight values
- Shows how decision uncertainty decreases with higher learning rates
- Lower entropy = more converged/specialised agents

costs_vs_weight.png
- Steady-state cost performance vs weight parameter
- Identifies optimal weight values for cost minimisation
```

### 2. Advanced Analysis Visualisations
```
performance_heatmap.png
- Normalised performance metrics across all weight values
- Cost, load balance, and convergence speed in single view

system_behaviour_radar.png
- Radar charts showing multi-agent system behaviour patterns
- Decision certainty, convergence speed, system coverage, exploration diversity, collective discovery

comparative_behaviour_radar.png
- Single radar chart comparing 4 weight values (0.1, 0.3, 0.7, 0.9)
- All 5 behaviour metrics overlaid for direct comparison

cost_evolution_comparison.png
- Time-series comparison of cost evolution 
- Multiple weight values with confidence intervals
```

### 3. Tradeoff Analysis Plots
```
cost_convergence_tradeoff.png
- Scatter plot: final cost vs convergence speed
- Weight values colour-coded to identify optimal regions

stability_performance_tradeoff.png
- Cost variance (stability) vs mean cost (performance)
- Identifies weights with best stability-performance balance
```

### 4. Distribution Analysis
```
weight_comparison_overview.png
- Mean total cost comparison across all weight configurations  
- Shows cost evolution, distributions, and summary statistics by weight value
- Y-axis properly labeled as "Mean Total Cost" (sum of costs across all resources)

performance_distributions.png
- Box plots of cost, load balance, and convergence speed
- Summary statistics table for all weight values
```

## Implementation Details ✓

### Implementation Location
```
resource_allocation_sim/experiments/weight_parameter_study.py
```
**Status**: Fully implemented with comprehensive analysis pipeline

### Main Class: `WeightParameterStudy`
Extends `ParameterSweepExperiment` with specialised analysis methods:

#### Core Analysis Methods
- `analyse_convergence_properties()`: Agent-level convergence analysis
- `analyse_performance_metrics()`: System performance evaluation  
- `extract_detailed_analysis_data()`: Comprehensive data extraction
- `generate_detailed_analysis()`: Statistical analysis and hypothesis testing
- `generate_comprehensive_report()`: Executive summary generation

#### Visualisation Methods
- `create_comprehensive_plots()`: Master plotting pipeline
- `_create_basic_sensitivity_plots()`: Parameter sensitivity analysis
- `_create_advanced_analysis_plots()`: Performance heatmaps and radar charts
- `_create_tradeoff_analysis_plots()`: Cost-convergence and stability analysis
- `_create_distribution_plots()`: Statistical distribution analysis

### Infrastructure Used
- **Base Class**: `ParameterSweepExperiment` (extends `BaseExperiment`)
- **Analysis Modules**: 
  - `evaluation.agent_analysis.analyse_agent_convergence()`
  - `evaluation.system_analysis.analyse_system_performance()`
  - `evaluation.metrics.calculate_entropy()`, `calculate_convergence_speed()`
- **Visualisation**: 
  - `visualisation.plots.plot_parameter_sensitivity()`
  - `visualisation.plots.plot_convergence_comparison()`
- **Statistical Analysis**: NumPy, Pandas, SciPy for comprehensive hypothesis testing

## Usage Instructions

### Quick Test Run (5 replications)
```bash
# Run from project root directory
python -m resource_allocation_sim.experiments.weight_parameter_study
```

### Custom Configuration
```python
from resource_allocation_sim.experiments.weight_parameter_study import run_weight_parameter_study

# Standard research run
study = run_weight_parameter_study(
    num_replications=30,        # Robust statistical sample
    output_dir="results/weight_study_full",
    show_plots=False           # Save plots without display
)

# Quick testing
test_study = run_weight_parameter_study(
    num_replications=5,         # Fast execution
    output_dir="test_results",
    show_plots=True            # Interactive display
)
```

### Output Structure
```
results/weight_parameter_study_YYYYMMDD_HHMMSS/
├── plots/                                    # All visualisations
│   ├── convergence_times_vs_weight.png
│   ├── final_entropy_vs_weight.png
│   ├── costs_vs_weight.png
│   ├── performance_heatmap.png
│   ├── agent_behaviour_radar.png
│   ├── cost_evolution_comparison.png
│   ├── cost_convergence_tradeoff.png
│   ├── stability_performance_tradeoff.png
│   ├── weight_comparison_overview.png
│   └── performance_distributions.png
├── weight_study_raw_data.csv                # Raw experimental data
├── weight_study_analysis.json               # Statistical analysis results
└── comprehensive_analysis_report.txt        # Executive summary
```

## Confirmed Results ✓

### Hypotheses Tested
1. **H1a**: Lower `w` values will show slower but more stable convergence
   -  **CONFIRMED**: Clear relationship observed
2. **H1b**: Higher `w` values will converge faster but with higher variance  
   -  **CONFIRMED**: Faster convergence, some variance increase
3. **H1c**: There exists an optimal `w` range that balances speed and stability
   -  **CONFIRMED**: Optimal range identified as 0.7-0.9
4. **H1d**: Very high `w` values (>0.8) will lead to unstable learning
   -  **REFUTED**: w=0.9 shows excellent performance and stability
5. **H1e**: Very low `w` values (<0.05) will be too slow to reach good solutions
   -  **CONFIRMED**: w=0.01, 0.05 show poor convergence within iteration limits

### Key Findings
- **Optimal Weight**: w = 0.9 provides best overall performance
- **Recommended Range**: 0.7-0.9 for most applications
- **Cost Sensitivity**: High - significant performance differences across weights
- **Convergence Pattern**: Clear monotonic improvement with higher weights (up to 0.9)
- **Stability**: Moderate weights (0.3-0.5) more stable but slower; w=0.9 good balance

### Practical Recommendations
1. **Default Recommendation**: Use w = 0.9 for most scenarios
2. **Conservative Approach**: Use w = 0.7 if stability is paramount
3. **Fast Prototyping**: w = 0.8-0.9 for quick convergence
4. **Avoid**: w < 0.2 (too slow) and w > 0.95 (potentially unstable) 