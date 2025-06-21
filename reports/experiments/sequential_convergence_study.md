# Sequential Convergence Hypothesis Study

## Hypothesis
**H2**: Agents with uniform initial distribution across available choices, sequentially converge to a degenerate distribution, with each agent becoming certain of a single choice one after the other.

## Research Questions
1. **Sequential Pattern**: Do agents converge one after another rather than simultaneously?
2. **Degenerate Distribution**: Do agents achieve near-certainty about a single resource choice?
3. **Convergence Order**: Is there a predictable order in which agents converge?
4. **Timing Analysis**: What is the temporal spacing between individual agent convergences?
5. **System Dynamics**: How does sequential convergence affect overall system performance?
6. **Trajectory Analysis**: How do agent probability distributions evolve through the simplex space?

## Key Concepts

### Degenerate Distribution
An agent has a **degenerate distribution** when:
- Maximum probability > 0.9 (high certainty)
- Entropy < 0.1 (low uncertainty)
- Clear preference for one resource over all others

### Sequential Convergence
**Sequential convergence** occurs when:
- Agent convergence times are well-separated
- Clear temporal ordering exists between agents
- Convergence events don't overlap significantly

### Barycentric Coordinate Analysis
With 3 resources, probability distributions can be visualised in **barycentric coordinates**:
- Each point represents a probability distribution over the 3 resources
- Vertices represent complete specialisation (degenerate distributions)
- Centre represents uniform distribution (1/3, 1/3, 1/3)
- Agent trajectories show evolution through the probability simplex

## Experimental Design

### Core Setup
- **Resources**: 3 (enables barycentric coordinate visualisation)
- **Agents**: 10 
- **Initial Distribution**: Uniform (0.333, 0.333, 0.333) for all agents
- **Iterations**: 2000 (longer to observe full convergence)
- **Replications**: 50 (to establish statistical patterns)
- **Weight Parameter**: 0.3 (moderate learning rate)
- **Capacity**: [1, 1, 1] (constant across resources)

### Step-by-Step Experimental Guide

#### Step 1: Quick Test Run 
```bash
# Fast test with minimal parameters
python -c "
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study
study = run_sequential_convergence_study(
    num_replications=5,     # Few replications for speed
    num_iterations=50,      # Short runs for quick testing
    show_plots=False        # No interactive plots
)
"
```

#### Step 2: Standard Research Run 
```bash
# Full research-quality analysis
python -c "
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study
study = run_sequential_convergence_study(
    num_replications=50,    # Robust statistical sample
    num_iterations=1000,    # Long enough for full convergence
    show_plots=False,       # Save plots without display
    output_dir='results/seq_conv_study_standard'
)
"
```

#### Step 3: High-Resolution Run 
```bash
# Maximum detail for publication
python -c "
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study
study = run_sequential_convergence_study(
    num_replications=100,   # High statistical power
    num_iterations=2000,    # Very long convergence period
    show_plots=False,
    output_dir='results/seq_conv_study_high_res'
)
"
```

#### Step 4: Custom Parameter Exploration
```python
# Example: Testing different convergence thresholds
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study

# Strict convergence criteria
strict_study = run_sequential_convergence_study(
    num_replications=30,
    num_iterations=1500,
    convergence_threshold_entropy=0.05,     # Very low entropy
    convergence_threshold_max_prob=0.95,    # Very high certainty
    output_dir='results/strict_convergence'
)

# Relaxed convergence criteria  
relaxed_study = run_sequential_convergence_study(
    num_replications=30,
    num_iterations=1500,
    convergence_threshold_entropy=0.2,      # Higher entropy allowed
    convergence_threshold_max_prob=0.7,     # Lower certainty required
    output_dir='results/relaxed_convergence'
)
```

#### Step 5: Interactive Analysis with Plots
```python
# Run with interactive plot display
study = run_sequential_convergence_study(
    num_replications=20,
    num_iterations=500,
    show_plots=True,        # Display plots interactively
    output_dir='results/interactive_analysis'
)

# Access results programmatically
print(f"Hypothesis Support: {study.analysis['hypothesis_support']['overall_support']}")
print(f"Sequential Index: {study.analysis['hypothesis_support']['evidence_strength']['sequential_index']:.3f}")
```

#### Parameter Guidelines

**Num Replications:**
- **5-10**: Quick testing and development
- **30-50**: Standard research analysis  
- **100+**: High-precision publication studies

**Num Iterations:**
- **50-100**: Fast testing (may not reach full convergence)
- **500-1000**: Standard analysis (good convergence)
- **2000+**: Comprehensive analysis (guaranteed convergence)

**Convergence Thresholds:**
- **Strict** (entropy < 0.05, max_prob > 0.95): High-certainty states only
- **Standard** (entropy < 0.1, max_prob > 0.9): Balanced criteria
- **Relaxed** (entropy < 0.2, max_prob > 0.7): Earlier convergence detection

### Multi-Parameter Experimental Configurations

#### Varying Number of Resources

**2 Resources (Binary Choice)**
```python
# Simplified binary choice scenario
from resource_allocation_sim.experiments.sequential_convergence_study import SequentialConvergenceStudy

study_2res = SequentialConvergenceStudy(
    results_dir="results/2_resources",
    experiment_name="seq_conv_2_resources"
)
study_2res.base_config.num_resources = 2
study_2res.base_config.relative_capacity = [0.5, 0.5]
study_2res.base_config.num_iterations = 800

results_2res = study_2res.run_experiment(num_episodes=50)
```

**4-5 Resources (Extended Choice)**
```python
# More complex choice scenarios
study_4res = SequentialConvergenceStudy(
    results_dir="results/4_resources",
    experiment_name="seq_conv_4_resources"
)
study_4res.base_config.num_resources = 4
study_4res.base_config.relative_capacity = [0.25, 0.25, 0.25, 0.25]
study_4res.base_config.num_iterations = 1200  # Longer convergence needed

study_5res = SequentialConvergenceStudy(
    results_dir="results/5_resources",
    experiment_name="seq_conv_5_resources"
)
study_5res.base_config.num_resources = 5
study_5res.base_config.relative_capacity = [0.2, 0.2, 0.2, 0.2, 0.2]
study_5res.base_config.num_iterations = 1500 

**Expected Behaviour by Resource Count:**
- **2 Resources**: Faster convergence, stronger sequential patterns
- **3 Resources**: Balanced complexity, ideal for barycentric analysis
- **4-5 Resources**: Slower convergence, potentially weaker sequential patterns
- **6+ Resources**: May require much longer iterations or different thresholds

#### Varying Number of Agents

**Small Groups (5 agents)**
```python
# Minimal group dynamics
study_5agents = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=600,     # Faster convergence with fewer agents
    output_dir="results/5_agents"
)
study_5agents.base_config.num_agents = 5
study_5agents.base_config.relative_capacity = [0.6, 0.6, 0.6]  # Higher per-agent capacity
```

**Medium Groups (15 agents)**
```python
# Standard group size
study_15agents = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=1200,
    output_dir="results/15_agents"
)
study_15agents.base_config.num_agents = 15
study_15agents.base_config.relative_capacity = [0.33, 0.33, 0.33]  # Standard capacity
```

**Large Groups (25+ agents)**
```python
# Large group dynamics
study_25agents = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=2000,    # Longer convergence needed
    output_dir="results/25_agents"
)
study_25agents.base_config.num_agents = 25
study_25agents.base_config.relative_capacity = [0.4, 0.4, 0.4]  # Slightly higher capacity
```

**Expected Behaviour by Agent Count:**
- **5 Agents**: Very clear sequential patterns, fast convergence
- **10 Agents**: Good balance, clear patterns (default recommendation)
- **15-20 Agents**: More complex dynamics, potentially longer sequences
- **25+ Agents**: May show sub-group formations, requires longer analysis

#### Varying Weight Parameters

**Low Learning Rate (w = 0.1)**
```python
# Slow, careful learning
study_low_weight = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=3000,    # Much longer needed for convergence
    output_dir="results/low_weight"
)
study_low_weight.base_config.weight = 0.1
```

**Moderate Learning Rate (w = 0.3)**
```python
# Balanced learning speed
study_moderate_weight = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=1000,
    output_dir="results/moderate_weight"
)
study_moderate_weight.base_config.weight = 0.3  

**High Learning Rate (w = 0.7)**
```python
# Fast learning
study_high_weight = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=500,     # Faster convergence expected
    output_dir="results/high_weight"
)
study_high_weight.base_config.weight = 0.7
```

**Very High Learning Rate (w = 0.9)**
```python
# Aggressive learning
study_very_high_weight = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=300,     # Very fast convergence
    output_dir="results/very_high_weight"
)
study_very_high_weight.base_config.weight = 0.9
```

**Expected Behaviour by Weight:**
- **w = 0.1**: Gradual convergence, may not show strong sequential patterns
- **w = 0.3**: Balanced, good sequential patterns (recommended)
- **w = 0.7**: Fast convergence, strong sequential patterns
- **w = 0.9**: Very fast, potentially simultaneous convergence

#### Varying Capacity Configurations

**Balanced Capacity (Equal Resources)**
```python
# Default balanced scenario
study_balanced = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=1000,
    output_dir="results/balanced_capacity"
)
study_balanced.base_config.relative_capacity = [0.33, 0.33, 0.33]
```

**Unbalanced Capacity (Heterogeneous Resources)**
```python
# One dominant resource
study_unbalanced = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=1000,
    output_dir="results/unbalanced_capacity"
)
study_unbalanced.base_config.relative_capacity = [0.6, 0.2, 0.2]  # Resource 1 dominates
```

**Constrained Capacity (Resource Scarcity)**
```python
# Limited total capacity
study_constrained = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=1500,    # May need longer due to competition
    output_dir="results/constrained_capacity"
)
study_constrained.base_config.relative_capacity = [0.2, 0.2, 0.2]  # Only 60% total capacity
```

**Abundant Capacity (Low Competition)**
```python
# High capacity, low competition
study_abundant = run_sequential_convergence_study(
    num_replications=50,
    num_iterations=800,
    output_dir="results/abundant_capacity"
)
study_abundant.base_config.relative_capacity = [0.8, 0.8, 0.8]  
```

**Expected Behaviour by Capacity:**
- **Balanced [0.33, 0.33, 0.33]**: Clear competition, good sequential patterns
- **Unbalanced [0.6, 0.2, 0.2]**: Agents prefer high-capacity resources, may reduce sequential patterns
- **Constrained [0.2, 0.2, 0.2]**: High competition, strong sequential patterns due to exclusion
- **Abundant [0.8, 0.8, 0.8]**: Low competition, may reduce sequential patterns

#### Systematic Parameter Comparison

**Comparative Analysis Across Parameters**
```python
# Run systematic comparison across multiple parameter configurations
from resource_allocation_sim.experiments.sequential_convergence_study import SequentialConvergenceStudy

parameter_configs = [
    # Baseline
    {"name": "baseline", "resources": 3, "agents": 10, "weight": 0.3, "capacity": [0.33, 0.33, 0.33], "iterations": 1000},
    
    # Resource variations
    {"name": "2_resources", "resources": 2, "agents": 10, "weight": 0.3, "capacity": [0.5, 0.5], "iterations": 800},
    {"name": "4_resources", "resources": 4, "agents": 10, "weight": 0.3, "capacity": [0.25, 0.25, 0.25, 0.25], "iterations": 1200},
    
    # Agent variations  
    {"name": "5_agents", "resources": 3, "agents": 5, "weight": 0.3, "capacity": [0.6, 0.6, 0.6], "iterations": 600},
    {"name": "20_agents", "resources": 3, "agents": 20, "weight": 0.3, "capacity": [0.35, 0.35, 0.35], "iterations": 1500},
    
    # Weight variations
    {"name": "low_weight", "resources": 3, "agents": 10, "weight": 0.1, "capacity": [0.33, 0.33, 0.33], "iterations": 2500},
    {"name": "high_weight", "resources": 3, "agents": 10, "weight": 0.7, "capacity": [0.33, 0.33, 0.33], "iterations": 500},
    
    # Capacity variations
    {"name": "constrained", "resources": 3, "agents": 10, "weight": 0.3, "capacity": [0.2, 0.2, 0.2], "iterations": 1200},
    {"name": "abundant", "resources": 3, "agents": 10, "weight": 0.3, "capacity": [0.8, 0.8, 0.8], "iterations": 800},
]

results_comparison = {}

for config in parameter_configs:
    print(f"Running {config['name']} configuration...")
    
    study = SequentialConvergenceStudy(
        results_dir=f"results/comparison_{config['name']}",
        experiment_name=f"seq_conv_comparison_{config['name']}"
    )
    
    # Configure parameters
    study.base_config.num_resources = config['resources']
    study.base_config.num_agents = config['agents']
    study.base_config.weight = config['weight']
    study.base_config.relative_capacity = config['capacity']
    study.base_config.num_iterations = config['iterations']
    
    # Run experiment
    results = study.run_experiment(num_episodes=30)  # Moderate replications for comparison
    results_comparison[config['name']] = {
        'study': study,
        'config': config,
        'results': results
    }

# Generate comparison report
print("\n" + "="*80)
print("PARAMETER COMPARISON RESULTS")
print("="*80)

for name, data in results_comparison.items():
    support = data['results']['analysis']['hypothesis_support']
    overall_support = support.get('overall_support', 'undetermined')
    
    if 'evidence_strength' in support:
        seq_index = support['evidence_strength'].get('sequential_index', 'N/A')
        deg_prop = support['evidence_strength'].get('degeneracy_proportion', 'N/A')
        
        print(f"{name.upper():<15} | Support: {overall_support.upper():<10} | "
              f"Sequential: {seq_index:.3f} | Degeneracy: {deg_prop:.3f}")
    else:
        print(f"{name.upper():<15} | Support: {overall_support.upper()}")
```

#### Best Practices for Parameter Selection

**1. Resource Count Guidelines:**
- **2 Resources**: Good for initial hypothesis testing, clearest sequential patterns
- **3 Resources**: Optimal balance, enables barycentric visualisation
- **4-5 Resources**: More realistic scenarios, requires longer convergence periods
- **6+ Resources**: Complex scenarios, may need relaxed convergence criteria

**2. Agent Count Guidelines:**
- **5-8 Agents**: Clear patterns, suitable for detailed analysis
- **10-15 Agents**: Standard group sizes, good statistical power
- **20+ Agents**: Complex group dynamics, may show sub-patterns

**3. Weight Parameter Guidelines:**
- **w = 0.1**: For studying gradual learning, requires 2000+ iterations
- **w = 0.3**: Balanced learning rate, good sequential patterns (recommended)
- **w = 0.5-0.7**: Fast learning, strong patterns, shorter experiments
- **w = 0.9**: Very aggressive learning, may lead to simultaneous convergence

**4. Capacity Configuration Guidelines:**
- **Balanced**: Default choice for hypothesis testing
- **Unbalanced**: Tests robustness of sequential patterns under resource asymmetry
- **Constrained**: Amplifies competition effects, strengthens sequential patterns
- **Abundant**: Tests hypothesis under low-competition conditions

**5. Iteration Count Guidelines:**
```python
# Recommended iteration counts based on other parameters
def recommend_iterations(num_resources, num_agents, weight):
    base_iterations = 1000
    
    # Adjust for resource complexity
    resource_multiplier = max(1.0, num_resources / 3.0)
    
    # Adjust for agent count  
    agent_multiplier = max(1.0, num_agents / 10.0)
    
    # Adjust for learning rate
    weight_multiplier = max(0.3, 1.0 / weight)
    
    recommended = int(base_iterations * resource_multiplier * agent_multiplier * weight_multiplier)
    
    return min(recommended, 5000)  # Cap at reasonable maximum

# Example usage
iterations_needed = recommend_iterations(num_resources=4, num_agents=15, weight=0.2)
print(f"Recommended iterations: {iterations_needed}")
```

**6. Convergence Threshold Guidelines:**
```python
# Adaptive thresholds based on experimental parameters
def get_adaptive_thresholds(num_resources, num_agents, weight):
    # More resources = higher entropy threshold (harder to achieve low entropy)
    if num_resources <= 2:
        entropy_threshold = 0.05
        max_prob_threshold = 0.95
    elif num_resources <= 3:
        entropy_threshold = 0.1
        max_prob_threshold = 0.9
    else:
        entropy_threshold = 0.15
        max_prob_threshold = 0.85
    
    # More agents = potentially slower convergence
    if num_agents > 20:
        entropy_threshold *= 1.2
        max_prob_threshold *= 0.95
    
    # Lower weights = potentially less decisive convergence
    if weight < 0.3:
        entropy_threshold *= 1.3
        max_prob_threshold *= 0.9
    
    return entropy_threshold, max_prob_threshold

# Example usage
entropy_thresh, prob_thresh = get_adaptive_thresholds(4, 15, 0.2)
print(f"Adaptive thresholds: entropy < {entropy_thresh:.3f}, max_prob > {prob_thresh:.3f}")
```

### Variables to Study
1. **Fixed Parameters**: Ensure controlled conditions
   - Same initial state for all agents (centre of ternary plot)
   - Same environment parameters
   - Same learning parameters

2. **Measured Variables**:
   - Individual agent convergence times
   - Final probability distributions (positions on ternary plot)
   - Entropy evolution patterns
   - Resource selection patterns
   - Trajectory paths through probability simplex

### Convergence Criteria
An agent is considered **converged** when:
```python
entropy < 0.1 AND max_probability > 0.9
```
This corresponds to reaching the edge regions of the ternary plot.

## Metrics and Analysis

### 1. Individual Agent Metrics
- **Convergence Time**: Iteration when agent reaches degenerate state
- **Degeneracy Score**: max(probabilities) at convergence
- **Preferred Resource**: Resource with highest final probability (vertex reached)
- **Convergence Speed**: Rate of entropy decrease
- **Trajectory Length**: Total distance travelled in probability space

### 2. System-Level Metrics  
- **Convergence Sequence**: Ordered list of agent convergence times
- **Sequential Index**: Measure of how sequential vs simultaneous convergence is
- **Convergence Spread**: Time difference between first and last agent to converge
- **Overlap Coefficient**: How much convergence windows overlap

### 3. Temporal Pattern Analysis
- **Inter-convergence Intervals**: Time between successive agent convergences
- **Convergence Rate**: Number of agents converging per time window
- **Sequential Correlation**: Statistical measure of ordering consistency

### 4. Resource Allocation Patterns
- **Resource Specialisation**: How agents distribute across resources (vertex occupation)
- **Load Balancing**: Evenness of final resource allocation
- **Competition Effects**: Whether agents avoid resources chosen by others

### 5. Spatial Pattern Analysis (Barycentric)
- **Trajectory Clustering**: Do agents follow similar paths?
- **Convergence Zones**: Which vertices are most commonly reached?
- **Path Directness**: Do agents move directly to vertices or wander?

## Required Visualisations

### Figure 1: Convergence Timeline Analysis
```
(a) Agent Convergence Timeline
    - Horizontal bars showing convergence intervals for each agent
    - Clear start/end times for each agent's convergence period
    
(b) Cumulative Convergence Plot  
    - Step function showing number of converged agents over time
    - Demonstrates sequential vs simultaneous patterns
    
(c) Entropy Evolution Heatmap
    - Agents on y-axis, time on x-axis, entropy as colour
    - Shows individual convergence patterns
    
(d) Sequential Index Distribution
    - Histogram of sequential indices across replications
    - Shows consistency of sequential patterns
```

### Figure 2: Probability Evolution Analysis  
```
(a) Individual Probability Trajectories
    - Line plots for each agent showing probability evolution
    - Separate lines for each of the 3 resources
    
(b) Resource Preference Heatmap
    - Agents vs Resources, final probabilities as colour intensity
    - Shows specialisation patterns
    
(c) Convergence Order Matrix
    - Shows which agent converged when across replications
    - Identifies consistent ordering patterns
    
(d) Degeneracy Score Distribution
    - Histogram of max probabilities at convergence
    - Tests if agents truly reach degenerate states
```

### Figure 3: System Dynamics Analysis
```
(a) Resource Competition Timeline
    - Shows how resource selection evolves over time
    - Stacked area chart or similar
    
(b) Performance During Convergence
    - System cost, load balance, efficiency over time
    - Marked with individual convergence events
    
(c) Agent Interaction Effects
    - Correlation between agent choices
    - Network-style visualisation if appropriate
    
(d) Convergence Speed vs Order
    - Scatter plot: convergence order vs convergence speed
    - Tests if later agents converge faster/slower
```

### Figure 4: Statistical Analysis 
```
(a) Convergence Time Distribution
    - Histogram with mean and median markers
    - Tests temporal clustering vs uniform distribution
    
(b) Sequential Pattern Validation  
    - Box plot of sequential indices across replications
    - Reference lines for random (0.5) and observed mean
    - Statistical significance indicators
    
(c) Replication Consistency Analysis
    - Scatter plot: sequential index vs replication number
    - Mean line showing pattern consistency
    - Variance assessment across runs
    
(d) Comprehensive Hypothesis Analysis Table 
    - HYPOTHESIS METRICS: Sequential index, degeneracy proportion, convergence times
    - STATISTICAL TESTS: t-test, KS-test, binomial test with ✓/✗ indicators  
    - HYPOTHESIS SUPPORT: Overall assessment (STRONG/MODERATE/WEAK/NONE)
    - Enhanced formatting with proper ∞ symbol for perfect sequential cases
```

### Figure 5: Barycentric Coordinate Analysis 
```
(a) All Agent Trajectories
    - Single ternary plot showing all agent paths
    - Color-coded by time progression
    - Start (centre) and end points (vertices) marked
    
(b) Final Distribution Positions
    - Ternary plot showing final positions of all agents
    - Demonstrates convergence to vertices (degeneracy)
    - Colour-coded by agent ID
    
(c) Individual Agent Trajectories
    - 2x2 subplot showing detailed trajectories for first 4 agents
    - Each subplot shows single agent's path through simplex
    - Demonstrates individual convergence patterns
    
(d) Trajectory Analysis
    - Path statistics: directness, speed, clustering
    - Distance from centre over time
    - Convergence zone analysis
```

## Implementation Plan

### File Structure
```
resource_allocation_sim/experiments/sequential_convergence_study.py
```

### Key Classes and Methods 
```python
class SequentialConvergenceStudy(BaseExperiment):
    def analyse_convergence_sequence_from_episodes()    # Core analysis pipeline
    def analyse_individual_convergence()                # Per-agent convergence analysis  
    def calculate_sequential_index()                    # Sequential pattern measurement
    def find_convergence_time()                         # Individual agent convergence detection
    def perform_statistical_tests()                     # Hypothesis testing suite
    def evaluate_hypothesis_support()                   # Overall assessment framework
    
    # Visualization Functions
    def create_convergence_timeline_plots()             # Figure 1: Timeline analysis
    def create_probability_evolution_plots()            # Figure 2: Probability evolution
    def create_system_dynamics_plots()                  # Figure 3: System dynamics
    def create_statistical_analysis_plots()             # Figure 4: Statistical analysis 
    def create_barycentric_trajectory_plots()           # Figure 5: Barycentric analysis
```

### System Analysis Functions Used 
```python
# From system_analysis.py
def perform_convergence_statistical_tests()            # Core statistical tests
def test_sequential_convergence_hypothesis()           # Hypothesis-specific tests
def calculate_system_convergence_metrics()             # System-level metrics
def evaluate_hypothesis_support()                      # Support evaluation framework
```

### Metrics and Analysis Functions
```python 
# From metrics.py
def calculate_probability_entropy()                    # Agent probability entropy 
def calculate_entropy()                                # Resource consumption entropy

# From agent_analysis.py  
def analyse_agent_convergence()                        # Individual agent patterns
def plot_visited_probabilities()                       # Barycentric trajectories 

# From system_analysis.py
def analyse_system_performance()                       # System-level effects
```

### Implemented Analysis Features 
-  Sequential index calculation and validation
-  Convergence order temporal analysis  
-  Statistical hypothesis testing suite
-  Degeneracy proportion measurement
-  Barycentric coordinate trajectory analysis
-  Comprehensive hypothesis support evaluation
-  Enhanced statistical analysis display with proper ∞ handling
-  System-level convergence pattern metrics

## Testable Predictions

### Primary Hypotheses
1. **H2a**: Agent convergence times will be significantly non-overlapping
2. **H2b**: >90% of agents will achieve max probability > 0.9 (reach ternary vertices)
3. **H2c**: Convergence order will be consistent across replications
4. **H2d**: Inter-convergence intervals will be roughly uniform
5. **H2e**: Agent trajectories will move directly from centre to vertices 

### Alternative Hypotheses
1. **H2a'**: Agents converge simultaneously (null hypothesis)
2. **H2b'**: Agents maintain diverse probability distributions (stay in ternary centre)
3. **H2c'**: Convergence order is random across replications
4. **H2d'**: Convergence times cluster rather than spread evenly
5. **H2e'**: Agent trajectories wander randomly through probability space 

## Statistical Tests & Metrics 

### Core Hypothesis Metrics
1. **Sequential Index**: Measures how sequential vs simultaneous convergence is
   - **Range**: 0.0 (simultaneous) to 1.0 (perfectly sequential)
   - **Calculation**: Based on temporal spacing of convergence events
   - **Threshold**: > 0.5 indicates more sequential than random

2. **Degeneracy Proportion**: Fraction of agents achieving degenerate distributions
   - **Definition**: Agents with max probability > 0.9 (entropy < 0.1)
   - **Target**: > 90% for strong hypothesis support
   - **Measurement**: Proportion reaching ternary plot vertices

3. **Convergence Time Statistics**: Temporal patterns of individual agent convergence
   - **Mean Convergence Time**: Average time to reach degenerate state
   - **Convergence Spread**: Range between first and last agent convergence
   - **Standard Deviation**: Variability in convergence timing

### Statistical Hypothesis Tests

#### Test 1: Sequential Pattern Analysis
- **Test**: One-sample t-test
- **Null Hypothesis (H₀)**: Sequential index = 0.5 (random convergence)
- **Alternative (H₁)**: Sequential index > 0.5 (sequential convergence)
- **Significance Level**: α = 0.05
- **Perfect Case**: t-statistic = ∞ when all replications show identical sequential patterns

#### Test 2: Convergence Time Distribution
- **Test**: Kolmogorov-Smirnov test
- **Null Hypothesis (H₀)**: Convergence times are uniformly distributed
- **Alternative (H₁)**: Convergence times follow non-uniform pattern
- **Purpose**: Test if agents converge at regular intervals vs clustered timing

#### Test 3: Degeneracy Achievement
- **Test**: Binomial test
- **Null Hypothesis (H₀)**: Proportion achieving degeneracy = expected threshold
- **Alternative (H₁)**: Proportion ≠ expected threshold  
- **Expected Threshold**: 0.9 (90% of agents)
- **Purpose**: Validate high degeneracy achievement rate

### Comprehensive Evaluation Framework

#### Hypothesis Support Levels
1. **Strong Support** (≥80% criteria met):
   - Sequential index > 0.5 with statistical significance
   - Degeneracy proportion > 0.8
   - Consistent patterns across replications

2. **Moderate Support** (≥60% criteria met):
   - Sequential index > 0.5 but may lack statistical significance
   - Degeneracy proportion 0.6-0.8
   - Some pattern consistency

3. **Weak Support** (≥40% criteria met):
   - Sequential index marginally > 0.5
   - Degeneracy proportion 0.4-0.6
   - Limited pattern consistency

4. **No Support** (<40% criteria met):
   - Sequential index ≤ 0.5
   - Low degeneracy proportion
   - Random or inconsistent patterns

### Enhanced Statistical Analysis Display

#### Figure 4d: Comprehensive Hypothesis Analysis Table
```
HYPOTHESIS METRICS
├── Sequential Index Mean      │ Value ± σ       │ Sequential/Random
├── Degeneracy Proportion      │ Value (n/total) │ High/Low
└── Mean Convergence Time      │ Value ± σ       │ Fast/Slow

STATISTICAL TESTS  
├── Sequential Pattern        │ t-statistic     │ p-value │ ✓/✗
├── Convergence Uniformity    │ KS-statistic    │ p-value │ ✓/✗  
└── Degeneracy Proportion     │ Binomial stat   │ p-value │ ✓/✗

HYPOTHESIS SUPPORT
└── Overall Assessment        │ STRONG/MODERATE/WEAK/NONE │ ✓/✗
```

### System-Level Convergence Metrics
- **Temporal Coordination**: Measure of how well-coordinated convergence timing is
- **Convergence Efficiency**: Agents converged per time unit
- **Pattern Consistency**: Variance in sequential indices across replications
- **Replication Success Rate**: Proportion of runs showing sequential pattern

## Success Criteria
1. **Clear Sequential Pattern**: Statistically significant evidence of sequential convergence
2. **High Degeneracy**: >90% of agents achieve degenerate distributions (reach vertices)
3. **Consistent Ordering**: Similar convergence sequences across replications
4. **Temporal Separation**: Non-overlapping convergence intervals
5. **Directed Trajectories**: Agents move efficiently from centre to vertices 

## Expected Outcomes
- **Strong Support**: Clear sequential pattern with directed trajectories to vertices
- **Partial Support**: Sequential pattern but with some simultaneity or wandering
- **Weak Support**: Some sequential elements but mostly simultaneous convergence
- **No Support**: Random convergence patterns, trajectories remain in centre

## Implementation Command

### Direct Module Execution
```bash
# Run with current default parameters (100 replications, 1000 iterations)
python -m resource_allocation_sim.experiments.sequential_convergence_study
```

### Programmatic Execution (Recommended)

#### Basic Usage
```python
from resource_allocation_sim.experiments.sequential_convergence_study import run_sequential_convergence_study

# Standard research run
study = run_sequential_convergence_study(
    num_replications=50,        # Number of independent runs
    num_iterations=1000,        # Iterations per simulation
    show_plots=False,           # Save plots without display
    output_dir="results/sequential_study"
)

# Print key results
print(f"Hypothesis Support: {study.analysis['hypothesis_support']['overall_support']}")
```

#### Advanced Configuration
```python
# Publication-quality analysis with custom thresholds
study = run_sequential_convergence_study(
    num_replications=100,                           # High statistical power
    num_iterations=2000,                            # Extended convergence period
    output_dir="results/publication_study",         # Custom output directory
    show_plots=False,                               # Batch processing mode
    convergence_threshold_entropy=0.1,              # Entropy threshold for convergence
    convergence_threshold_max_prob=0.9              # Max probability threshold
)

# Access detailed results
results = study.get_results()
analysis = results['analysis']

print(f"Sequential Index Mean: {analysis['sequential_indices']['mean']:.3f}")
print(f"Degeneracy Proportion: {analysis['degeneracy_scores']['proportion']:.3f}")
print(f"Mean Convergence Time: {analysis['convergence_times']['mean']:.1f}")
```

#### Quick Testing
```python
# Fast testing during development
test_study = run_sequential_convergence_study(
    num_replications=5,         # Minimal replications
    num_iterations=100,         # Short runs
    show_plots=True,            # Interactive display
    output_dir="test_results"
)
```

#### Batch Processing Multiple Configurations
```python
# Compare different parameter settings
configurations = [
    {"name": "strict", "entropy": 0.05, "max_prob": 0.95},
    {"name": "standard", "entropy": 0.1, "max_prob": 0.9},
    {"name": "relaxed", "entropy": 0.2, "max_prob": 0.7}
]

results = {}
for config in configurations:
    print(f"Running {config['name']} configuration...")
    study = run_sequential_convergence_study(
        num_replications=30,
        num_iterations=1000,
        convergence_threshold_entropy=config['entropy'],
        convergence_threshold_max_prob=config['max_prob'],
        output_dir=f"results/{config['name']}_study",
        show_plots=False
    )
    results[config['name']] = study

# Compare results across configurations
for name, study in results.items():
    support = study.analysis['hypothesis_support']['overall_support']
    seq_index = study.analysis['hypothesis_support']['evidence_strength']['sequential_index']
    print(f"{name.title()}: {support.upper()} support, Sequential Index: {seq_index:.3f}")
```

### Command Line Arguments (Future Enhancement)
```bash
# Planned future syntax for command-line configuration
python -m resource_allocation_sim.experiments.sequential_convergence_study \
    --replications 50 \
    --iterations 1000 \
    --output-dir "results/custom_study" \
    --convergence-entropy 0.1 \
    --convergence-max-prob 0.9
```

### Output Structure
After running, results are saved in the specified output directory:
```
results/
└── sequential_convergence_study_YYYYMMDD_HHMMSS/
    ├── experiment_results.json           # Raw numerical results
    ├── analysis_results.json             # Statistical analysis
    ├── convergence_timeline.png          # Figure 1: Timeline plots
    ├── probability_evolution.png         # Figure 2: Evolution plots  
    ├── system_dynamics.png               # Figure 3: System analysis
    ├── statistical_analysis.png          # Figure 4: Statistics
    ├── barycentric_trajectories.png      # Figure 5a: All trajectories
    ├── barycentric_final_positions.png   # Figure 5b: Final positions
    └── barycentric_individual.png        # Figure 5c: Individual paths
```

## Barycentric Coordinate Interpretation

### Key Features
- **Centre (0.33, 0.33, 0.33)**: Uniform distribution, maximum uncertainty
- **Vertices**: Complete specialisation on one resource
  - Top vertex: Resource 1 dominance
  - Bottom-left: Resource 2 dominance  
  - Bottom-right: Resource 3 dominance
- **Edges**: Two-resource combinations
- **Distance from centre**: Measure of specialisation/degeneracy

### Expected Trajectory Patterns
1. **Sequential Convergence**: Agents leave centre at different times, move to different vertices
2. **Simultaneous Convergence**: All agents leave centre together
3. **Directed Movement**: Straight lines from centre to vertices
4. **Wandering**: Circular or random movements before convergence

## Results Interpretation Guide 

### Statistical Test Outcomes

#### Sequential Pattern t-test Results
- **t = ∞, p = 0.000**: Perfect sequential pattern (all replications identical)
- **t > 2, p < 0.05**: Strong evidence for sequential convergence
- **t = 1-2, p = 0.05-0.10**: Moderate evidence for sequential pattern
- **t < 1, p > 0.10**: Weak or no evidence for sequential pattern

#### Degeneracy Proportion Results  
- **Proportion = 1.000**: All agents achieve perfect degeneracy (reach vertices)
- **Proportion > 0.900**: Strong support for degenerate convergence hypothesis
- **Proportion = 0.600-0.900**: Moderate support for degeneracy
- **Proportion < 0.600**: Weak support, agents remain in ternary centre

#### Warning Messages - Normal Behaviour
- **"Precision loss in moment calculation"**: Expected when sequential indices are identical
- **RuntimeWarning about catastrophic cancellation**: Normal for perfect sequential patterns
- **t-statistic = inf**: Indicates zero variance = perfect consistency (GOOD result!)

### Hypothesis Support Interpretation
- **STRONG**: Sequential index ≈ 1.0, degeneracy proportion ≈ 1.0, all tests significant
- **MODERATE**: Sequential index > 0.7, degeneracy proportion > 0.7, most tests significant  
- **WEAK**: Sequential index > 0.5, degeneracy proportion > 0.5, some tests significant
- **NONE**: Sequential index ≤ 0.5, low degeneracy proportion, tests not significant

### Performance Benchmarks
- **Fast Convergence**: Mean time < 100 iterations
- **High Sequential Index**: > 0.8 indicates strong temporal separation
- **Excellent Degeneracy**: > 0.9 shows most agents reach vertices
- **Consistent Patterns**: Low standard deviation across replications