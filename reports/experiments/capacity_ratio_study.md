# Capacity Ratio Study - Hypothesis 4

## Hypothesis
**H4**: Asymmetric capacity configurations create predictable agent specialisation patterns where high-capacity resources attract early sequential convergers whilst low-capacity resources either remain underutilised or attract late specialised convergers, resulting in hierarchical resource utilisation that reflects the capacity hierarchy.

## Research Questions
1. **Hierarchical Specialisation**: Do agents preferentially converge to high-capacity resources?
2. **Capacity-Utilisation Correlation**: Is there a positive correlation between resource capacity and final utilisation?
3. **Convergence Timing**: Do agents converge to high-capacity resources earlier than low-capacity ones?
4. **Performance Optimisation**: Do moderate asymmetries optimise system performance compared to extreme configurations?
5. **Specialisation Patterns**: How do different capacity configurations affect agent specialisation in ternary space?
6. **Load Balancing**: How does capacity asymmetry affect load distribution quality?

## Key Concepts

### Capacity-Driven Specialisation
**Capacity-driven specialisation** occurs when:
- High-capacity resources attract more agents
- Agent preferences correlate with resource capacities
- Hierarchical utilisation patterns emerge reflecting capacity hierarchy
- Early convergers prefer high-capacity resources

### Hierarchical Resource Utilisation
**Hierarchical utilisation** manifests as:
- Resource popularity ranking matches capacity ranking
- Predictable agent distribution across resources
- Consistent ordering across experimental replications
- Statistical correlation between capacity and utilisation

### Capacity Asymmetry Effects
**Capacity asymmetry** influences:
- **System Performance**: Cost efficiency and convergence rates
- **Load Distribution**: Balance quality and resource utilisation
- **Agent Behaviour**: Specialisation strength and convergence timing
- **Stability**: Consistency of hierarchical patterns

### Ternary Space Analysis
With 3 resources, capacity effects can be visualised in **ternary coordinates**:
- **Vertex Attraction**: High-capacity vertices attract more agents
- **Trajectory Bias**: Agent paths biased towards high-capacity vertices
- **Spatial Clustering**: Agents cluster near high-capacity resource vertices
- **Gradient Effects**: Capacity gradients create attraction fields

## Experimental Design

### Core Setup
- **Resources**: 3 (enables ternary coordinate visualisation)
- **Agents**: 10 
- **Learning Rate**: 0.3 (moderate learning rate for clear patterns)
- **Iterations**: 1000 (sufficient for convergence analysis)
- **Replications**: 50 (robust statistical analysis)
- **Initial Distribution**: Uniform (centre of ternary plot)

### Capacity Configurations Tested

#### 1. Symmetric Baseline
```python
[0.33, 0.33, 0.33]  # Perfect balance
```

#### 2. Moderate Asymmetry
```python
[0.50, 0.30, 0.20]  # Graduated hierarchy
[0.40, 0.40, 0.20]  # Two dominant, one minor
```

#### 3. High Asymmetry  
```python
[0.60, 0.30, 0.10]  # Strong primary resource
[0.70, 0.20, 0.10]  # Very dominant primary
```

#### 4. Extreme Asymmetry
```python
[0.80, 0.15, 0.05]  # Near-monopolistic primary
```

#### 5. Specialised Configurations
```python
[0.50, 0.25, 0.25]  # Single dominant resource
[0.45, 0.45, 0.10]  # Binary dominance
[0.60, 0.25, 0.15]  # Graduated three-tier
[0.55, 0.35, 0.10]  # Strong binary preference
```

### Step-by-Step Experimental Guide

#### Step 1: Quick Test Run
```bash
# Fast test with minimal parameters
python -c "
from resource_allocation_sim.experiments.capacity_ratio_study import run_capacity_ratio_study
study = run_capacity_ratio_study(
    num_replications=5,     # Few replications for speed
    show_plots=False        # No interactive plots
)
"
```

#### Step 2: Standard Research Run
```bash
# Full research-quality analysis
python -c "
from resource_allocation_sim.experiments.capacity_ratio_study import run_capacity_ratio_study
study = run_capacity_ratio_study(
    num_replications=50,    # Robust statistical sample
    show_plots=False,       # Save plots without display
    output_dir='results/capacity_study_standard'
)
"
```

#### Step 3: High-Resolution Analysis
```bash
# Maximum detail for publication
python -c "
from resource_allocation_sim.experiments.capacity_ratio_study import run_capacity_ratio_study
study = run_capacity_ratio_study(
    num_replications=100,   # High statistical power
    show_plots=False,
    output_dir='results/capacity_study_high_res'
)
"
```

#### Step 4: Custom Configuration Testing
```python
# Example: Testing specific capacity scenarios
from resource_allocation_sim.experiments.capacity_ratio_study import CapacityRatioStudy

# Real-world inspired configurations
real_world_study = CapacityRatioStudy(
    capacity_configurations=[
        [0.33, 0.33, 0.33],  # Baseline: equal infrastructure
        [0.50, 0.30, 0.20],  # Urban-suburban-rural distribution
        [0.60, 0.25, 0.15],  # Metropolitan dominance
        [0.45, 0.35, 0.20],  # Dual-city scenario
        [0.70, 0.20, 0.10],  # Megacity concentration
    ],
    results_dir="results/real_world_scenarios",
    experiment_name="real_world_capacity_scenarios"
)

results = real_world_study.run_experiment(num_episodes=30)
```

#### Step 5: Interactive Analysis with Plots
```python
# Run with interactive plot display
study = run_capacity_ratio_study(
    num_replications=20,
    show_plots=True,        # Display plots interactively
    output_dir='results/interactive_analysis'
)

# Access results programmatically
analysis = study.analysis_results
print(f"Hypothesis Support: {analysis['hypothesis_support']['overall_support']}")
print(f"Capacity-Utilisation Correlation: {analysis['hypothesis_support']['evidence_strength']['capacity_correlation']['value']:.3f}")
```

### Multi-Configuration Experimental Scenarios

#### Transportation Network Scenarios
```python
# Different transportation infrastructure scenarios
transport_configs = [
    [0.33, 0.33, 0.33],  # Balanced public transport
    [0.60, 0.25, 0.15],  # Highway-dominant system
    [0.45, 0.40, 0.15],  # Rail-highway system
    [0.70, 0.20, 0.10],  # Car-centric infrastructure
    [0.40, 0.35, 0.25],  # Multi-modal balance
]
```

#### Economic Resource Scenarios  
```python
# Economic resource allocation scenarios
economic_configs = [
    [0.33, 0.33, 0.33],  # Balanced economy
    [0.50, 0.30, 0.20],  # Service-manufacturing-agriculture
    [0.65, 0.25, 0.10],  # Technology-dominated economy
    [0.45, 0.45, 0.10],  # Dual-sector economy
    [0.55, 0.25, 0.20],  # Primary sector emphasis
]
```

#### Urban Planning Scenarios
```python
# Urban development capacity scenarios  
urban_configs = [
    [0.33, 0.33, 0.33],  # Balanced development
    [0.60, 0.30, 0.10],  # CBD concentration
    [0.40, 0.40, 0.20],  # Dual-centre model
    [0.50, 0.25, 0.25],  # Single dominant centre
    [0.45, 0.35, 0.20],  # Polycentric development
]
```

#### Parameter Guidelines

**Num Replications:**
- **5-10**: Quick testing and development
- **30-50**: Standard research analysis  
- **100+**: High-precision publication studies

**Capacity Configuration Selection:**
- **Symmetric**: [0.33, 0.33, 0.33] - Baseline control
- **Moderate**: [0.5, 0.3, 0.2] - Realistic asymmetry
- **High**: [0.7, 0.2, 0.1] - Strong dominance effects
- **Extreme**: [0.8, 0.15, 0.05] - Near-monopolistic scenarios

### Variables to Study
1. **Independent Variable**: Capacity configuration (relative_capacity)
2. **Dependent Variables**:
   - Final agent utilisation by resource
   - Convergence times by resource preference
   - System performance metrics
   - Hierarchy consistency measures
   - Specialisation indices

### Capacity Configuration Types
1. **Symmetric**: Equal capacities across resources
2. **Graduated**: Smooth hierarchy (high→medium→low)
3. **Binary**: Two dominant + one minor resource
4. **Monopolistic**: Single dominant resource
5. **Balanced Asymmetric**: Moderate differences

## Metrics and Analysis

### 1. Hierarchical Specialisation Metrics
- **Capacity-Preference Correlation**: Spearman rank correlation between capacity order and agent preference order
- **Resource Popularity Ranking**: Ordered list of resources by agent preference
- **Hierarchy Consistency**: Proportion of resources where capacity rank = popularity rank
- **Specialisation Strength**: Degree of agent concentration on high-capacity resources

### 2. Capacity-Utilisation Correlation Metrics
- **Pearson Correlation**: Linear relationship between capacity values and utilisation rates
- **Spearman Correlation**: Rank-order relationship between capacity and utilisation
- **Utilisation Distribution**: Final agent distribution across resources
- **Asymmetry Effects**: Relationship between capacity asymmetry and utilisation asymmetry

### 3. Convergence Timing Analysis
- **Early vs Late Convergers**: Resource preferences of agents converging in first/second half
- **Convergence Order by Capacity**: Analysis of whether high-capacity resources attract earlier convergers
- **Resource-Specific Convergence Rates**: Speed of convergence to different capacity resources
- **Temporal Hierarchy Formation**: Timeline of hierarchical pattern emergence

### 4. Performance Optimisation Metrics
- **Total System Cost**: Cumulative cost across all resources
- **Load Balance Quality**: Standard deviation of resource utilisation (lower = better)
- **System Efficiency**: Cost per unit of total consumption
- **Convergence Rate**: Proportion of agents achieving convergence
- **Configuration Performance Ranking**: Comparative analysis across configurations

### 5. Ternary Specialisation Analysis
- **Vertex Attraction Strength**: Distance of final agent positions from triangle centre
- **Capacity-Weighted Clustering**: Clustering of agents near high-capacity vertices
- **Trajectory Directness**: Efficiency of agent paths towards preferred vertices
- **Spatial Correlation**: Correlation between vertex capacity and agent density

## Required Visualisations

### Figure 1: Capacity-Utilisation Correlation Analysis
```
(a) Configuration Comparison
    - Bar chart: Capacity-utilisation correlation by configuration
    - Error bars showing variability across replications
    
(b) Asymmetry Effect Analysis  
    - Scatter plot: Capacity asymmetry vs utilisation asymmetry
    - Trend line showing relationship strength
    
(c) Correlation Strength Distribution
    - Histogram of correlation coefficients across all configurations
    - Mean correlation line and significance indicators
    
(d) Pearson vs Spearman Comparison
    - Scatter plot comparing linear vs rank correlations
    - Diagonal reference line (y=x)
```

### Figure 2: Hierarchical Specialisation Analysis
```
(a) Hierarchy Consistency by Configuration
    - Bar chart: Proportion of resources with correct hierarchy placement
    - Reference line at random level (0.33)
    
(b) Resource Popularity Patterns
    - Heatmap: Resource position in popularity ranking
    - Most asymmetric configuration highlighted
    
(c) Capacity vs Utilisation Scatter
    - All resource-utilisation pairs across configurations
    - Colour-coded by configuration type
    - Overall trend line with R² value
    
(d) Asymmetry vs Performance
    - Scatter: Configuration asymmetry vs hierarchy consistency
    - Individual configuration labels
    - Performance trend analysis
```

### Figure 3: System Performance Analysis
```
(a) Total Costs by Configuration
    - Bar chart: Mean system cost with error bars
    - Configuration ranking by performance
    
(b) Load Balance Quality
    - Bar chart: Load balance scores (lower = better)
    - Comparison across configuration types
    
(c) Convergence Rates
    - Bar chart: Proportion of agents converging
    - Configuration-specific convergence success
    
(d) Performance Trade-offs
    - Scatter: Cost vs convergence rate
    - Configuration labels and trend analysis
    - Pareto frontier identification
```

### Figure 4: Ternary Specialisation Comparison
```
(a-d) Representative Configuration Comparisons
    - Ternary plots for 4 key configurations:
      * Symmetric: [0.33, 0.33, 0.33]
      * Moderate: [0.50, 0.30, 0.20] 
      * High: [0.70, 0.20, 0.10]
      * Extreme: [0.80, 0.15, 0.05]
    - Agent positions colour-coded by capacity preference
    - Vertex labels showing capacity values
    - Clear demonstration of capacity-driven clustering
```

### Figure 5: Statistical Analysis Summary
```
(a) Hypothesis Support Summary Table
    - HYPOTHESIS 4 METRICS: Correlation, consistency, specialisation values
    - OVERALL ASSESSMENT: Support level and statistical significance
    - Status indicators (✓/✗) for each criterion
    
(b) Statistical Test Results
    - Bar chart: P-values for key statistical tests
    - Significance threshold line (0.05)
    - Log scale for p-value display
    
(c) Configuration Performance Ranking
    - Multi-metric comparison: Hierarchy, correlation, performance scores
    - Grouped bar chart across configurations
    - Best/worst configuration identification
    
(d) Asymmetry Effects Analysis
    - Scatter: Capacity asymmetry vs hierarchy consistency
    - Trend line with R² value
    - Configuration labels and clustering analysis
```

## Implementation Plan

### File Structure
```
resource_allocation_sim/experiments/capacity_ratio_study.py
```

### Key Classes and Methods
```python
class CapacityRatioStudy(ParameterSweepExperiment):
    def analyse_results()                               # Main analysis pipeline
    def analyse_hierarchical_specialisation()           # Hierarchy pattern analysis
    def analyse_capacity_utilisation_correlation()      # Capacity-utilisation relationships
    def analyse_convergence_timing_patterns()           # Temporal convergence analysis
    def analyse_performance_across_configurations()     # System performance metrics
    def analyse_agent_specialisation_patterns()         # Ternary space specialisation
    
    # Statistical Analysis
    def perform_capacity_statistical_tests()            # Hypothesis testing suite
    def evaluate_capacity_hypothesis_support()          # Evidence evaluation
    
    # Visualisation Methods
    def create_comprehensive_plots()                     # Main plotting pipeline
    def _create_capacity_correlation_plots()             # Figure 1: Correlation analysis
    def _create_hierarchy_analysis_plots()               # Figure 2: Hierarchy patterns
    def _create_performance_analysis_plots()             # Figure 3: Performance analysis
    def _create_ternary_specialisation_plots()           # Figure 4: Ternary analysis
    def _create_statistical_summary_plots()              # Figure 5: Statistical summary
```

### Analysis Pipeline
```python
# 1. Data Collection
configurations = generate_capacity_configurations()
results = run_multi_configuration_experiment()

# 2. Hierarchical Analysis
hierarchy_patterns = analyse_hierarchical_specialisation()
capacity_correlations = analyse_capacity_utilisation_correlation()

# 3. Performance Analysis  
system_performance = analyse_performance_across_configurations()
convergence_timing = analyse_convergence_timing_patterns()

# 4. Statistical Testing
statistical_tests = perform_capacity_statistical_tests()
hypothesis_support = evaluate_capacity_hypothesis_support()

# 5. Visualisation
comprehensive_plots = create_comprehensive_plots()
```

### Core Algorithm Implementation
```python
def analyse_hierarchical_specialisation():
    """Core hierarchy analysis algorithm."""
    for each_configuration:
        # Calculate agent preferences
        final_preferences = extract_agent_final_preferences()
        
        # Determine resource utilisation
        resource_utilisation = count_agents_per_resource()
        
        # Calculate hierarchy metrics
        capacity_order = sort_resources_by_capacity(desc=True)
        utilisation_order = sort_resources_by_utilisation(desc=True)
        
        # Spearman rank correlation
        hierarchy_correlation = spearman_correlation(capacity_order, utilisation_order)
        
        # Consistency measure
        consistency = proportion_matching_positions(capacity_order, utilisation_order)
        
        return hierarchy_metrics
```

## Testable Predictions

### Primary Hypotheses
1. **H4a**: Capacity-utilisation correlation will be significantly positive (r > 0.5, p < 0.05)
2. **H4b**: Hierarchy consistency will be significantly better than random (> 0.33, p < 0.05)
3. **H4c**: Early convergers will prefer high-capacity resources more than late convergers
4. **H4d**: Moderate asymmetry will optimise system performance vs extreme configurations
5. **H4e**: Agent clustering in ternary space will correlate with capacity hierarchy

### Alternative Hypotheses
1. **H4a'**: No significant capacity-utilisation correlation (null hypothesis)
2. **H4b'**: Random hierarchy patterns (consistency ≈ 0.33)
3. **H4c'**: No relationship between convergence timing and capacity preference
4. **H4d'**: Linear relationship between asymmetry and performance
5. **H4e'**: Random agent distribution in ternary space

## Statistical Tests & Metrics

### Core Hypothesis Metrics
1. **Capacity-Utilisation Correlation**: 
   - **Measurement**: Pearson/Spearman correlation between capacity values and final utilisation
   - **Target**: r > 0.5 for strong support
   - **Significance**: p < 0.05 via t-test

2. **Hierarchy Consistency**:
   - **Measurement**: Proportion of resources where capacity rank = utilisation rank
   - **Target**: > 0.6 for strong support (vs random 0.33)
   - **Significance**: p < 0.05 via one-sample t-test

3. **Specialisation Index**:
   - **Measurement**: 1 - (entropy of final distributions / log(3))
   - **Target**: > 0.8 for strong specialisation
   - **Analysis**: Across all configurations

### Statistical Hypothesis Tests

#### Test 1: Capacity-Utilisation Correlation
- **Test**: One-sample t-test on correlation coefficients
- **Null Hypothesis (H₀)**: Mean correlation = 0 (no relationship)
- **Alternative (H₁)**: Mean correlation > 0 (positive relationship)
- **Significance Level**: α = 0.05

#### Test 2: Hierarchy Consistency
- **Test**: One-sample t-test  
- **Null Hypothesis (H₀)**: Consistency = 1/3 (random)
- **Alternative (H₁)**: Consistency > 1/3 (better than random)
- **Significance Level**: α = 0.05

#### Test 3: Asymmetry-Performance Relationship
- **Test**: Pearson correlation test
- **Null Hypothesis (H₀)**: No correlation between asymmetry and performance
- **Alternative (H₁)**: Significant correlation exists
- **Purpose**: Test optimal asymmetry hypothesis

### Comprehensive Evaluation Framework

#### Hypothesis Support Levels
1. **Strong Support** (≥80% criteria met):
   - Capacity-utilisation correlation > 0.5 with p < 0.05
   - Hierarchy consistency > 0.6 with p < 0.05
   - Clear specialisation patterns (index > 0.8)
   - Consistent patterns across configurations

2. **Moderate Support** (≥60% criteria met):
   - Capacity-utilisation correlation > 0.3 with p < 0.1
   - Hierarchy consistency > 0.5 with p < 0.1
   - Moderate specialisation (index > 0.6)
   - Some pattern consistency

3. **Weak Support** (≥40% criteria met):
   - Capacity-utilisation correlation > 0.2
   - Hierarchy consistency > 0.4
   - Limited specialisation (index > 0.4)
   - Mixed pattern consistency

4. **No Support** (<40% criteria met):
   - No significant correlations
   - Random hierarchy patterns
   - Weak specialisation
   - Inconsistent or random patterns

## Success Criteria
1. **Clear Capacity Effects**: Statistically significant positive correlation between capacity and utilisation
2. **Hierarchical Organisation**: Consistent resource hierarchy reflecting capacity order
3. **Predictable Specialisation**: Agent clustering patterns correlate with capacity values
4. **Performance Relationships**: Identifiable optimal asymmetry levels
5. **Convergence Timing**: Early convergers prefer high-capacity resources

## Expected Outcomes
- **Strong Support**: Clear capacity-driven hierarchies with high correlations (r > 0.7)
- **Moderate Support**: Some capacity effects with moderate correlations (r = 0.4-0.7)
- **Weak Support**: Limited capacity influence with weak correlations (r = 0.2-0.4)
- **No Support**: Random patterns independent of capacity configuration

## Implementation Commands

### Direct Module Execution
```bash
# Run with current default parameters (50 replications)
python -m resource_allocation_sim.experiments.capacity_ratio_study
```

### Programmatic Execution (Recommended)

#### Basic Usage
```python
from resource_allocation_sim.experiments.capacity_ratio_study import run_capacity_ratio_study

# Standard research run
study = run_capacity_ratio_study(
    num_replications=50,        # Number of independent runs per configuration
    show_plots=False,           # Save plots without display
    output_dir="results/capacity_study"
)

# Print key results
analysis = study.analysis_results
print(f"Hypothesis Support: {analysis['hypothesis_support']['overall_support']}")
```

#### Advanced Configuration
```python
# Publication-quality analysis with custom configurations
from resource_allocation_sim.experiments.capacity_ratio_study import CapacityRatioStudy

custom_study = CapacityRatioStudy(
    capacity_configurations=[
        [0.33, 0.33, 0.33],  # Baseline
        [0.50, 0.30, 0.20],  # Moderate
        [0.70, 0.20, 0.10],  # High asymmetry
        [0.60, 0.25, 0.15],  # Custom scenario
    ],
    results_dir="results/custom_capacity_study",
    experiment_name="custom_capacity_analysis"
)

# Run experiment
results = custom_study.run_experiment(num_episodes=100)

# Access detailed analysis
custom_study.analyse_results()
analysis = custom_study.analysis_results

print(f"Capacity-Utilisation Correlation: {analysis['hypothesis_support']['evidence_strength']['capacity_correlation']['value']:.3f}")
print(f"Hierarchy Consistency: {analysis['hypothesis_support']['evidence_strength']['hierarchy_consistency']['value']:.3f}")
```

#### Quick Testing
```python
# Fast testing during development
test_study = run_capacity_ratio_study(
    num_replications=5,         # Minimal replications
    show_plots=True,            # Interactive display
    output_dir="test_results"
)
```

#### Batch Processing Multiple Scenarios
```python
# Compare different scenario types
scenarios = [
    {
        "name": "transportation", 
        "configs": [[0.33,0.33,0.33], [0.60,0.25,0.15], [0.45,0.40,0.15]]
    },
    {
        "name": "economic", 
        "configs": [[0.33,0.33,0.33], [0.50,0.30,0.20], [0.65,0.25,0.10]]
    },
    {
        "name": "urban", 
        "configs": [[0.33,0.33,0.33], [0.60,0.30,0.10], [0.40,0.40,0.20]]
    }
]

results = {}
for scenario in scenarios:
    print(f"Running {scenario['name']} scenario...")
    study = CapacityRatioStudy(
        capacity_configurations=scenario['configs'],
        results_dir=f"results/{scenario['name']}_scenario",
        experiment_name=f"capacity_study_{scenario['name']}"
    )
    
    scenario_results = study.run_experiment(num_episodes=30)
    study.analyse_results()
    results[scenario['name']] = study

# Compare results across scenarios
for name, study in results.items():
    support = study.analysis_results['hypothesis_support']['overall_support']
    corr = study.analysis_results['hypothesis_support']['evidence_strength']['capacity_correlation']['value']
    print(f"{name.title()}: {support.upper()} support, Correlation: {corr:.3f}")
```

### Output Structure
After running, results are saved in the specified output directory:
```
results/
└── capacity_ratio_study_YYYYMMDD_HHMMSS/
    ├── experiment_results.json                    # Raw numerical results
    ├── analysis_results.json                      # Statistical analysis
    ├── capacity_ratio_hypothesis_report.txt       # Comprehensive text report
    ├── plots/
    │   ├── capacity_correlation_analysis.png      # Figure 1: Correlation analysis
    │   ├── hierarchy_analysis.png                 # Figure 2: Hierarchy patterns
    │   ├── performance_analysis.png               # Figure 3: Performance analysis
    │   ├── ternary_specialisation_comparison.png  # Figure 4: Ternary comparison
    │   └── statistical_summary.png                # Figure 5: Statistical summary
    └── README.md                                  # Experiment summary
```

## Results Interpretation Guide

### Capacity-Utilisation Correlation Results
- **r = 0.8-1.0**: Very strong capacity-driven specialisation
- **r = 0.6-0.8**: Strong capacity effects with some variation
- **r = 0.4-0.6**: Moderate capacity influence
- **r = 0.2-0.4**: Weak capacity effects
- **r < 0.2**: No meaningful capacity-utilisation relationship

### Hierarchy Consistency Interpretation
- **Consistency = 1.0**: Perfect hierarchy (capacity order = utilisation order)
- **Consistency = 0.8-1.0**: Very strong hierarchical organisation
- **Consistency = 0.6-0.8**: Good hierarchical patterns
- **Consistency = 0.4-0.6**: Moderate hierarchy
- **Consistency ≈ 0.33**: Random (no hierarchy)

### Performance Benchmarks
- **Strong Hierarchy**: Consistency > 0.8, Correlation > 0.7
- **Clear Specialisation**: Specialisation index > 0.8
- **Optimal Configuration**: Best cost-performance ratio
- **Significant Results**: p-values < 0.05 for key tests

### Ternary Plot Interpretation
- **Vertex Clustering**: Agents cluster near high-capacity vertices
- **Gradient Effects**: Higher capacity creates stronger attraction
- **Spatial Correlation**: Agent density correlates with vertex capacity
- **Trajectory Bias**: Agent paths biased towards high-capacity vertices

## Real-World Applications

### Transportation Planning
- **Highway vs Rail vs Public Transport**: Capacity allocation effects on usage patterns
- **Infrastructure Investment**: Optimal capacity distribution strategies
- **Traffic Flow Management**: Predictable routing based on capacity

### Urban Development
- **Commercial District Planning**: Capacity-driven business location patterns
- **Residential Development**: Housing density following infrastructure capacity
- **Service Distribution**: Public service allocation based on capacity models

### Economic Resource Allocation
- **Market Concentration**: Capacity advantages leading to market dominance
- **Supply Chain Optimisation**: Warehouse and distribution capacity effects
- **Investment Allocation**: Capital flow patterns following capacity gradients

### Network Design
- **Server Load Balancing**: Capacity-based traffic distribution
- **Telecommunications**: Bandwidth allocation and usage patterns
- **Social Networks**: Platform capacity effects on user migration 