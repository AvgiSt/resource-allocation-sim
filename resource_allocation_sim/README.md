# Resource Allocation Simulation Framework - Analytical Guide

A comprehensive guide to experiments, evaluations, and visualizations in the resource allocation simulation framework. This guide follows a workflow-focused approach for both end-users and researchers.

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Built-in Experiments](#built-in-experiments)
3. [Evaluation Metrics & Theoretical Background](#evaluation-metrics--theoretical-background)
4. [Visualization Capabilities](#visualization-capabilities)
5. [Research Workflow Examples](#research-workflow-examples)
6. [Creating Custom Experiments](#creating-custom-experiments)
7. [External Tool Integration](#external-tool-integration)
8. [Results Interpretation Guide](#results-interpretation-guide)

## Framework Overview

The framework provides a systematic approach to studying multi-agent resource allocation through:

- **Probability-based learning agents** that adapt resource selection based on observed costs
- **Configurable environments** with varying resource capacities and cost functions
- **Comprehensive evaluation metrics** for system performance analysis
- **Rich visualization tools** for understanding system dynamics
- **Extensible experiment framework** for systematic studies

### Core Components

```
resource_allocation_sim/
├── core/           # Agent, Environment, Simulation engine
├── experiments/    # Built-in experiment types
├── evaluation/     # Metrics and analysis tools
├── visualization/  # Plotting and visualization
├── configs/        # Pre-configured experiment templates
└── utils/          # Configuration and I/O utilities
```

## Built-in Experiments

### 1. Base Experiment (`BaseExperiment`)

**Purpose**: Single simulation run with comprehensive data collection

**Configuration**:
```python
from resource_allocation_sim.experiments import BaseExperiment

experiment = BaseExperiment(
    num_agents=20,
    num_resources=3,
    num_iterations=1000,
    weight=0.6,
    capacity=[1.0, 1.0, 1.0]
)
```

**Outputs**:
- Agent probability evolution
- Resource consumption history
- Cost trajectories
- Convergence metrics

### 2. Grid Search Experiment (`GridSearchExperiment`)

**Purpose**: Systematic exploration of parameter combinations

**Configuration**:
```python
from resource_allocation_sim.experiments import GridSearchExperiment

experiment = GridSearchExperiment(
    parameter_grid={
        'weight': [0.3, 0.5, 0.7],
        'num_agents': [10, 20, 50],
        'capacity': [[1.0, 1.0, 1.0], [0.5, 1.0, 1.5]]
    },
    num_episodes=10
)
```

**Outputs**:
- Parameter combination results matrix
- Performance heatmaps
- Optimal parameter identification

### 3. Parameter Sweep Experiment (`ParameterSweepExperiment`)

**Purpose**: Detailed analysis of single parameter effects

**Configuration**:
```python
from resource_allocation_sim.experiments import ParameterSweepExperiment

experiment = ParameterSweepExperiment(
    parameter_name='weight',
    parameter_values=np.linspace(0.1, 0.9, 17),
    num_episodes=20
)
```

**Outputs**:
- Parameter sensitivity curves
- Statistical significance tests
- Optimal parameter ranges

### 4. Capacity Analysis Experiment (`CapacityAnalysisExperiment`)

**Purpose**: Study resource capacity effects on system behavior

**Configuration**:
```python
from resource_allocation_sim.experiments import CapacityAnalysisExperiment

experiment = CapacityAnalysisExperiment(
    capacity_scenarios={
        'symmetric': [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'asymmetric': [[0.5, 1.0, 1.5], [0.2, 0.8, 1.0]],
        'extreme': [[0.1, 0.1, 2.8], [2.0, 0.5, 0.5]]
    }
)
```

**Outputs**:
- Capacity utilization analysis
- Load balancing effectiveness
- System stability under different capacities

### 5. Comprehensive Study (`ComprehensiveStudy`)

**Purpose**: Complete system characterization using configuration files

**Using Configuration Files**:

#### Quick Study (`configs/quick_study.yaml`)
```yaml
# Fast testing configuration
study_name: "quick_resource_allocation_test"
base_simulation:
  num_iterations: 100
  num_episodes: 5
  num_agents: 10

analyses:
  weight_analysis: true      # Test weight sensitivity
  capacity_analysis: true    # Test capacity effects
  scaling_analysis: false    # Skip for speed
```

#### Comprehensive Study (`configs/comprehensive_study.yaml`)
```yaml
# Complete research configuration
study_name: "comprehensive_resource_allocation_study"
base_simulation:
  num_iterations: 1000
  num_episodes: 50
  num_agents: 20

analyses:
  weight_analysis: true           # Learning rate effects
  capacity_analysis: true         # Resource capacity effects
  scaling_analysis: true          # System size effects
  initial_condition_analysis: true # Starting condition effects
  convergence_analysis: true      # Convergence behavior
```

**Running Configuration-Based Studies**:
```python
from resource_allocation_sim.experiments import ComprehensiveStudy

# Load and run predefined study
study = ComprehensiveStudy.from_config('configs/comprehensive_study.yaml')
results = study.run()

# Generate comprehensive report
study.generate_report(save_path='study_results/')
```

## Evaluation Metrics & Theoretical Background

### 1. Shannon Entropy

**Theory**: Measures the uncertainty/randomness in resource consumption distribution.

**Formula**: `H(X) = -Σ p(x) log₂ p(x)`

**Implementation**:
```python
from resource_allocation_sim.evaluation import calculate_entropy

entropy = calculate_entropy(consumption)
```

**Interpretation**:
- **Low entropy (0-1)**: Concentrated resource usage, high specialization
- **Medium entropy (1-2)**: Balanced resource distribution
- **High entropy (>2)**: Uniform resource usage, low specialization

**Research Applications**:
- Measuring system organization
- Detecting phase transitions
- Comparing learning efficiency

### 2. Gini Coefficient

**Theory**: Measures inequality in resource consumption distribution.

**Formula**: `G = (n+1-2Σ(n+1-i)yᵢ)/(n·Σyᵢ)` where y is sorted consumption

**Implementation**:
```python
from resource_allocation_sim.evaluation import calculate_gini_coefficient

gini = calculate_gini_coefficient(consumption)
```

**Interpretation**:
- **Gini = 0**: Perfect equality (all resources used equally)
- **Gini = 1**: Perfect inequality (one resource dominates)
- **Gini ∈ [0.3, 0.7]**: Typical range for balanced systems

**Research Applications**:
- Fairness analysis
- Load balancing assessment
- Social choice theory applications

### 3. Resource Utilization Efficiency

**Theory**: Measures how effectively resources are used relative to capacity.

**Implementation**:
```python
from resource_allocation_sim.evaluation import calculate_resource_utilization

# Standard deviation (no capacity)
efficiency = calculate_resource_utilization(consumption)

# Utilization rates (with capacity)
utilization = calculate_resource_utilization(consumption, capacity)
```

**Interpretation**:
- **Low std deviation**: Balanced resource usage
- **High utilization rates**: Efficient capacity usage
- **Utilization > 1**: Over-capacity usage (potential bottlenecks)

### 4. Convergence Speed

**Theory**: Measures how quickly the system reaches stable behavior.

**Implementation**:
```python
from resource_allocation_sim.evaluation import calculate_convergence_speed

convergence_time = calculate_convergence_speed(
    cost_history, 
    window_size=10, 
    threshold=0.01
)
```

**Interpretation**:
- **Fast convergence (<100 iterations)**: Efficient learning
- **Slow convergence (>500 iterations)**: Complex dynamics
- **No convergence**: Chaotic or oscillatory behavior

### 5. Total System Cost

**Theory**: Aggregate cost considering capacity constraints and penalties.

**Formula**: `Cost = Σ cᵢ · f(cᵢ/capᵢ)` where f is the cost function

**Implementation**:
```python
from resource_allocation_sim.evaluation import calculate_total_cost

total_cost = calculate_total_cost(consumption, capacity)
```

**Interpretation**:
- **Linear growth**: Efficient resource allocation
- **Exponential growth**: Capacity violations and penalties
- **Optimal cost**: Balance between utilization and penalties

## Visualization Capabilities

### 1. Resource Distribution Plots (`plots.py`)

**Purpose**: Visualize consumption vs capacity for each resource

```python
from resource_allocation_sim.visualization import plot_resource_distribution

fig = plot_resource_distribution(
    consumption=results['final_consumption'],
    capacity=config.capacity,
    save_path='distribution.png'
)
```

**Output**: Bar chart with consumption bars and capacity lines

**Interpretation**:
- Bars above capacity lines indicate over-utilization
- Uniform bar heights suggest balanced allocation
- Large variations indicate specialization

### 2. Convergence Comparison Plots

**Purpose**: Compare convergence across different configurations

```python
from resource_allocation_sim.visualization import plot_convergence_comparison

fig = plot_convergence_comparison(
    results_dict={'Config A': results_a, 'Config B': results_b},
    metric='entropy',
    save_path='convergence.png'
)
```

**Output**: Multi-panel plot with time series, box plots, histograms, and statistics

**Interpretation**:
- Decreasing entropy indicates increasing organization
- Box plot spread shows configuration robustness
- Statistics table enables quantitative comparison

### 3. Parameter Sensitivity Plots

**Purpose**: Visualize metric response to parameter changes

```python
from resource_allocation_sim.visualization import plot_parameter_sensitivity

fig = plot_parameter_sensitivity(
    parameter_results=sweep_results,
    parameter_name='weight',
    metric_name='total_cost',
    save_path='sensitivity.png'
)
```

**Output**: Error bar plot and box plot distribution

**Interpretation**:
- Smooth curves indicate stable parameter effects
- Sharp transitions suggest critical parameter values
- Error bars show result variability

### 4. Ternary Diagrams (`ternary.py`)

**Purpose**: Visualize 3-resource consumption evolution

```python
from resource_allocation_sim.visualization import plot_ternary_distribution

fig = plot_ternary_distribution(
    consumption_history=results['consumption_history'],
    save_path='ternary.png'
)
```

**Output**: Triangular plot showing consumption trajectories

**Interpretation**:
- Corners represent single-resource specialization
- Center represents balanced allocation
- Trajectories show learning dynamics

### 5. Network Visualizations (`network.py`)

**Purpose**: Visualize agent state transitions and interactions

```python
from resource_allocation_sim.visualization import visualize_state_network

fig = visualize_state_network(
    agent_history=results['agent_history'],
    save_path='network.png'
)
```

**Output**: Network graph of agent states and transitions

**Interpretation**:
- Node size indicates state frequency
- Edge thickness shows transition probability
- Clusters reveal behavioral patterns

## Research Workflow Examples

### Workflow 1: Basic System Characterization

```python
# Step 1: Run comprehensive study
from resource_allocation_sim.experiments import ComprehensiveStudy

study = ComprehensiveStudy.from_config('configs/comprehensive_study.yaml')
results = study.run()

# Step 2: Analyze key metrics
from resource_allocation_sim.evaluation import calculate_system_metrics

for config_name, config_results in results.items():
    metrics = calculate_system_metrics(config_results, num_agents=20)
    print(f"{config_name}: Entropy={metrics['entropy']:.3f}, "
          f"Gini={metrics['gini_coefficient']:.3f}")

# Step 3: Generate visualizations
study.generate_plots(save_dir='analysis_plots/')

# Step 4: Create summary report
study.generate_report(save_path='system_characterization_report.html')
```

### Workflow 2: Parameter Optimization

```python
# Step 1: Parameter sweep
from resource_allocation_sim.experiments import ParameterSweepExperiment

experiment = ParameterSweepExperiment(
    parameter_name='weight',
    parameter_values=np.linspace(0.1, 0.9, 17),
    num_episodes=50
)
results = experiment.run()

# Step 2: Find optimal parameters
optimal_weight = experiment.find_optimal_parameter(
    metric='total_cost',
    optimization='minimize'
)

# Step 3: Validate optimal parameters
validation_experiment = BaseExperiment(weight=optimal_weight)
validation_results = validation_experiment.run()

# Step 4: Statistical analysis
experiment.statistical_analysis(save_path='optimization_analysis.csv')
```

### Workflow 3: Capacity Design Study

```python
# Step 1: Capacity analysis
from resource_allocation_sim.experiments import CapacityAnalysisExperiment

experiment = CapacityAnalysisExperiment(
    capacity_scenarios={
        'equal': [[1.0, 1.0, 1.0]],
        'graduated': [[0.5, 1.0, 1.5], [0.3, 1.0, 1.7]],
        'extreme': [[0.1, 0.1, 2.8], [2.0, 0.5, 0.5]]
    }
)
results = experiment.run()

# Step 2: Utilization analysis
utilization_analysis = experiment.analyze_utilization()

# Step 3: Cost-benefit analysis
cost_benefit = experiment.cost_benefit_analysis()

# Step 4: Design recommendations
recommendations = experiment.generate_design_recommendations()
```

## Creating Custom Experiments

### 1. Extending BaseExperiment

```python
from resource_allocation_sim.experiments import BaseExperiment
from resource_allocation_sim.evaluation import calculate_entropy
import numpy as np

class CustomConvergenceExperiment(BaseExperiment):
    """Custom experiment focusing on convergence dynamics."""
    
    def __init__(self, convergence_criteria=None, **kwargs):
        super().__init__(**kwargs)
        self.convergence_criteria = convergence_criteria or {
            'entropy_threshold': 0.01,
            'patience': 50
        }
    
    def run_single_episode(self):
        """Run single episode with convergence monitoring."""
        results = super().run_single_episode()
        
        # Add convergence analysis
        entropy_history = []
        for consumption in results['consumption_history']:
            entropy_history.append(calculate_entropy(consumption))
        
        # Detect convergence point
        convergence_point = self.detect_convergence(entropy_history)
        results['convergence_point'] = convergence_point
        results['entropy_history'] = entropy_history
        
        return results
    
    def detect_convergence(self, entropy_history):
        """Detect when system converges."""
        threshold = self.convergence_criteria['entropy_threshold']
        patience = self.convergence_criteria['patience']
        
        for i in range(patience, len(entropy_history)):
            window = entropy_history[i-patience:i]
            if np.std(window) < threshold:
                return i - patience
        
        return len(entropy_history)
    
    def analyze_results(self, results):
        """Custom analysis for convergence experiments."""
        convergence_times = [r['convergence_point'] for r in results]
        
        analysis = {
            'mean_convergence_time': np.mean(convergence_times),
            'std_convergence_time': np.std(convergence_times),
            'convergence_rate': len([t for t in convergence_times 
                                   if t < len(results[0]['entropy_history'])]) / len(results)
        }
        
        return analysis
```

### 2. Creating Custom Metrics

```python
from resource_allocation_sim.evaluation.metrics import calculate_entropy
import numpy as np

def calculate_specialization_index(consumption, capacity=None):
    """
    Custom metric: Specialization Index
    Measures how specialized the resource allocation is.
    """
    consumption = np.array(consumption)
    
    if capacity is not None:
        # Normalize by capacity
        capacity = np.array(capacity)
        normalized_consumption = consumption / capacity
    else:
        normalized_consumption = consumption
    
    # Calculate specialization as inverse of entropy
    entropy = calculate_entropy(normalized_consumption)
    max_entropy = np.log2(len(consumption))
    
    specialization_index = 1 - (entropy / max_entropy)
    return specialization_index

def calculate_stability_metric(consumption_history, window_size=10):
    """
    Custom metric: System Stability
    Measures how stable the resource allocation is over time.
    """
    if len(consumption_history) < window_size:
        return 0.0
    
    # Calculate variance in consumption over time
    consumption_array = np.array(consumption_history)
    
    # Rolling variance for each resource
    stabilities = []
    for i in range(consumption_array.shape[1]):  # For each resource
        resource_consumption = consumption_array[:, i]
        rolling_var = []
        
        for j in range(window_size, len(resource_consumption)):
            window_var = np.var(resource_consumption[j-window_size:j])
            rolling_var.append(window_var)
        
        if rolling_var:
            stabilities.append(1 / (1 + np.mean(rolling_var)))
    
    return np.mean(stabilities) if stabilities else 0.0
```

### 3. Custom Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
from resource_allocation_sim.visualization.plots import plot_resource_distribution

def plot_convergence_heatmap(results_dict, save_path=None):
    """
    Custom visualization: Convergence Heatmap
    Shows convergence times across different parameter combinations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    convergence_data = []
    for config_name, results in results_dict.items():
        convergence_times = [r.get('convergence_point', float('inf')) 
                           for r in results]
        convergence_data.append({
            'config': config_name,
            'mean_convergence': np.mean(convergence_times),
            'std_convergence': np.std(convergence_times)
        })
    
    # Create heatmap
    df = pd.DataFrame(convergence_data)
    pivot_data = df.pivot_table(values='mean_convergence', 
                               index='config', 
                               aggfunc='mean')
    
    sns.heatmap(pivot_data, annot=True, cmap='viridis_r', ax=ax)
    ax.set_title('Convergence Time Heatmap')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Mean Convergence Time')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_phase_diagram(parameter_results, x_param, y_param, metric, save_path=None):
    """
    Custom visualization: Phase Diagram
    Shows system behavior phases in parameter space.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract parameter values and metrics
    x_values = []
    y_values = []
    metric_values = []
    
    for params, results in parameter_results.items():
        x_val = params[x_param]
        y_val = params[y_param]
        metric_val = np.mean([r[metric] for r in results])
        
        x_values.append(x_val)
        y_values.append(y_val)
        metric_values.append(metric_val)
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(x_values, y_values, c=metric_values, 
                        cmap='viridis', s=100, alpha=0.7)
    
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f'Phase Diagram: {metric}')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(metric)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
```

## External Tool Integration

### 1. Jupyter Notebook Integration

```python
# Enable interactive plotting
%matplotlib widget
import ipywidgets as widgets
from IPython.display import display

# Interactive parameter exploration
def interactive_simulation():
    weight_slider = widgets.FloatSlider(
        value=0.6, min=0.1, max=0.9, step=0.1,
        description='Weight:'
    )
    
    agents_slider = widgets.IntSlider(
        value=20, min=5, max=100, step=5,
        description='Agents:'
    )
    
    def update_simulation(weight, agents):
        from resource_allocation_sim import SimulationRunner, Config
        
        config = Config()
        config.weight = weight
        config.num_agents = agents
        config.num_iterations = 200
        
        runner = SimulationRunner(config)
        runner.setup()
        results = runner.run()
        
        # Plot results
        plot_resource_distribution(
            results['final_consumption'], 
            config.capacity
        )
        plt.show()
    
    widgets.interact(update_simulation, 
                    weight=weight_slider, 
                    agents=agents_slider)

interactive_simulation()
```

### 2. Pandas Integration

```python
import pandas as pd
from resource_allocation_sim.experiments import ParameterSweepExperiment

# Run parameter sweep
experiment = ParameterSweepExperiment(
    parameter_name='weight',
    parameter_values=np.linspace(0.1, 0.9, 9),
    num_episodes=20
)
results = experiment.run()

# Convert to DataFrame for analysis
data_rows = []
for weight, episodes in results.items():
    for episode in episodes:
        row = {
            'weight': weight,
            'entropy': calculate_entropy(episode['final_consumption']),
            'gini': calculate_gini_coefficient(episode['final_consumption']),
            'total_cost': episode['total_cost'],
            'convergence_time': episode.get('convergence_point', 
                                          len(episode['cost_history']))
        }
        data_rows.append(row)

df = pd.DataFrame(data_rows)

# Statistical analysis with pandas
summary_stats = df.groupby('weight').agg({
    'entropy': ['mean', 'std'],
    'gini': ['mean', 'std'],
    'total_cost': ['mean', 'std'],
    'convergence_time': ['mean', 'std']
}).round(3)

print(summary_stats)

# Export for external analysis
df.to_csv('parameter_sweep_results.csv', index=False)
```

### 3. Scikit-learn Integration

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def cluster_system_behaviors(results_dict, n_clusters=3):
    """
    Use scikit-learn to cluster different system behaviors.
    """
    # Prepare feature matrix
    features = []
    labels = []
    
    for config_name, results in results_dict.items():
        for result in results:
            feature_vector = [
                calculate_entropy(result['final_consumption']),
                calculate_gini_coefficient(result['final_consumption']),
                result['total_cost'],
                len(result['cost_history']),  # convergence time proxy
                np.std(result['final_consumption'])  # consumption variance
            ]
            features.append(feature_vector)
            labels.append(config_name)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('System Behavior Clusters')
    plt.colorbar(scatter)
    plt.show()
    
    return {
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'pca_components': pca.components_,
        'feature_names': ['entropy', 'gini', 'total_cost', 'convergence_time', 'consumption_std']
    }
```

### 4. NetworkX Integration

```python
import networkx as nx
from resource_allocation_sim.visualization.network import create_state_transition_graph

def analyze_system_topology(agent_results):
    """
    Use NetworkX to analyze the topology of agent state transitions.
    """
    # Create transition graph
    G = create_state_transition_graph(agent_results)
    
    # Calculate network metrics
    metrics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G),
        'transitivity': nx.transitivity(G)
    }
    
    # Find important nodes
    centrality = nx.betweenness_centrality(G)
    important_states = sorted(centrality.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
    
    # Detect communities
    communities = nx.community.greedy_modularity_communities(G)
    
    # Calculate shortest paths
    try:
        avg_path_length = nx.average_shortest_path_length(G)
        metrics['average_path_length'] = avg_path_length
    except nx.NetworkXError:
        metrics['average_path_length'] = float('inf')
    
    return {
        'network_metrics': metrics,
        'important_states': important_states,
        'communities': list(communities),
        'graph': G
    }
```

### 5. Plotly Integration for Interactive Visualizations

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_interactive_dashboard(results_dict):
    """
    Create interactive dashboard using Plotly.
    """
    # Prepare data
    data_for_plotting = []
    for config_name, results in results_dict.items():
        for i, result in enumerate(results):
            data_for_plotting.append({
                'config': config_name,
                'episode': i,
                'entropy': calculate_entropy(result['final_consumption']),
                'gini': calculate_gini_coefficient(result['final_consumption']),
                'total_cost': result['total_cost']
            })
    
    df = pd.DataFrame(data_for_plotting)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Entropy Distribution', 'Gini Coefficient', 
                       'Total Cost', 'Entropy vs Gini'),
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # Add box plots
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        
        fig.add_trace(
            go.Box(y=config_data['entropy'], name=config, 
                  legendgroup=config, showlegend=True),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=config_data['gini'], name=config, 
                  legendgroup=config, showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Box(y=config_data['total_cost'], name=config, 
                  legendgroup=config, showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=config_data['entropy'], y=config_data['gini'],
                      mode='markers', name=config, 
                      legendgroup=config, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Interactive Results Dashboard")
    fig.show()
    
    return fig
```

## Results Interpretation Guide

### 1. Entropy Analysis

**Low Entropy (0-0.5)**:
- **Interpretation**: High specialization, agents concentrate on few resources
- **Implications**: Efficient but potentially fragile system
- **Research Questions**: What drives specialization? How robust is the system?

**Medium Entropy (0.5-1.5)**:
- **Interpretation**: Balanced resource usage with some specialization
- **Implications**: Good trade-off between efficiency and robustness
- **Research Questions**: What maintains this balance? How stable is it?

**High Entropy (>1.5)**:
- **Interpretation**: Uniform resource usage, little specialization
- **Implications**: Robust but potentially inefficient
- **Research Questions**: Why no specialization? Is this optimal?

### 2. Gini Coefficient Analysis

**Low Gini (0-0.3)**:
- **Interpretation**: Equal resource distribution
- **Implications**: Fair allocation, potential inefficiency
- **Research Context**: Social choice, fairness studies

**Medium Gini (0.3-0.7)**:
- **Interpretation**: Moderate inequality
- **Implications**: Some specialization while maintaining fairness
- **Research Context**: Optimal inequality studies

**High Gini (0.7-1.0)**:
- **Interpretation**: High inequality, strong specialization
- **Implications**: Efficient but potentially unfair
- **Research Context**: Efficiency vs. fairness trade-offs

### 3. Convergence Pattern Analysis

**Fast Convergence (<100 iterations)**:
- **Possible Causes**: Strong learning signal, simple environment
- **Implications**: Predictable system behavior
- **Research Focus**: Learning efficiency, parameter optimization

**Slow Convergence (100-500 iterations)**:
- **Possible Causes**: Complex dynamics, weak learning signal
- **Implications**: Rich system behavior, potential for multiple equilibria
- **Research Focus**: System complexity, learning dynamics

**No Convergence (>500 iterations)**:
- **Possible Causes**: Chaotic dynamics, conflicting objectives
- **Implications**: Complex system behavior, potential instability
- **Research Focus**: Chaos theory, system stability

### 4. Cost Analysis

**Linear Cost Growth**:
- **Interpretation**: Efficient resource allocation within capacity
- **Implications**: System operating optimally
- **Research Focus**: Capacity planning, efficiency optimization

**Exponential Cost Growth**:
- **Interpretation**: Capacity violations, penalty accumulation
- **Implications**: System stress, need for capacity adjustment
- **Research Focus**: Capacity design, penalty mechanisms

**Oscillating Costs**:
- **Interpretation**: Dynamic system behavior, potential cycles
- **Implications**: Complex system dynamics
- **Research Focus**: Dynamical systems, stability analysis

### 5. Multi-Metric Interpretation

**High Entropy + Low Gini**:
- **System State**: Uniform, fair allocation
- **Research Context**: Egalitarian systems, social welfare

**Low Entropy + High Gini**:
- **System State**: Specialized, unequal allocation
- **Research Context**: Efficiency-focused systems, market dynamics

**Medium Entropy + Medium Gini**:
- **System State**: Balanced system
- **Research Context**: Optimal system design, trade-off analysis

### 6. Statistical Significance

When comparing results across configurations:

```python
from scipy import stats

def compare_configurations(results_a, results_b, metric='entropy'):
    """Compare two configurations statistically."""
    values_a = [calculate_entropy(r['final_consumption']) for r in results_a]
    values_b = [calculate_entropy(r['final_consumption']) for r in results_b]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(values_a)-1)*np.var(values_a) + 
                         (len(values_b)-1)*np.var(values_b)) / 
                        (len(values_a)+len(values_b)-2))
    cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'effect_size': 'small' if abs(cohens_d) < 0.5 else 
                      'medium' if abs(cohens_d) < 0.8 else 'large'
    }
```

**Interpretation Guidelines**:
- **p < 0.05**: Statistically significant difference
- **Cohen's d > 0.8**: Large practical effect
- **Cohen's d 0.5-0.8**: Medium practical effect
- **Cohen's d < 0.5**: Small practical effect

---

This comprehensive guide provides the theoretical foundation and practical tools needed to conduct rigorous research using the resource allocation simulation framework. For additional support, consult the main documentation or contact me.
