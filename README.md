# Resource Allocation Simulation Framework

A comprehensive multi-agent simulation framework for studying resource allocation dynamics with probability-based learning agents. Designed for researchers investigating distributed resource allocation, game theory, and multi-agent systems.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Quick Installation](#quick-installation)
  - [Development Installation](#development-installation)
  - [Server/Cluster Installation](#servercluster-installation)
  - [Conda Environment Setup](#conda-environment-setup)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Running Experiments](#running-experiments)
  - [Visualisation Examples](#visualisation-examples)
- [Configuration](#configuration)
- [Performance Considerations](#performance-considerations)
- [Known Issues & Solutions](#known-issues--solutions)
- [Integration with Other Tools](#integration-with-other-tools)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

This framework simulates resource allocation scenarios where autonomous agents learn to select resources based on observed costs. Key features include:

- **Multi-agent learning**: Agents use probability-based learning to adapt resource selection
- **Flexible resource modelling**: Configurable resource capacities and cost functions
- **Comprehensive analysis**: Built-in metrics for entropy, Gini coefficient, convergence analysis
- **Extensive visualisation**: 2D plots, ternary diagrams, network visualisations
- **Experiment framework**: Grid search, parameter sweeps, capacity analysis
- **Scalable design**: Support for large-scale simulations with parallel processing

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM (8 GB+ recommended for larger simulations)
- **Storage**: 1 GB free space
- **OS**: Linux, macOS, or Windows

### Recommended for Research Use
- **Python**: 3.9-3.11 (optimal performance)
- **Memory**: 16 GB+ RAM for large-scale studies
- **CPU**: Multi-core processor (4+ cores for parallel experiments)
- **Storage**: 10 GB+ for extensive result storage

## Installation

### Quick Installation

For researchers who want to get started immediately:

```bash
git clone https://github.com/AvgiSt/resource-allocation-sim.git
cd resource-allocation-sim
pip install -e .
```

Test the installation:
```bash
resource-sim --help
resource-sim run --agents 5 --resources 3 --iterations 50
```

### Development Installation

For contributors and researchers who need full features:

```bash
git clone https://github.com/AvgiSt/resource-allocation-sim.git
cd resource-allocation-sim
chmod +x setup.sh
./setup.sh --dev --test
```

### Server/Cluster Installation

For high-performance computing environments:

```bash
# Load required modules (example for SLURM)
module load python/3.9
module load gcc/9.3.0

# Create environment
python -m venv ~/ras-env
source ~/ras-env/bin/activate

# Install with minimal dependencies
pip install -e .
```

### Conda Environment Setup

Alternative installation using conda:

```bash
conda create -n resource-sim python=3.9
conda activate resource-sim
pip install -e .

# Optional: Install with all features
pip install -e ".[full]"
```

## Dependencies

### Core Dependencies
- **numpy** (≥1.20.0): Numerical computations
- **matplotlib** (≥3.5.0): Basic plotting
- **pandas** (≥1.3.0): Data manipulation
- **scipy** (≥1.7.0): Statistical functions
- **seaborn** (≥0.11.0): Enhanced plotting
- **click** (≥8.0.0): CLI interface
- **pyyaml** (≥6.0): Configuration files

### Optional Dependencies
- **mpltern** (≥1.0.0): Ternary diagrams (`pip install -e ".[ternary]"`)
- **networkx** (≥2.6): Network visualisations (`pip install -e ".[network]"`)
- **plotly** (≥5.0.0): Interactive plots (`pip install -e ".[full]"`)

## Quick Start

### Command Line Interface

Run a basic simulation:
```bash
# Simple simulation with 10 agents and 3 resources
resource-sim run --agents 10 --resources 3 --iterations 100

# With custom capacity
resource-sim run --agents 15 --resources 4 --capacity 1.2 0.8 1.5 1.0

# Save results and generate plots
resource-sim run --agents 20 --resources 3 --save-results --plot
```

Run predefined experiments:
```bash
# Parameter sweep
resource-sim sweep --parameter weight --values 0.1 0.3 0.5 0.7 0.9

# Grid search
resource-sim grid-search --config experiments/grid_config.yaml

# Capacity analysis
resource-sim capacity --min-capacity 0.5 --max-capacity 2.0 --steps 10
```

### Python API

Basic simulation:
```python
from resource_allocation_sim import SimulationRunner, Config

# Create configuration
config = Config()
config.num_agents = 20
config.num_resources = 3
config.num_iterations = 200
config.weight = 0.6

# Run simulation
runner = SimulationRunner(config)
runner.setup()
results = runner.run()

print(f"Final consumption: {results['final_consumption']}")
print(f"Total cost: {results['total_cost']}")
```

### Running Experiments

Comprehensive study:
```python
from resource_allocation_sim.experiments import ComprehensiveStudy

study = ComprehensiveStudy()
study.run_convergence_analysis()
study.run_parameter_sensitivity()
study.run_capacity_analysis()
study.generate_report()
```

Parameter sweep:
```python
from resource_allocation_sim.experiments import ParameterSweepExperiment

# Sweep learning weight parameter
experiment = ParameterSweepExperiment(
    parameter_name='weight',
    parameter_values=[0.1, 0.3, 0.5, 0.7, 0.9],
    num_episodes=20
)
results = experiment.run()
experiment.plot_results()
```

### Visualisation Examples

Resource distribution:
```python
from resource_allocation_sim.visualisation import plot_resource_distribution

# After running simulation
plot_resource_distribution(
    consumption=results['final_consumption'],
    capacity=config.capacity,
    save_path='distribution.png'
)
```

Ternary diagram (requires mpltern):
```python
from resource_allocation_sim.visualisation import plot_ternary_distribution

# For 3-resource scenarios
plot_ternary_distribution(
    consumption_history=results['consumption_history'],
    save_path='ternary.png'
)
```

Network visualisation (requires networkx):
```python
from resource_allocation_sim.visualisation import visualise_state_network

# Visualise agent state transitions
visualise_state_network(
    agent_history=results['agent_history'],
    save_path='network.png'
)
```

## Configuration

### YAML Configuration Files

Create custom configurations:
```yaml
# config.yaml
num_agents: 50
num_resources: 4
num_iterations: 500
weight: 0.65

capacity: [1.2, 0.8, 1.5, 1.0]

# Agent initialisation
agent_initialisation_method: "dirichlet"  # uniform, dirichlet, softmax

# Experiment settings
num_episodes: 10
random_seed: 42

# Output
save_results: true
results_dir: "my_results"
plot_results: true
```

Load configuration:
```python
config = Config('config.yaml')
# Override specific parameters
config.update(num_agents=100, weight=0.7)
```

### Advanced Configuration Options

```python
config = Config()

# Learning parameters
config.weight = 0.6                    # Learning weight (0-1)
config.agent_initialisation_method = "uniform"  # How to initialise agent probabilities

# Environment parameters
config.capacity = [1.0, 1.5, 0.8]     # Resource capacities
config.cost_function = "quadratic"     # linear, quadratic, exponential

# Simulation parameters
config.num_iterations = 1000           # Steps per episode
config.num_episodes = 20               # Number of simulation runs
config.convergence_threshold = 0.001   # Early stopping criterion

# Performance settings
config.parallel_processing = True      # Use multiprocessing
config.num_workers = 4                 # Number of parallel workers
config.batch_size = 100               # Batch size for large simulations
```

## Performance Considerations

### Memory Usage

- **Small simulations** (≤100 agents, ≤1000 iterations): ~10 MB
- **Medium simulations** (≤1000 agents, ≤10000 iterations): ~100 MB  
- **Large simulations** (≥10000 agents, ≥100000 iterations): ~1+ GB

### Computational Complexity

- **Time complexity**: O(agents × iterations × resources)
- **Space complexity**: O(agents × resources + iterations) for history storage

### Optimisation Tips

```python
# For large-scale simulations
config.save_full_history = False       # Reduce memory usage
config.parallel_processing = True      # Enable multiprocessing
config.convergence_threshold = 0.01    # Enable early stopping

# For real-time analysis
config.streaming_mode = True           # Process results incrementally
config.checkpoint_frequency = 1000     # Save intermediate results
```

### Parallel Processing

```python
from resource_allocation_sim.experiments import GridSearchExperiment

# Automatically uses all available cores
experiment = GridSearchExperiment(
    parameter_grid={
        'weight': [0.3, 0.5, 0.7],
        'num_agents': [10, 20, 50]
    },
    parallel=True,
    n_jobs=-1  # Use all cores
)
```

## Known Issues & Solutions

### Installation Issues

**Issue**: `mpltern` installation fails
```bash
# Solution: Install dependencies first
pip install matplotlib numpy
pip install mpltern
```

**Issue**: `No module named resource_allocation_sim`
```bash
# Solution: Install in development mode
pip install -e .
```

### Runtime Issues

**Issue**: Memory error with large simulations
```python
# Solution: Reduce history storage
config.save_full_history = False
config.checkpoint_frequency = 1000
```

**Issue**: Slow convergence
```python
# Solution: Adjust learning parameters
config.weight = 0.8  # Increase learning rate
config.convergence_threshold = 0.01  # Relax convergence criteria
```

**Issue**: Visualisation errors
```bash
# Solution: Install optional dependencies
pip install -e ".[full]"
```

### Performance Issues

**Issue**: Simulation runs slowly
```python
# Solution: Enable optimisations
config.parallel_processing = True
config.use_numba = True  # If available
config.batch_processing = True
```

## Integration with Other Tools

### Jupyter Notebooks

```python
# Enable inline plotting
%matplotlib inline

from resource_allocation_sim import *
import matplotlib.pyplot as plt

# Interactive widgets (requires ipywidgets)
from ipywidgets import interact, FloatSlider

@interact(weight=FloatSlider(min=0.1, max=0.9, step=0.1))
def interactive_simulation(weight=0.5):
    config = Config()
    config.weight = weight
    runner = SimulationRunner(config)
    runner.setup()
    results = runner.run()
    plot_resource_distribution(results['final_consumption'], config.capacity)
    plt.show()
```

### Streamlit Dashboard

```python
import streamlit as st
from resource_allocation_sim import SimulationRunner, Config

st.title("Resource Allocation Simulation")

# Sidebar controls
num_agents = st.sidebar.slider("Number of Agents", 5, 100, 20)
weight = st.sidebar.slider("Learning Weight", 0.1, 0.9, 0.6)

# Run simulation
if st.button("Run Simulation"):
    config = Config()
    config.num_agents = num_agents
    config.weight = weight
    
    runner = SimulationRunner(config)
    runner.setup()
    results = runner.run()
    
    st.write(f"Final consumption: {results['final_consumption']}")
    st.pyplot(plot_resource_distribution(results['final_consumption'], config.capacity))
```

### SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=resource_sim
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load python/3.9
source ~/ras-env/bin/activate

# Run large-scale experiment
resource-sim grid-search --config large_experiment.yaml --parallel --n-jobs 16
```

## Contributing

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/AvgiSt/resource-allocation-sim.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`

### Code Style

- Use **Black** for formatting: `black .`
- Use **isort** for imports: `isort .`
- Use **mypy** for type checking: `mypy resource_allocation_sim/`
- Follow **PEP 8** conventions

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=resource_allocation_sim --cov-report=html

# Run specific test categories
pytest tests/test_core.py -v
pytest tests/test_experiments.py -v
pytest tests/test_visualisation.py -v
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Update documentation
4. Submit pull request

### Performance Testing

```bash
# Benchmark core components
python scripts/benchmark.py

# Profile memory usage
python scripts/memory_profile.py

# Test scalability
python scripts/scalability_test.py
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{resource_allocation_sim,
  author = {Avgi Stavrou},
  title = {Resource Allocation Simulation Framework},
  version = {1.0.0},
  year = {2024},
  url = {https://github.com/AvgiSt/resource-allocation-sim}
}
```

### Related Publications

- *Multi-Agent Resource Allocation with Probability-Based Learning* (2024)
- *Convergence Analysis in Distributed Resource Systems* (2024)
- *Game-Theoretic Approaches to Resource Distribution* (2023)

---

**Version**: 1.0.0  
**Author**: Avgi Stavrou   
**Documentation**: [Read the Docs](https://resource-allocation-sim.readthedocs.io/)  
**Issues**: [GitHub Issues](https://github.com/AvgiSt/resource-allocation-sim/issues)