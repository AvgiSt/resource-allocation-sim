# Agent Architecture and Behaviour Analysis

## Objective
Comprehensive analysis of agent decision-making, learning mechanisms, and partial observability in the resource allocation simulation with corrected mathematical model implementation.

## Agent Architecture Overview

### Core Components
1. **Belief State**: Probability distribution over resources (`self.probabilities`)
2. **Decision Mechanism**: Stochastic action selection based on beliefs
3. **Learning System**: Belief update using partial environmental feedback via mathematical model
4. **Memory**: Action and probability history tracking

## Environment Model

### Resource Capacity System
The simulation uses a **relative capacity system** for scale-invariant behaviour across different agent populations.

#### Relative Capacity Definition
- **Range**: 0.0 to 1.0+ (fractional representation)
- **Interpretation**: Fraction of total agents that can use a resource without causing congestion
- **Conversion**: `actual_capacity = relative_capacity × num_agents`

#### Capacity Examples
- **relative_capacity = 0.3**: 30% of agents can use resource without congestion
  - 4 agents → actual_capacity = 1.2
  - 10 agents → actual_capacity = 3.0  
  - 20 agents → actual_capacity = 6.0
- **relative_capacity = 1.0**: All agents can use resource simultaneously (no congestion possible)
- **relative_capacity > 1.0**: Over-provisioned resource (identical behaviour to 1.0)

#### Scale Invariance Benefits
- **Cross-experiment comparability**: Same relative capacity produces equivalent behaviour regardless of agent count
- **Intuitive interpretation**: 0.5 always means "half the agents can use this resource"
- **Experimental control**: Systematic variation of congestion levels

### Local Utility Function
The environment calculates costs using a conditional utility function based on consumption relative to capacity:

$$L(r,t) = \begin{cases} 
1 & \text{if } x_{r,t} \leq c_r \text{ (no congestion)} \\
e^{1-\frac{x_{r,t}}{c_r}} & \text{if } x_{r,t} > c_r \text{ (congestion penalty)}
\end{cases}$$

Where:
- $x_{r,t}$: consumption of resource $r$ at time $t$
- $c_r$: capacity of resource $r$ (actual capacity = relative_capacity × num_agents)

#### Cost Behaviour
- **No congestion**: Cost = 1.0 (good reinforcement signal)
- **Congestion**: Cost = exponentially decreasing penalty (weak reinforcement)
- **Severe congestion**: Cost approaches 0 (minimal reinforcement, drives agents away)

## Initialisation Methods

### 1. Uniform Initialisation
- **Method**: `create_uniform_agent()`
- **Description**: All resources start with equal probability (1/n)
- **Properties**: Deterministic, symmetric starting point
- **Use case**: Neutral baseline, no prior knowledge

### 2. Dirichlet Initialisation  
- **Method**: `create_dirichlet_agent()`
- **Description**: Random draw from Dirichlet distribution with α=1
- **Properties**: Stochastic, varied initial distributions
- **Use case**: Diverse population with random preferences

### 3. Softmax Initialisation
- **Method**: `create_softmax_agent()`
- **Description**: Random normal logits passed through softmax
- **Properties**: Stochastic, controlled randomness
- **Use case**: Structured random initialisation

## Decision Mechanism

### Stochastic Action Selection
```python
action = np.random.choice(self.num_resources, p=self.probabilities)
```

**Characteristics:**
- **Type**: Multinomial sampling from belief distribution
- **Exploration**: Built-in stochastic exploration proportional to probabilities
- **Exploitation**: Higher probability resources selected more frequently
- **Determinism**: Non-deterministic; same beliefs can yield different actions

## Partial Observability

### What Agents Observe
- **Selected Resource Cost**: Cost of their chosen resource only
- **No Social Information**: Cannot observe other agents' actions or choices
- **No Global State**: No knowledge of resource consumption levels
- **No Alternative Costs**: No information about costs of non-selected resources

### What Agents Cannot Observe
- Other agents' probability distributions
- Number of agents selecting each resource
- Costs of resources they didn't select
- Global system state or total consumption

### Information Flow
1. Agent selects resource based on current beliefs
2. Environment calculates costs for all resources based on total consumption
3. Agent receives only the cost for their selected resource
4. Agent updates beliefs using this partial information

## Belief Update Mechanism

### Mathematical Learning Model
The agent updates probabilities using the exact mathematical formula:

**Step 1: Calculate Scaling Factor**
$$\lambda = w_r \times L(r,t)$$

Where:
- $w_r$: agent's learning weight parameter
- $L(r,t)$: observed cost (local utility) for selected resource

**Step 2: Probability Update**
$$p(a|r) = \lambda \cdot I + (1-\lambda) \cdot p(a|r)$$

Where:
- $I$: unit vector with 1 for selected resource, 0 for others
- $p(a|r)$: current probability distribution

### Implementation Details
   ```python
# Calculate λ = w_r * L(r,t)
lambda_factor = self.weight * observed_cost

# Create unit vector I
unit_vector = np.zeros(self.num_resources)
unit_vector[selected_resource] = 1.0

# Apply update formula
self.probabilities = lambda_factor * unit_vector + (1 - lambda_factor) * self.probabilities
```

### Key Properties
- **Direct Reinforcement**: Higher costs create stronger reinforcement of selected action
- **Probability Conservation**: Total probability always sums to 1
- **Partial Information**: Updates based only on observed cost
- **Learning Rate**: Controlled by weight parameter and observed cost magnitude
- **Unit Vector Update**: Clean mathematical formulation with interpretable parameters

### Learning Dynamics
- **Low Cost (≈0.0009)**: λ ≈ 0.0005 → minimal probability update → agent drifts away
- **Good Cost (1.0)**: λ = weight → strong probability update → agent reinforces choice
- **Congestion Response**: Agents naturally avoid congested resources through weak reinforcement

## Emergent System Behaviour

### Capacity-Driven Load Balancing
The relative capacity system creates emergent load balancing through economic signals:

1. **Congestion Detection**: When consumption > capacity, costs become very small
2. **Weak Reinforcement**: Low costs provide minimal positive feedback
3. **Natural Avoidance**: Agents drift away from poorly reinforced choices
4. **Dynamic Rebalancing**: System continuously adapts to usage patterns

### Multi-Agent Coordination
- **Implicit Coordination**: No direct communication, coordination via cost signals
- **Distributed Learning**: Each agent learns independently with partial information
- **Emergent Efficiency**: System-level optimisation from individual learning
- **Robustness**: No single point of failure, adaptive to changing conditions

## Experimental Framework

### Core Research Questions
1. **Initialisation Impact**: How do different starting distributions affect convergence?
2. **Learning Efficiency**: Which method achieves better resource allocation faster?
3. **Capacity Effects**: How does relative capacity influence system dynamics?
4. **Partial Observability**: How does limited information affect learning quality?
5. **Scale Invariance**: Does relative capacity maintain behaviour across agent counts?

### Experimental Setup
- **Resources**: 3-5 (varies by experiment)
- **Agents**: 4-20 (test scaling behaviour)
- **Relative Capacity**: 0.1-1.0 (test congestion levels)
- **Learning Weight**: 0.1-0.9 (test learning rates)
- **Iterations**: 100-2000 (test convergence)
- **Replications**: 50+ (statistical significance)

### Metrics to Analyse
- **Convergence Speed**: Time to reach stable distributions
- **Load Distribution**: Evenness of final resource allocation
- **Cost Evolution**: System efficiency over time
- **Agent Specialisation**: Individual probability concentration
- **Capacity Utilisation**: Efficiency of resource usage

## Implementation Details

### Code References
- **Agent Class**: `resource_allocation_sim/core/agent.py`
- **Decision Logic**: Lines 48-53 (`select_action`)
- **Learning Algorithm**: Lines 55-82 (`update_probabilities`)
- **Initialisation Methods**: Lines 142-162
- **Environment Model**: `resource_allocation_sim/core/environment.py`
- **Cost Calculation**: Lines 75-85 (`_calculate_costs`)
- **Simulation Integration**: `resource_allocation_sim/core/simulation.py`

### Key Design Decisions
- **Mathematical Rigour**: Exact implementation of formal model equations
- **Partial Observability**: Deliberate limitation to model realistic scenarios
- **Relative Capacity**: Scale-invariant capacity system for comparable experiments
- **Stochastic Selection**: Maintains exploration throughout learning
- **Economic Signals**: Cost-based coordination mechanism

## Mathematical Model Validation
The implementation has been verified to correctly implement:
- **Environment**: L(r,t) = 1 (no congestion) or exp(1-x/c) (congestion penalty)
- **Agent Updates**: λ = w×L(r,t), p = λ·I + (1-λ)·p exactly as specified
- **Relative Capacity**: actual_capacity = relative_capacity × num_agents
- **Partial Observability**: Agents receive only their selected resource's cost

## Results
*[Results to be filled in after running experiments]* 