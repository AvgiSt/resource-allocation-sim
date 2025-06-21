# Documentation Overview

This directory contains technical documentation for the resource allocation simulation system with corrected mathematical model implementation.

## Mathematical Model Foundation

The simulation implements a rigorous multi-agent learning system based on:

**Environment Model:**
- **Local Utility Function**: L(r,t) = 1 (no congestion) or exp(1-x/c) (congestion penalty)
- **Relative Capacity System**: Scale-invariant capacity specification (0.0-1.0+ range)
- **Partial Observability**: Agents observe only their selected resource's cost

**Agent Learning Model:**
- **Scaling Factor**: λ = w_r × L(r,t)
- **Probability Update**: p(a|r) = λ·I + (1-λ)·p(a|r)
- **Unit Vector Reinforcement**: Direct mathematical formulation

## Key Documentation Files

### Agent Analysis
- **File**: `../analysis/agent_study.md`
- **Content**: Comprehensive agent architecture, learning mechanisms, and emergent behaviour analysis
- **Topics**: Relative capacity system, mathematical model validation, experimental framework

### Experimental Reports
- **Directory**: `../experiments/`
- **Content**: Detailed experimental studies and hypothesis testing
- **Examples**: Weight parameter studies, sequential convergence analysis

## System Properties

**Scale Invariance**: Relative capacity ensures consistent behaviour across different agent populations
**Mathematical Rigour**: Exact implementation of formal mathematical model
**Emergent Coordination**: Load balancing through economic signals without direct communication
**Validated Implementation**: Verified to match theoretical specifications exactly 