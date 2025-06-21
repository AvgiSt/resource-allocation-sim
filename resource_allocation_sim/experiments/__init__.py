"""Experiment framework for running systematic studies."""

from .base_experiment import BaseExperiment
from .grid_search import GridSearchExperiment
from .parameter_sweep import ParameterSweepExperiment
from .capacity_analysis import CapacityAnalysisExperiment

# Import hypothesis studies conditionally to avoid module execution conflicts
try:
    from .weight_parameter_study import WeightParameterStudy, run_weight_parameter_study
    from .sequential_convergence_study import SequentialConvergenceStudy, run_sequential_convergence_study
except ImportError:
    # Fallback if modules are being executed directly
    WeightParameterStudy = None
    run_weight_parameter_study = None
    SequentialConvergenceStudy = None
    run_sequential_convergence_study = None

__all__ = [
    "BaseExperiment",
    "GridSearchExperiment", 
    "ParameterSweepExperiment",
    "CapacityAnalysisExperiment",
    "WeightParameterStudy",
    "SequentialConvergenceStudy",
    "run_weight_parameter_study",
    "run_sequential_convergence_study",
    # Unified interface functions
    "run_hypothesis_1_study",
    "run_hypothesis_2_study"
]


def run_hypothesis_1_study(
    num_replications: int = 50,
    output_dir: str = "results/hypothesis1_weight_study",
    show_plots: bool = False,
    weight_values: list = None
):
    """
    Run Hypothesis 1: Weight Parameter Study
    
    Tests hypothesis that weight parameter significantly affects convergence speed,
    system performance, and learning stability.
    
    Args:
        num_replications: Number of replications per weight value
        output_dir: Output directory for results  
        show_plots: Whether to display plots interactively
        weight_values: List of weight values to test (default: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
        
    Returns:
        Completed study instance with full analysis
        
    Example:
        >>> study = run_hypothesis_1_study(num_replications=30)
        >>> print(f"Optimal weight: {study.analysis_results['recommendations']['optimal_for_cost']}")
    """
    # Import here to avoid circular import issues
    if run_weight_parameter_study is None:
        from .weight_parameter_study import run_weight_parameter_study as _run_weight_study
    else:
        _run_weight_study = run_weight_parameter_study
    
    print("=" * 80)
    print("HYPOTHESIS 1: WEIGHT PARAMETER EFFECT STUDY")
    print("=" * 80)
    print("Testing: Weight parameter significantly affects system performance and learning")
    print()
    
    study = _run_weight_study(
        num_replications=num_replications,
        output_dir=output_dir,
        show_plots=show_plots
    )
    
    # Custom weight values if provided
    if weight_values is not None:
        print(f"Note: Custom weight values were provided but study already completed with default values.")
        print(f"To use custom values, create WeightParameterStudy directly.")
    
    return study


def run_hypothesis_2_study(
    num_replications: int = 30,
    num_iterations: int = 1000,
    output_dir: str = "results/hypothesis2_sequential_study",
    show_plots: bool = False,
    convergence_threshold_entropy: float = None,
    convergence_threshold_max_prob: float = None
):
    """
    Run Hypothesis 2: Sequential Convergence Study
    
    Tests hypothesis that agents with uniform initial distribution across available choices,
    sequentially converge to a degenerate distribution, with each agent becoming certain
    of a single choice one after the other.
    
    Args:
        num_replications: Number of independent simulation runs
        num_iterations: Number of iterations per simulation
        output_dir: Output directory for results
        show_plots: Whether to display plots interactively
        convergence_threshold_entropy: Entropy threshold for convergence detection
        convergence_threshold_max_prob: Max probability threshold for degeneracy
        
    Returns:
        Completed study instance with full analysis
        
    Example:
        >>> study = run_hypothesis_2_study(num_replications=50)
        >>> support = study.analysis['hypothesis_support']['overall_support']
        >>> print(f"Hypothesis support: {support}")
    """
    print("=" * 80)
    print("HYPOTHESIS 2: SEQUENTIAL CONVERGENCE STUDY")  
    print("=" * 80)
    print("Testing: Agents sequentially converge to degenerate distributions")
    print()
    
    study = run_sequential_convergence_study(
        num_replications=num_replications,
        num_iterations=num_iterations,
        output_dir=output_dir,
        show_plots=show_plots,
        convergence_threshold_entropy=convergence_threshold_entropy,
        convergence_threshold_max_prob=convergence_threshold_max_prob
    )
    
    return study


# Convenient aliases for backward compatibility
def run_hypothesis1_study(*args, **kwargs):
    """Alias for run_hypothesis_1_study."""
    return run_hypothesis_1_study(*args, **kwargs)


def run_hypothesis2_study(*args, **kwargs):
    """Alias for run_hypothesis_2_study."""
    return run_hypothesis_2_study(*args, **kwargs) 