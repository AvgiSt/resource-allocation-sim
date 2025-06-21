"""Base experiment class for systematic studies."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime

from ..core.simulation import SimulationRunner
from ..utils.config import Config
from ..utils.io import save_results, ensure_directory
from ..evaluation.metrics import calculate_system_metrics


class BaseExperiment(ABC):
    """
    Base class for simulation experiments.
    
    Provides common functionality for running systematic studies,
    saving results, and generating reports.
    """
    
    def __init__(
        self,
        base_config: Optional[Config] = None,
        results_dir: str = "results/experiments",
        experiment_name: Optional[str] = None
    ):
        """
        Initialise base experiment.
        
        Args:
            base_config: Base simulation configuration
            results_dir: Directory to save results
            experiment_name: Name of the experiment
        """
        self.base_config = base_config or Config()
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name or self.__class__.__name__
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"{self.experiment_name}_{timestamp}"
        ensure_directory(self.experiment_dir)
        
        # Storage for results
        self.results: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate list of configurations to test.
        
        Returns:
            List of configuration dictionaries
        """
        pass
    
    @abstractmethod
    def analyse_results(self) -> Dict[str, Any]:
        """
        Analyse experimental results.
        
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def run_experiment(
        self,
        num_episodes: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Args:
            num_episodes: Number of episodes per configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing all results and analysis
        """
        configurations = self.generate_configurations()
        
        # Save experiment metadata
        self.metadata = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'num_configurations': len(configurations),
            'num_episodes': num_episodes,
            'base_config': self.base_config.to_dict()
        }
        
        # Run experiments
        for i, config_params in enumerate(configurations):
            if progress_callback:
                progress_callback(i, len(configurations), config_params)
            
            # Create configuration
            config = Config(**self.base_config.to_dict())
            config.update(**config_params)
            
            # Run episodes
            episode_results = []
            for episode in range(num_episodes):
                # Create custom agent factory if needed
                custom_agent_factory = None
                if hasattr(self, 'create_custom_agent_factory'):
                    custom_agent_factory = self.create_custom_agent_factory(config_params, config)
                
                runner = SimulationRunner(config, custom_agent_factory)
                runner.setup()
                result = runner.run()
                result['config_params'] = config_params
                result['episode'] = episode
                episode_results.append(result)
            
            # Store results
            self.results.append({
                'config_params': config_params,
                'episode_results': episode_results,
                'summary_metrics': self._summarize_episodes(episode_results)
            })
        
        # Complete metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        
        # Save results
        self._save_results()
        
        # Analyse results
        analysis = self.analyse_results()
        
        return {
            'metadata': self.metadata,
            'results': self.results,
            'analysis': analysis
        }
    
    def _summarize_episodes(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize metrics across episodes."""
        metrics_list = []
        for result in episode_results:
            metrics = calculate_system_metrics(result, self.base_config.num_agents)
            metrics_list.append(metrics)
        
        # Calculate summary statistics
        summary = {}
        if metrics_list:
            metric_names = metrics_list[0].keys()
            for metric in metric_names:
                values = [m[metric] for m in metrics_list]
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        return summary
    
    def _save_results(self) -> None:
        """Save experiment results."""
        # Save full results
        save_results(
            {
                'metadata': self.metadata,
                'results': self.results
            },
            'full_results',
            'pickle',
            self.experiment_dir
        )
        
        # Save metadata separately
        with open(self.experiment_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save summary CSV
        self._save_summary_csv()
    
    def _save_summary_csv(self) -> None:
        """Save summary results as CSV."""
        import pandas as pd
        
        summary_data = []
        for result in self.results:
            row = {}
            row.update(result['config_params'])
            row.update(result['summary_metrics'])
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(self.experiment_dir / 'summary_results.csv', index=False)
    
    def get_results_dir(self) -> Path:
        """Get experiment results directory."""
        return self.experiment_dir 