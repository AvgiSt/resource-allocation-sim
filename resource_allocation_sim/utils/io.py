"""Input/output utilities for saving and loading results."""

import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union, Optional
from datetime import datetime


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    results: Dict[str, Any], 
    filename: Union[str, Path],
    format: str = 'pickle',
    results_dir: Union[str, Path] = 'results'
) -> Path:
    """
    Save simulation results to file.
    
    Args:
        results: Results dictionary to save
        filename: Name of the file (without extension)
        format: Format to save in ('pickle', 'json', 'csv')
        results_dir: Directory to save results in
        
    Returns:
        Path to saved file
    """
    results_dir = ensure_directory(results_dir)
    
    # Add timestamp if not present
    if not any(char in str(filename) for char in ['_', '-']) or 'timestamp' not in str(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    if format == 'pickle':
        filepath = results_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'json':
        filepath = results_dir / f"{filename}.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = _make_json_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    elif format == 'csv':
        filepath = results_dir / f"{filename}.csv"
        # Convert to DataFrame if possible
        if isinstance(results, dict):
            df = pd.DataFrame(results)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError("Cannot save non-dict results as CSV")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filepath


def load_results(filepath: Union[str, Path]) -> Any:
    """
    Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Loaded results
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def _make_json_serializable(obj):
    """Convert object to JSON-serializable format."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def save_checkpoint(
    results: Dict[str, Any],
    checkpoint_dir: Union[str, Path] = 'results/checkpoints',
    checkpoint_name: Optional[str] = None
) -> Path:
    """
    Save a checkpoint of current results.
    
    Args:
        results: Results to checkpoint
        checkpoint_dir: Directory for checkpoints
        checkpoint_name: Optional name for checkpoint
        
    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = ensure_directory(checkpoint_dir)
    
    if checkpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}"
    
    return save_results(results, checkpoint_name, 'pickle', checkpoint_dir) 