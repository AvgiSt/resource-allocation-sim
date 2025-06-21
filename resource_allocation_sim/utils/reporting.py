"""Comprehensive reporting utilities for simulation analysis."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

def generate_comprehensive_report(
    results: Union[Dict, List[Dict]], 
    output_dir: str,
    report_name: str = "comprehensive_report"
) -> str:
    """
    Generate a comprehensive HTML report from simulation results.
    
    Args:
        results: Simulation results (single dict or list of dicts)
        output_dir: Directory to save the report
        report_name: Name of the report file (without extension)
        
    Returns:
        Path to the generated report file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert single result to list for consistency
    if isinstance(results, dict):
        results_list = [results]
    else:
        results_list = results
    
    # Generate report content
    html_content = _generate_html_report(results_list, output_dir)
    
    # Save report
    report_path = os.path.join(output_dir, f"{report_name}.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Generate summary JSON
    summary_data = _generate_summary_data(results_list)
    summary_path = os.path.join(output_dir, f"{report_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    return report_path

def _generate_html_report(results_list: List[Dict], output_dir: str) -> str:
    """Generate the HTML content for the report."""
    
    # Calculate statistics
    stats = _calculate_report_statistics(results_list)
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resource Allocation Simulation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .metric-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #27ae60;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Resource Allocation Simulation Report</h1>
        
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">Total Simulations</div>
                <div class="metric-value">{len(results_list)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Average Final Cost</div>
                <div class="metric-value">{stats['avg_cost']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Cost Standard Deviation</div>
                <div class="metric-value">{stats['std_cost']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Average Entropy</div>
                <div class="metric-value">{stats['avg_entropy']:.4f}</div>
            </div>
        </div>
        
        <h2>Simulation Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Number of Agents</td><td>{stats['config']['num_agents']}</td></tr>
            <tr><td>Number of Resources</td><td>{stats['config']['num_resources']}</td></tr>
            <tr><td>Number of Iterations</td><td>{stats['config']['num_iterations']}</td></tr>
            <tr><td>Learning Weight</td><td>{stats['config']['weight']}</td></tr>
            <tr><td>Resource Capacities</td><td>{stats['config']['capacity']}</td></tr>
        </table>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>
            <tr>
                <td>Final Cost</td>
                <td>{stats['avg_cost']:.4f}</td>
                <td>{stats['std_cost']:.4f}</td>
                <td>{stats['min_cost']:.4f}</td>
                <td>{stats['max_cost']:.4f}</td>
            </tr>
            <tr>
                <td>Entropy</td>
                <td>{stats['avg_entropy']:.4f}</td>
                <td>{stats['std_entropy']:.4f}</td>
                <td>{stats['min_entropy']:.4f}</td>
                <td>{stats['max_entropy']:.4f}</td>
            </tr>
            <tr>
                <td>Gini Coefficient</td>
                <td>{stats['avg_gini']:.4f}</td>
                <td>{stats['std_gini']:.4f}</td>
                <td>{stats['min_gini']:.4f}</td>
                <td>{stats['max_gini']:.4f}</td>
            </tr>
        </table>
        
        <h2>Resource Distribution</h2>
        <table>
            <tr><th>Resource</th><th>Average Consumption</th><th>Capacity</th><th>Utilisation (%)</th></tr>
"""
    
    # Add resource distribution table
    for i, (avg_consumption, capacity) in enumerate(zip(stats['avg_consumption'], stats['config']['capacity'])):
        utilisation = (avg_consumption / capacity) * 100 if capacity > 0 else 0
        html += f"""
            <tr>
                <td>Resource {i+1}</td>
                <td>{avg_consumption:.4f}</td>
                <td>{capacity:.4f}</td>
                <td>{utilisation:.2f}%</td>
            </tr>
"""
    
    html += """
        </table>
        
        <h2>Visualisations</h2>
        <p>Generated plots are available in the output directory:</p>
        <ul>
"""
    
    # List available plots
    plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    for plot_file in plot_files:
        html += f"            <li>{plot_file}</li>\n"
    
    html += f"""
        </ul>
        
        <h2>Detailed Results</h2>
        <p>Individual simulation results:</p>
        <table>
            <tr><th>Run</th><th>Final Cost</th><th>Entropy</th><th>Gini Coeff</th><th>Final Consumption</th></tr>
"""
    
    # Add individual results
    for i, result in enumerate(results_list):
        from ..evaluation.metrics import calculate_entropy, calculate_gini_coefficient
        
        final_consumption = result.get('final_consumption', [])
        entropy = calculate_entropy(np.array(final_consumption)) if final_consumption else 0
        gini = calculate_gini_coefficient(np.array(final_consumption)) if final_consumption else 0
        
        consumption_str = ', '.join([f"{x:.3f}" for x in final_consumption])
        
        html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{result.get('total_cost', 0):.4f}</td>
                <td>{entropy:.4f}</td>
                <td>{gini:.4f}</td>
                <td>[{consumption_str}]</td>
            </tr>
"""
    
    html += f"""
        </table>
        
        <div class="timestamp">
            Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""
    
    return html

def _calculate_report_statistics(results_list: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics for the report."""
    from ..evaluation.metrics import calculate_entropy, calculate_gini_coefficient
    
    # Extract metrics
    costs = [r.get('total_cost', 0) for r in results_list]
    entropies = []
    ginis = []
    consumptions = []
    
    for result in results_list:
        final_consumption = result.get('final_consumption', [])
        if final_consumption:
            consumption_array = np.array(final_consumption)
            entropies.append(calculate_entropy(consumption_array))
            ginis.append(calculate_gini_coefficient(consumption_array))
            consumptions.append(final_consumption)
    
    # Get configuration from first result
    config_data = results_list[0] if results_list else {}
    
    # Calculate average consumption per resource
    if consumptions:
        avg_consumption = np.mean(consumptions, axis=0).tolist()
    else:
        avg_consumption = []
    
    return {
        'avg_cost': np.mean(costs) if costs else 0,
        'std_cost': np.std(costs) if costs else 0,
        'min_cost': np.min(costs) if costs else 0,
        'max_cost': np.max(costs) if costs else 0,
        'avg_entropy': np.mean(entropies) if entropies else 0,
        'std_entropy': np.std(entropies) if entropies else 0,
        'min_entropy': np.min(entropies) if entropies else 0,
        'max_entropy': np.max(entropies) if entropies else 0,
        'avg_gini': np.mean(ginis) if ginis else 0,
        'std_gini': np.std(ginis) if ginis else 0,
        'min_gini': np.min(ginis) if ginis else 0,
        'max_gini': np.max(ginis) if ginis else 0,
        'avg_consumption': avg_consumption,
        'config': {
            'num_agents': config_data.get('num_agents', 'N/A'),
            'num_resources': config_data.get('num_resources', 'N/A'),
            'num_iterations': config_data.get('num_iterations', 'N/A'),
            'weight': config_data.get('weight', 'N/A'),
            'capacity': config_data.get('capacity', [])
        }
    }

def _generate_summary_data(results_list: List[Dict]) -> Dict[str, Any]:
    """Generate summary data for JSON export."""
    stats = _calculate_report_statistics(results_list)
    
    return {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'num_simulations': len(results_list),
            'report_version': '1.0'
        },
        'summary_statistics': {
            'cost': {
                'mean': stats['avg_cost'],
                'std': stats['std_cost'],
                'min': stats['min_cost'],
                'max': stats['max_cost']
            },
            'entropy': {
                'mean': stats['avg_entropy'],
                'std': stats['std_entropy'],
                'min': stats['min_entropy'],
                'max': stats['max_entropy']
            },
            'gini_coefficient': {
                'mean': stats['avg_gini'],
                'std': stats['std_gini'],
                'min': stats['min_gini'],
                'max': stats['max_gini']
            }
        },
        'configuration': stats['config'],
        'resource_utilisation': [
            {
                'resource_id': i+1,
                'average_consumption': consumption,
                'capacity': stats['config']['capacity'][i] if i < len(stats['config']['capacity']) else 1.0,
                'utilisation_percentage': (consumption / stats['config']['capacity'][i] * 100) 
                    if i < len(stats['config']['capacity']) and stats['config']['capacity'][i] > 0 else 0
            }
            for i, consumption in enumerate(stats['avg_consumption'])
        ]
    }

def generate_experiment_summary(
    experiment_results: Dict[str, List[Dict]],
    output_dir: str,
    experiment_name: str = "experiment_summary"
) -> str:
    """
    Generate a summary report for multiple experimental conditions.
    
    Args:
        experiment_results: Dictionary mapping condition names to result lists
        output_dir: Directory to save the report
        experiment_name: Name of the experiment
        
    Returns:
        Path to the generated summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = {}
    
    for condition_name, results_list in experiment_results.items():
        stats = _calculate_report_statistics(results_list)
        summary_data[condition_name] = {
            'num_runs': len(results_list),
            'mean_cost': stats['avg_cost'],
            'mean_entropy': stats['avg_entropy'],
            'mean_gini': stats['avg_gini'],
            'cost_std': stats['std_cost'],
            'entropy_std': stats['std_entropy'],
            'gini_std': stats['std_gini']
        }
    
    # Create comparison DataFrame
    df = pd.DataFrame(summary_data).T
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{experiment_name}_comparison.csv")
    df.to_csv(csv_path)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{experiment_name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    return json_path 