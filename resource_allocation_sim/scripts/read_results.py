# Read results from the sequential convergence study
# Read full_results_*.pkl and display the contents properly

import os
import json
import pickle
import pandas as pd
import numpy as np

results_dir = "results/sequential_convergence_study/sequential_convergence_study_20250602_152205"

print(f"Reading results from: {results_dir}")
print("=" * 60)

for file in os.listdir(results_dir):
    if file.endswith(".pkl"):
        print(f"\n📁 Processing file: {file}")
        
        with open(os.path.join(results_dir, file), "rb") as f:
            data = pickle.load(f)
        
        # Display metadata
        print("\n📊 METADATA:")
        metadata = data.get('metadata', {})
        for key, value in metadata.items():
            if key == 'base_config':
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # Process results
        print("\n📈 RESULTS SUMMARY:")
        results = data.get('results', [])
        print(f"  Number of configurations: {len(results)}")
        
        if results:
            config_result = results[0]
            print(f"  Configuration parameters: {config_result.get('config_params', {})}")
            
            episodes = config_result.get('episode_results', [])
            print(f"  Number of episodes: {len(episodes)}")
            
            if episodes:
                print("\n🎯 EPISODE ANALYSIS:")
                
                # Analyze each episode
                for ep_idx, episode in enumerate(episodes):
                    print(f"\n  Episode {ep_idx}:")
                    print(f"    Total cost: {episode.get('total_cost', 'N/A')}")
                    print(f"    Final consumption: {episode.get('final_consumption', 'N/A')}")
                    
                    # Agent analysis
                    agent_results = episode.get('agent_results', {})
                    print(f"    Number of agents: {len(agent_results)}")
                    
                    if agent_results:
                        print("    Agent final probabilities:")
                        for agent_id, agent_data in list(agent_results.items())[:3]:  # Show first 3 agents
                            final_probs = agent_data.get('prob', [])
                            if final_probs:
                                final_prob = final_probs[-1]
                                max_prob = np.max(final_prob)
                                preferred_resource = np.argmax(final_prob)
                                print(f"      Agent {agent_id}: max_prob={max_prob:.3f}, prefers resource {preferred_resource}")
                        
                        if len(agent_results) > 3:
                            print(f"      ... and {len(agent_results) - 3} more agents")
                
                # Summary statistics across all episodes
                print("\n📋 SUMMARY STATISTICS:")
                
                total_costs = [ep.get('total_cost', 0) for ep in episodes]
                print(f"  Total costs: min={min(total_costs):.3f}, max={max(total_costs):.3f}, mean={np.mean(total_costs):.3f}")
                
                # Create a simple DataFrame of episode summaries
                episode_summary = []
                for ep_idx, episode in enumerate(episodes):
                    summary = {
                        'episode': ep_idx,
                        'total_cost': episode.get('total_cost', 0),
                        'num_agents': len(episode.get('agent_results', {})),
                        'final_consumption': str(episode.get('final_consumption', []))
                    }
                    episode_summary.append(summary)
                
                df_episodes = pd.DataFrame(episode_summary)
                print(f"\n📊 EPISODES DATAFRAME:")
                print(df_episodes.to_string(index=False))
                
                # Agent convergence analysis
                print(f"\n🎲 AGENT CONVERGENCE ANALYSIS:")
                convergence_data = []
                
                for ep_idx, episode in enumerate(episodes):
                    agent_results = episode.get('agent_results', {})
                    for agent_id, agent_data in agent_results.items():
                        prob_history = agent_data.get('prob', [])
                        if prob_history:
                            final_probs = prob_history[-1]
                            max_prob = np.max(final_probs)
                            entropy = -np.sum(final_probs * np.log2(final_probs + 1e-10))
                            
                            convergence_data.append({
                                'episode': ep_idx,
                                'agent_id': agent_id,
                                'max_probability': max_prob,
                                'entropy': entropy,
                                'preferred_resource': np.argmax(final_probs),
                                'converged_strict': max_prob > 0.9 and entropy < 0.1,
                                'converged_relaxed': max_prob > 0.6 and entropy < 0.5
                            })
                
                if convergence_data:
                    df_convergence = pd.DataFrame(convergence_data)
                    print("Summary of agent convergence:")
                    print(df_convergence.groupby('episode').agg({
                        'max_probability': ['mean', 'max', 'min'],
                        'entropy': ['mean', 'max', 'min'],
                        'converged_strict': 'sum',
                        'converged_relaxed': 'sum'
                    }).round(3))

print("\n" + "=" * 60)
print("✅ Analysis complete!")