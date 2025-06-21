import numpy as np

from ..core.simulation import SimulationRunner
from ..utils.config import Config

# Create proper config object
config = Config(
    num_iterations=3,
    num_episodes=1,
    num_agents=10,  
    num_resources=3,
    weight=0.6,
    relative_capacity=[0.5, 0.5, 0.5]  # 50% relative capacity for testing
)

print("Testing updated mathematical model implementation...")
print(f"Config: {config.to_dict()}")
print("=" * 50)

# Run simulation
runner = SimulationRunner(config=config)
result = runner.run()

print("DETAILED RESULTS:")
print(result)

print("\nDETAILED ANALYSIS BY ITERATION:")

# Get number of iterations from any agent
num_iterations = len(list(result['agent_results'].values())[0]['prob'])

for iteration in range(num_iterations):
    print(f"\n=== ITERATION {iteration} ===")
    print("Agent | Action | Probabilities After Update")
    print("----- | ------ | --------------------------")
    
    for agent_id, agent_data in result['agent_results'].items():
        prob_history = agent_data['prob']
        action_history = agent_data['action']
        
        action = action_history[iteration] if iteration < len(action_history) else "N/A"
        probs = prob_history[iteration]
        prob_str = f"[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]"
        
        print(f"  {agent_id:>3} | {action:>6} | {prob_str}")
    
    # Show environment state for this iteration
    env_state = result['environment_state']
    if iteration < len(env_state['consumption_history']):
        consumption = env_state['consumption_history'][iteration]
        costs = env_state['cost_history'][iteration]
        print(f"\nConsumption: {consumption}")
        print(f"Costs: {costs}")

# Display key results
print("\n" + "=" * 50)
print("SUMMARY RESULTS:")
print(f"Final consumption: {result['final_consumption']}")
print(f"Total cost: {result['total_cost']:.4f}")

# Show agent convergence
print("\nFINAL AGENT STATE:")
for agent_id, agent_data in result['agent_results'].items():
    final_probs = agent_data['prob'][-1]
    max_prob = np.max(final_probs)
    preferred_resource = np.argmax(final_probs)
    print(f"Agent {agent_id}: max_prob={max_prob:.3f}, prefers resource {preferred_resource}")

print("\nTest completed successfully!")