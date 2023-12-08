import random
import pandas as pd
import json

# Load the policy from the file
with open('medium.policy', 'r') as file:
    policy = json.load(file)

# Convert policy keys from string to integer (if necessary)
policy = {int(k): v for k, v in policy.items()}

# Load the data
data = pd.read_csv('medium.txt', delimiter='\t')  # Ensure this matches your data file

# Unique states and actions
unique_states = data[['Job Title Numeric', 'Years of Experience in Months', 'P_Score', 'C_Rating']].drop_duplicates()
unique_actions = data['Training Program'].unique()

# State and action mapping to integers for Q-table
state_to_index = {tuple(state): idx for idx, state in enumerate(unique_states.values)}
action_to_index = {action: idx for idx, action in enumerate(unique_actions)}

def evaluate_policy(data, policy, max_steps = 100, discount_factor=1.0):
    """Evaluate the given policy for each unique state."""
    total_reward = 0
    num_states = len(unique_states)
    count = 0
    for _, state_row in unique_states.iterrows():
        # print(state_row)
        print(count)
        count= count + 1
        state = tuple(state_row)
        episode_reward = 0
        current_discount = 1.0
        for _ in range(max_steps):
            state_idx = state_to_index[state]
            action_idx = policy.get(state_idx, random.choice(list(action_to_index.values())))  # Default to random action if state not in policy
            action = unique_actions[action_idx]

            transitions = data[(data[['Job Title Numeric', 'Years of Experience in Months', 'P_Score', 'C_Rating']].values == state).all(axis=1) 
                               & (data['Training Program'] == action)]
            
            if transitions.empty:
                break  # No transitions available, end the episode

            sample = transitions.sample(n=1).iloc[0]
            reward = sample['Reward']
            episode_reward += current_discount * reward
            current_discount *= discount_factor

            state = (sample['Job Title Numeric'], sample['Years of Experience in Months'],
                     sample['P_Score'], sample['C_Rating'])

        total_reward += episode_reward

    # Average reward per unique state
    average_reward = total_reward / num_states
    return average_reward

def evaluate_random_policy(data, unique_actions, max_steps=100, discount_factor=1.0):
    """Evaluate a random policy for each unique state."""
    total_reward = 0
    num_states = len(unique_states)
    for _, state_row in unique_states.iterrows():
        state = tuple(state_row)
        episode_reward = 0
        current_discount = 1.0
        for _ in range(max_steps):
            action = random.choice(unique_actions)

            transitions = data[(data[['Job Title Numeric', 'Years of Experience in Months', 'P_Score', 'C_Rating']].values == state).all(axis=1) 
                               & (data['Training Program'] == action)]
            
            if transitions.empty:
                break  # No transitions available, end the episode

            sample = transitions.sample(n=1).iloc[0]
            reward = sample['Reward']
            episode_reward += current_discount * reward
            current_discount *= discount_factor

            state = (sample['Job Title Numeric'], sample['Years of Experience in Months'],
                     sample['P_Score'], sample['C_Rating'])

        total_reward += episode_reward

    # Average reward per unique state
    average_reward = total_reward / num_states
    return average_reward

average_reward_optimal = evaluate_policy(data, policy, 100, 1.0)
print(f"Average Reward (Optimal Policy): {average_reward_optimal}")
average_reward_random = evaluate_random_policy(data, unique_actions, 100, 1.0)
print(f"Average Reward (Random Policy): {average_reward_random}")