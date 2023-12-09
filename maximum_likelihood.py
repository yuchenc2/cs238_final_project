import pandas as pd
import numpy as np
from collections import defaultdict
import random
import os
import time
import json

input_file = 'large.txt'
data = pd.read_csv(input_file, delimiter='\t')

unique_states = data[['Job Title Numeric', 'Years of Experience in Months', 'P_Score', 'C_Rating']].drop_duplicates()
unique_actions = data['Training Program'].unique()
state_to_index = {tuple(state): idx for idx, state in enumerate(unique_states.values)}
action_to_index = {action: idx for idx, action in enumerate(unique_actions)}

def calculate_reward(row):
    return sum([
        row['Next P_Score'] - row['P_Score'],
        row['Next C_Rating'] - row['C_Rating']
    ])


data['Reward'] = data.apply(calculate_reward, axis=1)

def utility_function(row):
    return row['P_Score'] + row['C_Rating']

def calculate_rewards_and_transitions(data):
    rewards = defaultdict(lambda: defaultdict(float))
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    state_action_counts = defaultdict(lambda: defaultdict(int))

    for _, row in data.iterrows():
        state = tuple(row[:4])
        action = row[4]
        next_state = tuple(row[6:9])

        rewards[state][action] += row['Reward']
        transition_counts[state][action][next_state] += 1
        state_action_counts[state][action] += 1

    for state in rewards:
        for action in rewards[state]:
            rewards[state][action] /= state_action_counts[state][action]

    transition_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for state in transition_counts:
        for action in transition_counts[state]:
            for next_state in transition_counts[state][action]:
                count = transition_counts[state][action][next_state]
                transition_probabilities[state][action][next_state] = count / state_action_counts[state][action]

    return rewards, transition_probabilities

start_time = time.time()

rewards, transition_probabilities = calculate_rewards_and_transitions(data)

current_states = data.iloc[:, :4].drop_duplicates()
utilities = {tuple(row): utility_function(row) for _, row in current_states.iterrows()}

discount_factor = 0.5
convergence_threshold = 0.01
max_iterations = 100

for iteration in range(max_iterations):
    updated_utilities = utilities.copy()
    delta = 0
    print(f"Iteration {iteration}")

    print(len(utilities))
    for state in utilities:
        for action in transition_probabilities[state]:
            expected_utility = 0
            for next_state, prob in transition_probabilities[state][action].items():
                expected_utility += prob * utilities.get(next_state, 0)
            updated_utility = rewards[state][action] + discount_factor * expected_utility

            delta = max(delta, abs(updated_utilities[state] - updated_utility))
            updated_utilities[state] = updated_utility

    utilities = updated_utilities
    if delta < convergence_threshold:
        print("Converged")
        break

# Determine the optimal policy and final scores
optimal_policy = defaultdict(tuple)
for state in utilities:
    best_action = None
    best_utility = float('-inf')
    for action in transition_probabilities[state]:
        for next_state in transition_probabilities[state][action]:
            current_utility = transition_probabilities[state][action][next_state] * (utilities.get(next_state, 0))
            if current_utility > best_utility:
                best_utility = current_utility
                best_action = action
    optimal_policy[state] = best_action

optimal_policy_indices = {}

for state, action in optimal_policy.items():
    state_idx = state_to_index[state]
    action_idx = action_to_index[action]
    optimal_policy_indices[state_idx] = action_idx

base_name = os.path.splitext(input_file)[0]
output_file = f"{base_name}.policy"

with open(output_file, 'w') as file:
    json.dump(optimal_policy_indices, file)

print(f"Optimal policy saved to {output_file}")

end_time = time.time() 

duration = end_time - start_time
print(f"Process completed in {duration:.2f} seconds")