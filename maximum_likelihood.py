import pandas as pd
import numpy as np
from collections import defaultdict

# Load the data from the text file
data = pd.read_csv('sample.txt', delimiter='\t')

# Define the reward calculation based on the sum of score differences
def calculate_reward(row):
    return sum([
        row['Next E_Score'] - row['E_Score'],
        row['Next S_Score'] - row['S_Score'],
        row['Next WLB_Score'] - row['WLB_Score'],
        row['Next P_Score'] - row['P_Score'],
        row['Next C_Rating'] - row['C_Rating']
    ])

# Add the calculated reward to each row in the dataset
data['Reward'] = data.apply(calculate_reward, axis=1)

# Define utility function based on the state features
def utility_function(row):
    # Placeholder utility function, can be adjusted based on specific needs
    return row['E_Score'] + row['S_Score'] + row['WLB_Score'] + row['P_Score'] + row['C_Rating']

# Calculate rewards and transition probabilities
def calculate_rewards_and_transitions(data):
    rewards = defaultdict(lambda: defaultdict(float))
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    state_action_counts = defaultdict(lambda: defaultdict(int))

    for _, row in data.iterrows():
        state = tuple(row[:8])  # Current state features
        action = tuple(row[8:10])  # Action features
        next_state = tuple(row[11:19])  # Next state features

        rewards[state][action] += row['Reward']
        transition_counts[state][action][next_state] += 1
        state_action_counts[state][action] += 1

    # Average the rewards for each (state, action) pair
    for state in rewards:
        for action in rewards[state]:
            rewards[state][action] /= state_action_counts[state][action]

    # Calculate transition probabilities
    transition_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for state in transition_counts:
        for action in transition_counts[state]:
            for next_state in transition_counts[state][action]:
                count = transition_counts[state][action][next_state]
                transition_probabilities[state][action][next_state] = count / state_action_counts[state][action]

    return rewards, transition_probabilities

# Calculate rewards and transition probabilities
rewards, transition_probabilities = calculate_rewards_and_transitions(data)

# Initialize utilities for each unique state
current_states = data.iloc[:, :8].drop_duplicates()
utilities = {tuple(row): utility_function(row) for _, row in current_states.iterrows()}

# Define parameters for the iterative process
learning_rate = 0.1
convergence_threshold = 0.01
max_iterations = 1000

# Iterative update process
for iteration in range(max_iterations):
    updated_utilities = utilities.copy()
    delta = 0
    print(iteration)
    for _, row in data.iterrows():
        state = tuple(row[:8])
        action = tuple(row[8:10])
        next_state = tuple(row[11:19])
        reward = rewards[state][action]

        old_utility = utilities.get(state, 0)
        transition_prob = transition_probabilities[state][action].get(next_state, 0)
        updated_utilities[state] += learning_rate * (transition_prob * (reward + utilities.get(next_state, 0)) - utilities.get(state, 0))
        delta = max(delta, abs(old_utility - updated_utilities[state]))

    utilities = updated_utilities
    if delta < convergence_threshold:
        print("Converged")
        break

# Determine the optimal policy and final scores
optimal_policy = defaultdict(tuple)
final_scores = defaultdict(float)

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
    final_scores[state] = best_utility

# Extracting results
cumulative_reward_optimal_policy = sum(final_scores.values())
print(f"Final Score: {cumulative_reward_optimal_policy}")