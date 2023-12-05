import pandas as pd
import numpy as np
from collections import defaultdict
import random

# Load the data from the text file
data = pd.read_csv('sample.txt', delimiter='\t')

# Define the reward calculation based on the sum of score differences
def calculate_reward(row):
    return sum([
        row['Next P_Score'] - row['P_Score'],
        row['Next C_Rating'] - row['C_Rating']
    ])

# Add the calculated reward to each row in the dataset
data['Reward'] = data.apply(calculate_reward, axis=1)

# Define utility function based on the state features
def utility_function(row):
    return row['P_Score'] + row['C_Rating']

# Calculate rewards and transition probabilities
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
current_states = data.iloc[:, :4].drop_duplicates()
utilities = {tuple(row): utility_function(row) for _, row in current_states.iterrows()}

# Define parameters for the iterative process
learning_rate = 0.1
convergence_threshold = 0.01
max_iterations = 100

# Iterative update process
for iteration in range(max_iterations):
    updated_utilities = utilities.copy()
    delta = 0
    print(iteration)
    for _, row in data.iterrows():
        state = tuple(row[:4])
        action = row[4]
        next_state = tuple(row[6:9])
        reward = rewards[state][action]

        old_utility = utilities.get(state, 0)
        transition_prob = transition_probabilities[state][action].get(next_state, 0)
        updated_utilities[state] += learning_rate * (reward + transition_prob * ( utilities.get(next_state, 0)) - utilities.get(state, 0))
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
print(f"Final Score (Optimal): {cumulative_reward_optimal_policy}")

# Function to choose a random action for each state
def choose_random_action(state, transition_probabilities):
    if state not in transition_probabilities:
        return None
    actions = list(transition_probabilities[state].keys())
    return random.choice(actions) if actions else None

# Calculate the cumulative reward for a random policy
cumulative_reward_random_policy = 0

for state in utilities:
    random_action = choose_random_action(state, transition_probabilities)
    if random_action:
        for next_state in transition_probabilities[state][random_action]:
            transition_prob = transition_probabilities[state][random_action][next_state]
            reward = rewards[state][random_action]
            cumulative_reward_random_policy += reward + transition_prob * (utilities.get(next_state, 0))

print(f"Final Score (Random): {cumulative_reward_random_policy}")
