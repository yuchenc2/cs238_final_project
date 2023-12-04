import numpy as np
import random

# Constants
discountFactor = 1
learningRate = 0.1
exploreRate = 0.1
numEpisodes = 10000

# Unique states and actions
unique_states = data[['Job Title Numeric', 'Years of Experience', 'Time at Company', 'E_Score', 'S_Score', 'WLB_Score', 'P_Score', 'C_Rating']].drop_duplicates()
unique_actions = data['Training Program'].unique()

# State and action mapping to integers for Q-table
state_to_index = {tuple(state): idx for idx, state in enumerate(unique_states.values)}
action_to_index = {action: idx for idx, action in enumerate(unique_actions)}

# Initialize Q-table
num_states = len(unique_states)
num_actions = len(unique_actions)
QValues = np.zeros((num_states, num_actions))

def runQlearning(data, numEpisodes):
    for episode in range(numEpisodes):
        print(f"Episode {episode + 1}/{numEpisodes}")
        # Random initial state
        state = tuple(data.iloc[random.randrange(len(data))][['Job Title Numeric', 'Years of Experience', 'Time at Company', 'E_Score', 'S_Score', 'WLB_Score', 'P_Score', 'C_Rating']].values)
        state_idx = state_to_index[state]
        
        for _ in range(500):
            # Choose action
            if random.uniform(0, 1) < exploreRate:
                action = random.choice(list(action_to_index.values()))
            else:
                action = np.argmax(QValues[state_idx])

            # Find possible transitions
            transitions = data[(data[['Job Title Numeric', 'Years of Experience', 'Time at Company', 'E_Score', 'S_Score', 'WLB_Score', 'P_Score', 'C_Rating']].values == state).all(axis=1) & (data['Training Program'] == unique_actions[action])]

            if transitions.empty:
                # No valid transitions, end episode
                break
            else:
                # Sample a transition
                sample = transitions.sample(n=1).iloc[0]
                nextState = tuple(sample[['Job Title Numeric.1', 'New Years of Experience', 'New Time at Company', 'Next E_Score', 'Next S_Score', 'Next WLB_Score', 'Next P_Score', 'Next C_Rating']].values)
                nextState_idx = state_to_index[nextState]
                reward = sample['Reward']

                # Q-learning update
                td_target = reward + discountFactor * np.max(QValues[nextState_idx])
                QValues[state_idx, action] += learningRate * (td_target - QValues[state_idx, action])

                # Update state
                state_idx = nextState_idx

    # Policy: action with highest Q-value for each state
    return np.argmax(QValues, axis=1)

# Run Q-learning
policy = runQlearning(data, numEpisodes)

# Mapping back from index to action
policy_actions = [unique_actions[action_idx] for action_idx in policy]

# Save policy
policy_file_path = '/mnt/data/medium.policy'
with open(policy_file_path, 'w') as file:
    for action in policy_actions:
        file.write(f"{action}\n")

policy_file_path
