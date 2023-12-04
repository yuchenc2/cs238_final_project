def evaluate_policy(data, policy, state_to_index, action_to_index, num_episodes=1000):
    """Evaluate the given policy over a number of episodes."""
    total_reward = 0

    for _ in range(num_episodes):
        # Random initial state
        state = tuple(data.iloc[random.randrange(len(data))][['Job Title Numeric', 'Years of Experience', 'Time at Company', 'E_Score', 'S_Score', 'WLB_Score', 'P_Score', 'C_Rating']].values)
        state_idx = state_to_index[state]

        # Apply policy to the state
        action_idx = policy[state_idx]
        action = unique_actions[action_idx]

        # Find possible transitions
        transitions = data[(data[['Job Title Numeric', 'Years of Experience', 'Time at Company', 'E_Score', 'S_Score', 'WLB_Score', 'P_Score', 'C_Rating']].values == state).all(axis=1) & (data['Training Program'] == action)]

        if not transitions.empty:
            # Sample a transition and add reward
            sample = transitions.sample(n=1).iloc[0]
            reward = sample['Reward']
            total_reward += reward

    # Average reward per episode
    average_reward = total_reward / num_episodes
    return average_reward

def evaluate_random_policy(data, unique_actions, num_episodes=1000):
    """Evaluate a random policy over a number of episodes."""
    total_reward = 0

    for _ in range(num_episodes):
        # Random action
        action = random.choice(unique_actions)

        # Sample random transition for this action
        transitions = data[data['Training Program'] == action]
        if not transitions.empty:
            sample = transitions.sample(n=1).iloc[0]
            reward = sample['Reward']
            total_reward += reward

    # Average reward per episode
    average_reward = total_reward / num_episodes
    return average_reward

# Evaluate the learned policy
policy_value = evaluate_policy(data, policy_fixed_reduced, state_to_index, action_to_index)

# Evaluate a random policy
random_policy_value = evaluate_random_policy(data, unique_actions)

policy_value, random_policy_value
