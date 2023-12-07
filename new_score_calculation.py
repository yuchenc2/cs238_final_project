def evaluate_policy(data, policy, state_to_index, action_to_index, num_episodes=1000, max_steps=100):
    """Evaluate the given policy over a number of episodes."""
    total_reward = 0

    for _ in range(num_episodes):
        # Random initial state
        initial_row = data.iloc[random.randrange(len(data))]
        state = (initial_row['Job Title Numeric'], 
                 initial_row['Years of Experience in Months'],
                 initial_row['P_Score'], 
                 initial_row['C_Rating'])
        
        episode_reward = 0
        for _ in range(max_steps):
            state_idx = state_to_index[state]
            action_idx = policy[state_idx]
            action = unique_actions[action_idx]

            # Find possible transitions for this state-action pair
            transitions = data[(data[['Job Title Numeric', 
                                      'Years of Experience in Months', 
                                      'P_Score', 
                                      'C_Rating']].values == state).all(axis=1) 
                               & (data['Training Program'] == action)]
            
            if transitions.empty:
                break  # No transitions available, end the episode

            # Sample a transition
            sample = transitions.sample(n=1).iloc[0]
            reward = sample['Reward']
            episode_reward += reward

            # Update state to the next state
            state = (sample['Next Job Title Numeric'], 
                     sample['Next Years of Experience in Months'],
                     sample['Next P_Score'], 
                     sample['Next C_Rating'])

        total_reward += episode_reward

    # Average reward per episode
    average_reward = total_reward / num_episodes
    return average_reward


def evaluate_random_policy(data, unique_actions, num_episodes=1000, max_steps=100):
    """Evaluate a random policy over a number of episodes."""
    total_reward = 0

    for _ in range(num_episodes):
        # Random initial state
        initial_row = data.iloc[random.randrange(len(data))]
        state = (initial_row['Job Title Numeric'], 
                 initial_row['Years of Experience in Months'],
                 initial_row['P_Score'], 
                 initial_row['C_Rating'])

        episode_reward = 0
        for _ in range(max_steps):
            action = random.choice(unique_actions)

            # Find possible transitions for this state-action pair
            transitions = data[(data[['Job Title Numeric', 
                                      'Years of Experience in Months', 
                                      'P_Score', 
                                      'C_Rating']].values == state).all(axis=1) 
                               & (data['Training Program'] == action)]
            
            if transitions.empty:
                break  # No transitions available, end the episode

            # Sample a transition
            sample = transitions.sample(n=1).iloc[0]
            reward = sample['Reward']
            episode_reward += reward

            # Update state to the next state
            state = (sample['Next Job Title Numeric'], 
                     sample['Next Years of Experience in Months'],
                     sample['Next P_Score'], 
                     sample['Next C_Rating'])

        total_reward += episode_reward

    # Average reward per episode
    average_reward = total_reward / num_episodes
    return average_reward
