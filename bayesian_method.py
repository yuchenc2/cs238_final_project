import numpy as np
from scipy.stats import dirichlet, norm

TRAINING_CATEGORIES = [
    "Leadership and Management Training",
    "Project Management and Agile Methodologies",
    "Quality Assurance and Testing",
    "Soft Skills and Communication",
    "DevOps and Continuous Integration/Continuous Deployment (CI/CD)",
    "Security Training",
    "No Training"
]

class BayesianMDP:
    def __init__(self, num_states, num_actions, discount_factor):
        self.state_index_map = {}  # Map state tuples to indices
        self.current_state_index = 0
        self.S = np.arange(num_states)
        self.A = np.arange(num_actions)
        self.D = np.array([[dirichlet(np.ones(num_states)) for _ in self.A] for _ in self.S])
        self.R = np.array([[norm(loc=0, scale=1) for _ in self.A] for _ in self.S])
        self.gamma = discount_factor
        self.U = np.zeros(num_states)

    def get_state_index(self, state_tuple):
        if state_tuple not in self.state_index_map:
            self.state_index_map[state_tuple] = self.current_state_index
            self.current_state_index += 1
        return self.state_index_map[state_tuple]

    def update_reward_model(self, s, a, observed_reward):
        current_mean, current_std = self.R[s, a].mean(), self.R[s, a].std()
        new_mean = (current_mean + observed_reward) / 2
        new_std = np.sqrt((current_std**2 + (observed_reward - new_mean)**2) / 2)
        self.R[s, a] = norm(loc=new_mean, scale=new_std)

    def lookahead(self, s, a):
        n = sum(self.D[s, a].alpha)
        if n == 0:
            return 0.0
        expected_reward = self.R[s, a].mean()
        T = lambda s_prime: self.D[s, a].alpha[s_prime] / n
        return expected_reward + self.gamma * sum(T(s_prime) * self.U[s_prime] for s_prime in self.S)

    def update(self, s, a, r, s_prime):
        self.update_reward_model(s, a, r)
        alpha = self.D[s, a].alpha
        alpha[s_prime] += 1
        self.D[s, a] = dirichlet(alpha)
        self.U[s] = self.lookahead(s, a)
        return self

def parse_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first row containing column names
    data = [line.strip().split('\t') for line in lines]
    return data

def encode_action(training_program):
    return TRAINING_CATEGORIES.index(training_program)

def value_iteration(mdp, threshold=0.001):
    while True:
        delta = 0
        for s in mdp.S:
            v = mdp.U[s]
            mdp.U[s] = max(mdp.lookahead(s, a) for a in mdp.A)
            delta = max(delta, abs(v - mdp.U[s]))
        if delta < threshold:
            break

def calculate_cumulative_reward(mdp, start_probabilities):
    value_iteration(mdp)
    cumulative_reward = sum(start_probabilities[s] * mdp.U[s] for s in mdp.S)
    return cumulative_reward

def simulate_random_policy(mdp, start_probabilities, num_simulations=1000):
    np.random.seed(0)  # For reproducibility
    total_reward = 0
    for _ in range(num_simulations):
        s = np.random.choice(mdp.S, p=start_probabilities)
        a = np.random.choice(mdp.A)
        total_reward += mdp.R[s, a].mean()
    return total_reward / num_simulations

# Load and preprocess data
data = parse_data('sample.txt')

# Adjust number of states and actions based on your data
num_states = 9075
num_actions = len(TRAINING_CATEGORIES)
mdp = BayesianMDP(num_states, num_actions, discount_factor=0.9)

count = 0
# Train the model with the data
for row in data:
    s = (int(row[0]), int(row[1]), float(row[2]), float(row[3]))
    s_index = mdp.get_state_index(s)
    a = encode_action(row[4])
    r = float(row[5])
    s_prime = (int(row[0]), int(row[6]), float(row[7]), float(row[8]))
    s_prime_index = mdp.get_state_index(s_prime)
    mdp.update(s_index, a, r, s_prime_index)

# Assuming uniform probability for starting states
start_probabilities = np.ones(num_states) / num_states

# Calculate cumulative reward for learned optimal policy using the updated value function
optimal_cumulative_reward = sum(start_probabilities[s] * mdp.U[s] for s in mdp.S)
print(f"Cumulative Reward of the Learned Optimal Policy: {optimal_cumulative_reward}")

# Calculate cumulative reward for random policy
random_cumulative_reward = simulate_random_policy(mdp, start_probabilities)
print(f"Cumulative Reward of the Random Policy: {random_cumulative_reward}")