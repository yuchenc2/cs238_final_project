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
        self.𝒮 = np.arange(num_states)
        self.𝒜 = np.arange(num_actions)
        self.D = np.array([[dirichlet(np.ones(num_states)) for _ in self.𝒜] for _ in self.𝒮])
        self.R = np.array([[norm(loc=0, scale=1) for _ in self.𝒜] for _ in self.𝒮])
        self.γ = discount_factor
        self.U = np.zeros(num_states)

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
        return expected_reward + self.γ * sum(T(s_prime) * self.U[s_prime] for s_prime in self.𝒮)

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

def encode_action(training_program, training_completion):
    category_code = TRAINING_CATEGORIES.index(training_program)
    return category_code * 2 + int(training_completion)

def calculate_reward(old_scores, new_scores):
    return sum([float(new) - float(old) for old, new in zip(old_scores, new_scores)])

def value_iteration(mdp, threshold=0.001):
    while True:
        delta = 0
        for s in mdp.𝒮:
            v = mdp.U[s]
            mdp.U[s] = max(mdp.lookahead(s, a) for a in mdp.𝒜)
            delta = max(delta, abs(v - mdp.U[s]))
        if delta < threshold:
            break

def calculate_cumulative_reward(mdp, start_probabilities):
    value_iteration(mdp)
    cumulative_reward = sum(start_probabilities[s] * mdp.U[s] for s in mdp.𝒮)
    return cumulative_reward

# Load and preprocess data
data = parse_data('small.txt')

# Initialize the Bayesian MDP
num_states = 100  # Adjust based on your data
num_actions = len(TRAINING_CATEGORIES) * 2  # 14 unique actions
mdp = BayesianMDP(num_states, num_actions, discount_factor=0.9)

# Train the model with the data
for row in data:
    s = int(row[0])  # Convert state to an index
    a = encode_action(row[8], row[9])  # Encoded action
    old_scores = row[3:8]  # Old scores
    new_scores = row[14:19]  # New scores
    r = calculate_reward(old_scores, new_scores)  # Reward
    s_prime = int(row[11])  # Convert next state to an index
    mdp.update(s, a, r, s_prime)

# Assuming uniform probability for starting states
start_probabilities = np.ones(num_states) / num_states
cumulative_reward = calculate_cumulative_reward(mdp, start_probabilities)
print(f"Cumulative Reward of the Learned Optimal Policy: {cumulative_reward}")
