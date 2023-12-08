import numpy as np
import json
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
        self.optimal_policy = {}

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
    
    def derive_optimal_policy(self):
        """Derive the optimal policy based on the current utility values."""
        self.optimal_policy = np.zeros(len(self.S), dtype=int)
        for s in self.S:
            best_action = np.argmax([self.lookahead(s, a) for a in self.A])
            self.optimal_policy[s] = best_action

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
    print("done with value iteration")
    mdp.derive_optimal_policy()
    print("Derived optimal policy")

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

def create_action_index_map(categories):
    """Create a mapping from training categories to indices."""
    return {action: idx for idx, action in enumerate(categories)}

value_iteration(mdp)

# After running the value iteration and deriving the optimal policy
action_index_map = create_action_index_map(TRAINING_CATEGORIES)
optimal_policy_indices = {state_idx: action_index_map[TRAINING_CATEGORIES[action_idx]] for state_idx, action_idx in enumerate(mdp.optimal_policy)}

# Saving the optimal policy to a file
output_file = "optimal_policy.json"
with open(output_file, 'w') as file:
    json.dump(optimal_policy_indices, file)

print(f"Optimal policy saved to {output_file}")