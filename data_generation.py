# Full code to generate data for 10 employees with discretized scores

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# Constants
NUM_EMPLOYEES = 100
JOB_TITLES = ["software engineer", "senior software engineer", "tech lead"]
TRAINING_CATEGORIES = [
    "Leadership and Management Training",
    "Project Management and Agile Methodologies",
    "Quality Assurance and Testing",
    "Soft Skills and Communication",
    "DevOps and Continuous Integration/Continuous Deployment (CI/CD)",
    "Security Training",
    "No Training"  # Including 'No Training' as an option
]

# Initial data generation for each employee
np.random.seed(0)  # For reproducibility
data = []
for _ in range(NUM_EMPLOYEES):
    job_title = np.random.choice(JOB_TITLES)
    years_of_experience = round(np.random.normal(5, 2), 1)  # Average 5 years, std 2, rounded to 1 decimal place, converted to months
    time_at_company = round(np.random.uniform(0, 5), 1)  # Up to 5 years at company, rounded to 1 decimal place
    e_score = round(np.random.uniform(1, 5), 1)  # Engagement score, rounded to 1 decimal place
    s_score = round(np.random.uniform(1, 5), 1)  # Satisfaction score, rounded to 1 decimal place
    wlb_score = round(np.random.uniform(1, 5), 1)  # Work-life balance score, rounded to 1 decimal place
    p_score = round(np.random.uniform(1, 5), 1)  # Performance score, rounded to 1 decimal place
    c_rating = round(np.random.uniform(1, 5), 1)  # Current employee rating, rounded to 1 decimal place
    data.append([job_title, years_of_experience, time_at_company, e_score, s_score, wlb_score, p_score, c_rating])

# Define hypothetical covariance matrix (8x8 for 8 variables)
cov_matrix = np.array([
    [1, 0.6, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3],  # Job Title
    [0.6, 1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3],  # Years of Experience
    [0.5, 0.2, 1, 0.3, 0.3, 0.3, 0.3, 0.3],  # Time at Company
    [0.3, 0.3, 0.3, 1, 0.4, 0.4, 0.4, 0.4],  # E_Score
    [0.3, 0.3, 0.3, 0.4, 1, 0.4, 0.4, 0.4],  # S_Score
    [0.3, 0.3, 0.3, 0.4, 0.4, 1, 0.4, 0.4],  # WLB_Score
    [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 1, 0.4],  # P_Score
    [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 1]   # C_Rating
])

training_probabilities = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88]  # Adjust these probabilities as needed. Last one is No Training.

# Training impact assumptions
p_completion = 0.7  # Probability of completing training
training_impact = {
    "Leadership and Management Training": {"P_Score": 0.2, "C_Rating": 0.2},
    "Project Management and Agile Methodologies": {"E_Score": 0.15, "P_Score": 0.15},
    "Quality Assurance and Testing": {"P_Score": 0.1, "C_Rating": 0.1},
    "Soft Skills and Communication": {"E_Score": 0.15, "S_Score": 0.15, "C_Rating": 0.15},
    "DevOps and Continuous Integration/Continuous Deployment (CI/CD)": {"P_Score": 0.2},
    "Security Training": {"P_Score": 0.15, "C_Rating": 0.15},
    "No Training": {} }

# Generating the state, action, and next state for each employee over time
all_employee_data = []
job_title_numeric = {'software engineer': 0, 'senior software engineer': 1, 'tech lead': 2}

for row in data:
    job_title_numeric_value = job_title_numeric[row[0]]
    years_of_experience_in_months = int(row[1] * 12) # Convert years to months
    months_at_company = int(row[2] * 12)  # Convert years to months

    # Current performance metrics, will be updated in each loop iteration
    current_metrics = np.array(row[3:8])
    current_metrics = np.round(current_metrics, 1)  # Discretizing to one decimal place

    for month in range(months_at_company + 1):
        training_program = np.random.choice(TRAINING_CATEGORIES, p=training_probabilities)
        training_completed = 0
        if training_program != "No Training":
            training_completed = np.random.binomial(1, p_completion)

        # State Vector including current performance metrics
        state_vector = np.array([
            job_title_numeric_value,
            years_of_experience_in_months + month,
            month,  # Time at Company in months
            *current_metrics
        ])

        # Adjust mean improvements based on training program and completion
        mean_improvement = np.zeros(5)  # E_Score, S_Score, WLB_Score, P_Score, C_Rating
        if training_completed:
            for metric, improvement in training_impact.get(training_program, {}).items():
                index = ["E_Score", "S_Score", "WLB_Score", "P_Score", "C_Rating"].index(metric)
                mean_improvement[index] = improvement

        # Generate next state using the multivariate distribution with adjusted mean
        next_state_vector = multivariate_normal.rvs(mean=current_metrics + mean_improvement, cov=cov_matrix[3:, 3:])
        next_state_vector = np.round(next_state_vector, 1)  # Discretizing to one decimal place

        # Assuming weights for all scores are 1
        reward = (next_state_vector[0] - current_metrics[0]) + (next_state_vector[1] - current_metrics[1])\
              + (next_state_vector[2] - current_metrics[2]) + (next_state_vector[3] - current_metrics[3])

        # Append each month's data
        monthly_data = [job_title_numeric_value, years_of_experience_in_months + month, month, *current_metrics, 
                        training_program, training_completed, reward, job_title_numeric_value, years_of_experience_in_months + month + 1, month + 1, *next_state_vector]
        all_employee_data.append(monthly_data)

        # Update current metrics for the next iteration
        current_metrics = next_state_vector

# Create DataFrame for all employee data over time
columns = [
    "Job Title Numeric", "Years of Experience", "Time at Company", "E_Score", "S_Score", "WLB_Score", "P_Score", "C_Rating",
    "Training Program", "Training Completion", "Reward", "Job Title Numeric", "New Years of Experience", "New Time at Company",
    "Next E_Score", "Next S_Score", "Next WLB_Score", "Next P_Score", "Next C_Rating"
]


all_data_df = pd.DataFrame(all_employee_data, columns=columns)

# Saving the generated data to a file named 'sample.txt'
all_data_df.to_csv('sample.txt', index=False, sep='\t')