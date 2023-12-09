import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm

NUM_EMPLOYEES = 10000
JOB_TITLES = ["software engineer", "senior software engineer", "tech lead"]
TRAINING_CATEGORIES = [
    "Leadership and Management Training",
    "Project Management and Agile Methodologies",
    "Quality Assurance and Testing",
    "Soft Skills and Communication",
    "DevOps and Continuous Integration/Continuous Deployment (CI/CD)",
    "Security Training",
    "No Training"  
]

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# Initial data generation for each employee
np.random.seed(0)  
data = []
for _ in range(NUM_EMPLOYEES):
    job_title = np.random.choice(JOB_TITLES)
    years_of_experience = round(get_truncated_normal(mean=5, sd=2, low=0, upp=10).rvs(), 1)
    p_score = np.random.randint(1, 6)  
    c_rating = np.random.randint(1, 6) 
    data.append([job_title, years_of_experience, p_score, c_rating])

cov_matrix = np.array([
    [0.2, 0.2, 0.1, 0.1],  # Job Title
    [0.2, 0.2, 0.1, 0.1],  # Years of Experience
    [0.1, 0.1, 0.2, 0.1],  # P_Score
    [0.1, 0.1, 0.1, 0.2]   # C_Rating
])

training_probabilities = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88]  # Adjust these probabilities as needed. Last one is No Training.

training_impact = {
    "Leadership and Management Training": {"P_Score": 1.0, "C_Rating": 1.0},
    "Project Management and Agile Methodologies": {"P_Score": 0.45},
    "Quality Assurance and Testing": {"P_Score": 0.7, "C_Rating": 0.7},
    "Soft Skills and Communication": {"C_Rating": 0.7},
    "DevOps and Continuous Integration/Continuous Deployment (CI/CD)": {"P_Score": 1.0},
    "Security Training": {"P_Score": 0.7, "C_Rating": 0.7},
    "No Training": {}
}

all_employee_data = []
job_title_numeric = {'software engineer': 0, 'senior software engineer': 1, 'tech lead': 2}

for row in data:
    job_title_numeric_value = job_title_numeric[row[0]]
    years_of_experience = row[1]  
    months_at_company = int(years_of_experience * 12)  

    current_metrics = np.array(row[2:4])

    for month in range(months_at_company + 1):
        training_program = np.random.choice(TRAINING_CATEGORIES, p=training_probabilities)

        state_vector = np.array([
            job_title_numeric_value,
            month,
            *current_metrics
        ])

        mean_improvement = np.zeros_like(state_vector)  
        for metric, improvement in training_impact.get(training_program, {}).items():
            index = ["P_Score", "C_Rating"].index(metric) + 2 
            mean_improvement[index] = improvement

        next_state_vector = multivariate_normal.rvs(mean=state_vector + mean_improvement, cov=cov_matrix)
        next_state_vector[2:] = np.round(next_state_vector[2:]).astype(int) 
        next_state_vector[2:] = np.clip(next_state_vector[2:], 1, 5)  
        
        reward = sum(next_state_vector[2:] - current_metrics) 

        next_years_of_experience_in_months = month + 1

        monthly_data = [job_title_numeric_value, month, 
                            *current_metrics, training_program, 
                            reward, next_years_of_experience_in_months, 
                            *next_state_vector[2:]
                        ]
        all_employee_data.append(monthly_data)

        current_metrics = next_state_vector[2:]

columns = [
    "Job Title Numeric", "Years of Experience in Months", "P_Score", "C_Rating",
    "Training Program", "Reward", "Next Years of Experience in Months", "Next P_Score", "Next C_Rating"
]

all_data_df = pd.DataFrame(all_employee_data, columns=columns)
all_data_df.to_csv('large.txt', index=False, sep='\t')
