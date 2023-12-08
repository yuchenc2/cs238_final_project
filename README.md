# Employee Upskilling and Training Optimization using Reinforcement Learning

## Overview
This project applies Reinforcement Learning (RL) techniques to optimize decision-making processes in employee up-skilling and on-job training. Utilizing a Markov Decision Process framework, the project explores the effectiveness of various RL algorithms in predicting the most beneficial job training for employees based on their tenure.

## Algorithms
The project includes implementations of four RL algorithms:
- Q-learning
- SARSA
- Maximum Likelihood Estimate (MLE)
- Bayesian Method

## File Structure
- `bayesian_method.py`: Contains the implementation of the Bayesian approach for the RL model.
- `data_generation.py`: A script for generating synthetic datasets simulating employee performance and training impacts.
- `maximum_likelihood.py`: Implements the Maximum Likelihood Estimate method for the RL model.
- `score_calculation.py`: Used for calculating the policy scores and evaluating the performance of different RL algorithms.
- `PolicyOnlyQLearning.ipynb`: A Jupyter Notebook detailing the Q-learning algorithm implementation and its results.

## Datasets
- `small.txt`: Small-scale dataset for quick testing and development.
- `medium.txt`: Medium-scale dataset for more comprehensive testing.
- `large.txt`: Large-scale dataset for full-scale testing and performance evaluation.

## Usage
To run a specific RL algorithm, use the corresponding script. For instance, to execute the Bayesian method, run:

python bayesian_method.py

Similarly, use the data_generation.py script to generate new datasets and score_calculation.py to evaluate the performance of each algorithm.

## Requirements
Python 3.x
NumPy
Pandas
Matplotlib (for visualizations in Jupyter Notebook)

## Installation
Clone the repository and install the necessary dependencies:
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.