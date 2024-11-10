import numpy as np
import matplotlib.pyplot as plt
from gridworld import SimpleGrid
from metrics import GridScorer
import pandas as pd

# Shared utility functions
def random_valid_position(env):
    valid_positions = [
        [x, y] for x in range(env.grid_size) for y in range(env.grid_size)
        if [x, y] not in env.blocks
    ]
    return valid_positions[np.random.randint(len(valid_positions))]

def calculate_rate_map(experiences, env):
    occupancy_grid = np.zeros([env.grid_size, env.grid_size])
    for experience in experiences:
        occupancy_grid[tuple(env.state_to_point(experience[0]))] += 1
    rate_map = occupancy_grid / (np.sum(occupancy_grid) + 1e-10)
    return np.ma.masked_array(rate_map, mask=env.blocks)

# SARSA Agent
class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.stack([np.identity(state_size) for _ in range(action_size)])
        self.w = np.zeros([state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def Q_estimates(self, state):
        return np.matmul(self.M[:, state, :], self.w)
    
    def sample_action(self, state, epsilon=0.0):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.Q_estimates(state))
    
    def update(self, current_exp, next_exp):
        s, a, s_1, r, d = current_exp
        _, a_1, _, _, _ = next_exp
        
        # Update w
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error
        
        # Update SR
        I = np.eye(self.state_size)[s]
        if d:
            td_error = I + self.gamma * np.eye(self.state_size)[s_1] - self.M[a, s, :]
        else:
            td_error = I + self.gamma * self.M[a_1, s_1, :] - self.M[a, s, :]
        self.M[a, s, :] += self.learning_rate * td_error
        
        return np.mean(np.abs(td_error))

# WVF Agent
class WVFAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, goal_size):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.zeros((action_size, state_size, state_size))
        self.w = np.zeros((goal_size, state_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.goal_size = goal_size
        self.goals = np.zeros((goal_size, state_size))
        self.generate_goal_matrices()
    
    def generate_goal_matrices(self):
        available_states = list(range(self.state_size))
        np.random.shuffle(available_states)
        for i in range(self.goal_size):
            self.goals[i, available_states[i]] = 1
    
    def Q_estimates(self, state, goal):
        return np.matmul(self.M[:, state, :], self.w[goal])
    
    def sample_action(self, state, goal, epsilon=0.0):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.Q_estimates(state, goal))
    
    def update(self, experience, goal):
        s, a, s_1, r, d = experience
        
        # Update w
        for g in range(self.goal_size):
            error = r - np.dot(self.w[g], self.goals[g])
            self.w[g] += self.learning_rate * error * self.goals[g]
        
        # Update SR
        I = np.eye(self.state_size)[s]
        if d:
            td_error = I + self.gamma * np.eye(self.state_size)[s_1] - self.M[a, s, :]
        else:
            max_next_sr = np.max(self.M[:, s_1, :], axis=0)
            td_error = I + self.gamma * max_next_sr - self.M[a, s, :]
        self.M[a, s, :] += self.learning_rate * td_error
        
        return np.mean(np.abs(td_error))

# Training function
def train_agent(agent, env, episodes, episode_length, epsilon, goal_size=None):
    experiences = []
    for episode in range(episodes):
        agent_start = random_valid_position(env)
        goal_pos = random_valid_position(env)
        env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = env.observation
        goal = np.random.randint(goal_size) if goal_size else None
        
        for _ in range(episode_length):
            action = agent.sample_action(state, goal, epsilon) if goal_size else agent.sample_action(state, epsilon)
            reward = env.step(action)
            next_state = env.observation
            done = env.done
            experience = [state, action, next_state, reward, done]
            experiences.append(experience)
            
            if isinstance(agent, SARSAAgent):
                next_action = agent.sample_action(next_state, epsilon)
                next_experience = [next_state, next_action, None, None, None]
                agent.update(experience, next_experience)
            else:
                agent.update(experience, goal)
            
            state = next_state
            if done:
                break
    
    return calculate_rate_map(experiences, env)

# Main experiment function
def run_experiment(env, episodes, episode_length, learning_rate, gamma, epsilon, goal_sizes, num_runs):
    results = []
    
    for goal_size in goal_sizes:
        print(f"\nExperiment for goal size: {goal_size}")
        sarsa_scores = []
        wvf_scores = []
        
        for _ in range(num_runs):
            # SARSA
            sarsa_agent = SARSAAgent(env.state_size, env.action_size, learning_rate, gamma)
            sarsa_rate_map = train_agent(sarsa_agent, env, episodes, episode_length, epsilon)
            
            # WVF
            wvf_agent = WVFAgent(env.state_size, env.action_size, learning_rate, gamma, goal_size)
            wvf_rate_map = train_agent(wvf_agent, env, episodes, episode_length, epsilon, goal_size)
            
            # Calculate grid scores
            scorer = GridScorer(env.grid_size)
            sarsa_sac, _ = scorer.get_scores(sarsa_rate_map)
            wvf_sac, _ = scorer.get_scores(wvf_rate_map)
            
            sarsa_scores.append(scorer.grid_score_from_sac(sarsa_sac))
            wvf_scores.append(scorer.grid_score_from_sac(wvf_sac))
        
        results.append([goal_size, np.mean(sarsa_scores), np.mean(wvf_scores)])
    
    return pd.DataFrame(results, columns=['Goal Size', 'SARSA Grid Score', 'WVF Grid Score'])

# Setup and run experiment
grid_size = 12
env = SimpleGrid(grid_size, block_pattern="empty", obs_mode="index")

episodes = 30
episode_length = 100
learning_rate = 0.1
gamma = 0.9
epsilon = 0.1
goal_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_runs = 1

results = run_experiment(env, episodes, episode_length, learning_rate, gamma, epsilon, goal_sizes, num_runs)

# Save results
results.to_csv('sarsa_wvf_comparison.csv', index=False)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(results['Goal Size'], results['SARSA Grid Score'], '-o', label='SARSA')
plt.plot(results['Goal Size'], results['WVF Grid Score'], '-s', label='WVF')
plt.xlabel('Goal Size')
plt.ylabel('Grid Score')
plt.title('Comparison of SARSA and WVF Grid Scores vs Goal Size')
plt.legend()
plt.grid(True)
plt.savefig('sarsa_wvf_comparison.png')
plt.close()

print("Experiment completed. Results saved to sarsa_wvf_comparison.csv and sarsa_wvf_comparison.png")