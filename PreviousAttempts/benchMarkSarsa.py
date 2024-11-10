import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA
from gridworld import SimpleGrid
from tqdm import tqdm
import os
import matplotlib.animation as animation
from sklearn.decomposition import PCA 
import random
import pandas as pd
# Calculating the grid score
from metrics import GridScorer

# -------------------------- Class for SARSA based agents (Awjuliani) ------------------------
class SARSATabularSuccessorAgent(object):
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        self.w = np.zeros([state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma
        
    def Q_estimates(self, state, goal=None):
        # Generate Q values for all actions.
        if goal == None:
            goal = self.w
        else:
            goal = utils.onehot(goal, self.state_size)
        return np.matmul(self.M[:,state,:],goal)
    
    def sample_action(self, state, goal=None, epsilon=0.0):
        # Samples action using epsilon-greedy approach
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            action = np.argmax(Qs)
        return action
    
    def update_w(self, current_exp):
        # A simple update rule
        s_1 = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error        
        return error
    
    def update_sr(self, current_exp, next_exp):
        # SARSA TD learning rule
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        s_a_1 = next_exp[1]
        r = current_exp[3]
        d = current_exp[4]
        I = utils.onehot(s, env.state_size)
        if d:            
            td_error = (I + self.gamma * utils.onehot(s_1, env.state_size) - self.M[s_a, s, :])
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
        self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error

# helper function to calculate rate map
# def calculate_rate_map(experiences, env):
#     occupancy_grid = np.zeros([env.grid_size, env.grid_size])
#     for experience in experiences:
#         occupancy_grid[tuple(env.state_to_point(experience[0]))] += 1
#     rate_map = occupancy_grid / (np.sum(occupancy_grid) + 1e-10)  # Add small epsilon to avoid division by zero
#     return utils.mask_grid(rate_map, env.blocks)

def calculate_rate_map(experiences, env):
    states = np.array([env.state_to_point(experience[0]) for experience in experiences])
    occupancy_grid = np.zeros([env.grid_size, env.grid_size])
    np.add.at(occupancy_grid, (states[:, 0], states[:, 1]), 1)
    rate_map = occupancy_grid / (np.sum(occupancy_grid) + 1e-10)  # Normalize
    return utils.mask_grid(rate_map, env.blocks)


def run_sarsa(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon,goal_size):

    # ---------------------------Intermediate Setup --------------------------------
    # Initialize the SARSA agent
    SARSAagent = SARSATabularSuccessorAgent(env.state_size, env.action_size, lr, gamma)

    # Initialize the new Q-learning agent and environment
    # agent = TabularSuccessorAgent(env.state_size, env.action_size, lr, gamma, goal_size)

    # Filter out slices without goals
    # goals_with_targets = [slice_index for slice_index in range(agent.goals.shape[0]) if np.any(agent.goals[slice_index])]

    # print(f"Number of slices with goals: {len(goals_with_targets)}")

    # # Calculate episodes per goal
    # episodes_per_goal = episodes // len(goals_with_targets)
    # remaining_episodes = episodes % len(goals_with_targets)

    # # Shuffle goal order
    # goal_order = np.random.permutation(goal_size)
    # # Shuffle the order of goals with targets
    # np.random.shuffle(goals_with_targets)

    #  ---------------------- SARSA Training loop (Awjuliani) ----------------------
    SARSA_experiences = []
    SARSA_test_experiences = []
    SARSA_test_lengths = []
    SARSA_lifetime_td_errors = []

    # For Grid score
    SARSA_rate_map = np.zeros([env.grid_size, env.grid_size])

    for i in range(episodes):
        # Train phase
        agent_start = [0,0]
        # agent_start = random_valid_position(env)
        if i < episodes // 2:
            goal_pos = [0, grid_size-1]
        else:
            if i == episodes // 2:
                print("\nSwitched reward locations")
            goal_pos = [grid_size-1,grid_size-1]
        env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = env.observation
        episodic_error = []
        for j in range(train_episode_length):
            action = SARSAagent.sample_action(state, epsilon=initial_train_epsilon)
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            SARSA_experiences.append([state, action, state_next, reward, done])
            state = state_next
            if (j > 1):
                td_sr = SARSAagent.update_sr(SARSA_experiences[-2], SARSA_experiences[-1])
                td_w = SARSAagent.update_w(SARSA_experiences[-1])
                episodic_error.append(np.mean(np.abs(td_sr)))
            if env.done:
                td_sr = SARSAagent.update_sr(SARSA_experiences[-1], SARSA_experiences[-1])
                episodic_error.append(np.mean(np.abs(td_sr)))
                break

        SARSA_lifetime_td_errors.append(np.mean(episodic_error))

        if i % 50 == 0:
            print('\rEpisode {}/{}, TD Error: {}, Test Lengths: {}'
                    .format(i, episodes, np.mean(SARSA_lifetime_td_errors[-50:]), 
                            np.mean(SARSA_test_lengths[-50:])), end='')
            

    SARSA_rate_map = calculate_rate_map(SARSA_experiences, env)


# The Main experiment that compares the grid score from traditional SARSA against the new WVF Method
def experiment_sarsa_wvf(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon):
    
    # number of exepriments = goal slices size
    # The list that containt the number of goal sizes
    goal_sizes = [30, 35, 40, 45, 49]  # Example goal sizes (can be changed)

    # Initialize empty lists to store results
    results = []

    # # Run SARSA experiments
    # for run in range(num_runs):
    #     sarsa_grid_scores = run_sarsa(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon, goal_size=goal_sizes[0])
    #     sarsa_results.append(sarsa_grid_scores)
    # run SARSA with decreasing goal sizes (no effect)


    # run SARSA and WVF with decreasing goal sizes and store results together
    for goal_size in goal_sizes:
        print("\nExperiment for goal size: ", goal_size)
        # SARSA will be run in the awjuliani notebook
        sarsa_grid_score = run_sarsa(train_episode_length, test_episode_length, episodes, gamma, lr, initial_train_epsilon, epsilon_decay, test_epsilon, goal_size)
        # wvf_grid_score = run_wvf(train_episode_length, test_episode_length, episodes, gamma, lr, initial_train_epsilon, epsilon_decay, test_epsilon, goal_size)
        
        # append the goal size, SARSA score, and WVF score to combined_results
        results.append([goal_size, sarsa_grid_score])

    # dtore the results in a single CSV file
    combined_df = pd.DataFrame(results, columns=['Goal Size', 'SARSA Grid Score'])

    # Save to a single CSV
    combined_df.to_csv('sarsa_wvf_grid_scores.csv', index=False)

    print("Results saved to CSV.")

# --------------------Environment setup --------------------
cmap = plt.cm.viridis
cmap.set_bad(color='white')

grid_size = 7
pattern = "empty"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0, 0], goal_pos=[0, grid_size - 1])
# Plot the arena
# print("Four Rooms Arena: ")


# --------------------Training and Testing Parameters for Q-learning agents and SARSA agents --------------------------------
# parameters for training

# number of steps agent takes in envirnoment
train_episode_length = 500
test_episode_length = 500

# number of episodes per experiment
episodes = 5000

# parameters for agent
# 0.8
gamma = 0.95
# 0.01
lr = 5e-2
# 0.6
initial_train_epsilon = 1
epsilon_decay = 0.995
test_epsilon = 0.01

experiment_sarsa_wvf(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon)
