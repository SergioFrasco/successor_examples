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

# ------------------ Recording Functions --------------------------------
def record_agent_trajectories(env, agent, episodes, episode_length, epsilon, filename):
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    trajectories = []

    for episode in range(episodes):
        agent_start = random_valid_position(env)
        goal_pos = random_valid_position(env)
        env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = env.observation
        trajectory = [agent_start]

        for _ in range(episode_length):
            action = agent.sample_action(state, epsilon=epsilon)
            reward = env.step(action)
            next_state = env.observation
            trajectory.append(env.state_to_point(next_state))
            state = next_state
            if env.done:
                break

        trajectories.append(trajectory)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_title(f"Episode {frame + 1}")

        # Draw blocks
        for block in env.blocks:
            ax.add_patch(plt.Rectangle((block[1], block[0]), 1, 1, fill=True, color='gray'))

        # Draw trajectory
        trajectory = trajectories[frame]
        ax.plot([p[1] + 0.5 for p in trajectory], [p[0] + 0.5 for p in trajectory], 'b-')
        ax.plot(trajectory[0][1] + 0.5, trajectory[0][0] + 0.5, 'go', markersize=10)  # Start
        ax.plot(trajectory[-1][1] + 0.5, trajectory[-1][0] + 0.5, 'ro', markersize=10)  # End

        # Draw grid
        for i in range(env.grid_size + 1):
            ax.axhline(y=i, color='k', linestyle=':')
            ax.axvline(x=i, color='k', linestyle=':')

    anim = animation.FuncAnimation(fig, update, frames=episodes, interval=500, repeat=False)
    anim.save(filename, writer='pillow', fps=2)
    plt.close(fig)


# ------------plotting functions-----------

# Function to plot grid cells from the SR matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def plot_grid_cells(agent, env, title, num_grid_cells=16):
    # Reshape the SR matrix to be state_size x state_size for eigen decomposition
    sr_matrix = np.mean(agent.M, axis=0)  # Averaging over actions
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sr_matrix)
    
    # Sort the eigenvectors by the corresponding eigenvalues in descending order
    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate the grid layout
    grid_size = math.ceil(math.sqrt(num_grid_cells))
    
    # Plot the specified number of principal eigenvectors (grid cells)
    plt.figure(figsize=(4 * grid_size, 4 * grid_size))
    
    for i in range(num_grid_cells):
        plt.subplot(grid_size, grid_size, i + 1)
        grid_cell = np.reshape(eigenvectors[:, i], (env.grid_size, env.grid_size))
        plt.imshow(grid_cell, cmap='viridis')
        plt.title(f'Eigenvector {i + 1}')
        plt.colorbar()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Create a 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Save the plot as a PNG file
    plt.savefig(f'plots/{title}.png')
    
    # Close the plot to free up memory
    plt.close()

def plot_value_functions(agent, env, title, trained_goals=None):
    grid_size = env.grid_size
    state_size = env.state_size
    
    # If trained_goals is not provided, assume all goals were trained
    if trained_goals is None:
        trained_goals = range(state_size)
    
    # Compute value functions for trained goals
    value_functions = np.zeros((state_size, grid_size, grid_size))
    for goal in trained_goals:
        goal_reward = utils.onehot(goal, state_size)
        value_functions[goal] = np.max(np.matmul(agent.M, goal_reward), axis=0).reshape(grid_size, grid_size)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)
    
    # Plot value function for each goal state
    for y in range(grid_size):
        for x in range(grid_size):
            ax = axes[y, x]
            
            if [x, y] in env.blocks:
                ax.imshow(np.zeros((grid_size, grid_size)), cmap='viridis')
                ax.set_title(f'Goal: ({x},{y}) - Blocked')
            else:
                # Convert (x, y) to state index
                goal_state = y * grid_size + x
                if goal_state in trained_goals:
                    ax.imshow(utils.mask_grid(value_functions[goal_state], env.blocks), cmap='viridis')
                    ax.set_title(f'Goal: ({x},{y})')
                else:
                    ax.imshow(np.zeros((grid_size, grid_size)), cmap='viridis')
                    ax.set_title(f'Goal: ({x},{y}) - Untrained')
            
            ax.axis('off')
    
    plt.tight_layout()

    # Create a 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Save the plot as a PNG file
    plt.savefig(f'plots/{title}.png')
    
    # Close the plot to free up memory
    plt.close()

# display the WVF learned
def plot_wvf(agent, env):
    wvf = agent.compute_wvf()
    plt.figure(figsize=(10, 10))
    for goal in range(env.state_size):
        if env.state_to_point(goal) not in env.blocks:
            ax = plt.subplot(env.grid_size, env.grid_size, goal + 1)
            value_map = np.reshape(wvf[:, goal], (env.grid_size, env.grid_size))
            ax.imshow(utils.mask_grid(value_map, env.blocks), cmap='viridis')
            ax.set_title(f'Goal: {goal}')
    plt.tight_layout()
    # plt.show()

# Plotting the reward matrix slices for easy viewing
def plot_goal_matrices(goals, env):
    state_size = goals.shape[0]

    plt.figure(figsize=(30, 30))
    
    for slice_index in range(state_size):
        ax = plt.subplot(grid_size, grid_size, slice_index + 1)
        goal_matrix = goals[slice_index]
        ax.imshow(utils.mask_grid(goal_matrix, env.blocks), cmap='viridis')
        ax.set_title(f'Slice: {slice_index}')
        ax.axis('on')

    plt.tight_layout()
    # plt.show()


# Use the experiences to show where the agent was the most
def print_occupancy(experiences, env):
    occupancy_grid = np.zeros([env.grid_size, env.grid_size])
    for experience in experiences:
        occupancy_grid += env.state_to_grid(experience[0])
    occupancy_grid = np.sqrt(occupancy_grid)
    occupancy_grid = utils.mask_grid(occupancy_grid, env.blocks)
    plt.imshow(occupancy_grid, cmap='viridis')
    plt.colorbar()
    # plt.show()

def plot_srs(action, M, env):
    M = np.reshape(M, [env.action_size, env.state_size, env.grid_size, env.grid_size])
    M = np.sqrt(M)
    plt.figure(figsize=(env.grid_size*3, env.grid_size*3))
    for i in range(env.state_size):
        if env.state_to_point(i) not in env.blocks:
            ax = plt.subplot(env.grid_size, env.grid_size, i + 1)
            ax.imshow(utils.mask_grid(M[action, i, :, :], env.blocks), cmap='viridis')
    plt.tight_layout()
    # plt.show()

# plotting the raw SR matrix
def plot_raw_sr(sr, env, experiment_name):
    averaged_M = np.mean(sr, axis=0)
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(averaged_M, cmap='viridis')
    plt.colorbar(im, label='SR Value')
    plt.title(experiment_name)
    plt.xlabel('State Index')
    plt.ylabel('State Index')
    
    # Create a 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Save the plot as a PNG file
    plt.savefig(f'plots/{experiment_name}.png')
    
    # Close the plot to free up memory
    plt.close()

# --------------------------Class for Q-learning based Agents-----------------------
class TabularSuccessorAgent(object):
    def __init__(self, state_size, action_size, learning_rate, gamma, goal_size):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.zeros((action_size, state_size, state_size))
        self.w = np.zeros((goal_size, state_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.goal_size = goal_size
        self.goals = np.zeros((state_size, grid_size, grid_size), dtype=int)
        self.generate_goal_matrices()
    
    def Q_estimates(self, state, goal=None):
        if goal is None:
            goal = np.argmax(np.sum(self.w, axis=1))
        goal_reward = self.goals[goal].flatten()
        return np.matmul(self.M[:, state, :], self.w[goal])
    
    # epsilon greedy policy for action selection
    def sample_action(self, state, goal=None, epsilon=0.0):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            return np.argmax(Qs)
    

    def generate_goal_matrices(self):
        # Initialize the goal matrix with zeros
        self.goals = np.zeros((self.goal_size, grid_size, grid_size), dtype=int)
        
        # Get all available positions (excluding blocks)
        available_positions = [(x, y) for x in range(grid_size) for y in range(grid_size) if (x, y) not in env.blocks]
        
        # Check if we have enough available positions to place the goals
        if self.goal_size > len(available_positions):
            raise ValueError("Not enough available positions to place all goals.")
        
        # Shuffle the available positions to ensure random placement
        random.shuffle(available_positions)
        
        # Assign a unique random position to each slice
        for slice_index in range(self.goal_size):
            x, y = available_positions[slice_index]
            self.goals[slice_index, x, y] = 1
            # print(f"Slice {slice_index}: Position ({x}, {y}) set as goal")
        
        return self.goals
    
    def update_w(self, current_exp):
        s_1 = current_exp[2]  # next state
        r = current_exp[3]  # reward received
        
        for goal in range(self.goal_size):
            goal_reward = self.goals[goal].flatten()  # Use the goal matrix instead of onehot
            predicted_reward = np.dot(self.w[goal], goal_reward)
            error = r - predicted_reward
            self.w[goal] += self.learning_rate * error * goal_reward
        
        return error # might want to return an average error instead, ask supervisors
    
    # The core learning method that updates the SR using Q-learning.
    def update_sr(self, current_exp):
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        d = current_exp[4]
        I = utils.onehot(s, self.state_size)
        if d:            
            td_error = I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
        else:
            max_next_sr = np.max(self.M[:, s_1, :], axis=0)
            td_error = I + self.gamma * max_next_sr - self.M[s_a, s, :]
        self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error

    
    def compute_wvf(self):
        wvf = np.zeros((self.state_size, self.state_size)) 
        for goal in range(self.state_size):
            goal_reward = utils.onehot(goal, self.state_size)

            # This is the combination between the RS (M) and the rewards if they were in every state to create the WVF
            wvf[:, goal] = np.max(np.matmul(self.M, goal_reward), axis=0)
        return wvf


# --------------------Supporting Functions---------------------
# assists with finding positions for random agent initialization.
# to ensure uniform random distribution, first find the valid states, then randomly choose from them
def random_valid_position(env):
    # Create a list of all valid positions
    valid_positions = [
        [x, y] for x in range(env.grid_size) for y in range(env.grid_size)
        if [x, y] not in env.blocks
    ]
    
    # Randomly select a position from the list of valid positions
    return valid_positions[np.random.randint(len(valid_positions))]

# assists with choosing a random goal slice for the agent to learn off of
def get_goal_sequence(total_episodes, goal_size):
    episodes_per_goal = total_episodes // goal_size
    remaining_episodes = total_episodes % goal_size
    
    goal_sequence = []
    available_goals = list(range(goal_size))
    
    for _ in range(goal_size):
        if not available_goals:
            break
        goal = np.random.choice(available_goals)
        available_goals.remove(goal)
        goal_sequence.extend([goal] * episodes_per_goal)
    
    # Distribute remaining episodes
    for i in range(remaining_episodes):
        goal_sequence.append(np.random.choice(range(goal_size)))
    
    return goal_sequence

def calculate_rate_map(experiences, env):
    occupancy_grid = np.zeros([env.grid_size, env.grid_size])
    for experience in experiences:
        occupancy_grid[tuple(env.state_to_point(experience[0]))] += 1
    rate_map = occupancy_grid + 1e-10 / (np.sum(occupancy_grid) + 1e-10)  # Add small epsilon to avoid division by zero
    return utils.mask_grid(rate_map, env.blocks)



# --------------------Epsilon greedy Training Loop --------------------
# This loop trains the agent using a decaying epsilon greedy policy and trains it on goal slices in the arena.

def run_wvf(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon,goal_size):

    # Reinitialize the agent to ensure an independent learning process for the second loop
    epsilon_greedy_agent = TabularSuccessorAgent(env.state_size, env.action_size, lr, gamma, goal_size)

    # Filter out slices without goals
    goals_with_targets = [slice_index for slice_index in range(epsilon_greedy_agent.goals.shape[0]) if np.any(epsilon_greedy_agent.goals[slice_index])]

    experiences = []
    test_experiences = []
    test_lengths = []
    lifetime_td_errors = []

    # For grid score calculation
    WVF_rate_map = np.zeros([env.grid_size, env.grid_size])

    # Shuffle the order of goals with targets
    np.random.shuffle(goals_with_targets)

    # Calculate episodes per goal
    episodes_per_goal = episodes // len(goals_with_targets)
    remaining_episodes = episodes % len(goals_with_targets)

    # # Ensure the videos directory exists
    # if not os.path.exists('videos'):
    #     os.makedirs('videos')

    epsilon = initial_train_epsilon


    # print("plotting goals")
    # plot_goal_matrices(epsilon_greedy_agent.goals, env)

    for episode in range(episodes):
        goal_index = goals_with_targets[episode % len(goals_with_targets)]

        agent_start = random_valid_position(env)
        goal_pos = env.state_to_point(np.where(epsilon_greedy_agent.goals[goal_index] == 1)[0][0])

        env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = env.observation
        episodic_error = []

        for step in range(train_episode_length):
            action = epsilon_greedy_agent.sample_action(state, goal=goal_index, epsilon=epsilon)
            reward = env.step(action)
            next_state = env.observation
            done = env.done
            experiences.append([state, action, next_state, reward])
            experience = [state, action, next_state, reward, done]

            WVF_rate_map += env.state_to_grid(state)
            
            td_sr = epsilon_greedy_agent.update_sr(experience)
            td_w = epsilon_greedy_agent.update_w(experience)  # This now updates for all goals
            episodic_error.append(np.mean(np.abs(td_sr)))
            
            state = next_state
            if done:
                break
        
        lifetime_td_errors.append(np.mean(episodic_error))

        # Decay epsilon after each episode
        epsilon *= epsilon_decay
        epsilon = max(epsilon, 0.05)  # minimum epsilon value

    WVF_rate_map = calculate_rate_map(experiences, env)

        # Test phase
        # agent_start = random_valid_position(env)  
        # env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        # state = env.observation
        # for j in range(test_episode_length):
        #     action = epsilon_greedy_agent.sample_action(state, epsilon=test_epsilon)
        #     reward = env.step(action)
        #     state_next = env.observation
        #     test_experiences.append([state, action, state_next, reward])
        #     state = state_next
        #     if env.done:
        #         break
        # test_lengths.append(j)

    #     # Print progress every 50 episodes
    #     if (episode + 1) % 50 == 0:
    #         print(f"WVF training: Completed episode {episode + 1}")

    # print("\nWVF training completed.")
    
    nbins = grid_size  # value for number of bins
    wvf_scorer = GridScorer(nbins)

    # Get grid scores and spatial autocorrelation (SAC)
    sac, grid_props  = wvf_scorer.get_scores(WVF_rate_map)

    score = wvf_scorer.plot_grid_score(sac)
    grid_score = str(np.around(score[1]["gridscore"], decimals=4, out=None))
    # plt.savefig('plots/WVF Grid Score.png')
    # # SAC
    # wvf_scorer.plot_sac(sac, title="WVF Spatial Autocorrelogram", score="Grid Score: {}".format(sac))
    # # plt.show()
    # plt.savefig('plots/WVF Spatial Auto Correlation.png')

    # Grid-score
    # grid_score = grid_props['gridscore']
    
    # plt.show()
    
    # print("WVF Grid Score: ", grid_score)

    # # After epsilon-greedy policy training
    # plot_grid_cells(epsilon_greedy_agent, env, "WVF Grid Cells", num_grid_cells = 16)
    # plot_value_functions(epsilon_greedy_agent, env, "WVF Value Functions")
    # plot_raw_sr(epsilon_greedy_agent.M, env, "WVF SR Matrix")

    return float(grid_score)

# The Main experiment that compares the grid score from traditional SARSA against the new WVF Method
def experiment_sarsa_wvf(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon, num_runs):
    
    # number of exepriments = goal slices size
    # The list that containt the number of goal sizes
    goal_sizes = [1, 10, 20, 30 , 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # Example goal sizes (can be changed) 1, 10, 20, 30 , 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100

    # Initialize empty lists to store results
    results = []

    # # Run SARSA experiments
    # for run in range(num_runs):
    #     sarsa_grid_scores = run_sarsa(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon, goal_size=goal_sizes[0])
    #     sarsa_results.append(sarsa_grid_scores)
    # run SARSA with decreasing goal sizes (no effect)


    # run SARSA and WVF with decreasing goal sizes and store results together
    for goal_size in goal_sizes:
        print("\nWVF Experiment for goal size:", goal_size)
        total_score = 0.0  # Initialize a score accumulator

        # Run the SARSA experiment num_runs times
        for _ in range(num_runs):
            wvf_grid_score = run_wvf(train_episode_length, test_episode_length, episodes, gamma, lr, initial_train_epsilon, epsilon_decay, test_epsilon, goal_size)
            total_score += wvf_grid_score  # Accumulate the score

        # Calculate the average score for the current goal size
        average_score = total_score / num_runs
        
        # Append only the goal size and average SARSA score to the results
        results.append([goal_size, average_score])  # append the goal size, SARSA score, and WVF score to combined_results

    # dtore the results in a single CSV file
    combined_df = pd.DataFrame(results, columns=['Goal Size', 'WVF Grid Score'])

    # Save to a single CSV
    combined_df.to_csv('wvf_grid_scores.csv', index=False)

    print("WVF Results saved to CSV.")


# --------------------Environment setup --------------------
cmap = plt.cm.viridis
cmap.set_bad(color='white')

grid_size = 10

pattern = "empty"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0, 0], goal_pos=[0, grid_size - 1])

# --------------------Training and Testing Parameters for Q-learning agents and SARSA agents --------------------------------
# parameters for training

num_runs = 10

# number of steps agent takes in envirnoment
train_episode_length = 100
test_episode_length = 100

# number of episodes per experiment
episodes = 3000

# parameters for agent
gamma = 0.8
# gamma = 0.95
lr = 1
# lr = 0.1 grid cells
# lr = 1 gerauds
initial_train_epsilon = 0.6
# initial_train_epsilon = 1
epsilon_decay = 0.995
test_epsilon = 0.01

experiment_sarsa_wvf(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon,num_runs)
