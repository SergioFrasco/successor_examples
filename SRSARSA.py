import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA
from gridworld import SimpleGrid
from tqdm import tqdm
import os
import matplotlib.animation as animation
import random
import pandas as pd
# Calculating the grid score
# from metrics import GridScorer
from neuralplayground.comparison import GridScorer
from scipy.ndimage import gaussian_filter

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

# =============================================================================================

# -------------------------- Class for SARSA based agents (Awjuliani) ------------------------
class SARSATabularSuccessorAgent(object):
    def __init__(self, state_size, action_size, learning_rate, gamma, goal_size):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        self.w = np.zeros([state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.goal_size = goal_size
        self.goals = np.zeros((state_size, grid_size, grid_size), dtype=int)
        self.generate_goal_matrices()

        #From NeuralPlayground (for grid score)
        self.state_density = 2
        self.room_width = grid_size
        self.room_depth = grid_size
        self.resolution_width = int(self.state_density * self.room_width)
        self.resolution_depth = int(self.state_density * self.room_depth)
        
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
    
    # From neural playground: Stachenfeld2018 Agent
    def get_rate_map_matrix(self, M, eigen_vector: int = 10,):
        evals, evecs = np.linalg.eig(M)
        r_out_im = evecs[:, eigen_vector].reshape((self.resolution_width, self.resolution_depth)).real
        return r_out_im

# --------------------Supporting Functions---------

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



def calculate_rate_map(experiences, env, sigma=0.5):
    occupancy_grid = np.zeros([env.grid_size, env.grid_size])
    
    for experience in experiences:
        position = env.state_to_point(experience[0])
        occupancy_grid[tuple(position)] += 1
    
    total_steps = np.sum(occupancy_grid) + 1e-10
    rate_map = occupancy_grid / total_steps
    
    # Apply Gaussian smoothing
    smoothed_rate_map = gaussian_filter(rate_map, sigma=sigma)
    
    # Mask blocked areas and unvisited locations
    masked_rate_map = utils.mask_grid(smoothed_rate_map, env.blocks)
    masked_rate_map[occupancy_grid == 0] = np.nan
    
    return masked_rate_map # Apply masking if necessary



def run_sarsa(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon,goal_size,test_episodes):

    # ---------------------------Intermediate Setup --------------------------------
    # Initialize the SARSA agent
    SARSAagent = SARSATabularSuccessorAgent(env.state_size, env.action_size, lr, gamma, goal_size)

     # Filter out slices without goals
    goals_with_targets = [slice_index for slice_index in range(SARSAagent.goals.shape[0]) if np.any(SARSAagent.goals[slice_index])]

    SARSA_experiences = []
    SARSA_test_experiences = []
    SARSA_test_lengths = []
    SARSA_lifetime_td_errors = []

    # For Grid score
    # SARSA_rate_map = np.zeros([env.grid_size, env.grid_size])

    # Shuffle the order of goals with targets
    np.random.shuffle(goals_with_targets)


    for episode in range(episodes):
        goal_index = goals_with_targets[episode % len(goals_with_targets)]
        # Train phase
        # agent_start = [0,0]
        agent_start = random_valid_position(env)
        goal_pos = env.state_to_point(np.where(SARSAagent.goals[goal_index] == 1)[0][0])

        env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = env.observation
        episodic_error = []
        for j in range(train_episode_length):
            action = SARSAagent.sample_action(state, epsilon=initial_train_epsilon)
            reward = env.step(action)
            state_next = env.observation
            done = env.done
            SARSA_experiences.append([state, action, state_next, reward, done])

            # SARSA_rate_map += env.state_to_grid(state)

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

    # Testing phase
    for episode in range(test_episodes):
        agent_start = random_valid_position(env)
        goal_pos = random_valid_position(env)  # Or choose a specific test goal strategy
        env.reset(agent_pos=agent_start, goal_pos=goal_pos)
        state = env.observation
        
        for step in range(test_episode_length):
            action = SARSAagent.sample_action(state, epsilon=test_epsilon)
            reward = env.step(action)
            next_state = env.observation
            SARSA_test_experiences.append([state, action, next_state, reward])
            state = next_state
            if env.done:
                break

    # Eigen Vector 10
    r_out_im=SARSAagent.get_rate_map_matrix(SARSAagent.M, eigen_vector=10)

    GridScorer_SARSAagent = GridScorer(SARSAagent.resolution_width)
    GridScorer_SARSAagent.plot_grid_score(r_out_im=r_out_im, plot= True)
    score = GridScorer_SARSAagent.get_scores(r_out_im)
    grid_score_10 = score[1]['gridscore']

    # Eigen Vector 20
    r_out_im=SARSAagent.get_rate_map_matrix(SARSAagent.M, eigen_vector=20)

    GridScorer_SARSAagent = GridScorer(SARSAagent.resolution_width)
    GridScorer_SARSAagent.plot_grid_score(r_out_im=r_out_im, plot= True)
    score = GridScorer_SARSAagent.get_scores(r_out_im)
    grid_score_20 = score[1]['gridscore']

    # Eigen Vector 30
    r_out_im=SARSAagent.get_rate_map_matrix(SARSAagent.M, eigen_vector=30)

    GridScorer_SARSAagent = GridScorer(SARSAagent.resolution_width)
    GridScorer_SARSAagent.plot_grid_score(r_out_im=r_out_im, plot= True)
    score = GridScorer_SARSAagent.get_scores(r_out_im)
    grid_score_30 = score[1]['gridscore']

    # Eigen Vector 40
    r_out_im=SARSAagent.get_rate_map_matrix(SARSAagent.M, eigen_vector=40)

    GridScorer_SARSAagent = GridScorer(SARSAagent.resolution_width)
    GridScorer_SARSAagent.plot_grid_score(r_out_im=r_out_im, plot= True)
    score = GridScorer_SARSAagent.get_scores(r_out_im)
    grid_score_40 = score[1]['gridscore']

    # Eigen Vector 50
    r_out_im=SARSAagent.get_rate_map_matrix(SARSAagent.M, eigen_vector=50)

    GridScorer_SARSAagent = GridScorer(SARSAagent.resolution_width)
    GridScorer_SARSAagent.plot_grid_score(r_out_im=r_out_im, plot= True)
    score = GridScorer_SARSAagent.get_scores(r_out_im)
    grid_score_50 = score[1]['gridscore']

    return [grid_score_10,grid_score_20,grid_score_30,grid_score_40,grid_score_50]

    # Calculate grid score based on test experiences
    # test_rate_map = calculate_rate_map(SARSA_test_experiences, env)
    # grid_scorer = GridScorer(grid_size)
    # _, stGrd = grid_scorer.get_scores(test_rate_map)
    # grid_score = stGrd['gridscore']

    # return float(score)

    # nbins = grid_size 
    # SARSA_rate_map = calculate_rate_map(SARSA_experiences, env) 
    # grid_scorer = GridScorer(nbins)

    # # Get the grid score from the rate map
    # _, stGrd = grid_scorer.get_scores(SARSA_rate_map)
    # grid_score = stGrd['gridscore']

    # ---- current way to get grid score
     # value for number of bins
    # scorer = GridScorer(nbins)

    # # Get grid scores and spatial autocorrelation (SAC)
    # sac, grid_props  = scorer.get_scores(SARSA_rate_map)

    # score = scorer.plot_grid_score(sac)
    # grid_score = str(np.around(score[1]["gridscore"], decimals=4, out=None))
    # ----
    # plt.show()
    # # plt.savefig('plots/SARSA Grid Score.png')
    
    # # # Grid-score
    # # # grid_score = grid_props['gridscore']
    # print("SARSA Grid Score: ", grid_score)

    # # # SAC
    # # scorer.plot_sac(sac, title="SARSA Spatial Autocorrelogram", score="Grid Score: {}".format(sac))
    # # # plt.show()
    # # plt.savefig('plots/SARSA Spatial Auto Correlation.png')

    # # # After SARSA policy training
    # # plot_grid_cells(SARSAagent, env, "SARSA Grid Cells", num_grid_cells = 16)
    # # plot_value_functions(SARSAagent, env, "SARSA Value Functions")
    # # plot_raw_sr(SARSAagent.M, env, "SARSA SR Matrix")

    # return float(grid_score)



# The Main experiment that compares the grid score from traditional SARSA against the new WVF Method
def experiment_sarsa(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon, num_runs,test_episodes):
    
    # number of exepriments = goal slices size
    # The list that containt the number of goal sizes
    # 8, 16, 24, 32 , 40, 48, 54, 62
    goal_sizes = [4, 14, 24, 44, 64]   # Example goal sizes (can be changed) 1, 10, 20, 30 , 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100

    # Initialize empty lists to store results
    results = []

    # Run SARSA for each goal size and average over multiple runs
    for goal_size in goal_sizes:
        print("\nSARSA Experiment for goal size:", goal_size)
        
        total_scores = [0.0] * 5  # testing for 5 different eigen vectore

        # Run the SARSA experiment num_runs times
        for _ in range(num_runs):
            sarsa_grid_scores = run_sarsa(train_episode_length, test_episode_length, episodes, gamma, lr, initial_train_epsilon, epsilon_decay, test_epsilon, goal_size,test_episodes)
            # Check if the score is NaN, and set it to 0 if it is
            # Replace NaN values with -2.0
            current_grid_scores = np.where(np.isnan(sarsa_grid_scores), -2.0, sarsa_grid_scores)

            total_scores += current_grid_scores  # Accumulate the score

        # Calculate the average score for the current goal size
        average_scores = total_scores / num_runs
        
        # Append only the goal size and average SARSA score to the results. return the max eigenvector value as its most grid-like
        results.append([goal_size, max(average_scores)])

    # Store the results in a DataFrame
    combined_df = pd.DataFrame(results, columns=['Goal Size', 'Average SARSA Grid Score'])

    # Save to a CSV file
    combined_df.to_csv('sarsa_grid_scores.csv', index=False)

    print("SARSA Results saved to CSV.")


# --------------------Environment setup --------------------
cmap = plt.cm.viridis
cmap.set_bad(color='white')

grid_size = 8

pattern = "empty"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0, 0], goal_pos=[0, grid_size - 1])


# --------------------Training and Testing Parameters for Q-learning agents and SARSA agents --------------------------------
# parameters for training
num_runs = 10

# number of steps agent takes in envirnoment
train_episode_length = 400
test_episode_length = 200

# number of episodes per experiment
episodes = 5000
test_episodes = 500

# parameters for agent
# gamma = 0.8
gamma = 0.9
lr = 0.1
# lr = 0.1 grid cells
# lr = 1 gerauds
# initial_train_epsilon = 0.6
initial_train_epsilon = 1
epsilon_decay = 0.9995
test_epsilon = 0.01

experiment_sarsa(train_episode_length,test_episode_length,episodes,gamma,lr,initial_train_epsilon,epsilon_decay,test_epsilon,num_runs,test_episodes)


