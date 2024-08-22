import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA
from gridworld import SimpleGrid

cmap = plt.cm.viridis
cmap.set_bad(color='white')

grid_size = 7
pattern="four_rooms"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0,0], goal_pos=[0, grid_size-1])

# Plot the arena
print("Four Rooms Arena: ")
plt.show() 

class TabularSuccessorAgent(object):
    def __init__(self, state_size, action_size, learning_rate, gamma, goal_size):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.zeros((action_size, state_size, state_size))
        self.w = np.zeros(state_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    # Computes action values by cobining the SR with the current reward prediction (w) or a specified goal
    def Q_estimates(self, state, goal=None):
        if goal is None:
            goal = self.w
        else:
            goal = utils.onehot(goal, self.state_size)
        return np.matmul(self.M[:, state, :], goal)
    
    # epsilon greedy policy for action selection
    def sample_action(self, state, goal=None, epsilon=0.0):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            return np.argmax(Qs)
    
    # generate number of goals, if no goal size specified, goal_size = state_size
    def generate_goal_matrices(self, state_size, goal_size = 49):

        if goal_size > state_size:
            print("Goal size cannot be larger than state size!")
            return
        
        if goal_size != 1:
            print("Goal size must be 1 for this implementation!")
            return
        
        if state_size ** 2 < state_size:
            print("Not enough positions to place goals uniquely in each slice!")
            return
        
        # Initialize the goals matrix to zeros
        self.goals = np.zeros((state_size, state_size, state_size), dtype=int)
        
        # Generate a list of all possible positions
        all_positions = [(x, y) for x in range(state_size) for y in range(state_size)]
        
        # Shuffle positions to randomize the goal placements
        np.random.shuffle(all_positions)
        
        # Place one goal in each slice at a unique position

        for slice_index in range(state_size):
            if slice_index  <= goal_size:
                x, y = all_positions[slice_index]
                self.goals[slice_index, x, y] = 1
            else:
                break

    # Updates the reward prediction weights based on the rewards observed
    def update_w(self, current_exp):
        s_1 = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error        
        return error
    
    #  The core learning method that updates the SR using Q-learning.
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

# --------------------Training and Testing --------------------------------
# paramaters for training
train_episode_length = 50
test_episode_length = 50
episodes = 10000
gamma = 0.95
lr = 5e-2
train_epsilon = 0.5
test_epsilon = 0.1
goal_size = 49 # Testing with a goal from every state

# Initialize the agent and environment
agent = TabularSuccessorAgent(env.state_size, env.action_size, lr, gamma, goal_size)

experiences = []
test_experiences = []
test_lengths = []
lifetime_td_errors = []

for i in range(episodes):


    # training phase
    agent_start = [0,0] #set the agent to the top left

    # switch goals half way through episodes
    if i < episodes // 2:
        goal_pos = [0, grid_size-1]
    else:
        if i == episodes // 2:
            print("\nSwitched reward locations")
        goal_pos = [grid_size-1,grid_size-1]

    env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    state = env.observation
    episodic_error = [] # keeps track of error

    for j in range(train_episode_length):
        action = agent.sample_action(state, epsilon=train_epsilon) # get an action from epsilon greedy
        reward = env.step(action) # obtain reward from performing the action
        next_state = env.observation # obtain the next state
        done = env.done # check if we're at goal state
        experiences.append([state, action, next_state, reward]) # Collect the experiences
        experience = [state, action, next_state, reward, done]
        
        td_sr = agent.update_sr(experience)
        td_w = agent.update_w(experience)
        episodic_error.append(np.mean(np.abs(td_sr)))
        
        state = next_state
        if done:
            break
    
    lifetime_td_errors.append(np.mean(episodic_error))
    
    # Test phase
    env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    state = env.observation
    for j in range(test_episode_length):
        action = agent.sample_action(state, epsilon=test_epsilon)
        reward = env.step(action)
        state_next = env.observation
        test_experiences.append([state, action, state_next, reward])
        state = state_next
        if env.done:
            break
    test_lengths.append(j)
    
    if i % 50 == 0:
        print('\rEpisode {}/{}, TD Error: {}, Test Lengths: {}\n'
              .format(i, episodes, np.mean(lifetime_td_errors[-50:]), 
                      np.mean(test_lengths[-50:])), end='')

# ------------plotting functions-----------

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
    plt.show()

# Use the experiences to show where the agent was the most
def print_occupancy(experiences, grid_size):
    occupancy_grid = np.zeros([grid_size, grid_size])
    for experience in experiences:
        occupancy_grid += env.state_to_grid(experience[0])
    occupancy_grid = np.sqrt(occupancy_grid)
    occupancy_grid = utils.mask_grid(occupancy_grid, env.blocks)
    plt.imshow(occupancy_grid, cmap='viridis')
    plt.colorbar()
    plt.show()

def plot_srs(action, M, env):
    M = np.reshape(M, [env.action_size, env.state_size, env.grid_size, env.grid_size])
    M = np.sqrt(M)
    plt.figure(figsize=(env.grid_size*3, env.grid_size*3))
    for i in range(env.state_size):
        if env.state_to_point(i) not in env.blocks:
            ax = plt.subplot(env.grid_size, env.grid_size, i + 1)
            ax.imshow(utils.mask_grid(M[action,i,:,:], env.blocks), cmap='viridis')
    plt.tight_layout()
    plt.show()

# plotting the raw SR matrix
def plot_raw_sr(sr, env):
    averaged_M = np.mean(sr, axis=0)

    # Create a mapping from state index to grid coordinates
    state_to_coord = {i: env.state_to_point(i) for i in range(env.state_size)}

    # Create wall mask
    wall_mask = np.zeros_like(averaged_M, dtype=bool)
    for state, coord in state_to_coord.items():
        if coord in env.blocks:
            wall_mask[state, :] = True
            wall_mask[:, state] = True
    
    masked_M = np.ma.array(averaged_M, mask=wall_mask)
    
    plt.figure(figsize=(10, 10))
    im = plt.imshow(masked_M, cmap='viridis')
    plt.colorbar(im, label='SR Value')
    plt.title('Raw SR Matrix (Non-Wall States)')
    plt.xlabel('State Index')
    plt.ylabel('State Index')
    
    # Add grid lines to separate rooms
    for i in range(1, env.grid_size):
        plt.axhline(y=i * env.grid_size - 0.5, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=i * env.grid_size - 0.5, color='k', linestyle='-', linewidth=0.5)
    
    plt.show()

# ---------After training---------
# This shows the agent spends a lot of time in the top left perhaps because this is where they are initialized each run
print("Training Occupancy Plot\n")
print_occupancy(experiences, grid_size)

print("Test Occupancy Plot\n")
print_occupancy(test_experiences, grid_size)


# Call the function
print("Raw SR Matrix")
plot_raw_sr(agent.M, env)
print("plot_srs\n")
plot_srs(1, agent.M, env)
print("plot_wvf\n")
plot_wvf(agent, env)


# fig = plt.figure(figsize=(10, 6))

# ax = fig.add_subplot(2, 2, 1)
# ax.plot(lifetime_td_errors)
# ax.set_title("TD Error")
# ax = fig.add_subplot(2, 2, 2)
# ax.plot(test_lengths)
# ax.set_title("Episode Lengths")



# averaged_M = np.mean(agent.M, axis=0)
# plt.show()

# averaged_M = np.reshape(averaged_M, [env.state_size, grid_size, grid_size])

# cmap = plt.cm.viridis
# cmap.set_bad(color='white')

# plt.figure(1, figsize=(grid_size*3, grid_size*3))
# for i in range(env.state_size):
#     if env.state_to_point(i) not in env.blocks:
#         ax = plt.subplot(grid_size, grid_size, i + 1)
#         ax.imshow(utils.mask_grid(averaged_M[i,:,:], env.blocks), cmap=cmap)

# M_s = np.mean(agent.M[:,:,:], axis=0)
# colors = np.zeros([env.state_size])
# for bottleneck in env.bottlenecks:
#     grid = np.zeros([env.grid_size,env.grid_size])
#     grid[bottleneck[0],bottleneck[1]] = 1
#     grid = grid.flatten()
#     b_state = np.where(grid==1)[0][0]
#     colors[b_state] = 1
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(M_s[:])

# plt.scatter(pca_result[:,0], pca_result[:,1], c=colors)

# a = np.zeros([env.state_size])
# for i in range(env.state_size):
#     Qs = agent.Q_estimates(i)
#     V = np.mean(Qs)
#     a[i] = V
# V_Map = np.reshape(a, [grid_size, grid_size])
# V_Map = np.sqrt(V_Map)

# V_Map = utils.mask_grid(V_Map, env.blocks)

# plt.show()

