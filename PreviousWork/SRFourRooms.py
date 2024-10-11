import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA
from gridworld import SimpleGrid
from tqdm import tqdm

cmap = plt.cm.viridis
cmap.set_bad(color='white')

grid_size = 7
pattern = "four_rooms"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0, 0], goal_pos=[0, grid_size - 1])

# Plot the arena
print("Four Rooms Arena: ")
plt.show()

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

# Plotting the reward matrix slices for easy viewing
def plot_goal_matrices(goals, env):
    state_size = goals.shape[0]
    grid_size = 7  # Since we know it's a 7x7 grid

    plt.figure(figsize=(20, 20))
    
    for slice_index in range(state_size):
        ax = plt.subplot(grid_size, grid_size, slice_index + 1)
        goal_matrix = goals[slice_index]
        ax.imshow(utils.mask_grid(goal_matrix, env.blocks), cmap='viridis')
        ax.set_title(f'Slice: {slice_index}')
        ax.axis('on')

    plt.tight_layout()
    plt.show()


# Use the experiences to show where the agent was the most
def print_occupancy(experiences, env):
    occupancy_grid = np.zeros([env.grid_size, env.grid_size])
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
            ax.imshow(utils.mask_grid(M[action, i, :, :], env.blocks), cmap='viridis')
    plt.tight_layout()
    plt.show()

# plotting the raw SR matrix
def plot_raw_sr(sr, env, experiment_name):
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
    plt.title(experiment_name)
    plt.xlabel('State Index')
    plt.ylabel('State Index')
    
    # Add grid lines to separate rooms
    for i in range(1, env.grid_size):
        plt.axhline(y=i * env.grid_size - 0.5, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=i * env.grid_size - 0.5, color='k', linestyle='-', linewidth=0.5)
    
    plt.show()