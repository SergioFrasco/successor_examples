from gridworld import SimpleGrid
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
grid_size = 11
env = SimpleGrid(grid_size, block_pattern="four_rooms", obs_mode="index")

# WVF learning parameters
gamma = 1
maxiter = 5000
alpha = 1
epsilon = 0.1

def Goal_Oriented_Q_learning(env, epsilon, alpha, maxiter):
    # Initialize WVF
    WVF = {}
    for s in range(env.state_size):
        WVF[s] = {}
        for g in range(env.state_size):
            WVF[s][g] = np.zeros(env.action_size)
    
    stats = {"R": []}

    for i in range(maxiter):
        env.reset()
        s = env.observation
        g = np.random.randint(env.state_size)
        goal_pos = env.state_to_point(g)
        env.goal_pos = goal_pos  # Set the goal for this episode
        
        done = False
        episode_reward = 0
        steps = 0
        max_steps = 1000  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                a = np.random.randint(env.action_size)
            else:
                a = np.argmax(WVF[s][g])
            
            # Take action and observe next state and reward
            r = env.step(a)
            s_next = env.observation
            done = env.done
            
            # Q-learning update
            if done:
                WVF[s][g][a] += alpha * (r - WVF[s][g][a])
            else:
                WVF[s][g][a] += alpha * (r + gamma * np.max(WVF[s_next][g]) - WVF[s][g][a])
            
            s = s_next
            episode_reward += r
            steps += 1
        
        stats["R"].append(episode_reward)
        
        if i % 100 == 0:
            print(f"Episode {i}, Reward: {episode_reward}, Steps: {steps}")
    
    return WVF, stats

# Run WVF learning
WVF, stats = Goal_Oriented_Q_learning(env, epsilon, alpha, maxiter)

# Plot learning curve
plt.plot(stats["R"])
plt.title("WVF Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Visualize WVF
def plot_wvf(WVF, env):
    V = np.zeros((env.state_size, env.state_size))
    for s in range(env.state_size):
        for g in range(env.state_size):
            V[s, g] = np.max(WVF[s][g])
    
    plt.imshow(V, cmap='viridis')
    plt.colorbar()
    plt.title("World Value Function")
    plt.xlabel("Goal State")
    plt.ylabel("Current State")
    plt.show()

plot_wvf(WVF, env)