import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA
from gridworld import SimpleGrid

class SRWVFAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        self.w = np.zeros([state_size, state_size])  # One weight vector per goal state
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def Q_estimates(self, state, goal):
        return np.matmul(self.M[:,state,:], self.w[:, goal])
    
    def sample_action(self, state, goal, epsilon=0.0):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            action = np.argmax(Qs)
        return action
    
    def update_w(self, state, goal, reward):
        error = reward - self.w[state, goal]
        self.w[state, goal] += self.learning_rate * error
        return error
    
    def update_sr(self, current_exp, next_exp):
        s, s_a, s_1, r, d = current_exp
        s_a_1 = next_exp[1] if next_exp else None
        
        I = np.eye(self.state_size)[s]
        if d:
            td_error = (I + self.gamma * np.eye(self.state_size)[s_1] - self.M[s_a, s, :])
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
        
        self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error

def train_sr_wvf(env, agent, num_episodes, epsilon, max_steps):
    for episode in range(num_episodes):
        env.reset()
        state = env.observation
        goal = np.random.randint(env.state_size)
        goal_pos = env.state_to_point(goal)
        env.goal_pos = goal_pos  # Set the goal for this episode
        
        action = agent.sample_action(state, goal, epsilon)  # Choose first action
        
        total_reward = 0
        for step in range(max_steps):
            reward = env.step(action)
            next_state = env.observation
            done = env.done
            next_action = agent.sample_action(next_state, goal, epsilon)  # Choose next action
            
            current_exp = [state, action, next_state, reward, done]
            next_exp = [next_state, next_action, None, None, None]
            
            agent.update_w(next_state, goal, reward)
            agent.update_sr(current_exp, next_exp)
            
            total_reward += reward
            
            if done:
                break
            
            state, action = next_state, next_action
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {step+1}")

def visualize_sr(agent, env):
    plt.figure(figsize=(10, 10))
    averaged_M = np.mean(agent.M, axis=0)
    plt.imshow(averaged_M, cmap='viridis')
    plt.colorbar()
    plt.title("Averaged Successor Representation")
    plt.xlabel("Successor State")
    plt.ylabel("Current State")
    plt.show()

def visualize_wvf(agent, env, goal_state):
    plt.figure(figsize=(10, 10))
    V = np.max(np.array([agent.Q_estimates(s, goal_state) for s in range(env.state_size)]), axis=1)
    V = V.reshape(env.grid_size, env.grid_size)
    plt.imshow(V, cmap='viridis')
    plt.colorbar()
    plt.title(f"World Value Function for Goal State {goal_state}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

env = SimpleGrid(size=11, block_pattern="four_rooms", obs_mode="index")
agent = SRWVFAgent(env.state_size, env.action_size, learning_rate=0.1, gamma=0.99)
train_sr_wvf(env, agent, num_episodes=10000, epsilon=0.1, max_steps=100)

# After training
visualize_sr(agent, env)
visualize_wvf(agent, env, goal_state=env.state_size // 2)  # Visualize WVF for a middle state