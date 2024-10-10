import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'sarsa_wvf_grid_scores.csv' 
data = pd.read_csv(csv_file)

goal_size = data['Goal Size']
sarsa_scores = data['SARSA Grid Score']
wvf_scores = data['WVF Grid Score']

plt.figure(figsize=(10, 6))
plt.plot(goal_size, sarsa_scores, label='SARSA Grid Score', marker='o')
plt.plot(goal_size, wvf_scores, label='WVF Grid Score', marker='s')
plt.title('Comparison of SARSA and WVF Grid Scores vs Goal Size')
plt.xlabel('Goal Size')
plt.ylabel('Grid Score')
plt.legend()
plt.grid(True)
plt.savefig("Sarsa Vs. WVF Grid Score.png")
