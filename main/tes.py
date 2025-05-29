
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch as T
from agents.q_learner import Agent


def main():

  coefficients = [1, 2, 4, 6, 8, 10, 12]

  h_means = []
  h_ci_lower = []
  h_ci_upper = []

  m_means = []
  m_ci_lower = []
  m_ci_upper = []

  for coef in coefficients:
    harvest_timesteps = T.load(f'./data/assets/simulated/{coef}/harvest_timesteps.pt')
    h_mean = np.mean(harvest_timesteps)
    h_means.append(h_mean)
    ci = stats.t.interval(0.95, len(harvest_timesteps)-1, loc=h_mean, scale=stats.sem(harvest_timesteps))
    h_ci_lower.append(ci[0])
    h_ci_upper.append(ci[1])


    move_timesteps = T.load(f'./data/assets/simulated/{coef}/move_timesteps.pt')
    m_mean = np.mean(move_timesteps)

    
    m_means.append(m_mean)
    ci = stats.t.interval(0.95, len(move_timesteps)-1, loc=m_mean, scale=stats.sem(move_timesteps))
    m_ci_lower.append(ci[0])
    m_ci_upper.append(ci[1])

    





  
  x_values = np.arange(len(h_means))


  plt.figure(figsize=(8, 6))
  sns.lineplot(x=x_values, y=h_means, label='Harvest', color='blue')
  


  sns.lineplot(x=x_values, y=m_means, label='Move', color='purple')
  


  plt.xlabel('Operational cost coefficient')

  a=[0, 1, 2, 3, 4, 5, 6]
  b=[1, 2, 4, 6, 8, 10, 12]
  plt.xticks(a,b)
  
  
  plt.ylabel('Timestep')
  
  plt.savefig(f'./illustrations/results/episodic/optimal-move-harvest-per-operation-cost.png', format="png")
  plt.close()

if __name__ == '__main__':
  main()