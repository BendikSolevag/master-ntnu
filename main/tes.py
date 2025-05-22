
from scipy import stats
from environment import SalmonFarmEnv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():


  
  for htstep in tqdm(range(130)):
    for mtstep in tqdm(range(htstep)):
      for ep in range(50):
        env = SalmonFarmEnv(infinite=False)
        timesteps = 0
        total_reward = 0
        while True:
          action = 0
          if timesteps == mtstep:
            action = 2
          if timesteps == htstep:
            action = 1

          reward, done = env.step(action)
          total_reward += reward
          timesteps += 1

          if env.DONE:
            break
          
        





  means = [np.mean(y) for y in main_total_rewards]
  cis = [stats.t.interval(0.95, len(y)-1, loc=np.mean(y), scale=stats.sem(y)) for y in main_total_rewards]
  ci_lower = [ci[0] for ci in cis]
  ci_upper = [ci[1] for ci in cis]


  x_values = np.arange(len(main_total_rewards))

  plt.figure(figsize=(8, 6))
  sns.lineplot(x=x_values, y=means, label='Mean', color='blue')
  plt.fill_between(x_values, ci_lower, ci_upper, color='blue', alpha=0.2)


  plt.xlabel('Harvest timestep')
  
  plt.ylabel('Reward')
  
  plt.show()
  plt.close()

if __name__ == '__main__':
  main()