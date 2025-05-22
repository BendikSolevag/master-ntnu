
from scipy import stats
from environment import SalmonFarmEnv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
  env = SalmonFarmEnv(infinite=False)
  timesteps = 0
  total_reward = 0

  growth_hist = []
  lice_hist = []
  price_hist = []
  num_hist = []
  reward_hist = []

  while True:

    action = 0
    if timesteps == 40:
      action = 2
    if timesteps == 80:
      action = 1

    reward, done = env.step(action)

    growth_hist.append(env.GROWTH_OPEN + env.GROWTH_CLOSED)
    lice_hist.append(env.LICE)
    price_hist.append(env.PRICE)
    num_hist.append(env.NUMBER_CLOSED + env.NUMBER_OPEN)
    reward_hist.append(reward)


    if env.DONE == 1:
      break
    timesteps += 1

  

  fig, ax = plt.subplots(5, 1)
  fig.set_size_inches(12, 10)
  ax[0].plot(growth_hist)
  ax[0].title.set_text("Growth")

  ax[1].plot(lice_hist)
  ax[1].title.set_text("Lice")

  ax[2].plot(price_hist)
  ax[2].title.set_text("Price")

  ax[3].plot(num_hist)
  ax[3].title.set_text("Number")

  ax[4].plot(reward_hist)
  ax[4].title.set_text(f"Reward sum {sum(reward_hist)}")

  #plt.show()
  #plt.savefig('./envdynamics.png', format="png")
  plt.close()


  main_total_rewards = []
  for htstep in tqdm(range(130)):
    total_rewards = []
    for ep in range(50):
      env = SalmonFarmEnv(infinite=False)
      timesteps = 0
      total_reward = 0
      while True:

        action = 0
        if timesteps == 20:
          action = 2
        if timesteps == htstep:
          action = 1

        reward, done = env.step(action)
        total_reward += reward

        if env.DONE:
          break
        timesteps += 1
      total_rewards.append(total_reward / 1e7)
    main_total_rewards.append(total_rewards)





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