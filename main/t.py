
#from pretrain_episodic import TEnv
from environment import SalmonFarmEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats


main_total_rewards = []
for mtstep in tqdm(range(69)):
  total_rewards = []
  for ep in range(200):
    env = SalmonFarmEnv(infinite=False)
    timesteps = 0
    total_reward = 0
    while True:

      action = 0
      if timesteps == mtstep:
        action = 2
      if timesteps == 70:
        action = 1

      reward, done = env.step(action)
      total_reward += reward

      if env.DONE:
        break
      timesteps += 1
    total_rewards.append(total_reward)
  main_total_rewards.append(total_rewards)





means = [np.mean(y) for y in main_total_rewards]
cis = [stats.t.interval(0.95, len(y)-1, loc=np.mean(y), scale=stats.sem(y)) for y in main_total_rewards]
ci_lower = [ci[0] for ci in cis]
ci_upper = [ci[1] for ci in cis]


x_values = np.arange(len(main_total_rewards))

plt.figure(figsize=(8, 6))
sns.lineplot(x=x_values, y=means, label='Mean', color='blue')
plt.fill_between(x_values, ci_lower, ci_upper, color='blue', alpha=0.2)


plt.xlabel('Move Timestep')
plt.yticks([])
plt.ylabel('Reward')
plt.savefig('./illustrations/analysis/reward-len-fixed-70.png', format="png")
plt.close()