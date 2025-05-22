
from environment import SalmonFarmEnv
import numpy as np
import matplotlib.pyplot as plt

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

plt.savefig('./envdynamics.png', format="png")
plt.close()