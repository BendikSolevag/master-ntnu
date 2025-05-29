import torch as T
import numpy as np
from matplotlib import pyplot as plt

means_move = np.array(T.load('./means_move.pt'))
stds_move = np.array(T.load('./stds_move.pt'))
means_flat = np.array(T.load('./means_flat.pt'))
stds_flat = np.array(T.load('./stds_flat.pt'))


x = [1, 2, 4, 6, 8, 10, 12]


#plt.figure(figsize=(8, 5))
plt.plot(x, means_move, label="Hybrid", color="blue")
plt.fill_between(x, means_move - stds_move, means_move + stds_move, color="blue", alpha=0.2)

plt.plot(x, means_flat, label="Open", color="purple")
plt.fill_between(x, means_flat - stds_flat, means_flat + stds_flat, color="purple", alpha=0.2)

plt.ylabel("Discounted aggregated reward")
plt.xlabel("Operational cost coefficient")

plt.legend()
plt.savefig('./illustrations/results/episodic/total-discounted-reward-comparison-per-operating-cost.png')