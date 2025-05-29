
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch as T
from agents.q_learner import Agent
from environment import SalmonFarmEnv
from tqdm import tqdm


def main():

  coefficients = [1, 2, 4, 6, 8, 10, 12]

  
  means_move = []
  stds_move = []
  means_flat = []
  stds_flat = []
  pbar = tqdm(total=len(coefficients) * 2 * 1000)
  for coef in coefficients:
    agent = Agent(gamma=0.99956, lr=0.00001, input_dims=[6], batch_size=4, n_actions=4)
    agent.Q_eval.load_state_dict(T.load(f'./models/agent/episodic/q-{coef}.pt'))
    
    total_rewards = []
    for _ in range(1000):
        pbar.update(1)
        env = SalmonFarmEnv(closed_coefficient=coef, infinite=False)
        state = env.get_state()
        total_reward = 0
        timesteps = 0
        
        while True:
            action = agent.choose_action(state)
            if timesteps < 20 and action == 1:
                action = 0
            reward, done = env.step(action)
            total_reward += (reward * np.e**(-0.045 * timesteps/52))
            next_state = env.get_state()

            if env.DONE == 1:
                break
            if timesteps > 199:
                break
            timesteps += 1
            state = next_state

        total_rewards.append(total_reward)
    
    means_move.append(np.mean(total_rewards))
    stds_move.append(np.std(total_rewards))

    total_rewards = []
    for _ in range(1000):
        pbar.update(1)
        env = SalmonFarmEnv(closed_coefficient=coef, infinite=False)
        total_reward = 0
        timesteps = 0
        
        while True:
            action = agent.choose_action(state)
            if timesteps == 0:
              action = 2

            if timesteps < 20 and action == 1:
                action = 0
            

            reward, done = env.step(action)

                        
            total_reward += (reward * np.e**(-0.045 * timesteps/52))
            next_state = env.get_state()

            if env.DONE == 1:
                break
            if timesteps > 199:
                break
            timesteps += 1
            state = next_state

        total_rewards.append(total_reward)
    
    means_flat.append(np.mean(total_rewards))
    stds_flat.append(np.std(total_rewards))

  T.save(means_move, './means_move.pt')
  T.save(stds_move, './stds_move.pt')
  T.save(means_flat, './means_flat.pt')
  T.save(stds_flat, './stds_flat.pt')
    
  plt.plot(means_move, label="Open/Closed", color="blue")
  plt.plot(means_flat, label="Open", color="purple")
  plt.legend()
  plt.show()
    



if __name__ == '__main__':
  main()