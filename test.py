from matplotlib import pyplot as plt
import numpy as np
import torch as T
from tqdm import tqdm
from estimates.estimate_growth import GrowthNN
from main.n_step_actor_critic import Agent



class TEnv:
    def __init__(self):
        self.state = 0.2
        self.DONE  = False
        self.AGE = 0
        # Growth rate NN
        self.growth_model = GrowthNN(input_size=4)
        self.growth_model.load_state_dict(T.load('./models/growth/1743671011.288821-model.pt', weights_only=True))
        self.growth_model.eval()

    def resolve_growth_open(self):
      explanatory = [
          round(self.AGE), #generation_approx_age, 
          self.state * 0.015 * 30, #feedamountperfish, 
          self.state, #mean_size,
          0.25, #mean_voksne_hunnlus,
      ]
      pred = self.growth_model.forward(T.tensor(explanatory, dtype=T.float32)).item()  
      # Cap prediction within reasonable range
      pred = max(min(pred, 8), 0.1)
      # Adjust monthly to weekly
      g_rate = np.log(pred / self.state) / 4.345
      self.state *= np.exp(g_rate)

    def get_state(self):
        return [self.state]

    def step(self, action: int):
        reward = -0.01
        self.AGE += 1/52
        if action == 1:
            reward += self.state
            reward -= 7
            self.DONE = True
        self.resolve_growth_open()
        return reward, self.DONE


def visualize_env():
  trews = []
  for i in range(120):
    env = TEnv()
    trew = 0
    for j in range(i):
      rew, done = env.step(0)
      trew += rew
    rew, done = env.step(1)
    trew += rew
    trews.append(trew)
  return trews


def main():
  env   = TEnv()
  obs   = env.get_state()

  agent = Agent(lr=1e-3,
                input_dims=[len(obs)],
                n_actions=2,
                fc1_dims=4, fc2_dims=4,
                gamma=0.99,
                n_step=200)

  episode_lengths = []

  for ep in tqdm(range(1500)):
    env = TEnv()
    state = env.get_state()
    timesteps = 0

    while True:
      action = agent.choose_action(state)

      # safety stop: force terminate after 20 steps
      if timesteps > 200:
        action = 1

      reward, done = env.step(action)
      next_state = env.get_state()

      agent.remember_and_learn(state, action, reward, next_state, done)

      if done:
        break

      state = next_state
      timesteps += 1

    episode_lengths.append(timesteps)
  return episode_lengths




if __name__ == "__main__":

  fig, ax = plt.subplots(2, 1)
  
  trews = visualize_env()
  lens = main()

  ax[0].plot(trews)
  ax[1].plot(lens)
  plt.show()
