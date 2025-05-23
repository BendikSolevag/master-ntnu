"""

The main episodic version of the algorithm struggles to converge as the reward signal is affected by the scale. 
We therefore pretrain the model on this less noisy version of the algorithm

"""


from matplotlib import pyplot as plt
import numpy as np
import torch as T
from tqdm import tqdm
from util.growthmodel import GrowthNN
from agents.n_step_actor_critic import Agent



class TEnv:
  def __init__(self):
    self.GROWTH_CLOSED = 0.2
    self.GROWTH_OPEN = 0.0
    self.DONE  = False
    self.AGE = 0
    self.PRICE = 100
    self.NUMBER_CLOSED = 1000
    self.NUMBER_OPEN = 0
    self.LICE = 0.1
    self.OPEN = False
    self.TREATING = False

    self.LICE_TREAT_THRESHOLD = 0.5
    # Utility variables
    self.sliding_window_max = round((2/52)/(1/52))
    self.sliding_window_lice = [0 for _ in range(self.sliding_window_max)]
    
    # Growth rate NN
    self.growth_model = GrowthNN(input_size=4)
    self.growth_model.load_state_dict(T.load('./models/growth/1743671011.288821-model.pt', weights_only=True))
    self.growth_model.eval()


  def resolve_mortalityrate(self) -> float:
    categorydraw = np.random.uniform()
    if categorydraw >= 0.999**2:
      # gt 50% mortality
      return np.random.uniform(0.5, 0.9)
    if categorydraw >= 0.995**2:
      # gt 25% mortality
      return np.random.uniform(0.25, 0.5)
    if categorydraw >= 0.989**2:
      # gt 10% mortality
      return np.random.uniform(0.1, 0.25)
    if categorydraw >= 0.975*0.968:
      # gt 5% mortality
      return np.random.uniform(0.05, 0.1)
    if categorydraw >= 0.948*0.937:
      # gt 2.5% mortality
      return np.random.uniform(0.025, 0.05)
    if categorydraw >= 0.894*0.853:
      # gt 1% mortality
      return np.random.uniform(0.01, 0.025)
    return 0.0


  def resolve_lice(self):
    self.lice_kappa, self.lice_a, self.lice_b, self.lice_phi, self.lice_sigma, self.lice_t = 0.56451781,  0.17984971,  0.05243226, -0.62917791, 0.25959416, 0
    def theta(t, a, b, phi):
      return a + b * np.sin(2.0 * np.pi * (t / 52.0) + phi)
    seasonal_mean = theta(self.lice_t, self.lice_a, self.lice_b, self.lice_phi)
    self.LICE = (1 - self.lice_kappa) * self.LICE + self.lice_kappa * seasonal_mean + np.random.normal(0, self.lice_sigma)**2
  
  def resolve_treating(self):
    self.sliding_window_lice.pop(0)
    self.sliding_window_lice.append(self.LICE)
    window_exceeds = [True if x > self.LICE_TREAT_THRESHOLD else False for x in self.sliding_window_lice]
    self.TREATING = True if any(window_exceeds) else False

  def resolve_mortality(self):
    # Population loss due to treatment
    if self.NUMBER_OPEN > 0:
      if self.sliding_window_lice[0] > self.LICE_TREAT_THRESHOLD:
          mr = self.resolve_mortalityrate()
          self.NUMBER_OPEN = self.NUMBER_OPEN - mr * self.NUMBER_OPEN


  def resolve_growth_closed(self):
    if self.NUMBER_CLOSED <= 0:
      return
    explanatory = [
      round(self.AGE), #generation_approx_age, 
      self.GROWTH_CLOSED * 0.015 * 30, #feedamountperfish, 
      self.GROWTH_CLOSED, #mean_size,
      0, #mean_voksne_hunnlus, 0 as the system is closed
    ] 
    pred = self.growth_model.forward(T.tensor(explanatory, dtype=T.float32)).item()
    # Cap prediction within reasonable range
    pred = max(min(pred, 8), 0.1)
    # Adjust monthly to weekly
    g_rate = np.log(pred / self.GROWTH_CLOSED) / 4.345
    self.GROWTH_CLOSED *= np.exp(g_rate)
  
  def resolve_growth_open(self):
    if self.NUMBER_OPEN <= 0 or self.TREATING:
      return
    explanatory = [
      round(self.AGE), #generation_approx_age, 
      self.GROWTH_OPEN * 0.015 * 30, #feedamountperfish
      self.GROWTH_OPEN, #mean_size,
      self.LICE, #mean_voksne_hunnlus,
    ]
    pred = self.growth_model.forward(T.tensor(explanatory, dtype=T.float32)).item()  
    # Cap prediction within reasonable range
    pred = max(min(pred, 8), 0.1)
    # Adjust monthly to weekly
    g_rate = np.log(pred / self.GROWTH_OPEN) / 4.345
    self.GROWTH_OPEN *= np.exp(g_rate)


  def get_state(self):
    return [self.GROWTH_CLOSED, np.log(self.NUMBER_CLOSED + 1), self.GROWTH_OPEN, np.log(self.NUMBER_OPEN + 1), self.TREATING, self.LICE, np.log(self.PRICE)]

  def step(self, action: int):
    reward = -0.01 * (self.NUMBER_CLOSED + self.NUMBER_OPEN) * self.PRICE
    if not self.OPEN:
      reward -= 0.01 * self.NUMBER_CLOSED * self.PRICE
    if self.TREATING:
      reward -= 0.01 * self.NUMBER_OPEN * self.PRICE
        
    # Actions
    if action == 1:
      reward +=  self.GROWTH_OPEN * self.NUMBER_OPEN * self.PRICE
      reward -= 7 * (self.NUMBER_CLOSED + self.NUMBER_OPEN) * self.PRICE
      self.DONE = True

    if action == 2:
      self.OPEN = True
      self.NUMBER_OPEN = self.NUMBER_CLOSED
      self.NUMBER_CLOSED = 0
      self.GROWTH_OPEN = self.GROWTH_CLOSED
      self.GROWTH_CLOSED = 0
    
    # State vars
    self.AGE += 1/52

    self.resolve_growth_closed()
    if not self.TREATING:
      self.resolve_growth_open()
    
    self.resolve_lice()
    if self.OPEN:
      self.resolve_treating()
      self.resolve_mortality()

    self.PRICE = 100 + (np.random.beta(0.5, 0.5) - 0.5) * 60
    
    return reward, self.DONE


def main():
  env   = TEnv()
  obs   = env.get_state()

  agent = Agent(
    lr=1e-4,
    input_dims=[len(obs)],
    n_actions=3,
    fc1_dims=128, fc2_dims=128,
    gamma=0.99,
    n_step=200
  )

  episode_lengths = []
  move_timesteps = []
  
  for ep in tqdm(range(2500)):
    epsilon = np.random.uniform(0, 1)
    htstep = np.random.randint(60, 110)
    mtstep = np.random.randint(1, 59)
    
    env = TEnv()
    state = env.get_state()
    timesteps = 0
    move_timestep = 0


    while True:
      action = agent.choose_action(state)
      # If explore epoch, force random action 
      if epsilon < 0.1:
        action = 0
        if timesteps == mtstep:
           action = 2
        if timesteps == htstep:
           action = 1         

      # safety stop: force terminate after 20 steps
      if timesteps > 160:
        action = 1
      if action == 2 and move_timestep == 0:
         move_timestep = timesteps

      reward, done = env.step(action)
      reward = reward / (1000 * 100)
            
      next_state = env.get_state()

      agent.learn(state, action, reward, next_state, done)

      if done:
        break

      state = next_state
      timesteps += 1
    move_timesteps.append(move_timestep)
    episode_lengths.append(timesteps)
  return episode_lengths, move_timesteps


def visualize_env():
  trews = []
  for i in range(10, 120):
    env = TEnv()
    trew = 0
    for j in range(i):
      action = 0
      if j == int(i / 2):
        action = 2
      rew, done = env.step(action)
      trew += rew
    rew, done = env.step(1)

    trew += rew
    trews.append(trew)
  return trews

if __name__ == "__main__":

  fig, ax = plt.subplots(2, 1)
  
  trews = visualize_env()
  ax[0].plot(trews)

  lens, move_timesteps = main()
  ax[1].plot(lens)
  ax[1].plot(move_timesteps, color="pink", alpha=0.5)

  plt.show()
