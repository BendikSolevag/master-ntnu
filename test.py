"""
5-step Advantage Actor-Critic for the toy “wait-4-steps” environment.
The only material changes compared with your original script are:

  • A circular transition buffer inside Agent that holds n_step items.
  • Log-probabilities are **re-computed** at update time instead of being
    stored, preventing the autograd-in-place error.
  • The training loop now feeds (state, action, reward, next_state, done)
    to Agent.remember_and_learn(…).

Everything else (network, environment, plotting) is identical.
"""

import time
import collections
from matplotlib import pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from estimates.estimate_growth import GrowthNN


class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi  = nn.Linear(fc2_dims, n_actions)
        self.v   = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.pi(x), self.v(x)

class Agent:
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, gamma=0.99, n_step=200):
        self.gamma = gamma
        self.n_step = n_step
        self.net = ActorCriticNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims)
        self.buffer = collections.deque(maxlen=n_step)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.net.device)
        logits, _ = self.net(state)
        probs = F.softmax(logits, dim=1)
        dist  = T.distributions.Categorical(probs)
        action = dist.sample().item()
        return action

    def _flush_buffer(self, next_state, done):
        """
        Perform one gradient update for the *oldest* transition in the buffer
        using an n-step bootstrapped return.
        """
        # oldest transition we are updating now
        state, action, _ = self.buffer[0]

        # recompute log π(a|s) with current network parameters
        s = T.tensor([state], dtype=T.float32, device=self.net.device)
        logits, _ = self.net(s)
        dist = T.distributions.Categorical(F.softmax(logits, dim=1))
        logp = dist.log_prob(T.tensor([action], device=self.net.device))

        # estimated value from state_{t+n} (0 if terminal)
        with T.no_grad():
            ns = T.tensor([next_state], dtype=T.float32, device=self.net.device)
            _, v_approx = self.net(ns)
            v_approx *= (1 - int(done))

        # n-step discounted return G_t, ending with approximated value v
        G = v_approx
        for _, _, r in reversed(self.buffer):
            G = r + self.gamma * G

        # critic estimate current state V(s_t)
        _, v = self.net(s)
        delta = G - v

        # update
        self.net.optimizer.zero_grad()
        actor_loss  = -logp * delta.detach()   # stop grad into critic
        critic_loss = delta.pow(2)
        (actor_loss + critic_loss).backward()
        self.net.optimizer.step()


    def remember_and_learn(self, state, action, reward, next_state, done):
        """Store transition and, when appropriate, perform a learning step."""
        self.buffer.append((state, action, reward))

        # when buffer full or episode ends → learn & discard oldest item
        if len(self.buffer) == self.n_step or done:
            self._flush_buffer(next_state, done)
            self.buffer.popleft()

        # on episode end there may still be items left to learn from
        if done:
            while self.buffer:
                self._flush_buffer(next_state, done=True)
                self.buffer.popleft()


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
