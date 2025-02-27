import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment import SalmonFarmEnv


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 state_dim=5,
                 action_dim=4,
                 hidden_size=64,
                 gamma=0.99,
                 lr=1e-3,
                 batch_size=4,
                 buffer_size=10000,
                 min_replay_size=10,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=5000,
                 target_update_freq=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq

        # Q-network and target network
        self.q_net = QNetwork(state_dim, action_dim, hidden_size)
        self.target_net = QNetwork(state_dim, action_dim, hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.steps_done = 0

    def act(self, state):
        """
        Epsilon-greedy action selection
        """
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                  np.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            # choose best action from Q-network
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, transition):
        # transition is (state, action, reward, next_state, done)
        self.replay_buffer.append(transition)

    def train_step(self):
        """
        Sample a batch from replay buffer and update Q-network
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q estimates
        q_values = self.q_net(states_t).gather(1, actions_t)

        # Next Q values (from target net)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

        # Target for Q
        target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # Loss
        loss = nn.MSELoss()(q_values, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


def main():
    
    agent = DQNAgent(state_dim=6, action_dim=3)

    num_episodes = 20000
    max_steps_per_episode = 500  # or some large number

    rewards_history = []

    for ep in range(num_episodes):
        episode_reward = 0.0
        env = SalmonFarmEnv()
        state = np.array([env.PRICE, env.LICE, env.GROWTH, env.NUMBER, env.MOVED, env.TREATING])
        
        for step in range(max_steps_per_episode):    
            action = agent.act(state)
            
            reward, done = env.step(action)
            episode_reward += reward
            next_state = np.array([env.PRICE, env.LICE, env.GROWTH, env.NUMBER, env.MOVED, env.TREATING])

            agent.store_transition((state, action, reward, next_state, float(done)))
            agent.train_step()

            state = next_state
            if done:
                print(step)
                break

        rewards_history.append(episode_reward)
        if (ep+1) % 50 == 0:
            avg_rew = np.mean(rewards_history[-50:])
            print(f"Episode {ep+1}, Average Reward (last 50): {avg_rew:.2f}")

    # After training, we can see how the policy behaves (in a quick test)
    print("Training complete. Testing final policy...")
    env = SalmonFarmEnv()
    state = np.array([env.PRICE, env.LICE, env.GROWTH, env.NUMBER, env.MOVED, env.TREATING])
    done = False
    test_reward = 0
    while not done:
        action = agent.act(state)
        reward, done = env.step(action)
        test_reward += reward
        state = np.array([env.PRICE, env.LICE, env.GROWTH, env.NUMBER, env.MOVED, env.TREATING])
    print(f"Test episode reward: {test_reward:.2f}")


if __name__ == "__main__":
    main()