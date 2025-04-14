import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment import SalmonFarmEnv
import time
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
                 eps_start=0.9,
                 eps_end=0.001,
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

        q_values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

        target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


def main():
    
    agent = DQNAgent(state_dim=7, action_dim=4)

    num_episodes = 10000
    max_steps_per_episode = 200

    rewards_history = []
    terminating_step_history = []
    top_avg_rew = -1 * float('inf')
    top_timestamp = time.time()

    
    for ep in range(num_episodes):
        episode_reward = 0.0
        terminating_step = 0
        env = SalmonFarmEnv(infinite=False)

        state = np.array(env.get_state())

        growth_closeds = []
        growth_opens = []
        action_history = []

        
        
        for step in range(max_steps_per_episode):
            terminating_step += 1
            action = agent.act(state)
            
            reward, done = env.step(action)

            growth_closeds.append(env.GROWTH_CLOSED)
            growth_opens.append(env.GROWTH_OPEN)
            action_history.append(action)

            episode_reward += reward
            next_state = np.array(env.get_state())

            agent.store_transition((state, action, reward, next_state, float(env.DONE)))
            agent.train_step()

            state = next_state
            if env.DONE == 1:
                break
        
        if ep == (num_episodes-1):
            # Create 2 subplots (1 row, 2 columns)
            fig, axs = plt.subplots(2, 1)  # 1 row, 2 columns

            # Plot on the first subplot
            axs[0].plot(growth_closeds, label="closed")
            axs[0].plot(growth_opens, label="open")
            

            # Plot on the second subplot
            axs[1].plot(action_history)
            fig.tight_layout()
            fig.savefig('./a.png', format="png", dpi=600)
            

        rewards_history.append(episode_reward)
        terminating_step_history.append(terminating_step)
        if (ep+1) % 50 == 0:
            avg_rew = np.mean(rewards_history)
            
            if avg_rew >= top_avg_rew:
                top_avg_rew = avg_rew
                top_timestamp = time.time()
                torch.save(agent.q_net.state_dict(), f'./models/agent/{top_timestamp}-q_net.pt')
                torch.save(agent.target_net.state_dict(), f'./models/agent/{top_timestamp}-target_net.pt')

            print(f"Episode {ep+1}, Average reward (last 50): {avg_rew:.2f}")
            rewards_history = []
            avg_len = np.mean(terminating_step_history)
            print(f"Episode {ep+1}, Average length (last 50): {avg_len:.2f}")
            terminating_step_history = []

    return
    agent.q_net.load_state_dict(torch.load(f'./models/agent/{top_timestamp}-q_net.pt', weights_only=True))
    agent.target_net.load_state_dict(torch.load(f'./models/agent/{top_timestamp}-target_net.pt', weights_only=True))
    env = SalmonFarmEnv(infinite=False)

    state = np.array(env.get_state())
    test_reward = 0
    step = 0
    closed_pen_hist = []
    open_pen_hist = []
    while env.DONE != 1:
        action = agent.act(state)
        reward, done = env.step(action)
        test_reward += reward
        state = np.array(env.get_state())
        step += 1
        closed_pen_hist.append(env.GROWTH_CLOSED)
        open_pen_hist.append(env.GROWTH_OPEN)
    print("Test episode reward")
    print(env.lice_t)
    print(test_reward)

    plt.plot(closed_pen_hist, label="Closed pen history")
    plt.plot(open_pen_hist, label="Open pen history")
    plt.legend()
    plt.savefig('growth_histories.png', format="png", dpi=800)
    plt.close()



if __name__ == "__main__":
    #main()
    
    

    env = SalmonFarmEnv(infinite=False)
    total_r = 0
    
    
    for _ in range(10):
        r, d = env.step(0)
        total_r += r

    r, d = env.step(2)
    total_r += r

    for _ in range(10):
        r, d = env.step(0)
        total_r += r

    r, d = env.step(3)
    total_r += r

    print("{:e}".format(total_r))
    