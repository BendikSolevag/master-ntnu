import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from environment import SalmonFarmEnv
from tqdm import tqdm
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)

class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
                 gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()


def main():
    # The reinforce with baseline version is the most stable algorithm, but the problem is moreso with the design of the reward function. A 'least negative' reward signal works better than a 'most positive'. Why is this?
    env = SalmonFarmEnv(infinite=False)
    state = torch.tensor(env.get_state(), dtype=torch.float32)
    now = time.time()
    
    agent = Agent(0.001, [len(state)], 32, 32, 4)

    stats_rewards_list = [] # store stats for plotting in this
    stats_every = 100 # print stats every this many episodes
    fig, ax = plt.subplots(3, 1)

    top_rew = -float('inf')
    

    for ep in tqdm(range(1000000)):
        total_reward = 0
        timesteps = 0
        env = SalmonFarmEnv(infinite=False)
        state = env.get_state()
        
        state_list, action_list, reward_list = [], [], []

        growthlist_c = []
        growthlist_o = []
        treatlist = []
        
        while True:
            growthlist_c.append(env.GROWTH_CLOSED)
            growthlist_o.append(env.GROWTH_OPEN)
            treatlist.append(env.TREATING)
            timesteps += 1

            if env.DONE == 1:
                break
            if timesteps > 199:
                break
            
            action = agent.choose_action(state)
            if timesteps < 30:
                action = 0
            
            if timesteps > 120:
                action = 3
            

            
            reward, done = env.step(action)
            
            total_reward += reward
            next_state = env.get_state()

            agent.learn(state, reward, next_state, done)

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            
            state = next_state
        
        
        if False and ep == 500:
            ax[0].plot(growthlist_c)
            ax[1].plot(growthlist_o)
            ax[2].plot(treatlist)
            plt.show()
        stats_rewards_list.append((ep, total_reward, timesteps))

        if ep % stats_every == 0:
            if np.mean(stats_rewards_list[-stats_every:],axis=0)[1] > top_rew:
                top_rew = np.mean(stats_rewards_list[-stats_every:],axis=0)[1]
                print('new top: ', top_rew)
                sd = agent.actor_critic.state_dict()
                torch.save(sd, f'./models/agent/{now}-model.pt')

            
            print('feed', round(env.total_cost_feed / ((env.GROWTH_CLOSED * env.NUMBER_CLOSED) + (env.GROWTH_OPEN * env.NUMBER_OPEN)), 4))
            print('trea', round(env.total_cost_treatment / env.total_cost_feed, 4))
            print('harv', round(env.total_cost_harvest / env.total_cost_feed, 4))
            print('oper', round(env.total_cost_operation / env.total_cost_feed, 4))
            print('Episode: {}'.format(ep),
                'Total reward: {:.1f}'.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[1]),
                'Episode length: {:.1f}'.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[2])
            )


if __name__ == '__main__':
    main()

