import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from environment import SalmonFarmEnv
from tqdm import tqdm
from agent import Actor, Critic


def main():
    env = SalmonFarmEnv(infinite=False)
    state = torch.tensor(env.get_state(), dtype=torch.float32)
    
    agent = PGAgent(len(state), 4, 0.001, 0.001, 0.01)

    stats_rewards_list = [] # store stats for plotting in this
    stats_every = 10 # print stats every this many episodes
    stats_actor_loss, stats_vf_loss = 0., 0.
    

    for ep in range(25000):
        total_reward = 0
        timesteps = 0
        env = SalmonFarmEnv(infinite=False)
        state = env.get_state()
        
        state_list, action_list, reward_list = [], [], []
        
        while True:
            timesteps += 1

            if env.DONE == 1:
                break
            if timesteps > 199:
                break
            
            action = agent.select_action(state)
        
            reward, done = env.step(action)
        
            next_state = env.get_state()

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            
            state = next_state
        
        
        actor_loss, vf_loss = agent.train(state_list, action_list, reward_list)
        stats_rewards_list.append((ep, total_reward, timesteps))
        stats_actor_loss += actor_loss
        stats_vf_loss += vf_loss
        total_reward = 0
        
        if ep > 0 and ep % stats_every == 0:
            print('Episode: {}'.format(ep),
                'Timestep: {}'.format(timesteps),
                'Total reward: {:.1f}'.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[1]),
                'Episode length: {:.1f}'.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[2]),
                'Actor Loss: {:.4f}'.format(stats_actor_loss/stats_every), 
                'VF Loss: {:.4f}'.format(stats_vf_loss/stats_every))
            stats_actor_loss, stats_vf_loss = 0., 0.


class PGAgent():
    def __init__(self, state_size, action_size, actor_lr, vf_lr, discount ):
        self.action_size = action_size
        self.actor_net = Actor(action_size, state_size)
        self.vf_net = Critic(state_size)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.vf_optimizer = optim.Adam(self.vf_net.parameters(), lr=vf_lr)
        self.discount = discount
        
    def select_action(self, state):
        #get action probs then randomly sample from the probabilities
        with torch.no_grad():
            input_state = torch.FloatTensor(state)
            action_probs = self.actor_net(input_state)
            #detach and turn to numpy to use with np.random.choice()
            action_probs = action_probs.detach().cpu().numpy()
            action = np.random.choice(np.arange(self.action_size), p=action_probs)
        return action

    def train(self, state_list, action_list, reward_list):
        
        #turn rewards into return
        trajectory_len = len(reward_list)
        return_array = np.zeros((trajectory_len,))
        g_return = 0.
        for i in range(trajectory_len-1,-1,-1):
            g_return = reward_list[i] + self.discount*g_return
            return_array[i] = g_return
            
        # create tensors
        state_t = torch.tensor(state_list, dtype=torch.float)
        action_t = torch.tensor(action_list, dtype=torch.long).view(-1,1)
        return_t = torch.tensor(return_array, dtype=torch.float).view(-1,1)
        
        # get value function estimates
        vf_t = self.vf_net(state_t)
        with torch.no_grad():
            advantage_t = return_t - vf_t
        
        # calculate actor loss
        selected_action_prob = self.actor_net(state_t).gather(1, action_t)
        # REINFORCE loss:
        #actor_loss = torch.mean(-torch.log(selected_action_prob) * return_t)
        # REINFORCE Baseline loss:
        actor_loss = torch.mean(-torch.log(selected_action_prob) * advantage_t)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 

        # calculate vf loss
        loss_fn = nn.MSELoss()
        vf_loss = loss_fn(vf_t, return_t)
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step() 
        
        return actor_loss.detach().cpu().numpy(), vf_loss.detach().cpu().numpy()


if __name__ == "__main__":
    main()

