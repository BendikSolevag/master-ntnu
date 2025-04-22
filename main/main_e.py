import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment import SalmonFarmEnv
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = SalmonFarmEnv(infinite=False)
    state = torch.tensor(env.get_state(), dtype=torch.float32)
    
    #actor = Actor(4, len(state))
    #critic = Critic(len(state))
    R_bar = torch.tensor(0.0)

    agent = PGAgent(len(state), 4, 0.001, 0.001, 0.01)



    for ep in tqdm(range(25000)):

        env = SalmonFarmEnv(infinite=False)
        state = torch.tensor(env.get_state(), dtype=torch.float32)
        

        while True:
            if env.DONE == 1:
                break
        
            out = actor(state)
            probs = torch.distributions.Categorical(out)
            action = probs.sample()

            log_probs = probs.log_prob(action)        
        
            reward, done = env.step(action)
            next_state = torch.tensor(env.get_state(), dtype=torch.float32)
            
            delta = reward - R_bar + 0.99 * critic(next_state) - critic.forward(state)    
            

            critic_loss = delta**2
            actor_loss = -torch.sum(log_probs) * delta
            combined = actor_loss + critic_loss
            combined.backward()

            actor.opt.step(); actor.opt.zero_grad()
            critic.opt.step(); critic.opt.zero_grad()
        
        
            state = next_state


class PGAgent():
    def __init__(self, state_size, action_size, actor_lr, vf_lr, discount ):
        self.action_size = action_size
        self.actor_net = Actor(state_size, action_size).to(device)
        self.vf_net = Critic(state_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.vf_optimizer = optim.Adam(self.vf_net.parameters(), lr=vf_lr)
        self.discount = discount
        
    def select_action(self, state):
        #get action probs then randomly sample from the probabilities
        with torch.no_grad():
            input_state = torch.FloatTensor(state).to(device)
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
        state_t = torch.FloatTensor(state_list).to(device)
        action_t = torch.LongTensor(action_list).to(device).view(-1,1)
        return_t = torch.FloatTensor(return_array).to(device).view(-1,1)
        
        # get value function estimates
        vf_t = self.vf_net(state_t).to(device)
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

