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


def main():
    env = SalmonFarmEnv(infinite=True)
    state = torch.tensor(env.get_state(), dtype=torch.float32)
    

    actor = Actor(4, len(state))
    critic = Critic(len(state))
    R_bar = torch.tensor(0.0)


    num_episodes = 3
    max_steps_per_episode = 200

    
    accumulative_reward = 0
    accrew = []
    
    ii = 0
    
    
    
    for ep in tqdm(range(num_episodes * max_steps_per_episode)):

        
        out = actor(state)
        probs = torch.distributions.Categorical(out)
        #action = probs.sample()
        action = torch.tensor(0)
        if ii == 35:
            action = torch.tensor(2)
        if ii == 70:
            action = torch.tensor(3)
        if ii == 71:
            action = torch.tensor(1)
            ii = 0

        log_probs = probs.log_prob(action)        
        reward, done = env.step(action)
        
        

        
        
        accumulative_reward += reward
        accrew.append(reward)

        
        
        


        next_state = torch.tensor(env.get_state(), dtype=torch.float32)
        
        delta = reward - R_bar + 0.99 * critic(next_state) - critic.forward(state)    
        R_bar = ((1 - 1e-4) * R_bar + 1e-4 * delta).detach()

        critic_loss = delta**2
        actor_loss = -torch.sum(log_probs) * delta
        combined = actor_loss + critic_loss
        combined.backward()

        actor.opt.step()
        critic.opt.step()
        actor.opt.zero_grad()
        critic.opt.zero_grad()


        state = next_state

        ii = ii + 1

    plt.plot(accrew)
    plt.show()        




if __name__ == "__main__":
    main()
