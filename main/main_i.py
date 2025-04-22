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


    gro = []
    grc = []
    no = []
    nc = []
    rh = []
    ah = []
    ii = 0   
    
    fix, ax = plt.subplots(3, 2)
    
    for ep in tqdm(range(25000)):

        
        out = actor(state)
        probs = torch.distributions.Categorical(out)
        action = probs.sample()
        
        if ep < 5000:
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
        

        gro.append(env.GROWTH_OPEN)
        grc.append(env.GROWTH_CLOSED)
        no.append(env.NUMBER_OPEN)
        nc.append(env.NUMBER_CLOSED)
        rh.append(reward)
        ah.append(action)
        if len(gro) > 500:
            gro.pop(0)
            grc.pop(0)
            no.pop(0)
            nc.pop(0)
            rh.pop(0)
            ah.pop(0)

        next_state = torch.tensor(env.get_state(), dtype=torch.float32)
        
        delta = reward - R_bar + 0.99 * critic(next_state) - critic.forward(state)    
        R_bar = ((1 - 1e-2) * R_bar + 1e-2 * delta).detach()

        critic_loss = delta**2
        actor_loss = -torch.sum(log_probs) * delta
        combined = actor_loss + critic_loss
        combined.backward()

        actor.opt.step(); actor.opt.zero_grad()
        critic.opt.step(); critic.opt.zero_grad()
        
        

        state = next_state

        ii = ii + 1

    ax[0, 0].plot(grc)
    ax[0, 0].set_title("Growth rate closed")
    ax[0, 1].plot(gro)
    ax[0, 1].set_title("Growth rate open")
    ax[1, 0].plot(nc)
    ax[1, 0].set_title("n closed")
    ax[1, 1].plot(no)
    ax[1, 1].set_title("n open")

    ax[2, 0].plot(rh)
    ax[2, 0].set_title("reward hist")
    ax[2, 1].plot(ah)
    ax[2, 1].set_title("action hist")
    
    plt.show()        




if __name__ == "__main__":
    main()
