import torch
from environment import SalmonFarmEnv
from agents.q_learner import Agent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import torch as T
from torch import optim
from torch.nn import functional as F


def main():
    closed_coefficient = 1
    env = SalmonFarmEnv(infinite=True, closed_coefficient=closed_coefficient)
    state = env.get_state()
    
    agent = Agent(gamma=0.99, lr=0.000001, input_dims=[len(state)], batch_size=4, n_actions=4)
    agent.Q_eval.load_state_dict(T.load('./models/agent/infhor/q-1.pt'))
    
    
    
    htstep = 60 #+ np.random.uniform(-0.2, 0.2)
    mtstep = 23 #+ np.random.uniform(-0.2, 0.2)
    ptstep = 40 #+ np.random.uniform(-0.2, 0.2)

    weights_closed = []
    weights_open = []
    pricehist = []
    actionhist = []

    max_tsteps = 1000
    
    
    for ep in tqdm(range(max_tsteps)):

        action = agent.choose_action(state)
        
        # If there is no fish in any pen, force re-plant
        if env.NUMBER_CLOSED == 0 and env.NUMBER_OPEN == 0:
            action = 3

        # soft weight limits
        if env.AGE_CLOSED > 2:
            action = 2
        if env.AGE_OPEN > 3:
            action = 1
        
        reward, done = env.step(action)
        reward = reward / 1e7
        next_state =  env.get_state()

        state = next_state
        
        weights_closed.append(env.GROWTH_CLOSED)
        weights_open.append(env.GROWTH_OPEN)
        pricehist.append(env.PRICE)
        actionhist.append(action)
        

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(weights_open, label="open")
    ax[0].plot(weights_closed, label="closed")
    ax[1].plot(actionhist)
    plt.legend()
    plt.show()        




if __name__ == "__main__":
    main()
