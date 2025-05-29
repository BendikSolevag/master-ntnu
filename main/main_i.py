import torch
from environment import SalmonFarmEnv
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents.q_learner_inf import Agent
from torch import nn
import torch as T
from torch import optim
from torch.nn import functional as F


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


def main():

    env = SalmonFarmEnv(infinite=True)
    state = env.get_state()
    Q_eval = DeepQNetwork(lr=0.00001, input_dims=[len(state)], fc1_dims=64, fc2_dims=64, n_actions=4)
    state = T.tensor([state], dtype=T.float).to(Q_eval.device)
    
    
    htstep = 60 #+ np.random.uniform(-0.2, 0.2)
    mtstep = 23 #+ np.random.uniform(-0.2, 0.2)
    ptstep = 40 #+ np.random.uniform(-0.2, 0.2)

    weights_closed = []
    weights_open = []
    pricehist = []

    max_tsteps = 10000000
    r_bar = 0
    
    

    for ep in tqdm(range(max_tsteps)):

        actions = Q_eval.forward(state)
        action = T.argmax(actions).item()
        action = 0
        
        # If action is 2, force epsilon next iteration. Select move, plant, harvest timestamp.
        

        # One case for if generation has unharvested in move, one otherwise?
        if ep < 0.99 * max_tsteps:
            if round(env.AGE_CLOSED, 6) == round(mtstep * 1/52, 6):
                action = 2
                mtstep = -1
            if round(env.AGE_OPEN, 6) == round(ptstep * 1/52, 6):
                action = 3
                ptstep = -1
            if round(env.AGE_OPEN, 6) == round(htstep * 1/52, 6):
                
                action = 1
                htstep = -1


        # If there is no fish in any pen, force re-plant
        if env.NUMBER_CLOSED == 0 and env.NUMBER_OPEN == 0:
            action = 3
        if env.AGE_CLOSED > 2:
            action = 2
        if env.AGE_OPEN > 3:
            action = 1
        
        reward, done = env.step(action)
        reward = reward / 1e7
        next_state =  T.tensor([env.get_state()], dtype=T.float).to(Q_eval.device)

        # If age_open or age_closed greater than 2, punish (repeat success from episodic)
        if env.AGE_OPEN > 2 or env.AGE_CLOSED > 2:
            reward -= 10



        qot1 = Q_eval.forward(state)
        old_q = qot1[0, action]
        qnt1 = Q_eval.forward(next_state)
        qnt2 = T.argmax(qnt1).item()
        new_q = qnt1[0, qnt2]

        delta = reward - r_bar + new_q.detach() - old_q.detach()
        r_bar = r_bar + 0.005 + delta.detach()
        
        (-delta * old_q).backward()
        Q_eval.optimizer.zero_grad()
        Q_eval.optimizer.step()
        
        

        if action == 2 and ep < 0.99 * max_tsteps:
            htstep = np.random.randint(50, 120)            
            ptstep = np.random.randint(htstep / 2, htstep-1)
            mtstep = np.random.randint(0, ptstep)
            

        state = next_state
        
        
        weights_closed.append(env.GROWTH_CLOSED)
        weights_open.append(env.GROWTH_OPEN)
        pricehist.append(env.PRICE)
        if len(weights_closed) > max_tsteps*0.01:
            weights_closed.pop(0)
            weights_open.pop(0)
            pricehist.pop(0)


    plt.plot(weights_open, label="open")
    plt.plot(weights_closed, label="closed")
    plt.legend()
    plt.show()        




if __name__ == "__main__":
    main()
