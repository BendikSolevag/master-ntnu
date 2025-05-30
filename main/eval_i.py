import torch
from environment import SalmonFarmEnv
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    closed_coefficient = 1
    env = SalmonFarmEnv(infinite=True, closed_coefficient=closed_coefficient)
    state = env.get_state()
    lr = 0.001
    Q_eval = DeepQNetwork(lr=lr, input_dims=[len(state)], fc1_dims=64, fc2_dims=64, n_actions=4)
    Q_eval.load_state_dict(T.load(f'./models/agent/infhor/q-1'))
    state = T.tensor([state], dtype=T.float).to(Q_eval.device)
    
    
    htstep = 60 #+ np.random.uniform(-0.2, 0.2)
    mtstep = 23 #+ np.random.uniform(-0.2, 0.2)
    ptstep = 40 #+ np.random.uniform(-0.2, 0.2)

    weights_closed = []
    weights_open = []
    pricehist = []
    actionhist = []

    max_tsteps = 1000
    epsilon = False
    
    for ep in tqdm(range(max_tsteps)):

        actions = Q_eval.forward(state)
        action = T.argmax(actions).item()
        

        # One case for if generation has unharvested in move, one otherwise?
        if epsilon: #ep < 0.99 * max_tsteps:
            action = 0
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

        # soft weight limits
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


        if action == 2:# and ep < 0.99 * max_tsteps:
            epsilon = np.random.uniform() < 0.15
            htstep = np.random.randint(50, 120)            
            ptstep = np.random.randint(htstep / 2, htstep-1)
            mtstep = np.random.randint(0, ptstep)
            

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
