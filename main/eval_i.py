import sys
import numpy as np
from environment import SalmonFarmEnv
from agents.q_learner import Agent
import torch as T
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(closed_coefficient):
    env = SalmonFarmEnv(infinite=True, closed_coefficient=closed_coefficient)
    state = env.get_state()
    
    agent = Agent(gamma=0.99, lr=0.000001, input_dims=[len(state)], batch_size=4, n_actions=4)
    agent.Q_eval.load_state_dict(T.load(f'./models/agent/infhor/q-{closed_coefficient}.pt'))
    


    weights_closed = []
    weights_open = []
    pricehist = []
    actionhist = []

    max_tsteps = 520
    
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
        

    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 5)
    ax.plot(weights_open, label="open")
    ax.plot(weights_closed, label="closed")
    
    plt.legend()
    plt.savefig(f'./illustrations/results/infhor/production-cycle-q{closed_coefficient}.png', format="png")
    plt.close()

    sims = 500
    trews = []
    max_tsteps = 52 * 100
    paths = []
    for _ in tqdm(range(sims)):
        env = SalmonFarmEnv(infinite=True, closed_coefficient=closed_coefficient)
        state = env.get_state()
        path = []
        for t in range(max_tsteps):
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

            reward = reward * np.e**(-0.045 * (1/52) * t)
            path.append(reward)
            
            next_state =  env.get_state()
            state = next_state

        paths.append(path)
        trews.append(sum(path))
    
    T.save(paths, f'./data/results/infhor/rewardpaths/{closed_coefficient}.pt')
    print(np.mean(trews))




    onlyopenpaths = []
    onlyopentrews = []
    for _ in tqdm(range(sims)):
        env = SalmonFarmEnv(only_open=True, infinite=True, closed_coefficient=closed_coefficient)
        state = env.get_state()
        path = []

        

        for t in range(max_tsteps):
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

            reward = reward * np.e**(-0.045 * (1/52) * t)
            path.append(reward)            
            next_state =  env.get_state()
            state = next_state

        onlyopenpaths.append(path)
        onlyopentrews.append(sum(path))
    
    T.save(paths, f'./data/results/infhor/rewardpaths/{closed_coefficient}only_open.pt')
    print(np.mean(onlyopentrews))

        


if __name__ == "__main__": 
    coef = int(sys.argv[1])
    main(coef)
