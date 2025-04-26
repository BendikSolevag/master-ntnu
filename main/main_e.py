import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch as T
from environment import SalmonFarmEnv
from tqdm import tqdm
import numpy as np
from agents.n_step_actor_critic import Agent





def main():
    # The reinforce with baseline version is the most stable algorithm, but the problem is moreso with the design of the reward function. A 'least negative' reward signal works better than a 'most positive'. Why is this?
    env = SalmonFarmEnv(infinite=False)
    state = T.tensor(env.get_state(), dtype=T.float32)
    
    
    agent = Agent(lr=0.001, input_dims=[len(state)], n_actions=4)
    
    termtimesteps = []
    totalrewards = []
    for ep in tqdm(range(1000)):
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
            


            
            action = agent.choose_action(state)
            if timesteps < 30:
                action = 0
            
            if timesteps > 140:
                action = 3
            

            
            reward, done = env.step(action)
            
            total_reward += reward
            next_state = env.get_state()

            agent.learn(state, action, reward, next_state, done)

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)


            if env.DONE == 1:
                break
            if timesteps > 199:
                break


            timesteps += 1
            state = next_state
        
        termtimesteps.append(timesteps)
        totalrewards.append(total_reward)
    fix, ax = plt.subplots(2, 1)
    ax[0].plot(termtimesteps)
    ax[1].plot(totalrewards)
    
    
    plt.show()


if __name__ == '__main__':
    main()

