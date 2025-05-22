import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as T
from environment import SalmonFarmEnv
from tqdm import tqdm
import numpy as np
#from agents.n_step_actor_critic import Agent
from agents.q_learner import Agent



def main():
    # The reinforce with baseline version is the most stable algorithm, but the problem is moreso with the design of the reward function. A 'least negative' reward signal works better than a 'most positive'. Why is this?
    env = SalmonFarmEnv(infinite=False)
    state = T.tensor(env.get_state(), dtype=T.float32)
    
    agent = Agent(gamma=0.99956, lr=0.00001, input_dims=[len(state)], batch_size=4, n_actions=4)
    
    termtimesteps = []
    movetimesteps = []
    totalrewards = []
    for ep in tqdm(range(2000)):
        total_reward = 0
        timesteps = 0
        env = SalmonFarmEnv(infinite=False)
        state = env.get_state()
        
        state_list, action_list, reward_list = [], [], []

        growthlist_c = []
        growthlist_o = []
        move_timestep = -1

        epsilon = np.random.uniform() < 0.1
        htstep = np.random.randint(40, 130)
        mtstep = np.random.randint(0, htstep)

        
        while True:
            growthlist_c.append(env.GROWTH_CLOSED)
            growthlist_o.append(env.GROWTH_OPEN)
            
            


            
            action = agent.choose_action(state)

            if epsilon:
                if timesteps == htstep:
                    action = 1
                if timesteps == mtstep:
                    action = 2
            
            if timesteps < 20:
                action = 0
            
            if action == 2 and move_timestep == -1:
                move_timestep = timesteps
            

            
            reward, done = env.step(action)
            reward = reward / 1e7
            if timesteps > 110:
                reward -= 5
            
            total_reward += reward
            next_state = env.get_state()
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)

            if action == 2:
                print(reward)

            if env.DONE == 1:
                break
            if timesteps > 199:

                break


            timesteps += 1
            state = next_state
        movetimesteps.append(move_timestep)
        termtimesteps.append(timesteps)
        totalrewards.append(total_reward)
    fix, ax = plt.subplots(3, 1)
    ax[0].plot(termtimesteps)
    ax[1].plot(move_timestep)
    ax[2].plot(totalrewards)
    

    plt.show()


if __name__ == '__main__':
    main()

