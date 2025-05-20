import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as T

from tqdm import tqdm
import numpy as np
from agents.n_step_actor_critic import Agent
from pretrain_episodic import TEnv
import torch.nn.functional as F



def main():
    # The reinforce with baseline version is the most stable algorithm, but the problem is moreso with the design of the reward function. A 'least negative' reward signal works better than a 'most positive'. Why is this?

    env   = TEnv()
    obs   = env.get_state()

    agent = Agent(
        lr=1e-4,
        input_dims=[len(obs)],
        n_actions=4,
        fc1_dims=128, fc2_dims=128,
        gamma=0.99,
        n_step=200
    )

    agent.net_actor.load_state_dict(T.load("./models/agent/episodic/1747726111.0649228-actor-model-30000.pt", weights_only=True))
    agent.net_critic.load_state_dict(T.load("./models/agent/episodic/1747726111.0649228-critic-model-30000.pt", weights_only=True))
    
    agent.net_actor.eval()
    agent.net_critic.eval()
    
    termtimesteps = []
    move_timesteps = []
    for ep in tqdm(range(200)):
        total_reward = 0
        timesteps = 0
        env = TEnv()
        state = env.get_state()
        


        move_timestep = 0        
        
        while True:

            action = agent.choose_action(state)
            #if timesteps < 30:
            #    action = 0            
                        
            reward, done = env.step(action)

            if action == 2 and move_timestep == 0:
                move_timestep = timesteps
                #print('ep', ep, 'move timestep', move_timestep, 'action', action)

            
            total_reward += reward
            next_state = env.get_state()

            
            if env.DONE == 1:
                break
            if timesteps > 199:
                break

            


            timesteps += 1
            state = next_state
        
        if timesteps < 199:
            termtimesteps.append(timesteps)
        if move_timestep > 0:
            move_timesteps.append(move_timestep)
        

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(termtimesteps, label="Harvest")
    ax[0].legend()
    ax[1].plot(move_timesteps, label="Move")
    ax[1].legend()
    
    plt.show()


if __name__ == '__main__':
    main()

