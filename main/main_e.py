import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as T
from environment import SalmonFarmEnv
from tqdm import tqdm
import numpy as np
from agents.q_learner import Agent



def main(closed_coefficient):
    print('closed_coefficient', closed_coefficient)
    

    env = SalmonFarmEnv(closed_coefficient=closed_coefficient, infinite=False)
    state = T.tensor(env.get_state(), dtype=T.float32)
    
    agent = Agent(gamma=0.99956, lr=0.00001, input_dims=[len(state)], batch_size=4, n_actions=4)
    #agent.Q_eval.load_state_dict(T.load('./q.pt'))
    
    termtimesteps = []
    movetimesteps = []
    totalrewards = []
    running_return = []
    top_run_return = 0
    maxep = 0.5
    episodes = 10000
    
    for ep in tqdm(range(episodes)):
        total_reward = 0
        timesteps = 0
        env = SalmonFarmEnv(closed_coefficient=closed_coefficient, infinite=False)
        state = env.get_state()

        move_timestep = -1

        epsilon = np.random.uniform() < (maxep - maxep * (ep/episodes))
        htstep = np.random.randint(40, 130)
        mtstep = np.random.randint(0, htstep)

        while True:

            action = agent.choose_action(state)

            if epsilon:
                if timesteps == htstep:
                    action = 1
                if timesteps == mtstep:
                    action = 2
            
            if timesteps < 20 and action == 1:
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



            if env.DONE == 1:
                break
            if timesteps > 199:

                break


            timesteps += 1
            state = next_state

        
        #print('env.total_cost_operation_open', env.total_cost_operation_open)
        #print('env.total_cost_operation_closed', env.total_cost_operation_closed)
        #print('env.total_cost_feed', env.total_cost_feed)
        #print('env.total_cost_harvest', env.total_cost_harvest)
        #print('env.total_cost_treatment', env.total_cost_treatment)
        #print('env.total_num_treatment', env.total_num_treatments)
        #print(timesteps)
        #print(move_timestep)
            
        
        
        
        movetimesteps.append(move_timestep)
        termtimesteps.append(timesteps)
        totalrewards.append(total_reward)


        # Save best performing model to use later
        running_return.append(total_reward)
        if len(running_return) > 50:
            running_return.pop(0)
        mean_run_return = np.mean(running_return)
        if mean_run_return > top_run_return and len(running_return) > 48:
            top_run_return = mean_run_return
            #print('ep', ep, 'new top return', top_run_return)
            T.save(agent.Q_eval.state_dict(), f'./models/agent/episodic/q-{closed_coefficient}.pt')

    #fig, ax = plt.subplots(3, 1)
    #ax[0].plot(termtimesteps)
    #ax[1].plot(movetimesteps)
    #ax[2].plot(totalrewards)    
    #plt.show()

    T.save(termtimesteps, './ttsteps.pt')
    T.save(movetimesteps, './tmoves.pt')
    T.save(totalrewards, './trews.pt')
    
    return
    harvest_timesteps = []
    move_timesteps = []
    total_rewards = []
    agent.Q_eval.load_state_dict(T.load(f'./models/agent/episodic/q-{closed_coefficient}.pt'))
    for _ in range(1000):
        env = SalmonFarmEnv(closed_coefficient=closed_coefficient, infinite=False)
        total_reward = 0
        timesteps = 0
        move_timestep = -1
        while True:


            action = agent.choose_action(state)

            if timesteps < 20 and action == 1:
                action = 0
            
            if action == 2 and move_timestep == -1:
                move_timestep = timesteps
            

            
            reward, done = env.step(action)
            reward = reward / 1e7
                        
            total_reward += reward
            next_state = env.get_state()

            if env.DONE == 1:
                break
            if timesteps > 199:
                break
            timesteps += 1
            state = next_state
        harvest_timesteps.append(timesteps)
        move_timesteps.append(move_timestep)
        total_rewards.append(total_reward)
    
    #T.save(harvest_timesteps, f'./data/assets/simulated/{closed_coefficient}/harvest_timesteps.pt')
    #T.save(move_timesteps, f'./data/assets/simulated/{closed_coefficient}/move_timesteps.pt')
    #T.save(total_rewards, f'./data/assets/simulated/{closed_coefficient}/total_rewards.pt')

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(harvest_timesteps)
    ax[1].plot(move_timesteps)
    ax[2].plot(total_rewards)
    
    

    plt.show()


if __name__ == '__main__':
    #for coef in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 40, 64]:
    for coef in [4]:
        main(coef)

