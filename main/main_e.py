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
    #agent.Q_eval.load_state_dict(T.load('./q.pt'))
    
    termtimesteps = []
    movetimesteps = []
    totalrewards = []
    running_return = []
    top_run_return = 0
    for ep in tqdm(range(10000)):
        total_reward = 0
        timesteps = 0
        env = SalmonFarmEnv(infinite=False)
        state = env.get_state()

        move_timestep = -1

        epsilon = np.random.uniform() < 0.1
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

        running_return.append(total_reward)
        if len(running_return) > 50:
            running_return.pop(0)
        
        
        mean_run_return = np.mean(running_return)
        
        if mean_run_return > top_run_return and len(running_return) > 48:
            top_run_return = mean_run_return
            print('ep', ep, 'new top return', top_run_return)
            T.save(agent.Q_eval.state_dict(), './q.pt')
        
            


    fix, ax = plt.subplots(3, 1)
    ax[0].plot(termtimesteps)
    ax[1].plot(movetimesteps)
    ax[2].plot(totalrewards)
    

    plt.show()


if __name__ == '__main__':
    main()

