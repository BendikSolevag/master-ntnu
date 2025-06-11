
from environment import SalmonFarmEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents.q_learner_inf import Agent
import torch as T



def main(closed_coefficient):
    
    env = SalmonFarmEnv(infinite=True, closed_coefficient=closed_coefficient)
    state = env.get_state()
    
    agent = Agent(gamma=0.995, lr=0.000001, input_dims=[len(state)], batch_size=4, n_actions=4)
    #agent.Q_eval.load_state_dict(T.load(f'./models/agent/episodic/q-{closed_coefficient}.pt'))
    state = env.get_state()
    
    htstep = 60
    mtstep = 23
    ptstep = 40

    
    episodes = 1
    max_tsteps = 500000
    pbar = tqdm(total=episodes*max_tsteps)


    trews = []

    top_avg_period = -1 * float('inf')

    for ep in range(episodes):
        env = SalmonFarmEnv(infinite=True, closed_coefficient=closed_coefficient)
        total_reward = 0
        open_growthlist = []
        closed_growthlist = []
        open_agelist = []
        closed_agelist = []
        actionlist = []
        epsilon = True
        r_window = []
        for ts in range(max_tsteps):
            pbar.update(1)

            action = agent.choose_action(state)
            
            if len(r_window) > 200:
                r_window.pop(0)

            # One case for if generation has unharvested in move, one otherwise?
            if epsilon: #ep < 0.99 * max_tsteps:
                action = 0
                if round(env.AGE_CLOSED, 6) == round(mtstep * 1/52, 6):
                    action = 2
                if round(env.AGE_OPEN, 6) == round(ptstep * 1/52, 6):
                    action = 3
                if round(env.AGE_OPEN, 6) == round(htstep * 1/52, 6):
                    action = 1


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
            total_reward += reward
            r_window.append(reward)
            

            # If age_open or age_closed greater than 2, punish (repeat success from episodic)
            if env.AGE_OPEN > 2 or env.AGE_CLOSED > 2:
                reward -= 10
            
            
            next_state = env.get_state()
            r_bar = np.mean(r_window)
            agent.store_transition(state, action, reward, r_bar, next_state, done)
            agent.learn()
            
            if action == 2:# and ep < 0.99 * max_tsteps:
                epsilon = np.random.uniform() < 0.1
                htstep = np.random.randint(50, 120)            
                ptstep = np.random.randint(htstep / 2, htstep-1)
                mtstep = np.random.randint(0, ptstep)
                

            state = next_state
            open_growthlist.append(env.GROWTH_OPEN)
            closed_growthlist.append(env.GROWTH_CLOSED)
            open_agelist.append(env.AGE_OPEN)
            closed_agelist.append(env.AGE_CLOSED)
            actionlist.append(action)

        
        
        if False:
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(open_growthlist)
            ax[0].plot(closed_growthlist)
            ax[1].plot(actionlist)
            ax[2].plot(open_agelist)
            ax[2].plot(closed_agelist)
            plt.show()
            return
        
        #print(actionlist)

        trews.append(total_reward)

        
        if len(trews) > 50:
            meanrew = np.mean(trews[-50:])
            if meanrew > top_avg_period:
                top_avg_period = meanrew
                T.save(agent.Q_eval.state_dict(), f'./models/agent/infhor/q-{closed_coefficient}.pt')
            
    #plt.plot(trews)
    #plt.ylabel("Total reward")
    #plt.xlabel("Episode")
    #plt.savefig('./illustrations/results/infhor/train-convergence.png', format="png")
    #plt.show()
    #plt.close()
    





if __name__ == "__main__":
    closed_coefficients = [1, 2, 3, 4, 6, 8, 10, 12]
    for coef in closed_coefficients:
        main(coef)
 