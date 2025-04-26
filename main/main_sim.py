import numpy as np
from tqdm import tqdm
from environment import SalmonFarmEnv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch


def main():
    bottom_interval = 30
    top_interval = 120
    mc_sim_count = 200

    if True:
        env = SalmonFarmEnv(infinite=False)
        rewards_dict = {}
        pbar = tqdm(total=5995 * mc_sim_count)
        for harvest_timestep in range(bottom_interval, top_interval):
            rewards_dict[harvest_timestep] = {}
            for move_timestep in range(bottom_interval, harvest_timestep):
                
                total_rewards = []
                for _ in range(mc_sim_count):
                    pbar.update(1)

                    total_reward = 0            
                    timesteps = 0
                    env = SalmonFarmEnv(infinite=False)

                    while True:                
                        action = 0
                        if timesteps == move_timestep:
                            action = 2
                        if timesteps == harvest_timestep:
                            action = 3
                        reward, done = env.step(action)
                        total_reward += reward
                        timesteps += 1

                        if action == 3:
                            break

                    total_rewards.append(total_reward)
                rewards_dict[harvest_timestep][move_timestep] = total_rewards
        torch.save(rewards_dict, './assets/sim_rewards/sim_rewards_dict.pt')
        
    rewards_dict = torch.load('./sim_rewards_dict.pt')


    xs = [i for i in range(bottom_interval, top_interval)]
    ys = []
    for h_t in rewards_dict:
        maxrew = 0    
        for m_t in rewards_dict[h_t]:
            rewards_h_m = rewards_dict[h_t][m_t]
            
            mean = np.mean(rewards_h_m)
            if mean > maxrew:
                maxrew = mean
        ys.append(maxrew)
        
    
    plt.plot(xs, ys)
    plt.title("Best achievable reward at harvest timestep")
    plt.xlabel("Harvest timestep")
    plt.ylabel("Max mean reward")
    plt.savefig('./illustrations/analysis/top_reward_at_harvest_timestep.png', format="png", dpi=600)
    plt.close()

    top_rew_index = np.argmax(ys)
    top_harvest_timestep = xs[top_rew_index]
    top_harvest_dict = rewards_dict[top_harvest_timestep]
    xs = [i for i in range(bottom_interval, len(top_harvest_dict) + bottom_interval)]
    ys = []
    for m_t in top_harvest_dict:
        mean = np.mean(top_harvest_dict[m_t])
        ys.append(mean)
    
    plt.plot(xs, ys)
    plt.title("Best achievable reward at move timestep")
    plt.xlabel("Move timestep")
    plt.ylabel("Max mean reward")
    plt.savefig('./illustrations/analysis/top_reward_at_move_timestep.png', format="png", dpi=600)
    plt.close()


    

if __name__ == '__main__':
    main()    
