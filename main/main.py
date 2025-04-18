import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment import SalmonFarmEnv
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import Actor, Critic


def main():
    env = SalmonFarmEnv(infinite=True)
    state = torch.tensor(env.get_state(), dtype=torch.float32)
    

    actor = Actor(3, len(state))
    critic = Critic(len(state))
    R_bar = torch.tensor(0.0)


    num_episodes = 1000
    max_steps_per_episode = 200

    
    accumulative_reward = 0
    s = []
    actions = []

    
    for ep in tqdm(range(num_episodes * max_steps_per_episode)):

        
        out = actor(state)
        probs = torch.distributions.Categorical(out)
        action = probs.sample()
        actions.append(action)
        if len(actions) > 500:
            actions.pop(0)
        log_probs = probs.log_prob(action)

        print(action)
        reward, done = env.step(action)

        print(reward)
        return
        accumulative_reward += reward
        
        


        next_state = torch.tensor(env.get_state(), dtype=torch.float32)
        
        delta = reward - R_bar + 0.99 * critic(next_state) - critic.forward(state)    
        R_bar = ((1 - 1e-4) * R_bar + 1e-4 * delta).detach()

        critic_loss = delta**2
        actor_loss = -torch.sum(log_probs) * delta
        combined = actor_loss + critic_loss
        combined.backward()

        actor.opt.step()
        critic.opt.step()
        actor.opt.zero_grad()
        critic.opt.zero_grad()


        state = next_state

    plt.plot(actions)
    plt.show()        

    return
    agent.q_net.load_state_dict(torch.load(f'./models/agent/{top_timestamp}-q_net.pt', weights_only=True))
    agent.target_net.load_state_dict(torch.load(f'./models/agent/{top_timestamp}-target_net.pt', weights_only=True))
    env = SalmonFarmEnv(infinite=False)

    state = np.array(env.get_state())
    test_reward = 0
    step = 0
    closed_pen_hist = []
    open_pen_hist = []
    while env.DONE != 1:
        action = agent.act(state)
        reward, done = env.step(action)
        test_reward += reward
        state = np.array(env.get_state())
        step += 1
        closed_pen_hist.append(env.GROWTH_CLOSED)
        open_pen_hist.append(env.GROWTH_OPEN)
    print("Test episode reward")
    print(env.lice_t)
    print(test_reward)

    plt.plot(closed_pen_hist, label="Closed pen history")
    plt.plot(open_pen_hist, label="Open pen history")
    plt.legend()
    plt.savefig('growth_histories.png', format="png", dpi=800)
    plt.close()



if __name__ == "__main__":
    
    main()

    """
    data = []

    pbar = tqdm(total=(55-3) * 1000)
    
    for l in range(3, 55):
        lll = []
        for _ in range(1000):
            pbar.update(1)
            length = l * 2
            env = SalmonFarmEnv(infinite=False)
            total_r = 0
            for _ in range(int(length / 2)):
                r, d = env.step(0)
                total_r += r
            r, d = env.step(2)
            total_r += r
            for _ in range(int(length / 2)):
                r, d = env.step(0)
                total_r += r
            r, d = env.step(3)
            total_r += r
            lll.append(total_r)
        data.append(lll)    
    
    # Compute the mean and standard error for each list
    means = np.array([np.mean(d) for d in data])
    margin_errors_top = []
    margin_errors_bot = []
    for i in range(len(data)):
        lll = data[i]
        lll = sorted(lll)
        margin_errors_bot.append(means[i] - lll[50])
        margin_errors_top.append(lll[950] - means[i])

    # Calculate the margin of error for a 95% confidence interval (using 1.96 as the z-value)
    margin_errors = [margin_errors_top, margin_errors_bot]
    print(len(means))
    print(len(margin_errors_top))
    print(len(margin_errors_bot))

    # Prepare the x-axis values (e.g., index of each list)
    x = np.arange(1, len(data) + 1)
    x = x * 2

    # Plot using errorbar
    plt.errorbar(x, means, yerr=margin_errors, fmt='o', capsize=5, linestyle='none')
    plt.xlabel('Cycle length')
    plt.ylabel('Mean Value')
    plt.title('95% Confidence Intervals for Each List')
    plt.savefig('./environment_dynamics.png', format="png", dpi=500)
    plt.show()
    plt.close()
    """

