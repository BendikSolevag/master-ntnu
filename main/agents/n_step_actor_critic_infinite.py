import collections
import torch as T
import torch.nn.functional as F
from torch import nn
from torch import optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi  = nn.Linear(fc2_dims, n_actions)
        self.v   = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.pi(x), self.v(x)

class Agent:
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, gamma=0.99, n_step=200):
        self.gamma = gamma
        self.n_step = n_step
        self.net = ActorCriticNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims)
        self.buffer = collections.deque(maxlen=n_step)
        self.R_BAR = T.tensor(0, dtype=T.float32)
        

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.net.device)
        logits, _ = self.net(state)
        probs = F.softmax(logits, dim=1)
        dist  = T.distributions.Categorical(probs)
        action = dist.sample().item()
        return action

    def flush_buffer(self, next_state):
        """
        Perform one gradient update for the oldest transition in the buffer
        """
        # oldest transition we are updating now
        state, action, _ = self.buffer[0]

        # recompute log π(a|s) with current network parameters
        s = T.tensor([state], dtype=T.float32, device=self.net.device)
        logits, _ = self.net(s)
        dist = T.distributions.Categorical(F.softmax(logits, dim=1))
        logp = dist.log_prob(T.tensor([action], device=self.net.device))

        # critic estimate of n-th step state
        with T.no_grad():
            ns = T.tensor([next_state], dtype=T.float32, device=self.net.device)
            _, v_approx = self.net(ns)

        # n-step discounted return G_t, ending with approximated value v_approx
        G = v_approx
        for _, _, r in reversed(self.buffer):
            G = r - self.R_BAR.clone().detach() + self.gamma * G

        # critic estimate current state V(s_t)
        _, v = self.net(s)
        delta = G - v

        tmp = 1 / 300
        self.R_BAR = self.R_BAR * (1-tmp) + tmp * delta

        
        self.net.optimizer.zero_grad()
        actor_loss  = -logp * delta.detach()   # stop grad into critic
        critic_loss = delta.pow(2)
        (actor_loss + critic_loss).backward()
        self.net.optimizer.step()


    def learn(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward))

        # when buffer full or episode ends → learn & discard oldest item
        if len(self.buffer) == self.n_step:
            self.flush_buffer(next_state)
            self.buffer.popleft()
