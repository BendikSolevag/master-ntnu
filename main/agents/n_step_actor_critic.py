import collections
import torch as T
import torch.nn.functional as F
from torch import nn
from torch import optim

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi  = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=lr*0.01)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.pi(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v   = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=lr*0.01)
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)

class Agent:
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, gamma=0.999135, n_step=200):
        self.gamma = gamma
        self.n_step = n_step
        self.net_actor = ActorNetwork(lr, input_dims, n_actions, fc1_dims, fc2_dims)
        self.net_critic = CriticNetwork(lr, input_dims, fc1_dims, fc2_dims)
        self.buffer = collections.deque(maxlen=n_step)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.net_actor.device)
        logits = self.net_actor(state)
        probs = F.softmax(logits, dim=1)
        dist  = T.distributions.Categorical(probs)
        action = dist.sample().item()
        return action
        
        
    def _flush_buffer(self, next_state, done):
        """
        Perform one gradient update for the *oldest* transition in the buffer
        using an n-step bootstrapped return.
        """
        # oldest transition we are updating now
        state, action, _ = self.buffer[0]

        # recompute policy log probs with current network parameters
        s = T.tensor([state], dtype=T.float32, device=self.net_actor.device)
        logits = self.net_actor(s)
        dist = T.distributions.Categorical(F.softmax(logits, dim=1))
        logp = dist.log_prob(T.tensor([action], device=self.net_actor.device))

        # estimated value from state_{t+n} (0 if terminal)
        with T.no_grad():
            ns = T.tensor([next_state], dtype=T.float32, device=self.net_critic.device)
            v_approx = self.net_critic(ns)
            v_approx *= (1 - int(done))

        # n-step discounted return G_t, ending with approximated value v
        G = v_approx
        for _, _, r in reversed(self.buffer):
            G = r + self.gamma * G

        # critic estimate current state V(s_t)
        v = self.net_critic(s)
        delta = G - v

        # update
        self.net_actor.optimizer.zero_grad()
        self.net_critic.optimizer.zero_grad()
        actor_loss  = -logp * delta.detach()   # stop grad into critic
        critic_loss = delta.pow(2)
        (actor_loss + critic_loss).backward()
        
        self.net_actor.optimizer.step()
        self.net_critic.optimizer.step()



    def learn(self, state, action, reward, next_state, done):
        """Store transition and, when appropriate, perform a learning step."""
        self.buffer.append((state, action, reward))

        # when buffer full or episode ends → learn & discard oldest item
        if len(self.buffer) == self.n_step or done:
            self._flush_buffer(next_state, done)
            self.buffer.popleft()

        # on episode end there may still be items left to learn from
        if done:
            while self.buffer:
                self._flush_buffer(next_state, done=True)
                self.buffer.popleft()