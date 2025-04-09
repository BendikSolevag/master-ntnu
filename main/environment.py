import numpy as np
import torch
from torch import nn

# Seasonal mean function
def theta(t, a, b, phi):
    return a + b * np.sin(2.0 * np.pi * (t / 52.0) + phi)

class GrowthNN(nn.Module):
    def __init__(self, input_size):
        super(GrowthNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class SalmonFarmEnv:
    """
    Toy environment for salmon farming with state = (m, N, L, X, P).
      - m (float): average mass of fish (kg)
      - N (int): number of fish
      - L (float): approximate lice population
      - X (0 or 1): whether currently in treatment
      - P (0 or 1): pen type (0=closed, 1=open)
    Actions:
      - 0: do nothing
      - 1: treat
      - 2: move pen to open
      - 3: harvest (terminal)
    """
    def __init__(self,
                 max_time=20,
                 max_fish=1000,
                 fish_price=5.0,
                 cost_closed=1.0,
                 cost_open=0.5,
                 cost_treatment=1e5,
                 cost_move=0.5,
                 cost_feed=1e-10,
                 cost_harvest=1e5,
                 discount=0.99, 
                 time_step_size=1/52):
        
        N_ZERO=1500
        G_ZERO=0.5       
        L_ZERO=150

        # State variables
        self.NUMBER = N_ZERO
        self.GROWTH = G_ZERO
        self.LICE = L_ZERO

        self.PRICE_GENERATOR = schwartz_two_factor_generator(100, 0.01, 0.045, 0.01, 0.1, 0.05, 0.2, 0.2, 0.8, time_step_size)
        self.PRICE = next(self.PRICE_GENERATOR)

        self.TREATING = 0
        self.MOVED = 0
        self.DONE = 0

        # Constants
        self.time_step_size = time_step_size
        self.max_time = max_time
        self.max_fish = max_fish
        self.fish_price = fish_price
        self.cost_closed = cost_closed
        self.cost_open = cost_open
        self.cost_treatment = cost_treatment
        self.cost_move = cost_move
        self.cost_feed = cost_feed
        self.cost_harvest = cost_harvest
        self.discount = discount
        self.action_space = 3  # Do nothing, Move, Harvest

        # Growth rates
        # Salmon weight maximum
        self.G_max = 7
        # Salmon Mortality Treatment
        self.kappa_SMT = 1
        # Growth Increase
        self.kappa_GI = 0.1
        # Growth Decrease
        self.kappa_GD = 0.5
        
        
        # Lice SDE
        self.lice_kappa, self.lice_a, self.lice_b, self.lice_phi, self.lice_sigma, self.lice_t = 0.56451781,  0.17984971,  0.05243226, -0.62917791, 0.25959416, 0
        self.LICE_TREAT_THRESHOLD = 0.5

        # Utility variables
        self.sliding_window_max = round((2/52)/self.time_step_size)
        self.sliding_window_lice = [0 for _ in range(self.sliding_window_max)]

        # Growth rate NN
        self.growth_model = GrowthNN(input_size=4)
        self.growth_model.load_state_dict(torch.load('./models/growth/1743671011.288821-model.pt', weights_only=True))
        self.growth_model.eval()



    def step(self, action: int) -> tuple[tuple[float, float, float, float, int, float], float, bool]:
        # Set reward equal zero
        reward = 0.0
        # Iterate price to next value:
        self.PRICE = next(self.PRICE_GENERATOR)
        self.lice_t += 1/52
        
        if self.DONE:
            raise ValueError("Episode already done. Reset the environment.")
        
        if action == 1:
            reward += self.cost_move
            self.MOVED = 1

        # If action is harvest => immediate reward, episode ends
        if action == 2:
            harvest_revenue = self.GROWTH * self.NUMBER * self.PRICE
            reward += harvest_revenue
            reward -= self.cost_harvest
            self.DONE = True
            return reward, self.DONE


        #
        # Update state variables
        #

        # Lice
        seasonal_mean = theta(self.lice_t, self.lice_a, self.lice_b, self.lice_phi)
        self.LICE = (1 - self.lice_kappa) * self.LICE + self.lice_kappa * seasonal_mean + np.random.normal(0, self.lice_sigma)**2


        # Treatment
        self.sliding_window_lice.pop(0)
        self.sliding_window_lice.append(self.LICE)
        window_exceeds = [True if x > self.LICE_TREAT_THRESHOLD else False for x in self.sliding_window_lice]
        self.TREATING = 1.0 if any(window_exceeds) else 0.0

        # Population
        self.NUMBER += -(self.kappa_SMT*self.TREATING)*self.NUMBER*self.time_step_size

        # Weight
        explanatory = torch.tensor(
            [
            self.TREATING, #badebehandling_in_month, 
            self.TREATING, #forbehandling_in_month, 
            self.TREATING, #mekanisk_in_month, 
            round(self.lice_t), #generation_approx_age, 
            self.GROWTH * 0.015 * (1-self.TREATING), #feedamountperfish, 
            self.GROWTH, #mean_size,
            self.LICE, #mean_voksne_hunnlus,
            ], dtype=torch.float32
        )




        explanatory = [
                round(i / 52), #generation_approx_age, 
                self.GROWTH * 0.015 * 30, #feedamountperfish, 
                self.GROWTH, #mean_size,
                0, #mean_voksne_hunnlus,
            ]
        pred = self.growth_model.forward(torch.tensor(explanatory, dtype=torch.float32)).item()
        # Adjust monthly to weekly
        g_rate = np.log(pred / self.GROWTH) / 4.345

        
        self.GROWTH *= np.exp(g_rate * np.sqrt(1 - (self.GROWTH / 8)))

        

        #
        # Apply costs
        #
        cost_operation = self.cost_closed if self.MOVED == 0 else self.cost_open
        reward -= cost_operation

        cost_treatment = self.cost_treatment if self.TREATING else 0
        reward -= cost_treatment

        cost_feed = 0.015 * self.GROWTH * self.NUMBER * self.cost_feed
        reward -= cost_feed

        # Reset window of treatment occurs in current timestep (threshold reahed 2 weeks ago)
        if self.sliding_window_lice[0] > self.LICE_TREAT_THRESHOLD:
          self.LICE = 0.1 * self.LICE + 0.9 * self.LICE * np.random.beta(2.5, 5)
          self.sliding_window_lice = [0 for _ in range(self.sliding_window_max)]
          self.sliding_window_lice[-1] = self.LICE/self.NUMBER

        # If population reaches 0, force episode to end
        if self.NUMBER < 1:
            self.DONE = True

        return reward, self.DONE
    



def schwartz_two_factor_generator(P0, delta0, r, lambda_, kappa, alpha, sigma1, sigma2, rho, dt):
    """
    P0      : Initial spot price.
    delta0  : Initial convenience yield.
    r       : Risk-free rate.
    lambda_ : Expected return on the spot price.
    kappa   : Mean reversion speed of the convenience yield.
    alpha   : Long-term mean of the convenience yield.
    sigma1  : Volatility of the spot price.
    sigma2  : Volatility of the convenience yield.
    rho     : Correlation between spot price and convenience yield.
    dt      : Time step size.
    """

    # Initial values
    P_t = P0
    delta_t = delta0

    # Cholesky decomposition for correlated Brownian motions
    L = np.linalg.cholesky([[1, rho], [rho, 1]])

    while True:
        # Generate correlated Brownian motion increments
        dW = np.random.randn(2) * np.sqrt(dt)
        dZ1, dZ2 = L @ dW

        # Update convenience yield (Ornstein-Uhlenbeck process)
        delta_t += kappa * (alpha - delta_t) * dt + sigma2 * dZ2

        # Update spot price (Geometric Brownian motion with convenience yield)
        P_t *= np.exp((lambda_ - delta_t - 0.5 * sigma1 ** 2) * dt + sigma1 * dZ1)

        yield P_t 
    
