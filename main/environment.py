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
                 infinite=True,
                 max_time=20,
                 max_fish=1000,
                 fish_price=5.0,
                 cost_closed=1.0,
                 cost_open=0.5,
                 cost_treatment=1e5,
                 cost_plant=1e3,
                 cost_move=1e5,
                 cost_feed=0.14 * 1e-4,
                 cost_harvest=1e5,
                 discount=0.99, 
                 time_step_size=1/52):
        
        N_ZERO=150000
        G_ZERO=0.2       
        L_ZERO=150
        self.infinite = infinite

        # State variables
        self.N_ZERO = N_ZERO
        self.G_ZERO = G_ZERO
        self.NUMBER_CLOSED = N_ZERO
        self.NUMBER_OPEN = 0
        self.GROWTH_CLOSED = G_ZERO
        self.GROWTH_OPEN = 0
        self.LICE = L_ZERO
        self.AGE_CLOSED = 0
        self.AGE_OPEN = 0

        self.PRICE_GENERATOR = schwartz_two_factor_generator(100, 0.01, 0.045, 0.01, 0.1, 0.05, 0.2, 0.2, 0.8, time_step_size)
        self.PRICE = next(self.PRICE_GENERATOR)

        self.TREATING = 0
        self.DONE = 0

        # Constants
        self.time_step_size = time_step_size
        self.max_time = max_time
        self.max_fish = max_fish
        self.fish_price = fish_price
        self.cost_closed = cost_closed
        self.cost_open = cost_open
        self.cost_treatment = cost_treatment
        self.cost_plant = cost_plant
        self.cost_move = cost_move
        self.cost_feed = cost_feed
        self.cost_harvest = cost_harvest
        self.discount = discount
        self.action_space = 4  # Do nothing, Plant, Move, Harvest
        self.max_biomass = N_ZERO * 6.5 # Assume max biomass is hit when one generation hits 6.5 kg
        
        self.feed_per_fish = 0.015
        
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

    def get_state(self):
        return [self.PRICE, self.LICE, self.GROWTH_CLOSED, self.NUMBER_CLOSED, self.GROWTH_OPEN, self.NUMBER_OPEN, self.TREATING]

    def resolve_mortalityrate(self):
        categorydraw = np.random.uniform()
        if categorydraw >= 0.999**2:
            # gt 50% mortality
            return np.random.uniform(0.5, 0.9)
        if categorydraw >= 0.995**2:
            # gt 25% mortality
            return np.random.uniform(0.25, 0.5)
        if categorydraw >= 0.989**2:
            # gt 10% mortality
            return np.random.uniform(0.1, 0.25)
        if categorydraw >= 0.975*0.968:
            # gt 5% mortality
            return np.random.uniform(0.05, 0.1)
        if categorydraw >= 0.948*0.937:
            # gt 2.5% mortality
            return np.random.uniform(0.025, 0.05)
        if categorydraw >= 0.894*0.853:
            # gt 1% mortality
            return np.random.uniform(0.01, 0.025)
        return 0

    def step(self, action: int) -> tuple[tuple[float, float, float, float, int, float], float, bool]:
        # Set reward equal zero
        reward = 0.0
        # Iterate price to next value:
        self.PRICE = next(self.PRICE_GENERATOR)
        self.lice_t += 1/52
        self.AGE_CLOSED += 1/52
        self.AGE_OPEN += 1/52
        
        if self.DONE:
            raise ValueError("Episode already done. Reset the environment.")
        
        # If action is plant => add new generation to closed pool
        if action == 1:
            if self.infinite:
                # One could argue a penalty should be incurred if planting into a pen where fish already exist. Instead we rely on the repeated incurred feed/operating cost to do this.
                self.NUMBER_CLOSED = self.N_ZERO
                self.GROWTH_CLOSED = self.G_ZERO
                self.AGE_CLOSED = 0
            reward -= self.cost_plant * self.N_ZERO

        # If action is move => move closed individuals to open, reset closed
        if action == 2:
            reward -= self.cost_move
            self.NUMBER_OPEN = self.NUMBER_CLOSED
            self.GROWTH_OPEN = self.GROWTH_CLOSED
            self.AGE_OPEN = self.AGE_CLOSED
            self.NUMBER_CLOSED = 0
            self.GROWTH_CLOSED = 0
            self.AGE_CLOSED = 0

        # If action is harvest => give reward, reset open
        if action == 3:
            harvest_revenue = self.GROWTH_OPEN * self.NUMBER_OPEN * self.PRICE
            reward += harvest_revenue
            reward -= self.cost_harvest
            self.GROWTH_OPEN = 0
            self.NUMBER_OPEN = 0
            self.AGE_OPEN = 0
            self.DONE = 1
            
            if not self.infinite:
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

        # Population loss due to treatment
        if self.sliding_window_lice[0] > self.LICE_TREAT_THRESHOLD:
            mr = self.resolve_mortalityrate()
            self.NUMBER_OPEN = self.NUMBER_OPEN - mr * self.NUMBER_OPEN

        #
        # Grow population in open/closed system
        #

        if self.NUMBER_CLOSED > 0:
            # Closed
            explanatory = [
                round(self.AGE_CLOSED), #generation_approx_age, 
                self.GROWTH_CLOSED * self.feed_per_fish * 30, #feedamountperfish, 
                self.GROWTH_CLOSED, #mean_size,
                0, #mean_voksne_hunnlus, 0 as the system is closed
            ] 
            pred = self.growth_model.forward(torch.tensor(explanatory, dtype=torch.float32)).item()
            # Adjust monthly to weekly
            g_rate = np.log(pred / self.GROWTH_CLOSED) / 4.345
            greater_factor = (8 - g_rate) / np.abs(8 - g_rate) 
            self.GROWTH_CLOSED *= np.exp(g_rate * greater_factor * np.sqrt(np.abs(1 - (self.GROWTH_CLOSED / 8))))

        if self.NUMBER_OPEN > 0 and self.TREATING == 0.0:
            # Open
            explanatory = [
                round(self.AGE_OPEN), #generation_approx_age, 
                self.GROWTH_OPEN * self.feed_per_fish * 30, #feedamountperfish, 
                self.GROWTH_OPEN, #mean_size,
                self.LICE, #mean_voksne_hunnlus,
            ] 
            pred = self.growth_model.forward(torch.tensor(explanatory, dtype=torch.float32)).item()
            # Adjust monthly to weekly
            g_rate = np.log(pred / self.GROWTH_OPEN) / 4.345
            greater_factor = (8 - g_rate) / np.abs(8 - g_rate) 
            self.GROWTH_OPEN *= np.exp(g_rate * greater_factor * np.sqrt(np.abs(1 - (self.GROWTH_OPEN / 8))))        


        #
        # Apply costs
        #
        cost_operation = 0
        if self.NUMBER_CLOSED > 0:
            cost_operation += self.cost_closed
        if self.NUMBER_OPEN > 0: 
            cost_operation += self.cost_open
        reward -= cost_operation

        cost_treatment = self.cost_treatment if self.TREATING else 0
        reward -= cost_treatment

        cost_feed_closed = self.feed_per_fish * self.GROWTH_CLOSED * self.NUMBER_CLOSED * self.cost_feed
        cost_feed_open = self.feed_per_fish * self.GROWTH_OPEN * self.NUMBER_OPEN * self.cost_feed
        reward -= (cost_feed_closed + cost_feed_open)

        # If max biomass is exceeded, punish reward
        if self.NUMBER_OPEN * self.GROWTH_OPEN + self.NUMBER_CLOSED * self.GROWTH_CLOSED >= self.max_biomass:
            reward -= 1e8

        # Reset window of treatment occurs in current timestep (threshold reahed 2 weeks ago)
        if self.sliding_window_lice[0] > self.LICE_TREAT_THRESHOLD:
          self.LICE = 0.1 * self.LICE + 0.9 * self.LICE * np.random.beta(2.5, 5)
          self.sliding_window_lice = [0 for _ in range(self.sliding_window_max)]
          if self.NUMBER_OPEN > 0:  
            self.sliding_window_lice[-1] = self.LICE/self.NUMBER_OPEN

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
        #dZ1, dZ2 = 0, 0
        

        # Update convenience yield (Ornstein-Uhlenbeck process)
        delta_t += kappa * (alpha - delta_t) * dt + sigma2 * dZ2

        # Update spot price (Geometric Brownian motion with convenience yield)
        P_t *= np.exp((lambda_ - delta_t - 0.5 * sigma1 ** 2) * dt + sigma1 * dZ1)

        yield P_t 
    
