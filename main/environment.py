import numpy as np
import torch as T

from util.growthmodel import GrowthNN


# Seasonal mean function
def theta(t, a, b, phi):
    return a + b * np.sin(2.0 * np.pi * (t / 52.0) + phi)

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
                 only_open=False,
                 closed_coefficient=1,
                 infinite=True,
                 max_time=20,
                 max_fish=1000,
                 fish_price=5.0,
                 cost_closed=5.1e3,
                 cost_open=5.1e3,
                 cost_treatment=1.5e5,
                 cost_plant=1e5,
                 cost_move=1e6,
                 cost_feed=5,
                 cost_harvest=7.3e5,
                 discount=0.99, 
                 time_step_size=1/52):
        
        N_ZERO=150000
        G_ZERO=0.2       
        self.infinite = infinite

        # State variables
        self.only_open = only_open
        self.N_ZERO = N_ZERO
        self.G_ZERO = G_ZERO
        self.NUMBER_CLOSED = N_ZERO
        self.NUMBER_OPEN = 0
        self.GROWTH_CLOSED = G_ZERO
        self.GROWTH_OPEN = 0
        self.LICE = 0.3
        self.AGE_CLOSED = 0
        self.AGE_OPEN = 0

        self.PRICE_GENERATOR = schwartz_two_factor_generator(100, 0.01, 0.045, 0.01, 0.1, 0.05, 0.2, 0.2, 0.8, time_step_size)
        self.PRICE = 100 #next(self.PRICE_GENERATOR)

        self.DONE = 0

        # Constants
        self.time_step_size = time_step_size
        self.max_time = max_time
        self.max_fish = max_fish
        self.fish_price = fish_price
        self.cost_closed = closed_coefficient * cost_closed
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

        # Growth rate NN
        self.growth_model = GrowthNN(input_size=4)
        self.growth_model.load_state_dict(T.load('./models/growth/1743671011.288821-model.pt', weights_only=True))
        self.growth_model.eval()

        # Track total costs
        self.total_cost_feed = 0
        self.total_cost_harvest = 0
        self.total_cost_operation_closed = 0
        self.total_cost_operation_open = 0
        self.total_cost_treatment = 0
        self.total_num_treatments = 0

    def get_state(self):
        return [self.GROWTH_CLOSED, np.log(self.NUMBER_CLOSED + 1), self.GROWTH_OPEN, np.log(self.NUMBER_OPEN + 1), self.LICE, np.log(self.PRICE)]

    def resolve_mortalityrate(self) -> float:
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
        return 0.0


    def resolve_lice(self):
        seasonal_mean = theta(self.lice_t, self.lice_a, self.lice_b, self.lice_phi)
        self.LICE = (1 - self.lice_kappa) * self.LICE + self.lice_kappa * seasonal_mean + np.random.normal(0, self.lice_sigma)**2


    def resolve_mortality(self):
        mr = self.resolve_mortalityrate()
        # Population loss due to treatment
        if self.NUMBER_OPEN > 0 and self.LICE > self.LICE_TREAT_THRESHOLD:
            
            self.NUMBER_OPEN = self.NUMBER_OPEN - mr * self.NUMBER_OPEN
        if self.only_open:
            self.NUMBER_CLOSED = self.NUMBER_CLOSED - mr * self.NUMBER_CLOSED
            

    def resolve_growth_closed(self):
        if self.only_open and self.LICE > self.LICE_TREAT_THRESHOLD:
            self.GROWTH_OPEN -= self.GROWTH_OPEN * 0.05
            return

        if self.NUMBER_CLOSED <= 0:
            return
        # Closed
        explanatory = [
            round(self.AGE_CLOSED), #generation_approx_age, 
            self.GROWTH_CLOSED * self.feed_per_fish * 30, #feedamountperfish, 
            self.GROWTH_CLOSED, #mean_size,
            0, #mean_voksne_hunnlus, 0 as the system is closed
        ] 
        pred = self.growth_model.forward(T.tensor(explanatory, dtype=T.float32)).item()
        # Cap prediction within reasonable range
        pred = max(min(pred, 8), 0.1)
        # Adjust monthly to weekly
        g_rate = np.log(pred / self.GROWTH_CLOSED) / 4.345
        self.GROWTH_CLOSED *= np.exp(g_rate)
    
    def resolve_growth_open(self):
        if self.LICE > self.LICE_TREAT_THRESHOLD:
            self.GROWTH_OPEN -= self.GROWTH_OPEN * 0.05
            return
        if self.NUMBER_OPEN <= 0:
            return
    
        explanatory = [
            round(self.AGE_OPEN), #generation_approx_age, 
            self.GROWTH_OPEN * self.feed_per_fish * 30, #feedamountperfish, 
            self.GROWTH_OPEN, #mean_size,
            self.LICE, #mean_voksne_hunnlus,
        ] 
        pred = self.growth_model.forward(T.tensor(explanatory, dtype=T.float32)).item()  
        # Cap prediction within reasonable range
        pred = max(min(pred, 8), 0.1)
        # Adjust monthly to weekly
        g_rate = np.log(pred / self.GROWTH_OPEN) / 4.345
        self.GROWTH_OPEN *= np.exp(g_rate)



    def step(self, action: int):
        # Set reward equal zero
        reward = 0.0
        # Iterate price to next value:
        self.PRICE = 100 #next(self.PRICE_GENERATOR)
        self.lice_t += 1/52
        if self.NUMBER_CLOSED > 0:
            self.AGE_CLOSED += 1/52
        if self.NUMBER_OPEN > 0:
            self.AGE_OPEN += 1/52
        
        if not self.infinite and self.DONE:
            raise ValueError("Episode already done. Reset the environment.")

        # If action is harvest => give reward, reset open
        if action == 1:
            harvest_revenue = self.GROWTH_OPEN * self.NUMBER_OPEN * self.PRICE
            reward += harvest_revenue
            reward -= self.cost_harvest
            self.total_cost_harvest += self.cost_harvest
            self.GROWTH_OPEN = 0
            self.NUMBER_OPEN = 0
            self.AGE_OPEN = 0
            
            if not self.infinite:
                self.DONE = 1
                return reward, self.DONE

            #reward -= self.cost_plant * self.N_ZERO

        # If action is move => move closed individuals to open, reset closed
        if action == 2:
            reward -= self.cost_move
            if self.NUMBER_CLOSED > 0:
                self.NUMBER_OPEN = self.NUMBER_CLOSED
                self.GROWTH_OPEN = self.GROWTH_CLOSED
                self.AGE_OPEN = self.AGE_CLOSED
                self.NUMBER_CLOSED = 0
                self.GROWTH_CLOSED = 0
                self.AGE_CLOSED = 0

        # If action is plant => add new generation to closed pool
        if action == 3:
            if self.infinite:
                # One could argue a penalty should be incurred if planting into a pen where fish already exist. Instead we rely on the repeated incurred feed/operating cost to do this.
                self.NUMBER_CLOSED = self.N_ZERO
                self.GROWTH_CLOSED = self.G_ZERO
                self.AGE_CLOSED = 0
            reward -= (self.cost_plant * self.N_ZERO) / 1000




        # Update state variables
        self.resolve_lice()
        self.resolve_mortality()
        self.resolve_growth_closed()
        self.resolve_growth_open()


        #
        # Apply costs
        #
        cost_operation = 0
        if not self.only_open and self.NUMBER_CLOSED > 0:
            cost_operation += self.cost_closed
            self.total_cost_operation_closed += cost_operation

        if self.NUMBER_OPEN > 0: 
            cost_operation += self.cost_open
            if self.only_open:
                cost_operation += self.cost_open
            self.total_cost_operation_open += cost_operation

        reward -= cost_operation

        
        cost_treatment = 0
        if self.LICE > self.LICE_TREAT_THRESHOLD and (self.NUMBER_OPEN > 0 or self.only_open):
            cost_treatment = self.cost_treatment
        reward -= cost_treatment
        self.total_cost_treatment += cost_treatment

        cost_feed_closed = self.feed_per_fish * self.GROWTH_CLOSED * self.NUMBER_CLOSED * self.cost_feed
        cost_feed_open = self.feed_per_fish * self.GROWTH_OPEN * self.NUMBER_OPEN * self.cost_feed
        
        reward -= (cost_feed_closed + cost_feed_open)
        self.total_cost_feed += (cost_feed_closed + cost_feed_open)

        # If max biomass is exceeded, punish reward
        if self.NUMBER_OPEN * self.GROWTH_OPEN + self.NUMBER_CLOSED * self.GROWTH_CLOSED >= self.max_biomass:
            #if action == 3:
            #    print(1e8)
            reward -= 3

        

        # Resolve next price
        self.PRICE = next(self.PRICE_GENERATOR)
        # Reset lice
        if self.LICE > self.LICE_TREAT_THRESHOLD:
            self.LICE = 0
            self.total_num_treatments += 1
        
        reward = reward

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
        P_t *= np.exp( sigma1 * dZ1)

        yield P_t 
    
