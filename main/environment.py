import numpy as np

def environment(N_zero=150000, G=0.2, delta_t=0.01):
  sliding_window_max = round((2/52)/delta_t)
  sliding_window_lice = [0 for _ in range(sliding_window_max)]


  N = N_zero
  L_MOVED = 15000
  L = 0

  # 
  moved = False
  
  # Lice per salmon threshold (treat when over)
  L_bar = 1

  # Salmon weight maximum
  G_max = 7

  # Constant which determines lice growth
  kappa_GL =  1

  # Constnt which determines lice mortality
  kappa_ML = 0.1

  # Constant which determines lice growth factor decrease per density
  kappa_LD = 0.2

  # Salmon Mortality Lice
  kappa_SML = 1e-10

  # Salmon Mortality Treatment
  kappa_SMT = 1

  # Growth Increase
  kappa_GI = 0.1
  
  # Growth Decrease
  kappa_GD = 0.5

  while True:
    # Update lice population
    L += (kappa_GL*(N/N_zero) - kappa_ML - kappa_LD*(L / N))*L*delta_t

    # Update lice history
    sliding_window_lice.pop(0)
    sliding_window_lice.append(L/N)

    # Determine if plant is in treatment stage
    window_exceeds = [True if x > L_bar else False for x in sliding_window_lice]
    T = 1 if any(window_exceeds) else 0

    # Update salmon population
    N += -(kappa_SMT*T)*N*delta_t

    # Update salmon weight
    G += (kappa_GI*(1-(G/G_max))*(1-T) - kappa_GD*T)*G*delta_t



    received = yield L, T, N, G

    if moved != received:
       L = L_MOVED
    if sliding_window_lice[0] > L_bar:
      L = 0.1 * L + 0.9 * L * np.random.beta(2.5, 5)
      sliding_window_lice = [0 for _ in range(sliding_window_max)]
      sliding_window_lice[-1] = L/N

    moved = received
    

def schwartz_two_factor_generator(P0, delta0, r, lambda_, kappa, alpha, sigma1, sigma2, rho, dt):
    """
    Generator function implementing the Schwartz (1997) two-factor model.

    Parameters:
    - P0      : Initial spot price.
    - delta0  : Initial convenience yield.
    - r       : Risk-free rate.
    - lambda_ : Expected return on the spot price.
    - kappa   : Mean reversion speed of the convenience yield.
    - alpha   : Long-term mean of the convenience yield.
    - sigma1  : Volatility of the spot price.
    - sigma2  : Volatility of the convenience yield.
    - rho     : Correlation between spot price and convenience yield.
    - dt      : Time step size.

    Yields:
    - The next spot price in the stochastic process.
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

        yield P_t  # Yield the next spot price
    
