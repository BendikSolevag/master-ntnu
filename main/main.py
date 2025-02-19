import matplotlib.pyplot as plt
from environment import schwartz_two_factor_generator, environment

def main():

  # Parameters
  P0 = 50          # Initial spot price
  delta0 = 0.02    # Initial convenience yield
  r = 0.03         # Risk-free rate
  lambda_ = 0.05   # Expected return on spot price
  kappa = 0.1      # Mean reversion speed of convenience yield
  alpha = 0.02     # Long-term mean of convenience yield
  sigma1 = 0.2     # Volatility of spot price
  sigma2 = 0.1     # Volatility of convenience yield
  rho = 0.5        # Correlation between spot price and convenience yield
  dt = 1/100       # Time step (0.01 years)

  # Create generator instance
  price_generator = schwartz_two_factor_generator(P0, delta0, r, lambda_, kappa, alpha, sigma1, sigma2, rho, dt)

  environment_generator = environment()
  L, T, N, G = next(environment_generator)
  

  

  L_hist = []
  T_hist = []
  N_hist = []
  G_hist = []

  # Generate and print the next 10 spot prices
  for i in range(100):
    L, T, N, G = environment_generator.send(i > 50)
    L_hist.append(L)
    T_hist.append(T)
    N_hist.append(N)
    G_hist.append(G)


  # Create subplots: 4 rows, 1 column; share x-axis if desired
  fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

  # Plot L_hist in the first subplot
  axs[0].plot(L_hist, color='blue')
  axs[0].set_title('L_hist')
  axs[0].set_ylabel('Value')

  # Plot T_hist in the second subplot
  axs[1].plot(T_hist, color='green')
  axs[1].set_title('T_hist')
  axs[1].set_ylabel('Value')

  # Plot N_hist in the third subplot
  axs[2].plot(N_hist, color='red')
  axs[2].set_title('N_hist')
  axs[2].set_ylabel('Value')

  # Plot G_hist in the fourth subplot
  axs[3].plot(G_hist, color='orange')
  axs[3].set_title('G_hist')
  axs[3].set_ylabel('Value')
  axs[3].set_xlabel('Index')  # x-axis label only for the bottom plot

  # Improve layout spacing
  plt.tight_layout()

  # Display the plot
  plt.show()

  

if __name__ == '__main__':
  main()