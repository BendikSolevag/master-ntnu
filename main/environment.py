
import numpy as np


def environment(N_zero=150000, L=15000, G=0.2, delta_t=0.01):
  sliding_window_max = round((2/52)/delta_t)
  sliding_window_lice = [L/N for _ in range(sliding_window_max)]
  

  N = N_zero
  
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
  kappa_SML = 0.01

  # Salmon Mortality Treatment
  kappa_SMT = 0.1

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
    T = 1 if any(sliding_window_lice, lambda x: x > L_bar) else 0

    # Update salmon population
    N += -(kappa_SML * (L / N) + kappa_SMT*T)*N*delta_t

    # Update salmon weight
    G += (kappa_GI*(1-(G/G_max))*(1-T) - kappa_GD*T)*G*delta_t



    yield L, T, N, G

