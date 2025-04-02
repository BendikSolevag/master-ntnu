import numpy as np
import pandas as pd
from scipy.optimize import minimize


df = pd.read_csv('./data/HubOcean/parsed-aggregated.csv')
df = df[df['produksjonsomradenr'] == 3]
df = df[df['lokalitetsnummer'] == 11543]
print(df)


X = df['voksne_hunnlus'].values
T = len(X)

time_index = np.arange(T)  

def theta(t, a, b, phi):
    return a + b * np.sin(2.0 * np.pi * (t/52.0) + phi)

def neg_log_likelihood(params, X, t):
    kappa, a, b, phi, sigma = params
    
    if sigma <= 0 or kappa < 0 or kappa > 1:
        return np.inf
    
    nll = 0.0
    for i in range(T - 1):
        mu = (1 - kappa)*X[i] + kappa*theta(t[i], a, b, phi)
        diff = X[i+1] - mu
        nll += 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * (diff**2 / sigma**2)
    return nll


init_params = [0.5,  # kappa
               np.mean(X),  # a
               0.5,  # b
               0.0,  # phi
               1.0]  # sigma

res_mle = minimize(neg_log_likelihood, x0=init_params, args=(X, time_index),
                   method='L-BFGS-B',
                   bounds=[(1e-6, 1-1e-6),  # 0 < kappa < 1
                           (None, None),    # a
                           (None, None),    # b
                           (None, None),    # phi
                           (1e-6, None)])   # sigma>0

best_params_mle = res_mle.x
print("Best params (MLE) = ", best_params_mle)

import matplotlib.pyplot as plt
import numpy as np

# Estimated parameters (already computed)
kappa, a, b, phi, sigma = best_params_mle

# Time range
T_sim = T
sim_time = np.arange(T_sim)

# Seasonal mean function
def theta(t, a, b, phi):
    return a + b * np.sin(2.0 * np.pi * (t / 52.0) + phi)

# Simulate sample path
X_sim = np.zeros(T_sim)
X_sim[0] = X[0]  # Initial value from real data

for t in range(T_sim - 1):
    seasonal_mean = theta(t, a, b, phi)
    X_sim[t + 1] = (1 - kappa) * X_sim[t] + kappa * seasonal_mean + np.random.normal(0, sigma)**2

# Calculate seasonal curve
theta_vals = theta(sim_time, a, b, phi)

# Plot everything
plt.figure(figsize=(12, 6))
plt.plot(sim_time, X, label='Original Data', linewidth=2)
plt.plot(sim_time, X_sim, label='Simulated Path', linestyle='--')
plt.plot(sim_time, theta_vals, label='Seasonal Mean (theta)', linestyle=':', color='gray')
plt.xlabel('Time (weeks)')
plt.ylabel('Adult Female Lice')
plt.title('Lice Levels: Data, Simulated Path, and Seasonal Mean')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()