import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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


init_params = [0.5, np.mean(X), 0.5, 0.0, 1.0]  
res_mle = minimize(neg_log_likelihood, x0=init_params, args=(X, time_index), method='L-BFGS-B', bounds=[(1e-6, 1-1e-6), (None, None), (None, None), (None, None), (1e-6, None)])   

best_params_mle = res_mle.x
print(best_params_mle)

kappa, a, b, phi, sigma = best_params_mle
sim_time = np.arange(T)

X_sim = np.zeros(T)
X_sim[0] = X[0]

for t in range(T - 1):
    seasonal_mean = theta(t, a, b, phi)
    X_sim[t + 1] = (1 - kappa) * X_sim[t] + kappa * seasonal_mean + np.random.normal(0, sigma)

theta_vals = theta(sim_time, a, b, phi)

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