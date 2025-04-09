import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


X = np.load('./data/featurized/growth/X.npy')  
age = X[:, 0]
feed = X[:, 1]
size = X[:, 2]
temp = X[:, 3]
label = np.load('./data/featurized/growth/y.npy')

T = len(X)


def predict_next_weight(W_t, A_t, F_t, Temp_t, alpha, beta_w, beta_a, beta_f, beta_temp):
  """
  Returns the predicted weight at time t+1 given data at time t.
  """
  return alpha \
    + beta_w   * W_t \
    + beta_a   * A_t \
    + beta_f   * F_t \
    + beta_temp * Temp_t


def sse(params, W, age, feed, temp, label):
  alpha, beta_w, beta_a, beta_f, beta_temp = params
  residuals = []
  for i in range(len(W)):
    W_pred = predict_next_weight(W[i], age[i], feed[i], temp[i],
                                  alpha, beta_w, beta_a, beta_f, beta_temp)
    W_actual = label[i]
    residuals.append(W_actual - W_pred)

  return np.sum(np.array(residuals)**2)

init_params = [0.1,  # alpha
               0.01, # beta_w
               0.01, # beta_a
               0.01, # beta_f
               0.01] # beta_temp

res = minimize(sse,
  x0=init_params,
  args=(size, age, feed, temp, label),
  method='L-BFGS-B',
  bounds=[(None, None),  # alpha
    (None, None),        # beta_w
    (None, None),        # beta_a
    (None, None),        # beta_f
    (None, None)])       # beta_temp

best_params = res.x
print("Best params (SSE) = ", best_params)

alpha_fit, beta_w_fit, beta_a_fit, beta_f_fit, beta_temp_fit = best_params

predictions = []
for i in range(200):
  pred = predict_next_weight(size[i], age[i], feed[i], temp[i],
                              alpha_fit, 
                              beta_w_fit, 
                              beta_a_fit, 
                              beta_f_fit, 
                              beta_temp_fit,
  )
  predictions.append(pred)

plt.plot(predictions)
plt.show()