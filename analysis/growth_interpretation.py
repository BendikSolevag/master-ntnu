import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

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
    


TARGET_GROWTH = 8.0
DECAY = 0.05  # You can tune this parameter

model = GrowthNN(input_size=4)
model.load_state_dict(torch.load('./models/growth/1743671011.288821-model.pt', weights_only=True))
model.eval()

curve = []
weight = 0.15
total_feedcost = 0
for i in range(200):
  feed_amount_per_fish = weight * 0.015 * 30
  explanatory = [
    round(i / 52), #generation_approx_age, 
    feed_amount_per_fish, #feedamountperfish, 
    weight, #mean_size,
    0, #mean_voksne_hunnlus,
  ]

  pred = model.forward(torch.tensor(explanatory, dtype=torch.float32)).item()
  curve.append(weight)
  g_rate = np.log(pred / weight) / 4.345
  
  weight *= (np.exp(g_rate * np.sqrt(1 - (weight / 8)) ))

  # This is how weekly feed cost was calculated
  total_feedcost += 0.14 * feed_amount_per_fish
  
  if weight >= 5:
      break

total_feedcost = total_feedcost / (weight * 0.84)
plt.title(total_feedcost)
plt.plot(curve)
plt.show()
