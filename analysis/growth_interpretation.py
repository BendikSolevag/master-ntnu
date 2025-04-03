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
    


model = GrowthNN(input_size=7)
model.load_state_dict(torch.load('./models/growth/model.pt', weights_only=True))
model.eval()

curve = []
weight = 0.2
for i in range(200):
  explanatory = [
    1, #badebehandling_in_month, 
    1, #forbehandling_in_month, 
    1, #mekanisk_in_month, 
    round(i / 52), #generation_approx_age, 
    weight * 0.015 * 30, #feedamountperfish, 
    weight, #mean_size,
    0, #mean_voksne_hunnlus,
  ]
  pred = model.forward(torch.tensor(explanatory, dtype=torch.float32)).item()
  curve.append(weight)
  weight *= (pred**(1/4.33))


plt.plot(curve)
plt.show()
