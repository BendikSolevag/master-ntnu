import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.metrics import r2_score as r2s
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import r2_score



class MortalityDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


class MortalityNN(nn.Module):
    def __init__(self, input_size):
        super(MortalityNN, self).__init__()
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
    

def main():
  X = np.load('./data/featurized/mortality/X.npy')
  y = np.load('./data/featurized/mortality/y.npy')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

  model = LinearRegression().fit(X_train, y_train)
  preds = model.predict(X_test)
  print(r2s(preds, y_test))

  train_dataset = MortalityDataset(X_train, y_train)
  test_dataset = MortalityDataset(X_test, y_test)
  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  batch_size = 8
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  model = MortalityNN(input_size=X.shape[1])
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  lossfunc = nn.MSELoss()
  epoch_losses_train = []
  epoch_losses_test = []

  for _ in tqdm(range(7)):
    epoch_loss_train = 0
    model.train()
    for X_batch, y_batch in train_loader:
      if len(y_batch) != batch_size:
         continue
      optimizer.zero_grad()
      pred = model.forward(X_batch).squeeze()
      loss = lossfunc(pred, y_batch)
      loss.backward()
      optimizer.step()
      epoch_loss_train += loss.item()
    epoch_losses_train.append(epoch_loss_train / len(train_loader))
    
    epoch_loss_test = 0
    model.eval()
    for X_batch, y_batch in test_loader:
      if len(y_batch) != batch_size:
         continue
      pred = model.forward(X_batch).squeeze()
      loss = lossfunc(pred, y_batch)
      epoch_loss_test += loss.item()
    epoch_losses_test.append(epoch_loss_test / len(test_loader))

  test_pred = model.forward(X_test).squeeze()
  r2 = r2_score(test_pred, y_test)
  print(r2)
  

  plt.plot(epoch_losses_train, label="train")
  plt.plot(epoch_losses_test, label="test")
  plt.legend()
  plt.show()
    

  

if __name__ == '__main__':
  main()

