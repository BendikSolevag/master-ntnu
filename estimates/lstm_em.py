from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox


# Sample dataset preparation class
class FishMortalityDataset(Dataset):
  def __init__(self, data):
    """
    Args:
        data (list): A 2D list where:
            - Each outer list corresponds to a farm facility.
            - Each inner list corresponds to the time-series data (tuple of input tensor, target value).
    """
    self.data = []
    for facility_data in data:
        inputs = torch.stack([entry[0] for entry in facility_data])  # Stack inputs (features)
        targets = torch.tensor([entry[1] for entry in facility_data], dtype=torch.float32)  # Stack targets
        self.data.append((inputs, targets))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

def collate_fn(batch):
  inputs, targets = zip(*batch)
  seq_lengths = torch.tensor([len(seq) for seq in inputs])  # Store lengths before padding
  inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
  targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0)
  mask = (targets_padded != 0).float()  # 1 for valid values, 0 for padding
  return inputs_padded, targets_padded, mask, seq_lengths

# Define the LSTM model
class LSTMRegressor(nn.Module):
  def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
    super(LSTMRegressor, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    output = self.fc(lstm_out)
    return output.squeeze(-1)


def main():
  # Example dataset: (xrandom tensors for input, random float for target)

  data = torch.load('./data/featurized/mortality/lstm/lstm.pt')

  p_values = []
  for series in data:
    ys = [y[1].item() for y in series]
    result = acorr_ljungbox(ys, lags=[1], return_df=True)  # test lag 1 autocorrelation
    
    p_values.append(result["lb_pvalue"].iloc[0])
  
  p_values = [p_value for p_value in p_values if str(p_value) != 'nan']

  
  plt.hist(p_values, bins=100)
  plt.ylabel("Frequency")
  plt.xlabel("Ljung-Box p-value (1 lag)")
  plt.savefig('./illustrations/mortality/mortality_autocorrelations.png', format="png", dpi=600)
  plt.close()

  from scipy.stats import combine_pvalues
  

  stat, p_combined = combine_pvalues(p_values, method='fisher')
  print(stat, p_combined)

  

  

  data_train, data_test = train_test_split(data, test_size=0.15)

  print(len(data_train), len(data_test))



  


  data_train = FishMortalityDataset(data_train)
  data_test = FishMortalityDataset(data_test)
  dataloader_train = DataLoader(data_train, batch_size=1, collate_fn=collate_fn, shuffle=True)
  dataloader_test = DataLoader(data_test, batch_size=1, collate_fn=collate_fn, shuffle=True)

  input_size = 9  # Assuming each time step has 5 features

  model = LSTMRegressor(input_size)
  criterion = nn.L1Loss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  model.train()
  train_losses = []
  for epoch in tqdm(range(25)):
    total_loss = 0.0
    for inputs, targets, mask, seq_length in dataloader_train:
      optimizer.zero_grad()
      predictions = model(inputs)
      loss = criterion(predictions * mask, targets * mask)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss)
  torch.save(model.state_dict(), './model_params.pt')


  ys = []
  y_hats = []
  model.eval()
  for inputs, targets, mask, seq_length in dataloader_train:
    predictions = model(inputs)
    
    for i in range(len(predictions)):
      if mask[0, i] == 0:
        break
      ys.append(targets[0, i].item())
      y_hats.append(predictions[0, i].item())

  print('In sample r2', r2_score(ys, y_hats))

  ys = []
  y_hats = []
  model.eval()
  for inputs, targets, mask, seq_length in dataloader_test:
    predictions = model(inputs)
    
    for i in range(len(predictions)):
      if mask[0, i] == 0:
        break
      ys.append(targets[0, i].item())
      y_hats.append(predictions[0, i].item())

  print('test set r2', r2_score(ys, y_hats))



  #plt.plot(train_losses)
  #plt.show()
  

# Example usage
if __name__ == "__main__":
  main()