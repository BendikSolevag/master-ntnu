import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
import torch

def main():
  df_bio = pd.read_csv(f'./data/Fiskeridirektoratet/lokalitet/csv/aggregated.csv', sep=";")
  df_bio = df_bio[df_bio['FISKEARTID'] == 71101]
  print(len(df_bio['LOKNR'].unique()))
    
  # ['ÅR', 'MÅNED_KODE', 'MÅNED', 'PO_KODE', 'PO_NAVN', 'ARTSID', 'UTSETTSÅR', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'UTTAK_SLØYD_KG', 'UTTAK_HODEKAPPET_KG', 'UTTAK_RUNDVEKT_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']  

  df_lice = pd.read_csv('./data/HubOcean/parsed-aggregated.csv')

  

  Y = []
  X = []

  LSTM_dataset = []

  #df_main = df_main[df_main['PO_KODE'] == '03']
  
  bio_locnums = df_bio['LOKNR'].unique()
  lice_locnums = df_lice['lokalitetsnummer'].unique()
  for lokalitet in tqdm(bio_locnums):
    if not lokalitet in lice_locnums:
      # Lice data does not exist for current lokalitet
      continue





    # Create dataframes concerning only current lokalitet
    dfs = df_bio[df_bio['LOKNR'] == lokalitet]
    dfl = df_lice[df_lice['lokalitetsnummer'].astype(int) == int(lokalitet)]

    
    for utsettår in dfs['ARSKLASSE'].unique():
      generation = dfs[dfs['ARSKLASSE'].astype(int) == int(utsettår)]
      series = []
      for i in range(1, len(generation) - 1):

        prev = generation.iloc[i - 1]
        curr = generation.iloc[i]
        
        if curr.ARSKLASSE == 2011:
          continue

        licerows = dfl[(dfl['month'].astype(int) == int(curr.MAANED)) & (dfl['year'].astype(int) == int(curr.AAR))]
        
        
        if len(licerows) == 0:
          
          # If we do not have lice data for the current month, skip it
          continue
        # There was no stocked fish last month, meaning we have no label
        if prev.FISKEBEHOLDNING_ANTALL == 0:
          continue

        

        # Predictive variable: We want to predict mortality
        label = curr.TAP_DODFISK_STK / prev.FISKEBEHOLDNING_ANTALL



        
        # Feature 1: Total number of badebehandling treatments divided by number of rows in month
        badebehandling_in_month = 1 if len(licerows[licerows['badebehandling']]) > 0 else 0
        # Feature 2: Total number of forbehandling treatments divided by number of facilities
        forbehandling_in_month = 1 if len(licerows[licerows['forbehandling']]) > 0 else 0
        # Feature 3: Total number of mekanisk fjerning treatments divided by number of facilities
        mekanisk_in_month = 1 if len(licerows[licerows['mekanisk_fjerning']]) > 0 else 0
        # Feature 5: Treatment any
        treatment_any = 1 if badebehandling_in_month or forbehandling_in_month or mekanisk_in_month else 0

        # Feature 6: Water temperatre
        mean_temp = licerows['sjotemperatur'].mean()

        # Feature 7: Lice level
        mean_voksne_hunnlus = licerows['voksne_hunnlus'].mean()
        mean_bevegelige_lus = licerows['bevegelige_lus'].mean()
        mean_fastsittende_lus = licerows['fastsittende_lus'].mean()

        # Feature X: Age
        generation_approx_age = curr.AAR - curr.ARSKLASSE

        explanatory = [badebehandling_in_month, forbehandling_in_month, mekanisk_in_month, treatment_any, mean_temp, mean_voksne_hunnlus, mean_bevegelige_lus, mean_fastsittende_lus, generation_approx_age]
        Y.append(label)
        X.append(explanatory)

        series.append((torch.tensor(explanatory), torch.tensor(label)))
      if len(series) > 1:
        LSTM_dataset.append(series)




  
  Y = np.array(Y)



  # Handle outliers
  Q1 = np.percentile(Y, 25)
  Q3 = np.percentile(Y, 75)
  IQR = Q3 - Q1
  upper_bound = Q3 + 3 * IQR
  lower_bound = Q1 - 3 * IQR

  print(upper_bound)
  print(lower_bound)
  
  
  X_filtered = []
  y_filtered = []
  for i in range(len(X)):
    if Y[i] > 1:
      continue
    #if Y[i] > upper_bound or Y[i] < lower_bound:
    #  continue
    X_filtered.append(X[i])
    y_filtered.append(Y[i])
  
  X = np.array(X_filtered)
  y = np.array(y_filtered)



  np.save('./data/featurized/mortality/X.npy', X)
  np.save('./data/featurized/mortality/y.npy', y)
  torch.save(LSTM_dataset, './data/featurized/mortality/lstm/lstm.pt')


  
  


if __name__ == '__main__':
  main()