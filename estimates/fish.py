import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
  df_main = pd.read_csv(f'./data/Fiskeridirektoratet/biomasse.csv', sep=";")
  df_main = df_main[df_main['ARTSID'] == 'LAKS']
  df_main = df_main.dropna()
  df_main = df_main[df_main['PO_KODE'] != '(null)']
  df_main = df_main[['ÅR', 'MÅNED_KODE', 'UTSETTSÅR', 'PO_KODE', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']]
  # ['ÅR', 'MÅNED_KODE', 'MÅNED', 'PO_KODE', 'PO_NAVN', 'ARTSID', 'UTSETTSÅR', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'UTTAK_SLØYD_KG', 'UTTAK_HODEKAPPET_KG', 'UTTAK_RUNDVEKT_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']  

  df_lice = pd.read_csv('./data/HubOcean/parsed-aggregated.csv')

  Y = []
  X = []

  #df_main = df_main[df_main['PO_KODE'] == '03']
  
  
  for po_kode in tqdm(df_main['PO_KODE'].unique()):
    df = df_main[df_main['PO_KODE'] == po_kode]
    lice = df_lice[df_lice['produksjonsomradenr'].astype(int) == int(po_kode)]

    for utsettår in df['UTSETTSÅR'].unique():
      generation = df[df['UTSETTSÅR'].astype(int) == int(utsettår)]

      for i in range(len(generation)-1):
        
        prev = generation.iloc[i]
        curr = generation.iloc[i+1]
        lice_in_month = lice[(lice['month'].astype(int) == int(curr.MÅNED_KODE)) & (lice['year'].astype(int) == int(curr.ÅR))]
        if len(lice_in_month) == 0:
          # If we do not have lice data for the current month, skip it
          continue
        # There was no stocked fish last month, meaning we have no label
        if prev.BEHFISK_STK == 0:
          continue

        # Predictive variable: We want to predict mortality
        label = curr.DØDFISK_STK / prev.BEHFISK_STK
        #path.append(prev.BEHFISK_STK + 1)
        #path.append(np.log(label))
        

        n_facilities = len(lice_in_month['lokalitetsnummer'].unique())

        # Feature 1: Total number of badebehandling treatments divided by number of facilities
        badebehandling_in_month = len(lice_in_month[lice_in_month['badebehandling']]) / n_facilities
        # Feature 2: Total number of forbehandling treatments divided by number of facilities
        forbehandling_in_month = len(lice_in_month[lice_in_month['forbehandling']]) / n_facilities
        # Feature 3: Total number of mekanisk fjerning treatments divided by number of facilities
        mekanisk_in_month = len(lice_in_month[lice_in_month['mekanisk_fjerning']]) / n_facilities



        # Feature 2: Which production facility are we currently viewing
        feature_po_kode = int(po_kode)

        # Feature 3: Which month is it currently?
        feature_month = int(curr.MÅNED_KODE)

        Y.append(label)
        X.append([badebehandling_in_month, forbehandling_in_month, mekanisk_in_month, feature_po_kode, feature_month])


  X = np.array(X)
  y = np.array(Y)


  model = LinearRegression()
  model.fit(X, y)
  print(model.score(X, y))
  print(model.coef_, model.intercept_)
  


if __name__ == '__main__':
  main()