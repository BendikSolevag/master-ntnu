import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
import random



def main():
  df_main = pd.read_csv(f'./data/Fiskeridirektoratet/biomasse.csv', sep=";")
  df_main = df_main[df_main['ARTSID'] == 'LAKS']
  df_main = df_main.dropna()
  df_main = df_main[df_main['PO_KODE'] != '(null)']
  df_main = df_main[['ÅR', 'MÅNED_KODE', 'UTSETTSÅR', 'PO_KODE', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']]
  # ['ÅR', 'MÅNED_KODE', 'MÅNED', 'PO_KODE', 'PO_NAVN', 'ARTSID', 'UTSETTSÅR', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'UTTAK_SLØYD_KG', 'UTTAK_HODEKAPPET_KG', 'UTTAK_RUNDVEKT_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']  

  df_lice = pd.read_csv('./data/HubOcean/parsed-aggregated.csv')
  print(df_lice)

  Y = []
  X = []

  global_growthrate = []
  global_mean_weight = []
  global_mean_temp = []
  global_mortalityrate = []
  global_feedamountperfish = []

  

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
        if prev.BEHFISK_STK == 0 or prev.BIOMASSE_KG == 0 or curr.BEHFISK_STK == 0 or curr.BIOMASSE_KG == 0:
          continue

        # Predictive variable: We want to predict growth rate
        label = (curr.BIOMASSE_KG / curr.BEHFISK_STK) / (prev.BIOMASSE_KG / prev.BEHFISK_STK)

        global_growthrate.append(label)
        global_mean_weight.append((prev.BIOMASSE_KG / prev.BEHFISK_STK))

        temp = np.mean(lice_in_month['sjotemperatur'])
        global_mean_temp.append(temp)

        mortality = curr.DØDFISK_STK / prev.BEHFISK_STK
        global_mortalityrate.append(mortality)

        feedamountperfish = curr.FORFORBRUK_KG / curr.BEHFISK_STK
        global_feedamountperfish.append(feedamountperfish)
        
        
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

  df_ana = pd.DataFrame({
    "growth": global_growthrate, 
    "weight": global_mean_weight, 
    "temp": global_mean_temp,
    "mort": global_mortalityrate,
    "fperfish": global_feedamountperfish,
  })
  print(df_ana)

  Q1 = np.percentile(df_ana['growth'], 25)
  Q3 = np.percentile(df_ana['growth'], 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_ana = df_ana[(df_ana['growth'] >= lower_bound) & (df_ana['growth'] <= upper_bound)]

  Q1 = np.percentile(df_ana['weight'], 25)
  Q3 = np.percentile(df_ana['weight'], 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_ana = df_ana[(df_ana['weight'] >= lower_bound) & (df_ana['weight'] <= upper_bound)]

  Q1 = np.percentile(df_ana['temp'], 25)
  Q3 = np.percentile(df_ana['temp'], 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_ana = df_ana[(df_ana['temp'] >= lower_bound) & (df_ana['temp'] <= upper_bound)]

  Q1 = np.percentile(df_ana['mort'], 25)
  Q3 = np.percentile(df_ana['mort'], 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_ana = df_ana[(df_ana['mort'] >= lower_bound) & (df_ana['mort'] <= upper_bound)]
  

  Q1 = np.percentile(df_ana['fperfish'], 25)
  Q3 = np.percentile(df_ana['fperfish'], 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_ana = df_ana[(df_ana['fperfish'] >= lower_bound) & (df_ana['fperfish'] <= upper_bound)]
  print(df_ana)


  fig = plt.figure(figsize=(12, 12))
  ax = fig.add_subplot(projection='3d')
  ax.scatter(df_ana['growth'], df_ana['weight'], df_ana['fperfish'])
  ax.set_xlabel("Growth rate")
  ax.set_ylabel("Weight")
  ax.zaxis.set_rotate_label(False) 
  ax.set_zlabel('Feed per fish', rotation = 0)
  plt.show()

  #plt.scatter(df_ana['growth'], df_ana['fperfish'])
  #plt.show()
  
  #plt.hist(global_growthrate, bins=50, edgecolor="black")
  #plt.xlabel("Monthly growth rate")
  #plt.ylabel("Frequency")
  #plt.show()

  #X = np.array(X)
  #y = np.array(Y)
  #model = LinearRegression()
  #model.fit(X, y)
  #print(model.score(X, y))
  #print(model.coef_, model.intercept_)

if __name__ == '__main__':


  main()