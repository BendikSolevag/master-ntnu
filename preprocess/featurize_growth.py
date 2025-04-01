import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm




def main():
  df_bio = pd.read_csv(f'./data/Fiskeridirektoratet/lokalitet/csv/aggregated.csv', sep=";")
  df_bio = df_bio[df_bio['FISKEARTID'] == 71101]

  df_bio = df_bio[df_bio['FISKEARTID'] == 71101]
  
  
  # ['ÅR', 'MÅNED_KODE', 'MÅNED', 'PO_KODE', 'PO_NAVN', 'ARTSID', 'UTSETTSÅR', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'UTTAK_SLØYD_KG', 'UTTAK_HODEKAPPET_KG', 'UTTAK_RUNDVEKT_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']  

  df_lice = pd.read_csv('./data/HubOcean/parsed-aggregated.csv')

  Y = []
  X = []

  global_growthrate = []
  global_mean_weight = []
  global_mean_temp = []
  global_mortalityrate = []
  global_feedamountperfish = []

  bio_locnums = df_bio['LOKNR'].unique()
  lice_locnums = df_lice['lokalitetsnummer'].unique()
  
  for lokalitet in tqdm(bio_locnums):
    if not lokalitet in lice_locnums:
      # If we do not have lice data for lokalitet, ignore
      continue

    df = df_bio[df_bio['LOKNR'] == lokalitet]
    dfl = df_lice[df_lice['lokalitetsnummer'].astype(int) == int(lokalitet)]

    for utsettår in df['ARSKLASSE'].unique():
      generation = df[df['ARSKLASSE'].astype(int) == int(utsettår)]

      for i in range(len(generation)-1):
        
        prev = generation.iloc[i]
        curr = generation.iloc[i+1]
        licerows = dfl[(dfl['month'].astype(int) == int(curr.MAANED)) & (dfl['year'].astype(int) == int(curr.AAR))]
        
        if len(licerows) == 0:
          # If we do not have lice data for the current month, skip it
          continue

        
        # There was no stocked fish last month, meaning we have no label
        if prev.FISKEBEHOLDNING_ANTALL == 0 or prev.BIOMASSE_KG == 0 or curr.FISKEBEHOLDNING_ANTALL == 0 or curr.BIOMASSE_KG == 0:
          continue

        
        label = (float(curr.BIOMASSE_KG) / int(curr.FISKEBEHOLDNING_ANTALL)) / (float(prev.BIOMASSE_KG) / int(prev.FISKEBEHOLDNING_ANTALL))

        global_growthrate.append(label)
        global_mean_weight.append(float(prev.BIOMASSE_KG) / int(prev.FISKEBEHOLDNING_ANTALL))

        temp = np.mean(licerows['sjotemperatur'])
        global_mean_temp.append(temp)

        mortality = curr.TAP_DODFISK_STK / prev.FISKEBEHOLDNING_ANTALL
        global_mortalityrate.append(mortality)

        feedamountperfish = curr.FORFORBRUK_KG / curr.FISKEBEHOLDNING_ANTALL
        global_feedamountperfish.append(feedamountperfish)
        

        badebehandling_in_month = len(licerows[licerows['badebehandling']])
        forbehandling_in_month = len(licerows[licerows['forbehandling']])
        mekanisk_in_month = len(licerows[licerows['mekanisk_fjerning']])
        generation_approx_age = curr.AAR - curr.ARSKLASSE
        mean_size = float(prev.BIOMASSE_KG) / int(prev.FISKEBEHOLDNING_ANTALL)
        mean_voksne_hunnlus = np.mean(licerows['voksne_hunnlus'].values)

        # consider adding: temp. does not significantly improve r squared
        explanatory = [
          #badebehandling_in_month, 
          #forbehandling_in_month, 
          #mekanisk_in_month, 
          generation_approx_age, 
          feedamountperfish, 
          mean_size,
          mean_voksne_hunnlus,
          #temp
        ]

        Y.append(label)
        X.append(explanatory)

  print(len(X))
  print(len(Y))
  
  # Handle outliers
  Q1 = np.percentile(Y, 25)
  Q3 = np.percentile(Y, 75)
  print(Q1, Q3)
  IQR = Q3 - Q1
  upper_bound = Q3 + 3 * IQR
  lower_bound = Q1 - 3 * IQR

  print(upper_bound)
  print(lower_bound)
  
  
  X_filtered = []
  y_filtered = []
  for i in range(len(X)):
    if Y[i] > upper_bound or Y[i] < lower_bound:
      continue
    #if Y[i] > upper_bound or Y[i] < lower_bound:
    #  continue
    X_filtered.append(X[i])
    y_filtered.append(Y[i])
  
  X = np.array(X_filtered)
  y = np.array(y_filtered)

  print(len(X))
  print(len(y))



  np.save('./data/featurized/growth/X.npy', X)
  np.save('./data/featurized/growth/y.npy', y)

  df_ana = pd.DataFrame({
    "growth": global_growthrate, 
    "weight": global_mean_weight, 
    "temp": global_mean_temp,
    "mort": global_mortalityrate,
    "fperfish": global_feedamountperfish,
  })

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
  ax.scatter(df_ana['growth'], df_ana['weight'], df_ana['temp'],)
  ax.set_xlabel("Growth rate")
  ax.set_ylabel("Weight")
  ax.zaxis.set_rotate_label(False) 
  ax.set_zlabel('Temperature', rotation = 0)
  
  fig.savefig('./illustrations/growth/weight-feed-growth3d.png', bbox_inches=None)
  plt.close('all')

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