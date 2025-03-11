import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():

  main_df = pd.DataFrame()
  paths = []
  for i in tqdm(range(10)):
    df = pd.read_csv(f'./data/HubOcean/lusedata{2015 + i}.csv')
    # ['referansenummer', 'uke', 'sjotemperatur', 'voksne_hunnlus', 'bevegelige_lus', 'fastsittende_lus', 'beholdning', 'dato', 'rensefisk_aktiv', 'badebehandling', 'forbehandling', 'mekanisk_fjerning', 'utsett_av_rensefisk', 'year', 'lokalitetsnummer', 'lokalitet', 'kommunenummer', 'kommune', 'geometry', 'produksjonsomradenr', 'produksjonsomrade', 'plassering_id', 'plassering', 'fylkenummer', 'fylke', 'sone_id', 'sone', 'omrade_id', 'omrade', 'region_id', 'region', 'total_voksne_hunnlus', 'total_bevegelige_lus', 'total_fastsittende_lus', 'longitude', 'latitude', 'timestamp']
    df = df[['dato', 'uke', 'year', 'voksne_hunnlus', 'rensefisk_aktiv', 'badebehandling', 'forbehandling', 'mekanisk_fjerning', 'utsett_av_rensefisk', 'lokalitetsnummer',  'produksjonsomradenr', 'plassering_id']]
    df = df.dropna()
    if i == 0:
      main_df = df
    else:
      main_df = pd.concat([main_df, df], ignore_index=True)


    groups = df.groupby(['uke'])
    means = []
    for uke, group in groups:
      mean = group['voksne_hunnlus'].mean()
      means.append(mean)
    plt.plot(means, alpha=0.5, label=f"{2015+i}")
    
    paths.append(means)

  paths = np.array(paths)
  meanpath = paths.mean(axis=0)
  plt.plot(meanpath, color="black", label="Mean")
  plt.legend(loc=2, prop={'size': 8})
  plt.xlabel('Week number')
  plt.ylabel('Adult female lice per individual')
  plt.savefig('./illustrations/lice/seasonality.png', format='png', dpi=600)
  plt.close()

  ser = [*main_df['produksjonsomradenr'].values]
  ser = [str(round(val)) if val >= 10 else f"0{round(val)}" for val in ser]
  main_df['produksjonsomradenr'] = ser

  month = [*main_df['dato'].values]
  month = [int(val[5:7]) for val in month]
  main_df['month'] = month


  print(main_df)
  main_df.to_csv('./data/HubOcean/parsed-aggregated.csv')
  
  

  
  


if __name__ == '__main__':
  main()