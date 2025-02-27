import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  paths = []
  for i in range(10):
    df = pd.read_csv(f'./data/HubOcean/lusedata{2015 + i}.csv')
    # ['referansenummer', 'uke', 'sjotemperatur', 'voksne_hunnlus', 'bevegelige_lus', 'fastsittende_lus', 'beholdning', 'dato', 'rensefisk_aktiv', 'badebehandling', 'forbehandling', 'mekanisk_fjerning', 'utsett_av_rensefisk', 'year', 'lokalitetsnummer', 'lokalitet', 'kommunenummer', 'kommune', 'geometry', 'produksjonsomradenr', 'produksjonsomrade', 'plassering_id', 'plassering', 'fylkenummer', 'fylke', 'sone_id', 'sone', 'omrade_id', 'omrade', 'region_id', 'region', 'total_voksne_hunnlus', 'total_bevegelige_lus', 'total_fastsittende_lus', 'longitude', 'latitude', 'timestamp']
    df = df[['uke', 'sjotemperatur', 'voksne_hunnlus', 'rensefisk_aktiv', 'badebehandling', 'forbehandling', 'mekanisk_fjerning', 'utsett_av_rensefisk', 'lokalitetsnummer',  'produksjonsomradenr', 'plassering_id']]
    df = df.dropna()
    
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
  
  

  
  


if __name__ == '__main__':
  main()