import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  
  df = pd.read_csv(f'./data/Fiskeridirektoratet/biomasse.csv', sep=";")
  df = df[df['ARTSID'] == 'LAKS']
  df = df.dropna()
  df = df[df['PO_KODE'] != '(null)']
  # ['ÅR', 'MÅNED_KODE', 'MÅNED', 'PO_KODE', 'PO_NAVN', 'ARTSID', 'UTSETTSÅR', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'UTTAK_SLØYD_KG', 'UTTAK_HODEKAPPET_KG', 'UTTAK_RUNDVEKT_KG', 'DØDFISK_STK', 'UTKAST_STK', 'RØMMING_STK', 'ANDRE_STK']
  df = df[['ÅR', 'MÅNED_KODE', 'PO_NAVN', 'BEHFISK_STK', 'BIOMASSE_KG', 'UTSETT_SMOLT_STK', 'UTSETT_SMOLT_STK_MINDRE_ENN_500G', 'FORFORBRUK_KG', 'UTTAK_STK', 'UTTAK_KG', 'DØDFISK_STK', 'ANDRE_STK']]
  print(df)

  # Bruk dette for å regne ut generell dødelighet. Prøv å finne en måte å koble sammen med lus, regne ut luserelatert dødelighet
  
  

  
  


if __name__ == '__main__':
  main()