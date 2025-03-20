import pandas as pd
# ['AAR', 'MAANED', 'LOKNR', 'ARSKLASSE', 'FISKEARTID', 'UTSATT_SMOLT_STK', 'UTSATT_TOTALT_STK', 'FISKEBEHOLDNING_ANTALL', 'BIOMASSE_KG', 'FORFORBRUK_KG', 'TAP_DODFISK_STK', 'TAP_UTKAST_STK', 'TAP_ROMMING_STK', 'TAP_ANNET_STK', 'UTTAK_SLAKT_STK', 'UTTAK_SLAKT_KG', 'UTTAK_LEVENDE_STK', 'UTTAK_LEVENDE_KG', 'PROD_OMR', 'LATITUDE', 'LONGITUDE']

#df_main = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/aggregated.csv', sep=";")
#print(df_main)
#df_main = df_main.dropna(subset=['LOKNR'])
#print(df_main)


df_main = pd.read_csv('./data/HubOcean/parsed-aggregated.csv')
print(df_main)
df_main = df_main.dropna(subset=['lokalitetsnummer'])
print(df_main)