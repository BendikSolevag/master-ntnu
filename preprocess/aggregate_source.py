
import pandas as pd
import numpy as np

old = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/2012-2016-Tabell 1.csv', sep=";")
med = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/2017-2021-Tabell 1.csv', sep=";")
new = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/2022-2023-Tabell 1.csv', sep=";")





# Aggregate different year dataframes
df = pd.concat([old, med, new], ignore_index=True)

df = df[['AAR', 'MAANED', 'LOKNR', 'LOKNAVN', 'ARSKLASSE', 'FISKEARTID', 'UTSATT_SMOLT_STK', 'UTSATT_TOTALT_STK', 'FISKEBEHOLDNING_ANTALL', 'BIOMASSE_KG', 'FORFORBRUK_KG', 'TAP_DODFISK_STK', 'TAP_UTKAST_STK', 'TAP_ROMMING_STK', 'TAP_ANNET_STK', 'UTTAK_SLAKT_STK', 'UTTAK_SLAKT_KG', 'UTTAK_LEVENDE_STK', 'UTTAK_LEVENDE_KG', 'PROD_OMR', 'LATITUDE', 'LONGITUDE']]
df = df[['AAR', 'MAANED', 'LOKNR', 'ARSKLASSE', 'FISKEARTID', 'UTSATT_SMOLT_STK', 'UTSATT_TOTALT_STK', 'FISKEBEHOLDNING_ANTALL', 'BIOMASSE_KG', 'FORFORBRUK_KG', 'TAP_DODFISK_STK', 'TAP_UTKAST_STK', 'TAP_ROMMING_STK', 'TAP_ANNET_STK', 'UTTAK_SLAKT_STK', 'UTTAK_SLAKT_KG', 'UTTAK_LEVENDE_STK', 'UTTAK_LEVENDE_KG', 'PROD_OMR', 'LATITUDE', 'LONGITUDE']]
df.to_csv('./data/Fiskeridirektoratet/lokalitet/csv/aggregated.csv', sep=";")

