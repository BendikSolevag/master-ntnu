
import pandas as pd
import numpy as np
import re

def parse_number(s):
    s = str(s)
    
    # Remove all non-digit, non-decimal, non-negative-sign characters
    s = re.sub(r'[^\d.,-]', '', s)
    
    # Handle different decimal/thousands separators
    if s.count(',') > 1 and '.' not in s:
        s = s.replace(',', '')
    elif s.count('.') > 1 and ',' not in s:
        s = s.replace('.', '')
    elif ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(' ', '').replace(',', '')  # fallback
    try:
        return float(s)
    except ValueError:
        return None  # or np.nan





old = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/2012-2016-Tabell 1.csv', sep=";")
old['BIOMASSE_KG'] = old['BIOMASSE_KG'].apply(parse_number)

med = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/2017-2021-Tabell 1.csv', sep=";")
med['BIOMASSE_KG'] = med['BIOMASSE_KG'].apply(parse_number)

new = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/2022-2023-Tabell 1.csv', sep=";")
new['BIOMASSE_KG'] = new['BIOMASSE_KG'].apply(parse_number)

print(old.dtypes)
print(med.dtypes)
print(new.dtypes)


# Aggregate different year dataframes
df = pd.concat([old, med, new], ignore_index=True)

df = df[['AAR', 'MAANED', 'LOKNR', 'LOKNAVN', 'ARSKLASSE', 'FISKEARTID', 'UTSATT_SMOLT_STK', 'UTSATT_TOTALT_STK', 'FISKEBEHOLDNING_ANTALL', 'BIOMASSE_KG', 'FORFORBRUK_KG', 'TAP_DODFISK_STK', 'TAP_UTKAST_STK', 'TAP_ROMMING_STK', 'TAP_ANNET_STK', 'UTTAK_SLAKT_STK', 'UTTAK_SLAKT_KG', 'UTTAK_LEVENDE_STK', 'UTTAK_LEVENDE_KG', 'PROD_OMR', 'LATITUDE', 'LONGITUDE']]
df = df[['AAR', 'MAANED', 'LOKNR', 'ARSKLASSE', 'FISKEARTID', 'UTSATT_SMOLT_STK', 'UTSATT_TOTALT_STK', 'FISKEBEHOLDNING_ANTALL', 'BIOMASSE_KG', 'FORFORBRUK_KG', 'TAP_DODFISK_STK', 'TAP_UTKAST_STK', 'TAP_ROMMING_STK', 'TAP_ANNET_STK', 'UTTAK_SLAKT_STK', 'UTTAK_SLAKT_KG', 'UTTAK_LEVENDE_STK', 'UTTAK_LEVENDE_KG', 'PROD_OMR', 'LATITUDE', 'LONGITUDE']]
df.to_csv('./data/Fiskeridirektoratet/lokalitet/csv/aggregated.csv', sep=";")

