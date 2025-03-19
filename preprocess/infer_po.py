import pandas as pd
# ['AAR', 'MAANED', 'LOKNR', 'ARSKLASSE', 'FISKEARTID', 'UTSATT_SMOLT_STK', 'UTSATT_TOTALT_STK', 'FISKEBEHOLDNING_ANTALL', 'BIOMASSE_KG', 'FORFORBRUK_KG', 'TAP_DODFISK_STK', 'TAP_UTKAST_STK', 'TAP_ROMMING_STK', 'TAP_ANNET_STK', 'UTTAK_SLAKT_STK', 'UTTAK_SLAKT_KG', 'UTTAK_LEVENDE_STK', 'UTTAK_LEVENDE_KG', 'PROD_OMR', 'LATITUDE', 'LONGITUDE']

df_main = pd.read_csv('./data/Fiskeridirektoratet/lokalitet/csv/aggregated.csv', sep=";")

df = df_main[~df_main['PROD_OMR'].isin([None, "(null)"]) & df_main['PROD_OMR'].notna()]
POs = df['PROD_OMR'].unique()


# Identify rows where inferring PO-code is possible
po_inferrable = df_main[
    (df_main['PROD_OMR'].isnull() | (df_main['PROD_OMR'] == '(null)')) & 
    df_main['LATITUDE'].notnull() & 
    (df_main['LATITUDE'] != '(null)') &
    df_main['LONGITUDE'].notnull() & 
    (df_main['LONGITUDE'] != '(null)')
]



# Load lice

# 