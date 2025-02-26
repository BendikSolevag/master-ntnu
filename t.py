

import xarray as xr

ds = xr.open_dataset('data/data.nc')
df = ds.to_dataframe()

print(df[df['C10'] != 0.0])