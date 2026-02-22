
import re
import os
import sys

import zarr
import yaml
from glob import glob
# from datetime import datetime, timedelta

import numpy as np
import xarray as xr
# import pandas as pd

# sys.path.insert(0, os.path.realpath('../../libs/'))
# import verif_utils as vu

ds_geo = xr.open_zarr('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr')
XLONG = ds_geo['XLONG'].values
XLAT = ds_geo['XLAT'].values

exp_name = 'GDAS'
for year in range(2021, 2025):
    
    ds_final = xr.open_zarr(f'/glade/campaign/ral/hap/ksha/GWC_results/FINAL_run/final_{exp_name}_{year}_WY.zarr', chunks={})
    
    # Add global attribute
    ds_final.attrs['Conventions'] = 'CF-1.11'
    
    # Decode time
    ds_final['time'] = xr.decode_cf(ds_final[['time']]).time
    
    # Replace indexing dims with coordinate dims and standard names
    if 'XLAT' in ds_final and 'XLONG' in ds_final:
        ds_final = ds_final.set_coords(['XLAT', 'XLONG'])
        ds_final['XLAT'] = ds_final['XLAT'].rename({'latitude': 'south_north', 'longitude': 'west_east'})
        ds_final['XLONG'] = ds_final['XLONG'].rename({'latitude': 'south_north', 'longitude': 'west_east'})
    
        ds_final['XLAT'].attrs.update({
            "standard_name": "latitude",
            "units": "degrees_north",
            "long_name": "latitude"
        })
        ds_final['XLONG'].attrs.update({
            "standard_name": "longitude",
            "units": "degrees_east",
            "long_name": "longitude"
        })
    
    # Replace lat/lon dims with south_north/west_east
    dim_map_3d = ('time', 'level', 'south_north', 'west_east')
    dim_map_2d = ('time', 'south_north', 'west_east')
    
    for var in ds_final.data_vars:
        dims = ds_final[var].dims
        dim_rename = {}
        if 'latitude' in dims:
            dim_rename['latitude'] = 'south_north'
        if 'longitude' in dims:
            dim_rename['longitude'] = 'west_east'
        
        if dim_rename:
            ds_final[var] = ds_final[var].rename(dim_rename)
    
    output_name = f'/glade/derecho/scratch/ksha/GWC_Results/final_{exp_name}_{year}_WY.nc'
    ds_final.to_netcdf(output_name, format='NETCDF4_CLASSIC')
    
    print(output_name)
