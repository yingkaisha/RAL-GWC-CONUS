
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

for year in range(2021, 2025):
    # Load dataset lazily
    ds_C404 = xr.open_zarr(
        f'/glade/campaign/ral/hap/ksha/GWC_results/FINAL_run/target_{year}_WY.zarr', chunks={}
    )

    ds_C404 = ds_C404.drop_vars(['level'])
    
    # Rename dimensions
    ds_C404 = ds_C404.rename({
        'bottom_top': 'level',
        'south_north': 'latitude',
        'west_east': 'longitude'
    })
    
    # Assign level coordinate
    ds_C404 = ds_C404.assign_coords(level=[0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42])
    
    # Drop old lat/lon coords if present, then assign new ones
    ds_C404 = ds_C404.drop_vars(['latitude', 'longitude'], errors='ignore').assign_coords(
        XLAT=(('latitude', 'longitude'), XLAT),
        XLONG=(('latitude', 'longitude'), XLONG)
    )
    
    # Set global attribute
    ds_C404.attrs['Conventions'] = 'CF-1.11'
    
    # Decode CF time
    ds_C404['time'] = xr.decode_cf(ds_C404[['time']]).time
    
    # Set coordinate variables and rename dims
    if 'XLAT' in ds_C404 and 'XLONG' in ds_C404:
        ds_C404 = ds_C404.set_coords(['XLAT', 'XLONG'])
    
        # Rename only the coordinate dims, not data
        ds_C404['XLAT'] = ds_C404['XLAT'].rename({'latitude': 'south_north', 'longitude': 'west_east'})
        ds_C404['XLONG'] = ds_C404['XLONG'].rename({'latitude': 'south_north', 'longitude': 'west_east'})
    
        # Set CF metadata
        ds_C404['XLAT'].attrs.update({
            "standard_name": "latitude",
            "units": "degrees_north",
            "long_name": "latitude"
        })
        ds_C404['XLONG'].attrs.update({
            "standard_name": "longitude",
            "units": "degrees_east",
            "long_name": "longitude"
        })
    
    # Rename lat/lon dims in data variables
    for var in ds_C404.data_vars:
        dims = ds_C404[var].dims
        rename_dims = {}
        if 'latitude' in dims:
            rename_dims['latitude'] = 'south_north'
        if 'longitude' in dims:
            rename_dims['longitude'] = 'west_east'
        if rename_dims:
            ds_C404[var] = ds_C404[var].rename(rename_dims)
            
    output_name = f'/glade/derecho/scratch/ksha/GWC_Results/target_{year}_WY.nc'
    ds_C404.to_netcdf(output_name, format='NETCDF4_CLASSIC')
    print(output_name)
