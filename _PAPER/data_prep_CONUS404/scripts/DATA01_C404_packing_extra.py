import os
import sys
import time
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

base_dir_extra = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/raw_404_Q_extra/'

start_time = time.time() 

ds_year = xr.open_zarr(f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/all_in_one/C404_GP_{year}.zarr')
fn_year_extra = sorted(glob(base_dir_extra+f'C404_GP_{year}*extra.zarr'))

if len(fn_year_extra) > 0:
    file_collect_extra = []
    
    for fn in fn_year_extra:
        ds = xr.open_zarr(fn)
        file_collect_extra.append(ds)
        
    ds_year_extra = xr.concat(file_collect_extra, dim='time')

    # merge all
    ds_year = xr.merge([ds_year, ds_year_extra])
    
    save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_{year}.zarr'
    ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True)
    print(save_name)
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    print(f'Skip year {year}')

