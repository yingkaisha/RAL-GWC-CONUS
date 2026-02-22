import re
import os
import sys
import zarr
import yaml
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

source_dir = f'/glade/derecho/scratch/ksha/DWC/RAW_OUTPUT/CONUS_GP_diag_GDAS/*{year}*/*{year}*'
fn_all = sorted(glob(source_dir))

ds_collect = []

for fn in fn_all:
    ds = xr.open_dataset(fn)
    ds_collect.append(ds)

ds_final = xr.concat(ds_collect, dim='time')

ds_final = ds_final.rename({'latitude': 'south_north', 'longitude': 'west_east'})
ds_final['west_east'] = np.arange(336).astype(np.float32)
ds_final['south_north'] = np.arange(336).astype(np.float32)
ds_final = ds_final.drop_vars(['forecast_hour'])

load_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/prog_outputs/opt_GDAS_{year}.zarr'
ds_prog = xr.open_zarr(load_name)

ds_full = xr.merge([ds_final, ds_prog])
ds_full = ds_full.chunk({'time': 12, 'bottom_top': 12, 'south_north': 336, 'west_east': 336})

# ========================================================================== #
# encoding 
dict_encoding = {}
varnames = list(ds_final.keys())
varname_4D = ['WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05', 'WRF_P']

chunk_size_3d = dict(chunks=(12, 336, 336))
chunk_size_4d = dict(chunks=(12, 12, 336, 336))
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/full_output/opt_GDAS_{year}_full.zarr'
ds_full.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)






