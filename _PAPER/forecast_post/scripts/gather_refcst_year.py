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

fn_target = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_{year}.zarr'
ds_target = xr.open_zarr(fn_target)
ds_target = ds_target.isel(time=slice(1, None))

source_dir = '/glade/derecho/scratch/ksha/DWC/RAW_OUTPUT/CONUS_GP_refcst/'
fn_all = sorted(vu.get_nc_files(source_dir))

L = len(fn_all)

for i in range(L):
    sample_file = os.path.basename(fn_all[i][0])
    if str(year) in sample_file:
        ind_pick = i
        break

ds_collect = []

# fn_pick = sorted(fn_all[ind_pick])
fn_pick = sorted(fn_all[ind_pick], key=lambda x: int(re.search(r'_(\d+)\.nc$', x).group(1)))

for fn in fn_pick:
    ds = xr.open_dataset(fn)
    ds_collect.append(ds)

ds_final = xr.concat(ds_collect, dim='time')

ds_final = ds_final.rename({'latitude': 'south_north', 'longitude': 'west_east', 'level': 'bottom_top'})
ds_final['west_east'] = ds_target['west_east']
ds_final['south_north'] = ds_target['south_north']
ds_final['bottom_top'] = ds_target['bottom_top']

ds_final = xr.merge([ds_final, ds_target[['WRF_precip_025', 'WRF_radar_composite_025', 'WRF_TCC', 'WRF_OLR']]])
ds_final = ds_final.chunk({'time': 12, 'bottom_top': 15, 'south_north': 336, 'west_east': 336})

# zarr encodings
dict_encoding = {}
varnames = list(ds_final.keys())
varname_4D = ['WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05', 'WRF_P']

chunk_size_3d = dict(chunks=(12, 336, 336))
chunk_size_4d = dict(chunks=(12, 15, 336, 336))
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/refcst/refcst_{year}-01-01T00Z.zarr'
ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

