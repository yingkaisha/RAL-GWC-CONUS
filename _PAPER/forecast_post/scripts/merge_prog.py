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

IND_max = 24*366-1
INDs = np.arange(0, IND_max+996, 996)
INDs[-1] = IND_max

base_dir = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/prog_outputs/source/'

ds_collect = []
for i, ind_start in enumerate(INDs[:-1]):
    
    ind_end = INDs[i+1]
    load_name = base_dir + f'opt_single_clean_{ind_start:04d}_{ind_end:04d}_2020-01-01T00Z.zarr'
    ds_collect.append(xr.open_zarr(load_name))

ds_final = xr.concat(ds_collect, dim='time')

ds_final = ds_final.chunk({'time': 12, 'bottom_top': 12, 'south_north': 336, 'west_east': 336})

# =================================================== #
# zarr encodings
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

base_dir = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/prog_outputs/'
save_name = base_dir + f'opt_single_clean_2020-01-01T00Z.zarr'
ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

fn_target = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_2021.zarr'
ds_target = xr.open_zarr(fn_target)
ds_save = ds_target.isel(time=slice(120))
save_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/prog_outputs/opt_single_clean_2021-01-01T00Z.zarr'
ds_save.to_zarr(save_name, mode='w', consolidated=True, compute=True)


