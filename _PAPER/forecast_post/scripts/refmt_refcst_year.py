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

# # zarr encodings
# dict_encoding = {}
# varnames = list(ds_final.keys())
# varname_4D = ['WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05', 'WRF_P']

# chunk_size_3d = dict(chunks=(12, 336, 336))
# chunk_size_4d = dict(chunks=(12, 15, 336, 336))
# compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

# for i_var, var in enumerate(varnames):
#     if var in varname_4D:
#         dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
#     else:
#         dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

load_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/refcst/refcst_{year}-01-01T00Z.zarr'

ds_final = xr.open_zarr(load_name)
ds_final = ds_final.isel(time=slice(None, -1))

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/refcst_new/refcst_{year}-01-01T00Z.zarr'

ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True)



