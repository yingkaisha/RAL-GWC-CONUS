'''

Yingkai Sha
ksha@ucar.edu
'''
import re
import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xesmf as xe
import xarray as xr
import pandas as pd
from glob import glob
# from dask.utils import SerializableLock

# import calendar
from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta

# sys.path.insert(0, os.path.realpath('../libs/'))
# import verif_utils as vu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

def select_fn_fmt(fn_fmt_list, year):
    """
    Return (ind_year, fn_fmt) where fn_fmt_list[ind_year] covers `year`.
    Filenames must contain ranges like ...YYYYMMDDHH-YYYYMMDDHH.nc
    """
    for i, s in enumerate(fn_fmt_list):
        m = re.search(r'(\d{4})\d{4}-(\d{4})\d{4}\.nc$', s)
        if not m:
            continue
        start, end = map(int, m.groups())
        if start <= year <= end:
            return i, s
    raise ValueError(f"No filename pattern covers year {year}.")

year = int(args['year'])
flag_leap = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


# C404 static
static_WRF_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr'
ds_WRF_static = xr.open_zarr(static_WRF_name)
XLAT = ds_WRF_static['XLAT'].values
XLONG = ds_WRF_static['XLONG'].values
ds_WRF_static = ds_WRF_static.assign_coords(lat=(("south_north", "west_east"), XLAT))
ds_WRF_static = ds_WRF_static.assign_coords(lon=(("south_north", "west_east"), XLONG))
domain_inds = np.arange(336).astype(np.float32)

static_cesm = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static_cesm/ERA5_mlevel_cesm_static.zarr'
ds_cesm_static = xr.open_zarr(static_cesm)
ds_cesm_static = ds_cesm_static.rename({'longitude': 'lon', 'latitude': 'lat'})
ds_cesm_static = ds_cesm_static.sortby('lat')

levels_hPa = [1000, 950, 850, 700, 600, 500, 400, 300, 200, 100, 50]
levels_Pa = np.array(levels_hPa, dtype=np.int32) * 100  # Pa

RDGAS = 287.05   # J/kg/K
GRAVITY = 9.80665
LAPSE_RATE = 0.0065 # K/m
ALPHA = LAPSE_RATE * RDGAS / GRAVITY # ≈ 0.19026
target_height = 2.0  # meters

base_dir = "/gdex/data/d651056/CESM2-LE/atm/proc/tseries/day_1/"

ens_name = [
    '1011.001', '1031.002', '1051.003', '1071.004', '1091.005', 
    '1111.006', '1131.007', '1151.008', '1171.009', '1191.010'
]

ens = ens_name[-1] # last member only
   
varnames_surf = ['TMQ', 'PRECT', 'TREFHTMN', 'TREFHTMX']

fn_fmt_surf = [
    'b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h1.{}.20650101-20741231.nc',
    'b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h1.{}.20750101-20841231.nc',
    'b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h1.{}.20850101-20941231.nc',
    'b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h1.{}.20950101-21001231.nc']

ind_year, fn_fmt = select_fn_fmt(fn_fmt_surf, year)

dict_fn_surf = {}

for varname in varnames_surf:
    fn_ = fn_fmt.format(ens, varname)
    dict_fn_surf[varname] = os.path.join(base_dir, varname, fn_)

# =============================================== # 
# gather CESM2 data

ds_collcetion_surf = []

for varname in varnames_surf:
    fn_ = dict_fn_surf[varname]
    ds = xr.open_dataset(fn_)[[varname,]].sel(time=slice(f'{year}-01-01T00', f'{year}-12-31T23'))
    ds_collcetion_surf.append(ds)

ds = xr.merge(ds_collcetion_surf)

# =============================================== # 
# interpolation

ds['lon'] = (ds['lon']  + 180) % 360 - 180

regridder = xe.Regridder(ds, ds_WRF_static, method='bilinear')

ds_ERA5_interp = regridder(ds)

ds_ERA5_interp = ds_ERA5_interp.assign_coords(
    south_north=domain_inds, 
    west_east=domain_inds
)

ds_ERA5_interp = ds_ERA5_interp.drop_vars(['lon', 'lat'])

ds_final = ds_ERA5_interp

dt64 = ds_final.indexes["time"].to_datetimeindex()
ds_final = ds_final.assign_coords(time=dt64)

ds_final = ds_final.chunk({'time': 12, 'south_north': 336, 'west_east': 336})

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_CESM_SSP_daily/CESM_GP_{year}.zarr'
ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True)
print(save_name)

print('...all done...')

