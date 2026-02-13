'''

Yingkai Sha
ksha@ucar.edu
'''

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
from datetime import datetime, timedelta
import pygrib

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
parser.add_argument('i_ens', help='i_ens')
args = vars(parser.parse_args())

def insert_feb29_6h_linear(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """
    For each leap year in `ds`, linearly interpolate Feb 29 00/06/12/18 values
    between Feb 28 18 and Mar 1 00, then insert them. Assumes 6-hourly sampling.
    Works with dask.
    """
    t = pd.DatetimeIndex(ds[time_dim].values)
    if len(t) == 0:
        return ds

    # candidate years present
    years = np.unique(t.year)
    leap_years = [y for y in years if pd.Timestamp(y, 1, 1).is_leap_year]

    new_chunks = []
    for y in leap_years:
        prev_ts = pd.Timestamp(y, 2, 28, 18)  # 18Z on Feb 28
        next_ts = pd.Timestamp(y, 3, 1, 0)    # 00Z on Mar 1
        if prev_ts not in t or next_ts not in t:
            continue  # skip if endpoints missing

        # endpoints
        A = ds.sel({time_dim: prev_ts})
        B = ds.sel({time_dim: next_ts})

        # target Feb 29 stamps (00, 06, 12, 18)
        targets = pd.to_datetime([f"{y}-02-29 00:00",
                                  f"{y}-02-29 06:00",
                                  f"{y}-02-29 12:00",
                                  f"{y}-02-29 18:00"])

        # Only add if not already present
        targets = targets[~pd.Index(targets).isin(t)]
        if len(targets) == 0:
            continue

        # Linear weights between A (t=0) and B (t=30h). Positions are 6h,12h,18h,24h -> w = 1/5,2/5,3/5,4/5
        offsets_hours = (targets - prev_ts).astype("timedelta64[h]").astype(int)
        w = xr.DataArray(np.array(offsets_hours) / 30.0, dims=[time_dim])

        # Broadcast A and B to the "time" dim of the targets
        A_rep = A.expand_dims({time_dim: targets})
        B_rep = B.expand_dims({time_dim: targets}).assign_coords({time_dim: targets})

        # Interpolated Feb 29 slices
        feb29 = (1 - w) * A_rep + w * B_rep
        feb29 = feb29.assign_coords({time_dim: targets})

        new_chunks.append(feb29)

    if not new_chunks:
        return ds

    # Insert new times and sort
    ds_out = xr.concat([ds] + new_chunks, dim=time_dim).sortby(time_dim)
    return ds_out

# ==================================================================================== #
# get year from input
year = int(args['year'])
if year % 4 == 0:
    flag_leap = True
else:
    flag_leap = False

i_ens = int(args['i_ens'])

print(f'{year} - {i_ens}')

# ==================================================================================== #
# C404 static
static_WRF_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr'
ds_WRF_static = xr.open_zarr(static_WRF_name)
XLAT = ds_WRF_static['XLAT'].values
XLONG = ds_WRF_static['XLONG'].values
ds_WRF_static = ds_WRF_static.assign_coords(lat=(("south_north", "west_east"), XLAT))
ds_WRF_static = ds_WRF_static.assign_coords(lon=(("south_north", "west_east"), XLONG))
domain_inds = np.arange(336).astype(np.float32)

# ==================================================================================== #
# CESM static
static_cesm = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static_cesm/ERA5_mlevel_cesm_static.zarr'
ds_cesm_static = xr.open_zarr(static_cesm)
ds_cesm_static = ds_cesm_static.rename({'longitude': 'lon', 'latitude': 'lat'})
ds_cesm_static = ds_cesm_static.sortby('lat')

# ==================================================================================== #
# param
levels_hPa = [1000, 950, 850, 700, 600, 500, 400, 300, 200, 100, 50]
levels_Pa = np.array(levels_hPa, dtype=np.int32) * 100  # Pa

RDGAS = 287.05   # J/kg/K
GRAVITY = 9.80665
LAPSE_RATE = 0.0065 # K/m
ALPHA = LAPSE_RATE * RDGAS / GRAVITY # ≈ 0.19026
target_height = 2.0  # meters

# ==================================================================================== #
base_dir = "/gdex/data/d651056/CESM2-LE/atm/proc/tseries/hour_6/"
varnames = ['T', 'Q', 'U', 'V', 'Z3']
ens_name = [
    '1011.001', '1031.002', '1051.003', '1071.004', '1091.005', 
    '1111.006', '1131.007', '1151.008', '1171.009', '1191.010'
]
fn_fmt = 'b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h5.{}.2025010100-2034123100.nc'

dict_fn = {}
for ens in ens_name:
    fn_list = []
    for varname in varnames:
        fn_ = fn_fmt.format(ens, varname)
        fn_list.append(os.path.join(base_dir, varname, fn_))
        
    dict_fn[ens] = fn_list

varnames_surf = ['PS', 'PSL', 'UBOT', 'VBOT']

fn_fmt_surf = 'b.e21.BSSP370smbb.f09_g17.LE2-{}.cam.h2.{}.2025010100-2034123100.nc'
dict_fn_surf = {}
for ens in ens_name:
    fn_list = []
    for varname in varnames_surf:
        fn_ = fn_fmt_surf.format(ens, varname)
        fn_list.append(os.path.join(base_dir, varname, fn_))
        
    dict_fn_surf[ens] = fn_list

# ==================================================================================== #
ens = ens_name[i_ens]
fn_list = dict_fn[ens]
fn_list_surf = dict_fn_surf[ens]

# =============================================== # 
# gather CESM2 data
ds_collcetion = []
ds_collcetion_surf = []

for i_fn, fn in enumerate(fn_list):
    ds = xr.open_dataset(fn)[[varnames[i_fn],]]
    ds = ds.sel(time=slice(f'{year}-01-01T00', f'{year}-12-31T23')) #.isel(time=slice(10))
    ds_collcetion.append(ds)
    
ds_upper = xr.merge(ds_collcetion)

for i_fn, fn in enumerate(fn_list_surf):
    ds = xr.open_dataset(fn)[[varnames_surf[i_fn],]]
    ds = ds.sel(time=slice(f'{year}-01-01T00', f'{year}-12-31T23')) #.isel(time=slice(10))
    ds_collcetion_surf.append(ds)
    
ds_surf = xr.merge(ds_collcetion_surf)
ds = xr.merge([ds_upper, ds_surf])
# correct to bottom --> top
# ds = ds.isel(lev=slice(None, None, -1))

# ========================================================== #
# 2-m temperature
surface_height = ds_cesm_static['Z_GDS4_SFC']

# bottom model level fields
T_bot = ds['T'].isel(lev=-1)
P_bot = 992.556095 * 100.0

# surface pressure
PS = ds['PS']
# project temperature from bottom model level to the surface
T_surface = T_bot * (PS / P_bot) ** ALPHA
# T_surface = T_bot + ALPHA * T_bot * (PS / P_bot - 1)

# correct from surface to 2 m AGL
T_2m = (T_surface - LAPSE_RATE * target_height).rename('VAR_2T')
ds = ds.assign(VAR_2T=T_2m)

# =============================================== # 
# interpolation

# drop not needed vars
ds = ds.drop_vars(['Z3', 'PS'])
ds = ds.rename({'UBOT': 'VAR_10U', 'VBOT': 'VAR_10V', 'PSL': 'MSL'})

ds['lon'] = (ds['lon']  + 180) % 360 - 180

regridder = xe.Regridder(ds, ds_WRF_static, method='bilinear')

ds_ERA5_interp = regridder(ds)

ds_ERA5_interp = ds_ERA5_interp.assign_coords(
    south_north=domain_inds, 
    west_east=domain_inds
)

ds_ERA5_interp = ds_ERA5_interp.drop_vars(['lon', 'lat'])
ds_ERA5_interp = ds_ERA5_interp.rename({'lev': 'level'})

lev_src = ds_ERA5_interp["level"]
target = xr.DataArray(levels_hPa, dims=("level",), coords={"level": levels_hPa})
target_clamped = target.clip(min=lev_src.min(), max=lev_src.max())
ds_final = ds_ERA5_interp.interp({"level": target_clamped}, method="linear")

dt64 = ds_final.indexes["time"].to_datetimeindex()
ds_final = ds_final.assign_coords(time=dt64)

if flag_leap:
    ds_final = insert_feb29_6h_linear(ds_final, time_dim="time")

ds_final = ds_final.chunk({'time': 12, 'level': 11, 'south_north': 336, 'west_east': 336})

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_CESM/CESM_GP_mem{i_ens}_{year}.zarr'
ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True)
print(save_name)

print('...all done...')

