
import os
import zarr
from glob import glob

import numpy as np
import xarray as xr

def _normalize_one_year(ds):
    # Drop leap day
    is_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
    ds = ds.sel(time=~is_feb29)

    # Add day-of-year and hour coords
    ds = ds.assign_coords(
        doy=("time", ds.time.dt.dayofyear.values),
        hour=("time", ds.time.dt.hour.values),
    )

    # Reshape to [doy, hour, ...space...] for this one year
    ydh = (
        ds.set_index(time=["doy", "hour"]).unstack("time").transpose("doy", "hour", ...)
    )

    # Add explicit 'year' dimension
    year_label = int(ds.time.dt.year.values[0])
    ydh = ydh.expand_dims(year=[year_label])

    return ydh


pad = 15 # 7-day time window

# 30 year data
fn_all = sorted(glob('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_*.zarr'))[10:40]

ds_collection = []

for fn in fn_all:
    ds = xr.open_zarr(fn)
    ds_collection.append(ds)

varname_pick = list(ds.keys())
varname_4D = ['WRF_P', 'WRF_Q_tot', 'WRF_Q_tot_05', 'WRF_T', 'WRF_U', 'WRF_V', 'WRF_W', 'WRF_Z']

# Normalize each already-opened yearly dataset to [year, doy, hour, space...]
pieces = [_normalize_one_year(ds) for ds in ds_collection]

# Concatenate across the small 'year' dimension (fast & metadata-robust)
ds_ydh = xr.concat(
    pieces,
    dim="year",
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="override",
)

# Mean over years -> (doy=365, hour=24, space...)
clim_dayhour = ds_ydh.mean("year")

# 15-day centered moving window along 'doy' with cyclic wrap
pad = int(pad)
win = 2 * pad + 1

# pad at both ends (wrap) -> roll -> crop
clim_wrap = xr.concat(
    [
        clim_dayhour.isel(doy=slice(-pad, None)),
        clim_dayhour,
        clim_dayhour.isel(doy=slice(0, pad)),
    ],
    dim="doy",
)

clim = (
    clim_wrap.rolling(doy=win, center=True, min_periods=win).mean().isel(doy=slice(pad, -pad)).transpose("doy", "hour", ...)
)

clim = clim.chunk({'doy': 365, 'hour': 24, 'south_north': 336, 'west_east': 336, 'bottom_top': 12, 'pressure_approx': 12})

dict_encoding = {}
chunk_size_4d = dict(chunks=(1, 1, 12, 336, 336))
chunk_size_3d = dict(chunks=(1, 1, 336, 336))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varname_pick):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}
    
save_name = f'/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/clim/CONUS_GP_clim_{pad}d.zarr'
clim.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)


# ===================================================================================== #
# pad on 366 days
fn = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/clim/CONUS_GP_clim_15d.zarr'
ds = xr.open_zarr(fn)

idx = np.concatenate([np.arange(59), [58], np.arange(59, ds.sizes['doy'])])
# idx length = 366: 0..58, 58 again (Feb 29 as a copy of Feb 28), then 59..364

ds_366 = ds.isel(doy=idx)

# Fix the 'doy' coordinate labels to be contiguous again.
# If your current 'doy' coords are 1..365:
if 'doy' in ds.coords and int(ds['doy'].values[0]) == 1:
    ds_366 = ds_366.assign_coords(doy=np.arange(1, 367))
else:
    # If they are 0..364 (or there is no coord), keep 0-based style:
    ds_366 = ds_366.assign_coords(doy=np.arange(366))

save_name = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/clim/CONUS_GP_clim_15d_366.zarr'
ds_366.to_zarr(save_name, mode='w', consolidated=True, compute=True)

