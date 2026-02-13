
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
qs = [0.05, 0.10, 0.90, 0.95]

varname_pick = ['WRF_T2', 'WRF_TD2', 'WRF_precip', 'WRF_U10', 'WRF_V10', 'WRF_MSLP', 'WRF_SP', 'WRF_PWAT']
fn_all = sorted(glob('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_*.zarr'))[:40]

ds_collection = []

for fn in fn_all:
    ds = xr.open_zarr(fn)[varname_pick]
    ds_collection.append(ds)

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

win = 2 * pad + 1

# cyclic wrap in 'doy' so the centered window is valid at the ends
ydh_wrap = xr.concat(
    [
        ds_ydh.isel(doy=slice(-pad, None)),
        ds_ydh,
        ds_ydh.isel(doy=slice(0, pad)),
    ],
    dim="doy",
)

# Build explicit 15-day windows along 'doy' (keeps year & hour intact)
# -> dims: (year, doy, doy_win=15, hour, south_north, west_east)
ydh_win = (
    ydh_wrap.rolling(doy=win, center=True, min_periods=win).construct("doy_win").isel(doy=slice(pad, -pad))
)

# un-chunk 
ydh_win = ydh_win.chunk(dict(year=-1, doy_win=-1))

# ================================================================== #
# compute quantiles across both the 15-day window and years

# reduces dims ('year','doy_win'), keeping (doy, hour, space...)
pct_15d = ydh_win.quantile(q=qs, dim=("year", "doy_win"), skipna=True)

# assign new dim and dimension
pct_15d = (
    pct_15d.rename(quantile="percentile").assign_coords(percentile=(np.array(qs) * 100).astype(int)).transpose("percentile", "doy", "hour", ...)
)

# re-chunk
pct_15d = pct_15d.chunk({'percentile': 4, 'doy': 1, 'hour': 1, 'south_north': 336, 'west_east': 336})

# ================================================================== #
dict_encoding = {}
chunk_size_4d = dict(chunks=(4, 1, 1, 336, 336))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varname_pick):
    dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    
save_name = f'/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/clim/CONUS_GP_percentile_{pad}d.zarr'
pct_15d.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

