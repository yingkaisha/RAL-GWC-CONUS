"""Data.py contains modules for processing training data.

Heper functions:
    - generate_datetime(start_time, end_time, interval_hr)
    - hour_to_nanoseconds(input_hr)
    - nanoseconds_to_year(nanoseconds_value)
    - extract_month_day_hour(dates)
    - find_common_indices(list1, list2)
    - concat_and_reshape(x1, x2)
    - reshape_only(x1)
    - get_forward_data(filename)
    - drop_var_from_dataset()
    - previous_hourly_steps()
    - next_n_hour()
    - encode_datetime64()

Sample class:
    - Sample
    - Sample_WRF
    - Sample_dscale
    - Sample_diag
"""

# system tools
from typing import TypedDict, Union, List, Sequence

# data utils
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import cftime

# Pytorch utils
import torch
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

#
Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ("historical_ERA5_images", "target_ERA5_images")


def device_compatible_to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safely move tensor to device, with float32 casting on MPS (Metal Performance Shaders). Addresses runtime error in OSX about MPS not supporting float64.

    Args:
        tensor (torch.Tensor): Input tensor to move.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Tensor moved to device (cast to float32 if device is MPS).
    """

    if device.type == "mps":
        return tensor.to(dtype=torch.float32, device=device)
    else:
        return tensor.to(device)


def ensure_numpy_datetime(value):
    """
    Converts an input value (or array) to numpy.datetime64.
    Handles numpy arrays, pandas timestamps, cftime objects, and strings.
    """
    # If the value is an array, extract the first element
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()  # Extract scalar value
        else:
            raise TypeError(f"Cannot convert array with multiple elements: {value}")

    if isinstance(value, np.datetime64):
        return value  # Already correct
    elif isinstance(value, pd.Timestamp):
        return np.datetime64(value)  # Convert from pandas Timestamp
    elif isinstance(value, str):
        try:
            return np.datetime64(value)  # Convert from string
        except ValueError:
            pass  # If it fails, let it fall through
    elif isinstance(value, cftime.datetime):
        return np.datetime64(value.strftime("%Y-%m-%dT%H:%M:%S"))  # Convert from cftime
    elif isinstance(value, object):  # Catch-all for potential unexpected object types
        try:
            return np.datetime64(pd.to_datetime(value))
        except Exception:
            raise TypeError(f"Cannot convert type {type(value)} to numpy.datetime64")
    else:
        raise TypeError(f"Unsupported type {type(value)} for datetime conversion")


def generate_datetime(start_time, end_time, interval_hr):
    """Generate a list of datetime.datetime based on stat, end times, and hour interval.

    Args:
        start_time (datetime.datetime): start time
        end_time (datetime.datetime): end time
        interval_hr (int): hour interval

    """
    # Define the time interval (e.g., every hour)
    interval = datetime.timedelta(hours=interval_hr)

    # Generate the list of datetime objects
    datetime_list = []
    current_time = start_time
    while current_time <= end_time:
        datetime_list.append(current_time)
        current_time += interval
    return datetime_list


def hour_to_nanoseconds(input_hr):
    """Convert hour to nanoseconds."""
    # hr * min_per_hr * sec_per_min * nanosec_per_sec
    return input_hr * 60 * 60 * 1000000000


def nanoseconds_to_year(nanoseconds_value):
    """Given datetime info as nanoseconds, compute which year it belongs to."""
    return np.datetime64(nanoseconds_value, "ns").astype("datetime64[Y]").astype(int) + 1970


def extract_month_day_hour(dates):
    """Given an 1-d array of np.datatime64[ns], extract their mon, day, hr into a zipped list."""
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    days = (dates - dates.astype("datetime64[M]") + 1).astype("timedelta64[D]").astype(int)
    hours = dates.astype("datetime64[h]").astype(int) % 24
    return list(zip(months, days, hours))


def find_common_indices(list1, list2):
    """Find indices of common elements between two lists."""
    # Find common elements
    common_elements = set(list1).intersection(set(list2))

    # Find indices of common elements in both lists
    indices_list1 = [i for i, x in enumerate(list1) if x in common_elements]
    indices_list2 = [i for i, x in enumerate(list2) if x in common_elements]

    return indices_list1, indices_list2


def concat_and_reshape(x1, x2):
    """Flattening the "level" coordinate of upper-air variables and concatenate it will surface variables."""
    # print("x1 shape: ", x1.shape)
    # print("x2 shape: ", x2.shape)
    x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
    x_concat = torch.cat((x1, x2), dim=2)
    return x_concat.permute(0, 2, 1, 3, 4)


def reshape_only(x1):
    """Flattening the "level" coordinate of upper-air variables.

    As in "concat_and_reshape", but no concat.
    """
    x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
    return x1.permute(0, 2, 1, 3, 4)


def get_forward_data(filename) -> xr.Dataset:
    """Check nc vs. zarr files and open file as xr.Dataset."""
    if filename[-3:] == ".nc" or filename[-4:] == ".nc4":
        dataset = xr.open_dataset(filename)
    else:
        dataset = xr.open_zarr(filename)
    return dataset


def flatten_list(list_of_lists):
    """Flatten a list of lists.

    Parameters
    ----------
    - list_of_lists (list): A list containing sublists.

    Returns
    -------
    - flattened_list (list): A flattened list containing all elements from sublists.

    """
    return [item for sublist in list_of_lists for item in sublist]


def generate_integer_list_around(number, spacing=10):
    """Generate a list of integers on either side of a given number with a specified spacing.

    Parameters
    ----------
    - number (int): The central number around which the list is generated.
    - spacing (int): The spacing between consecutive integers in the list. Default is 10.

    Returns
    -------
    - integer_list (list): List of integers on either side of the given number.

    """
    lower_limit = number - spacing
    upper_limit = number + spacing + 1  # Adding 1 to include the upper limit
    integer_list = list(range(lower_limit, upper_limit))

    return integer_list


def find_key_for_number(input_number, data_dict):
    """Find the key in the dictionary based on the given number.

    Parameters
    ----------
    - input_number (int): The number to search for in the dictionary.
    - data_dict (dict): The dictionary with keys and corresponding value lists.

    Returns
    -------
    - key_found (str): The key in the dictionary where the input number falls within the specified range.

    """
    for key, value_list in data_dict.items():
        if value_list[1] <= input_number <= value_list[2]:
            return key

    # Return None if the number is not within any range
    return None


def drop_var_from_dataset(xarray_dataset, varname_keep):
    """Preserve a given set of variables from an xarray.Dataset, and drop the rest.

    It will raise error if `varname_key` is missing from `xarray_dataset`.
    """
    varname_all = list(xarray_dataset.keys())

    for varname in varname_all:
        if varname not in varname_keep:
            xarray_dataset = xarray_dataset.drop_vars(varname)

    varname_clean = list(xarray_dataset.keys())

    varname_diff = list(set(varname_keep) - set(varname_clean))
    assert len(varname_diff) == 0, "Variable name: {} missing".format(varname_diff)

    return xarray_dataset


def keep_dataset_vars(xarray_dataset: xr.Dataset, varnames_keep: List[str]):
    """Return a version of an xarray dataset with only a selected subset of variables.

    Args:
        xarray_dataset (xr.Dataset): The xarray dataset.
        varnames_keep (List[str]): a list of variable names to be kept.

    Returns:
        xr.Dataset with only the variables in varnames_keep included.

    """
    return xarray_dataset[varnames_keep]

def random_patch(
    WRF_input, WRF_target, ds_outside, patch_size=(256, 256), rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    ph, pw = patch_size
    ny = WRF_input.sizes["yIndex"]
    nx = WRF_input.sizes["xIndex"]

    # random top-left corner
    y0 = rng.integers(0, ny - ph + 1)
    x0 = rng.integers(0, nx - pw + 1)


    ys = slice(y0, y0 + ph)
    xs = slice(x0, x0 + pw)
    return (
        WRF_input.isel(yIndex=ys, xIndex=xs),
        WRF_target.isel(yIndex=ys, xIndex=xs),
        ds_outside.isel(yIndex=ys, xIndex=xs),
    )

def random_land_patch(
    WRF_input, WRF_target, ds_outside, patch_size=(336, 336), 
    min_land_frac=0.2, max_tries=2000, rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    ph, pw = patch_size
    ny = WRF_input.sizes["south_north"]
    nx = WRF_input.sizes["west_east"]

    # Use LANDMASK; collapse extra dims if needed (e.g., time)
    landmask = WRF_input["LANDMASK"]
    for dim in ("time", "level"):
        if dim in landmask.dims:
            landmask = landmask.isel({dim: 0})

    mask = landmask.values.astype(np.float32)  # 2D array (ny, nx)

    for _ in range(max_tries):
        # random top-left corner
        y0 = rng.integers(0, ny - ph + 1)
        x0 = rng.integers(0, nx - pw + 1)

        submask = mask[y0:y0+ph, x0:x0+pw]
        land_frac = submask.mean()

        if land_frac >= min_land_frac:
            ys = slice(y0, y0 + ph)
            xs = slice(x0, x0 + pw)
            return (
                WRF_input.isel(south_north=ys, west_east=xs),
                WRF_target.isel(south_north=ys, west_east=xs),
                ds_outside.isel(south_north=ys, west_east=xs),
            )
            
    # Fallback to central patch
    warnings.warn(
        "random_land_patch: could not find a patch meeting land fraction "
        f"≥ {min_land_frac} after {max_tries} tries; using central patch instead."
    )

    y0_c = (ny - ph) // 2
    x0_c = (nx - pw) // 2
    ys = slice(y0_c, y0_c + ph)
    xs = slice(x0_c, x0_c + pw)

    return (
        WRF_input.isel(south_north=ys, west_east=xs),
        WRF_target.isel(south_north=ys, west_east=xs),
        ds_outside.isel(south_north=ys, west_east=xs),
    )



def subset_patch(
    ds: xr.Dataset,
    input_size,
    start,  # (ilat0, ilon0). If None → center crop
    lat_name="yIndex",
    lon_name="xIndex",
) -> xr.Dataset:
    """
    Return a spatial subset of shape (time, input_size[0], input_size[1]).
    Assumes ds has dims (time, lat, lon).
    """
    H = ds.dims[lat_name]
    W = ds.dims[lon_name]
    h, w = input_size

    if h > H or w > W:
        raise ValueError(f"Requested patch {h}x{w} exceeds dataset size {H}x{W}")

    if start is None:
        i0 = (H - h) // 2
        j0 = (W - w) // 2
    else:
        i0, j0 = start
        if i0 < 0 or j0 < 0 or i0 + h > H or j0 + w > W:
            raise ValueError(f"Start {(i0, j0)} with size {(h, w)} is out of bounds for {H}x{W}")

    i1 = i0 + h
    j1 = j0 + w

    return ds.isel({lat_name: slice(i0, i1), lon_name: slice(j0, j1)})


def encode_datetime64(dt_array):
    dt_array = np.atleast_1d(dt_array).astype("datetime64[ns]")
    dt_s = dt_array.astype("datetime64[s]")

    # Time components
    seconds_in_day = 86400
    seconds_since_midnight = (dt_s - dt_s.astype("datetime64[D]")).astype("timedelta64[s]").astype(int)
    hour = seconds_since_midnight / 3600.0

    # Day of year
    year_start = dt_s.astype("datetime64[Y]")
    day_of_year = (dt_s - year_start).astype("timedelta64[D]").astype(int) + 1

    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)

    return np.concatenate((hour_sin, hour_cos, doy_sin, doy_cos), axis=0)


def next_n_hour(dt, period_hours, key="h"):
    """
    Round dt forward to the next N-hour boundary.

    Parameters:
    - dt: np.datetime64[ns] or array of such values
    - period_hours: int, the interval in hours (e.g., 3, 6)

    Returns:
    - np.datetime64[ns] rounded forward to the next period_hours boundary
    """
    period_ns = int(np.timedelta64(period_hours, key) / np.timedelta64(1, "ns"))
    ns = dt.astype("int64")
    out = (ns // period_ns + 1) * period_ns
    return out.astype("datetime64[ns]")


def next_n_second(dt, period_second, key="s"):
    dt_ns = np.asarray(dt).astype("datetime64[ns]")
    period_ns = int(np.timedelta64(int(period_second), key) / np.timedelta64(1, "ns"))
    ns = dt_ns.astype("int64")
    out = ((ns + period_ns - 1) // period_ns) * period_ns
    return out.astype("datetime64[ns]")


def previous_hourly_steps(time_pick, hour, step):
    """
    Given a datetime64[ns] time_pick, compute time_pick - step * hours.
    """
    return time_pick - np.timedelta64(hour * step, "h")


def previous_second_steps(time_pick, second, step):
    """
    Given a datetime64[ns] time_pick, compute time_pick - step * hours.
    """
    return time_pick - np.timedelta64(second * step, "s")

def filter_ds(ds: xr.Dataset, varnames_keep: Sequence[str]) -> xr.Dataset:
    """
    Return a new Dataset containing only the variables in varnames_keep.
    Raises if any var in varnames_keep is missing.
    """
    missing = set(varnames_keep) - set(ds.data_vars)
    if missing:
        raise KeyError(f"Missing variables in dataset: {missing}")
    # this builds the new Dataset by iterating only over varnames_keep
    return ds[list(varnames_keep)]


class Sample(TypedDict):
    historical_ERA5_images: Array
    target_ERA5_images: Array
    datetime_index: Array


class Sample_WRF(TypedDict):
    WRF_input: Array
    WRF_target: Array
    boundary_input: Array
    time_encode: Array
    datetime_index: Array


class Sample_dscale(TypedDict):
    LR_input: Array
    HR_input: Array
    HR_target: Array
    time_encode: Array
    datetime_index: Array


class Sample_diag(TypedDict):
    WRF_input: Array
    WRF_target: Array
    time_encode: Array
    datetime_index: Array
    