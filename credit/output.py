"""
output.py
-------------------------------------------------------
Content:
    - load_metadata()
    - make_xarray()
    - save_netcdf_increment()
"""

import os
import yaml
import logging
import traceback
import xarray as xr
from credit.data import drop_var_from_dataset
from inspect import signature
from credit.credit_ptype import CreditPostProcessor

logger = logging.getLogger(__name__)


def load_metadata(conf):
    """
    Load metadata attributes from yaml file in credit/metadata directory
    """
    # set priorities for user-specified metadata
    if conf["predict"]["metadata"]:
        meta_file = conf["predict"]["metadata"]
        with open(meta_file) as f:
            meta_data = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        print("conf['predict']['metadata'] not given. Skip.")
        meta_data = False

    return meta_data


def split_and_reshape(tensor, conf):
    """
    Split the output tensor of the model to upper air variables and diagnostics/surface variables.

    Upperair level arrangement: top-of-atmosphere--> near-surface --> single layer
    An example: U (top-of-atmosphere) --> U (near-surface) --> V (top-of-atmosphere) --> V (near-surface)
    The shape of the output tensor is (variables, latitude, longitude)

    Args:
        tensor: PyTorch Tensor containing output of the AI NWP model
        conf: config file for the model

    """

    # get the number of levels
    levels = conf["model"]["levels"]

    # get number of channels
    channels = len(conf["data"]["variables"])
    single_level_channels = len(conf["data"]["surface_variables"]) + len(conf["data"]["diagnostic_variables"])

    # subset upper air variables
    tensor_upper_air = tensor[:, : int(channels * levels), :, :]

    shape_upper_air = tensor_upper_air.shape
    tensor_upper_air = tensor_upper_air.reshape(shape_upper_air[0], channels, levels, shape_upper_air[-2], shape_upper_air[-1])

    # subset surface variables
    tensor_single_level = tensor[:, -int(single_level_channels) :, :, :]

    # return x, surf for B, c, lat, lon output
    return tensor_upper_air, tensor_single_level


def make_xarray(pred, forecast_datetime, lat, lon, conf):
    """
    Create two xarray.DataArray objects for upper air and surface variables.

    Args
        pred (torch.Tensor or np.ndarray): Prediction tensor containing both upper air and surface variables.
    forecast_datetime (datetime): The forecast initialization datetime.
    lat (np.ndarray or list): Latitude values.
    lon (np.ndarray or list): Longitude values.
    conf (dict): Configuration dictionary containing details about the data structure
        and variables.

    Returns:
        darray_upper_air (xarray.DataArray): DataArray containing upper air variables with dimensions
            [time, vars, level, latitude, longitude].
    darray_single_level (xarray.DataArray): DataArray containing surface variables with dimensions
        [time, vars, latitude, longitude].
    """
    # subset upper air and surface variables
    tensor_upper_air, tensor_single_level = split_and_reshape(pred, conf)

    # -------------------------------------------- #
    # level inds
    if "level_ids" in conf["data"].keys():
        level_ids = conf["data"]["level_ids"]
    else:
        level_ids = range(conf["model"]["levels"])

    # save upper air variables
    varname_upper = conf["data"]["variables"]

    # make xr.DatasArray
    darray_upper_air = xr.DataArray(
        tensor_upper_air,
        dims=["time", "vars", "level", "latitude", "longitude"],
        coords=dict(
            vars=varname_upper,
            time=[forecast_datetime],
            level=level_ids,
            latitude=lat,
            longitude=lon,
        ),
    )

    # save surface variables
    varname_single_level = conf["data"]["surface_variables"] + conf["data"]["diagnostic_variables"]

    if len(varname_single_level) > 0:
        # make xr.DatasArray
        darray_single_level = xr.DataArray(
            tensor_single_level.squeeze(2),
            dims=["time", "vars", "latitude", "longitude"],
            coords=dict(
                vars=varname_single_level,
                time=[forecast_datetime],
                latitude=lat,
                longitude=lon,
            ),
        )

        # return x-arrays as outputs
        return darray_upper_air, darray_single_level
    else:
        return darray_upper_air


def make_xarray_diag(pred, forecast_datetime, lat, lon, conf):
    tensor_single_level = pred

    varname_single_level = conf["data"]["diagnostic_variables"]

    # make xr.DatasArray
    darray_single_level = xr.DataArray(
        tensor_single_level.squeeze(2),
        dims=["time", "vars", "latitude", "longitude"],
        coords=dict(
            vars=varname_single_level,
            time=[forecast_datetime],
            latitude=lat,
            longitude=lon,
        ),
    )
    return darray_single_level

def save_netcdf_clean(
    darray_upper_air,
    darray_single_level,
    nc_filename,
    forecast_hour,
    meta_data,
    conf,
    use_logger=True,
):
    if use_logger:
        logging.basicConfig(level=logging.INFO)  # ensures this process logs
        logger = logging.getLogger(__name__)

    # if no single-level field
    if darray_single_level is None:
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_merged = ds_upper
    else:
        # merge upper-air and single-level
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")
        ds_merged = xr.merge([ds_upper, ds_single])

    # add forecast_hour coordinate
    ds_merged["forecast_hour"] = forecast_hour

    if use_logger:
        logger.info(f"Process forecast hour {forecast_hour} to {nc_filename}")

    save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
    os.makedirs(save_location, exist_ok=True)

    unique_filename = os.path.join(save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc")

    # ---------------------------------------------------- #
    # If conf['predict']['save_vars'] provided --> drop useless vars
    if "save_vars" in conf["predict"]:
        if len(conf["predict"]["save_vars"]) > 0:
            ds_merged = drop_var_from_dataset(ds_merged, conf["predict"]["save_vars"])

    # ---------------------------------------------------- #
    # handle meta data and time encoding
    encoding_dict = {}

    if meta_data is not False:
        # Add metadata attributes to every model variable if available
        for var in ds_merged.variables:
            if var in meta_data.keys():
                if var != "time":
                    # use attrs.update for non-datetime variables
                    ds_merged[var].attrs.update(meta_data[var])
                else:
                    # use time.encoding for datetime variables/coords
                    for metadata_time in meta_data["time"]:
                        ds_merged.time.encoding[metadata_time] = meta_data["time"][metadata_time]
    else:
        # if not metadata available, apply time encoding based on gregorian calendar
        time_encoding = {
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }

        encoding_dict = {"time": time_encoding}

    # ---------------------------------------------------- #
    # save
    if use_logger:
        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")

    ds_merged.to_netcdf(unique_filename, mode="w", encoding=encoding_dict)


def save_netcdf_diag(
    darray_single_level: xr.DataArray,
    nc_foldername: str,
    nc_filename: str,
    forecast_hour: int,
    meta_data: dict,
    conf: dict,
    use_logger=True,
):
    if use_logger:
        logging.basicConfig(level=logging.INFO)  # ensures this process logs
        logger = logging.getLogger(__name__)

    # Convert DataArrays to Datasets
    ds_single = darray_single_level.to_dataset(dim="vars")

    ds_merged = ds_single
    ds_merged["forecast_hour"] = forecast_hour
    ds_merged.attrs["Conventions"] = "CF-1.11"

    if use_logger:
        logger.info(f"Process forecast hour {forecast_hour} to {nc_filename}")

    save_location = os.path.join(conf["predict"]["save_forecast"], nc_foldername)
    os.makedirs(save_location, exist_ok=True)

    unique_filename = os.path.join(
        save_location,
        f"pred_{nc_filename}.nc",  # _{forecast_hour:03d}.nc"
    )
    # ---------------------------------------------------- #
    # If conf['predict']['save_vars'] provided --> drop useless vars
    if "save_vars" in conf["predict"]:
        if len(conf["predict"]["save_vars"]) > 0:
            ds_merged = drop_var_from_dataset(ds_merged, conf["predict"]["save_vars"])

    # when there's no metafile --> meta_data = False
    encoding_dict = {}

    if meta_data is not False:
        # Add metadata attributes to every model variable if available
        for var in ds_merged.variables:
            if var in meta_data.keys():
                if var != "time":
                    # use attrs.update for non-datetime variables
                    ds_merged[var].attrs.update(meta_data[var])
                else:
                    # use time.encoding for datetime variables/coords
                    for metadata_time in meta_data["time"]:
                        ds_merged.time.encoding[metadata_time] = meta_data["time"][metadata_time]
    else:
        time_encoding = {
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }
        encoding_dict = {"time": time_encoding}

    # ---------------------------------------------------- #
    # save
    if use_logger:
        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")

    ds_merged.to_netcdf(unique_filename, mode="w", encoding=encoding_dict)
