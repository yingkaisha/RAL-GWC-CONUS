import logging
import numpy as np
from torchvision import transforms as tforms
from credit.transforms.transforms_wrf import Normalize_WRF, ToTensor_WRF
from credit.transforms.transforms_dscale import Normalize_Dscale, ToTensor_Dscale
from credit.transforms.transforms_diag import Normalize_Diag, ToTensor_Diag

logger = logging.getLogger(__name__)


def load_transforms(conf, scaler_only=False):
    """Load transforms.

    Args:
        conf (str): path to config
        scaler_only (bool): True --> retrun scaler; False --> return scaler and ToTensor

    Returns:
        tf.tensor: transform

    """
    # ------------------------------------------------------------------- #
    # transform class
    if conf["data"]["scaler_type"] == "std-wrf":
        transform_scaler = Normalize_WRF(conf)

    elif conf["data"]["scaler_type"] == "std-dscale":
        transform_scaler = Normalize_Dscale(conf)

    elif conf["data"]["scaler_type"] == "std-diag":
        transform_scaler = Normalize_Diag(conf)

    else:
        logger.log("scaler type not supported.")
        raise

    if scaler_only:
        return transform_scaler

    # ------------------------------------------------------------------- #
    # ToTensor class
    
    if conf["data"]["scaler_type"] == "std-wrf":
        to_tensor_scaler = ToTensor_WRF(conf)

    elif conf["data"]["scaler_type"] == "std-dscale":
        to_tensor_scaler = ToTensor_Dscale(conf)

    elif conf["data"]["scaler_type"] == "std-diag":
        to_tensor_scaler = ToTensor_Diag(conf)

    else:
        logger.log("ToTensor type not supported.")
        raise

    # ------------------------------------------------------------------- #
    # combine transform and ToTensor

    if transform_scaler is not None:
        transforms = [transform_scaler, to_tensor_scaler]

    else:
        transforms = [to_tensor_scaler]

    return tforms.Compose(transforms)
