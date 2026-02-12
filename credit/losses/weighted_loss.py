import torch
import xarray as xr
import numpy as np
import logging

from credit.losses.base_losses import base_losses

logger = logging.getLogger(__name__)


def latitude_weights(conf):
    
    # Open the dataset and extract latitude and longitude information
    ds = xr.open_dataset(conf["loss"]["latitude_weights"])
    lat = torch.from_numpy(ds["latitude"].values).float()
    lon_dim = ds["longitude"].shape[0]

    # Calculate weights using PyTorch operations
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()

    # Create a 2D tensor of weights
    L = weights.unsqueeze(1).expand(-1, lon_dim)

    return L


def variable_weights(conf, channels, frames):
    """Create variable-specific weights for different atmospheric
    and surface channels.

    This function loads weights for different atmospheric variables
    (e.g., U, V, T, Q) and surface variables (e.g., SP, t2m) from
    the configuration file. It then combines them into a single
    weight tensor for use in loss calculations.

    Args:
        conf (dict): Configuration dictionary containing the
            variable weights.
        channels (int): Number of channels for atmospheric variables.
        frames (int): Number of time frames.

    Returns:
        torch.Tensor: A tensor containing the combined weights for
            all variables.
    """
    # Load weights for U, V, T, Q
    varname_upper_air = conf["data"]["variables"]
    varname_surface = conf["data"]["surface_variables"]
    varname_diagnostics = conf["data"]["diagnostic_variables"]

    # surface + diag channels
    N_channels_single = len(varname_surface) + len(varname_diagnostics)

    weights_upper_air = torch.tensor([conf["loss"]["variable_weights"][var] for var in varname_upper_air]).view(1, channels * frames, 1, 1)

    weights_single = torch.tensor([conf["loss"]["variable_weights"][var] for var in (varname_surface + varname_diagnostics)]).view(1, N_channels_single, 1, 1)

    # Combine all weights along the color channel
    var_weights = torch.cat([weights_upper_air, weights_single], dim=1)

    return var_weights


class VariableTotalLoss2D(torch.nn.Module):
    """Custom loss function class for 2D geospatial data
    with optional spectral and power loss components.

    This class defines a loss function that combines a base loss
    (e.g., L1, MSE) with optional spectral and power loss components
    for 2D geospatial data. The loss function can incorporate latitude
    and variable-specific weights.

    Args:
        conf (dict): Configuration dictionary containing loss
            function settings and weights.
        validation (bool, optional): If True, the loss function
            is used in validation mode. Defaults to False.
    """

    def __init__(self, conf, validation=False):
        
        super(VariableTotalLoss2D, self).__init__()

        self.conf = conf
        self.training_loss = conf["loss"]["training_loss"]

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.lat_weights = None
        if conf["loss"]["use_latitude_weights"]:
            logger.info("Using latitude weights in loss calculations")
            self.lat_weights = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # ------------------------------------------------------------- #
        # variable weights
        # order: upper air --> surface --> diagnostics
        self.var_weights = None
        if conf["loss"]["use_variable_weights"]:
            logger.info("Using variable weights in loss calculations")

            var_weights = [value if isinstance(value, list) else [value] for value in conf["loss"]["variable_weights"].values()]

            var_weights = np.array([item for sublist in var_weights for item in sublist])

            self.var_weights = torch.from_numpy(var_weights)
        # # ------------------------------------------------------------- #

        self.validation = validation # True / False
        
        if self.validation:
            if "validation_loss" in conf["loss"]:
                self.loss_fn = base_losses(conf, reduction="none", validation=True)
            else:
                self.loss_fn = torch.nn.L1Loss(reduction="none")
        else:
            self.loss_fn = base_losses(conf, reduction="none", validation=False)

    def forward(self, target, pred):
        
        # User defined loss
        loss = self.loss_fn(target, pred)

        # Latitutde and variable weights
        loss_dict = {}
        for i, var in enumerate(self.vars):
            var_loss = loss[:, i]

            if self.lat_weights is not None:
                var_loss = torch.mul(var_loss, self.lat_weights.to(target.device))

            if self.var_weights is not None:
                var_loss *= self.var_weights[i].to(target.device)

            loss_dict[f"loss_{var}"] = var_loss.mean()

        loss = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss
